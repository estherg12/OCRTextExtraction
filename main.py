import os
from pathlib import Path
import time
import cv2
from fpdf import FPDF
import numpy as np
import re
from PIL import Image, ImageOps
import torch
from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

FILE = "sample_invoice.jpg"
REMOTE_REPO_ID = "ifesther/trocr-spanish-handwritten"
LOCAL_MODEL_DIR = "./trocr_spanish_final"

num_cores = max(1, (os.cpu_count() or 4) - 1)
torch.set_num_threads(num_cores)

# Common cursive visual confusion corrections for Spanish text
CORRECTION_MAP = {
    r"\bense\b": "eso",
    r"\bsonos\b": "sueños",
    r"\bperseír\b": "perseguir",
}

def clean_spanish_transcription(text: str) -> str:
    """Corrects visual character-confusion artifacts."""
    cleaned = text
    for wrong, right in CORRECTION_MAP.items():
        cleaned = re.sub(wrong, right, cleaned, flags=re.IGNORECASE)
    return cleaned

def load_model_pipeline():
    """Load the fine-tuned TrOCR model.

    Prefers the local directory if available, falling back to Hugging Face Hub
    for external users.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = (
        LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else REMOTE_REPO_ID
    )

    print(f"Loading Spanish handwriting model from '{checkpoint}' on {device}...")

    image_processor = ViTImageProcessor.from_pretrained(checkpoint)
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
    processor = TrOCRProcessor(
        image_processor=image_processor, tokenizer=tokenizer
    )

    model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(device)
    model.eval()

    return processor, model, device

def pad_to_natural_aspect_ratio(
    pil_crop: Image.Image, target_height: int = 120
) -> Image.Image:
    """Pad the top and bottom with paper-white borders.

    Prevents ViT from squashing thin text lines into unnatural squares.
    """
    w, h = pil_crop.size
    if h == 0 or w == 0:
        return pil_crop

    # Scale proportionally so the text height is consistent
    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    resized = pil_crop.resize((new_w, target_height), Image.Resampling.BILINEAR)

    # Add 25px top/bottom padding to give the Transformer clean margins
    padded = ImageOps.expand(resized, border=(15, 20, 15, 20), fill=(255, 255, 255))
    return padded

def segment_document_lines(image_path: str) -> list[Image.Image]:
    """Segments document pages into clean, single-line crops using
    smoothed horizontal profile valley detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h_img, w_img, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize ink vs. background
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Horizontal Projection Profile (sum ink along each row)
    profile = np.sum(thresh, axis=1) / 255.0

    # Smooth the profile to eliminate intra-letter spikes
    kernel_size = max(5, int(h_img * 0.02))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(
        profile.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    # Find ink regions (rows where smoothed density is above baseline noise)
    noise_floor = np.max(smoothed) * 0.08
    in_line = False
    start_y = 0
    raw_cuts = []

    for y, val in enumerate(smoothed):
        if val > noise_floor and not in_line:
            in_line = True
            start_y = y
        elif val <= noise_floor and in_line:
            in_line = False
            raw_cuts.append((start_y, y))

    if in_line:
        raw_cuts.append((start_y, h_img))

    # Filter out tiny artifacts and merge lines that are unnaturally thin
    min_line_height = int(h_img * 0.04)  # At least 4% of image height
    valid_bands = [
        (y1, y2) for y1, y2 in raw_cuts if (y2 - y1) >= min_line_height
    ]

    # Fallback: If no clean valleys detected (single-line crop like phrase1.jpg)
    if not valid_bands:
        raw_crop = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return [pad_to_natural_aspect_ratio(raw_crop)]

    # Extract crops with tight lateral ink bounds
    debug_dir = Path("debug_crops")
    debug_dir.mkdir(exist_ok=True)
    for f in debug_dir.glob("*.png"):
        f.unlink()

    line_crops = []
    pad_y = 6

    for idx, (y1, y2) in enumerate(valid_bands):
        top = max(0, y1 - pad_y)
        bottom = min(h_img, y2 + pad_y)

        line_strip = image[top:bottom, :]
        strip_thresh = thresh[top:bottom, :]

        # Find horizontal ink extents inside this strip (tight x-padding)
        col_profile = np.sum(strip_thresh, axis=0)
        ink_cols = np.where(col_profile > 0)[0]

        if len(ink_cols) > 0:
            left = max(0, ink_cols[0] - 10)
            right = min(w_img, ink_cols[-1] + 10)
        else:
            left, right = 0, w_img

        raw_strip = image[top:bottom, left:right]
        pil_raw = Image.fromarray(cv2.cvtColor(raw_strip, cv2.COLOR_BGR2RGB))

        # Preserve natural strokes with clean letterbox padding
        padded_line = pad_to_natural_aspect_ratio(pil_raw)
        padded_line.save(debug_dir / f"line_{idx:02d}.png")
        line_crops.append(padded_line)

    print(
        f"Segmented {len(line_crops)} line(s). Saved inspectable crops to 'debug_crops/'."
    )
    return line_crops


def transcribe_lines(
    crops: list[Image.Image], processor, model, device: str
) -> list[str]:
    """Transcribes line crops using beam search to resolve cursive character ambiguity."""
    transcriptions = []

    for crop in crops:
        pixel_values = processor(crop, return_tensors="pt").pixel_values.to(
            device
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=40,
                num_beams=2,  # Explores top 2 hypotheses to resolve ambiguous connections
                length_penalty=1.0,
                early_stopping=True,
            )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        cleaned = text.strip()
        cleaned = clean_spanish_transcription(cleaned)
        if cleaned:
            transcriptions.append(cleaned)

    return transcriptions


def export_to_txt(lines: list[str], output_path: str):
    """Export lines to a UTF-8 text file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved text to: {output_path}")


def export_to_pdf(lines: list[str], output_path: str):
    """Export lines to a PDF file handling Spanish characters and accents."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    for line in lines:
        # Clean latin-1 mapping for accents (á, é, í, ó, ú, ñ)
        encoded_line = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(w=pdf.epw, h=9, text=encoded_line)

    pdf.output(output_path)
    print(f"Saved transcript to: {output_path}")


if __name__ == "__main__":
    # Test on full poem image or any cropped phrase
    input_file = "%s" % FILE

    if not Path(input_file).exists():
        # Fallback to images folder if located inside images/
        input_file = "images/%s" % FILE

    base_name = Path(input_file).stem
    output_dir = Path("output")

    # Time tracking
    t0 = time.time()
    proc, mdl, dev = load_model_pipeline()
    load_duration = time.time() - t0

    t1 = time.time()
    line_crops = segment_document_lines(input_file)
    seg_duration = time.time() - t1

    t2 = time.time()
    results = transcribe_lines(line_crops, proc, mdl, dev)
    ocr_duration = time.time() - t2

    print("\n================== Extracted Text ==================")
    print("\n".join(results))
    print("====================================================")
    print(
        f"Timing: Model Load={load_duration:.2f}s | Segmentation={seg_duration:.2f}s | OCR Inference={ocr_duration:.2f}s\n"
    )

    export_to_txt(results, str(output_dir / f"{base_name}_transcription.txt"))
    export_to_pdf(results, str(output_dir / f"{base_name}_transcription.pdf"))