from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

def load_trocr():
    """Load the trained Spanish TrOCR model directly from Hugging Face Hub."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "ifesther/trocr-spanish-handwritten"

    # Automatically downloads and caches your weights from Hugging Face Hub
    image_processor = ViTImageProcessor.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    processor = TrOCRProcessor(
        image_processor=image_processor, tokenizer=tokenizer
    )

    model = VisionEncoderDecoderModel.from_pretrained(repo_id).to(device)
    model.eval()

    return processor, model, device

def segment_lines(image_path: str):
    """Segment handwritten text lines with visual debugging."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert binary threshold (ink becomes white, background becomes black)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Wide horizontal kernel to fuse cursive letters into continuous line blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 7))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours strictly top-to-bottom
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda b: b[1])

    line_crops = []
    h_img, w_img = gray.shape

    # Filter out dust, specks, and isolate full text lines
    for x, y, w, h in bounding_boxes:
        if w > 40 and h > 18:
            pad_y = max(0, y - 6)
            pad_x = max(0, x - 8)
            crop = image[
                pad_y : min(h_img, y + h + 6), pad_x : min(w_img, x + w + 8)
            ]
            line_crops.append(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            )

    # Save debug crops so you can see each segmented line
    debug_dir = Path("debug_crops")
    debug_dir.mkdir(exist_ok=True)
    for idx, crop in enumerate(line_crops):
        crop.save(debug_dir / f"line_{idx:02d}.png")

    print(
        f"Segmented {len(line_crops)} text lines (saved previews to debug_crops/)."
    )
    return line_crops


def transcribe_image(image_path: str) -> str:
    processor, model, device = load_trocr()
    line_images = segment_lines(image_path)

    recognized_lines = []
    for idx, line_img in enumerate(line_images):
        pixel_values = processor(
            line_img, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=3,  # Beam search produces cleaner Spanish sentence syntax
                early_stopping=True,
            )
        line_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        recognized_lines.append(line_text.strip())

    return "\n".join(recognized_lines)


if __name__ == "__main__":
    test_img = "images/sample_invoice2.png"

    print(f"Transcribing {test_img} with fine-tuned Spanish model...")
    result = transcribe_image(test_img)

    print("\n--- Output Transcription ---")
    print(result)
    print("----------------------------\n")

    with open("handwriting_output.txt", "w", encoding="utf-8") as f:
        f.write(result)