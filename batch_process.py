import os
import sys
from pathlib import Path
from count_lines import measure_text_lines, crop_and_save_lines, TextLine

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def get_image_files(directory_path: str | Path) -> list[Path]:
    """
    Returns a sorted list of all supported image files (.jpg, .jpeg, .png)
    found inside directory_path (non-recursive).
    """
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    images = [
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name)


def cleanup_line_crops(output_dir: str | Path = "output", prefix: str | None = None) -> None:
    """Deletes temporary cropped line images while preserving final transcriptions."""
    out = Path(output_dir)
    if not out.exists():
        return
    for item in out.glob("*.png"):
        if "_line_" in item.name:
            if prefix is None or item.name.startswith(f"{prefix}_line_"):
                try:
                    item.unlink()
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")


def export_to_txt(lines: list[str], output_path: str | Path) -> None:
    """Export lines to a UTF-8 text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved transcript to: {output_path}")


def export_to_pdf(lines: list[str], output_path: str | Path) -> None:
    """Export lines to a PDF file handling Spanish characters and accents."""
    from fpdf import FPDF

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF(format='letter')
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)

    for line in lines:
        pdf.cell(200, 10, text=line, align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln()

    pdf.output(str(output_path))
    print(f"Saved transcript to: {output_path}")


def transcribe_lines(
    line_image_paths: list[str | Path],
    processor=None,
    model=None,
    device: str = "cpu",
    transcribe_fn=None,
) -> tuple[list[str], list[str]]:
    """
    Transcribes a list of line image crops into raw and NLP-corrected texts.
    Accepts either a custom transcribe_fn(img_path, i) or (processor, model, device).
    """
    raw_lines: list[str] = []
    nlp_lines: list[str] = []

    for i, line_img in enumerate(line_image_paths, start=1):
        if transcribe_fn is not None:
            raw, nlp = transcribe_fn(str(line_img), i)
        else:
            from main import test_single_phrase
            raw, nlp = test_single_phrase(str(line_img), processor, model, i, device=device)
        raw_lines.append(raw)
        nlp_lines.append(nlp)

    return raw_lines, nlp_lines


def process_image(
    image_path: str | Path,
    processor=None,
    model=None,
    output_dir: str | Path = "output",
    device: str = "cpu",
    transcribe_fn=None,
) -> dict[str, str]:
    """
    Processes a single image:
    1. Measures text lines using measure_text_lines()
    2. Crops and saves line segments into output_dir
    3. Transcribes each line segment
    4. Cleans up intermediate line crops
    5. Exports raw and NLP transcriptions as .txt and .pdf named after the input image
    """
    img_path = Path(image_path)
    if not img_path.exists():
        fallback = Path("images") / image_path
        if fallback.exists():
            img_path = fallback
        else:
            raise FileNotFoundError(f"Could not find image '{image_path}'.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing Image: {img_path.name} ---")
    detected_lines = measure_text_lines(str(img_path))
    print(f"Detected {len(detected_lines)} line(s).")

    if not detected_lines:
        print(f"No lines detected in {img_path.name}.")
        return {}

    crop_and_save_lines(str(img_path), detected_lines, output_dir=str(out_dir))

    base_name = img_path.stem
    line_image_paths = [
        out_dir / f"{base_name}_line_{i}.png"
        for i in range(1, len(detected_lines) + 1)
    ]

    raw_lines, nlp_lines = transcribe_lines(
        line_image_paths,
        processor=processor,
        model=model,
        device=device,
        transcribe_fn=transcribe_fn,
    )

    # Clean up intermediate line crops
    cleanup_line_crops(output_dir=out_dir, prefix=base_name)

    txt_raw = out_dir / f"{base_name}_transcription_raw.txt"
    pdf_raw = out_dir / f"{base_name}_transcription_raw.pdf"
    txt_nlp = out_dir / f"{base_name}_transcription_nlp.txt"
    pdf_nlp = out_dir / f"{base_name}_transcription_nlp.pdf"

    export_to_txt(raw_lines, txt_raw)
    export_to_pdf(raw_lines, pdf_raw)
    export_to_txt(nlp_lines, txt_nlp)
    export_to_pdf(nlp_lines, pdf_nlp)

    return {
        "txt_raw": str(txt_raw),
        "pdf_raw": str(pdf_raw),
        "txt_nlp": str(txt_nlp),
        "pdf_nlp": str(pdf_nlp),
    }


def batch_process(
    directory_path: str | Path,
    processor=None,
    model=None,
    output_dir: str | Path = "output",
    device: str = "cpu",
    transcribe_fn=None,
) -> list[dict[str, str]]:
    """
    Processes all supported images in a directory.
    """
    image_files = get_image_files(directory_path)
    print(f"Found {len(image_files)} image(s) to process in '{directory_path}'.")

    results = []
    for img_file in image_files:
        res = process_image(
            img_file,
            processor=processor,
            model=model,
            output_dir=output_dir,
            device=device,
            transcribe_fn=transcribe_fn,
        )
        if res:
            results.append(res)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process directory of images for OCR line extraction and transcription.")
    parser.add_argument("directory", nargs="?", default="images", help="Path to folder containing images (default: images)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--model-dir", "-m", default="./trocr_spanish_final", help="Path to local TrOCR model directory")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Local model directory '{args.model_dir}' missing.")

    import torch
    from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from '{args.model_dir}' on {device}...")
    image_processor = ViTImageProcessor.from_pretrained(args.model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    batch_process(args.directory, processor=processor, model=model, output_dir=args.output, device=device)
