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
    """Load TrOCR bypassing the fast tokenizer conversion issue."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "microsoft/trocr-base-handwritten"

    # Explicitly load feature extractor and slow tokenizer separately
    image_processor = ViTImageProcessor.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)

    processor = TrOCRProcessor(
        image_processor=image_processor, tokenizer=tokenizer
    )
    model = VisionEncoderDecoderModel.from_pretrained(repo_id).to(device)

    return processor, model, device


def segment_lines(image_path: str):
    """Segment an unlined or handwritten page into individual line crops."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert and blur
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Dilate horizontally to connect cursive letters into solid text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find line contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours top-to-bottom
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda b: b[1])

    line_crops = []
    h_img, w_img = gray.shape

    # Save each crop to see what TrOCR is actually looking at
    Path("debug_crops").mkdir(exist_ok=True)
    for idx, crop in enumerate(line_crops):
        crop.save(f"debug_crops/crop_{idx}.png")

    for x, y, w, h in bounding_boxes:
        # Filter out minor specks or isolated dots
        if w > 30 and h > 15:
            # Add small padding around the crop
            pad_y = max(0, y - 5)
            pad_x = max(0, x - 5)
            crop = image[
                pad_y : min(h_img, y + h + 5), pad_x : min(w_img, x + w + 5)
            ]
            line_crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    return line_crops


def transcribe_image(image_path: str) -> str:
    processor, model, device = load_trocr()
    line_images = segment_lines(image_path)

    recognized_lines = []
    for line_img in line_images:
        pixel_values = processor(line_img, return_tensors="pt").pixel_values.to(
            device
        )
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
        line_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        recognized_lines.append(line_text)

    return "\n".join(recognized_lines)


if __name__ == "__main__":
    test_img = "images/sample_invoice2.png" if Path("images/sample_invoice2.png").exists() else "sample_invoice2.png"
    result = transcribe_image(test_img)
    print("--- Detected Text ---")
    print(result)

    with open("handwriting_output.txt", "w", encoding="utf-8") as f:
        f.write(result)