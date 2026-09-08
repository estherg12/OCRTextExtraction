from pathlib import Path
import cv2
from fpdf import FPDF
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def preprocess_image(image_path: str):
    """Enhance image contrast and remove noise for better OCR accuracy."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Denoise and threshold (Otsu's thresholding works well for text on paper)
    denoised = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


def extract_text(image_path: str, preprocess: bool = True) -> str:
    """Extract text from an image path using pytesseract."""
    if preprocess:
        processed_img = preprocess_image(image_path)
        # Convert OpenCV numpy array back to PIL Image
        pil_img = Image.fromarray(processed_img)
    else:
        pil_img = Image.open(image_path)

    # --oem 3: Default LSTM engine; --psm 3: Fully automatic page segmentation
    custom_config = r"--oem 3 --psm 3"
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    return text.strip()


def export_to_txt(text: str, output_path: str):
    """Save extracted text to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text to: {output_path}")


def export_to_pdf(text: str, output_path: str):
    """Save extracted text into a structured .pdf file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    # Clean and write non-empty lines
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            # Note: parameter is now 'text' instead of 'txt'
            # Using pdf.epw (effective page width) guarantees fitting inside margins
            pdf.multi_cell(w=pdf.epw, h=8, text=line)
        else:
            # Add a vertical gap for blank lines/paragraphs
            pdf.ln(4)

    pdf.output(output_path)
    print(f"Saved PDF to: {output_path}")


def process_document(image_path: str, output_format: str = "both"):
    """Main pipeline: image -> OCR -> export (.txt, .pdf, or both)."""
    input_file = Path(image_path)
    base_name = input_file.stem

    print(f"Processing '{input_file.name}'...")
    extracted_text = extract_text(str(input_file), preprocess=True)

    if not extracted_text:
        print("Warning: No text was detected in the image.")
        return

    if output_format in ("txt", "both"):
        export_to_txt(extracted_text, f"{base_name}_transcription.txt")

    if output_format in ("pdf", "both"):
        export_to_pdf(extracted_text, f"{base_name}_transcription.pdf")


if __name__ == "__main__":
    # Test on your own image
    sample_image = "sample_invoice.png"
    process_document(sample_image, output_format="both")