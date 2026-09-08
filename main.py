from pathlib import Path
import easyocr
from fpdf import FPDF

def extract_handwriting(image_path: str, languages: list = ["es", "en"]) -> str:
    """Extract handwritten text using EasyOCR."""
    # Initializes reader (uses GPU if CUDA is available, otherwise CPU)
    reader = easyocr.Reader(languages, gpu=False)

    # paragraph=True groups detected words into coherent lines/sentences
    results = reader.readtext(image_path, detail=0, paragraph=True)

    return "\n".join(results)


def export_to_txt(text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text to: {output_path}")


def export_to_pdf(text: str, output_path: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Use Helvetica or load a TTF font if you have special symbols
    pdf.set_font("Helvetica", size=11)

    for line in text.splitlines():
        # Encode as latin-1 to avoid standard PDF font encoding errors on Spanish accents
        clean_line = line.encode("latin-1", "replace").decode("latin-1")
        if clean_line.strip():
            pdf.multi_cell(w=pdf.epw, h=8, text=clean_line)
        else:
            pdf.ln(4)

    pdf.output(output_path)
    print(f"Saved PDF to: {output_path}")


if __name__ == "__main__":
    image_file = "images/sample_invoice2.png"
    base_name = Path(image_file).stem

    print("Running handwriting recognition...")
    extracted_text = extract_handwriting(image_file, languages=["es"])

    print("\n--- Extracted Text ---")
    print(extracted_text)
    print("----------------------\n")

    export_to_txt(extracted_text, f"{base_name}_transcription.txt")
    export_to_pdf(extracted_text, f"{base_name}_transcription.pdf")