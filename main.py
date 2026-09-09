import os, shutil
from pathlib import Path
import re
import torch
from PIL import Image
from fpdf import FPDF
from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)
from spellchecker import SpellChecker
from count_lines import run_as_main

# Target a single-phrase test image directly
IMAGE_FILE = "test3"
LOCAL_MODEL_DIR = "./trocr_spanish_final"

def correct_spanish_text(text: str) -> str:
    """Certifies words against the Spanish vocabulary and corrects typos/confusions."""
    spell = SpellChecker(language='es')
    words = text.split()
    corrected_words = []

    for word in words:
        # Extract alphanumeric content to check against dictionary
        clean_word = re.sub(r'[^\wáéíóúüñÁÉÍÓÚÜÑ]', '', word)
        if clean_word and not clean_word.isdigit():
            # If the word is unknown/misspelled, find the most probable valid Spanish substitute
            unknowns = spell.unknown([clean_word.lower()])
            if clean_word.lower() in unknowns:
                correction = spell.correction(clean_word.lower())
                if correction:
                    if clean_word.istitle():
                        correction = correction.capitalize()
                    elif clean_word.isupper():
                        correction = correction.upper()
                    word = word.replace(clean_word, correction)
        corrected_words.append(word)

    return " ".join(corrected_words)

def test_single_phrase(image_path: str, processor: TrOCRProcessor, model: VisionEncoderDecoderModel, i: int) -> tuple[str, str]:
    # Locate image
    target_path = image_path
    if not Path(target_path).exists():
        target_path = f"images/{image_path}"
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Could not find image '{image_path}' anywhere.")

    print(f"Predicting text for single-phrase image: '{target_path}'...")
    img = Image.open(target_path).convert("RGB")

    # Raw single-pass inference without any extra preprocessing
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=32,
            num_beams=1,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    corrected_text = correct_spanish_text(text)

    """
    print(f"\nLine {i}")
    print(f"Raw OCR Output    : {text.strip()}")
    print(f"NLP Certified Text: {corrected_text.strip()}")
    """

    return text, corrected_text

def delete_output_content():
    folder = "output"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def export_to_txt(lines: list[str], output_path: str):
    """Export lines to a UTF-8 text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved transcript to: {output_path}")


def export_to_pdf(lines: list[str], output_path: str):
    """Export lines to a PDF file handling Spanish characters and accents."""
    pdf = FPDF(format='letter')
    #pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)

    for line in lines:
        pdf.cell(200, 10, text=line,
                 ln=1, align='C')
        pdf.ln()

    pdf.output(output_path)
    print(f"Saved transcript to: {output_path}")

if __name__ == "__main__":
    detected_lines = run_as_main("images/"+IMAGE_FILE+".png")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(f"Local model directory '{LOCAL_MODEL_DIR}' missing.")

    print(f"\nLoading local model from '{LOCAL_MODEL_DIR}' on {device}...")

    # Load processor, tokenizer, and model directly
    image_processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = RobertaTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(LOCAL_MODEL_DIR).to(device)
    model.eval()

    RAW_LINES = []
    NLP_LINES = []

    for i, line in enumerate(detected_lines, start=1):
        txt_tuple = test_single_phrase(f"output/{IMAGE_FILE}_line_{i}.png", processor, model, i)

        raw_text = txt_tuple[0]
        corrected_text = txt_tuple[1]

        RAW_LINES.append(raw_text)
        NLP_LINES.append(corrected_text)

    print(f"\nRaw Lines:\n{RAW_LINES}")
    print(f"\nNLP Lines:\n{NLP_LINES}")

    delete_output_content()
    print("\nOutput content deleted.")

    base_name = Path(IMAGE_FILE).stem

    txt_out_raw = f"output/{base_name}_transcription_raw.txt"
    pdf_out_raw = f"output/{base_name}_transcription_raw.pdf"
    txt_out_nlp = f"output/{base_name}_transcription_nlp.txt"
    pdf_out_nlp = f"output/{base_name}_transcription_nlp.pdf"

    export_to_txt(RAW_LINES, txt_out_raw)
    export_to_pdf(RAW_LINES, pdf_out_raw)
    export_to_txt(NLP_LINES, txt_out_nlp)
    export_to_pdf(NLP_LINES, pdf_out_nlp)

    print("Transcription files saved.")