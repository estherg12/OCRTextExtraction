# OCRTextExtraction: Modern Spanish Handwriting OCR

An end-to-end Python pipeline capable of transcribing **modern handwritten Spanish text** (including cursive styles, accents `á, é, í, ó, ú`, and `ñ`) from raw images, exporting structured results to `.txt` and `.pdf`.

Powered by a fine-tuned Vision-Encoder-Decoder model hosted on Hugging Face:  
**Model Weights:** 
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ifesther/trocr-spanish-handwritten)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Key Features

- **Zero English Hallucination:** Standard TrOCR models default to English syntax and hallucinate random English text when processing Spanish sentences. This model is fine-tuned specifically on modern Spanish vocabulary and orthography.
- **Diacritics & Special Glyphs:** Full character coverage for `á, é, í, ó, ú, ü`, `ñ`, and punctuation (`¿`, `¡`).
- **OpenCV Line Segmentation:** Automated morphological text-line detection and isolation from full documents.
- **Export Formats:** Direct export to clean `.txt` transcripts and formatted `.pdf` reports.

---

## Installation & Setup

1. **Clone the repository:**
   ```
   git clone [https://github.com/estherg12/OCRTextExtraction.git](https://github.com/estherg12/OCRTextExtraction.git)
   cd OCRTextExtraction
   ```
2. Create and activate a virtual environment:
   ```python -m venv .venv```
   - Windows: 
  ```.venv\Scripts\activate```
   - Linux/macOS:
  ```source .venv/bin/activate```
3. Install dependencies:
  ```pip install -r requirements.txt```
4. Run Transcription: transcribe an image containing handwriting 
```
python main.py --image samples/test_note.png --format pdf
```

---

## Model Checkpoint
The core engine is based on a Vision-Encoder-Decoder fine-tuned on synthetic Spanish notebook text generated across 41 handwriting fonts.
- **Hugging Face Hub**: [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ifesther/trocr-spanish-handwritten)
- **Base Checkpoint**: ```microsoft/trocr-base-handwritten```
You can load the model directly in Python with Transformers:
```
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained(
    "ifesther/trocr-spanish-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "ifesther/trocr-spanish-handwritten"
)
```

---

## Repository Architecture
- ```main.py```: entry point for full-page processing and document export (.txt / .pdf).
- ```transcribe_handwriting.py```: line segmentation and TrOCR inference pipeline.
- ```generate_synthetic_dataset.py```: synthetic generator used to create diverse handwriting line crops.
- ```train_trocr.py```: fine-tuning script (only needed if retraining from scratch).

---

## License
This project is licensed under the MIT License.
