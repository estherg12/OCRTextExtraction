import gradio as gr
from PIL import Image
import torch
from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

# Load your local weights
CHECKPOINT_DIR = "./trocr_spanish_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = ViTImageProcessor.from_pretrained(CHECKPOINT_DIR)
tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT_DIR)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()


def transcribe_crop(image):
    if image is None:
        return "Upload an image / Sube una imagen"

    pil_img = Image.fromarray(image).convert("RGB")
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(
        device
    )

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values, max_new_tokens=64, num_beams=3, early_stopping=True
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


demo = gr.Interface(
    fn=transcribe_crop,
    inputs=gr.Image(
        type="numpy",
        label="Single Line Handwriting Crop / Recorte de línea manuscrita",
    ),
    outputs=gr.Textbox(label="Detected Transcription / Transcripción Detectada"),
    title="TrOCR Spanish Handwriting Reader",
    description="Locally hosted demo testing the fine-tuned Spanish model checkpoint.",
)

if __name__ == "__main__":
    # share=True creates a public temporary https://xxxx.gradio.live link for 72 hours
    demo.launch(share=False)