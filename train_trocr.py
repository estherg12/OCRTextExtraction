import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

# Verification
assert torch.cuda.is_available(), "GPU not found! Enable T4 GPU under Runtime > Change runtime type."
device = torch.device("cuda")
print(f"Using compute accelerator: {torch.cuda.get_device_name(0)}")

# Paths
DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
METADATA_PATH = DATASET_DIR / "metadata.tsv"
OUTPUT_DIR = Path("trocr_spanish_final")
CHECKPOINT_DIR = Path("trocr_temp_checkpoint")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# 3. Dataset definition
class SpanishHTRDataset(Dataset):
    def __init__(self, metadata_path, images_dir, processor, max_target_length=64):
        self.df = pd.read_csv(metadata_path, sep="\t").dropna().reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["file_name"]
        image = Image.open(img_path).convert("RGB")
        text = str(row["text"]).strip()

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# Model & Processor initialization
BASE_MODEL = "microsoft/trocr-base-handwritten"

image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Check if a temporary checkpoint exists to resume from
if (CHECKPOINT_DIR / "pytorch_model.bin").exists() or (CHECKPOINT_DIR / "model.safetensors").exists():
    print(f"Resuming training from temporary checkpoint in {CHECKPOINT_DIR}...")
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_DIR)
else:
    print(f"Loading fresh base architecture: {BASE_MODEL}...")
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

# Unfreeze encoder and decoder
for param in model.encoder.parameters():
    param.requires_grad = True
for param in model.decoder.parameters():
    param.requires_grad = True

model.to(device)

# Dataloader & Training Hyperparameters
train_dataset = SpanishHTRDataset(METADATA_PATH, IMAGES_DIR, processor)
BATCH_SIZE = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

EPOCHS = 2
LEARNING_RATE = 4e-5
total_training_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_training_steps * 0.08)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)

print(
    f"Total samples: {len(train_dataset)} | Steps per epoch: {len(train_loader)} | Total steps: {total_training_steps}")

# Training Loop with Periodic Checkpointing
model.train()
global_step = 0

for epoch in range(EPOCHS):
    running_loss = 0.0
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        global_step += 1

        # Save a backup checkpoint every 500 steps so you never lose progress
        if global_step % 500 == 0:
            print(f"Saving interim checkpoint at step {global_step}...")
            model.save_pretrained(CHECKPOINT_DIR)
            processor.save_pretrained(CHECKPOINT_DIR)

        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Step [{step + 1}/{len(train_loader)}] | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            running_loss = 0.0

# Final Export
print(f"\nTraining complete. Saving final model and processor to '{OUTPUT_DIR}'...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Saved successfully.")