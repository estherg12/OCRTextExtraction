import evaluate
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    DefaultDataCollator,
    RobertaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)

# Dataset Class
class SpanishHandwritingDataset(Dataset):
    def __init__(self, df, processor, max_target_length=64):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = Image.open(item["file_path"]).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize the Spanish ground truth string
        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # PyTorch CrossEntropyLoss ignores index -100 (padding tokens)
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# CER Metric Evaluation
cer_metric = evaluate.load("cer")

def compute_metrics(pred, tokenizer):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# Training Execution
def train():
    device = "cpu"
    repo_id = "microsoft/trocr-base-handwritten"

    # Load processor using explicit RobertaTokenizer to prevent Windows fast-tokenizer errors
    image_processor = ViTImageProcessor.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    processor = TrOCRProcessor(
        image_processor=image_processor, tokenizer=tokenizer
    )

    model = VisionEncoderDecoderModel.from_pretrained(repo_id)
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.to(device)

    # Configure model sequence generation settings
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Load dataset
    df = pd.read_csv("dataset/metadata.csv")
    df["file_path"] = "dataset/" + df["file_name"]

    # 90/10 train/validation split
    train_df = df.sample(n=1500, random_state=42)
    val_df = df.drop(train_df.index).sample(n=150, random_state=42)

    train_dataset = SpanishHandwritingDataset(train_df, processor)
    eval_dataset = SpanishHandwritingDataset(val_df, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./trocr_spanish_checkpoint",
        per_device_train_batch_size=8,  # Increased batch size reduces step overhead
        per_device_eval_batch_size=8,
        predict_with_generate=False,  # False to skip slow autoregressive eval steps on CPU
        eval_strategy="no",  # Skip eval during training to save time; evaluate only at the end
        save_strategy="epoch",
        logging_steps=25,
        num_train_epochs=2,  # 2 epochs is sufficient for the decoder to adapt
        learning_rate=1e-4,
        dataloader_num_workers=0,  # Windows stability
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
    )

    print(
        f"Starting accelerated CPU training using {num_cores} threads (Encoder Frozen)..."
    )
    trainer.train()

    # Save final Spanish checkpoint
    trainer.save_model("./trocr_spanish_final")
    processor.save_pretrained("./trocr_spanish_final")
    print("Model saved to ./trocr_spanish_final")


if __name__ == "__main__":
    train()