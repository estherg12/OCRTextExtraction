import evaluate
import numpy as np
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

from datasets import load_dataset

# Loads Spanish marriage records (Esposalles) containing line images + transcripts
dataset = load_dataset("perellon/esposalles-ocr")

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "microsoft/trocr-base-handwritten"

    # Load processor using explicit RobertaTokenizer to prevent Windows fast-tokenizer errors
    image_processor = ViTImageProcessor.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(repo_id)
    model.to(device)

    # Configure model sequence generation settings
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Load dataset
    df = pd.read_csv("dataset/metadata.csv")
    df["file_path"] = "dataset/" + df["file_name"]

    # 90/10 train/validation split
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = SpanishHandwritingDataset(train_df, processor)
    eval_dataset = SpanishHandwritingDataset(val_df, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./trocr_spanish_checkpoint",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=10,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),  # Speeds up training on NVIDIA GPUs
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Save final Spanish checkpoint
    trainer.save_model("./trocr_spanish_final")
    processor.save_pretrained("./trocr_spanish_final")
    print("Model saved to ./trocr_spanish_final")


if __name__ == "__main__":
    train()