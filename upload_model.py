from huggingface_hub import HfApi

api = HfApi()

# Replace it with your actual Hugging Face username and chosen repository name
REPO_ID = "ifesther/trocr-spanish-handwritten"

print(f"Uploading trocr_spanish_final to https://huggingface.co/{REPO_ID}...")

api.upload_folder(
    folder_path="trocr_spanish_final",
    repo_id=REPO_ID,
    repo_type="model",
)

print("Upload complete!")