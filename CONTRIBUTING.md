# Contributing to OCR Text Extraction

Thank you for your interest in improving this OCR pipeline! We welcome bug reports, feature requests, and pull requests.

## How to Contribute

1. **Fork the repository** and clone it locally.
2. **Set up your environment**:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Create a new branch** for your feature or bug fix:
   ```git checkout -b feature/your-feature-name```
4. **Test your changes** by running ```python main.py``` and ensuring the segmentation and transcription still work correctly on the sample images.
5. **Commit changes** with a clear, descriptive commit message.
6. **Push to your fork** and submit a Pull Request against the ```main``` branch.

## Areas we need Help with:
- Improving the OpenCV horizontal projection logic for highly irregular handwriting.
- Adding support for batch processing entire directories.
- Expanding the NLP dictionary corrections in ```clean_spanish_transcription()```.
