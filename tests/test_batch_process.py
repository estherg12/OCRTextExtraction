from pathlib import Path
import pytest
from batch_process import (
    get_image_files,
    cleanup_line_crops,
    export_to_txt,
    export_to_pdf,
    transcribe_lines,
    process_image,
    batch_process,
)


def test_get_image_files(tmp_path):
    # Create supported and unsupported test files
    (tmp_path / "img1.png").touch()
    (tmp_path / "img2.JPG").touch()
    (tmp_path / "img3.jpeg").touch()
    (tmp_path / "doc.txt").touch()
    (tmp_path / "notes.pdf").touch()
    (tmp_path / "subfolder").mkdir()

    files = get_image_files(tmp_path)
    filenames = [f.name for f in files]

    assert len(files) == 3
    assert filenames == ["img1.png", "img2.JPG", "img3.jpeg"]


def test_get_image_files_nonexistent():
    with pytest.raises(FileNotFoundError):
        get_image_files("/nonexistent/directory/path/12345")


def test_get_image_files_not_a_directory(tmp_path):
    file_path = tmp_path / "single_file.png"
    file_path.touch()
    with pytest.raises(NotADirectoryError):
        get_image_files(file_path)


def test_cleanup_line_crops(tmp_path):
    (tmp_path / "test_line_1.png").touch()
    (tmp_path / "test_line_2.png").touch()
    (tmp_path / "other_line_1.png").touch()
    (tmp_path / "test_transcription_raw.txt").touch()
    (tmp_path / "test_transcription_raw.pdf").touch()

    # Clean only 'test' prefix
    cleanup_line_crops(tmp_path, prefix="test")
    remaining = [f.name for f in tmp_path.iterdir()]

    assert "test_line_1.png" not in remaining
    assert "test_line_2.png" not in remaining
    assert "other_line_1.png" in remaining
    assert "test_transcription_raw.txt" in remaining
    assert "test_transcription_raw.pdf" in remaining


def test_export_to_txt_and_pdf(tmp_path):
    lines = ["Primera línea", "Segunda línea con acento: áéíóú", "Tercera línea: ñ y ¿preguntas?"]
    txt_path = tmp_path / "test.txt"
    pdf_path = tmp_path / "test.pdf"

    export_to_txt(lines, txt_path)
    assert txt_path.exists()
    content = txt_path.read_text(encoding="utf-8")
    assert "Segunda línea con acento: áéíóú" in content

    export_to_pdf(lines, pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_export_to_pdf_unicode_punctuation(tmp_path):
    # Tests non-Latin-1 characters like em-dashes and curly quotes
    lines = ["Nota con guión — y comillas “españolas”", "Puntos suspensivos… y viñeta •"]
    pdf_path = tmp_path / "unicode_test.pdf"

    export_to_pdf(lines, pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_transcribe_lines_with_custom_fn():
    mock_images = ["output/sample_line_1.png", "output/sample_line_2.png"]

    def mock_transcribe(img_path, idx):
        return f"Raw text {idx}", f"NLP text {idx}"

    raw, nlp = transcribe_lines(mock_images, transcribe_fn=mock_transcribe)
    assert raw == ["Raw text 1", "Raw text 2"]
    assert nlp == ["NLP text 1", "NLP text 2"]


def test_process_image_end_to_end(tmp_path):
    output_dir = tmp_path / "output"
    sample_image = "images/test3.png"

    def mock_transcribe(img_path, idx):
        return f"Raw line {idx}", f"Corrected line {idx}"

    result = process_image(
        sample_image,
        output_dir=output_dir,
        transcribe_fn=mock_transcribe,
    )

    assert result != {}
    assert "txt" in result
    assert "pdf" in result
    assert "txt_raw" in result
    assert "pdf_raw" in result
    assert "txt_nlp" in result
    assert "pdf_nlp" in result

    # Check generated files
    assert Path(result["txt"]).exists()
    assert Path(result["pdf"]).exists()
    assert Path(result["txt_raw"]).exists()
    assert Path(result["pdf_raw"]).exists()
    assert Path(result["txt_nlp"]).exists()
    assert Path(result["pdf_nlp"]).exists()

    # Ensure output files are named exactly after the input image stem
    assert result["txt"].endswith("test3.txt")
    assert result["pdf"].endswith("test3.pdf")
    assert "test3_transcription_raw.txt" in result["txt_raw"]
    assert "test3_transcription_raw.pdf" in result["pdf_raw"]

    # Verify transcript contents
    main_content = Path(result["txt"]).read_text(encoding="utf-8")
    assert "Corrected line 1" in main_content
    raw_content = Path(result["txt_raw"]).read_text(encoding="utf-8")
    assert "Raw line 1" in raw_content

    # Ensure temporary crop images are cleaned up
    remaining_crops = list(output_dir.glob("*_line_*.png"))
    assert len(remaining_crops) == 0


def test_batch_process_directory(tmp_path):
    # Setup test input folder with copies of small test images
    in_dir = tmp_path / "input_images"
    in_dir.mkdir()
    out_dir = tmp_path / "output_dir"

    import shutil
    shutil.copy("images/test1.png", in_dir / "sample1.png")
    shutil.copy("images/test3.png", in_dir / "sample2.png")

    def mock_transcribe(img_path, idx):
        return f"Text {idx}", f"NLP {idx}"

    results = batch_process(
        in_dir,
        output_dir=out_dir,
        transcribe_fn=mock_transcribe,
    )

    assert len(results) == 2
    # Verify outputs for both images exist, including exact-named files
    assert (out_dir / "sample1.txt").exists()
    assert (out_dir / "sample1.pdf").exists()
    assert (out_dir / "sample1_transcription_raw.txt").exists()
    assert (out_dir / "sample1_transcription_raw.pdf").exists()
    assert (out_dir / "sample2.txt").exists()
    assert (out_dir / "sample2.pdf").exists()
    assert (out_dir / "sample2_transcription_raw.txt").exists()
    assert (out_dir / "sample2_transcription_raw.pdf").exists()
