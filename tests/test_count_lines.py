import pytest
from count_lines import count_text_lines

# Dictionary of test images and their verified line counts.
# Ensure these specific images are committed to a 'tests/test_images/' or 'images/' folder.
EXPECTED_COUNTS = {
    "phrase1.jpg": 1,
    "test1.jpg": 4,
    "test3.png": 3,
    "test4.png": 6,
    "test6.png": 12,
}

@pytest.mark.parametrize("image_filename, expected_lines", EXPECTED_COUNTS.items())
def test_line_counting_accuracy(image_filename, expected_lines):
    """
    Verifies that the horizontal projection profiling accurately 
    counts the number of text lines in a given image.
    """
    # Call the segmentation counter
    actual_lines = count_text_lines(image_filename)
    
    # Assert the output matches the verified manual count
    assert actual_lines == expected_lines, (
        f"Segmentation failure on {image_filename}: "
        f"Expected {expected_lines} lines, but detected {actual_lines}."
    )
