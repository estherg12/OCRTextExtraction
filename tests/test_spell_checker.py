import pytest

from main import correct_spanish_text


def test_preserves_acronyms():
    text = "La NASA estudia Marte"
    result = correct_spanish_text(text)

    assert "NASA" in result


def test_preserves_proper_nouns():
    text = "García vive en Madrid"
    result = correct_spanish_text(text)

    assert "García" in result


def test_preserves_mixed_case_names():
    text = "McDonald trabaja aquí"
    result = correct_spanish_text(text)

    assert "McDonald" in result


def test_preserves_hyphenated_names():
    text = "García-López llegó ayer"
    result = correct_spanish_text(text)

    assert "García-López" in result


def test_preserves_alphanumeric_tokens():
    text = "El vuelo A320 llegó"
    result = correct_spanish_text(text)

    assert "A320" in result


def test_preserves_numbers():
    text = "Tengo 25 años"
    result = correct_spanish_text(text)

    assert "25" in result