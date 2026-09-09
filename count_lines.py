"""Counts the lines of handwritten text in a photo and measures each one's height.

Works without OCR. The ink is isolated from the paper, and words are merged into
horizontal streaks, giving a row-by-row ink profile. Lines are then cut apart at
the troughs of that profile, spaced by the line rhythm found in it. Each cut
is trimmed back to the ink it actually holds -- so a reported height covers the
ascenders and descenders of the line, not just its x-height.

Cutting at troughs rather than at a fixed ink level is what lets one page of
tightly ruled writing and one loosely spaced note be measured by the same code.
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

IMAGE_FILE = "images/test4.png"

# Tuning constants
WORK_WIDTH = 1200        # images are downscaled to this width before analysis
BACKGROUND_KERNEL = 41   # size of the blob used to estimate the paper brightness
MIN_PITCH = 8            # a line rhythm shorter than this many rows is not credible
PITCH_STRENGTH = 0.20    # autocorrelation peak needed to accept a line rhythm
TROUGH_SPACING = 0.30    # a cut is looked for within this fraction of a pitch
INK_THRESHOLD = 0.08     # a row counts as "ink" above this fraction of the peak
EDGE_THRESHOLD = 0.02    # faint rows above this fraction of the peak still count as ink
BLANK_THRESHOLD = 0.03   # cuts peaking below this fraction of the peak hold no text
MIN_BAND_RATIO = 0.35    # lines thinner than this fraction of the median are specks
MIN_GAP_RATIO = 0.50     # gaps thinner than this fraction of the typical gap merge
SLIVER_RATIO = 0.35      # bands covering less of the width than this are stroke slivers


class TextLine(NamedTuple):
    """One detected line of text, measured in original-photo pixels."""
    top: int
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    """Returns a mask where the handwriting is white (255) and the paper is black."""
    # Photographed paper has uneven lighting, so estimate the local background
    # brightness and divide it out before thresholding globally.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BACKGROUND_KERNEL, BACKGROUND_KERNEL))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(gray, background, scale=255)

    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Drop specks of paper texture that survived the threshold.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return ink


def _merge_words(ink: np.ndarray) -> np.ndarray:
    """Closes the gaps between letters and words so each line becomes one streak."""
    width = ink.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 25, 9), 1))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)


def _row_profile(mask: np.ndarray) -> np.ndarray:
    """Counts the ink pixels on each row, smoothed so ragged rows do not split lines."""
    profile = (mask > 0).sum(axis=1).astype(np.float32)
    window = max(int(round(mask.shape[0] * 0.005)) * 2 + 1, 3)
    return cv2.GaussianBlur(profile.reshape(-1, 1), (1, window), 0).ravel()


def _line_pitch(profile: np.ndarray) -> float | None:
    """Estimates the rows between consecutive baselines, or None if there is no rhythm."""
    signal = profile - profile.mean()
    if not signal.any():
        return None

    correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    # Walk down out of the zero-lag peak, then take the strongest lag left: for
    # evenly written lines that lag is the distance from one line to the next.
    lag = 1
    while lag < len(correlation) - 1 and correlation[lag] <= correlation[lag - 1]:
        lag += 1
    search = correlation[lag:len(correlation) // 2 + 1]
    if len(search) == 0:
        return None

    pitch = lag + int(np.argmax(search))
    if correlation[pitch] < PITCH_STRENGTH or pitch < MIN_PITCH:
        return None

    # The strongest lag may be a multiple of the true pitch, so prefer a half
    # that is almost as strong -- otherwise every other line would be missed.
    while pitch // 2 >= MIN_PITCH and correlation[pitch // 2] > correlation[pitch] * 0.6:
        pitch //= 2
    return float(pitch)


def _cut_at_troughs(profile: np.ndarray, pitch: float) -> list[tuple[int, int]]:
    """Splits the profile into one segment per line by tracking down its troughs."""
    ink_rows = np.flatnonzero(profile > profile.max() * BLANK_THRESHOLD)
    if len(ink_rows) == 0:
        return []

    # Walk down the page one line at a time: from the line just cut, the next
    # boundary has to lie roughly one pitch further on, so take the emptiest row
    # in that window. Following the rhythm instead of ranking every dip on the
    # page keeps a shallow dip inside a line from being mistaken for a gap, and
    # lets the walk absorb the wider blank between two paragraphs.
    near = max(int(round(pitch * (1.0 - TROUGH_SPACING / 2))), 1)
    far = max(int(round(pitch * (1.0 + TROUGH_SPACING / 2))), near + 1)

    cuts = [0]
    row = int(ink_rows[0])
    while row < int(ink_rows[-1]):
        low, high = row + near, min(row + far, len(profile))
        if high <= low:
            break
        row = low + int(np.argmin(profile[low:high]))
        cuts.append(row)

    cuts.append(len(profile))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


def _threshold_bands(profile: np.ndarray) -> list[tuple[int, int]]:
    """Finds the row ranges whose ink clears a share of the peak, for arrhythmic pages."""
    if profile.max() == 0:
        return []

    is_ink = profile > profile.max() * INK_THRESHOLD
    bands = []
    start = None
    for row, filled in enumerate(is_ink):
        if filled and start is None:
            start = row
        elif not filled and start is not None:
            bands.append((start, row))
            start = None
    if start is not None:
        bands.append((start, len(is_ink)))
    return bands


def _width_coverage(streaks: np.ndarray, band: tuple[int, int]) -> float:
    """Returns the fraction of the image width that a band puts ink on."""
    start, end = band
    return float((streaks[start:end] > 0).any(axis=0).mean())


def _merge_slivers(bands: list[tuple[int, int]], streaks: np.ndarray) -> list[tuple[int, int]]:
    """Reunites a line with the thin sliver its ascenders or descenders were cut into."""
    if len(bands) < 2:
        return bands

    median_height = float(np.median([end - start for start, end in bands]))
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
    typical_gap = float(np.median(gaps)) if len(gaps) >= 3 else median_height

    # A row of ascender tips or descender tails reaches across only a little of
    # the width, whereas even a one-word line is a solid streak. That makes
    # coverage the signal for "leftover stroke, not a line", and unlike an ink
    # density test it does not punish short lines.
    coverage = [_width_coverage(streaks, band) for band in bands]
    sliver_limit = max(coverage) * SLIVER_RATIO

    merged = [bands[0]]
    was_sliver = coverage[0] < sliver_limit
    for band, cover in zip(bands[1:], coverage[1:]):
        start, end = band
        prev_start, prev_end = merged[-1]
        is_sliver = cover < sliver_limit
        # Demanding a small gap as well keeps two genuinely tight lines apart.
        if (is_sliver or was_sliver) and start - prev_end < typical_gap * MIN_GAP_RATIO:
            merged[-1] = (prev_start, end)
            was_sliver = is_sliver and was_sliver
        else:
            merged.append(band)
            was_sliver = is_sliver
    return merged


def _trim_to_ink(segments: list[tuple[int, int]], profile: np.ndarray) -> list[tuple[int, int]]:
    """Shrinks each segment to the rows it really holds ink on, dropping blank ones."""
    if profile.max() == 0:
        return []

    floor = profile.max() * EDGE_THRESHOLD
    trimmed = []
    for start, end in segments:
        rows = np.flatnonzero(profile[start:end] > floor)
        if len(rows) == 0:
            continue
        trimmed.append((start + int(rows[0]), start + int(rows[-1]) + 1))
    return trimmed


def _drop_specks(lines: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Discards bands far too thin to be a line, such as a lone accent or smudge."""
    if not lines:
        return []

    median_height = float(np.median([end - start for start, end in lines]))
    return [(s, e) for s, e in lines if e - s >= median_height * MIN_BAND_RATIO]


def _find_lines(ink: np.ndarray) -> list[tuple[int, int]]:
    """Locates every line of text in an ink mask and returns its row extent."""
    # Words are merged into streaks so that a line reads as one bump in the
    # profile regardless of how its words are spaced.
    streaks = _merge_words(ink)
    body_profile = _row_profile(streaks)
    pitch = _line_pitch(body_profile)

    if pitch is not None:
        # Several lines with a steady rhythm: cut between them at the troughs.
        segments = _cut_at_troughs(body_profile, pitch)
        # Keep only the cuts that hold real ink; the rest are margins and the
        # blank space between paragraphs.
        keep = body_profile.max() * BLANK_THRESHOLD
        segments = [(a, b) for a, b in segments if body_profile[a:b].max() > keep]
    else:
        # No rhythm to lock on to, which usually means a single line. Fall back
        # to a plain ink threshold and stitch the stray strokes back on.
        segments = _merge_slivers(_threshold_bands(body_profile), streaks)

    # Extents are measured on the raw ink, where a lone thin stroke such as an
    # accent or a descender tail still registers.
    return _drop_specks(_trim_to_ink(segments, _row_profile(ink)))


def measure_text_lines(image_path: str, debug_path: str | None = None) -> list[TextLine]:
    """Finds every line of text in the image and measures its height in photo pixels."""
    target_path = Path(image_path)
    if not target_path.exists():
        target_path = Path("images") / image_path
    if not target_path.exists():
        raise FileNotFoundError(f"Could not find image '{image_path}' anywhere.")

    image = cv2.imread(str(target_path))
    if image is None:
        raise ValueError(f"'{target_path}' is not a readable image.")

    original_height = image.shape[0]
    scale = min(1.0, WORK_WIDTH / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    bands = _find_lines(_binarize_ink(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

    if debug_path:
        _write_debug_image(image, bands, debug_path)

    # Report in the coordinates of the photo the user passed in, not the
    # downscaled working copy.
    factor = original_height / image.shape[0]
    return [TextLine(round(top * factor), round(bottom * factor)) for top, bottom in bands]


def count_text_lines(image_path: str, debug_path: str | None = None) -> int:
    """Counts the lines of text in the image, optionally saving an annotated copy."""
    return len(measure_text_lines(image_path, debug_path))


def _write_debug_image(image: np.ndarray, bands: list[tuple[int, int]], debug_path: str) -> None:
    """Saves a copy of the working image with every measured line boxed and labelled."""
    annotated = image.copy()
    for index, (top, bottom) in enumerate(bands, start=1):
        cv2.rectangle(annotated, (0, top), (annotated.shape[1] - 1, bottom - 1), (0, 0, 255), 1)
        cv2.putText(annotated, f"{index}: {bottom - top}px", (4, max(top + 12, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(debug_path, annotated)
    print(f"Annotated image written to '{debug_path}'.")


def _print_report(image_path: str, lines: list[TextLine]) -> None:
    """Prints the line count followed by the height of each individual line."""
    print(f"\nLines of text detected in '{image_path}': {len(lines)}")
    if not lines:
        return

    print(f"\n{'Line':>4}  {'Top':>6}  {'Bottom':>6}  {'Height':>7}")
    for index, line in enumerate(lines, start=1):
        print(f"{index:>4}  {line.top:>6}  {line.bottom:>6}  {line.height:>5} px")

    heights = [line.height for line in lines]
    print(f"\nTallest: {max(heights)} px   Shortest: {min(heights)} px   "
          f"Average: {round(sum(heights) / len(heights))} px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the lines of text in a photo and show the height of each line.")
    parser.add_argument("image", nargs="?", default=IMAGE_FILE,
                        help=f"image path or file name under images/ (default: {IMAGE_FILE})")
    parser.add_argument("--debug", metavar="PATH", nargs="?", const="output/line_boxes.png",
                        help="save a copy with the measured lines boxed in red")
    args = parser.parse_args()

    _print_report(args.image, measure_text_lines(args.image, args.debug))
