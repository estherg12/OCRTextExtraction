"""Checks count_lines against hand-counted line totals for every image in images/.

The expected counts were read off the photos by eye; run this after touching any
of the tuning constants in count_lines to see what the change costs elsewhere.
"""

from count_lines import measure_text_lines

EXPECTED = {
    "phrase1.jpg": 1, "phrase2.jpg": 1, "phrase3.jpg": 1, "phrase4.jpg": 1,
    "phrase5.jpg": 1, "phrase6.jpg": 1, "phrase7.png": 1, "phrase8.png": 1,
    "phrase9.png": 1, "phrase10.png": 1, "phrase11.png": 1, "phrase12.png": 1,
    "phrase13.png": 1, "phrase14.png": 1, "phrase15.png": 1, "phrase16.png": 1,
    "test1.jpg": 4, "test2.png": 4, "test3.png": 3, "test4.png": 6,
    "test5.png": 4, "test6.png": 12, "test7.png": 10, "test8.png": 5,
    "test9.png": 8, "test10.png": 11, "test11.png": 16, "test12.png": 10,
    "test13.png": 14,
}

if __name__ == "__main__":
    failures = []
    for name, expected in EXPECTED.items():
        got = len(measure_text_lines(f"images/{name}"))
        if got != expected:
            failures.append(f"{name}: expected {expected}, got {got}")
        print(f"{'ok  ' if got == expected else 'FAIL'} {name:<16} "
              f"expected={expected:<3} got={got}")

    print(f"\n{len(EXPECTED) - len(failures)}/{len(EXPECTED)} correct")
    for failure in failures:
        print(f"  {failure}")
