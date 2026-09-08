import csv
import glob
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Directories
OUTPUT_DIR = Path("dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Corrected & Expanded Modern Spanish Vocabulary
SUBJECTS = [
    "No te rindas,",
    "La mujer",
    "El hombre",
    "El amor de mi vida",
    "Tú",
    "La mujer que amas",
    "El hombre que amas",
    "Tu presencia",
    "El viaje",
    "Cada amanecer",
    "Cada atardecer",
    "La vida",
    "Nuestra memoria",
    "Vuestra memoria",
    "El camino",
    "Tu mirada",
    "Su mirada",
    "Agradezco",
    "Si buscas el éxito,",
    "El mayor tesoro",
    "La perseverancia",
    "Un corazón sincero",
    "Un corazón rebelde",
    "El tiempo compartido",
    "Tu sonrisa",
    "Su sonrisa",
    "La paciencia",
    "Caminar juntos",
    "Un buen recuerdo",
    "La libertad",
    "Escribir poemas",
    "Mi mujer soñada",
    "Mi hombre soñado",
    "Mi mujer",
    "Mi hombre",
    "Eres mi mejor compañía",
    "Eres la razón",
    "Tu amor",
    "Si quieres ser mi estrella,",
]

VERBS_PREDICATES = [
    "hace que todo sea más hermoso.",
    "me quiere más que a nada.",
    "me cuida como a una flor.",
    "huele a rosas frescas.",
    "sabe protegerme siempre.",
    "tiene un corazón rebelde.",
    "cuida sus flores cada día.",
    "pertenece a mi vida.",
    "sabe que todo vale la pena.",
    "me dice que todo saldrá bien.",
    "es un regalo que no debemos olvidar.",
    "abre senderos que antes no existían.",
    "ilumina cada rincón del alma.",
    "construye puentes sobre mares agitados.",
    "nos enseña a valorar lo simple.",
    "vence cualquier obstáculo con calma.",
    "guía los pasos hacia nuevos horizontes.",
    "deja huellas imborrables en el tiempo.",
    "merece vivirse con toda la intensidad.",
    "despierta pasiones dormidas.",
    "es la razón de sonreír a diario.",
    "transforma la rutina en pura magia.",
    "alcanza cimas que parecían lejanas.",
    "acompaña las noches de invierno.",
    "da sentido a los pequeños instantes.",
    "y mi mayor apoyo.",
    "por la que sonrío todos los días.",
    "ilumina mi vida por completo.",
    "prometo ser tu cielo.",
]

AUTHORS_TAGS = [
    "- Mario Benedetti.",
    "- Gabriel García Márquez.",
    "- Federico García Lorca.",
    "- Jorge Luis Borges.",
    "- Isabel Allende.",
    "- Antonio Machado.",
    "- Gustavo Adolfo Bécquer.",
    "- Pablo Neruda.",
    "- Octavio Paz.",
    "- Julio Cortázar.",
    "- Saludos desde Madrid.",
    "- Notas del día 15-08-24.",
    "- Ref: 984-A / Cuaderno.",
    "- Saludos desde Cádiz",
    "- Madrid, 03/09/92",
    "L.L.L. ~ 02-10-23",
]


def check_font_spanish_support(font_path: str) -> bool:
    """Check if the font contains Spanish glyphs (accents, ñ, symbols)."""
    test_chars = "áéíóúÁÉÍÓÚñÑüÜ¿¡~"
    try:
        font = ImageFont.truetype(font_path, size=24)
        for ch in test_chars:
            # Check bounding box: missing glyphs evaluate to empty/zero width in PIL
            bbox = font.getbbox(ch)
            if bbox is None or (bbox[2] - bbox[0]) == 0:
                return False
        return True
    except Exception:
        return False


def generate_phrases(min_count=1200):
    phrases = set()
    while len(phrases) < min_count:
        mode = random.choice([1, 2, 3])
        if mode == 1:
            phrases.add(
                f"{random.choice(SUBJECTS)} {random.choice(VERBS_PREDICATES)}"
            )
        elif mode == 2:
            phrases.add(
                f"{random.choice(SUBJECTS)} {random.choice(VERBS_PREDICATES)} {random.choice(AUTHORS_TAGS)}"
            )
        else:
            phrases.add(random.choice(AUTHORS_TAGS))
    return list(phrases)


def create_paper_background(w, h):
    # Realistic paper tonalities
    bg_color = (
        random.randint(238, 255),
        random.randint(235, 252),
        random.randint(225, 248),
    )
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 40% probability of notebook grid / ruled lines
    if random.random() < 0.40:
        grid_color = (
            random.randint(190, 220),
            random.randint(200, 230),
            random.randint(220, 245),
        )
        step = random.randint(18, 26)
        # Horizontal rules
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=grid_color, width=1)
        # Vertical grid columns
        if random.random() < 0.5:
            for x in range(0, w, step):
                draw.line([(x, 0), (x, h)], fill=grid_color, width=1)

    return img


def render_synthetic_line(text, font_path, output_path):
    font_size = random.randint(26, 38)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Extra padding prevents cursive descenders (g, j, y, p) from clipping
    pad_x = random.randint(25, 40)
    pad_y = random.randint(20, 35)
    w = max(100, text_w + pad_x * 2)
    h = max(50, text_h + pad_y * 2)

    img = create_paper_background(w, h)
    draw = ImageDraw.Draw(img)

    # Ink spectrum: Ballpoint blue, deep black, aged dark sepia
    ink_types = [
        (random.randint(15, 45), random.randint(20, 50), random.randint(90, 160)),
        (random.randint(20, 45), random.randint(20, 45), random.randint(25, 45)),
        (random.randint(30, 60), random.randint(20, 40), random.randint(40, 70)),
    ]
    ink = random.choice(ink_types)

    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=ink)

    np_img = np.array(img)

    # Perspective distortion
    if random.random() < 0.5:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        shift = random.uniform(-3, 3)
        pts2 = np.float32([[shift, 0], [w + shift, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        np_img = cv2.warpPerspective(
            np_img, matrix, (w, h), borderValue=(245, 243, 238)
        )

    # Stroke bleed simulation
    if random.random() < 0.4:
        np_img = cv2.GaussianBlur(np_img, (3, 3), 0)

    # Sensor noise
    noise = np.random.normal(0, random.uniform(1.0, 3.5), np_img.shape).astype(
        np.float32
    )
    noisy_img = np.clip(np_img.astype(np.float32) + noise, 0, 255).astype(
        np.uint8
    )

    Image.fromarray(noisy_img).save(output_path)


def generate_dataset(num_samples=3000):
    all_fonts = glob.glob("fonts/*.ttf") + glob.glob("fonts/*.otf")
    if not all_fonts:
        raise FileNotFoundError(
            "No handwriting fonts found in the 'fonts/' folder."
        )

    # Validate fonts for Spanish glyph coverage
    valid_fonts = [f for f in all_fonts if check_font_spanish_support(f)]
    print(f"Total fonts detected: {len(all_fonts)}")
    print(f"Fonts with full Spanish glyph support: {len(valid_fonts)}")

    if not valid_fonts:
        print(
            "Warning: None of the fonts fully passed glyph verification. Falling back to all fonts."
        )
        valid_fonts = all_fonts

    print(
        f"Generating {num_samples} synthetic Spanish handwriting samples across {len(valid_fonts)} fonts..."
    )
    phrases = generate_phrases(min_count=num_samples // 2)

    metadata_records = []

    for i in range(num_samples):
        text = random.choice(phrases)
        font_path = random.choice(valid_fonts)
        filename = f"synthetic_line_{i:05d}.png"
        file_path = IMAGES_DIR / filename

        render_synthetic_line(text, font_path, str(file_path))
        metadata_records.append(
            {"file_name": f"images/{filename}", "text": text}
        )

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{num_samples} crops...")

    # Write metadata.csv
    csv_file = OUTPUT_DIR / "metadata.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(metadata_records)

    print(
        f"Completed: {num_samples} samples generated with labels at {csv_file}"
    )


if __name__ == "__main__":
    generate_dataset(num_samples=3000)