import csv
import glob
import math
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Ensure target directories exist
OUTPUT_DIR = Path("dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Base vocabulary & modern Spanish sentence templates
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
    "Mi hombre"
]

VERBS_PREDICATES = [
    "hace que todo sea más hermoso.",
    "me queire más que a nada.",
    "me cuida como a una flor.",
    "huele a rosas.",
    "sabe protegerme.",
    "tiene un corazón rebelde.",
    "cuida sus flores.",
    "pertenece a mi vida.",
    "sabe que todo vale la pena.",
    "me dice que todo vale la pena.",
    "me dice que estará bien.",
    "solo vivir es una deuda que no debemos olvidar.",
    "quererte no sabe.",
    "besarte quiere",
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
    "- Cádiz, 02-10-23.",
    "- Madrid 03/09/92"
]


def generate_phrases(min_count=550):
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
    # Base cream / white paper tone
    bg_color = (
        random.randint(238, 255),
        random.randint(235, 252),
        random.randint(225, 248),
    )
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 30% chance of notebook lines / grid
    if random.random() < 0.35:
        grid_color = (
            random.randint(190, 220),
            random.randint(200, 230),
            random.randint(220, 245),
        )
        # Horizontal lines
        for y in range(0, h, random.randint(18, 28)):
            draw.line([(0, y), (w, y)], fill=grid_color, width=1)
        # Grid vertical lines
        if random.random() < 0.5:
            for x in range(0, w, random.randint(18, 28)):
                draw.line([(x, 0), (x, h)], fill=grid_color, width=1)

    return img


def render_synthetic_line(text, font_path, output_path):
    font_size = random.randint(26, 38)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Calculate text bounding box
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Canvas dimensions with padding
    pad_x = random.randint(15, 30)
    pad_y = random.randint(10, 20)
    w = text_w + pad_x * 2
    h = text_h + pad_y * 2

    # Step 1: Generate background texture
    img = create_paper_background(w, h)
    draw = ImageDraw.Draw(img)

    # Step 2: Realistic ink colors (blue, black, blue-black)
    ink_types = [
        (random.randint(15, 45), random.randint(20, 50), random.randint(90, 160)),  # Ballpoint blue
        (random.randint(20, 45), random.randint(20, 45), random.randint(25, 45)),   # Black ink
        (random.randint(30, 60), random.randint(20, 40), random.randint(40, 70)),   # Dark sepia
    ]
    ink = random.choice(ink_types)

    draw.text((pad_x, pad_y), text, font=font, fill=ink)

    # Step 3: Add realistic scanner/camera artifacts (OpenCV)
    np_img = np.array(img)

    # Slight slant/perspective jitter
    if random.random() < 0.6:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        shift = random.uniform(-4, 4)
        pts2 = np.float32(
            [[shift, 0], [w + shift, 0], [0, h], [w, h]]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        np_img = cv2.warpPerspective(
            np_img, matrix, (w, h), borderValue=(250, 248, 245)
        )

    # Mild Gaussian blur to mimic pen stroke bleed and camera lens
    if random.random() < 0.5:
        k = random.choice([3, 5])
        np_img = cv2.GaussianBlur(np_img, (k, k), 0)

    # Add subtle sensor noise
    noise = np.random.normal(0, random.uniform(1.5, 4.0), np_img.shape).astype(np.float32)
    noisy_img = np.clip(np_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    final_pil = Image.fromarray(noisy_img)
    final_pil.save(output_path)


def generate_dataset(num_samples=1200):
    fonts = glob.glob("fonts/*.ttf") + glob.glob("fonts/*.otf")
    if not fonts:
        raise FileNotFoundError(
            "No handwriting fonts found! Place at least 1-3 .ttf or .otf fonts inside the 'fonts/' folder."
        )

    print(f"Found {len(fonts)} fonts. Generating {num_samples} synthetic Spanish handwriting samples...")
    phrases = generate_phrases(min_count=num_samples // 2)

    metadata_records = []

    for i in range(num_samples):
        text = random.choice(phrases)
        font_path = random.choice(fonts)
        filename = f"synthetic_line_{i:05d}.png"
        file_path = IMAGES_DIR / filename

        render_synthetic_line(text, font_path, str(file_path))
        metadata_records.append({"file_name": f"images/{filename}", "text": text})

        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{num_samples} crops...")

    # Write metadata.csv
    csv_file = OUTPUT_DIR / "metadata.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(metadata_records)

    print(f"Finished! Successfully generated {num_samples} samples and {csv_file}")


if __name__ == "__main__":
    generate_dataset(num_samples=1200)