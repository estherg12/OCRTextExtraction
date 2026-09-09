import os
import re
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pypdf import PdfReader

# Directorios de trabajo
PDF_DIR = Path("training_pdf")
OUTPUT_DIR = Path("dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
FONTS_DIR = Path("fonts")
METADATA_FILE = OUTPUT_DIR / "metadata.tsv"

TARGET_SAMPLES = 30000


# EXTRACCIÓN Y LIMPIEZA DE FRASES DESDE LOS PDFS

def clean_and_split_text(raw_text: str) -> list[str]:
    """Limpia el texto bruto y lo divide en líneas/frases de longitud idónea para TrOCR."""
    # Reemplazar saltos de línea y múltiples espacios por un espacio simple
    text = re.sub(r"\s+", " ", raw_text)

    # Separar por puntos, signos de exclamación, interrogación, guiones largos o comas mayores
    raw_phrases = re.split(r"(?<=[.?!;])\s+|(?<=[,])\s+", text)

    valid_lines = []
    for phrase in raw_phrases:
        p = phrase.strip()

        # Eliminar comillas y caracteres huérfanos
        p = re.sub(r'^["\'«»—–\-\s]+|["\'«»—–\-\s]+$', '', p).strip()

        # Filtrar números de página, índices numéricos o encabezados vacíos
        if not p or len(p) < 12 or len(p) > 85:
            continue

        words = p.split()
        # Una línea de caligrafía ideal para TrOCR contiene entre 3 y 11 palabras
        if len(words) < 3 or len(words) > 11:
            continue

        # Conservar solo texto con caracteres legítimos en español
        if not re.search(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]", p):
            continue

        # Descartar URLs, rutas o artefactos de digitalización
        if re.search(r"(http|www|\.com|\[|\]|\{|\}|\\|/|_|\*|<|>)", p):
            continue

        valid_lines.append(p)

    return valid_lines


def build_sentence_bank_from_pdfs(pdf_folder: Path, target_count: int) -> list[str]:
    """Lee todos los archivos PDF presentes en la carpeta y extrae frases únicas."""
    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron archivos PDF dentro de '{pdf_folder.resolve()}'.")

    print(f"Extrayendo texto de {len(pdf_files)} libros/documentos PDF...")
    all_sentences = set()

    for pdf_path in pdf_files:
        print(f" -> Procesando '{pdf_path.name}'...")
        try:
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    lines = clean_and_split_text(extracted)
                    all_sentences.update(lines)

                # Si ya hemos acumulado holgadamente más frases de las necesarias, detenemos la lectura
                if len(all_sentences) >= target_count * 1.3:
                    break
        except Exception as e:
            print(f" [Aviso] Error leyendo '{pdf_path.name}': {e}")
            continue

        if len(all_sentences) >= target_count * 1.3:
            break

    sentence_list = list(all_sentences)
    random.seed(42)
    random.shuffle(sentence_list)

    if len(sentence_list) < target_count:
        print(f" [Aviso] Se extrajeron {len(sentence_list)} frases únicas (se esperaban {target_count}).")
        # Si faltasen, rellenamos combinando fragmentos para alcanzar el volumen objetivo
        augmented = []
        while len(sentence_list) + len(augmented) < target_count:
            s1 = random.choice(sentence_list)
            augmented.append(s1)
        sentence_list.extend(augmented)

    final_bank = sentence_list[:target_count]
    print(f"Banco de datos construido con éxito: {len(final_bank)} frases reales en español.\n")
    return final_bank


# RENDERIZADO SINTÉTICO RÁPIDO EN PARALELO

def apply_fast_degradations(img: Image.Image) -> Image.Image:
    """Aplica ruido de grano y leve desenfoque de absorción de tinta optimizado en NumPy."""
    np_img = np.array(img, dtype=np.int16)

    # Grano de textura de papel ligero
    noise = np.random.randint(-4, 5, np_img.shape, dtype=np.int16)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)

    result_img = Image.fromarray(np_img)
    if random.random() < 0.35:
        result_img = result_img.filter(ImageFilter.BoxBlur(radius=0.4))
    return result_img


def render_single_crop(task_data: tuple) -> str:
    """Función de renderizado para cada hilo del multiprocessing pool."""
    idx, text, font_path = task_data

    font_size = random.randint(30, 42)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    # Medir caja de texto
    dummy_img = Image.new("RGB", (1, 1))
    draw_dummy = ImageDraw.Draw(dummy_img)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Margen de seguridad para descendentes y ascendentes
    pad_x = random.randint(15, 25)
    pad_y = random.randint(10, 20)
    img_w = max(80, text_w + pad_x * 2)
    img_h = max(45, text_h + pad_y * 2)

    # Fondo claro tipo papel (blanco marfil, crema, gris claro)
    bg_val = random.randint(240, 255)
    bg_color = (bg_val, max(0, bg_val - random.randint(0, 4)), max(0, bg_val - random.randint(0, 8)))

    # Color de tinta (negro, azul marino oscuro, gris carbón)
    if random.random() < 0.6:
        ink_color = (random.randint(15, 45), random.randint(15, 45), random.randint(15, 45))
    else:
        ink_color = (random.randint(10, 30), random.randint(15, 40), random.randint(55, 95))

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=ink_color)

    # Añadir leve ruido de soporte
    final_img = apply_fast_degradations(img)

    file_name = f"crop_{idx:06d}.png"
    out_path = IMAGES_DIR / file_name
    final_img.save(out_path, format="PNG", optimize=False)

    return f"{file_name}\t{text}\n"


def generate_dataset(num_samples: int = TARGET_SAMPLES):
    """Orquesta la extracción, balanceo de fuentes y renderizado multiproceso."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Obtener tipografías
    fonts = list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf"))
    if not fonts:
        raise FileNotFoundError(f"No se encontraron fuentes TTF/OTF en '{FONTS_DIR.resolve()}'.")
    print(f"Fuentes caligráficas cargadas: {len(fonts)}")

    # Obtener frases reales de los libros
    sentences = build_sentence_bank_from_pdfs(PDF_DIR, num_samples)

    # Preparar tareas
    tasks = []
    for idx, phrase in enumerate(sentences):
        assigned_font = random.choice(fonts)
        tasks.append((idx, phrase, assigned_font))

    # Renderizado en paralelo usando todos los núcleos físicos/lógicos
    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Renderizando {len(tasks)} imágenes en disco usando {workers} núcleos...")

    metadata_lines = ["file_name\ttext\n"]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count, line in enumerate(executor.map(render_single_crop, tasks, chunksize=100), 1):
            metadata_lines.append(line)
            if count % 2500 == 0 or count == len(tasks):
                print(f" -> Progreso: {count}/{len(tasks)} imágenes generadas...")

    # Guardar metadatos TSV
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.writelines(metadata_lines)

    print(f"\nGeneración completada: {len(tasks)} imágenes en '{IMAGES_DIR}' y metadatos en '{METADATA_FILE}'.")


if __name__ == "__main__":
    generate_dataset(TARGET_SAMPLES)