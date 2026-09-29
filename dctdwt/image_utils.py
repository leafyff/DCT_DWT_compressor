"""Завантаження та збереження зображень у відтінках сірого; вбудовані зразки."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SIZE = 512   # розмір, до якого типово зводиться зображення
ALIGN = 64   # без масштабування розміри обрізаються до кратних 64 (блоки 8×8 і щонайменше 6 рівнів Хаара)

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC = DATA_DIR / "synthetic.png"  # синтетичне 512×512: градієнти, контури, текст, дрібні текстури
PHOTO = DATA_DIR / "raccoon.png"        # реальне фото: єнот, 1024×768 (набір SciPy «face», public domain)


def prepare(img: Image.Image, resize: bool = True) -> np.ndarray:
    """Переводить зображення у відтінки сірого та зводить до розміру, зручного для перетворень.
    resize=True — центральне обрізання до квадрата і масштабування до 512×512;
    resize=False — лише обрізання країв до розмірів, кратних 64.
    """
    img = ImageOps.exif_transpose(img)  # поворот за EXIF (фото з телефона), як у переглядачах зображень
    if img.mode.startswith("I"):  # 16 біт (PNG, TIFF, PGM): 0…65535 → 0…255; convert("L") лише обрізав би до 255
        img = Image.fromarray(np.clip(np.round(np.asarray(img) / 257), 0, 255).astype(np.uint8))
    img = img.convert("L")
    w, h = img.size
    if resize:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    else:
        nw, nh = w - w % ALIGN, h - h % ALIGN
        if not nw or not nh:
            raise ValueError(f"Зображення {w}×{h} замале (потрібно щонайменше {ALIGN}×{ALIGN})")
        left, top = (w - nw) // 2, (h - nh) // 2
        img = img.crop((left, top, left + nw, top + nh))
    return np.array(img, dtype=np.uint8)


def load_grayscale(path: str | Path, resize: bool = True) -> np.ndarray:
    """Читає файл зображення і готує його функцією prepare."""
    with Image.open(path) as img:
        return prepare(img, resize)


def save_grayscale(image: np.ndarray, path: str | Path) -> None:
    """Зберігає масив uint8 як зображення у відтінках сірого."""
    Image.fromarray(image.astype(np.uint8)).save(path)
