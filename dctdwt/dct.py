"""Двовимірне DCT-II блоками 8×8 через матричне представлення.

Для кожного блоку X розміром 8×8:
    пряме DCT:     Y = C · X · Cᵀ
    обернене DCT:  X = Cᵀ · Y · C
де C — ортонормована матриця DCT-II. Оскільки C · Cᵀ = I, обернена матриця дорівнює транспонованій.
"""
import numpy as np

BLOCK = 8  # розмір блоку


def dct_matrix(n: int = BLOCK) -> np.ndarray:
    """Ортонормована матриця DCT-II розміру n×n.

    C[k, m] = a(k) · cos(π · (2m + 1) · k / (2n)),  a(0) = √(1/n),  a(k > 0) = √(2/n).
    Рядок k — дискретна косинусоїда частоти k.
    """
    k = np.arange(n).reshape(-1, 1)  # номер частоти (рядок)
    m = np.arange(n).reshape(1, -1)  # номер відліку (стовпець)
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * m + 1) * k / (2 * n))
    c[0, :] = np.sqrt(1.0 / n)
    return c


def _zigzag_order(n: int = BLOCK) -> np.ndarray:
    """Індекси 0..n²−1 у порядку зигзаг-сканування (як у JPEG): від низьких частот до високих."""
    cells = [(i, j) for i in range(n) for j in range(n)]
    # діагоналі i + j по черзі; на парних рух угору-праворуч, на непарних — униз-ліворуч
    cells.sort(key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))
    return np.array([i * n + j for i, j in cells])


C = dct_matrix()          # матриця перетворення 8×8
ZIGZAG = _zigzag_order()  # порядок сканування коефіцієнтів блоку


def to_blocks(image: np.ndarray) -> np.ndarray:
    """(H, W) -> (H/8, W/8, 8, 8): blocks[i, j] — блок у рядку i та стовпці j."""
    h, w = image.shape
    if h % BLOCK or w % BLOCK:
        raise ValueError(f"Розмір {w}×{h} не кратний {BLOCK}")
    return image.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2)


def from_blocks(blocks: np.ndarray) -> np.ndarray:
    """(H/8, W/8, 8, 8) -> (H, W): операція, обернена до to_blocks."""
    rows, cols = blocks.shape[:2]
    return blocks.swapaxes(1, 2).reshape(rows * BLOCK, cols * BLOCK)


def dct2_blocks(image: np.ndarray) -> np.ndarray:
    """Пряме DCT кожного блоку 8×8: Y = C · X · Cᵀ.

    Результат має розмір зображення: коефіцієнти блоку лежать на місці цього блоку
    (DC — у його лівому верхньому куті). Матричне множення виконується для всіх блоків одразу.
    """
    return from_blocks(C @ to_blocks(image) @ C.T)


def idct2_blocks(coefs: np.ndarray) -> np.ndarray:
    """Обернене DCT кожного блоку 8×8: X = Cᵀ · Y · C."""
    return from_blocks(C.T @ to_blocks(coefs) @ C)


def scan_coefficients(coefs: np.ndarray) -> np.ndarray:
    """Коефіцієнти в одновимірному порядку для кодування.

    Спершу DC усіх блоків, потім перший за зигзагом AC-коефіцієнт усіх блоків і т. д.
    Нулі високих частот утворюють довгі серії, які добре стискаються.
    """
    per_block = to_blocks(coefs).reshape(-1, BLOCK * BLOCK)  # рядок = 64 коефіцієнти одного блоку
    return per_block[:, ZIGZAG].T.ravel()
