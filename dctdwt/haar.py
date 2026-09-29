"""Багаторівневе двовимірне дискретне вейвлет-перетворення Хаара (ортонормоване).

Один крок 1D-перетворення для пари сусідніх відліків (x₀, x₁):
    a = (x₀ + x₁) / √2 — апроксимація (низькі частоти)
    d = (x₀ − x₁) / √2 — деталі (високі частоти)
Обернений крок: x₀ = (a + d) / √2, x₁ = (a − d) / √2.
Це еквівалентно множенню на ортонормовану матрицю Хаара, але без її явної побудови.

Один 2D-рівень — крок по рядках, потім по стовпцях. Результат зберігається у форматі Маллата:
    ┌────┬────┐
    │ LL │ HL │   LL — зменшена копія зображення, до неї застосовується наступний рівень;
    ├────┼────┤   HL, LH, HH — деталі: вертикальні, горизонтальні та діагональні контури.
    │ LH │ HH │
    └────┴────┘
"""
import numpy as np

SQRT2 = np.sqrt(2.0)


def _step_rows(x: np.ndarray) -> np.ndarray:
    """Один крок Хаара вздовж кожного рядка: [a | d]."""
    even, odd = x[:, 0::2], x[:, 1::2]
    return np.hstack(((even + odd) / SQRT2, (even - odd) / SQRT2))


def _inverse_step_rows(y: np.ndarray) -> np.ndarray:
    """Обернений до _step_rows: з [a | d] відновлює парні й непарні відліки."""
    half = y.shape[1] // 2
    a, d = y[:, :half], y[:, half:]
    x = np.empty_like(y)
    x[:, 0::2] = (a + d) / SQRT2
    x[:, 1::2] = (a - d) / SQRT2
    return x


def max_levels(shape: tuple[int, int]) -> int:
    """Скільки разів обидва розміри можна поділити навпіл (максимальна кількість рівнів)."""
    h, w = shape
    levels = 0
    while h % 2 == 0 and w % 2 == 0 and min(h, w) > 1:
        h, w, levels = h // 2, w // 2, levels + 1
    return levels


def haar2d(image: np.ndarray, levels: int) -> np.ndarray:
    """Пряме перетворення: `levels` рівнів, кожен наступний — над піддіапазоном LL попереднього."""
    if not 1 <= levels <= max_levels(image.shape):
        raise ValueError(f"Неприпустима кількість рівнів {levels} для розміру {image.shape}")
    coefs = image.astype(np.float64)  # копія: вхідний масив не змінюється
    h, w = coefs.shape
    for _ in range(levels):
        coefs[:h, :w] = _step_rows(_step_rows(coefs[:h, :w]).T).T  # рядки, потім стовпці
        h, w = h // 2, w // 2
    return coefs


def ihaar2d(coefs: np.ndarray, levels: int) -> np.ndarray:
    """Обернене перетворення: рівні відновлюються від найгрубшого до найдетальнішого."""
    image = coefs.astype(np.float64)
    h, w = image.shape[0] >> (levels - 1), image.shape[1] >> (levels - 1)  # розмір найгрубшого рівня
    for _ in range(levels):
        image[:h, :w] = _inverse_step_rows(_inverse_step_rows(image[:h, :w].T).T)  # стовпці, потім рядки
        h, w = h * 2, w * 2
    return image


def subbands(shape: tuple[int, int], levels: int) -> list[tuple[slice, slice]]:
    """Області піддіапазонів у форматі Маллата від найгрубшого до найдетальнішого: LLₙ, HLₙ, LHₙ, HHₙ, …, HH₁."""
    h, w = shape
    bands = [(slice(0, h >> levels), slice(0, w >> levels))]  # LLₙ
    for lev in range(levels, 0, -1):
        bh, bw = h >> lev, w >> lev  # розмір піддіапазонів рівня lev
        bands += [
            (slice(0, bh), slice(bw, 2 * bw)),       # HL
            (slice(bh, 2 * bh), slice(0, bw)),       # LH
            (slice(bh, 2 * bh), slice(bw, 2 * bw)),  # HH
        ]
    return bands


def scan_coefficients(coefs: np.ndarray, levels: int) -> np.ndarray:
    """Коефіцієнти в одновимірному порядку для кодування: піддіапазони від грубих до детальних."""
    return np.concatenate([coefs[s].ravel() for s in subbands(coefs.shape, levels)])
