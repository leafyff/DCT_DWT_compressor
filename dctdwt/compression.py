"""Стиснення: відкидання коефіцієнтів, відновлення зображення та метрики якості.

Обидва перетворення ортонормовані (зберігають енергію сигналу), тому однаковий параметр стиснення
(частка, поріг, крок квантування) має для DCT і DWT однаковий зміст, і результати можна порівнювати.
Зображення відновлюється з тих самих цілих символів, що записуються у «файл», тож PSNR і розмір
даних описують один і той самий результат стиснення.
"""
import zlib
from dataclasses import dataclass

import numpy as np

from dctdwt import dct, haar

LEVEL_SHIFT = 128.0  # центрування яскравості перед перетворенням (як у JPEG): [0, 255] -> [-128, 127]


# ---------------------------------------------------------------- способи зменшення коефіцієнтів

@dataclass(frozen=True)
class Mode:
    """Опис способу зменшення кількості коефіцієнтів для інтерфейсу та експериментів."""
    key: str                    # ідентифікатор: fraction / threshold / quant
    title: str                  # назва для користувача
    param: str                  # назва параметра
    vmin: float                 # допустимий діапазон параметра
    vmax: float
    default: float              # початкове значення
    levels: tuple[float, ...]   # рівні для серії експериментів (від слабкого стиснення до сильного)
    description: str            # пояснення способу


MODES = {m.key: m for m in (
    Mode("fraction", "Частка найбільших коефіцієнтів", "Частка, %", 0.1, 100.0, 10.0,
         (50, 20, 10, 5, 2, 1),
         "Зберігаються k = p·N коефіцієнтів з найбільшим |c| (відбір по всьому зображенню, однакове k "
         "для DCT і DWT) з округленням до цілих, решта обнуляються. Якщо ненульових коефіцієнтів "
         "менше за k, зберігаються всі."),
    Mode("threshold", "Поріг (hard thresholding)", "Поріг T", 1.0, 1000.0, 20.0,
         (5, 10, 20, 40, 80, 160),
         "Коефіцієнти з |c| < T обнуляються, решта зберігаються з округленням до цілих."),
    Mode("quant", "Рівномірне квантування", "Крок Q", 1.0, 1000.0, 20.0,
         (5, 10, 20, 40, 80, 160),
         "c → Q·round(c / Q). Зберігаються цілі індекси round(c / Q); "
         "малі коефіцієнти стають нулями, решта — огрублюються."),
)}


def keep_largest(coefs: np.ndarray, fraction: float) -> np.ndarray:
    """Залишає рівно k = round(fraction·N) коефіцієнтів з найбільшим модулем, решту обнуляє.

    Серед коефіцієнтів з однаковим модулем на межі відбору беруться ті, що раніше в масиві, — тож
    результат не залежить від реалізації часткового сортування. Якщо ненульових коефіцієнтів менше
    за k, зберігаються всі ненульові (решта відібраних — нулі, які нічого не коштують).
    """
    flat = coefs.ravel()
    n = flat.size
    k = min(max(int(round(fraction * n)), 1), n)
    mag = np.abs(flat)
    cut = np.partition(mag, n - k)[n - k]  # k-й за величиною модуль (без повного сортування)
    keep = mag > cut
    ties = np.flatnonzero(mag == cut)      # модуль дорівнює межі: добираються по порядку до k
    keep[ties[:k - np.count_nonzero(keep)]] = True
    return np.where(keep, flat, 0.0).reshape(coefs.shape)


def reduce_coefficients(coefs: np.ndarray, mode: str, value: float) -> tuple[np.ndarray, np.ndarray]:
    """Зменшує кількість значущих коефіцієнтів.

    Повертає (коефіцієнти для оберненого перетворення, цілі символи, які записуються у «файл»).
    Коефіцієнти відновлюються із символів — так само, як це зробив би декодер «файлу».
    Для квантування символи — індекси round(c / Q), інакше — округлені до цілих відібрані коефіцієнти.
    """
    if mode == "quant":
        symbols = np.round(coefs / value)
        return symbols * value, symbols
    if mode == "fraction":
        reduced = keep_largest(coefs, value / 100.0)
    elif mode == "threshold":
        reduced = np.where(np.abs(coefs) >= value, coefs, 0.0)
    else:
        raise ValueError(f"Невідомий спосіб стиснення: {mode}")
    symbols = np.round(reduced)  # у «файл» відібрані коефіцієнти записуються цілими
    return symbols, symbols      # символи й є відновленими коефіцієнтами (крок 1)


# ---------------------------------------------------------------- кодеки

class DCTCodec:
    """Блочне DCT 8×8: коефіцієнти зображення та обернене перетворення."""
    name = "DCT"
    label = "DCT (блоки 8×8)"

    def __init__(self, image: np.ndarray):
        """Обчислює коефіцієнти DCT зображення (після зсуву яскравості)."""
        self.coefs = dct.dct2_blocks(image - LEVEL_SHIFT)

    @staticmethod
    def inverse(coefs: np.ndarray) -> np.ndarray:
        """Відновлює яскравості пікселів із коефіцієнтів."""
        return dct.idct2_blocks(coefs) + LEVEL_SHIFT

    @staticmethod
    def scan(coefs: np.ndarray) -> np.ndarray:
        """Порядок запису коефіцієнтів у «файл»: від низьких частот до високих."""
        return dct.scan_coefficients(coefs)


class HaarCodec:
    """Багаторівневе DWT Хаара: коефіцієнти зображення та обернене перетворення."""
    name = "DWT"

    def __init__(self, image: np.ndarray, levels: int):
        """Обчислює коефіцієнти `levels`-рівневого перетворення Хаара (після зсуву яскравості)."""
        self.levels = levels
        self.label = f"DWT Хаара ({levels} рівн.)"
        self.coefs = haar.haar2d(image - LEVEL_SHIFT, levels)

    def inverse(self, coefs: np.ndarray) -> np.ndarray:
        """Відновлює яскравості пікселів із коефіцієнтів."""
        return haar.ihaar2d(coefs, self.levels) + LEVEL_SHIFT

    def scan(self, coefs: np.ndarray) -> np.ndarray:
        """Порядок запису коефіцієнтів у «файл»: піддіапазони від грубих до детальних."""
        return haar.scan_coefficients(coefs, self.levels)


Codec = DCTCodec | HaarCodec


# ---------------------------------------------------------------- метрики

def mse(original: np.ndarray, restored: np.ndarray) -> float:
    """Середньоквадратична похибка між двома зображеннями."""
    diff = original.astype(np.float64) - restored
    return float(np.mean(diff * diff))


def psnr(error: float, peak: float = 255.0) -> float:
    """PSNR = 10·log10(peak² / MSE), дБ. Для MSE = 0 (без втрат) — ∞."""
    return float("inf") if error == 0 else float(10.0 * np.log10(peak * peak / error))


def encoded_size(symbols: np.ndarray) -> int:
    """Оцінка розміру закодованих даних у байтах.

    Цілі символи (у порядку від низьких частот до високих) записуються як int16
    (int32, якщо не вміщуються) і стискаються zlib. Заголовок файлу не враховується.
    """
    ints = symbols.astype(np.int64)
    dtype = np.int16 if np.abs(ints).max(initial=0) < 2 ** 15 else np.int32
    return len(zlib.compress(ints.astype(dtype).tobytes()))


@dataclass
class Result:
    """Результат стиснення одним методом з одним параметром."""
    method: str           # "DCT" або "DWT"
    label: str            # назва методу для підписів
    image: np.ndarray     # відновлене зображення, uint8
    coefs: np.ndarray     # коефіцієнти після відкидання (відновлені з цілих символів «файлу»)
    kept: int             # кількість ненульових коефіцієнтів (= ненульових символів у «файлі»)
    mse: float
    psnr: float
    encoded_bytes: int    # оцінка розміру закодованих даних

    @property
    def total(self) -> int:
        """Загальна кількість коефіцієнтів N (дорівнює кількості пікселів)."""
        return self.coefs.size

    @property
    def fraction(self) -> float:
        """Частка збережених (ненульових) коефіцієнтів."""
        return self.kept / self.total

    @property
    def cr_coef(self) -> float:
        """Коефіцієнт стиснення за кількістю коефіцієнтів: N / N_ненульових."""
        return self.total / self.kept if self.kept else float("inf")

    @property
    def cr_file(self) -> float:
        """Коефіцієнт стиснення за розміром даних: (H·W байт при 8 біт/піксель) / розмір zlib."""
        return self.image.size / self.encoded_bytes

    @property
    def bpp(self) -> float:
        """Бітова швидкість: біт на піксель."""
        return 8.0 * self.encoded_bytes / self.image.size


def compress(codec: Codec, original: np.ndarray, mode: str, value: float) -> Result:
    """Повний цикл: відкидання коефіцієнтів -> обернене перетворення -> відновлення -> метрики."""
    reduced, symbols = reduce_coefficients(codec.coefs, mode, value)
    restored = np.clip(np.round(codec.inverse(reduced)), 0, 255).astype(np.uint8)
    error = mse(original, restored)
    return Result(
        method=codec.name,
        label=codec.label,
        image=restored,
        coefs=reduced,
        kept=int(np.count_nonzero(reduced)),
        mse=error,
        psnr=psnr(error),
        encoded_bytes=encoded_size(codec.scan(symbols)),
    )
