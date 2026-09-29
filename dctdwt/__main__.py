"""Командний рядок: графічний інтерфейс або пакетна серія експериментів.

    dctdwt                                   — інтерфейс із синтетичним тестовим зображенням
    dctdwt photo.jpg                         — інтерфейс із вашим зображенням
    dctdwt --photo                           — інтерфейс із вбудованим фото (єнот)
    dctdwt --photo --mode quant              — інтерфейс, одразу зі способом «квантування»
    dctdwt --photo --batch                   — серія експериментів без інтерфейсу, звіт у results/
    dctdwt --batch --mode quant --levels "5;20;80"

Без встановлення пакета замість `dctdwt` — `python -m dctdwt`.
"""
import argparse
import io
import sys
from pathlib import Path
from typing import NoReturn

from dctdwt import experiments as ex
from dctdwt import haar
from dctdwt.compression import MODES, DCTCodec, HaarCodec
from dctdwt.image_utils import PHOTO, SYNTHETIC, load_grayscale


class _Parser(argparse.ArgumentParser):
    """Розбір аргументів; без консолі (запуск через pythonw) повідомлення про помилку показується у вікні."""

    def error(self, message: str) -> NoReturn:
        if sys.stderr is None:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(self.prog, f"{message}\n\n{self.format_usage()}", parent=root)
            root.destroy()
        super().error(message)


def run_batch(image_path: Path, resize: bool, dwt_levels: int, mode_key: str, levels: list[float] | None,
              out_dir: Path) -> None:
    """Серія експериментів без інтерфейсу: таблиця в консоль, звіт (PNG, CSV, Markdown) у нову підтеку out_dir."""
    original = load_grayscale(image_path, resize)
    mode = MODES[mode_key]
    values = levels or list(mode.levels)
    codecs = (DCTCodec(original), HaarCodec(original, min(dwt_levels, haar.max_levels(original.shape))))

    experiments = ex.run_experiments(codecs, original, mode.key, values)
    curves = ex.sweep(codecs, original, mode.key)
    out = ex.save_report(ex.report_dir(out_dir, image_path, mode.key), original, experiments, curves, mode.key,
                         image_path.name)

    h, w = original.shape
    print(f"{image_path.name}, {w}×{h}; {codecs[0].label} vs {codecs[1].label}; спосіб: {mode.title}\n")
    print(ex.markdown_table(experiments, mode.key))
    print(f"\nЗвіт збережено: {out.resolve()}")


def main() -> None:
    """Розбирає аргументи командного рядка і запускає інтерфейс або пакетний режим."""
    for stream in (sys.stdout, sys.stderr):  # інакше консоль Windows не виведе кирилицю у файл чи канал
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = _Parser(prog="dctdwt", description="Стиснення зображень: DCT vs DWT (Хаар)")
    parser.add_argument("image", nargs="?", type=Path, help="зображення (типово — синтетичне тестове 512×512)")
    parser.add_argument("--photo", action="store_true", help="вбудоване фото єнота")
    parser.add_argument("--no-resize", action="store_true",
                        help="не зводити до 512×512, лише обрізати краї до розмірів, кратних 64")
    parser.add_argument("--dwt-levels", type=int, default=3, choices=range(1, 9), metavar="1..8",
                        help="рівнів декомпозиції Хаара (типово 3)")
    parser.add_argument("--batch", action="store_true", help="виконати серію експериментів без інтерфейсу")
    parser.add_argument("--mode", choices=list(MODES), default="fraction",
                        help="fraction — частка найбільших коеф. (%%), threshold — поріг T, quant — крок Q")
    parser.add_argument("--levels", help='рівні серії експериментів через ";", напр. "50;10;2"')
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="тека, у якій створюється підтека звіту (режим --batch)")
    args = parser.parse_args()
    image = PHOTO if args.photo else (args.image or SYNTHETIC)
    try:
        levels = ex.parse_levels(args.levels, args.mode) if args.levels else None
    except ValueError as err:  # некоректні рівні
        parser.error(str(err))

    if args.batch:
        try:
            run_batch(image, not args.no_resize, args.dwt_levels, args.mode, levels, args.out)
        except (OSError, ValueError) as err:  # немає файлу, не зображення, замале зображення
            parser.error(str(err))
    else:
        from dctdwt.gui import run  # Tkinter завантажується лише для графічного режиму
        run(image, not args.no_resize, args.dwt_levels, args.mode, levels)


if __name__ == "__main__":
    main()
