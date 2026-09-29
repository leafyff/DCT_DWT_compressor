"""Серії експериментів: кілька рівнів стиснення, таблиця результатів, графіки PSNR, збереження звіту."""
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from dctdwt.compression import MODES, Codec, Result, compress
from dctdwt.image_utils import save_grayscale

# оформлення графіків: кольори й маркери серій, основний і допоміжний текст, сітка
COLORS = {"DCT": "#2a78d6", "DWT": "#eb6834"}
MARKERS = {"DCT": "o", "DWT": "s"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# стовпці метрик (спільні для таблиці в інтерфейсі та для звіту) і повна шапка таблиці експериментів
ZLIB_COLUMN = "Розмір zlib, КБ"  # у зведеній таблиці під зображеннями цей стовпець не показується
METRIC_HEADERS = ("Збережено коеф.", "Частка, %", "CR (коеф.)", ZLIB_COLUMN, "CR (файл)", "bpp", "MSE",
                  "PSNR, дБ")
HEADERS = ("Рівень", "Метод") + METRIC_HEADERS


@dataclass
class Experiment:
    """Один рівень стиснення: значення параметра і результати обох методів."""
    value: float
    results: dict[str, Result]  # {"DCT": …, "DWT": …}


# ---------------------------------------------------------------- обчислення та форматування

def format_level(mode: str, value: float) -> str:
    """Підпис рівня стиснення: «10 %», «T = 20» або «Q = 20»."""
    return {"fraction": f"{value:g} %", "threshold": f"T = {value:g}", "quant": f"Q = {value:g}"}[mode]


def format_psnr(value: float) -> str:
    """PSNR із двома знаками після коми або «∞» для стиснення без втрат."""
    return "∞" if math.isinf(value) else f"{value:.2f}"


def format_mse(value: float) -> str:
    """MSE із двома знаками після коми; дуже малі значення — в експоненційному записі."""
    return f"{value:.2f}" if value == 0 or value >= 0.01 else f"{value:.1e}"


def file_tag(mode: str, value: float) -> str:
    """Частина імені файлу для рівня стиснення: p10, T20, Q20 (крапка замінюється на _)."""
    prefix = {"fraction": "p", "threshold": "T", "quant": "Q"}[mode]
    return f"{prefix}{value:g}".replace(".", "_")


def parse_levels(text: str, mode: str) -> list[float]:
    """«50; 20; 10,5» -> [50, 20, 10.5] з перевіркою діапазону параметра способу `mode`.

    Роздільники — «;» або пробіли, десяткова кома допускається.
    """
    m = MODES[mode]
    try:
        values = [float(part.replace(",", ".")) for part in re.split(r"[;\s]+", text.strip()) if part]
    except ValueError:
        values = []
    if not values or any(not m.vmin <= v <= m.vmax for v in values):
        raise ValueError(f"Некоректні рівні «{text}»: потрібні числа від {m.vmin:g} до {m.vmax:g}, розділені «;»")
    return values


def format_levels(values) -> str:
    """[50, 20, 10.5] -> «50; 20; 10.5» (обернене до parse_levels)."""
    return "; ".join(f"{v:g}" for v in values)


def run_experiments(codecs: tuple[Codec, ...], original: np.ndarray, mode: str, values) -> list[Experiment]:
    """Стискає зображення кожним методом для кожного рівня."""
    return [Experiment(v, {c.name: compress(c, original, mode, v) for c in codecs}) for v in values]


def _curve(results: list[Result]) -> np.ndarray:
    """Точки [частка коеф. %, PSNR, bpp], відсортовані за часткою; результати з PSNR = ∞ не наносяться."""
    rows = sorted((100 * r.fraction, r.psnr, r.bpp) for r in results if r.kept and math.isfinite(r.psnr))
    return np.array(rows).reshape(-1, 3)


def sweep(codecs: tuple[Codec, ...], original: np.ndarray, mode: str, points: int = 24) -> dict[str, np.ndarray]:
    """Густий перебір параметра (логарифмічна сітка) для побудови кривих PSNR. Повертає {метод: _curve}."""
    m = MODES[mode]
    # частка — до 60 %: ближче до 100 % зберігаються вже всі ненульові цілі коефіцієнти,
    # і криві сплющуються на межі, яку дає округлення коефіцієнтів до цілих
    top = 60.0 if mode == "fraction" else m.vmax
    values = np.geomspace(m.vmin, top, points)
    return {c.name: _curve([compress(c, original, mode, v) for v in values]) for c in codecs}


def metric_cells(r: Result) -> tuple[str, ...]:
    """Відформатовані метрики одного результату (у порядку METRIC_HEADERS)."""
    return (f"{r.kept} / {r.total}", f"{100 * r.fraction:.2f}", f"{r.cr_coef:.1f}", f"{r.encoded_bytes / 1024:.1f}",
            f"{r.cr_file:.2f}", f"{r.bpp:.3f}", format_mse(r.mse), format_psnr(r.psnr))


def table_rows(experiments: list[Experiment], mode: str) -> list[tuple[str, ...]]:
    """Рядки таблиці результатів (у порядку HEADERS): по рядку на кожен рівень і метод."""
    return [(format_level(mode, e.value), r.method, *metric_cells(r))
            for e in experiments for r in e.results.values()]


def markdown_table(experiments: list[Experiment], mode: str) -> str:
    """Таблиця результатів у форматі Markdown."""
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    lines += ["| " + " | ".join(row) + " |" for row in table_rows(experiments, mode)]
    return "\n".join(lines)


# ---------------------------------------------------------------- графіки

def _style_axes(ax: Axes) -> None:
    """Стримане оформлення осей: світла сітка, без верхньої та правої рамки."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)


def draw_curves(fig: Figure, experiments: list[Experiment], curves: dict[str, np.ndarray], mode: str) -> None:
    """Два графіки: PSNR від частки збережених коефіцієнтів і PSNR від bpp (оцінка zlib).

    Лінії — густий перебір параметра (sweep), маркери — рівні з таблиці експериментів.
    """
    fig.clear()
    axes = fig.subplots(1, 2)
    specs = ((0, "Частка збережених коефіцієнтів, % (лог. шкала)", "PSNR від кількості коефіцієнтів"),
             (2, "Бітова швидкість (оцінка zlib), біт/піксель (лог. шкала)", "PSNR від розміру даних"))
    for ax, (col, xlabel, title) in zip(axes, specs):
        for name, line in curves.items():
            points = _curve([e.results[name] for e in experiments])
            ax.plot(line[:, col], line[:, 1], color=COLORS[name], linewidth=2,
                    label=experiments[0].results[name].label)
            ax.plot(points[:, col], points[:, 1], linestyle="none", marker=MARKERS[name], markersize=8,
                    markerfacecolor=COLORS[name], markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))  # 0.1, 1, 10 замість 10⁻¹…
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PSNR, дБ")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
        _style_axes(ax)
    fig.suptitle(f"Лінії — густий перебір параметра, маркери — експерименти ({MODES[mode].title.lower()})",
                 fontsize=9, color=INK_MUTED, x=0.01, ha="left")


def _show(ax: Axes, image: np.ndarray, title: str, fontsize: int = 9) -> None:
    """Показує 8-бітове зображення без поділок осей і з підписом."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])


def image_title(r: Result, level: str) -> str:
    """Підпис відновленого зображення: метод, рівень, частка коефіцієнтів і PSNR."""
    return f"{r.method} · {level}\n{100 * r.fraction:.2f} % коеф. · PSNR {format_psnr(r.psnr)} дБ"


def draw_grid(fig: Figure, original: np.ndarray, experiments: list[Experiment], mode: str) -> None:
    """Усі рівні поряд: верхній рядок DCT, нижній DWT, у першому стовпці — оригінал.

    Осі спільні, тож лупа збільшує однаковий фрагмент на всіх зображеннях.
    """
    fig.clear()
    axes = fig.subplots(2, len(experiments) + 1, sharex=True, sharey=True, squeeze=False)
    for row, name in zip(axes, ("DCT", "DWT")):
        _show(row[0], original, "Оригінал")
        for ax, e in zip(row[1:], experiments):
            _show(ax, e.results[name].image, image_title(e.results[name], format_level(mode, e.value)), 8)


def save_comparison(out_dir: Path, tag: str, original: np.ndarray, results: dict[str, Result], level: str) -> None:
    """Зберігає compare_<tag>.png (Original | DCT | DWT з підписами) і відновлені dct_<tag>.png, dwt_<tag>.png."""
    h, w = original.shape
    fig = Figure(figsize=(3 * w / 100 + 0.6, h / 100 + 0.9), dpi=100, layout="constrained")  # ≈ 1:1
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    _show(axes[0], original, "Оригінал", 11)
    for ax, r in zip(axes[1:], results.values()):
        _show(ax, r.image, image_title(r, level), 11)
        save_grayscale(r.image, out_dir / f"{r.method.lower()}_{tag}.png")
    fig.savefig(out_dir / f"compare_{tag}.png")


# ---------------------------------------------------------------- звіт

def report_dir(base: str | Path, source: str | Path, mode: str) -> Path:
    """Нова тека звіту <base>/<ім'я зображення>_<спосіб>_<дата>_<час>: попередні звіти не перезаписуються."""
    return Path(base) / f"{Path(source).stem}_{mode}_{datetime.now():%Y%m%d_%H%M%S}"


def save_report(out_dir: str | Path, original: np.ndarray, experiments: list[Experiment],
                curves: dict[str, np.ndarray], mode: str, source: str) -> Path:
    """Зберігає зображення, таблиці (CSV, Markdown) і графіки серії експериментів. Повертає теку звіту."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_grayscale(original, out / "original.png")
    for e in experiments:
        save_comparison(out, file_tag(mode, e.value), original, e.results, format_level(mode, e.value))

    fig = Figure(figsize=(2.4 * (len(experiments) + 1), 5.6), dpi=110, layout="constrained")
    draw_grid(fig, original, experiments, mode)
    fig.savefig(out / "overview.png")

    fig = Figure(figsize=(12, 4.8), dpi=120, layout="constrained")
    draw_curves(fig, experiments, curves, mode)
    fig.savefig(out / "psnr_curves.png")

    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:  # BOM — щоб Excel прочитав UTF-8
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADERS)
        # десяткова кома, як і роздільник «;», — під українську Excel: число з крапкою вона читає як текст,
        # а деякі (напр. «12.1») — як дату
        writer.writerows([cell.replace(".", ",") for cell in row] for row in table_rows(experiments, mode))

    h, w = original.shape
    m = MODES[mode]
    labels = "; ".join(r.label for r in experiments[0].results.values())
    lines = [
        "# Результати експериментів DCT vs DWT",
        "",
        f"- Дата: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Зображення: {source}, {w}×{h}, 8 біт/піксель ({h * w} байт без стиснення)",
        f"- Методи: {labels}",
        f"- Спосіб стиснення: {m.title}. {m.description}",
        "",
        markdown_table(experiments, mode),
        "",
        "Позначення:",
        "- **Збережено коеф.** — кількість ненульових коефіцієнтів після відкидання / загальна кількість.",
        "- **CR (коеф.)** = N / N_ненульових — коефіцієнт стиснення за кількістю коефіцієнтів.",
        "- **Розмір zlib** — оцінка розміру закодованих даних: цілі коефіцієнти (для квантування — індекси) "
        "у порядку від низьких частот до високих, int16, стиснення zlib; заголовок не враховано. "
        "Зображення відновлюється з цих самих цілих значень.",
        "- **CR (файл)** = (H·W байт) / розмір zlib; **bpp** = 8·розмір / (H·W).",
        "- **PSNR** = 10·log10(255² / MSE).",
        "",
        "Файли: `original.png` — оригінал, `compare_*.png` — Original | DCT | DWT для кожного рівня, "
        "`overview.png` — усі рівні, `psnr_curves.png` — графіки, `dct_*.png` / `dwt_*.png` — відновлені "
        "зображення, `results.csv` — таблиця для Excel.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
