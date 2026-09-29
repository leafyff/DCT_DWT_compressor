"""Графічний інтерфейс (Tkinter + Matplotlib).

Вкладки:
  «Порівняння»                   — Original | DCT | DWT, керування стисненням, таблиця метрик;
  «Експерименти: таблиця і PSNR» — серія рівнів стиснення, таблиця результатів, графіки PSNR;
  «Експерименти: зображення»     — усі рівні серії поряд.
"""
import math
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from dctdwt import experiments as ex
from dctdwt import haar
from dctdwt.compression import MODES, DCTCodec, HaarCodec, Mode, Result, compress
from dctdwt.image_utils import PHOTO, SYNTHETIC, load_grayscale

SLIDER_STEPS = 1000   # кількість положень повзунка (шкала логарифмічна)
MAX_DWT_LEVELS = 8    # верхня межа рівнів Хаара в інтерфейсі
MUTED = "#52514e"     # колір допоміжного тексту
IMAGE_TYPES = [("Зображення", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.pgm *.webp"), ("Усі файли", "*.*")]
VIEWS = {  # режими показу вкладки «Порівняння»
    "recon": "Відновлені зображення",
    "error": "Карта похибки |x − x̂|",
    "coefs": "Збережені коефіцієнти log(1 + |c|)",
}
CR_NOTE = ("CR (коеф.) = N / N_ненульових.   CR (файл) = H·W байт / розмір закодованих даних, "
           "bpp = 8·розмір / (H·W); розмір оцінюється так: цілі коефіцієнти (int16, від низьких частот "
           "до високих), стиснені zlib.")
# у зведеній таблиці під зображеннями розмір у КБ не показуємо — його передають CR (файл) і bpp
SUMMARY_HEADERS = tuple(h for h in ex.METRIC_HEADERS if h != ex.ZLIB_COLUMN)


def _group(parent: tk.Misc, title: str) -> ttk.LabelFrame:
    """Рамка з заголовком для групи елементів лівої панелі."""
    box = ttk.LabelFrame(parent, text=title, padding=8)
    box.pack(fill="x", pady=(0, 8))
    return box


def _note(parent: tk.Misc, var: tk.StringVar) -> None:
    """Допоміжний сірий текст із переносом рядків."""
    label = ttk.Label(parent, textvariable=var, style="Muted.TLabel", wraplength=310, justify="left")
    label.pack(anchor="w", pady=(6, 0))


class FigurePanel(ttk.Frame):
    """Фігура Matplotlib з панеллю інструментів (лупа, переміщення, збереження) і підказкою під нею."""

    def __init__(self, master: tk.Misc, hint: str):
        """Створює фігуру; `footer` — місце для додаткових віджетів між фігурою та підказкою."""
        super().__init__(master)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        ttk.Label(self, text=hint, style="Muted.TLabel").pack(side="bottom", anchor="w", padx=6)
        self.footer = ttk.Frame(self, padding=(6, 0))
        self.footer.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def redraw(self, new_axes: bool = False) -> None:
        """Перемальовує фігуру; після створення нових осей скидає історію масштабування (кнопка «Home»)."""
        if new_axes:
            self.toolbar.update()
        self.canvas.draw_idle()


class App(tk.Tk):
    """Головне вікно: стан (зображення, кодеки, результати), побудова інтерфейсу та обробники подій."""

    def __init__(self, image_path: Path = SYNTHETIC, resize: bool = True, dwt_levels: int = 3,
                 mode: str = "fraction", levels: list[float] | None = None):
        """Будує інтерфейс і відкриває зображення (якщо не вдалося — синтетичне тестове).

        mode — початковий спосіб стиснення (ключ MODES), levels — рівні серії експериментів для нього.
        """
        super().__init__()
        self.title("Стиснення зображень: DCT vs DWT (Хаар)")
        self._fit_window()
        self._setup_style()

        # змінні Tk, прив'язані до віджетів
        self.mode_var = tk.StringVar(value=mode)         # спосіб стиснення (ключ MODES)
        self.value_var = tk.StringVar()                  # значення параметра в полі вводу
        self.levels_var = tk.IntVar(value=dwt_levels)    # рівні Хаара
        self.view_var = tk.StringVar(value="recon")      # режим показу (ключ VIEWS)
        self.resize_var = tk.BooleanVar(value=resize)    # зводити зображення до 512×512
        self.info_var = tk.StringVar()                   # опис відкритого зображення
        self.mode_desc_var = tk.StringVar()              # пояснення способу стиснення
        self.param_name_var = tk.StringVar()             # назва параметра
        self.range_var = tk.StringVar()                  # діапазон параметра
        self.verdict_var = tk.StringVar()                # який метод дав вищий PSNR
        self.exp_title_var = tk.StringVar()              # спосіб стиснення серії експериментів
        self.exp_param_var = tk.StringVar()              # підпис поля рівнів
        self.exp_levels_var = tk.StringVar()             # рівні серії, через «;»
        self.exp_status_var = tk.StringVar()             # стан серії

        # стан: зображення, кодеки, результати
        image = self._read_image(image_path)
        if image is None:
            image_path, image = SYNTHETIC, load_grayscale(SYNTHETIC)
        self.source_path = image_path                    # файл відкритого зображення
        self.original = image
        self.dct_codec, self.dwt_codec = self._make_codecs(image)
        self.results: dict[str, Result] = {}             # поточне стиснення: {"DCT": …, "DWT": …}
        self.experiments: list[ex.Experiment] = []       # остання серія експериментів
        self.curves: dict[str, np.ndarray] = {}          # криві PSNR для графіків серії
        self.exp_mode = mode                             # спосіб стиснення останньої серії
        self.exp_stale = True                            # серія не відповідає поточним параметрам
        self.param_values = {k: m.default for k, m in MODES.items()}  # останнє значення параметра кожного способу
        self.exp_levels = {k: ex.format_levels(m.levels) for k, m in MODES.items()}  # рівні серії кожного способу
        if levels:
            self.exp_levels[mode] = ex.format_levels(levels)
        self.exp_levels_mode = mode                      # спосіб, рівні якого зараз у полі
        self.exp_levels_var.set(self.exp_levels[mode])
        self.images = []                                 # три AxesImage вкладки «Порівняння»
        self.subband_lines = []                          # межі піддіапазонів DWT (режим «Коефіцієнти»)
        self._pending = False                            # перерахунок заплановано, але ще не виконано
        self._syncing = False                            # повзунок рухається програмно

        # віджети
        self._build_toolbar()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.scale, self.levels_spin, self.compare_panel, self.metrics = self._build_compare_tab()
        self.exp_table, self.curves_panel = self._build_experiments_tab()
        self.grid_panel = FigurePanel(self.notebook, "Верхній рядок — DCT, нижній — DWT. "
                                      "Лупа збільшує однаковий фрагмент на всіх зображеннях.")
        self.notebook.add(self.grid_panel, text="  Експерименти: зображення  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.bind("<<Recompute>>", self._on_recompute)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._on_mode_change()
        self._on_image_changed()

    # ------------------------------------------------------------ побудова інтерфейсу

    def _fit_window(self) -> None:
        """Розмір вікна під екран; на Windows вікно розгортається на весь екран."""
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1480, sw - 40), min(900, sh - 100)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+10")
        self.minsize(min(1100, w), min(640, h))
        try:
            self.state("zoomed")
        except tk.TclError:  # стан «zoomed» підтримується не в усіх віконних системах
            pass

    def _setup_style(self) -> None:
        """Висота рядків таблиць під розмір шрифту та стиль допоміжного тексту."""
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=tkfont.nametofont("TkDefaultFont").metrics("linespace") + 6)
        style.configure("Muted.TLabel", foreground=MUTED)

    def _build_toolbar(self) -> None:
        """Верхня панель: вибір зображення, масштабування до 512×512, опис зображення, збереження."""
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Button(bar, text="Відкрити зображення…", command=self._open).pack(side="left")
        ttk.Button(bar, text="Синтетичне тестове", command=lambda: self._load(SYNTHETIC)).pack(side="left", padx=4)
        ttk.Button(bar, text="Фото (єнот)", command=lambda: self._load(PHOTO)).pack(side="left")
        ttk.Checkbutton(bar, text="Привести до 512×512", variable=self.resize_var,
                        command=lambda: self._load(self.source_path)).pack(side="left", padx=10)
        ttk.Label(bar, textvariable=self.info_var, style="Muted.TLabel").pack(side="left", padx=10)
        ttk.Button(bar, text="Зберегти поточне порівняння…", command=self._save_current).pack(side="right")

    def _build_compare_tab(self) -> tuple[ttk.Scale, ttk.Spinbox, FigurePanel, ttk.Treeview]:
        """Вкладка «Порівняння»: ліва панель керування, три зображення і таблиця метрик під ними."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Порівняння Original | DCT | DWT  ")
        side = ttk.Frame(tab, padding=(8, 8, 4, 8))
        side.pack(side="left", fill="y")

        box = _group(side, "Спосіб зменшення коефіцієнтів")
        for key, m in MODES.items():
            ttk.Radiobutton(box, text=m.title, value=key, variable=self.mode_var,
                            command=self._on_mode_change).pack(anchor="w")
        _note(box, self.mode_desc_var)

        box = _group(side, "Рівень стиснення")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, textvariable=self.param_name_var).pack(side="left")
        entry = ttk.Entry(row, textvariable=self.value_var, width=9, justify="right")
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", self._on_entry)
        entry.bind("<FocusOut>", self._on_entry)
        ttk.Button(row, text="OK", width=4, command=self._on_entry).pack(side="left")
        scale = ttk.Scale(box, from_=0, to=SLIDER_STEPS, orient="horizontal", command=self._on_scale)
        scale.pack(fill="x", pady=(8, 0))
        _note(box, self.range_var)

        box = _group(side, "DWT (вейвлет Хаара)")
        ttk.Label(box, text="Рівнів декомпозиції:").pack(side="left")
        spin = ttk.Spinbox(box, from_=1, to=MAX_DWT_LEVELS, width=4, state="readonly",
                           textvariable=self.levels_var, command=self._on_levels_change)
        spin.pack(side="left", padx=6)

        box = _group(side, "Відображення")
        for key, title in VIEWS.items():
            ttk.Radiobutton(box, text=title, value=key, variable=self.view_var,
                            command=self._refresh_plots).pack(anchor="w")

        panel = FigurePanel(tab, "Лупа на панелі нижче збільшує фрагмент одразу на всіх трьох зображеннях — "
                                 "так зручно порівнювати артефакти.")
        panel.pack(side="left", fill="both", expand=True)
        columns = ("Метод",) + SUMMARY_HEADERS
        metrics = ttk.Treeview(panel.footer, columns=columns, show="headings", height=2, selectmode="none")
        for col in columns:
            metrics.heading(col, text=col)
            metrics.column(col, width=110, anchor="w" if col == "Метод" else "e")
        for name in ("DCT", "DWT"):
            metrics.insert("", "end", iid=name)
        metrics.pack(fill="x", pady=(4, 0))
        ttk.Label(panel.footer, textvariable=self.verdict_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(panel.footer, text=CR_NOTE, style="Muted.TLabel").pack(anchor="w")
        return scale, spin, panel, metrics

    def _build_experiments_tab(self) -> tuple[ttk.Treeview, FigurePanel]:
        """Вкладка серії експериментів: рівні стиснення, таблиця результатів і графіки PSNR."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Експерименти: таблиця і PSNR  ")
        top = ttk.Frame(tab, padding=8)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.exp_title_var).pack(anchor="w")
        row = ttk.Frame(top)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, textvariable=self.exp_param_var).pack(side="left")
        entry = ttk.Entry(row, textvariable=self.exp_levels_var, width=36)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda _event: self._run_experiments())
        ttk.Button(row, text="Запустити", command=self._run_experiments).pack(side="left")
        ttk.Button(row, text="Зберегти звіт…", command=self._save_report).pack(side="left", padx=6)
        ttk.Label(row, textvariable=self.exp_status_var, style="Muted.TLabel").pack(side="left", padx=10)

        paned = ttk.PanedWindow(tab, orient="vertical")  # межу між таблицею і графіками можна перетягувати
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        frame = ttk.Frame(paned)
        table = ttk.Treeview(frame, columns=ex.HEADERS, show="headings", height=8, selectmode="browse")
        for col in ex.HEADERS:
            table.heading(col, text=col)
            table.column(col, width=110, anchor="w" if col in ("Рівень", "Метод") else "e")
        table.tag_configure("DWT", background="#f4f3ef")  # рядки DWT — з легким фоном
        scroll = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        table.pack(fill="both", expand=True)
        paned.add(frame, weight=1)
        panel = FigurePanel(paned, "Лупа збільшує вибрану ділянку графіка.")
        paned.add(panel, weight=2)
        return table, panel

    # ------------------------------------------------------------ зображення

    def _read_image(self, path: Path) -> np.ndarray | None:
        """Читає зображення; у разі помилки показує повідомлення і повертає None."""
        try:
            return load_grayscale(path, self.resize_var.get())
        except (OSError, ValueError) as err:  # немає файлу, не зображення або зображення замале
            messagebox.showerror("Помилка", f"Не вдалося відкрити зображення:\n{err}")
            return None

    def _make_codecs(self, image: np.ndarray) -> tuple[DCTCodec, HaarCodec]:
        """Кодеки DCT і DWT для зображення; кількість рівнів Хаара обмежується його розміром."""
        self.levels_var.set(min(self.levels_var.get(), haar.max_levels(image.shape), MAX_DWT_LEVELS))
        return DCTCodec(image), HaarCodec(image, self.levels_var.get())

    def _open(self) -> None:
        """Діалог вибору файлу зображення."""
        path = filedialog.askopenfilename(title="Оберіть зображення", filetypes=IMAGE_TYPES)
        if path:
            self._load(Path(path))

    def _load(self, path: Path) -> None:
        """Відкриває нове зображення і перераховує все."""
        image = self._read_image(path)
        if image is not None:
            self.source_path, self.original = path, image
            self.dct_codec, self.dwt_codec = self._make_codecs(image)
            self._on_image_changed()

    def _on_image_changed(self) -> None:
        """Оновлює підпис, межі рівнів Хаара, осі зображень і результати після зміни зображення."""
        h, w = self.original.shape
        self.info_var.set(f"{self.source_path.name} — {w}×{h}, відтінки сірого")
        self.levels_spin.config(to=min(haar.max_levels(self.original.shape), MAX_DWT_LEVELS))
        self._init_plots()
        self._mark_stale()
        self._update()
        self._on_tab_changed()  # якщо відкрита вкладка експериментів — одразу перерахувати серію

    def _on_levels_change(self) -> None:
        """Змінено кількість рівнів Хаара."""
        self.dwt_codec = HaarCodec(self.original, self.levels_var.get())
        self._draw_subband_lines()
        self._mark_stale()
        self._schedule_update()

    # ------------------------------------------------------------ параметр стиснення

    def _mode(self) -> Mode:
        """Поточний спосіб стиснення."""
        return MODES[self.mode_var.get()]

    def _slider_to_value(self, pos: float) -> float:
        """Положення повзунка 0…SLIDER_STEPS -> значення параметра (логарифмічно, 3 значущі цифри)."""
        m = self._mode()
        return float(f"{m.vmin * (m.vmax / m.vmin) ** (pos / SLIDER_STEPS):.3g}")

    def _value_to_slider(self, value: float) -> float:
        """Значення параметра -> положення повзунка (обернене до _slider_to_value)."""
        m = self._mode()
        return SLIDER_STEPS * math.log(value / m.vmin) / math.log(m.vmax / m.vmin)

    def _apply_value(self, value: float, move_slider: bool = True) -> None:
        """Запам'ятовує значення параметра, показує його в полі вводу (і на повзунку), планує перерахунок."""
        self.param_values[self.mode_var.get()] = value
        self.value_var.set(f"{value:g}")
        if move_slider:
            self._syncing = True  # scale.set викликає _on_scale — цей виклик треба пропустити
            self.scale.set(self._value_to_slider(value))
            self._syncing = False
        self._schedule_update()

    def _on_scale(self, pos: str) -> None:
        """Користувач рухає повзунок."""
        if not self._syncing:
            self._apply_value(self._slider_to_value(float(pos)), move_slider=False)

    def _on_entry(self, _event: tk.Event | None = None) -> None:
        """Користувач ввів значення (Enter, «OK» або вихід із поля); некоректне введення скасовується."""
        m = self._mode()
        try:
            value = float(self.value_var.get().replace(",", "."))
        except ValueError:
            value = math.nan
        if math.isnan(value):  # не число (зокрема «nan», яке float() приймає) — лишається попереднє значення
            value = self.param_values[m.key]
        self._apply_value(min(max(value, m.vmin), m.vmax))

    def _on_mode_change(self) -> None:
        """Змінено спосіб стиснення: оновлює підписи, повзунок і рівні серії експериментів."""
        m = self._mode()
        stronger = "ліворуч" if m.key == "fraction" else "праворуч"
        self.mode_desc_var.set(m.description)
        self.param_name_var.set(f"{m.param}:")
        self.range_var.set(f"Діапазон {m.vmin:g}…{m.vmax:g}, шкала логарифмічна; {stronger} — сильніше стиснення.")
        self.exp_title_var.set(f"Спосіб стиснення: {m.title} (змінюється на вкладці «Порівняння»).")
        self.exp_param_var.set(f"Рівні ({m.param}), через «;»:")
        self.exp_levels[self.exp_levels_mode] = self.exp_levels_var.get()  # рівні попереднього способу не губляться
        self.exp_levels_mode = m.key
        self.exp_levels_var.set(self.exp_levels[m.key])
        self._mark_stale()
        self._apply_value(self.param_values[m.key])

    # ------------------------------------------------------------ стиснення і показ

    def _schedule_update(self) -> None:
        """Планує перерахунок після обробки подій, що вже стоять у черзі; повторні запити ігноруються.

        Подія <<Recompute>> стає в кінець черги, тож усі рухи повзунка, накопичені за час попереднього
        перерахунку, обробляються одним перерахунком — оновлення йдуть так часто, як встигають обчислюватися.
        """
        if not self._pending:
            self._pending = True
            self.event_generate("<<Recompute>>", when="tail")

    def _on_recompute(self, _event: tk.Event) -> None:
        """Обробник <<Recompute>>: виконує перерахунок, якщо його ще не зробили напряму."""
        if self._pending:
            self._update()

    def _update(self) -> None:
        """Стискає зображення обома методами з поточним параметром і оновлює зображення та метрики."""
        self._pending = False
        mode = self.mode_var.get()
        value = self.param_values[mode]
        self.results = {c.name: compress(c, self.original, mode, value) for c in (self.dct_codec, self.dwt_codec)}
        self._refresh_plots()
        self._refresh_metrics()

    def _init_plots(self) -> None:
        """Створює три осі (Original | DCT | DWT) під розмір поточного зображення."""
        fig = self.compare_panel.figure
        fig.clear()
        self.images = []
        for ax in fig.subplots(1, 3, sharex=True, sharey=True):  # спільні осі: лупа діє на всі три
            self.images.append(ax.imshow(self.original, cmap="gray", vmin=0, vmax=255, interpolation="nearest"))
            ax.set_xticks([])
            ax.set_yticks([])
        self.subband_lines = []  # старі лінії зникли разом з осями
        self._draw_subband_lines()
        self.compare_panel.redraw(new_axes=True)

    def _draw_subband_lines(self) -> None:
        """Малює межі піддіапазонів DWT поверх карти коефіцієнтів (видно лише в режимі «Коефіцієнти»)."""
        for line in self.subband_lines:
            line.remove()
        ax = self.images[2].axes
        h, w = self.original.shape
        self.subband_lines = []
        for lev in range(1, self.dwt_codec.levels + 1):
            bh, bw = h >> lev, w >> lev  # розмір піддіапазону рівня lev; межі проходять між пікселями (−0.5)
            self.subband_lines += ax.plot([-0.5, 2 * bw - 0.5], [bh - 0.5, bh - 0.5], color="#4fc3f7", lw=0.8)
            self.subband_lines += ax.plot([bw - 0.5, bw - 0.5], [-0.5, 2 * bh - 0.5], color="#4fc3f7", lw=0.8)
        for line in self.subband_lines:
            line.set_visible(self.view_var.get() == "coefs")

    def _refresh_plots(self) -> None:
        """Оновлює зображення DCT і DWT відповідно до режиму показу (масштаб лупи зберігається)."""
        view = self.view_var.get()
        results = list(self.results.values())
        if view == "recon":
            data = [r.image for r in results]
            cmap, top = "gray", 255.0
            titles = [f"{r.label}\nPSNR = {ex.format_psnr(r.psnr)} дБ · коеф.: {100 * r.fraction:.2f} %"
                      for r in results]
        elif view == "error":
            data = [np.abs(self.original.astype(np.int16) - r.image) for r in results]
            top = max(1.0, *(float(np.percentile(d, 99.5)) for d in data))  # спільна шкала для обох карт
            cmap = "magma"
            titles = [f"|x − x̂|, {r.label}\nMSE = {ex.format_mse(r.mse)} · шкала 0…{top:.0f}" for r in results]
        else:
            data = [np.log1p(np.abs(r.coefs)) for r in results]
            cmap, top = "magma", max(float(d.max()) for d in data)
            titles = [f"Коефіцієнти, {r.label}\nненульових: {r.kept} з {r.total}" for r in results]

        self.images[0].axes.set_title("Оригінал", fontsize=10)
        for im, d, title in zip(self.images[1:], data, titles):
            im.set_data(d)
            im.set_cmap(cmap)
            im.set_clim(0, top)
            im.axes.set_title(title, fontsize=10)
        for line in self.subband_lines:
            line.set_visible(view == "coefs")
        self.compare_panel.redraw()

    def _refresh_metrics(self) -> None:
        """Заповнює таблицю метрик під зображеннями та висновок щодо PSNR."""
        for r in self.results.values():
            cells = dict(zip(ex.METRIC_HEADERS, ex.metric_cells(r)))
            self.metrics.item(r.method, values=(r.label, *(cells[h] for h in SUMMARY_HEADERS)))
        dct_psnr, dwt_psnr = self.results["DCT"].psnr, self.results["DWT"].psnr
        best = "DCT" if dct_psnr > dwt_psnr else "DWT"
        if math.isinf(dct_psnr) and math.isinf(dwt_psnr):
            verdict = "Обидва методи відновили зображення без втрат."
        elif math.isinf(max(dct_psnr, dwt_psnr)):
            verdict = f"Без втрат зображення відновив лише {best}."
        elif abs(dct_psnr - dwt_psnr) < 0.05:
            verdict = "PSNR практично однаковий."
        else:
            verdict = f"Вищий PSNR: {best} (на {abs(dct_psnr - dwt_psnr):.2f} дБ)."
        self.verdict_var.set(verdict)

    # ------------------------------------------------------------ серія експериментів

    def _mark_stale(self) -> None:
        """Позначає серію експериментів застарілою після зміни зображення чи параметрів."""
        self.exp_stale = True
        if self.experiments:
            self.exp_status_var.set("Параметри змінилися — натисніть «Запустити», щоб оновити результати.")

    def _on_tab_changed(self, _event: tk.Event | None = None) -> None:
        """На вкладках експериментів застаріла серія перераховується автоматично."""
        if self.notebook.index("current") > 0 and self.exp_stale:
            self._run_experiments()

    def _run_experiments(self) -> bool:
        """Виконує серію експериментів і оновлює таблицю, графіки та сітку зображень. Повертає успіх."""
        m = self._mode()
        try:
            values = ex.parse_levels(self.exp_levels_var.get(), m.key)
        except ValueError as err:
            messagebox.showerror("Рівні стиснення", str(err))
            return False

        self.config(cursor="watch")
        self.exp_status_var.set("Обчислення…")
        self.update_idletasks()
        try:
            codecs = (self.dct_codec, self.dwt_codec)
            self.experiments = ex.run_experiments(codecs, self.original, m.key, values)
            self.curves = ex.sweep(codecs, self.original, m.key)
            self.exp_mode, self.exp_stale = m.key, False
        finally:
            self.config(cursor="")

        self.exp_table.delete(*self.exp_table.get_children())
        for row in ex.table_rows(self.experiments, m.key):
            self.exp_table.insert("", "end", values=row, tags=(row[1],))  # тег = метод (для фону рядка)
        ex.draw_curves(self.curves_panel.figure, self.experiments, self.curves, m.key)
        self.curves_panel.redraw(new_axes=True)
        ex.draw_grid(self.grid_panel.figure, self.original, self.experiments, m.key)
        self.grid_panel.redraw(new_axes=True)
        hint = "" if len(values) >= 3 else " За завданням потрібно щонайменше 3 рівні."
        self.exp_status_var.set(f"Готово ({self.source_path.name}): рівнів стиснення — {len(values)}.{hint}")
        return True

    # ------------------------------------------------------------ збереження

    def _save_current(self) -> None:
        """Зберігає поточне порівняння Original | DCT | DWT і відновлені зображення в обрану теку."""
        folder = filedialog.askdirectory(title="Тека для збереження")
        if not folder:
            return
        mode = self.mode_var.get()
        value = self.param_values[mode]
        tag = ex.file_tag(mode, value)
        ex.save_comparison(Path(folder), tag, self.original, self.results, ex.format_level(mode, value))
        messagebox.showinfo("Збережено", f"compare_{tag}.png, dct_{tag}.png, dwt_{tag}.png\nу теці {folder}")

    def _save_report(self) -> None:
        """Зберігає звіт серії експериментів у нову підтеку (за потреби спершу перераховує серію)."""
        if self.exp_stale and not self._run_experiments():
            return
        base = filedialog.askdirectory(title="Тека для звіту")
        if not base:
            return
        out = ex.report_dir(base, self.source_path, self.exp_mode)
        ex.save_report(out, self.original, self.experiments, self.curves, self.exp_mode, self.source_path.name)
        messagebox.showinfo("Звіт збережено", f"Файли збережено в теці:\n{out}")

    def _quit(self) -> None:
        """Завершує цикл подій і закриває вікно (інакше процес може лишитися через Matplotlib)."""
        self.quit()
        self.destroy()


def run(image_path: Path = SYNTHETIC, resize: bool = True, dwt_levels: int = 3, mode: str = "fraction",
        levels: list[float] | None = None) -> None:
    """Запускає графічний інтерфейс."""
    try:  # чіткий (не розмитий) інтерфейс на екранах з масштабуванням Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):  # не Windows або стара версія Windows
        pass
    App(image_path, resize, dwt_levels, mode, levels).mainloop()
