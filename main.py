"""
Human Perception & Performance Tester
=====================================

Run the experiment app:      python main.py

Contents
  1. Imports
  2. Settings
       2.1 Experiment conditions (contrast, size, Fitts' trial layout)
       2.2 Experimenter panel look
       2.3 Identification task image
       2.4 CSV logging
  3. Shared helpers
  4. Data logging              (was logger.py)
  5. Tasks                     (was tasks.py)
       5.1 Fitts' law task
       5.2 Identification task
  6. Results summary window    (was summary.py)
  7. Main application window   (was main.py)
  8. Offline analysis          (was analysis.py)
  9. Entry point
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import csv
import os
import random
import statistics
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import ttk


try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


# =============================================================================
# 2. SETTINGS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 2.1 Experiment conditions ----------
# Each contrast level is a (text color, background color) pair.
CONTRAST_LEVELS = {
    "High":   ("#000000", "#FFFFFF"),   # black on white
    "Medium": ("#5A5A5A", "#E0E0E0"),   # dark grey on light grey
    "Low":    ("#A0A0A0", "#D8D8D8"),   # light grey on light grey
}

# Font sizes approximate "visual angle": bigger font = bigger visual angle.
SIZE_LEVELS = {"Small": 10, "Medium": 16, "Large": 28}

# Display order for the text-valued settings (summary window and analysis)
ORDER = {
    "contrast": ["High", "Medium", "Low"],
    "size": ["Small", "Medium", "Large"],
}

# Fitts' law trial layout
DISTANCES = [150, 300, 450]     # px, start-button centre to target centre
TARGET_SIZES = [20, 50, 100]    # px, side of the square target
REPS = 3                        # repetitions of each combination

# Start button geometry (fixed, on the left, vertically centred)
START_X = 100

# ---------- 2.2 Experimenter panel look (not part of the experiment) ----------
PANEL_BG = "#F4F5F7"
CARD_BORDER = "#D0D4DA"
ACCENT = "#2F6FED"
ACCENT_DARK = "#1F55C4"
TEXT = "#1F2933"
FONT = "Segoe UI"

# ---------- 2.3 Identification task image ----------
# Right-hand edge of the vase, top to bottom, as (offset from centre, y) points.
# Mirrored on the left, the two dark gaps read as facing face profiles.
VASE_PROFILE = [(55, 0), (40, 20), (38, 50), (34, 75), (37, 90), (20, 112),
                (33, 125), (27, 145), (34, 155), (27, 168), (36, 185),
                (28, 210), (44, 235), (62, 265), (72, 290), (72, 300)]
ANSWERS = ["Vase", "Two faces", "Both", "Can't tell"]

# ---------- 2.4 CSV logging ----------
CSV_PATH = os.path.join(BASE_DIR, "results.csv")

COLUMNS = [
    "participant_id", "timestamp", "task_type",
    "contrast", "size", "brightness",
    "distance_px", "target_size_px", "misses",
    "response", "time_taken_s", "ease_of_use_1_5",
]

# ---------- 2.5 Analysis ----------
EXCLUDE_IDS = {"TEST"}      # participant IDs dropped from the analysis (practice runs)
CHART_DPI = 150


# =============================================================================
# 3. SHARED HELPERS
# =============================================================================
def scale_color(hex_color, factor):
    """Darken a hex color by factor (1.0 = unchanged, 0.3 = very dark)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# 4. DATA LOGGING
# =============================================================================
# Rows saved since the app was opened (the summary window reads these)
SESSION_ROWS = []


def log_result(row):
    """Append one result row to results.csv, creating it with headers if needed.
    Columns a task doesn't use (e.g. distance for the identification task)
    are left blank."""
    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    SESSION_ROWS.append({k: str(v) for k, v in row.items()})

    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS,
                                restval="", extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# 5. TASKS
# =============================================================================

# -----------------------------------------------------------------------------
# 5.1 Fitts' law task
# -----------------------------------------------------------------------------
class FittsTask(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.results = []          # one dict per trial, written to CSV after the rating
        self.rating_frame = None

        # 3 distances x 3 sizes x 3 reps = 27 trials, shuffled
        self.trials = [(d, s) for d in DISTANCES
                       for s in TARGET_SIZES for _ in range(REPS)]
        random.shuffle(self.trials)
        self.trial_index = 0
        self.current = None
        self.misses = 0
        self.t0 = 0.0

        # Current look (set by the app through apply_theme)
        self.fg, self.bg, self.font_size = "#000000", "#FFFFFF", 16

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        self.mode = "waiting_start"   # waiting_start / waiting_target / rating / done
        self.redraw()

    # ---------- Theme ----------
    def apply_theme(self, fg, bg, font_size):
        self.fg, self.bg, self.font_size = fg, bg, font_size
        self.redraw()

    # ---------- Drawing ----------
    def center_y(self):
        return self.canvas.winfo_height() // 2

    def redraw(self):
        # The rating buttons only exist in "rating" mode
        if self.mode != "rating" and self.rating_frame is not None:
            self.rating_frame.destroy()
            self.rating_frame = None

        c = self.canvas
        c.delete("all")
        c.config(bg=self.bg)
        w, cy = c.winfo_width(), self.center_y()
        font = ("Arial", self.font_size)

        if self.mode in ("waiting_start", "waiting_target"):
            if self.mode == "waiting_start":
                msg = "Click START, then click the square as fast as you can."
            else:
                msg = "Click the square!"
            c.create_text(w // 2, 15, anchor="n", justify="center", fill=self.fg,
                          font=font, width=max(w - 40, 100),
                          text=f"Trial {self.trial_index + 1}/{len(self.trials)}\n{msg}")
            self.draw_start(cy)
            if self.mode == "waiting_target":
                self.draw_target(cy)
        elif self.mode == "rating":
            self.draw_rating(w)
        else:
            c.create_text(w // 2, cy, justify="center", fill=self.fg, font=font,
                          width=max(w - 40, 100),
                          text="Block complete! Thank you.\n"
                               "Change the settings and start another block if needed.")

    def start_half_size(self):
        """Start button half-width/half-height, grown to fit the label."""
        f = tkfont.Font(family="Arial", size=self.font_size)
        return (f.measure("START") + 30) // 2, (f.metrics("linespace") + 16) // 2

    def draw_start(self, cy):
        hw, hh = self.start_half_size()
        self.canvas.create_rectangle(START_X - hw, cy - hh,
                                     START_X + hw, cy + hh,
                                     outline=self.fg, width=3)
        self.canvas.create_text(START_X, cy, text="START", fill=self.fg,
                                font=("Arial", self.font_size))

    def draw_target(self, cy):
        d, s = self.current
        tx = START_X + d
        self.canvas.create_rectangle(tx - s / 2, cy - s / 2, tx + s / 2, cy + s / 2,
                                     fill=self.fg, outline=self.fg)

    def draw_rating(self, w):
        font = ("Arial", self.font_size)
        self.canvas.create_text(w // 2, 100, justify="center", fill=self.fg, font=font,
                                text="How easy was this task?\n1 = very difficult, 5 = very easy")
        self.rating_frame = tk.Frame(self, bg=self.bg)
        for n in range(1, 6):
            tk.Button(self.rating_frame, text=str(n), width=3, font=font,
                      bg=self.bg, fg=self.fg, activebackground=self.bg,
                      activeforeground=self.fg,
                      command=lambda n=n: self.submit_rating(n)).pack(side="left", padx=8)
        self.rating_frame.place(relx=0.5, rely=0.5, anchor="center")

    # ---------- Click handling ----------
    def inside(self, event, cx, half_w, half_h):
        return abs(event.x - cx) <= half_w and abs(event.y - self.center_y()) <= half_h

    def on_click(self, event):
        if self.mode == "waiting_start":
            hw, hh = self.start_half_size()
            if self.inside(event, START_X, hw, hh):
                self.begin_trial()
        elif self.mode == "waiting_target":
            d, s = self.current
            if self.inside(event, START_X + d, s / 2, s / 2):
                self.finish_trial()
            else:
                self.misses += 1

    # ---------- Trial flow ----------
    def begin_trial(self):
        self.current = self.trials[self.trial_index]
        self.misses = 0
        self.mode = "waiting_target"
        self.redraw()
        self.canvas.update_idletasks()      # make sure the target is on screen
        self.t0 = time.perf_counter()       # start the timer

    def finish_trial(self):
        elapsed = time.perf_counter() - self.t0
        d, s = self.current
        # Read the settings per trial so the CSV matches what was on screen
        row = dict(self.app.get_conditions())
        row.update({
            "task_type": "Target Hitting",
            "distance_px": d,
            "target_size_px": s,
            "time_taken_s": round(elapsed, 3),
            "misses": self.misses,
        })
        self.results.append(row)

        self.trial_index += 1
        if self.trial_index >= len(self.trials):
            self.mode = "rating"
        else:
            self.mode = "waiting_start"
        self.redraw()

    def submit_rating(self, n):
        # The rating belongs to the whole block, so store it once (first row)
        # to avoid counting it 27 times in the summary and analysis.
        for row in self.results:
            row["ease_of_use_1_5"] = ""
        self.results[0]["ease_of_use_1_5"] = n
        for row in self.results:
            log_result(row)
        self.mode = "done"
        self.redraw()


# -----------------------------------------------------------------------------
# 5.2 Identification task
# -----------------------------------------------------------------------------
class IdentificationTask(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.overlay = None         # frame holding the current buttons
        self.row = {}
        self.t0 = 0.0
        self.conditions = {}
        self.fg, self.bg, self.font_size = "#000000", "#FFFFFF", 16

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        self.mode = "ready"         # ready / identify / rating / done
        self.redraw()

    # ---------- Theme ----------
    def apply_theme(self, fg, bg, font_size):
        self.fg, self.bg, self.font_size = fg, bg, font_size
        self.redraw()

    # ---------- Drawing ----------
    def redraw(self):
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None

        c = self.canvas
        c.delete("all")
        c.config(bg=self.bg)
        w, h = c.winfo_width(), c.winfo_height()

        if self.mode == "ready":
            self.draw_text(w, "Click START when you are ready,\n"
                              "then choose what you see first.")
            self.show_buttons(["START"], lambda _: self.begin_trial(),
                              w // 2, h // 2)
        elif self.mode == "identify":
            top = 80
            mid = top + (h - top) // 2
            k = self.font_size / 16 * 0.6      # image size follows the Size setting
            self.draw_text(w, "What do you see?")
            self.draw_vase(w * 0.3, mid, k)
            self.show_buttons(ANSWERS, self.on_answer, int(w * 0.72), mid)
        elif self.mode == "rating":
            self.draw_text(w, "How easy was it to see the image?\n"
                              "1 = very difficult, 5 = very easy")
            self.show_buttons(range(1, 6), self.submit_rating,
                              w // 2, h // 2, horizontal=True)
        else:
            c.create_text(w // 2, h // 2, justify="center", fill=self.fg,
                          font=("Arial", self.font_size), width=max(w - 40, 100),
                          text="Thank you!\nChange the settings to repeat "
                               "under a new condition.")

    def draw_text(self, w, text):
        self.canvas.create_text(w // 2, 15, anchor="n", justify="center",
                                fill=self.fg, font=("Arial", self.font_size),
                                width=max(w - 40, 100), text=text)

    def draw_vase(self, cx, cy, k):
        """Dark panel with a vase-shaped cut-out; k scales the whole picture."""
        top = cy - 150 * k
        self.canvas.create_rectangle(cx - 110 * k, top - 12 * k,
                                     cx + 110 * k, top + 312 * k,
                                     fill=self.fg, outline=self.fg)
        right = [(cx + dx * k, top + y * k) for dx, y in VASE_PROFILE]
        left = [(cx - dx * k, top + y * k) for dx, y in reversed(VASE_PROFILE)]
        self.canvas.create_polygon(right + left, fill=self.bg, outline=self.bg)

    def show_buttons(self, labels, command, x, y, horizontal=False):
        """Place a group of themed buttons centred at (x, y)."""
        self.overlay = tk.Frame(self, bg=self.bg)
        for label in labels:
            tk.Button(self.overlay, text=str(label), font=("Arial", self.font_size),
                      bg=self.bg, fg=self.fg, activebackground=self.bg,
                      activeforeground=self.fg,
                      command=lambda l=label: command(l)
                      ).pack(side="left" if horizontal else "top",
                             fill="x", padx=8, pady=6)
        self.overlay.place(x=x, y=y, anchor="center")

    # ---------- Trial flow ----------
    def begin_trial(self):
        self.conditions = self.app.get_conditions()
        self.mode = "identify"
        self.redraw()
        self.canvas.update_idletasks()       # make sure the image is on screen
        self.t0 = time.perf_counter()        # start the timer

    def on_answer(self, answer):
        elapsed = time.perf_counter() - self.t0
        self.row = dict(self.conditions)
        self.row.update({
            "task_type": "Identification",
            "response": answer,
            "time_taken_s": round(elapsed, 3),
        })
        self.mode = "rating"
        self.redraw()

    def submit_rating(self, n):
        self.row["ease_of_use_1_5"] = n
        log_result(self.row)
        self.mode = "done"
        self.redraw()


# =============================================================================
# 6. RESULTS SUMMARY WINDOW
# =============================================================================
_summary_window = None      # the open summary window, if any


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def sort_key(column, value):
    """Sort High/Medium/Low and Small/Medium/Large sensibly, numbers numerically."""
    if column in ORDER:
        return ORDER[column].index(value) if value in ORDER[column] else 99
    number = to_float(value)
    return number if number is not None else 0


def group_stats(rows, column, value_column):
    """Group rows by `column`; return (key, n, mean, median) of `value_column`."""
    groups = {}
    for r in rows:
        key = r.get(column, "")
        value = to_float(r.get(value_column))
        if key != "" and value is not None:
            groups.setdefault(key, []).append(value)
    stats = [(k, len(v), statistics.mean(v), statistics.median(v))
             for k, v in groups.items()]
    return sorted(stats, key=lambda s: sort_key(column, s[0]))


def table(title, stats, label):
    """Format one small table as text."""
    lines = [title, f"  {label:<14}{'n':>5}{'mean':>9}{'median':>9}"]
    if not stats:
        lines.append("  (no data)")
    for key, n, mean, median in stats:
        lines.append(f"  {key:<14}{n:>5}{mean:>9.2f}{median:>9.2f}")
    return "\n".join(lines)


def response_counts(rows):
    """Identification answers per contrast/size condition."""
    counts = {}
    for r in rows:
        cond = f"{r['contrast']}/{r['size']}"
        counts.setdefault(cond, {})
        answer = r.get("response") or "?"
        counts[cond][answer] = counts[cond].get(answer, 0) + 1
    lines = ["Answers by condition (contrast/size)"]
    if not counts:
        lines.append("  (no data)")
    for cond in sorted(counts):
        answers = ", ".join(f"{a}: {n}" for a, n in sorted(counts[cond].items()))
        lines.append(f"  {cond:<14}{answers}")
    return "\n".join(lines)


def build_summary():
    rows = SESSION_ROWS
    if not rows:
        return ("No results yet this session.\n\n"
                "Run a task and click the 1-5 rating at the end;\n"
                "the summary will appear here automatically.")

    fitts = [r for r in rows if r["task_type"] == "Target Hitting"]
    ident = [r for r in rows if r["task_type"] == "Identification"]
    people = sorted({r["participant_id"] for r in rows})

    sections = [
        f"This session: {len(rows)} rows   (Fitts' trials: {len(fitts)}, "
        f"identification runs: {len(ident)})\n"
        f"Participants: {', '.join(people)}",

        "=== FITTS' LAW TASK (time in seconds) ===",
        table("By target size (px)", group_stats(fitts, "target_size_px", "time_taken_s"), "Size px"),
        table("By distance (px)", group_stats(fitts, "distance_px", "time_taken_s"), "Distance px"),
        table("By contrast", group_stats(fitts, "contrast", "time_taken_s"), "Contrast"),
        table("By size setting", group_stats(fitts, "size", "time_taken_s"), "Size"),
        table("By brightness (%)", group_stats(fitts, "brightness", "time_taken_s"), "Brightness"),
        table("Misses per trial, by target size", group_stats(fitts, "target_size_px", "misses"), "Size px"),

        "=== IDENTIFICATION TASK ===",
        table("Response time (s) by contrast", group_stats(ident, "contrast", "time_taken_s"), "Contrast"),
        table("Response time (s) by size setting", group_stats(ident, "size", "time_taken_s"), "Size"),
        response_counts(ident),

        "=== EASE OF USE (1 = very difficult, 5 = very easy) ===",
        table("By contrast", group_stats(rows, "contrast", "ease_of_use_1_5"), "Contrast"),
        table("By size setting", group_stats(rows, "size", "ease_of_use_1_5"), "Size"),
        table("By brightness (%)", group_stats(rows, "brightness", "ease_of_use_1_5"), "Brightness"),
        table("By task", group_stats(rows, "task_type", "ease_of_use_1_5"), "Task"),
    ]
    return "\n\n".join(sections)


def show_summary(parent):
    """Open the summary window. It updates itself every second."""
    global _summary_window
    if _summary_window is not None and _summary_window.winfo_exists():
        _summary_window.lift()          # already open: just bring it to the front
        return

    win = tk.Toplevel(parent)
    win.title("Results summary (this session)")
    win.geometry("640x680")
    _summary_window = win

    body = ttk.Frame(win)
    body.pack(fill="both", expand=True)
    text = tk.Text(body, font=("Consolas", 10), wrap="none", padx=12, pady=12)
    scroll = ttk.Scrollbar(body, command=text.yview)
    text.config(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)

    shown = {"content": None}

    def update():
        if not win.winfo_exists():
            return
        content = build_summary()
        if content != shown["content"]:         # only redraw when data changed
            position = text.yview()[0]          # keep the scroll position
            text.config(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", content)
            text.config(state="disabled")
            text.yview_moveto(position)
            shown["content"] = content
        win.after(1000, update)

    update()


# =============================================================================
# 7. MAIN APPLICATION WINDOW
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Human Perception & Performance Tester")
        self.geometry("1000x760")
        self.minsize(900, 700)
        self.configure(bg=PANEL_BG)

        # Experimenter settings (shared by all tasks)
        self.contrast_var = tk.StringVar(value="High")
        self.size_var = tk.StringVar(value="Medium")
        self.brightness_var = tk.IntVar(value=100)
        self.participant_var = tk.StringVar(value="P1")
        self.active_task = None

        self.setup_styles()
        self.build_control_panel()
        self.build_task_area()
        self.apply_theme()

    # ---------- Styling of the experimenter panel ----------
    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Card.TLabelframe", background=PANEL_BG,
                        bordercolor=CARD_BORDER, lightcolor=CARD_BORDER,
                        darkcolor=CARD_BORDER)
        style.configure("Card.TLabelframe.Label", background=PANEL_BG,
                        foreground=TEXT, font=(FONT, 10, "bold"))
        style.configure("Panel.TLabel", background=PANEL_BG,
                        foreground=TEXT, font=(FONT, 10))

        # Segmented-control look for the radio buttons
        style.configure("Seg.Toolbutton", font=(FONT, 10), padding=(12, 6),
                        background="#E3E6EC", foreground=TEXT,
                        relief="flat", borderwidth=0)
        style.map("Seg.Toolbutton",
                  background=[("selected", ACCENT), ("active", "#D3D8E1")],
                  foreground=[("selected", "white")])

        # Main action button
        style.configure("Accent.TButton", font=(FONT, 10, "bold"),
                        padding=(14, 8), background=ACCENT,
                        foreground="white", borderwidth=0)
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)])

        style.configure("Horizontal.TScale", background=PANEL_BG,
                        troughcolor=CARD_BORDER)

    # ---------- UI construction ----------
    def add_segmented_card(self, parent, column, title, options, variable):
        """A titled card containing a row of segmented option buttons."""
        card = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe",
                              padding=10)
        card.grid(row=0, column=column, padx=(0, 10), sticky="nsew")
        for option in options:
            ttk.Radiobutton(card, text=option, value=option, variable=variable,
                            style="Seg.Toolbutton", command=self.apply_theme
                            ).pack(side="left", padx=2)

    def build_control_panel(self):
        """Experimenter panel. Kept at a fixed style so it stays readable
        even when the task area is set to low contrast or small size."""
        panel = ttk.Frame(self, style="Panel.TFrame", padding=12)
        panel.pack(side="top", fill="x")

        self.add_segmented_card(panel, 0, "Contrast",
                                CONTRAST_LEVELS, self.contrast_var)
        self.add_segmented_card(panel, 1, "Size",
                                SIZE_LEVELS, self.size_var)

        # Brightness card
        card = ttk.LabelFrame(panel, text="Brightness",
                              style="Card.TLabelframe", padding=10)
        card.grid(row=0, column=2, padx=(0, 10), sticky="nsew")
        ttk.Scale(card, from_=30, to=100, orient="horizontal", length=170,
                  variable=self.brightness_var, command=self.on_brightness
                  ).pack(side="left")
        self.brightness_label = ttk.Label(card, text="100%", width=5,
                                          anchor="e", style="Panel.TLabel")
        self.brightness_label.pack(side="left", padx=(8, 0))

        # Session card: participant ID + task buttons
        card = ttk.LabelFrame(panel, text="Session",
                              style="Card.TLabelframe", padding=10)
        card.grid(row=0, column=3, sticky="nsew")
        row = ttk.Frame(card, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Participant ID", style="Panel.TLabel"
                  ).pack(side="left")
        ttk.Entry(row, textvariable=self.participant_var, width=8
                  ).pack(side="left", padx=(8, 0))
        ttk.Button(card, text="▶  Fitts' Law Task", style="Accent.TButton",
                   command=lambda: self.start_task(FittsTask)
                   ).pack(fill="x", pady=(8, 0))
        ttk.Button(card, text="▶  Identification Task", style="Accent.TButton",
                   command=lambda: self.start_task(IdentificationTask)
                   ).pack(fill="x", pady=(6, 0))
        ttk.Button(card, text="View Summary", style="Accent.TButton",
                   command=lambda: show_summary(self)
                   ).pack(fill="x", pady=(6, 0))

    def build_task_area(self):
        """The area participants see. Everything here gets restyled."""
        self.task_area = tk.Frame(self, highlightthickness=1,
                                  highlightbackground=CARD_BORDER)
        self.task_area.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Placeholder preview content
        self.title_label = tk.Label(self.task_area, text="Preview Area")
        self.title_label.pack(pady=(40, 10))

        self.body_label = tk.Label(
            self.task_area,
            text="This text changes with contrast, size and brightness.",
            wraplength=700)
        self.body_label.pack(pady=10)

        self.sample_button = tk.Button(self.task_area, text="Sample Button")
        self.sample_button.pack(pady=20)

        # Every widget that should respond to the theme
        self.themed_widgets = [self.title_label, self.body_label,
                               self.sample_button]

    # ---------- Tasks ----------
    def start_task(self, task_class):
        """Hide the preview and start a fresh instance of the chosen task."""
        for w in self.themed_widgets:
            w.pack_forget()
        if self.active_task is not None:
            self.active_task.destroy()
        self.active_task = task_class(self.task_area, self)
        self.active_task.pack(fill="both", expand=True)
        self.apply_theme()

    # ---------- Theme logic ----------
    def get_conditions(self):
        """Current settings, ready to be written to the CSV."""
        return {
            "contrast": self.contrast_var.get(),
            "size": self.size_var.get(),
            "brightness": self.brightness_var.get(),
            "participant_id": self.participant_var.get().strip() or "unknown",
        }

    def on_brightness(self, _value=None):
        self.brightness_label.config(text=f"{self.brightness_var.get()}%")
        self.apply_theme()

    def apply_theme(self):
        """Read all three controls and restyle the task area."""
        fg, bg = CONTRAST_LEVELS[self.contrast_var.get()]
        factor = self.brightness_var.get() / 100
        fg, bg = scale_color(fg, factor), scale_color(bg, factor)
        font_size = SIZE_LEVELS[self.size_var.get()]

        self.task_area.config(bg=bg)
        for w in self.themed_widgets:
            w.config(bg=bg, fg=fg, font=("Arial", font_size))
        self.sample_button.config(activebackground=bg, activeforeground=fg)

        # Restyle the running task if one is on screen
        if self.active_task is not None and self.active_task.winfo_exists():
            self.active_task.apply_theme(fg, bg, font_size)


# =============================================================================
# 8. OFFLINE ANALYSIS  (python main.py analyse)
# =============================================================================
# Outputs: console tables, charts.png, fitts_fit.png, ident_charts.png

# ---------- 8.1 Helpers ----------
def heading(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def ordered(index, column):
    """Return `index` in the preferred display order for `column`."""
    wanted = ORDER.get(column)
    if wanted is None:
        return index
    return [v for v in wanted if v in index]


def summarize(data, by, value, title=None):
    """Print and return count / mean / median / std of `value` grouped by `by`."""
    result = data.groupby(by)[value].agg(["count", "mean", "median", "std"])
    result = result.reindex(ordered(result.index, by)).round(2)
    print(f"\n{title or f'{value} by {by}'}")
    print(result.to_string())
    return result


def mean_by(data, by, value):
    """Mean of `value` per group of `by`, in display order (for charts)."""
    means = data.groupby(by)[value].mean()
    return means.reindex(ordered(means.index, by))


def bar(ax, series, title, xlabel, ylabel=None):
    series.plot.bar(ax=ax, rot=0, color=ACCENT)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def save_chart(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, filename), dpi=CHART_DPI)
    print(f"Saved {filename}")


# ---------- 8.2 Load data ----------
def load_data():
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"results.csv not found at {CSV_PATH}. Run some trials first.")

    df = pd.read_csv(CSV_PATH)
    df = df[~df["participant_id"].isin(EXCLUDE_IDS)]

    # Make sure numeric columns really are numeric (blank cells become NaN)
    for col in ["brightness", "distance_px", "target_size_px", "misses",
                "time_taken_s", "ease_of_use_1_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        raise SystemExit("No rows left after filtering. Nothing to analyse.")
    return df


# ---------- 8.3 Fitts' law task ----------
def fitts_tables(fitts):
    heading("FITTS' LAW TASK: time taken (s)")
    for column in ["target_size_px", "distance_px", "contrast", "size", "brightness"]:
        summarize(fitts, column, "time_taken_s")

    heading("FITTS' LAW TASK: misses per trial")
    summarize(fitts, "target_size_px", "misses")
    summarize(fitts, "contrast", "misses")


def fitts_regression(fitts):
    """Fit MT = a + b * ID, with ID = log2(D / W + 1) (Shannon formulation)."""
    heading("FITTS' LAW REGRESSION")

    fitts = fitts.copy()
    fitts["ID"] = np.log2(fitts["distance_px"] / fitts["target_size_px"] + 1)

    slope, intercept = np.polyfit(fitts["ID"], fitts["time_taken_s"], 1)
    r = fitts["ID"].corr(fitts["time_taken_s"])

    print(f"MT = {intercept:.3f} + {slope:.3f} * ID")
    print(f"R^2 = {r ** 2:.3f}   (n = {len(fitts)} trials)")
    print(f"Throughput approx. {1 / slope:.2f} bits/s")

    # Plot: raw trials + mean per ID + fitted line
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitts["ID"], fitts["time_taken_s"], alpha=0.3, label="Trials")
    per_id = fitts.groupby("ID")["time_taken_s"].mean()
    ax.scatter(per_id.index, per_id.values, color="black", zorder=3, label="Mean per ID")
    xs = np.linspace(fitts["ID"].min(), fitts["ID"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="red",
            label=f"MT = {intercept:.2f} + {slope:.2f}·ID  (R²={r ** 2:.2f})")
    ax.set_xlabel("Index of difficulty (bits)")
    ax.set_ylabel("Movement time (s)")
    ax.set_title("Fitts' law fit")
    ax.legend()
    save_chart(fig, "fitts_fit.png")


def fitts_charts(fitts):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    bar(axes[0], mean_by(fitts, "target_size_px", "time_taken_s"),
        "Time by target size", "Target size (px)", "Mean time (s)")
    bar(axes[1], mean_by(fitts, "distance_px", "time_taken_s"),
        "Time by distance", "Distance (px)")
    bar(axes[2], mean_by(fitts, "contrast", "time_taken_s"),
        "Time by contrast", "Contrast")
    bar(axes[3], mean_by(fitts, "brightness", "time_taken_s"),
        "Time by brightness", "Brightness (%)")
    save_chart(fig, "charts.png")


def analyse_fitts(df):
    fitts = df[df["task_type"] == "Target Hitting"]
    if fitts.empty:
        print("\nNo Fitts' law data found.")
        return
    fitts_tables(fitts)
    fitts_regression(fitts)
    fitts_charts(fitts)


# ---------- 8.4 Identification task ----------
def analyse_identification(df):
    ident = df[df["task_type"] == "Identification"]
    if ident.empty:
        print("\nNo identification data found.")
        return

    heading("IDENTIFICATION TASK")
    summarize(ident, "contrast", "time_taken_s", "Response time (s) by contrast")
    summarize(ident, "size", "time_taken_s", "Response time (s) by size setting")

    answers = pd.crosstab(ident["contrast"], ident["response"])
    answers = answers.reindex(ordered(answers.index, "contrast"))
    print("\nAnswers by contrast")
    print(answers.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bar(axes[0], mean_by(ident, "contrast", "time_taken_s"),
        "Response time by contrast", "Contrast", "Mean time (s)")
    answers.plot.bar(ax=axes[1], rot=0, title="Answers by contrast")
    axes[1].set_xlabel("Contrast")
    axes[1].set_ylabel("Count")
    save_chart(fig, "ident_charts.png")


# ---------- 8.5 Ease of use ratings ----------
def analyse_ease_of_use(df):
    rated = df.dropna(subset=["ease_of_use_1_5"])   # blank rows are skipped
    if rated.empty:
        print("\nNo ease-of-use ratings found.")
        return

    heading("EASE OF USE (1 = very difficult, 5 = very easy)")
    for column in ["task_type", "contrast", "size", "brightness"]:
        summarize(rated, column, "ease_of_use_1_5")


# ---------- 8.6 Run everything ----------
def run_analysis():
    if not ANALYSIS_AVAILABLE:
        raise SystemExit("Analysis needs pandas, numpy and matplotlib:\n"
                         "  pip install pandas numpy matplotlib")

    df = load_data()
    people = ", ".join(sorted(df["participant_id"].astype(str).unique()))
    print(f"Loaded {len(df)} rows. Participants: {people}")

    analyse_fitts(df)
    analyse_identification(df)
    analyse_ease_of_use(df)

    plt.show()


# =============================================================================
# 9. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("analyse", "analyze"):
        run_analysis()
    else:
        App().mainloop()