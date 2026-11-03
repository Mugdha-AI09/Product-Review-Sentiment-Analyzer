# ============================================================
# 🖥️  gui.py  —  Product Review Sentiment Analyzer
#     Run: python gui.py
#     Requires: models.pkl to exist (run pipeline_3_train first)
# ==========exit==================================================

import tkinter as tk
from tkinter import font as tkfont
import threading
import pickle
import re
import math
import time
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ── NLTK setup ────────────────────────────────────────────────
for pkg in ["stopwords", "wordnet", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))
negation_words = {"no", "not", "nor", "never", "neither", "without",
                  "nobody", "nothing", "nowhere", "hardly", "barely",
                  "doesn't", "isn't", "wasn't", "wouldn't", "couldn't"}
stop_words -= negation_words

# ── Colors ────────────────────────────────────────────────────
BG_DARK      = "#0A0E1A"
BG_CARD      = "#111827"
BG_INPUT     = "#1C2333"
ACCENT_GOLD  = "#F5C518"
ACCENT_BLUE  = "#3B82F6"
TEXT_PRIMARY = "#F1F5F9"
TEXT_MUTED   = "#64748B"
TEXT_DIM     = "#94A3B8"
BORDER       = "#1E2D45"

POS_COLOR    = "#10B981"   # emerald
NEU_COLOR    = "#F59E0B"   # amber
NEG_COLOR    = "#EF4444"   # red

SENTIMENT_COLORS = {
    "positive": POS_COLOR,
    "neutral" : NEU_COLOR,
    "negative": NEG_COLOR,
}
SENTIMENT_EMOJI = {
    "positive": "◉ POSITIVE",
    "neutral" : "◎ NEUTRAL",
    "negative": "◌ NEGATIVE",
}
SENTIMENT_DESC = {
    "positive": "This review expresses a positive sentiment.",
    "neutral" : "This review appears neutral or mixed.",
    "negative": "This review expresses a negative sentiment.",
}

# ── NLP helpers ───────────────────────────────────────────────
def _get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def preprocess(text: str) -> str:
    text   = str(text).lower()
    text   = re.sub(r"<.*?>",          " ", text)
    text   = re.sub(r"http\S+|www\S+", " ", text)
    text   = re.sub(r"[^a-z\s]",       " ", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(tok, _get_wordnet_pos(pos)) for tok, pos in tagged]
    return " ".join(tokens)

# ── Load model ────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_PKL = os.path.join(BASE_DIR, "models.pkl")

MODEL_DATA = None
def load_model():
    global MODEL_DATA
    try:
        with open(MODELS_PKL, "rb") as f:
            MODEL_DATA = pickle.load(f)
    except FileNotFoundError:
        MODEL_DATA = None

# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────
class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Product Review Sentiment Analyzer")
        self.geometry("860x720")
        self.minsize(760, 640)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # Fonts
        self.f_display = tkfont.Font(family="Georgia",        size=22, weight="bold")
        self.f_sub     = tkfont.Font(family="Georgia",        size=10, slant="italic")
        self.f_label   = tkfont.Font(family="Courier New",    size=9,  weight="bold")
        self.f_input   = tkfont.Font(family="Georgia",        size=12)
        self.f_result  = tkfont.Font(family="Georgia",        size=28, weight="bold")
        self.f_stat_v  = tkfont.Font(family="Courier New",    size=16, weight="bold")
        self.f_stat_l  = tkfont.Font(family="Courier New",    size=8)
        self.f_btn     = tkfont.Font(family="Courier New",    size=11, weight="bold")
        self.f_desc    = tkfont.Font(family="Georgia",        size=11, slant="italic")
        self.f_model   = tkfont.Font(family="Courier New",    size=9)

        self._build_ui()
        self._animate_header()

        # Load model in background
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    # ── Build UI ──────────────────────────────────────────────
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # ── Header ────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.grid(row=0, column=0, sticky="ew", padx=40, pady=(32, 4))

        self.title_lbl = tk.Label(
            hdr, text="PRODUCT REVIEW", font=self.f_display,
            bg=BG_DARK, fg=TEXT_PRIMARY, anchor="w"
        )
        self.title_lbl.pack(anchor="w")

        tk.Label(
            hdr, text="SENTIMENT  ANALYZER",
            font=tkfont.Font(family="Georgia", size=22, weight="bold"),
            bg=BG_DARK, fg=ACCENT_GOLD, anchor="w"
        ).pack(anchor="w")

        tk.Label(
            hdr, text="— Powered by ML Ensemble  ·  NLP Pipeline  ·  TF-IDF",
            font=self.f_sub, bg=BG_DARK, fg=TEXT_MUTED, anchor="w"
        ).pack(anchor="w", pady=(2, 0))

        # thin gold rule
        tk.Frame(self, bg=ACCENT_GOLD, height=1).grid(
            row=1, column=0, sticky="ew", padx=40, pady=(8, 0))

        # ── Input card ────────────────────────────────────────
        inp_card = tk.Frame(self, bg=BG_CARD, bd=0)
        inp_card.grid(row=2, column=0, sticky="nsew", padx=40, pady=20)
        inp_card.grid_rowconfigure(1, weight=1)
        inp_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            inp_card,
            text="ENTER PRODUCT REVIEW",
            font=self.f_label, bg=BG_CARD, fg=ACCENT_GOLD, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 6))

        # text area with scrollbar
        txt_frame = tk.Frame(inp_card, bg=BORDER, bd=1)
        txt_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 16))
        txt_frame.grid_rowconfigure(0, weight=1)
        txt_frame.grid_columnconfigure(0, weight=1)

        self.text_input = tk.Text(
            txt_frame, font=self.f_input,
            bg=BG_INPUT, fg=TEXT_PRIMARY,
            insertbackground=ACCENT_GOLD,
            relief="flat", bd=0,
            wrap="word",
            padx=14, pady=12,
            selectbackground=ACCENT_BLUE,
            spacing3=4
        )
        self.text_input.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(txt_frame, command=self.text_input.yview,
                          bg=BG_CARD, troughcolor=BG_INPUT,
                          relief="flat", bd=0, width=10)
        sb.grid(row=0, column=1, sticky="ns")
        self.text_input.config(yscrollcommand=sb.set)

        # placeholder
        self._placeholder_text = "Type or paste your product review here..."
        self._placeholder_active = True
        self.text_input.insert("1.0", self._placeholder_text)
        self.text_input.config(fg=TEXT_MUTED)
        self.text_input.bind("<FocusIn>",  self._clear_placeholder)
        self.text_input.bind("<FocusOut>", self._restore_placeholder)
        self.text_input.bind("<Return>",   self._on_enter)

        # char counter
        self.char_lbl = tk.Label(
            inp_card, text="0 characters",
            font=self.f_label, bg=BG_CARD, fg=TEXT_MUTED, anchor="e"
        )
        self.char_lbl.grid(row=2, column=0, sticky="e", padx=20, pady=(0, 14))
        self.text_input.bind("<KeyRelease>", self._update_char_count)

        # ── Analyze button ────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.grid(row=3, column=0, sticky="ew", padx=40, pady=(0, 16))

        self.analyze_btn = tk.Button(
            btn_row,
            text="  ANALYZE SENTIMENT  →",
            font=self.f_btn,
            bg=ACCENT_GOLD, fg="#0A0E1A",
            activebackground="#D4A017",
            activeforeground="#0A0E1A",
            relief="flat", bd=0,
            padx=28, pady=12,
            cursor="hand2",
            command=self._analyze
        )
        self.analyze_btn.pack(side="left")

        self.clear_btn = tk.Button(
            btn_row,
            text="  CLEAR",
            font=self.f_btn,
            bg=BG_CARD, fg=TEXT_MUTED,
            activebackground=BG_INPUT,
            activeforeground=TEXT_PRIMARY,
            relief="flat", bd=0,
            padx=20, pady=12,
            cursor="hand2",
            command=self._clear
        )
        self.clear_btn.pack(side="left", padx=(10, 0))

        self.status_lbl = tk.Label(
            btn_row, text="⬤  Model loading...",
            font=self.f_label, bg=BG_DARK, fg=TEXT_MUTED
        )
        self.status_lbl.pack(side="right")

        # ── Results panel ─────────────────────────────────────
        self.result_frame = tk.Frame(self, bg=BG_CARD)
        self.result_frame.grid(row=4, column=0, sticky="ew", padx=40, pady=(0, 28))
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(1, weight=1)
        self.result_frame.grid_columnconfigure(2, weight=1)
        self.result_frame.grid_columnconfigure(3, weight=2)

        # Sentiment label
        self.sent_var = tk.StringVar(value="—")
        self.sent_lbl = tk.Label(
            self.result_frame,
            textvariable=self.sent_var,
            font=self.f_result,
            bg=BG_CARD, fg=TEXT_MUTED,
            anchor="center", pady=18
        )
        self.sent_lbl.grid(row=0, column=0, columnspan=4,
                           sticky="ew", padx=24, pady=(20, 4))

        self.desc_var = tk.StringVar(value="Submit a review to see analysis.")
        tk.Label(
            self.result_frame,
            textvariable=self.desc_var,
            font=self.f_desc,
            bg=BG_CARD, fg=TEXT_MUTED,
            anchor="center"
        ).grid(row=1, column=0, columnspan=4, sticky="ew", padx=24, pady=(0, 16))

        # thin divider
        tk.Frame(self.result_frame, bg=BORDER, height=1).grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=24)

        # Stat boxes
        self.conf_val  = tk.StringVar(value="—")
        self.acc_val   = tk.StringVar(value="—")
        self.model_val = tk.StringVar(value="—")

        self._make_stat(self.result_frame, 3, 0, "CONFIDENCE",   self.conf_val)
        self._make_stat(self.result_frame, 3, 1, "MODEL ACCURACY", self.acc_val)
        self._make_stat_wide(self.result_frame, 3, 2, "MODEL USED", self.model_val)

        # confidence bar
        bar_frame = tk.Frame(self.result_frame, bg=BG_CARD)
        bar_frame.grid(row=4, column=0, columnspan=4,
                       sticky="ew", padx=24, pady=(4, 20))
        bar_frame.grid_columnconfigure(0, weight=1)

        tk.Label(bar_frame, text="CONFIDENCE SCORE",
                 font=self.f_label, bg=BG_CARD, fg=TEXT_MUTED,
                 anchor="w").grid(row=0, column=0, sticky="w")

        bar_bg = tk.Frame(bar_frame, bg=BG_INPUT, height=6)
        bar_bg.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bar_bg.grid_propagate(False)
        bar_bg.grid_columnconfigure(0, weight=1)

        self.conf_bar = tk.Frame(bar_bg, bg=TEXT_MUTED, height=6, width=0)
        self.conf_bar.place(x=0, y=0, height=6, width=0)
        self._bar_bg = bar_bg

    def _make_stat(self, parent, row, col, label, var):
        f = tk.Frame(parent, bg=BG_CARD)
        f.grid(row=row, column=col, sticky="nsew", padx=(24 if col==0 else 8, 8), pady=16)
        tk.Label(f, text=label, font=self.f_stat_l,
                 bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w")
        tk.Label(f, textvariable=var, font=self.f_stat_v,
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")

    def _make_stat_wide(self, parent, row, col, label, var):
        f = tk.Frame(parent, bg=BG_CARD)
        f.grid(row=row, column=col, columnspan=2,
               sticky="nsew", padx=(8, 24), pady=16)
        tk.Label(f, text=label, font=self.f_stat_l,
                 bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w")
        tk.Label(f, textvariable=var, font=self.f_model,
                 bg=BG_CARD, fg=TEXT_DIM, wraplength=300, justify="left").pack(anchor="w")

    # ── Placeholder ───────────────────────────────────────────
    def _clear_placeholder(self, _=None):
        if self._placeholder_active:
            self.text_input.delete("1.0", "end")
            self.text_input.config(fg=TEXT_PRIMARY)
            self._placeholder_active = False

    def _restore_placeholder(self, _=None):
        if not self.text_input.get("1.0", "end").strip():
            self.text_input.insert("1.0", self._placeholder_text)
            self.text_input.config(fg=TEXT_MUTED)
            self._placeholder_active = True

    def _update_char_count(self, _=None):
        if not self._placeholder_active:
            n = len(self.text_input.get("1.0", "end").strip())
            self.char_lbl.config(text=f"{n} character{'s' if n != 1 else ''}")

    def _on_enter(self, event):
        if not (event.state & 0x1):   # Shift+Enter = newline, Enter alone = analyze
            self._analyze()
            return "break"

    # ── Model loading ─────────────────────────────────────────
    def _load_model_thread(self):
        load_model()
        self.after(0, self._model_loaded)

    def _model_loaded(self):
        if MODEL_DATA:
            self.status_lbl.config(text="⬤  Model ready", fg=POS_COLOR)
        else:
            self.status_lbl.config(
                text="⬤  models.pkl not found — run pipeline_3 first",
                fg=NEG_COLOR)

    # ── Analyze ───────────────────────────────────────────────
    def _analyze(self, _=None):
        if MODEL_DATA is None:
            self.sent_var.set("MODEL NOT LOADED")
            self.desc_var.set("Please run pipeline_3_train.py first to generate models.pkl")
            self.sent_lbl.config(fg=NEG_COLOR)
            return

        raw = self.text_input.get("1.0", "end").strip()
        if not raw or self._placeholder_active:
            self._flash_input()
            return

        # Disable button while processing
        self.analyze_btn.config(state="disabled", text="  ANALYZING...  ")
        self.update()

        def _run():
            processed  = preprocess(raw)
            vectorizer = MODEL_DATA["vectorizer"]
            model      = MODEL_DATA["best_model"]
            le         = MODEL_DATA["label_encoder"]
            results    = MODEL_DATA.get("results", {})
            best_name  = MODEL_DATA["best_name"]

            vec       = vectorizer.transform([processed])
            pred_idx  = model.predict(vec)[0]
            pred_lbl  = le.inverse_transform([pred_idx])[0].lower()

            confidence = None
            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(vec)[0]
                confidence = proba[pred_idx] * 100

            # Best model accuracy
            acc = None
            if best_name in results:
                acc = results[best_name].get("accuracy")

            self.after(0, lambda: self._show_result(
                pred_lbl, confidence, acc, best_name))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, pred_lbl, confidence, acc, best_name):
        color = SENTIMENT_COLORS.get(pred_lbl, TEXT_DIM)
        label = SENTIMENT_EMOJI.get(pred_lbl, pred_lbl.upper())
        desc  = SENTIMENT_DESC.get(pred_lbl, "")

        self.sent_var.set(label)
        self.sent_lbl.config(fg=color)
        self.desc_var.set(desc)

        if confidence is not None:
            self.conf_val.set(f"{confidence:.1f}%")
        else:
            self.conf_val.set("N/A")

        if acc is not None:
            self.acc_val.set(f"{acc:.1f}%")
        else:
            self.acc_val.set("N/A")

        self.model_val.set(best_name)

        # update stat colors
        for widget in [self.conf_bar]:
            widget.config(bg=color)

        # animate confidence bar
        if confidence is not None:
            self._animate_bar(confidence / 100, color)

        self.analyze_btn.config(state="normal", text="  ANALYZE SENTIMENT  →")

    def _animate_bar(self, ratio, color, step=0):
        steps  = 30
        target = ratio
        if step > steps:
            return
        current = target * (step / steps)
        self.update_idletasks()
        w = self._bar_bg.winfo_width()
        fill = int(w * current)
        self.conf_bar.place(x=0, y=0, height=6, width=fill)
        self.conf_bar.config(bg=color)
        self.after(16, lambda: self._animate_bar(ratio, color, step + 1))

    def _flash_input(self):
        orig = BG_INPUT
        flash = "#2A1A2E"
        self.text_input.config(bg=flash)
        self.after(200, lambda: self.text_input.config(bg=orig))

    def _clear(self):
        self.text_input.delete("1.0", "end")
        self._placeholder_active = False
        self._restore_placeholder()
        self.char_lbl.config(text="0 characters")
        self.sent_var.set("—")
        self.sent_lbl.config(fg=TEXT_MUTED)
        self.desc_var.set("Submit a review to see analysis.")
        self.conf_val.set("—")
        self.acc_val.set("—")
        self.model_val.set("—")
        self.conf_bar.place(x=0, y=0, height=6, width=0)

    # ── Header animation ──────────────────────────────────────
    def _animate_header(self, tick=0):
        colors = [TEXT_PRIMARY, "#E2E8F0", TEXT_PRIMARY, "#CBD5E1"]
        self.title_lbl.config(fg=colors[tick % len(colors)])
        self.after(1800, lambda: self._animate_header(tick + 1))


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
