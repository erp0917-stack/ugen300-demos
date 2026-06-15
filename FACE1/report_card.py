# -*- coding: utf-8 -*-
"""
report_card.py — 滿版卡片式 A4 直式臉部美學分析報告 PNG（6 列網格）
列1 banner / 列2 三卡 / 列3 兩卡 / 列4 兩卡 / 列5 tips 五列 / 列6 免責
"""

import random
import datetime
import textwrap
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from face_helpers import SCORE_KEYS, TIP_KEYS

_DISCLAIMER = (
    "顏值評分為主觀且具偏見，僅供娛樂，勿用於未成年或作為真實評斷；"
    "基於靜態照片，不代表個人價值。"
)
_DISCLAIMER_EN = (
    "Beauty scores are subjective and biased — for entertainment only. "
    "Do not apply to minors or treat as genuine evaluation. "
    "Based on a static photo; does not represent personal worth."
)

_BG       = "#ffffff"
_CARD     = "#f5f5f5"
_BORDER   = "#cccccc"
_DARK     = "#1a1a1a"
_MID      = "#444444"
_LIGHT    = "#888888"
_BAR_ON   = "#2c2c2c"
_BAR_OFF  = "#e0e0e0"
_HEAD_BG  = "#2c2c2c"
_HEAD_FG  = "#ffffff"


# ── 字型 ────────────────────────────────────────────────────────────────
def _font():
    matplotlib.rcParams["font.family"]        = ["Microsoft JhengHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ── 卡片框 helper ────────────────────────────────────────────────────────
def _card(ax, title="", tfs=8.5):
    ax.set_facecolor(_CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if title:
        ax.set_title(title, fontsize=tfs, fontweight="bold",
                     color=_DARK, loc="left", pad=3)


def _off(ax):
    ax.axis("off")


# ── 列 1：Banner ─────────────────────────────────────────────────────────
def _row_banner(fig, spec, data, lang, L):
    ax = fig.add_subplot(spec)
    ax.set_facecolor(_HEAD_BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    _off(ax)

    ax.text(0.012, 0.68, L["report_title"], transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=_HEAD_FG, va="center")
    ax.text(0.012, 0.28, L["report_sub"], transform=ax.transAxes,
            fontsize=8, color="#aaaaaa", va="center")

    today = datetime.date.today()
    rid   = f"{today.strftime('%Y%m%d')}-{random.randint(1000,9999)}"
    meta  = data.get("meta", {})
    info  = (f"No.{rid}    {today}    "
             f"{L['age_range']} {meta.get('age_range','')}")
    ax.text(0.988, 0.5, info, transform=ax.transAxes,
            fontsize=7, color="#aaaaaa", va="center", ha="right")


# ── 列 2：照片 | 評分+summary | 五大維度 bar ────────────────────────────
def _row2(fig, spec, crop_bgr, data, L):
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=spec, wspace=0.010, width_ratios=[1, 1, 1.15])

    # 卡 A：照片
    ax_p = fig.add_subplot(gs[0])
    _card(ax_p, L["photo"])
    _off(ax_p)
    ax_p.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), aspect="auto")
    for sp in ax_p.spines.values():
        sp.set_visible(True); sp.set_edgecolor(_BORDER); sp.set_linewidth(1.2)

    # 卡 B：大分數 + summary
    ax_s = fig.add_subplot(gs[1])
    _card(ax_s, L["overall"])
    _off(ax_s)
    ax_s.text(0.5, 0.70, str(data["overall"]),
              transform=ax_s.transAxes, ha="center", va="center",
              fontsize=58, fontweight="bold", color=_DARK)
    ax_s.text(0.5, 0.48, "/ 100",
              transform=ax_s.transAxes, ha="center", fontsize=12, color=_LIGHT)
    ax_s.plot([0.12, 0.88], [0.38, 0.38], color=_BORDER, lw=0.7,
              transform=ax_s.transAxes)
    summary = data.get("summary", "")
    wrapped = "\n".join(textwrap.wrap(summary, width=14))
    ax_s.text(0.5, 0.28, wrapped,
              transform=ax_s.transAxes, ha="center", va="top",
              fontsize=7.5, color=_MID, multialignment="center")

    # 卡 C：五大維度橫條
    ax_b = fig.add_subplot(gs[2])
    _card(ax_b, L.get("scores_title", "五大維度評分"))
    n = len(SCORE_KEYS)
    ax_b.set_xlim(-28, 112)
    ax_b.set_ylim(-0.6, n - 0.4)
    ax_b.axis("off")
    for i, k in enumerate(reversed(SCORE_KEYS)):
        v = data["scores"][k]
        # 背景條
        ax_b.barh(i, 100, height=0.52, left=0, color=_BAR_OFF, zorder=1)
        # 實際條（加粗感）
        ax_b.barh(i, v,   height=0.52, left=0, color=_BAR_ON,  zorder=2)
        ax_b.text(-1, i, L[k], ha="right", va="center",
                  fontsize=8, color=_MID, fontweight="bold")
        ax_b.text(v + 2, i, str(v), ha="left", va="center",
                  fontsize=7.5, color=_DARK, fontweight="bold")


# ── 列 3：臉型結構 | 各部位評估 ─────────────────────────────────────────
def _row3(fig, spec, sketch_gray, data, L):
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=spec, wspace=0.010, width_ratios=[1, 1.1])

    # 卡 A：整塊單一 axes。
    # 列2 photo card 佔 1/3.15 ≈ 31.7% 行寬；列3 left card 佔 1/2.1 ≈ 47.6%。
    # sketch inset 寬度 = 31.7/47.6 ≈ 0.63，使視覺上與正上方原照片對齊。
    ax_left = fig.add_subplot(gs[0])
    _card(ax_left, "")
    _off(ax_left)

    # ── 線稿圖：inset 對齊上方照片欄寬 ──
    # [x0, y0, width, height] in axes fraction
    _iw = 0.63   # 與 row2 photo card 同寬
    ax_img = ax_left.inset_axes([0.02, 0.06, _iw, 0.86])
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.8)
    ax_img.imshow(sketch_gray, cmap="gray", aspect="auto")
    # 三庭五眼 grid lines（transAxes，永遠落在 inset 框內）
    _gc, _ga, _glw = "#888888", 0.55, 0.9
    for yp in [0.30, 0.55, 0.80]:
        ax_img.plot([0, 1], [yp, yp], transform=ax_img.transAxes,
                    color=_gc, lw=_glw, alpha=_ga, clip_on=True)
    for xp in [0.35, 0.65]:
        ax_img.plot([xp, xp], [0, 1], transform=ax_img.transAxes,
                    color=_gc, lw=_glw, alpha=_ga, clip_on=True)
    # 「結構線稿」標題放線稿圖左上方
    ax_left.text(0.02, 0.97, L["sketch"], transform=ax_left.transAxes,
                 fontsize=8.5, fontweight="bold", color=_DARK, va="top")

    # ── 右半：臉型結構五項（x 從 inset 右邊留 0.03 間距）──
    _tx = _iw + 0.02 + 0.03   # ≈ 0.68
    ax_left.text(_tx, 0.97, L["face_struct"], transform=ax_left.transAxes,
                 fontsize=8.5, fontweight="bold", color=_DARK, va="top")
    fs = data.get("face_struct", {})
    keys_labels = [
        ("shape", L["shape"]), ("forehead", L["forehead"]),
        ("cheekbone", L["cheekbone"]), ("jawline", L["jawline"]),
        ("midface", L["midface"]),
    ]
    n_items = len(keys_labels)
    y_top, y_bot = 0.85, 0.24          # 固定上下邊界，5 項均分不溢出
    row_h = (y_top - y_bot) / n_items  # = 0.122 per item
    for i, (k, lbl) in enumerate(keys_labels):
        yr = y_top - i * row_h
        ax_left.text(_tx, yr, f"▸ {lbl}", transform=ax_left.transAxes,
                     fontsize=7.5, fontweight="bold", color=_DARK, va="top")
        val = fs.get(k, "")
        short = textwrap.shorten(val, width=11, placeholder="…")
        ax_left.text(_tx + 0.03, yr - 0.072, short,
                     transform=ax_left.transAxes,
                     fontsize=7, color=_MID, va="top", clip_on=True)

    # 卡 B：parts 六項表格
    ax_pt = fig.add_subplot(gs[1])
    _card(ax_pt, L["parts"])
    _off(ax_pt)
    # 表頭
    cx = [0.03, 0.26, 0.44, 1.0]
    ax_pt.text(cx[1], 0.95, L["score_label"], transform=ax_pt.transAxes,
               fontsize=7.5, fontweight="bold", color=_DARK, va="top")
    ax_pt.text(cx[2], 0.95, L["note_label"],  transform=ax_pt.transAxes,
               fontsize=7.5, fontweight="bold", color=_DARK, va="top")
    ax_pt.plot([0, 1], [0.91, 0.91], color=_BORDER, lw=0.7,
               transform=ax_pt.transAxes)
    parts = data.get("parts", [])
    row_h = 0.135
    y = 0.87
    for i, p in enumerate(parts):
        bg = "#ececec" if i % 2 == 0 else _CARD
        ax_pt.add_patch(FancyBboxPatch(
            (0, y - row_h + 0.005), 1, row_h - 0.005,
            boxstyle="square,pad=0", transform=ax_pt.transAxes,
            fc=bg, ec="none", zorder=0))
        ax_pt.text(cx[0], y - 0.045, p.get("name", ""),
                   transform=ax_pt.transAxes,
                   fontsize=7.5, fontweight="bold", color=_DARK, va="center")
        sc = p.get("score", 0)
        ax_pt.text(cx[1], y - 0.045, str(sc),
                   transform=ax_pt.transAxes,
                   fontsize=7.5, fontweight="bold", color=_DARK, va="center")
        # 小橫條
        bw = 0.09
        ax_pt.barh(y - 0.045, bw, height=0.055,
                   left=cx[1] + 0.068, color=_BAR_OFF, zorder=1,
                   transform=ax_pt.transAxes, clip_on=False)
        ax_pt.barh(y - 0.045, sc / 100 * bw, height=0.055,
                   left=cx[1] + 0.068, color=_BAR_ON,  zorder=2,
                   transform=ax_pt.transAxes, clip_on=False)
        note = p.get("note", "")
        ax_pt.text(cx[2], y - 0.045, note,
                   transform=ax_pt.transAxes,
                   fontsize=6.8, color=_MID, va="center", clip_on=True)
        y -= row_h


# ── 列 4：優勢 | 可改善 ──────────────────────────────────────────────────
def _row4(fig, spec, data, L):
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.010)

    for ax_key, title, items, bullet in [
        (gs[0], L["pros"], data.get("pros", []), "✦"),
        (gs[1], L["cons"], data.get("cons", []), "◈"),
    ]:
        ax = fig.add_subplot(ax_key)
        _card(ax, title)
        _off(ax)
        y = 0.88
        for item in items:
            lines = textwrap.wrap(item, width=22)
            for j, line in enumerate(lines):
                prefix = f"{bullet} " if j == 0 else "   "
                ax.text(0.03, y, prefix + line,
                        transform=ax.transAxes,
                        fontsize=8, color=_MID, va="top", clip_on=True)
                y -= 0.115
            y -= 0.03
            if y < 0.04:
                break


# ── 列 5：造型保養建議（五列 × 三欄對照表）──────────────────────────
def _row5(fig, spec, data, L):
    ax = fig.add_subplot(spec)
    _card(ax, L["tips"])
    _off(ax)
    tips = data.get("tips", {})
    tip_map = [
        ("hair",     L["hair"]),
        ("brow",     L.get("brow_tip", L.get("brow", "眉型"))),
        ("makeup",   L["makeup"]),
        ("skincare", L["skincare"]),
        ("style",    L["style"]),
    ]

    # 三欄起點（比例約 1 : 2.5 : 2.5）
    cx = [0.015, 0.165, 0.575]

    # 表頭
    y = 0.89
    ax.text(cx[0], y, "項目",  transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color=_DARK, va="top")
    ax.text(cx[1], y, "取向 A", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color=_DARK, va="top")
    ax.text(cx[2], y, "取向 B", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color=_DARK, va="top")
    y -= 0.065
    ax.plot([0.01, 0.99], [y + 0.015, y + 0.015],
            color=_BORDER, lw=0.8, transform=ax.transAxes)

    row_h = 0.155
    for k, lbl in tip_map:
        val = tips.get(k, "")
        # 用「；」切兩半，再去掉每半開頭的「標籤：」前綴
        halves = val.split("；", 1)
        def _strip_prefix(s):
            return s.split("：", 1)[-1] if "：" in s else s
        col_a = _strip_prefix(halves[0]) if len(halves) > 0 else ""
        col_b = _strip_prefix(halves[1]) if len(halves) > 1 else ""

        mid_y = y - row_h / 2 + 0.01
        ax.text(cx[0], mid_y, lbl,   transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=_DARK, va="center")
        ax.text(cx[1], mid_y, col_a, transform=ax.transAxes,
                fontsize=7.5, color=_MID, va="center", clip_on=True)
        ax.text(cx[2], mid_y, col_b, transform=ax.transAxes,
                fontsize=7.5, color=_MID, va="center", clip_on=True)

        y -= row_h
        ax.plot([0.01, 0.99], [y + 0.015, y + 0.015],
                color=_BORDER, lw=0.5, transform=ax.transAxes)


# ── 列 6：免責聲明 ───────────────────────────────────────────────────────
def _row6(fig, spec, lang):
    ax = fig.add_subplot(spec)
    ax.set_facecolor(_BG)
    _off(ax)
    txt = _DISCLAIMER_EN if lang == "en" else _DISCLAIMER
    ax.text(0.5, 0.5, txt, transform=ax.transAxes,
            ha="center", va="center", fontsize=6,
            color="#aaaaaa", style="italic", multialignment="center")


# ── 主函式 ────────────────────────────────────────────────────────────────
def make_report_png(crop_bgr, sketch_gray, data, lang, lang_cfg, out_path):
    _font()

    L = lang_cfg["labels"]

    fig = plt.figure(figsize=(8.27, 11.69), facecolor=_BG, dpi=150)
    fig.patch.set_facecolor(_BG)

    gs = gridspec.GridSpec(
        6, 1, figure=fig,
        height_ratios=[0.055, 0.260, 0.235, 0.185, 0.215, 0.030],
        hspace=0.012,
        left=0.022, right=0.978,
        top=0.988, bottom=0.008,
    )

    _row_banner(fig, gs[0], data, lang, L)
    _row2(fig, gs[1], crop_bgr, data, L)
    _row3(fig, gs[2], sketch_gray, data, L)
    _row4(fig, gs[3], data, L)
    _row5(fig, gs[4], data, L)
    _row6(fig, gs[5], lang)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, pad_inches=0.04)
    plt.close(fig)
    print(f"[報告] PNG 已儲存：{out_path}")
