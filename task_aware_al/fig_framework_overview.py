"""
Framework overview figure — EdgeQuery mid-term report.
Saves: task_aware_al/report/figures/fig_framework_overview.png
Run:   python task_aware_al/fig_framework_overview.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "report", "figures", "fig_framework_overview.png")

# ── palette ────────────────────────────────────────────────────────────────────
H = ["#1a4a8a", "#1c7c3e", "#5a2285", "#bf4000"]   # top panel headers (blue/green/purple/orange)
H5, H6 = "#1a5878", "#7a2020"                        # bottom panel headers
CL, CT, CU = "#3a7bd5", "#ef7f3a", "#9e9e9e"        # L / T / U node colours
BX  = ["#ddeaf8", "#d5f0e0", "#ead0f8", "#fde8d8"]  # sub-box fills 1-4
BX5, BX6 = "#cce6f4", "#f8d8d8"
WHITE, INK, DIM = "#ffffff", "#0d0d0d", "#555555"

# ── figure setup ───────────────────────────────────────────────────────────────
# Wider aspect ratio so Word doesn't clip top/bottom when scaling to 6.5" width
FW, FH = 16.0, 8.5
fig = plt.figure(figsize=(FW, FH), facecolor="white")

# ── layout constants (figure fraction) ────────────────────────────────────────
G       = 0.012    # gutter between panels (wider for arrow space)
TOP_B   = 0.370    # bottom of top-row panels
TOP_H   = 0.560    # height of top-row panels  (in figure fraction)
BOT_B   = 0.055    # bottom of bottom-row panels
BOT_H   = 0.240    # height of bottom-row panels
PW      = (1 - 5*G) / 4
BW      = (1 - 3*G) / 2

# ── axes constants (axes fraction) ─────────────────────────────────────────────
HDR_H   = 0.095   # header height in axes fraction
SP      = 0.018   # sub-box inner vertical padding
LH      = 0.082   # one text line height in axes fraction
BG      = 0.014   # gap between sub-boxes
FS      = 9.8     # base font size (scaled up for readability in Word)

# derived: space used by one sub-box with n body lines
def box_h(n_body): return 2*SP + (1 + n_body)*LH

# check: 4 boxes × box_h(2) + 3×BG + HDR_H + SP ≤ 1.0
# = 4*(2*0.018 + 3*0.082) + 3*0.014 + 0.092 + 0.018
# = 4*0.282 + 0.042 + 0.11 = 1.128 + 0.152 → too much with 4 boxes of 3 lines
# 4*(0.036+2*0.082)+0.042+0.11 = 4*0.2+0.152 = 0.952 ✓  (2 body lines per box)

def make_ax(left, bottom, w, h):
    ax = fig.add_axes([left, bottom, w, h])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")
    return ax

axT = [make_ax(G + i*(PW+G), TOP_B, PW, TOP_H) for i in range(4)]
axB = [make_ax(G + i*(BW+G), BOT_B, BW, BOT_H) for i in range(2)]

# ── panel outline ──────────────────────────────────────────────────────────────
def outline(ax, c):
    ax.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.01",
                                facecolor="#f7f7f7", edgecolor=c, lw=1.8,
                                transform=ax.transAxes, clip_on=False, zorder=1))

def header(ax, c, num, title, fs=11.5):
    ax.add_patch(FancyBboxPatch((0, 1-HDR_H),1,HDR_H,
                                boxstyle="round,pad=0.01",
                                facecolor=c, edgecolor="none",
                                transform=ax.transAxes, clip_on=False, zorder=2))
    ax.text(0.04, 1-HDR_H/2, f"({num})", transform=ax.transAxes,
            fontsize=11.0, fontweight="bold", color=WHITE, va="center", zorder=3)
    ax.text(0.53, 1-HDR_H/2, title, transform=ax.transAxes,
            fontsize=fs, fontweight="bold", color=WHITE,
            va="center", ha="center", zorder=3)

def sbox(ax, y_top, title, body_lines, fill, ec, lp=0.035, rp=0.035):
    """Draw sub-box; returns y_bottom of box."""
    bh = box_h(len(body_lines))
    y0 = y_top - bh
    ax.add_patch(FancyBboxPatch((lp, y0), 1-lp-rp, bh,
                                boxstyle="round,pad=0.007",
                                facecolor=fill, edgecolor=ec, lw=0.85,
                                transform=ax.transAxes, clip_on=True, zorder=2))
    ty = y_top - SP
    ax.text(lp+0.025, ty, title, transform=ax.transAxes,
            fontsize=FS, fontweight="bold", color=INK, va="top", zorder=3)
    ty -= LH
    for line in body_lines:
        ax.text(lp+0.025, ty, line, transform=ax.transAxes,
                fontsize=FS-0.4, color=INK, va="top", zorder=3)
        ty -= LH
    return y0

def t(ax, x, y, s, sz=FS, bold=False, col=INK, ha="left"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=sz,
            fontweight="bold" if bold else "normal", color=col,
            va="top", ha=ha, clip_on=False)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Environment / Network
# ═══════════════════════════════════════════════════════════════════════════════
ax = axT[0]; outline(ax, H[0]); header(ax, H[0], 1, "Environment / Network")

# --- mini graph (upper portion of panel body) ---
NP = {"L1":(0.18,0.80),"L2":(0.52,0.90),"T1":(0.18,0.62),
      "T2":(0.52,0.72),"T3":(0.82,0.82),"U1":(0.82,0.62),"U2":(0.52,0.50)}
NC = {"L1":CL,"L2":CL,"T1":CT,"T2":CT,"T3":CT,"U1":CU,"U2":CU}
NL = {k: k[:1] for k in NP}
EDGES = [("L1","L2",True),("L1","T1",True),("L2","T2",True),
         ("T2","T3",True),("T1","T2",False),("T3","U1",False),
         ("T2","U2",False),("L2","T3",False)]
for a,b,obs in EDGES:
    x0,y0=NP[a]; x1,y1=NP[b]
    ax.plot([x0,x1],[y0,y1], color="#555" if obs else "#ccc",
            lw=1.2 if obs else 0.8, ls="-" if obs else (0,(4,3)),
            transform=ax.transAxes, zorder=2)
R=0.048
for name,(nx,ny) in NP.items():
    ax.add_patch(plt.Circle((nx,ny),R, color=NC[name], ec="#333",
                             lw=0.7, transform=ax.transAxes, zorder=3))
    ax.text(nx,ny, NL[name], transform=ax.transAxes, ha="center",
            va="center", fontsize=9.5, fontweight="bold", color=WHITE, zorder=4)

# --- text info ---
y = 0.42
t(ax, 0.04, y, "Contextual SBM (cSBM):", bold=True, col=H[0], sz=10.2)
info_lh = 0.062
for line in ["n = 90,  k = 3 balanced classes",
             "p_in = 0.13  |  p_out = 0.006",
             "~189 true edges per random seed",
             "12-dim attributes  (weak signal)",
             "15% edges observed at t = 0"]:
    y -= info_lh; t(ax, 0.06, y, "• "+line, sz=9.2)

# --- node legend ---
lx, ly = 0.04, 0.025
for col, lbl in [(CL,"L: Labelled"),(CT,"T: Target"),(CU,"U: Unlabelled")]:
    ax.add_patch(plt.Circle((lx+0.03, ly), 0.019, color=col,
                             ec="#444", lw=0.5, transform=ax.transAxes, zorder=3))
    t(ax, lx+0.060, ly+0.008, lbl, sz=9.2)
    lx += 0.32

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Input Representation
# ═══════════════════════════════════════════════════════════════════════════════
ax = axT[1]; outline(ax, H[1]); header(ax, H[1], 2, "Input Representation")
y = 1 - HDR_H - BG
for title, body in [
    ("Observed Graph  G_t = (V, E_t)",
     ["A_t after t queries;  15% visible at t=0"]),
    ("Labelled Set  L ⊂ V",
     ["Known labels y_L ∈ {1,…,k};  |L|=15–40%"]),
    ("Target Set  T ⊂ V\\L  (|T| = 15)",
     ["y_T hidden;  goal: maximise Acc_T"]),
    ("Budget  B = 200 queries",
     ["C = all unqueried pairs (i, j)"]),
]:
    y = sbox(ax, y, title, body, BX[1], H[1]) - BG

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Query Strategy
# ═══════════════════════════════════════════════════════════════════════════════
ax = axT[2]; outline(ax, H[2]); header(ax, H[2], 3, "Query Strategy")
y = 1 - HDR_H - BG
for title, body in [
    ("Step 1 — Edge probability",
     ["P̂(ij) = p̂_in·P_same + p̂_out·(1−P_same)"]),
    ("Step 2 — Confidence gain",
     ["ΔÂ_T = Â_T(A+e_ij) − Â_T(A)"]),
    ("Step 3 — Score & select",
     ["score(i,j) = P̂(ij)×ΔÂ_T;  argmax"]),
    ("Step 4 — Planning mode",
     ["Adaptive: re-score;  Static: rank once"]),
]:
    y = sbox(ax, y, title, body, BX[2], H[2]) - BG

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Oracle & Graph Update
# ═══════════════════════════════════════════════════════════════════════════════
ax = axT[3]; outline(ax, H[3]); header(ax, H[3], 4, "Oracle & Graph Update")

# mini diagram
def obox(ax, x, y, w, h, c, lbl):
    ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.01",
                                facecolor=c+"44", edgecolor=c, lw=1.4,
                                transform=ax.transAxes, zorder=3))
    ax.text(x+w/2, y+h/2, lbl, transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=c, zorder=4)

obox(ax, 0.04, 0.83, 0.40, 0.075, H[2], "Query\nStrategy")
obox(ax, 0.56, 0.83, 0.40, 0.075, H[3], "Oracle\n(Truth)")
ax.annotate("", xy=(0.56,0.868), xytext=(0.44,0.868),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=10))
ax.text(0.50, 0.912, "(i, j) ?", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=10.5)

y = 0.80
for fill, ec, title, body in [
    (BX[1], H[1], "Answer = 1 (edge exists)",
     ["A_{t+1} = A_t + e_{ij};  add to G_t"]),
    ("#f0f0f0", "#999", "Answer = 0 (no edge)",
     ["Pair (i,j) removed from C"]),
    (BX[3], H[3], "Repeat until budget B done",
     ["t = 1, 2, …, B;  evaluate on T"]),
]:
    y = sbox(ax, y, title, body, fill, ec) - BG

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Label Propagation
# ═══════════════════════════════════════════════════════════════════════════════
ax = axB[0]; outline(ax, H5); header(ax, H5, 5, "Label Propagation", fs=11.5)

CX1, CX2, CW = 0.02, 0.51, 0.46
y1 = 1 - HDR_H - BG
y2 = 1 - HDR_H - BG
for title, body, cx in [
    ("Closed-Form Solution:",
     ["S = D⁻¹A   (row-norm. adj.)",
      "F_U = (I−αS_UU)⁻¹ αS_UL Y_L",
      "α = 0.99;   F_L = Y_L  (clamped)"],
     CX1),
    ("Predictions & Confidence:",
     ["ŷ_u = argmax_c F_uc",
      "conf_u = max_c q_u(c)",
      "Unreached node → prior guess"],
     CX1),
]:
    y1 = sbox(ax, y1, title, body, BX5, H5, lp=cx, rp=1-cx-CW) - BG*0.7

for title, body, cx in [
    ("Why Label Propagation?",
     ["• Exact; no retraining needed",
      "• Per-edge ΔAcc computable instantly",
      "• All candidates scored per query"],
     CX2),
    ("Target Confidence Estimate:",
     ["Â_T(A) = (1/|T|) Σ_{u∈T} conf_u",
      "Used inside the value scorer"],
     CX2),
]:
    y2 = sbox(ax, y2, title, body, BX5, H5, lp=cx, rp=1-cx-CW) - BG*0.7

# --- mini propagation diagram (bottom-right of panel 5) ---
gx0, gy0 = 0.60, 0.06
mini_nodes = [(gx0+0.00, gy0+0.05, CL, "L"),
              (gx0+0.10, gy0+0.05, CL, "L"),
              (gx0+0.20, gy0+0.05, CT, "T"),
              (gx0+0.30, gy0+0.05, CU, "U")]
for j in range(3):
    x0n = mini_nodes[j]; x1n = mini_nodes[j+1]
    ax.plot([x0n[0], x1n[0]], [x0n[1], x1n[1]],
            color="#888", lw=1.0, transform=ax.transAxes, zorder=2)
for mx, my, mc, ml in mini_nodes:
    ax.add_patch(plt.Circle((mx, my), 0.018, color=mc, ec="#444",
                             lw=0.5, transform=ax.transAxes, zorder=3))
    ax.text(mx, my, ml, transform=ax.transAxes, ha="center",
            va="center", fontsize=7.5, fontweight="bold", color=WHITE, zorder=4)
ax.annotate("", xy=(gx0+0.14, gy0+0.10), xytext=(gx0+0.06, gy0+0.10),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=H5, lw=1.0, mutation_scale=8))
ax.text(gx0+0.10, gy0+0.13, "propagate", ha="center", va="bottom",
        fontsize=8.5, color=H5, style="italic", transform=ax.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════
ax = axB[1]; outline(ax, H6); header(ax, H6, 6, "Evaluation Metrics", fs=11.5)

CX1, CX2, CW = 0.02, 0.51, 0.46
y1 = 1 - HDR_H - BG
y2 = 1 - HDR_H - BG
for title, body, cx in [
    ("[1]  Accuracy Gain",
     ["Acc_T(B) − Acc_T(0)",
      "Improvement on 15 target nodes"],
     CX1),
    ("[2]  Gap Closed",
     ["(Acc_T(B)−Acc_T(0)) / (Acc_Tᵦᵘᴸᴸ−Acc_T(0))",
      "Fraction of achievable gain"],
     CX1),
]:
    y1 = sbox(ax, y1, title, body, BX6, H6, lp=cx, rp=1-cx-CW) - BG*0.7

for title, body, cx in [
    ("[3]  Target Reach",
     ["|{u∈T : u connected to L}| / |T|",
      "Primary mechanism for accuracy gain"],
     CX2),
    ("[4]  True Edges Found",
     ["|{(i,j) queried : A*(i,j)=1}|",
      "Link recall (not the goal)"],
     CX2),
]:
    y2 = sbox(ax, y2, title, body, BX6, H6, lp=cx, rp=1-cx-CW) - BG*0.7

# --- mini bar chart (gap-closed sketch) ---
bar_x0, bar_y0, bar_w = 0.06, 0.055, 0.06
bar_vals = [("Value", 0.50, H[2]), ("Heur.", 0.13, "#888"), ("Rand.", 0.08, "#bbb")]
for idx, (blbl, bval, bcol) in enumerate(bar_vals):
    bx = bar_x0 + idx * (bar_w + 0.025)
    bh = bval * 0.16
    ax.add_patch(FancyBboxPatch((bx, bar_y0), bar_w, bh,
                                boxstyle="round,pad=0.003",
                                facecolor=bcol, edgecolor="none",
                                transform=ax.transAxes, zorder=3))
    ax.text(bx + bar_w/2, bar_y0 - 0.008, blbl, ha="center", va="top",
            fontsize=8.0, color=DIM, transform=ax.transAxes)
    ax.text(bx + bar_w/2, bar_y0 + bh + 0.005, f"{bval:.0%}", ha="center",
            va="bottom", fontsize=8.0, fontweight="bold", color=bcol,
            transform=ax.transAxes)

ax.text(0.62, 0.035,
        "30 seeds + 30 held-out  |  "
        "Paired t-test, Wilcoxon, bootstrap CI, Holm",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=9.0, color=DIM, style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW ARROWS — between top panels (extend into panels so they're visible)
# ═══════════════════════════════════════════════════════════════════════════════
arrow_y = TOP_B + TOP_H * 0.52
ARROW_PAD = 0.015
for i in range(3):
    x0 = G + i*(PW+G) + PW - ARROW_PAD
    x1 = G + (i+1)*(PW+G) + ARROW_PAD
    fig.add_artist(FancyArrowPatch(
        (x0, arrow_y), (x1, arrow_y), transform=fig.transFigure,
        arrowstyle="-|>", color="#333", lw=2.5, mutation_scale=18, zorder=8))

# ── vertical arrows: bottom panels feed back to query loop ─────────────────────
vert_targets = [
    (0, H5, G + 1*(PW+G) + PW*0.5),
    (1, H6, G + 2*(PW+G) + PW*0.5),
]
for i, col, x_target in vert_targets:
    x_bot = G + i*(BW+G) + BW/2
    fig.add_artist(FancyArrowPatch(
        (x_bot, BOT_B+BOT_H+0.005), (x_target, TOP_B-0.005),
        transform=fig.transFigure,
        arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14, zorder=7,
        connectionstyle="arc3,rad=0.0"))

# ── simulation loop (dashed curve under all panels) ────────────────────────────
BOT_TOP = BOT_B + BOT_H          # top edge of bottom panels = 0.302
LY      = (BOT_TOP + TOP_B) / 2  # centre of inter-row gap ≈ 0.344
X_RIGHT = G + 3*(PW+G) + PW + 0.005
X_LEFT  = G - 0.005

loop_xs = [X_RIGHT, X_RIGHT, X_LEFT, X_LEFT]
loop_ys = [TOP_B + 0.03, LY, LY, TOP_B + 0.03]
fig.add_artist(Line2D(loop_xs, loop_ys, color=H[3], lw=2.0,
                      ls=(0,(5,3)), transform=fig.transFigure, zorder=5))
# arrowhead pointing up at left end
fig.add_artist(FancyArrowPatch(
    (X_LEFT, LY+0.01), (X_LEFT, TOP_B+0.025),
    transform=fig.transFigure,
    arrowstyle="-|>", color=H[3], lw=2.0, mutation_scale=14, zorder=5))
fig.text((X_LEFT+X_RIGHT)/2, LY-0.008,
         "Simulation Loop — repeat until budget B exhausted",
         ha="center", va="top", fontsize=12.0, color=H[3],
         fontweight="bold", transform=fig.transFigure)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL TITLE
# ═══════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.965,
         "EdgeQuery: Task-Aware Active Link Acquisition Framework "
         "for Graph-Based Inference",
         ha="center", va="top", fontsize=16.0, fontweight="bold",
         color=INK, transform=fig.transFigure)

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
ly_leg = 0.014
lx = 0.03
for col, desc in [(CL, "L = Labelled node"),
                  (CT, "T = Target  (y_T hidden)"),
                  (CU, "U = Unlabelled node")]:
    fig.add_artist(plt.Circle((lx, ly_leg), 0.0045, color=col, ec="#444",
                              lw=0.5, transform=fig.transFigure, zorder=5))
    fig.text(lx+0.012, ly_leg, desc+"   ", va="center",
             fontsize=10.5, transform=fig.transFigure, color=INK)
    lx += 0.17

fig.text(lx+0.01, ly_leg,
         "——  observed edge    - - -  hidden edge    "
         "n=90, k=3, |T|=15, B=200",
         va="center", fontsize=10.5, transform=fig.transFigure, color=DIM)

# ═══════════════════════════════════════════════════════════════════════════════
plt.savefig(OUT, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print("Saved:", OUT)
