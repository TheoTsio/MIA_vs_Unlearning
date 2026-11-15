"""Flow diagram of the loss-threshold MIA (Yeom et al., 2018), drawn in the style of
Figure 1 (MIA - Flow Diagram) of the paper.

Outputs: fig_mia_loss_flow.pdf (vector, for the paper) and fig_mia_loss_flow.png (preview).
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle, Circle, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GREEN_NET = "#1DB66E"     # target model (as in Fig. 1)
FRAME = "#7ED957"         # light green frame (as in Fig. 1)
PILL_RED = "#F2453D"      # non-membership
PILL_GREEN = "#1B5E3B"    # membership
PILL_ORANGE = "#F5A05A"   # score pill (as "Confidence" in Fig. 1)
SEP = "#2B5DA8"           # dotted separator
INK = "black"

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(0, 104)
ax.set_ylim(0, 46)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


# ------------------------------------------------------------------ icons
def arrow(x0, y0, x1, y1, ls="-", lw=1.3, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls,
                                mutation_scale=9, shrinkA=0, shrinkB=0))


def cylinder(cx, cy, w=6.0, h=7.5, lw=1.4):
    top, bot = cy + h / 2, cy - h / 2
    ry = w * 0.17
    ax.add_patch(Rectangle((cx - w / 2, bot), w, h, fc="white", ec="none", zorder=2))
    ax.plot([cx - w / 2, cx - w / 2], [bot, top], color=INK, lw=lw, zorder=3)
    ax.plot([cx + w / 2, cx + w / 2], [bot, top], color=INK, lw=lw, zorder=3)
    t = np.linspace(np.pi, 2 * np.pi, 60)
    for yy in (bot, cy - h / 6, cy + h / 6):  # bottom rim + two bands
        ax.plot(cx + w / 2 * np.cos(t), yy + ry * np.sin(t), color=INK, lw=lw, zorder=3)
    ax.add_patch(Ellipse((cx, top), w, 2 * ry, fc="white", ec=INK, lw=lw, zorder=4))


def network(cx, cy, color, layers=(3, 4, 2), dx=3.0, dy=2.3, r=0.75):
    pos = []
    for i, n in enumerate(layers):
        x = cx + (i - (len(layers) - 1) / 2) * dx
        pos.append([(x, cy + (j - (n - 1) / 2) * dy) for j in range(n)])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.9, zorder=2)
    for layer in pos:
        for p in layer:
            ax.add_patch(Circle(p, r, fc=color, ec=color, zorder=3))


def record(cx, cy, s=3.4):
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc="white", ec=INK, lw=1.2, zorder=3))
    ax.add_patch(Polygon([(cx - s / 2 + 0.4, cy - s / 2 + 0.5), (cx - 0.2, cy + 0.3),
                          (cx + 0.5, cy - 0.4), (cx + 0.9, cy), (cx + s / 2 - 0.4, cy - s / 2 + 0.5)],
                         closed=True, fc="white", ec=INK, lw=1.0, zorder=4))
    ax.add_patch(Circle((cx + 0.7, cy + 0.85), 0.38, fc="white", ec=INK, lw=1.0, zorder=4))


def pill(cx, cy, text, color, w, h=3.2, fs=8.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=color, ec="none", zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=4)


def label(x, y, text, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "top")
    kw.setdefault("fontsize", 8)
    ax.text(x, y, text, linespacing=1.15, **kw)


# ------------------------------------------------------------------ model owner side
SEPX = 36.5
ax.text(18, 45, "Model owner (black-box)", ha="center", va="top", fontsize=8,
        color="0.35", style="italic")

cylinder(7, 30)
label(7, 24.6, "Target\ntraining set")
arrow(10.8, 30, 19.5, 30)
label(15.2, 33.4, "Train", va="bottom")

network(25, 30, GREEN_NET)
label(25, 24.6, "Target\nModel")

record(25, 9)
label(25, 6.2, "Target record\n$(x, y)$")
arrow(25, 11.2, 25, 20.0)
label(23.6, 15.5, "Query $x$", ha="right", va="center")

# ------------------------------------------------------------------ separator
ax.plot([SEPX, SEPX], [2, 44], ls=(0, (1, 2)), color=SEP, lw=1.6)
ax.text(52, 45, "Adversary", ha="center", va="top", fontsize=8, color="0.35", style="italic")

# ------------------------------------------------------------------ posterior vector
col_x, cell_w, cell_h, n_cells, hi = 41.5, 3.2, 2.0, 6, 0
col_bot = 30 - n_cells * cell_h / 2
arrow(30, 30, col_x - 0.2, 30)
label(SEPX, 31.0, "Inference", va="bottom", bbox=dict(fc="white", ec="none", pad=0.5))
for i in range(n_cells):
    ax.add_patch(Rectangle((col_x, col_bot + i * cell_h), cell_w, cell_h,
                           fc=PILL_ORANGE if i == hi else "white", ec=INK, lw=1.0, zorder=3))
label(col_x + cell_w / 2, col_bot + n_cells * cell_h + 0.6, "Posterior\n$p(\\cdot \\mid x)$", va="bottom")
label(col_x + cell_w / 2, col_bot - 0.6, "$p(y \\mid x)$", color="#C0601A")

# ------------------------------------------------------------------ loss score
PX, PW = 57.0, 16.5
arrow(col_x + cell_w + 0.3, 30, PX - PW / 2 - 0.3, 30)
pill(PX, 30, "Loss  $-\\log\\, p(y \\mid x)$", PILL_ORANGE, w=PW, fs=7.8)

# true label y is needed: dashed path from the record to the loss pill
ax.plot([27.2, PX], [9, 9], ls=(0, (3, 2)), color=INK, lw=1.1)
arrow(PX, 9, PX, 28.2, ls=(0, (3, 2)), lw=1.1)
label(47.0, 9.6, "true label $y$", va="bottom", bbox=dict(fc="white", ec="none", pad=0.4))

# ------------------------------------------------------------------ decision frame
fx0, fx1, fy0, fy1 = 69.0, 94.5, 4.5, 41.5
ax.add_patch(FancyBboxPatch((fx0, fy0), fx1 - fx0, fy1 - fy0,
                            boxstyle="round,pad=0,rounding_size=2.2",
                            fc="none", ec=FRAME, lw=2.6, zorder=1))
ax.text((fx0 + fx1) / 2, fy1 - 1.2, "Threshold Rule", ha="center", va="top",
        fontsize=8.5, fontweight="bold")
arrow(PX + PW / 2 + 0.3, 30, fx0 - 0.3, 30)


def gamma_pdf(x, k, theta):
    out = np.zeros_like(x)
    m = x > 0
    out[m] = np.exp((k - 1) * np.log(x[m]) - x[m] / theta - math.lgamma(k) - k * math.log(theta))
    return out


# mini loss distributions (members = green, non-members = red, as the pills)
lx = np.linspace(0, 3.0, 400)
mem, non = gamma_pdf(lx, 2.2, 0.20), gamma_pdf(lx, 4.0, 0.30)
px0, px1, py0, ph = 71.0, 91.0, 21.0, 12.5
sx = lambda v: px0 + v / 3.0 * (px1 - px0)
sy = lambda v: py0 + v / mem.max() * ph
ax.plot([px0, px1], [py0, py0], color=INK, lw=0.9)
ax.plot(sx(lx), sy(mem), color=PILL_GREEN, lw=1.5)
ax.plot(sx(lx), sy(non), color=PILL_RED, lw=1.5, ls="--")
msk = (lx > 0.3) & (lx < 1.5)
tau = lx[msk][np.argmin(np.abs(mem[msk] - non[msk]))]
ax.plot([sx(tau), sx(tau)], [py0, py0 + ph + 0.6], ls=(0, (1, 1.5)), color="0.3", lw=1.1)
ax.text(sx(tau) + 0.4, py0 + ph + 0.6, "$\\tau$", fontsize=9, va="top", ha="left")
ax.text(px1, py0 - 0.5, "loss", fontsize=7, ha="right", va="top", color="0.3")
ax.text(sx(0.95), sy(1.30), "members", fontsize=7, color=PILL_GREEN, ha="left", va="center")
ax.text(sx(1.55), sy(0.62), "non-members", fontsize=7, color=PILL_RED, ha="left", va="bottom")

for yy, cond, txt, col in ((15.0, "loss $< \\tau$", "membership", PILL_GREEN),
                           (10.6, "loss $\\geq \\tau$", "non-membership", PILL_RED)):
    ax.text(fx0 + 1.3, yy, cond, fontsize=7.5, ha="left", va="center")
    pill(fx0 + 16.6, yy, txt, col, w=16.0, h=3.0, fs=7.3)
ax.text((fx0 + fx1) / 2, 6.6, "sweep $\\tau$ $\\rightarrow$ AUC", fontsize=7.3, ha="center",
        va="center", color="0.3", style="italic")

# ------------------------------------------------------------------ output
OX = 100.0
ax.plot([fx1 + 0.2, OX], [30, 30], color=INK, lw=1.3, ls=(0, (1, 1.5)))
arrow(OX, 30, OX, 34.8, ls=(0, (1, 1.5)))
ax.text(OX, 42.0, "?", fontsize=17, fontweight="bold", ha="center", va="center")
ax.text(OX, 37.4, "Member or\nnot", fontsize=8, fontweight="bold", ha="center",
        va="center", linespacing=1.1)

ax.text((fx0 + fx1) / 2, 2.2, "no shadow models, no attack model", ha="center",
        va="center", fontsize=7.3, color="0.35", style="italic")

fig.savefig("fig_mia_loss_flow.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("fig_mia_loss_flow.png", dpi=220, bbox_inches="tight", pad_inches=0.03)

# ------------------------------------------------------------------------------------------


"""Concept figure: loss-based membership inference and its use for evaluating unlearning.

Schematic only: the curves are idealized gamma densities, not experimental data.
Outputs: fig_mia_loss_concept.pdf (vector, for the paper) and fig_mia_loss_concept.png (preview).
"""
import math

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # editable text in the PDF
    "ps.fonttype": 42,
})

C_MEMBER = "#0072B2"     # members / forget set (colour-blind safe blue)
C_NONMEMBER = "#D55E00"  # non-members (colour-blind safe vermillion)
C_NEUTRAL = "#555555"    # threshold, annotations


def gamma_pdf(x, k, theta):
    """Gamma density; loss is non-negative and right-skewed, so gamma is a natural shape."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    log_pdf = ((k - 1) * np.log(x[pos]) - x[pos] / theta
               - math.lgamma(k) - k * math.log(theta))
    out[pos] = np.exp(log_pdf)
    return out


x = np.linspace(0, 4.0, 2000)
members = gamma_pdf(x, 2.2, 0.20)        # low loss, concentrated near 0
nonmembers = gamma_pdf(x, 4.0, 0.30)     # higher, wider
retrained = gamma_pdf(x, 4.1, 0.295)     # forget set after ideal unlearning ~ non-members
over = gamma_pdf(x, 8.0, 0.33)           # forget set after over-unlearning

# Threshold at the crossing of the two densities (Bayes-optimal for equal priors)
mask = (x > 0.3) & (x < 1.5)
tau = x[mask][np.argmin(np.abs(members[mask] - nonmembers[mask]))]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
ymax = members.max() * 1.45
lead = dict(arrowstyle="-", color=C_NEUTRAL, lw=0.5)

# ---------------------------------------------------------------- panel (a)
ax = ax_a
ax.fill_between(x, np.minimum(members, nonmembers), color="0.85", lw=0, zorder=1)
ax.plot(x, members, color=C_MEMBER, lw=1.6, ls="-", zorder=3)
ax.plot(x, nonmembers, color=C_NONMEMBER, lw=1.6, ls="--", zorder=3)
ax.axvline(tau, color=C_NEUTRAL, lw=1.0, ls=":", zorder=2)
ax.text(tau + 0.05, ymax * 0.97, r"$\tau$", ha="left", va="top", color=C_NEUTRAL, fontsize=10)

ax.text(1.45, ymax * 0.97,
        "Decision rule\n"
        r"loss $<\tau$  $\rightarrow$  “member”" "\n"
        r"loss $\geq\tau$  $\rightarrow$  “non-member”" "\n"
        r"sweeping $\tau$  $\rightarrow$  AUC",
        ha="left", va="top", fontsize=8, color=C_NEUTRAL, linespacing=1.4)

ax.annotate("members\n(training data)", xy=(0.42, float(np.interp(0.42, x, members))),
            xytext=(0.95, 1.30), color=C_MEMBER, fontsize=8, va="center", arrowprops=lead)
ax.annotate("non-members\n(unseen data)", xy=(1.6, float(np.interp(1.6, x, nonmembers))),
            xytext=(2.05, 0.95), color=C_NONMEMBER, fontsize=8, va="center", arrowprops=lead)
ax.annotate("overlap =\nattack errors", xy=(0.80, 0.18), xytext=(2.55, 0.42),
            color=C_NEUTRAL, fontsize=8, va="center", arrowprops=lead)


# ---------------------------------------------------------------- panel (b)
ax = ax_b
ax.fill_between(x, nonmembers, color=C_NONMEMBER, alpha=0.10, lw=0, zorder=1)
h_nm, = ax.plot(x, nonmembers, color=C_NONMEMBER, lw=1.6, ls="--", zorder=2)
h_or, = ax.plot(x, members, color=C_MEMBER, lw=1.6, ls="-", zorder=3)
h_re, = ax.plot(x, retrained, color=C_MEMBER, lw=1.6, ls="-.", zorder=3)
h_ov, = ax.plot(x, over, color=C_MEMBER, lw=1.6, ls=":", zorder=3)

ax.legend([h_or, h_re, h_ov, h_nm],
          ["forget set, original model",
           "forget set, ideal unlearning (≈ retrained)",
           "forget set, over-unlearning",
           "non-members (reference)"],
          loc="upper right", fontsize=7.2, frameon=False, handlelength=2.6)

pk_o = (x[np.argmax(members)], members.max())
pk_r = (x[np.argmax(retrained)], retrained.max())
pk_v = (x[np.argmax(over)], over.max())

ax.text(pk_o[0] + 0.10, pk_o[1], r"AUC $\gg$ 0.5", color=C_MEMBER, fontsize=8, va="center")
ax.text(1.25, pk_r[1] + 0.14, r"AUC $\approx$ 0.5", color=C_MEMBER, fontsize=8, va="center")
ax.text(pk_v[0] + 0.30, pk_v[1] + 0.12, r"AUC $\ll$ 0.5", color=C_MEMBER, fontsize=8,
        ha="left", va="bottom")

ax.annotate("", xy=(pk_r[0] - 0.08, pk_r[1] + 0.10), xytext=(pk_o[0] + 0.22, pk_o[1] * 0.70),
            arrowprops=dict(arrowstyle="->", color=C_NEUTRAL, lw=0.9,
                            connectionstyle="arc3,rad=0.3"))
ax.text(0.95, 1.20, "unlearning", fontsize=8, color=C_NEUTRAL, style="italic")
ax.annotate("", xy=(pk_v[0] - 0.05, pk_v[1] + 0.06), xytext=(1.45, 0.58),
            arrowprops=dict(arrowstyle="->", color=C_NEUTRAL, lw=0.9,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(1.62, 0.70, "too far", fontsize=8, color=C_NEUTRAL, style="italic")

# ---------------------------------------------------------------- shared axes
for ax in (ax_a, ax_b):
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"loss  $\ell(x,y) = -\log p(y \mid x)$")
ax_a.set_ylabel("density")

fig.tight_layout(w_pad=1.5)
fig.savefig("fig_mia_loss_concept.pdf", bbox_inches="tight")
fig.savefig("fig_mia_loss_concept.png", dpi=200, bbox_inches="tight")
print(f"tau = {tau:.3f}")


# ------------------------------------------------------------------------------------------

"""Flow diagram of the confidence-based MIA (Salem et al., 2019, ML-Leaks), drawn in the style of
Figure 1 (MIA - Flow Diagram) of the paper.

Outputs: fig_mia_conf_flow.pdf (vector, for the paper) and fig_mia_conf_flow.png (preview).
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle, Circle, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GREEN_NET = "#1DB66E"     # target model (as in Fig. 1)
FRAME = "#7ED957"         # light green frame (as in Fig. 1)
PILL_RED = "#F2453D"      # non-membership
PILL_GREEN = "#1B5E3B"    # membership
PILL_ORANGE = "#F5A05A"   # score pill (as "Confidence" in Fig. 1)
SEP = "#2B5DA8"           # dotted separator
INK = "black"

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(0, 106)
ax.set_ylim(0, 46)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


# ------------------------------------------------------------------ icons
def arrow(x0, y0, x1, y1, ls="-", lw=1.3, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls,
                                mutation_scale=9, shrinkA=0, shrinkB=0))


def cylinder(cx, cy, w=6.0, h=7.5, lw=1.4):
    top, bot = cy + h / 2, cy - h / 2
    ry = w * 0.17
    ax.add_patch(Rectangle((cx - w / 2, bot), w, h, fc="white", ec="none", zorder=2))
    ax.plot([cx - w / 2, cx - w / 2], [bot, top], color=INK, lw=lw, zorder=3)
    ax.plot([cx + w / 2, cx + w / 2], [bot, top], color=INK, lw=lw, zorder=3)
    t = np.linspace(np.pi, 2 * np.pi, 60)
    for yy in (bot, cy - h / 6, cy + h / 6):  # bottom rim + two bands
        ax.plot(cx + w / 2 * np.cos(t), yy + ry * np.sin(t), color=INK, lw=lw, zorder=3)
    ax.add_patch(Ellipse((cx, top), w, 2 * ry, fc="white", ec=INK, lw=lw, zorder=4))


def network(cx, cy, color, layers=(3, 4, 2), dx=3.0, dy=2.3, r=0.75):
    pos = []
    for i, n in enumerate(layers):
        x = cx + (i - (len(layers) - 1) / 2) * dx
        pos.append([(x, cy + (j - (n - 1) / 2) * dy) for j in range(n)])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.9, zorder=2)
    for layer in pos:
        for p in layer:
            ax.add_patch(Circle(p, r, fc=color, ec=color, zorder=3))


def record(cx, cy, s=3.4):
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc="white", ec=INK, lw=1.2, zorder=3))
    ax.add_patch(Polygon([(cx - s / 2 + 0.4, cy - s / 2 + 0.5), (cx - 0.2, cy + 0.3),
                          (cx + 0.5, cy - 0.4), (cx + 0.9, cy), (cx + s / 2 - 0.4, cy - s / 2 + 0.5)],
                         closed=True, fc="white", ec=INK, lw=1.0, zorder=4))
    ax.add_patch(Circle((cx + 0.7, cy + 0.85), 0.38, fc="white", ec=INK, lw=1.0, zorder=4))


def pill(cx, cy, text, color, w, h=3.2, fs=8.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=color, ec="none", zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=4)


def label(x, y, text, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "top")
    kw.setdefault("fontsize", 8)
    ax.text(x, y, text, linespacing=1.15, **kw)


# ------------------------------------------------------------------ model owner side
SEPX = 36.5
ax.text(18, 45, "Model owner (black-box)", ha="center", va="top", fontsize=8,
        color="0.35", style="italic")

cylinder(7, 30)
label(7, 24.6, "Target\ntraining set")
arrow(10.8, 30, 19.5, 30)
label(15.2, 33.4, "Train", va="bottom")

network(25, 30, GREEN_NET)
label(25, 24.6, "Target\nModel")

record(25, 9)
label(25, 6.2, "Target record\n$x$  (no label)")
arrow(25, 11.2, 25, 20.0)
label(23.6, 15.5, "Query $x$", ha="right", va="center")

# ------------------------------------------------------------------ separator
ax.plot([SEPX, SEPX], [2, 44], ls=(0, (1, 2)), color=SEP, lw=1.6)
ax.text(52, 45, "Adversary", ha="center", va="top", fontsize=8, color="0.35", style="italic")

# ------------------------------------------------------------------ posterior vector
col_x, cell_w, cell_h, n_cells, hi = 41.5, 3.2, 2.0, 6, 0
col_bot = 30 - n_cells * cell_h / 2
arrow(30, 30, col_x - 0.2, 30)
label(SEPX, 31.0, "Inference", va="bottom", bbox=dict(fc="white", ec="none", pad=0.5))
for i in range(n_cells):
    ax.add_patch(Rectangle((col_x, col_bot + i * cell_h), cell_w, cell_h,
                           fc=PILL_ORANGE if i == hi else "white", ec=INK, lw=1.0, zorder=3))
label(col_x + cell_w / 2, col_bot + n_cells * cell_h + 0.6, "Posterior\n$p(\\cdot \\mid x)$", va="bottom")
label(col_x + cell_w / 2, col_bot - 0.6, "$\\mathrm{max}_c\\, p(c \\mid x)$", color="#C0601A")

# ------------------------------------------------------------------ loss score
PX, PW = 56.5, 14.0
arrow(col_x + cell_w + 0.3, 30, PX - PW / 2 - 0.3, 30)
pill(PX, 30, "Confidence", PILL_ORANGE, w=PW, fs=8.2)


# ------------------------------------------------------------------ decision frame
fx0, fx1, fy0, fy1 = 70.0, 96.5, 4.5, 41.5
ax.add_patch(FancyBboxPatch((fx0, fy0), fx1 - fx0, fy1 - fy0,
                            boxstyle="round,pad=0,rounding_size=2.2",
                            fc="none", ec=FRAME, lw=2.6, zorder=1))
ax.text((fx0 + fx1) / 2, fy1 - 1.2, "Threshold Rule", ha="center", va="top",
        fontsize=8.5, fontweight="bold")
arrow(PX + PW / 2 + 0.3, 30, fx0 - 0.3, 30)


def gamma_pdf(x, k, theta):
    out = np.zeros_like(x)
    m = x > 0
    out[m] = np.exp((k - 1) * np.log(x[m]) - x[m] / theta - math.lgamma(k) - k * math.log(theta))
    return out


# mini confidence distributions (members near 1 = green, non-members lower = red)
def beta_pdf(x, a, b):
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    out[m] = np.exp((a - 1) * np.log(x[m]) + (b - 1) * np.log(1 - x[m])
                    + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    return out


cx_ = np.linspace(0, 1, 500)
mem, non = beta_pdf(cx_, 7.0, 1.6), beta_pdf(cx_, 3.0, 2.4)
px0, px1, py0, ph = fx0 + 2.0, fx1 - 2.0, 21.0, 12.5
sx = lambda v: px0 + v * (px1 - px0)
sy = lambda v: py0 + v / mem.max() * ph
ax.plot([px0, px1], [py0, py0], color=INK, lw=0.9)
ax.plot(sx(cx_), sy(non), color=PILL_RED, lw=1.5, ls="--")
ax.plot(sx(cx_), sy(mem), color=PILL_GREEN, lw=1.5)
msk = (cx_ > 0.6) & (cx_ < 0.97)
tau = cx_[msk][np.argmin(np.abs(mem[msk] - non[msk]))]
ax.plot([sx(tau), sx(tau)], [py0, py0 + ph + 0.6], ls=(0, (1, 1.5)), color="0.3", lw=1.1)
ax.text(sx(tau) - 0.4, py0 + ph + 0.6, "$\\tau$", fontsize=9, va="top", ha="right")
ax.text(px0, py0 - 0.5, "0", fontsize=7, ha="center", va="top", color="0.3")
ax.text(px1, py0 - 0.5, "1", fontsize=7, ha="center", va="top", color="0.3")
ax.text((px0 + px1) / 2, py0 - 0.5, "max confidence", fontsize=7, ha="center", va="top", color="0.3")
ax.text(sx(tau) - 0.6, py0 + 0.62 * ph, "members", fontsize=7, color=PILL_GREEN, ha="right", va="center")
ax.text(sx(0.02), py0 + 0.82 * ph, "non-members", fontsize=7, color=PILL_RED, ha="left", va="bottom")

for yy, cond, txt, col in ((15.0, "conf $\\geq \\tau$", "membership", PILL_GREEN),
                           (10.6, "conf $< \\tau$", "non-membership", PILL_RED)):
    ax.text(fx0 + 1.3, yy, cond, fontsize=7.5, ha="left", va="center")
    pill(fx0 + 17.4, yy, txt, col, w=16.0, h=3.0, fs=7.3)
ax.text((fx0 + fx1) / 2, 6.6, "sweep $\\tau$ $\\rightarrow$ AUC", fontsize=7.3, ha="center",
        va="center", color="0.3", style="italic")

# ------------------------------------------------------------------ output
OX = 102.0
ax.plot([fx1 + 0.2, OX], [30, 30], color=INK, lw=1.3, ls=(0, (1, 1.5)))
arrow(OX, 30, OX, 34.8, ls=(0, (1, 1.5)))
ax.text(OX, 42.0, "?", fontsize=17, fontweight="bold", ha="center", va="center")
ax.text(OX, 37.4, "Member or\nnot", fontsize=8, fontweight="bold", ha="center",
        va="center", linespacing=1.1)

ax.text((fx0 + fx1) / 2, 2.2, "no shadow models, no attack model, no label", ha="center",
        va="center", fontsize=7.3, color="0.35", style="italic")

fig.savefig("fig_mia_conf_flow.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("fig_mia_conf_flow.png", dpi=220, bbox_inches="tight", pad_inches=0.03)