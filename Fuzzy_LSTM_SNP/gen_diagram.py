#!/usr/bin/env python3
"""
Generate LSTM-SNP with Fuzzy Feature Augmentation diagram
Matching the paper's black-and-white schematic style.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 8), dpi=200)
ax.set_xlim(-2, 22)
ax.set_ylim(-4.5, 6)
ax.set_aspect('equal')
ax.axis('off')

# Colors
WHITE = '#ffffff'
GRAY = '#c0c0c0'
BG = '#ffffff'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

def draw_circle(ax, cx, cy, r, label, filled=False, fontsize=10):
    """Draw a gate circle (white) or operation circle (gray)."""
    color = GRAY if filled else WHITE
    circ = plt.Circle((cx, cy), r, facecolor=color, edgecolor='black', linewidth=1.2, zorder=5)
    ax.add_patch(circ)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold' if not filled else 'normal', zorder=6)
    return (cx, cy)

def draw_box(ax, cx, cy, w, h, label, fontsize=9):
    """Draw a rectangular box node."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h, 
                           boxstyle="round,pad=0.05", 
                           facecolor=WHITE, edgecolor='black', linewidth=1.2, zorder=5)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize, zorder=6)

def arrow(ax, x1, y1, x2, y2, **kwargs):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2, 
                                shrinkA=0, shrinkB=0, **kwargs),
                zorder=4)

def line(ax, x1, y1, x2, y2, **kwargs):
    """Draw a line (no arrowhead)."""
    ls = kwargs.pop('linestyle', '-')
    lw = kwargs.pop('lw', 1.2)
    ax.plot([x1, x2], [y1, y2], color='black', linewidth=lw, linestyle=ls, zorder=3, **kwargs)

def draw_fuzzy_box(ax, cx, cy, w=1.6, h=1.0):
    """Draw the fuzzy feature augmentation box with a Gaussian curve and sigma."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.08",
                           facecolor=WHITE, edgecolor='black', linewidth=1.2, zorder=5)
    ax.add_patch(rect)
    
    # Draw Gaussian curve
    xs = np.linspace(-1.8, 1.8, 80)
    ys = np.exp(-xs**2 / (2 * 0.5**2))
    # Scale and position
    scale_x = w * 0.25
    scale_y = h * 0.3
    gx = cx + xs * scale_x
    gy = cy + 0.05 + ys * scale_y
    ax.plot(gx, gy, color='black', linewidth=1.5, zorder=6)
    
    # Sigma label below the curve
    ax.text(cx, cy - h * 0.32, r'$\sigma$', ha='center', va='center', fontsize=11, 
            fontstyle='italic', zorder=6)

def draw_cell(ax, ox, oy, t_label, show_u_prev=True):
    """
    Draw one LSTM-SNP cell with fuzzy feature augmentation at input.
    ox, oy: origin (bottom-left of the cell box)
    t_label: time step label string ("1", "2", "T")
    """
    R = 0.3  # gate radius
    
    # Cell box dimensions
    bw, bh = 4.0, 4.5
    bcx, bcy = ox + bw/2, oy + bh/2
    
    # Draw cell bounding box
    rect = FancyBboxPatch((ox, oy), bw, bh,
                           boxstyle="round,pad=0.15",
                           facecolor=WHITE, edgecolor='black', linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    
    # Gate positions (relative to cell origin)
    # Bottom row: r gate, × (multiply)
    r_x, r_y = ox + 0.8, oy + 0.7
    rmul_x, rmul_y = ox + 2.0, oy + 0.7
    
    # Middle row: subtraction, u(t)
    sub_x, sub_y = ox + 2.0, oy + 1.6
    u_x, u_y = ox + 3.2, oy + 1.6
    
    # Upper-middle row: c gate, × (multiply), f gate
    c_x, c_y = ox + 0.8, oy + 2.5
    cmul_x, cmul_y = ox + 2.0, oy + 2.5
    f_x, f_y = ox + 3.2, oy + 2.5
    
    # Top row: o gate, × (multiply)
    o_x, o_y = ox + 0.8, oy + 3.6
    omul_x, omul_y = ox + 3.2, oy + 3.6
    
    # Draw gates
    draw_circle(ax, r_x, r_y, R, r'$r$', filled=False)
    draw_circle(ax, rmul_x, rmul_y, R, r'$\times$', filled=True)
    draw_circle(ax, sub_x, sub_y, R, r'$-$', filled=False)
    draw_circle(ax, c_x, c_y, R, r'$c$', filled=False)
    draw_circle(ax, cmul_x, cmul_y, R, r'$\times$', filled=True)
    draw_circle(ax, f_x, f_y, R, r'$f$', filled=False)
    draw_circle(ax, o_x, o_y, R, r'$o$', filled=False)
    draw_circle(ax, omul_x, omul_y, R, r'$\times$', filled=True)
    
    # u(t) box
    draw_box(ax, u_x, u_y, 0.7, 0.4, f'$u({t_label})$', fontsize=8)
    
    # --- Internal connections ---
    # r -> rmul
    arrow(ax, r_x + R, r_y, rmul_x - R, rmul_y)
    # rmul -> sub (up)
    arrow(ax, rmul_x, rmul_y + R, sub_x, sub_y - R)
    # sub -> u(t) (right)
    arrow(ax, sub_x + R, sub_y, u_x - 0.35, u_y)
    # sub -> cmul (up)
    arrow(ax, sub_x, sub_y + R, cmul_x, cmul_y - R)
    # c -> cmul (right)
    arrow(ax, c_x + R, c_y, cmul_x - R, cmul_y)
    # cmul -> f (right)
    arrow(ax, cmul_x + R, cmul_y, f_x - R, f_y)
    # f -> omul (up)
    arrow(ax, f_x, f_y + R, omul_x, omul_y - R)
    # o -> right then up to omul
    arrow(ax, o_x + R, o_y, omul_x - R, omul_y)
    
    # --- Left vertical input bus ---
    input_bus_x = ox + 0.15
    line(ax, input_bus_x, oy - 0.1, input_bus_x, oy + 3.6)
    # Taps from bus to gates
    arrow(ax, input_bus_x, r_y, r_x - R, r_y)
    arrow(ax, input_bus_x, c_y, c_x - R, c_y)
    arrow(ax, input_bus_x, o_y, o_x - R, o_y)
    
    # --- Right vertical input (x(t) direct path feeding rmul bottom) ---
    right_bus_x = ox + 3.8
    line(ax, right_bus_x, oy - 0.1, right_bus_x, oy + 3.6)
    # Tap to rmul from right
    arrow(ax, right_bus_x, rmul_y, rmul_x + R, rmul_y)
    # Tap to f from right
    arrow(ax, right_bus_x, f_y, f_x + R, f_y)
    # Tap to omul from right  
    arrow(ax, right_bus_x, omul_y, omul_x + R, omul_y)
    
    # --- Output h(t) going up ---
    h_top_y = oy + bh + 0.7
    arrow(ax, omul_x, omul_y + R, omul_x, h_top_y)
    ax.text(omul_x + 0.15, h_top_y + 0.15, f'$h({t_label})$', ha='center', va='bottom', fontsize=11)
    
    # --- Recurrent connection: h(t) wraps around left side ---
    # Goes up from omul, left over the top, down the left side
    line(ax, omul_x, h_top_y, ox - 0.5, h_top_y)
    line(ax, ox - 0.5, h_top_y, ox - 0.5, oy - 0.8)
    arrow(ax, ox - 0.5, oy - 0.8, input_bus_x, oy - 0.1)
    
    # --- FUZZY FEATURE AUGMENTATION BLOCK ---
    fuz_cx = ox + bw / 2
    fuz_cy = oy - 2.2
    draw_fuzzy_box(ax, fuz_cx, fuz_cy)
    
    # --- Input x(t) from bottom ---
    x_bottom_y = oy - 3.7
    ax.text(ox + bw / 2, x_bottom_y - 0.15, f'$x({t_label})$', ha='center', va='top', fontsize=11)
    
    # x(t) arrow up to fuzzy box
    arrow(ax, fuz_cx, x_bottom_y + 0.15, fuz_cx, fuz_cy - 0.5)
    
    # Fuzzy box output -> split into two paths
    # Left path: augmented features -> left bus (into cell)
    arrow(ax, fuz_cx - 0.4, fuz_cy + 0.5, input_bus_x, oy - 0.1)
    
    # Right path: original x(t) also goes up the right bus 
    arrow(ax, fuz_cx + 0.4, fuz_cy + 0.5, right_bus_x, oy - 0.1)
    
    # Label: "Fuzzy Feature" near the box
    ax.text(fuz_cx, fuz_cy + 0.7, 'Fuzzy\nFeature', ha='center', va='bottom', fontsize=7,
            fontstyle='italic', color='gray')
    
    # --- u(t-1) dashed line coming from left ---
    if show_u_prev:
        if t_label == "1":
            u_prev_label = "$u(0)$"
        elif t_label == "2":
            u_prev_label = "$u(1)$"
        else:
            u_prev_label = f"$u({t_label}\\!-\\!1)$"
        ax.text(ox - 1.2, oy + 1.6, u_prev_label, ha='center', va='center', fontsize=9)
        line(ax, ox - 0.7, oy + 1.6, ox + 0.15, oy + 1.6, linestyle='--')
    
    # --- Time label below ---
    ax.text(ox + bw/2, x_bottom_y - 0.6, f'$t={t_label}$', ha='center', va='top', fontsize=11,
            fontstyle='italic')
    
    return ox + bw

# ============================================================
# Draw three cells
# ============================================================

# Cell t=1
x_end1 = draw_cell(ax, 0, 0, "1", show_u_prev=True)

# Dots between cell 1 and cell 2
ax.text(x_end1 + 0.7, 2.2, r'$\cdots\cdots$', ha='center', va='center', fontsize=16)

# Cell t=2
x_end2 = draw_cell(ax, x_end1 + 1.4, 0, "2", show_u_prev=True)

# Dots between cell 2 and cell T
ax.text(x_end2 + 0.7, 2.2, r'$\cdots\cdots$', ha='center', va='center', fontsize=16)

# Cell t=T
draw_cell(ax, x_end2 + 1.4, 0, "T", show_u_prev=True)

plt.tight_layout()
plt.savefig('/Users/satabarto/Research/Fuzzy_LSTM_SNP/fuzzy_feature_augmentation_diagram.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/satabarto/Research/Fuzzy_LSTM_SNP/fuzzy_feature_augmentation_diagram.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved PNG and PDF")
