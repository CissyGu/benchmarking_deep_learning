import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
C_INPUT = '#E0E0E0'
C_CONV = '#FFF9C4'
C_POOL = '#FFE0B2'
C_LSTM = '#C8E6C9'
C_ATTN = '#F8BBD0'
C_NORM = '#E1BEE7'
C_DENSE = '#BBDEFB'


def draw_box(ax, center, width, height, text, color, subtext=None):
    # Shadow
    shadow = patches.FancyBboxPatch(
        (center[0] - width / 2 + 0.02, center[1] - height / 2 - 0.02), width, height,
        boxstyle="round,pad=0.03", ec="none", fc='#BBBBBB', zorder=1
    )
    ax.add_patch(shadow)
    # Main box
    box = patches.FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2), width, height,
        boxstyle="round,pad=0.03", ec='#333333', fc=color, zorder=2, linewidth=1.2
    )
    ax.add_patch(box)
    # Text - Compact Fonts
    ax.text(center[0], center[1] + (0.12 if subtext else 0), text,
            ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
    if subtext:
        ax.text(center[0], center[1] - 0.15, subtext,
                ha='center', va='center', fontsize=10, color='#333333', zorder=3)


def draw_arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))


def setup_canvas(height=6):  # REDUCED HEIGHT from 10 to 6
    fig, ax = plt.subplots(figsize=(3, height))  # Narrower width to keep aspect ratio sane
    ax.set_xlim(0, 4)
    ax.set_ylim(0, height)
    ax.axis('off')
    return fig, ax


# ==========================================
# 1. CNN (Compact)
# ==========================================
def gen_cnn():
    fig, ax = setup_canvas(6)

    draw_box(ax, (2, 0.6), 2.8, 0.6, "Input", C_INPUT)
    draw_arrow(ax, (2, 0.9), (2, 1.3))

    draw_box(ax, (2, 1.7), 2.8, 0.6, "Conv1D", C_CONV, "64f")
    draw_arrow(ax, (2, 2.0), (2, 2.4))

    draw_box(ax, (2, 2.8), 2.8, 0.6, "MaxPool", C_POOL)
    draw_arrow(ax, (2, 3.1), (2, 3.5))

    draw_box(ax, (2, 3.9), 2.8, 0.6, "Conv1D", C_CONV, "32f")
    draw_arrow(ax, (2, 4.2), (2, 4.6))

    draw_box(ax, (2, 5.2), 2.8, 0.8, "Dense", C_DENSE)

    plt.tight_layout()
    plt.savefig("arch_CNN.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 2. LSTM (Compact)
# ==========================================
def gen_lstm():
    fig, ax = setup_canvas(4.5)  # Even shorter

    draw_box(ax, (2, 0.6), 2.8, 0.6, "Input", C_INPUT)
    draw_arrow(ax, (2, 0.9), (2, 1.4))

    draw_box(ax, (2, 1.9), 2.8, 0.8, "LSTM", C_LSTM, "64 units")
    draw_arrow(ax, (2, 2.3), (2, 2.8))

    draw_box(ax, (2, 3.2), 2.8, 0.6, "Last Step", C_POOL)
    draw_arrow(ax, (2, 3.5), (2, 3.9))

    draw_box(ax, (2, 4.1), 2.8, 0.4, "Dense", C_DENSE)

    plt.tight_layout()
    plt.savefig("arch_LSTM.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. CNN-LSTM (Compact)
# ==========================================
def gen_cnn_lstm():
    fig, ax = setup_canvas(6)

    draw_box(ax, (2, 0.6), 2.8, 0.6, "Input", C_INPUT)
    draw_arrow(ax, (2, 0.9), (2, 1.3))

    draw_box(ax, (2, 1.7), 2.8, 0.6, "Conv1D", C_CONV)
    draw_arrow(ax, (2, 2.0), (2, 2.4))

    draw_box(ax, (2, 2.8), 2.8, 0.6, "MaxPool", C_POOL)
    draw_arrow(ax, (2, 3.1), (2, 3.5))

    draw_box(ax, (2, 4.0), 2.8, 0.8, "LSTM", C_LSTM)
    draw_arrow(ax, (2, 4.4), (2, 5.0))

    draw_box(ax, (2, 5.4), 2.8, 0.6, "Dense", C_DENSE)

    plt.tight_layout()
    plt.savefig("arch_CNNLSTM.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 4. Transformer (Compact)
# ==========================================
def gen_transformer():
    fig, ax = setup_canvas(6.5)

    draw_box(ax, (2, 0.6), 2.8, 0.6, "Input", C_INPUT)
    draw_arrow(ax, (2, 0.9), (2, 1.3))

    draw_box(ax, (2, 1.7), 2.8, 0.6, "ConvProj", C_CONV)
    draw_arrow(ax, (2, 2.0), (2, 2.4))

    draw_box(ax, (2, 3.0), 2.8, 0.8, "Multi-Head\nAttention", C_ATTN)
    draw_arrow(ax, (2, 3.4), (2, 3.8))

    draw_box(ax, (2, 4.2), 2.8, 0.6, "Add&Norm", C_NORM)
    draw_arrow(ax, (2, 4.5), (2, 4.9))

    draw_box(ax, (2, 5.3), 2.8, 0.6, "GlobalPool", C_POOL)
    draw_arrow(ax, (2, 5.6), (2, 6.0))

    draw_box(ax, (2, 6.2), 2.8, 0.4, "Dense", C_DENSE)

    plt.tight_layout()
    plt.savefig("arch_Transformer.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 5. LSTM-Transformer (Compact)
# ==========================================
def gen_lstm_trans():
    fig, ax = setup_canvas(7)

    draw_box(ax, (2, 0.6), 2.8, 0.6, "Input", C_INPUT)
    draw_arrow(ax, (2, 0.9), (2, 1.3))

    draw_box(ax, (2, 1.8), 2.8, 0.8, "LSTM", C_LSTM)
    draw_arrow(ax, (2, 2.2), (2, 2.6))

    draw_box(ax, (2, 3.2), 2.8, 0.8, "Attention", C_ATTN)
    draw_arrow(ax, (2, 3.6), (2, 4.0))

    draw_box(ax, (2, 4.4), 2.8, 0.6, "Add&Norm", C_NORM)
    draw_arrow(ax, (2, 4.7), (2, 5.1))

    # Res connection
    draw_box(ax, (2, 5.5), 2.8, 0.6, "Res+Proj", C_CONV)
    ax.text(3.6, 5.5, "+", fontsize=12, fontweight='bold', color='#555555')
    draw_arrow(ax, (2, 5.8), (2, 6.2))

    draw_box(ax, (2, 6.6), 2.8, 0.6, "Dense", C_DENSE)

    plt.tight_layout()
    plt.savefig("arch_LSTMTrans.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 6. Tri-Hybrid (Compact)
# ==========================================
def gen_tri_hybrid():
    fig, ax = setup_canvas(7)

    draw_box(ax, (2, 0.6), 2.8, 0.6, "Input", C_INPUT)
    draw_arrow(ax, (2, 0.9), (2, 1.3))

    # 3 Branches
    draw_box(ax, (2, 2.5), 3.8, 1.8, "", '#EEEEEE')  # Background container
    ax.text(2, 3.2, "Parallel Branches", fontsize=7, color='#999999')

    draw_box(ax, (0.8, 2.2), 1.0, 0.6, "Conv", C_CONV)
    draw_box(ax, (2.0, 2.2), 1.0, 0.6, "LSTM", C_LSTM)
    draw_box(ax, (3.2, 2.2), 1.0, 0.6, "Attn", C_ATTN)

    # Arrows to branches
    ax.annotate("", xy=(0.8, 1.9), xytext=(2, 1.3), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(2.0, 1.9), xytext=(2, 1.3), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(3.2, 1.9), xytext=(2, 1.3), arrowprops=dict(arrowstyle="->", lw=1))

    # Arrows from branches
    ax.annotate("", xy=(2, 4.0), xytext=(0.8, 2.5), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(2, 4.0), xytext=(2.0, 2.5), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(2, 4.0), xytext=(3.2, 2.5), arrowprops=dict(arrowstyle="->", lw=1))

    draw_box(ax, (2, 4.3), 2.8, 0.6, "Concat", C_POOL)
    draw_arrow(ax, (2, 4.6), (2, 5.0))

    draw_box(ax, (2, 5.3), 2.8, 0.6, "GlobalPool", C_NORM)
    draw_arrow(ax, (2, 5.6), (2, 6.0))

    draw_box(ax, (2, 6.4), 2.8, 0.6, "Dense", C_DENSE)

    plt.tight_layout()
    plt.savefig("arch_TriHybrid.png", dpi=300, bbox_inches='tight')
    plt.close()


gen_cnn()
gen_lstm()
gen_cnn_lstm()
gen_transformer()
gen_lstm_trans()
gen_tri_hybrid()
print("Generated 6 COMPACT architecture diagrams.")