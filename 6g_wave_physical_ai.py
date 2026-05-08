# ============================================================
# WAVE-NATIVE PHYSICAL AI
# Graphene Axon + hBN Synapse
# FULL WORKING VERSION
# ============================================================

# Includes:
# - 2D THz interference field
# - Serpentine graphene geometry
# - hBN synaptic phase memory
# - Polariton cavity resonance
# - Adaptive learning
# - Hysteresis memory
# - Solstice/equinox drift
# - Multi-layer wave propagation
# - XOR wave logic
# - Animated GIF generation
# - Final static image save
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# ============================================================
# PARAMETERS
# ============================================================

c = 3e8
frequency = 1e12                  # 1 THz

wavelength = c / frequency

k = 2 * np.pi / wavelength

# ============================================================
# GRAPHENE AXON CHANNELS
# ============================================================

N = 28

# ============================================================
# SPATIAL DOMAIN
# ============================================================

L = 25e-6

grid_size = 300

x = np.linspace(-L, L, grid_size)
y = np.linspace(-L, L, grid_size)

X, Y = np.meshgrid(x, y)

# ============================================================
# SERPENTINE GRAPHENE GEOMETRY
# ============================================================

spacing = 1.5e-6

x_src = np.linspace(-N/2, N/2, N) * spacing

serp_amp = 4e-6

y_src = serp_amp * np.sin(
    np.linspace(0, 6*np.pi, N)
)

# ============================================================
# 23.5 DEGREE TILT
# ============================================================

tilt = np.radians(23.5)

x_rot = (
    x_src*np.cos(tilt)
    - y_src*np.sin(tilt)
)

y_rot = (
    x_src*np.sin(tilt)
    + y_src*np.cos(tilt)
)

# ============================================================
# INITIAL PHASE MEMORY
# ============================================================

phi = np.linspace(0, np.pi, N)

# ============================================================
# LEARNING PARAMETERS
# ============================================================

eta = 0.08

gamma = 0.015

beta = 0.12

memory = np.zeros(N)

# ============================================================
# MULTI-LAYER NETWORK
# ============================================================

layers = 3

layer_shift = 2.5e-6

# ============================================================
# GLOBAL FINAL FRAME STORAGE
# ============================================================

last_attention = None

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9,8))

# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame):

    global phi
    global memory
    global last_attention

    ax.clear()

    # ========================================================
    # FIELD INITIALIZATION
    # ========================================================

    field = np.zeros_like(X, dtype=complex)

    # ========================================================
    # SOLSTICE / EQUINOX DRIFT
    # ========================================================

    seasonal_phase = 0.9 * np.sin(
        2*np.pi*frame/80
    )

    # ========================================================
    # MULTI-LAYER PROPAGATION
    # ========================================================

    for layer in range(layers):

        layer_phase = layer * 0.8

        for n in range(N):

            x_n = x_rot[n]

            y_n = y_rot[n] + layer*layer_shift

            # =================================================
            # DISTANCE FIELD
            # =================================================

            r = np.sqrt(
                (X - x_n)**2 +
                (Y - y_n)**2
            )

            # =================================================
            # POLARITON CAVITY RESONANCE
            # =================================================

            cavity = np.cos(
                2*np.pi*r/(2*wavelength)
            )

            # =================================================
            # AUTONOMOUS OSCILLATION
            # =================================================

            oscillation = 0.5 * np.sin(
                0.15*frame + n*0.3
            )

            # =================================================
            # XOR WAVE LOGIC
            # =================================================

            A = (frame // 25) % 2

            B = (frame // 40) % 2

            xor_phase = 0

            if A == 1 and B == 1:

                xor_phase = np.pi

            # =================================================
            # TOTAL PHASE
            # =================================================

            total_phase = (
                phi[n]
                + seasonal_phase
                + oscillation
                + layer_phase
                + xor_phase
            )

            # =================================================
            # FIELD CONTRIBUTION
            # =================================================

            contribution = cavity * np.exp(
                1j * (k*r + total_phase)
            )

            field += contribution

    # ========================================================
    # ATTENTION FIELD
    # ========================================================

    intensity = np.abs(field)**2

    intensity /= np.max(intensity)

    attention = intensity**1.7

    # ========================================================
    # SAVE LAST FRAME
    # ========================================================

    last_attention = attention.copy()

    # ========================================================
    # HYSTERESIS MEMORY
    # ========================================================

    mean_attention = np.mean(attention)

    memory = (
        (1-beta)*memory
        + beta*mean_attention
    )

    # ========================================================
    # LOCAL ATTENTION LEARNING
    # ========================================================

    local_attention = np.zeros(N)

    for n in range(N):

        idx_x = np.argmin(
            np.abs(x - x_rot[n])
        )

        idx_y = np.argmin(
            np.abs(y - y_rot[n])
        )

        local_attention[n] = attention[
            idx_y,
            idx_x
        ]

    # ========================================================
    # PHASE LEARNING
    # ========================================================

    noise = 0.02 * np.random.randn(N)

    phi += (
        eta*local_attention
        - gamma*phi
        + noise
    )

    # ========================================================
    # PLOT ATTENTION FIELD
    # ========================================================

    im = ax.imshow(

        attention,

        extent=[
            x.min()*1e6,
            x.max()*1e6,
            y.min()*1e6,
            y.max()*1e6
        ],

        origin='lower',

        cmap='inferno',

        animated=True
    )

    # ========================================================
    # GRAPHENE AXON LOCATIONS
    # ========================================================

    ax.scatter(

        x_rot*1e6,
        y_rot*1e6,

        color='cyan',

        s=20
    )

    # ========================================================
    # TITLES
    # ========================================================

    ax.set_title(
        'Wave-Native Physical AI\n'
        f'Time = {frame}     '
        f'Memory = {memory.mean():.3f}'
    )

    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')

    return [im]

# ============================================================
# CREATE ANIMATION
# ============================================================

ani = FuncAnimation(

    fig,

    update,

    frames=120,

    interval=80,

    blit=False
)

# ============================================================
# SAVE GIF
# ============================================================

print("Saving GIF...")

ani.save(

    "wave_physical_ai.gif",

    writer=PillowWriter(fps=15)
)

print("GIF saved.")

# ============================================================
# FINAL STATIC IMAGE
# ============================================================

plt.figure(figsize=(10,8))

plt.imshow(

    last_attention,

    extent=[
        x.min()*1e6,
        x.max()*1e6,
        y.min()*1e6,
        y.max()*1e6
    ],

    origin='lower',

    cmap='inferno'
)

# ============================================================
# SOURCE LOCATIONS
# ============================================================

plt.scatter(

    x_rot*1e6,
    y_rot*1e6,

    color='cyan',

    s=25
)

# ============================================================
# LABELS
# ============================================================

plt.title(
    'Final Wave-Native Physical AI State'
)

plt.xlabel('x (um)')
plt.ylabel('y (um)')

plt.colorbar(
    label='Attention Intensity'
)

plt.tight_layout()

# ============================================================
# SAVE MP4
# ============================================================

print("Saving MP4...")

ani.save(
    "wave_physical_ai.mp4",
    writer="ffmpeg",
    fps=20
)

print("MP4 saved.")
