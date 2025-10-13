#!/usr/bin/env python3
"""
perlin_with_scipy_display.py

Generates Perlin-based fBm like your original script, thresholds to capacity,
composes RGB, upscales using scipy.ndimage.zoom, and displays using scipy
(if available) or matplotlib as a fallback.
"""
import os
from math import floor
from random import Random
import numpy as np
from scipy.ndimage import binary_opening
from src.config import ROOT_DIR

# SciPy imports for resizing and optional display
import scipy.ndimage as ndi
try:
    import scipy.misc as smisc  # for legacy imshow if present
except Exception:
    smisc = None

import matplotlib.pyplot as plt

# ---------- Perlin implementation (2D) ----------
class Perlin2D:
    def __init__(self, seed=0):
        rng = Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        self.perm = p + p
        self.grad2 = (
            (1,1), (-1,1), (1,-1), (-1,-1),
            (1,0), (-1,0), (0,1), (0,-1)
        )

    @staticmethod
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(self, hash_val, x, y):
        h = hash_val & 7
        gx, gy = self.grad2[h]
        return gx * x + gy * y

    def perlin(self, x, y):
        xi = int(floor(x)) & 255
        yi = int(floor(y)) & 255
        xf = x - floor(x)
        yf = y - floor(y)

        u = self.fade(xf)
        v = self.fade(yf)

        p = self.perm
        aa = p[p[xi    ] + yi    ]
        ab = p[p[xi    ] + yi + 1]
        ba = p[p[xi + 1] + yi    ]
        bb = p[p[xi + 1] + yi + 1]

        x1 = self.lerp(self.grad(aa, xf    , yf    ),
                       self.grad(ba, xf - 1, yf    ), u)
        x2 = self.lerp(self.grad(ab, xf    , yf - 1),
                       self.grad(bb, xf - 1, yf - 1), u)
        return self.lerp(x1, x2, v)

# ---------- fBm ----------
def fbm_noise2(perlin_obj, x, y, octaves=6, persistence=0.5, lacunarity=2.0):
    amplitude = 1.0
    frequency = 1.0
    value = 0.0
    max_ampl = 0.0
    for _ in range(octaves):
        value += perlin_obj.perlin(x * frequency, y * frequency) * amplitude
        max_ampl += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return value / max_ampl

# ---------- Create tiles (returns capacity and rgb as numpy arrays) ----------
def create_tiles_and_rgb(n_rows, n_cols,
                         scale=5.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=0):
    perlin = Perlin2D(seed=seed)
    pnoise = np.zeros((n_rows, n_cols), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            x = i / scale
            y = j / scale
            pnoise[i, j] = fbm_noise2(perlin, x, y, octaves=octaves,
                                      persistence=persistence, lacunarity=lacunarity)

    is_soil = np.zeros_like(pnoise, dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            v = pnoise[i, j]
            if  v < -0.18:#-0.125:
                is_soil[i, j] = True
                
    # Apply binary opening with periodic (wrap-around) boundaries
    # Pad the array by wrapping edges, apply opening, then crop back
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    pad_width = 1  # padding size based on structure size
    
    # Create periodic padding by wrapping edges
    padded = np.pad(is_soil, pad_width, mode='wrap')
    
    # Apply binary opening to padded array
    opened_padded = binary_opening(padded, structure=structure)
    
    # Crop back to original size
    is_soild = opened_padded[pad_width:-pad_width, pad_width:-pad_width]

    # compose rgb (0..1 floats)
    soil = np.array((120., 72, 0.)) / 255.0
    grass = np.array((85., 168, 74.)) / 255.0

    rgb = soil * is_soild[..., None] + (1 - is_soild[..., None]) * grass  # float in [0,1]

    return is_soild, rgb

# ---------- Upscale using scipy.ndimage.zoom and display ----------
def upscale_and_display(rgb, target_size=(1024, 1024)):
    # rgb shape (H, W, 3) float [0,1] -> convert to uint8
    small_h, small_w = rgb.shape[:2]
    out_h, out_w = target_size

    # compute zoom factors per axis: (zoom_y, zoom_x, zoom_channels)
    zoom_y = out_h / small_h
    zoom_x = out_w / small_w

    # Use order=0 for nearest neighbour (to mimic scipy.misc.imresize(..., interp='nearest'))
    # We do not zoom the channels axis (keep 1)
    zoom_factors = (zoom_y, zoom_x, 1.0)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # scipy.ndimage.zoom wants float or numeric array; we can zoom uint8 too.
    up = ndi.zoom(rgb_uint8, zoom_factors, order=0)

    # Try to display with scipy.misc.imshow if available; it's often not.
    if smisc is not None and hasattr(smisc, 'imshow'):
        try:
            smisc.imshow(up)  # legacy API
            return
        except Exception:
            pass

    # Fallback: matplotlib
    plt.figure(figsize=(6,6))
    plt.imshow(up)
    plt.axis('off')
    plt.show()

# ---------- Main ----------
if __name__ == '__main__':
    n_rows = 96
    n_cols = 96
    max_capacity = 5
    seed = 2

    is_soil, rgb = create_tiles_and_rgb(n_rows, n_cols,
                                         scale=15.0, octaves=4, persistence=0.5, lacunarity=20.0, seed=seed)
    print(f'{is_soil.sum()=}')
    with open(os.path.join(ROOT_DIR, f'resources/is_soil_{n_rows}_{n_cols}.bin'), 'wb') as f:
        f.write(is_soil.tobytes())

    nonzero = ~is_soil
    print('is soil:', is_soil.sum())
    print('Max capacity:', nonzero * max_capacity)
    print('Sustained capacity:', nonzero * 0.15)  # growth_rate 0.2 as before

    # display using scipy (zoom) + fallback
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    # upscale_and_display(rgb, target_size=(128, 128))
