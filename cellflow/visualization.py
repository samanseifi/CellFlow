"""Rendering helpers: per-step frames and GIF assembly.

Kept separate from the solver so simulations can run headless. Set the
matplotlib backend to "Agg" before importing pyplot for batch/cron runs.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import imageio


def render_frame(nutrient_field, cells, physical_size, step, output_dir):
    """Render one frame (nutrient field + cells) to a PNG and return its path.

    Cells are drawn from their shape_axes() (an area-conserving ellipse); with
    cell-shape mechanics off this is exactly a circle, so default output is
    unchanged."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(nutrient_field, cmap='viridis', origin='lower',
              extent=[0, physical_size, 0, physical_size])

    for cell in cells:
        a, b, angle = cell.shape_axes()
        # Inscribe the ellipse within the collision radius (scale so the long
        # axis = radius) so a deformed cell never renders beyond where the
        # circular mechanics actually keep neighbours apart.
        scale = cell.radius / max(a, b)
        a, b = a * scale, b * scale
        color = 'red' if cell.phase == 'DIVISION' else 'white'
        ax.add_patch(Ellipse(cell.position, 2 * a, 2 * b, angle=np.degrees(angle),
                             color=color, alpha=0.75, ec='black', lw=0.6))

    ax.set_title(f'Step: {step}, Cells: {len(cells)}')
    ax.set_xlim(0, physical_size)
    ax.set_ylim(0, physical_size)

    filepath = os.path.join(output_dir, f'frame_{step:04d}.png')
    plt.savefig(filepath)
    plt.close(fig)
    return filepath


def create_gif(frames, config_name):
    """Stitch saved PNG frames into a GIF, then delete the intermediate PNGs."""
    with imageio.get_writer(f"{config_name}_simulation.gif", mode='I', duration=0.1) as writer:
        for filename in frames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in frames:
        os.remove(filename)
