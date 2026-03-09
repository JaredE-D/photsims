# SPSq/field_gif.py
# Low-res simulation to generate a GIF of Hz field evolution in a microring resonator.
# Usage: python3 -u field_gif.py

from pathlib import Path
import sys
import numpy as np
import meep as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Parameters (2.5D EIM) ────────────────────────────────────────────────
ring_w = 0.450
wg_w   = 0.450
n_core = 2.8217
n_clad = 1.44
pml    = 1.5
pad    = 2.0
resolution = 30

# m=140 candidate
R   = 24.7751
gap = 0.2853
lambda0 = 1.55
fcen = 1.0 / lambda0
fwidth = 0.1 * fcen

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def build_geometry():
    core = mp.Medium(index=n_core)
    clad = mp.Medium(index=n_clad)
    outer = mp.Cylinder(radius=R + ring_w / 2, material=core)
    inner = mp.Cylinder(radius=R - ring_w / 2, material=clad)
    bus_y = R + ring_w / 2 + gap + wg_w / 2
    bus_len = 2 * (R + pad + pml)
    bus = mp.Block(size=mp.Vector3(bus_len, wg_w, mp.inf),
                   center=mp.Vector3(0, bus_y), material=core)
    return [outer, inner, bus], bus_y


def main():
    geometry, bus_y = build_geometry()

    sx = 2 * (R + pad + pml)
    sy = 2 * (R + ring_w / 2 + gap + wg_w / 2 + pad + pml)
    cell = mp.Vector3(sx, sy)

    # EigenModeSource on bus waveguide for clean excitation
    src_x = -sx / 2 + pml+ 1.5
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=mp.Vector3(src_x, bus_y),
            size=mp.Vector3(0, 3 * wg_w),
            eig_band=1,
            eig_match_freq=True,
            direction=mp.X,
            eig_parity=mp.EVEN_Z,
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=[mp.PML(pml)],
        geometry=geometry,
        sources=sources,
        default_material=mp.Medium(index=n_clad),
    )

    # Save frames to disk to avoid OOM
    frame_dir = OUT / "_frames_tmp"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = 5.0  # MEEP time units between frames
    total_time = 1125.0    # total sim time
    n_frames = int(total_time / frame_interval)

    print(f"Simulating R={R} µm, gap={gap} µm, res={resolution}")
    print(f"Cell: {sx:.1f} × {sy:.1f} µm, frames: {n_frames}")
    sys.stdout.flush()

    # Track global max for color scaling
    global_absmax = 0.0

    for i in range(n_frames):
        sim.run(until=frame_interval)
        hz = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)
        hz = hz.copy().T  # transpose for correct orientation
        np.save(frame_dir / f"frame_{i:04d}.npy", hz)
        nonzero = np.abs(hz[hz != 0])
        if len(nonzero) > 0:
            global_absmax = max(global_absmax, np.percentile(nonzero, 100))
        del hz
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames} (t={sim.meep_time():.1f})")
            sys.stdout.flush()

    # Free simulation memory before building GIF
    sim.reset_meep()
    del sim
    import gc; gc.collect()
    import matplotlib.colors as mcolors
    # Build GIF from saved frames
    print(f"\nBuilding GIF from {n_frames} frames...")
    sys.stdout.flush()

    vmax1 = global_absmax if global_absmax > 0 else 1.0
    gif_path = OUT / f"field_Hz_R{R}_gap{gap}_res{resolution}.gif"

    # Render and append frames one at a time to avoid holding all PIL images
    first_image = None
    frame_paths = []
    norm1 = mcolors.SymLogNorm(linthresh=vmax1/25,vmin=-vmax1,vmax=vmax1)

    for i in range(n_frames):
        hz = np.load(frame_dir / f"frame_{i:04d}.npy")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            hz, origin="lower", cmap="RdBu_r",
            norm=norm1,
            extent=[-sx/2, sx/2, -sy/2, sy/2],
        )
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        # ax.set_xlabel("x [µm]")
        # ax.set_ylabel("y [µm]")

        # Draw ring outline
        theta = np.linspace(0, 2*np.pi, 200)
        for r_ring in [R - ring_w/2, R + ring_w/2]:
            ax.plot(r_ring * np.cos(theta), r_ring * np.sin(theta),
                    'k-', linewidth=0.5, alpha=0.3)
        # Bus waveguide edges
        ax.axhline(bus_y - wg_w/2, color='k', linewidth=0.5, alpha=0.3,
                    xmin=0.05, xmax=0.95)
        ax.axhline(bus_y + wg_w/2, color='k', linewidth=0.5, alpha=0.3,
                    xmin=0.05, xmax=0.95)

        fig.tight_layout()
        png_path = frame_dir / f"render_{i:04d}.png"
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(png_path)
        del hz

        if (i + 1) % 50 == 0:
            print(f"  Rendered {i+1}/{n_frames} frames")
            sys.stdout.flush()

    # Assemble GIF by streaming frames (no need to hold all in memory)
    print("Assembling GIF...")
    sys.stdout.flush()
    first_image = Image.open(frame_paths[0]).copy()
    append_frames = []
    for p in frame_paths[1:]:
        append_frames.append(Image.open(p).copy())

    first_image.save(gif_path, save_all=True, append_images=append_frames,
                     duration=40, loop=0)
    del first_image, append_frames

    # Clean up temporary files
    for p in frame_paths:
        p.unlink(missing_ok=True)
    for p in frame_dir.glob("frame_*.npy"):
        p.unlink(missing_ok=True)
    frame_dir.rmdir()

    print(f"[OK] GIF saved → {gif_path}")
    print(f"     {n_frames} frames, {total_time:.0f} MEEP time units")


if __name__ == "__main__":
    main()
