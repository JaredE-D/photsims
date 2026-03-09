# SPSq/run_transmission_clean.py
# Clean transmission spectra using EigenModeSource for proper bus waveguide
# excitation. Two-pass normalization (reference run without ring, then with ring)
# to eliminate Fabry-Perot artifacts.

from pathlib import Path
import sys
import csv
import numpy as np
import meep as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Fixed parameters (2.5D EIM) ──────────────────────────────────────────
ring_w = 0.450
wg_w   = 0.450
n_core = 2.8217
n_clad = 1.44
pml    = 1.5
pad    = 2.0
resolution = 22

OUT = Path(__file__).resolve().parent / "characterization_results"
OUT.mkdir(parents=True, exist_ok=True)

# m=140 candidate: closest to 1550nm
CONFIGS = [
    # {"R": 24.4949, "gap": 0.2916, "label": "highestq",
    #  "Q_harminv": 261051527, "lambda_res": 1.540905},
    {"R": 24.7751, "gap": 0.2853, "label": "secondhighestq",
     "Q_harminv": 249786, "lambda_res": 1.5541361},
]


def build_ring_geometry(R, ring_w, gap, wg_w):
    core = mp.Medium(index=n_core)
    clad = mp.Medium(index=n_clad)
    bus_y = (R + ring_w / 2) + gap + wg_w / 2
    outer = mp.Cylinder(radius=R + ring_w / 2, material=core)
    inner = mp.Cylinder(radius=R - ring_w / 2, material=clad)
    bus_len = 2 * (R + pad + pml)
    bus = mp.Block(size=mp.Vector3(bus_len, wg_w, mp.inf),
                   center=mp.Vector3(0, bus_y), material=core)
    return [outer, inner, bus], bus_y


def build_bus_only(R, ring_w, gap, wg_w):
    core = mp.Medium(index=n_core)
    bus_y = (R + ring_w / 2) + gap + wg_w / 2
    bus_len = 2 * (R + pad + pml)
    bus = mp.Block(size=mp.Vector3(bus_len, wg_w, mp.inf),
                   center=mp.Vector3(0, bus_y), material=core)
    return [bus], bus_y


def run_flux(geometry, bus_y, R, fcen, fwidth, nfreq):
    sx = 2 * (R + pad + pml)
    sy = 2 * (R + ring_w / 2 + 0.20 + wg_w / 2 + pad + pml)
    cell = mp.Vector3(sx, sy)

    src_x = -sx / 2 + pml + 1.5

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=mp.Vector3(src_x, bus_y),
            size=mp.Vector3(0, 3 * wg_w),
            eig_band=1,
            direction=mp.X,
            eig_parity=mp.EVEN_Z,  # TE: Hz is even under z-mirror
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

    output_x = sx / 2 - pml - 1.5
    flux_mon = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(
            center=mp.Vector3(output_x, bus_y),
            size=mp.Vector3(0, 3 * wg_w),
            direction=mp.X,
        ),
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Hz, mp.Vector3(output_x, bus_y), 1e-9))

    freqs = np.array(mp.get_flux_freqs(flux_mon))
    fluxes = np.array(mp.get_fluxes(flux_mon))
    return freqs, fluxes


def run_normalized_transmission(R, gap, lambda_center, nfreq=1500):
    lam_lo = lambda_center - 0.020  # ±30 nm window for FSR visibility
    lam_hi = lambda_center + 0.020
    fcen = 1.0 / lambda_center
    fwidth = 1.0 / lam_lo - 1.0 / lam_hi

    print("  [1/2] Reference run (bus waveguide only)...")
    sys.stdout.flush()
    geom_ref, bus_y = build_bus_only(R, ring_w, gap, wg_w)
    freqs, flux_ref = run_flux(geom_ref, bus_y, R, fcen, fwidth, nfreq)

    print("  [2/2] Full run (bus + ring)...")
    sys.stdout.flush()
    geom_full, bus_y = build_ring_geometry(R, ring_w, gap, wg_w)
    _, flux_full = run_flux(geom_full, bus_y, R, fcen, fwidth, nfreq)

    transmission = flux_full / np.where(np.abs(flux_ref) > 1e-20, flux_ref, 1e-20)
    return freqs, transmission


def extract_resonances(freqs, transmission):
    wavelengths = 1.0 / freqs
    T = np.clip(transmission, 1e-12, None)
    dip_indices = []
    for i in range(2, len(T) - 2):
        if T[i] < T[i - 1] and T[i] < T[i + 1] and T[i] < 0.95:
            dip_indices.append(i)

    results = []
    for idx in dip_indices:
        lam_res = wavelengths[idx]
        T_min = T[idx]
        half_level = 0.5 * (1.0 + T_min)
        left = idx
        while left > 0 and T[left] < half_level:
            left -= 1
        right = idx
        while right < len(T) - 1 and T[right] < half_level:
            right += 1
        if left == 0 or right == len(T) - 1:
            continue
        fwhm = abs(wavelengths[left] - wavelengths[right])
        if fwhm < 1e-6:
            continue
        Q_loaded = lam_res / fwhm
        ER_dB = -10.0 * np.log10(max(T_min, 1e-12))
        results.append(dict(lambda_res=lam_res, T_min=T_min, fwhm=fwhm,
                            Q=Q_loaded, ER_dB=ER_dB))

    if len(results) >= 2:
        results.sort(key=lambda r: r["lambda_res"])
        for i in range(len(results) - 1):
            results[i]["FSR_nm"] = abs(results[i+1]["lambda_res"] - results[i]["lambda_res"]) * 1e3

    return results


def plot_and_save(freqs, transmission, resonances, config):
    R = config["R"]
    gap = config["gap"]
    label = config["label"]
    Q_harm = config["Q_harminv"]
    lam_center = config["lambda_res"]

    wavelengths = 1.0 / freqs * 1e3
    lam_center_nm = lam_center * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    ax.plot(wavelengths, transmission, "b-", linewidth=0.8)
    ax.set_ylabel("Normalized Transmission T(λ)", fontsize=11)
    ax.set_title(f"Normalized Transmission — R={R} µm, gap={gap} µm, ring_w={ring_w} µm\n"
                 f"(Harminv Q = {Q_harm:,.0f}, resolution={resolution})", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Wavelength [nm]")

    for r in resonances:
        lam_nm = r["lambda_res"] * 1e3
        fsr_str = f"\nFSR={r['FSR_nm']:.2f}nm" if "FSR_nm" in r else ""
        ax.annotate(f"Q={r['Q']:,.0f}\nER={r['ER_dB']:.1f}dB{fsr_str}",
                    xy=(lam_nm, r["T_min"]),
                    xytext=(0, -45), textcoords="offset points",
                    fontsize=6.5, ha="center", color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.7))

    ax2 = axes[1]
    zoom_hw = 3.0
    mask = (wavelengths > lam_center_nm - zoom_hw) & (wavelengths < lam_center_nm + zoom_hw)
    if np.any(mask):
        ax2.plot(wavelengths[mask], transmission[mask], "b-", linewidth=1.2,
                 marker=".", markersize=3)
    ax2.set_xlabel("Wavelength [nm]", fontsize=11)
    ax2.set_ylabel("T(λ)", fontsize=11)
    ax2.set_title(f"Zoom: {lam_center_nm-zoom_hw:.0f}–{lam_center_nm+zoom_hw:.0f} nm "
                  f"(around Harminv λ={lam_center_nm:.1f} nm)", fontsize=10)
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax2.axvline(lam_center_nm, color="red", ls=":", lw=1.0, alpha=0.6,
                label=f"Harminv λ={lam_center_nm:.1f} nm")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = OUT / f"transmission_{label}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved → {out}")

    csv_path = OUT / f"transmission_{label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_inv_um", "wavelength_nm", "normalized_transmission"])
        for freq, t in zip(freqs, transmission):
            writer.writerow([freq, 1e3 / freq, t])
    print(f"[OK] CSV saved → {csv_path}")


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    configs_to_run = [CONFIGS[idx]] if idx >= 0 else CONFIGS

    for config in configs_to_run:
        R, gap = config["R"], config["gap"]
        print(f"\n{'='*60}")
        print(f"Normalized transmission: R={R} µm, gap={gap} µm  (res={resolution})")
        print(f"{'='*60}")
        sys.stdout.flush()

        freqs, T = run_normalized_transmission(R, gap, config["lambda_res"])
        resonances = extract_resonances(freqs, T)
        plot_and_save(freqs, T, resonances, config)

        if resonances:
            print(f"\n  Found {len(resonances)} resonance(s):")
            for r in resonances:
                fsr_str = f", FSR={r['FSR_nm']:.2f} nm" if "FSR_nm" in r else ""
                print(f"    λ={r['lambda_res']*1e3:.2f} nm, Q={r['Q']:,.0f}, "
                      f"ER={r['ER_dB']:.1f} dB, FWHM={r['fwhm']*1e3:.4f} nm{fsr_str}")
        else:
            print("  No resonance dips found.")


if __name__ == "__main__":
    main()
