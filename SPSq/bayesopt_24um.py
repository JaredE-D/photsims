# SPSq/bayesopt_24um.py
# Bayesian optimization of Q near R≈24µm.
# Usage: python3 -u bayesopt_24um.py [--seed-csv <path>] [--R-center <val>]
#
# Search space: R ± 1µm around center, ring_w 0.35-0.55µm, gap=0.25 fixed
# Objective: maximize Q (minimize -Q) using skopt GP

from pathlib import Path
import csv
import sys
import time
import argparse
import numpy as np
import meep as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ── Design parameters (2.5D EIM) ──────────────────────────────────────────
wg_w   = 0.450
n_core = 2.8217
n_clad = 1.44
pml    = 1.5
pad    = 2.0
resolution = 23

lambda0 = 1.55
fcen   = 1.0 / lambda0
fwidth = 0.1 * fcen
n_eff_te0 = 2.5675

gap = 0.25  # fixed

# ── Output ──────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent / "characterization_results"
OUT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = OUT / f"bayesopt_24um_{timestamp}.csv"
FIELDNAMES = ["R_um", "ring_w_um", "gap_um", "resolution", "freq",
              "lambda_res_um", "Q", "amp", "cost"]


def build_geometry(R, ring_w):
    core = mp.Medium(index=n_core)
    clad = mp.Medium(index=n_clad)
    outer = mp.Cylinder(radius=R + ring_w / 2, material=core)
    inner = mp.Cylinder(radius=R - ring_w / 2, material=clad)
    bus_y = R + ring_w / 2 + gap + wg_w / 2
    bus_len = 2 * (R + pad + pml)
    bus = mp.Block(size=mp.Vector3(bus_len, wg_w, mp.inf),
                   center=mp.Vector3(0, bus_y), material=core)
    return [outer, inner, bus]


def harminv_time(R):
    round_trip = 2 * np.pi * R * n_eff_te0
    return max(int(round_trip * 10), 2500)


def run_harminv(R, ring_w):
    sx = 2 * (R + pad + pml)
    sy = 2 * (R + ring_w / 2 + gap + wg_w / 2 + pad + pml)
    cell = mp.Vector3(sx, sy)
    geometry = build_geometry(R, ring_w)

    theta = -np.pi / 2
    src_pos = mp.Vector3(R * np.cos(theta), R * np.sin(theta))
    Qmeas_pos = mp.Vector3(R, 0)

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            component=mp.Hz,
            center=src_pos,
            amplitude=1.0,
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

    t_harm = harminv_time(R)
    harm = mp.Harminv(mp.Hz, Qmeas_pos, fcen, fwidth)
    sim.run(mp.after_sources(harm), until_after_sources=t_harm)

    modes = harm.modes
    physical = [m for m in modes if m.freq > 0 and m.Q > 0 and abs(m.amp) > 1e-8]
    if not physical:
        return None, None, None, None

    best = max(physical, key=lambda m: m.Q)
    return best.freq, best.Q, 1.0 / best.freq, abs(best.amp)


call_count = 0

def objective(params):
    global call_count
    call_count += 1
    R_val, ring_w_val = params
    t0 = time.time()

    freq, Q, lam, amp = run_harminv(R_val, ring_w_val)
    elapsed = time.time() - t0

    cost = -Q if Q else 0.0  # penalty for no mode

    row = {
        "R_um": round(R_val, 4), "ring_w_um": round(ring_w_val, 4),
        "gap_um": gap, "resolution": resolution,
        "freq": round(freq, 6) if freq else "",
        "lambda_res_um": round(lam, 6) if lam else "",
        "Q": round(Q, 1) if Q else "",
        "amp": round(amp, 8) if amp else "",
        "cost": round(cost, 1),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    if Q:
        print(f"  [{call_count:3d}] R={R_val:.3f} ring_w={ring_w_val:.3f} "
              f"→ λ={lam*1e3:.2f}nm Q={Q:,.0f} ({elapsed:.0f}s)")
    else:
        print(f"  [{call_count:3d}] R={R_val:.3f} ring_w={ring_w_val:.3f} "
              f"→ no mode ({elapsed:.0f}s)")
    sys.stdout.flush()

    return cost


def load_seed_csv(path):
    """Load previous results as seed points for BayesOpt."""
    x0, y0 = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Q"] == "":
                continue
            R = float(row["R_um"])
            # Use ring_w from CSV if available, else default
            rw = float(row.get("ring_w_um", 0.450))
            Q = float(row["Q"])
            x0.append([R, rw])
            y0.append(-Q)
    return x0, y0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-csv", type=str, default=None,
                        help="CSV from sweep to seed the optimizer")
    parser.add_argument("--R-center", type=float, default=24.0,
                        help="Center R for search range (default 24.0)")
    parser.add_argument("--n-calls", type=int, default=60,
                        help="Number of BayesOpt evaluations")
    args = parser.parse_args()

    R_lo = args.R_center - 1.0
    R_hi = args.R_center + 1.0
    space = [
        Real(R_lo, R_hi, name="R"),
        Real(0.350, 0.550, name="ring_w"),
    ]

    print(f"Bayesian Optimization: R=[{R_lo:.1f}, {R_hi:.1f}], "
          f"ring_w=[0.350, 0.550], gap={gap}")
    print(f"Resolution={resolution}, n_calls={args.n_calls}")
    print(f"Output → {csv_path}")

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

    kwargs = dict(
        func=objective,
        dimensions=space,
        n_calls=args.n_calls,
        acq_func="EI",
        random_state=42,
    )

    if args.seed_csv:
        x0, y0 = load_seed_csv(args.seed_csv)
        print(f"Seeded with {len(x0)} points from {args.seed_csv}")
        # Filter to within bounds
        x0_filt, y0_filt = [], []
        for x, y in zip(x0, y0):
            if R_lo <= x[0] <= R_hi and 0.350 <= x[1] <= 0.550:
                x0_filt.append(x)
                y0_filt.append(y)
        if x0_filt:
            kwargs["x0"] = x0_filt
            kwargs["y0"] = y0_filt
            kwargs["n_initial_points"] = max(5, args.n_calls // 6)
            print(f"  {len(x0_filt)} seed points within bounds")
        else:
            kwargs["n_initial_points"] = max(10, args.n_calls // 4)
            print("  No seed points within bounds, using random initial")
    else:
        kwargs["n_initial_points"] = max(10, args.n_calls // 4)

    sys.stdout.flush()
    t0 = time.time()
    result = gp_minimize(**kwargs)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"BAYESIAN OPTIMIZATION COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"Best R      = {result.x[0]:.4f} µm")
    print(f"Best ring_w = {result.x[1]:.4f} µm")
    print(f"Best Q      = {-result.fun:,.0f}")
    print(f"Gap         = {gap} µm (fixed)")
    print(f"All evals   → {csv_path}")
    print(f"{'='*60}")

    # Save convergence plot
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(result.func_vals)
    best_so_far = np.minimum.accumulate(result.func_vals)
    ax.plot(range(1, n + 1), -np.array(result.func_vals), "o", alpha=0.4,
            markersize=4, label="Each eval Q")
    ax.plot(range(1, n + 1), -best_so_far, "r-", lw=2, label="Best Q so far")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Q factor")
    ax.set_title(f"BayesOpt Convergence — R≈{args.R_center}µm, gap={gap}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / f"bayesopt_24um_convergence_{timestamp}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[OK] Convergence plot → {out}")


if __name__ == "__main__":
    main()
