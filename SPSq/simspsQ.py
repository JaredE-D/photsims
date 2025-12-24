# sim/meep_sps_ring.py
# --------------------
# Requires: meep, numpy, matplotlib (optional for saving a field image)


from pathlib import Path
import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from numpy._core.numeric import True_


R = 29.44 #11.766 
ring_w = 0.2988
gap = 0.06
wg_w = 0.5

bus_y = (R + ring_w/2) + gap + wg_w/2 # bus above ring (horizontal)

lambda0 = 1.55 # target wavelength [um]
fcen = 1 / lambda0 # center frequency [1/um]
fwidth = 0.1 * fcen # fractional bandwidth for Gaussian source


n_core = 3.45 # effective index for Si
n_clad = 1.44 # cladding (SiO2/air effective)

# Simulation domain
pml = 2.0 # PML thickness [um]
pad = 4.0 # padding around geometry [um]
# Resolution

resolution = 20 # pixels per µm (raise to 40+ for higher accuracy)


#Sweep Simulation Parameters Space
SPACE = [
    Real(11.0,50, name ="R"),
    Real(0.0,0.1, name ="gap"),
    Real(0.2,0.38, name ="ring_w")
]


# Results directory
OUT = Path(__file__).resolve().parent / "resultsQ"
OUT.mkdir(parents=True, exist_ok=True)


RESULTS_DIR = Path("results_bayesoptQ")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = RESULTS_DIR / f"sps_bo_log_{timestamp}.csv"

FIELDNAMES = [
    "R_um",
    "gap_um",
    "ring_w",
    "freq",
    "score",
    "cost",
]



def build_geometry(include_ring=True, include_bus=True, R=2.466, ring_w = 0.5, gap = 0.2, wg_w = 0.5):
    """Returns geometry list for Meep."""
    
        
    # ---------------------
    # Physical / geometry params (µm)
    # ---------------------
    

    # Derived placement
    ring_center = mp.Vector3(0, 0)
    bus_y = (R + ring_w/2) + gap + wg_w/2 # bus above ring (horizontal)

    geom = []
    clad = mp.Medium(index=n_clad)
    core = mp.Medium(index=n_core)


    # Background is clad (set as default_material in Simulation)


    if include_ring:
    # Outer core cylinder then inner clad cylinder to make an annulus
        outer = mp.Cylinder(radius=R + ring_w/2, material=core, center=ring_center)
        inner = mp.Cylinder(radius=R - ring_w/2, material=clad, center=ring_center)
        geom += [outer, inner]


    if include_bus:
        # Horizontal bus waveguide above the ring
        bus_len = 2 * (R + pad + pml) # long enough to cross the domain
        bus = mp.Block(size=mp.Vector3(bus_len, wg_w, mp.inf),
        center=mp.Vector3(0, bus_y), material=core)
        geom += [bus]


    return geom




EDGE_EPS = 0.2  # small offset from PML to avoid touching it

def preview_geometry(include_ring=True, include_bus=True, R=R, ring_w = ring_w, gap = gap, wg_w = wg_w):
    """Builds a Meep Simulation and plots the dielectric distribution (ε)
    without running a time-domain simulation."""
    # Same cell size you use in run_sim
    sx = 2 * (R + pad + pml)
    sy = 2 * (R + (wg_w/2 + ring_w/2 + gap if include_bus else 0) + pad + pml)

    cell = mp.Vector3(sx, sy)
    boundary_layers = [mp.PML(pml)]

    geometry = build_geometry(include_ring=include_ring, include_bus=include_bus, R=R, ring_w = ring_w, gap=gap, wg_w=wg_w)

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=geometry,
        default_material=mp.Medium(index=n_clad),
    )
    theta = -np.pi/2 # top of the ring
    r_src = R # radial position ~ ring center radius
    src_pos = mp.Vector3(r_src * np.cos(theta), r_src * np.sin(theta))


    # Initialize the simulation (sets up grid, materials, etc.)
    sim.init_sim()

    # Get dielectric distribution on the full cell
    eps = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

    plt.figure(figsize=(10, 10))
    # Rotate so axes look natural (x horizontal, y vertical)
    plt.imshow(
        eps.T,
        extent=[-sx/2, sx/2, -sy/2, sy/2],
        interpolation="nearest",
        origin="lower",
    )
    plt.scatter(src_pos.x, src_pos.y)
    plt.xlabel("x [µm]")
    plt.ylabel("y [µm]")
    plt.title("Dielectric profile (ε)")
    plt.colorbar(label="ε")
    plt.tight_layout()
    plt.show()


def save_fieldsfunc(sim):
    t = sim.meep_time()
    filename = f"Ez_t{t:.1f}.h5"
    sim.output_hdf5(mp.Ez, filename=filename)

def run_simQ(R1, gap1, ring_w1, include_ring=True, include_bus=True, save_fields=False):
    # Cell size
    R1 = round(R1,2)
    gap1 = round(gap1,2)
    ring_w1 = round(ring_w1,2)
    sx = 2 * (R1 + pad + pml)
    sy = 2 * (R1 + (wg_w/2 + ring_w1/2 + gap1 if include_bus else 0) + pad + pml)
    

    cell = mp.Vector3(sx, sy)
    boundary_layers = [mp.PML(pml)]

    
    # Build geometry
    geometry = build_geometry(include_ring=include_ring, include_bus=include_bus, R=R1, ring_w = ring_w1, gap=gap1)
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=geometry,
        default_material=mp.Medium(index=n_clad),
        symmetries=[mp.Mirror(direction=mp.X)],
        )
    sim.use_output_directory("imgresultsQ")


    # Dipole source placed near ring outer edge at top (away from bus)
    # Choose polarization matching dominant ring mode (Ez out-of-plane in 2D TE-like)
    theta = -np.pi/2 # bottom of the ring
    r_src = R1 # radial position ~ ring center radius
    src_pos = mp.Vector3(r_src * np.cos(theta), r_src * np.sin(theta))

    Qmeas_pos = mp.Vector3(r_src , 0) # right side of ring

    sources = [
    mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
    component=mp.Ez, # TE-like (2D)
    center=src_pos,
    amplitude=1.0)
    ]
    
    harm = mp.Harminv(mp.Ez, Qmeas_pos, fcen, fwidth)
    
    sim.sources = sources
    sim.run(
        mp.after_sources(harm),
        until_after_sources=1000
    )
    modes = harm.modes
    best = min(modes, key=lambda modes: abs(modes.freq - fcen))
    
    return best.freq, best.Q

def get_prev_results(csvname, lim=SPACE):
    data = np.loadtxt(csvname, skiprows=1,delimiter=',', dtype=float, encoding='utf-8')
    x0 = [list(row[0:3]) for row in data]
    y0 = [row[-1] for row in data]
    x0np = np.array(x0)
    y0np = np.array(y0)
    lower_bounds = np.array([dim.low for dim in lim])  
    upper_bounds = np.array([dim.high for dim in lim])   
    is_above_min = x0 >= lower_bounds
    is_below_max = x0 <= upper_bounds
    is_in_range = is_above_min & is_below_max
    valid_indices_mask = np.all(is_in_range, axis=1)
    # print(x0np[valid_indices_mask].tolist())
    # raise OSError
    return x0np[valid_indices_mask].tolist(), y0np[valid_indices_mask].tolist()



#TBD
#def runsimsweepQ(include_ring=True, include_bus=True, R1= 2.433, ring_w=0.5, gap=0.5):
    # Cell size
    
                
@use_named_args(SPACE)
def objective(R, gap, ring_w):
    # Run the expensive sim
    freq1, qload = run_simQ(R, gap, ring_w)

    score = qload
    cost = -score  # gp_minimize tries to MINIMIZE

    # Prepare row for CSV
    row = {
        "R_um": R,
        "gap_um": gap,
        "ring_w": ring_w,
        "freq": freq1,
        "score": score,
        "cost": cost,
    }
    
    # Append to CSV
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


    return cost

mainpross = 3

#1 for preview geometry
#2 for run 1 sim
#3 is sweep run
#4 is for sweep run using previous data
#5 is for inverse design
## Used for purcell factor approximate calculatino using power

if __name__ == "__main__":
    # 1) Full device: ring + bus (β-factor numerator and total power)
    match(mainpross):
        
        case 1:
            preview_geometry()
        case 2:
                    # Create a unique filename carrying info about the geometry + timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_name = f"sps_R{R}_g{gap}_w{ring_w}_bus{wg_w}_lam{lambda0}_{timestamp}.json"
            json_path = OUT / json_name
            freq, Q = run_simQ(R, ring_w, gap)
            # Package results into a dictionary
            results = {
                "lambda0_um": float(lambda0),
                "parameters": {
                    "R_um": float(R),
                    "ring_width_um": float(ring_w),
                    "bus_width_um": float(wg_w),
                    "gap_um": float(gap)
                },
                "metrics": {
                    "freq": float(freq),
                    "Q": float(Q)
                }, 
                "timestamp": timestamp
            }
            # Write JSON file
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)

            print(f"[OK] Wrote JSON → {json_path}") 
        case 3:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            
            result = gp_minimize(
            func=objective,
            dimensions=SPACE,
            n_calls=300,           # how many evaluations (Meep runs / 2 because we run 2 simulations per )
            n_initial_points=20,   # random starts
            acq_func="EI",        # Expected Improvement
            random_state=0
            )

            print("\n[ DONE ]")
            print("Best parameters found:")
            print(f"  R   = {result.x[0]:.4f} um")           
            print(f"  gap = {result.x[1]:.4f} um")
            print(f"Best cost (minimized) = {result.fun:.4f}")
            
            # Save best params as JSON
            best = {
                "R_um": result.x[0],
                "gap_um": result.x[1],
                "ring_w": result.x[2],
                "best_cost": result.fun,
            }
            best_json = RESULTS_DIR / f"best_params_{timestamp}.json"
            with open(best_json, "w") as f:
                json.dump(best, f, indent=2)
            print(f"[OK] Saved best params → {best_json}")
            print(f"[OK] Logged all evals → {csv_path}")
        case 4:
            x1, y1 = get_prev_results("results_bayesoptQ/sps_bo_log_20251128-185833.csv", lim=SPACE)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            
            result = gp_minimize(
            func=objective,
            dimensions=SPACE,
            n_calls=100,           # how many evaluations
            n_initial_points=0,   # random starts
            acq_func="EI",        # Expected Improvement
            random_state=100,
            x0=x1,
            y0=y1
            )

            print("\n[ DONE ]")
            print("Best parameters found:")
            print(f"  R   = {result.x[0]:.4f} um")
            print(f"  gap = {result.x[1]:.4f} um")
            print(f"Best cost (minimized) = {result.fun:.4f}")
            
            # Save best params as JSON
            best = {
                "R_um": result.x[0],
                "gap_um": result.x[1],
                "ring_w": result.x[2],
                "best_cost": result.fun,
            }
            best_json = RESULTS_DIR / f"best_params_{timestamp}.json"
            with open(best_json, "w") as f:
                json.dump(best, f, indent=2)
            print(f"[OK] Saved best params → {best_json}")
            print(f"[OK] Logged all evals → {csv_path}")
                
        case _:
            print("Error match / case is not a proper number")
            