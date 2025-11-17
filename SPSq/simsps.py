# sim/meep_sps_ring.py
# --------------------
# Requires: meep, numpy, matplotlib (optional for saving a field image)


from pathlib import Path
import numpy as np
import meep as mp
import matplotlib.pyplot as plt
from numpy._core.numeric import True_


# ---------------------
# Physical / geometry params (µm)
# ---------------------
lambda0 = 1.55 # target wavelength [um]
fcen = 1 / lambda0 # center frequency [1/um]
fwidth = 0.1 * fcen # fractional bandwidth for Gaussian source


n_core = 3.45 # effective index for Si
n_clad = 1.44 # cladding (SiO2/air effective)


R = 2.466 # ring center radius [um]
ring_w = 0.5 # ring waveguide width [um]
wg_w = 0.5 # bus waveguide width [um]


gap = 0.20 # bus gap to ring outer edge [um]


# Simulation domain
pml = 1.0 # PML thickness [um]
pad = 4.0 # padding around geometry [um]


# Derived placement
ring_center = mp.Vector3(0, 0)
bus_y = (R + ring_w/2) + gap + wg_w/2 # bus above ring (horizontal)


# Resolution

resolution =20 # pixels per µm (raise to 40+ for higher accuracy)


# Results directory
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)




def build_geometry(include_ring=True, include_bus=True):
    """Returns geometry list for Meep."""
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
        bus_len = 2 * (R + pad) # long enough to cross the domain
        bus = mp.Block(size=mp.Vector3(bus_len, wg_w, mp.inf),
        center=mp.Vector3(0, bus_y), material=core)
        geom += [bus]


    return geom




EDGE_EPS = 0.2  # small offset from PML to avoid touching it

def add_flux_monitors(sim, sx, sy):
    """Return dict of flux objects: wg, left, right, top, bottom."""
    # Waveguide monitor (measures +x flux)
    x_flux = 0.5 * sx - pml - EDGE_EPS
    wg = sim.add_flux(fcen, 0, 1, mp.FluxRegion(
        center=mp.Vector3(x_flux, bus_y),
        size=mp.Vector3(0, 1.5*wg_w),
        direction=mp.X,             # explicitly +x
    ))

    # Total-power boundary: four lines just inside PML
    halfx = 0.5 * sx - pml - EDGE_EPS
    halfy = 0.5 * sy - pml - EDGE_EPS
    L = sim.add_flux(fcen, 0, 1, mp.FluxRegion(  # normal = +x (points right)
        center=mp.Vector3(-halfx, 0),
        size=mp.Vector3(0, 2*halfy),
        direction=mp.X,
    ))
    R = sim.add_flux(fcen, 0, 1, mp.FluxRegion(  # normal = +x (points outward)
        center=mp.Vector3(+halfx, 0),
        size=mp.Vector3(0, 2*halfy),
        direction=mp.X,
    ))
    B = sim.add_flux(fcen, 0, 1, mp.FluxRegion(  # normal = +y (points upward)
        center=mp.Vector3(0, -halfy),
        size=mp.Vector3(2*halfx, 0),
        direction=mp.Y,
    ))
    T = sim.add_flux(fcen, 0, 1, mp.FluxRegion(  # normal = +y (points outward)
        center=mp.Vector3(0, +halfy),
        size=mp.Vector3(2*halfx, 0),
        direction=mp.Y,
    ))

    return dict(wg=wg, L=L, R=R, B=B, T=T)


def preview_geometry(include_ring=True, include_bus=True):
    """Builds a Meep Simulation and plots the dielectric distribution (ε)
    without running a time-domain simulation."""
    # Same cell size you use in run_sim
    sx = 2 * (R + pad + pml)
    sy = 2 * (R + (bus_y if include_bus else R) + pad + pml)

    cell = mp.Vector3(sx, sy)
    boundary_layers = [mp.PML(pml)]

    geometry = build_geometry(include_ring=include_ring, include_bus=include_bus)

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=geometry,
        default_material=mp.Medium(index=n_clad),
    )

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
    plt.xlabel("x [µm]")
    plt.ylabel("y [µm]")
    plt.title("Dielectric profile (ε)")
    plt.colorbar(label="ε")
    plt.tight_layout()
    plt.show()

def run_sim(include_ring=True, include_bus=True, save_fields=False):
    # Cell size
    sx = 2 * (R + pad + pml)
    sy = 2 * (R + bus_y + pad + pml) if include_bus else 2 * (R + pad + pml)


    cell = mp.Vector3(sx, sy)
    boundary_layers = [mp.PML(pml)]


    # Build geometry
    geometry = build_geometry(include_ring=include_ring, include_bus=include_bus)
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=geometry,
        default_material=mp.Medium(index=n_clad),
        )


    # Dipole source placed near ring outer edge at top (away from bus)
    # Choose polarization matching dominant ring mode (Ez out-of-plane in 2D TE-like)
    theta = np.pi/2 # top of the ring
    r_src = R # radial position ~ ring center radius
    src_pos = mp.Vector3(r_src * np.cos(theta), r_src * np.sin(theta))


    sources = [
    mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
    component=mp.Ez, # TE-like (2D)
    center=src_pos,
    amplitude=1.0)
    ]


    sim.sources = sources


    # Monitors
    mons = add_flux_monitors(sim, sx, sy)

    # Run long enough for fields to decay
    sim.run(until=1200)
    P_wg = mp.get_fluxes(mons["wg"])[0]

    # Net outward power outward On the waveguides axis:
    # Right (+x outward) Plus Left (outward is -x) Minus
    # Top (+y outward)   Plus Bottom (outward is -y)
    P_R = mp.get_fluxes(mons["R"])[0]
    P_L = mp.get_fluxes(mons["L"])[0]
    P_T = mp.get_fluxes(mons["T"])[0]
    P_B = mp.get_fluxes(mons["B"])[0]
    P_tot = (P_R + P_L) - (P_T + P_B)


    if save_fields:
        eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
        ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
        plt.figure(figsize=(10,10))
        plt.imshow(eps_data.T, interpolation='spline36', extent=[-sx/2, sx/2, -sy/2, sy/2])
        plt.imshow(np.real(ez_data).T, interpolation='spline36', alpha=0.7, extent=[-sx/2, sx/2, -sy/2, sy/2])
        plt.xlabel('x [µm]'); plt.ylabel('y [µm]'); plt.title('ε and Re(Ez) snapshot')
        out_png = OUT / 'fields.png'
        plt.savefig(out_png, dpi=180, bbox_inches='tight')
        print(f"[OK] Saved field snapshot → {out_png}")


    return P_wg, P_tot



get_geo = False
if __name__ == "__main__":
    # 1) Full device: ring + bus (β-factor numerator and total power)
    if(get_geo):
        preview_geometry()
    else:
        P_wg_device, P_tot_device = run_sim(include_ring=True, include_bus=True, save_fields=True)
        
        # 2) Control: homogeneous cladding only (Purcell proxy denominator)
        # Temporarily disable geometry for control
        P_wg_ctrl, P_tot_ctrl = run_sim(include_ring=False, include_bus=False, save_fields=False)

        
        beta = P_wg_device / max(P_tot_device, 1e-12)
        purcell_proxy = P_tot_device / max(P_tot_ctrl, 1e-12)


        txt = OUT / 'summary.txt'
        with open(txt, 'w') as f:
            f.write(f"lambda0 [um]: {lambda0}\n")
            f.write(f"ring R={R} um, w={ring_w} um, bus w={wg_w} um, gap={gap} um\n")
            f.write(f"Beta (≈ P_bus / P_total): {beta:.3f}\n")
            f.write(f"Purcell proxy (P_total / P_clad): {purcell_proxy:.2f}\n")
            f.write(f"P_bus_device: {P_wg_device:.4e}\n")
            f.write(f"P_total_device: {P_tot_device:.4e}\n")
            f.write(f"P_total_cladding: {P_tot_ctrl:.4e}\n")
            print(f"[OK] Wrote → {txt}")