from pathlib import Path
import gdsfactory as gf
import gplugins.gmeep as gm
import gdsklivehelp
import gplugins
"""
Project Parameters 
The goal of this project is to get better at designing photonic assets on a large scale, and sweeping over 
multiple important parameters in order to optimize our waveguidesconda install -c conda-forge pymeep pymeep-extras.

In this case we will be working with a directional coupler, a basic component of photonic systems.

"""

WAVELENGTH = 1.55
WG_WIDTH = 0.5
BEND_RADIUS = 10.0


# Sweeps


GAPS = [0.15 ]
COUPLING_LENGTHS = [10]

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
GDS_PATH = OUT / "dc_sweep.gds"




def make_dc(gap: float = 0.2, length: float = 20.0, wg_width: float = WG_WIDTH) -> gf.Component:
    """Parametric directional coupler using GDSFactory's built-in PCell.
    Ports (default): o1,o2 (inputs), o3,o4 (outputs) depending on orientation.
    """
    c = gf.components.coupler(
        gap=gap,
        length=length,
        dx=length/2,
        dy=wg_width/2, # differential width
        bend="bend_s",
        cross_section='strip',
        allow_min_radius_violation=True
    )
    c.info["gap_um"] = gap
    c.info["Lc_um"] = length
    c.info["w_um"] = wg_width
    c.info["lambda_um"] = WAVELENGTH
    return c



def build_library():
    cells = []
    for gap in GAPS:
        for Lc in COUPLING_LENGTHS:
            dc = make_dc(gap=gap, length=Lc)
            dc_copy = dc.copy()
            dc_copy.name = f"dc_gap{gap:.2f}_Lc{Lc:.1f}_w{WG_WIDTH:.2f}_lam{WAVELENGTH:.2f}"
            cells.append(dc)


# Pack devices into as few tiles as possible for a neat GDS gallery
    packed = gf.pack(cells, max_size=(5000, 5000), spacing=20)
    top = gf.Component("DC_SWEEP")
    for i, p in enumerate(packed):
        top.add_ref(p).move((0, -i * 0)) # already laid out by pack


    top.write_gds(GDS_PATH)
    print(f"[OK] Wrote {GDS_PATH}")


def showbuild_library():
    top = gf.import_gds(GDS_PATH)

    gdsklivehelp.show_in_klive(top)

def sim():
    core_material = gplugins.get_effective_indices(
        core_material=3.4777,
        clad_materialding=1.444,
        nsubstrate=1.444,
        thickness=0.22,
        wavelength=1.55,
        polarization="te",
    )[0]
    top = gf.import_gds(GDS_PATH)
    sp = gm.write_sparameters_meep(
        top,
        resolution=20,
        is_3d=False,
        material_name_to_meep=dict(si=core_material)
    )   
    
    gplugins.plot.plot_sparameters(sp, keys=("o2@0,o1@0",))



if __name__ == "__main__":
    if False :
        build_library() 
    else:
        sim()