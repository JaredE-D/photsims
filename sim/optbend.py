import os
import sys
import pathlib
import numpy as np

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
import gplugins.gmeep as gm

from gplugins.gmeep.get_simulation import get_simulation
from gdsklivehelp import show_in_klive
import ../../gdsklivehelp

# --- Activate Generic PDK (provides layer_stack, cross-sections, etc.) ---
PDK = get_generic_pdk()
PDK.activate()


r = gf.Component()

@gf.cell
def flatbend(size = 2):
    c = gf.Component()
    c.add_polygon([(0,0),( size,0 ),( size + size/np.tan(1.178097), size ),
                   ( size, size+size/np.tan(1.178097) ),( 0,size )], layer=(1,0))
    s1 = gf.components.rectangle()
    s2 = gf.components.rectangle()
    return c


# --- Choose / parametrize a component to demo ---
# You can swap this for e.g. gf.components.ring_single() or your own custom cell.
#c = gf.components.mmi1x2(width=0.5, width_taper=0.5, length_mmi=2.5)

## Optional: add short tapers + straight to give some room for ports for simulation
#left = gf.components.straight(length=5.0)
#right = gf.components.straight(length=5.0)
#wg = gf.components.straight(length=2.0)

## Connect a bit of routing to ensure ports face outward (nice for Meep)
#r = gf.Component("mmi1x2_sim_routed")
#mmi = r << c
#l = r << left
#l.connect("o2", mmi.ports["o1"])
#o2 = r << right
#o2.connect("o1", mmi.ports["o2"])
#o3 = r << right
#o3.connect("o1", mmi.ports["o3"])
r = flatbend()
## Expose ports
r.add_port("o_in", port=l.ports["o1"])
#r.add_port("o_top", port=o2.ports["o2"])
#r.add_port("o_bot", port=o3.ports["o2"])

if __name__ == "__main__":
    show_in_klive(r)
    
    



