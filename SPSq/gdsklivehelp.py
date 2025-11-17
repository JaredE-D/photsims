# utils_show.py (WSL)
import json, socket, subprocess
from pathlib import Path

"""
This file is here to help with WSL connection to Windows KLayout / Klive, to allow live streaming of GDS files.
"""
def show_in_klive(component, out_dir="/mnt/c/Users/adskf/gds_tmp",
                  name="design.gds", host="127.0.0.1", port=8082):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gds_linux = out / name
    component.write_gds(gds_linux)

    # Convert /mnt/c/... -> C:\...\... using wslpath (no hardcoded username)
    win_path = subprocess.check_output(["wslpath", "-w", str(gds_linux)]).decode().strip()

    line = json.dumps({"gds": win_path}) + "\n"  # one JSON line
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(line.encode("utf-8"))
        s.shutdown(socket.SHUT_WR)               # signal EOF so Klive stops “receiving”
        s.settimeout(1.0)
        try:
            while s.recv(1024):                  # linger briefly to avoid races
                pass
        except Exception:
            pass

        
        