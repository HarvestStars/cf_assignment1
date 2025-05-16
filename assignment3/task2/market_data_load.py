import numpy as np

raw = np.load("../raw_data/raw_ivol_surfaces.npy", allow_pickle=True).item()
interp = np.load("../raw_data/interp_ivol_surfaces.npy", allow_pickle=True).item()

date = "2023 11 01"

raw_vols = raw[date]['vols']
interp_vols = interp[date]['vols']

print("raw vols:", raw_vols)
print("interp vols:", interp_vols)