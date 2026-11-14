"""ANALYSIS: how visible is occupant heat in the per-zone model residual? python src/occ_analyze.py <npz files written by occ_log.py>"""
import sys

import numpy as np

for f in sys.argv[1:]:
    d = np.load(f)
    R, TH, OCC, N, H = d["R"], d["TH"], d["OCC"], d["N"], d["H"]
    # implied extra heat (kW) = residual (K/step) / response (K per kW per step)
    th = np.where(np.abs(TH) > 1e-4, TH, np.nan)
    P = R / th
    day = (H >= 8) & (H < 19) & (N > 0)
    occ = OCC > 0
    rows = []
    for z in range(R.shape[1]):
        a = P[day & occ[:, z], z]
        b = P[day & ~occ[:, z], z]
        if len(a) > 20 and len(b) > 20:
            rows.append((np.nanmean(a) - np.nanmean(b), np.nanstd(P[day, z]), np.nanmean(OCC[day & occ[:, z], z])))
    rows = np.array(rows)
    print(f"{f.split('/')[-1]:34s} zones={R.shape[1]} n_comp={len(rows)} median dP(occ-empty)={np.median(rows[:,0])*1000:.0f} W "
          f"median sd={np.median(rows[:,1])*1000:.0f} W  mean occupants when occ={np.mean(rows[:,2]):.2f}  "
          f"theta5 median={np.nanmedian(TH[-1]):.4f}")
    # step response at departures: residual change 6 steps after a zone goes from occupied to empty, afternoons
    ch = []
    for z in range(R.shape[1]):
        o = occ[:, z]
        idx = np.where(o[:-1] & ~o[1:] & (H[1:] >= 13) & (H[1:] < 19))[0]
        for i in idx:
            if i >= 6 and i + 8 < len(R):
                ch.append((np.nanmean(P[i + 3:i + 9, z]) - np.nanmean(P[i - 6:i, z])) * 1000)
    if ch:
        ch = np.array(ch)
        print(f"   departures={len(ch)} power change after departure: median {np.median(ch):.0f} W, "
              f"frac < -50W {np.mean(ch < -50):.2f}")
