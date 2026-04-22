import os, pickle, numpy as np
from oxford_extractor import extract_oxford_dataset
from run_idtcn import build_dataset

MAT_FILE = "Oxford_Battery_Degradation_Dataset_1.mat"
ARTIFACTS = "artifacts"

print("Regenerating scalers (with actual max cycle count)...")
# Use build_dataset to get the new scalers dict with max_cyc
_, _, _, _, _, scalers = build_dataset(MAT_FILE)

with open(os.path.join(ARTIFACTS, "per_cell_scalers.pkl"), "wb") as f:
    pickle.dump(scalers, f)

print("Regenerating test samples with RAW data and absolute cycles...")
raw = extract_oxford_dataset(MAT_FILE)
test_samples = []

for cell_name in sorted(raw.keys()):
    cycles = sorted(raw[cell_name].keys())
    total_c = scalers[cell_name]["total_cycles"]
    
    # Pick 3 interesting cycles
    indices = [0, len(cycles)//2, len(cycles)-1]
    
    # We need to find the c0 for this cell to compute SOH
    c0 = None
    for cyc in cycles:
        q = raw[cell_name][cyc].get("C1dc", {}).get("q")
        if q is not None and len(q) >= 2:
            c0 = abs(float(q[-1]) - float(q[0]))
            break
            
    for idx, cyc in enumerate(cycles):
        dc = raw[cell_name][cyc].get("C1dc", {})
        v, q, T = dc.get("v"), dc.get("q"), dc.get("T")
        if q is not None and len(q) >= 2:
            cap = abs(float(q[-1]) - float(q[0]))
            if idx in indices:
                curr_cyc = int(cyc.replace("cyc",""))
                test_samples.append({
                    "cell": cell_name, "cycle": cyc,
                    "v": list(map(float, v)), "q": list(map(float, q)), "T": list(map(float, T)),
                    "soh_true": round(cap/c0, 4),
                    "rul_true_cycles": int(total_c - curr_cyc),
                    "total_cycles": int(total_c)
                })

with open(os.path.join(ARTIFACTS, "test_samples.pkl"), "wb") as f:
    pickle.dump(test_samples, f)

print("Done! per_cell_scalers.pkl and test_samples.pkl updated with absolute cycles.")
