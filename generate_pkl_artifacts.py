"""
generate_pkl_artifacts.py
=========================
Run this ONCE locally to train both models on the Oxford dataset
and export all necessary artifacts for Render deployment.

Usage:
    python generate_pkl_artifacts.py

Outputs (commit the entire artifacts/ directory):
    artifacts/idtcn_model.pth        — ID-TCN weights
    artifacts/wavenet_model.pth      — WaveNet-CNN weights
    artifacts/global_scalers.pkl     — global v/q/T min-max bounds
    artifacts/metrics_idtcn.pkl      — training metrics + history
    artifacts/metrics_wavenet.pkl    — training metrics + history
    artifacts/bayesian_idtcn.pkl     — Bayesian opt result (ID-TCN)
    artifacts/bayesian_wavenet.pkl   — Bayesian opt result (WaveNet)
    artifacts/test_samples.pkl       — 12 raw test cycles for frontend demo
"""

import os, sys, io, pickle, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from oxford_extractor import extract_oxford_dataset
from run_idtcn import IDTCNModel, build_dataset, resample
from run_wavenet import WaveNetCNN

# ── Config ─────────────────────────────────────────────────────────────────
MAT_FILE   = "Oxford_Battery_Degradation_Dataset_1.mat"
ARTIFACTS  = "artifacts"
SEQ_LEN    = 200
EPOCHS     = 60
BATCH_SIZE = 16
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(ARTIFACTS, exist_ok=True)
np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

print("=" * 60)
print("  Enerlytics — Artifact Generation Script")
print("=" * 60)

# ── Step 1: Extract & preprocess dataset ───────────────────────────────────
print("\n[1/6] Extracting Oxford Dataset...")
X, Y_s, Y_r, PHYS, cell_ids, per_cell_scalers = build_dataset(MAT_FILE)
print(f"  Dataset shape: X={X.shape}, cells={np.unique(cell_ids)}")

with open(os.path.join(ARTIFACTS,"per_cell_scalers.pkl"),"wb") as f:
    pickle.dump(per_cell_scalers, f)
print(f"  Saved per_cell_scalers.pkl with {len(per_cell_scalers)} cells.")

# ── Step 2: Compute global scalers ─────────────────────────────────────────
print("\n[2/6] Computing global normalization scalers...")
# X shape: (N, 200, 3)  channels: [v, q, T]
global_scalers = {
    "v_min": float(X[:,:,0].min()), "v_max": float(X[:,:,0].max()),
    "q_min": float(X[:,:,1].min()), "q_max": float(X[:,:,1].max()),
    "T_min": float(X[:,:,2].min()), "T_max": float(X[:,:,2].max()),
}
with open(os.path.join(ARTIFACTS,"global_scalers.pkl"),"wb") as f:
    pickle.dump(global_scalers, f)
print(f"  Saved global_scalers.pkl: {global_scalers}")

# ── Step 3: Split data ─────────────────────────────────────────────────────
uc = np.unique(cell_ids)
tr_c, te_c = train_test_split(uc, test_size=0.2, random_state=42)
tm = np.isin(cell_ids, tr_c); em = np.isin(cell_ids, te_c)
X_tr,Ys_tr,Yr_tr = X[tm], Y_s[tm], Y_r[tm]
X_te,Ys_te,Yr_te = X[em], Y_s[em], Y_r[em]
PHYS_te = PHYS[em]

def make_loader(Xd,Ysd,Yrd,shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(Xd),torch.tensor(Ysd),torch.tensor(Yrd)),
                      batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_tr,Ys_tr,Yr_tr,True)
test_loader  = make_loader(X_te,Ys_te,Yr_te)

# ── Helper: run Bayesian optimization ──────────────────────────────────────
def run_bayesian(PHYS_te, rul_preds):
    vi = ~np.isnan(PHYS_te).any(1) & ~np.isinf(PHYS_te).any(1)
    Pc = PHYS_te[vi]; rc = np.array(rul_preds)[vi]
    sur = RandomForestRegressor(50, random_state=42); sur.fit(Pc, rc)
    sp = [Real(Pc[:,i].min(), Pc[:,i].max(), name=n)
          for i,n in enumerate(["voltage","current","dod"])]
    @use_named_args(sp)
    def obj(voltage, current, dod): return -sur.predict([[voltage,current,dod]])[0]
    res = gp_minimize(obj, sp, n_calls=30, random_state=42)
    bv, bc, bd = res.x; br = -res.fun
    return {"best_voltage": float(bv), "best_current": float(bc),
            "best_dod": float(bd), "best_rul": float(br)}

# ── Step 4: Train ID-TCN ───────────────────────────────────────────────────
print("\n[3/6] Training ID-TCN (60 epochs)...")
model_i = IDTCNModel().to(DEVICE)
opt_i = optim.Adam(model_i.parameters(), lr=LR, weight_decay=1e-5)
sched_i = optim.lr_scheduler.CosineAnnealingLR(opt_i, T_max=EPOCHS)
crit = nn.L1Loss(); hl_i,hs_i,hr_i = [],[],[]

for ep in range(EPOCHS):
    model_i.train(); el=0
    for xb,ys,yr in train_loader:
        xb,ys,yr = xb.to(DEVICE),ys.to(DEVICE),yr.to(DEVICE); opt_i.zero_grad()
        ps,pr=model_i(xb); loss=0.6*crit(ps,ys)+0.4*crit(pr,yr)
        loss.backward(); nn.utils.clip_grad_norm_(model_i.parameters(),1.0); opt_i.step(); el+=loss.item()
    sched_i.step(); model_i.eval(); sp,rp,st,rt=[],[],[],[]
    with torch.no_grad():
        for xb,ys,yr in test_loader:
            ps,pr=model_i(xb.to(DEVICE)); sp.extend(ps.cpu().numpy())
            rp.extend(pr.cpu().numpy()); st.extend(ys.numpy()); rt.extend(yr.numpy())
    sm=mean_absolute_error(st,sp); rm=mean_absolute_error(rt,rp)
    hl_i.append(el/len(train_loader)); hs_i.append(sm); hr_i.append(rm)
    if (ep+1)%10==0: print(f"  IDTCN Ep[{ep+1}/{EPOCHS}] SOH={sm:.4f} RUL={rm:.4f}")

torch.save(model_i.state_dict(), os.path.join(ARTIFACTS,"idtcn_model.pth"))
sr2_i=r2_score(st,sp); rr2_i=r2_score(rt,rp)
m_i={"soh_mae":float(sm),"rul_mae":float(rm),"soh_r2":float(sr2_i),"rul_r2":float(rr2_i),
     "history_loss":hl_i,"history_soh_val":hs_i,"history_rul_val":hr_i,
     "soh_trues":list(map(float,st)),"soh_preds":list(map(float,sp)),
     "rul_trues":list(map(float,rt)),"rul_preds":list(map(float,rp))}
with open(os.path.join(ARTIFACTS,"metrics_idtcn.pkl"),"wb") as f: pickle.dump(m_i,f)
print(f"  ID-TCN → SOH MAE={sm:.4f} R²={sr2_i:.3f} | RUL MAE={rm:.4f} R²={rr2_i:.3f}")

print("  Running Bayesian Opt for ID-TCN...")
bay_i = run_bayesian(PHYS_te, rp)
with open(os.path.join(ARTIFACTS,"bayesian_idtcn.pkl"),"wb") as f: pickle.dump(bay_i,f)
print(f"  Best → V={bay_i['best_voltage']:.4f} I={bay_i['best_current']:.4f} DoD={bay_i['best_dod']:.4f} RUL={bay_i['best_rul']:.4f}")

# ── Step 5: Train WaveNet-CNN ──────────────────────────────────────────────
print("\n[4/6] Training WaveNet-CNN (60 epochs)...")
model_w = WaveNetCNN().to(DEVICE)
opt_w = optim.Adam(model_w.parameters(), lr=LR, weight_decay=1e-5)
sched_w = optim.lr_scheduler.CosineAnnealingLR(opt_w, T_max=EPOCHS)
hl_w,hs_w,hr_w = [],[],[]

for ep in range(EPOCHS):
    model_w.train(); el=0
    for xb,ys,yr in train_loader:
        xb,ys,yr = xb.to(DEVICE),ys.to(DEVICE),yr.to(DEVICE); opt_w.zero_grad()
        ps,pr=model_w(xb); loss=0.7*crit(ps,ys)+0.3*crit(pr,yr)
        loss.backward(); nn.utils.clip_grad_norm_(model_w.parameters(),1.0); opt_w.step(); el+=loss.item()
    sched_w.step(); model_w.eval(); sp,rp,st,rt=[],[],[],[]
    with torch.no_grad():
        for xb,ys,yr in test_loader:
            ps,pr=model_w(xb.to(DEVICE)); sp.extend(ps.cpu().numpy())
            rp.extend(pr.cpu().numpy()); st.extend(ys.numpy()); rt.extend(yr.numpy())
    sm=mean_absolute_error(st,sp); rm=mean_absolute_error(rt,rp)
    hl_w.append(el/len(train_loader)); hs_w.append(sm); hr_w.append(rm)
    if (ep+1)%10==0: print(f"  WaveNet Ep[{ep+1}/{EPOCHS}] SOH={sm:.4f} RUL={rm:.4f}")

torch.save(model_w.state_dict(), os.path.join(ARTIFACTS,"wavenet_model.pth"))
sr2_w=r2_score(st,sp); rr2_w=r2_score(rt,rp)
m_w={"soh_mae":float(sm),"rul_mae":float(rm),"soh_r2":float(sr2_w),"rul_r2":float(rr2_w),
     "history_loss":hl_w,"history_soh_val":hs_w,"history_rul_val":hr_w,
     "soh_trues":list(map(float,st)),"soh_preds":list(map(float,sp)),
     "rul_trues":list(map(float,rt)),"rul_preds":list(map(float,rp))}
with open(os.path.join(ARTIFACTS,"metrics_wavenet.pkl"),"wb") as f: pickle.dump(m_w,f)
print(f"  WaveNet → SOH MAE={sm:.4f} R²={sr2_w:.3f} | RUL MAE={rm:.4f} R²={rr2_w:.3f}")

print("  Running Bayesian Opt for WaveNet...")
bay_w = run_bayesian(PHYS_te, rp)
with open(os.path.join(ARTIFACTS,"bayesian_wavenet.pkl"),"wb") as f: pickle.dump(bay_w,f)
print(f"  Best → V={bay_w['best_voltage']:.4f} I={bay_w['best_current']:.4f} DoD={bay_w['best_dod']:.4f} RUL={bay_w['best_rul']:.4f}")

# ── Step 6: Save test samples pkl ─────────────────────────────────────────
print("\n[5/6] Saving test samples for frontend upload demo...")
raw = extract_oxford_dataset(MAT_FILE)

# First, compute total cycles per cell for un-normalization
cell_total_cycles = {}
for cell_name in raw.keys():
    # Only count cycles that have the required discharge data
    count = 0
    for cyc in raw[cell_name].keys():
        if "C1dc" in raw[cell_name][cyc] and "q" in raw[cell_name][cyc]["C1dc"]:
            count += 1
    cell_total_cycles[cell_name] = count

with open(os.path.join(ARTIFACTS, "cell_cycles.pkl"), "wb") as f:
    pickle.dump(cell_total_cycles, f)

test_samples = []
# Pick just one cell for test samples, per user request
test_cell = te_c[0] if te_c else sorted(raw.keys())[0]

for cell_name in [test_cell]:
    if cell_name not in raw: continue
    cycles = sorted(raw[cell_name].keys())
    total_c = cell_total_cycles.get(cell_name, len(cycles))
    
    # Pick a variety of cycles (beginning, middle, end) to show degradation
    indices = [0, int(len(cycles)*0.2), int(len(cycles)*0.5), int(len(cycles)*0.8), len(cycles)-1]
    
    vi = 0
    for idx, cyc in enumerate(cycles):
        if len(test_samples) >= 10: break
        
        cap_val = None
        dc = raw[cell_name][cyc].get("C1dc", {})
        v, q, T = dc.get("v"), dc.get("q"), dc.get("T")
        
        if q is not None and len(q) >= 2:
            cap_val = abs(float(q[-1]) - float(q[0]))
        
        if cap_val is None or cap_val <= 1e-6:
            if cap_val is not None: vi += 1
            continue
            
        # Only add if it's one of our "interesting" indices or we need more samples
        if idx in indices or len(test_samples) < 5:
            c0 = None
            # Find first valid capacity for this cell
            for c_idx in cycles:
                d_c = raw[cell_name][c_idx].get("C1dc", {})
                q_c = d_c.get("q")
                if q_c is not None and len(q_c) >= 2:
                    c0 = abs(float(q_c[-1]) - float(q_c[0]))
                    break
            
            soh = float(cap_val) / c0
            rul_norm = (total_c - vi - 1) / max(total_c - 1, 1)
            rul_cycles = total_c - vi - 1
            
            test_samples.append({
                "cell": cell_name, "cycle": cyc,
                "v": list(map(float,v)), "q": list(map(float,q)), "T": list(map(float,T)),
                "soh_true": round(soh,4), 
                "rul_true_norm": round(rul_norm,4),
                "rul_true_cycles": int(rul_cycles),
                "total_cycles": int(total_c)
            })
        vi += 1

with open(os.path.join(ARTIFACTS,"test_samples.pkl"),"wb") as f: pickle.dump(test_samples,f)
print(f"  Saved {len(test_samples)} test samples to artifacts/test_samples.pkl")

# ── Done ───────────────────────────────────────────────────────────────────
print("\n[6/6] Artifact generation complete!")
print(f"\nFiles in {ARTIFACTS}/:")
for fn in os.listdir(ARTIFACTS): print(f"  {fn:40s}  {os.path.getsize(os.path.join(ARTIFACTS,fn))//1024} KB")
print("\nNow commit artifacts/ and push to GitHub → Render will deploy automatically.")
