"""
run_wavenet.py — All data loading is INSIDE train_wavenet() so this file
can be safely imported by main.py / generate_pkl_artifacts.py.
"""
import os, pickle
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings; warnings.filterwarnings("ignore")

SEQ_LEN = 200; EPOCHS = 60; BATCH_SIZE = 16; LR = 1e-3
BASE_OUTPUT_DIR = "final_outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resample(arr, length=SEQ_LEN):
    if len(arr) < 2: return np.zeros(length, dtype=np.float32)
    return interp1d(np.linspace(0,1,len(arr)), arr, kind="linear",
                    fill_value="extrapolate")(np.linspace(0,1,length)).astype(np.float32)

def create_dirs(name):
    out = os.path.join(BASE_OUTPUT_DIR, name)
    dirs = {k: os.path.join(out,k) for k in ["models","plots","summaries","metrics"]}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    return dirs

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ─── Model (importable without side-effects) ───────────────────────────────
class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv_f = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.conv_g = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.trim = pad; self.res = nn.Conv1d(channels, channels, 1)
    def forward(self, x):
        f = torch.tanh(self.conv_f(x)[...,:-self.trim] if self.trim>0 else self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x)[...,:-self.trim] if self.trim>0 else self.conv_g(x))
        z = f * g; return self.res(z) + x[...,:z.shape[-1]]

class WaveNetCNN(nn.Module):
    def __init__(self, input_size=3, channels=64, kernel_size=2, dilations=(1,2,4,8,16,32,64)):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, channels, 1)
        self.blocks = nn.ModuleList([WaveNetBlock(channels, kernel_size, d) for d in dilations])
        self.drop = nn.Dropout(0.2)
        self.soh_head = nn.Linear(channels,1); self.rul_head = nn.Linear(channels,1)
    def forward(self, x):
        x = self.input_proj(x.transpose(1,2))
        for b in self.blocks: x = b(x)
        z = self.drop(x.mean(-1))
        return self.soh_head(z).squeeze(-1), self.rul_head(z).squeeze(-1)

# ─── Data builder (reuse from run_idtcn to avoid duplication) ─────────────
def build_dataset(mat_file):
    from run_idtcn import build_dataset as _bd
    return _bd(mat_file)

def make_loader(X,Ys,Yr,shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X),torch.tensor(Ys),torch.tensor(Yr)),
                      batch_size=BATCH_SIZE, shuffle=shuffle)

# ─── Training ──────────────────────────────────────────────────────────────
def train_wavenet(mat_file="Oxford_Battery_Degradation_Dataset_1.mat"):
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    name = "WaveNet_CNN"; dirs = create_dirs(name)
    X,Y_s,Y_r,PHYS,cell_ids = build_dataset(mat_file)
    uc = np.unique(cell_ids); tr_c,te_c = train_test_split(uc,test_size=0.2,random_state=42)
    tm,em = np.isin(cell_ids,tr_c), np.isin(cell_ids,te_c)
    train_loader = make_loader(X[tm],Y_s[tm],Y_r[tm],True)
    test_loader  = make_loader(X[em],Y_s[em],Y_r[em])
    PHYS_te = PHYS[em]
    model = WaveNetCNN().to(DEVICE)
    with open(os.path.join(dirs["summaries"],"model_summary.txt"),"w") as f:
        f.write(str(model)+f"\n\nParams: {count_params(model):,}")
    opt = optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit = nn.L1Loss(); hl,hsoh,hrul = [],[],[]
    print(f"\nTraining {name}...")
    for ep in range(EPOCHS):
        model.train(); el=0
        for xb,ys,yr in train_loader:
            xb,ys,yr = xb.to(DEVICE),ys.to(DEVICE),yr.to(DEVICE); opt.zero_grad()
            ps,pr=model(xb); loss=0.7*crit(ps,ys)+0.3*crit(pr,yr)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); el+=loss.item()
        sched.step(); model.eval(); sp,rp,st,rt=[],[],[],[]
        with torch.no_grad():
            for xb,ys,yr in test_loader:
                ps,pr=model(xb.to(DEVICE)); sp.extend(ps.cpu().numpy()); rp.extend(pr.cpu().numpy())
                st.extend(ys.numpy()); rt.extend(yr.numpy())
        sm=mean_absolute_error(st,sp); rm=mean_absolute_error(rt,rp)
        hl.append(el/len(train_loader)); hsoh.append(sm); hrul.append(rm)
        if (ep+1)%10==0: print(f"  Ep[{ep+1}/{EPOCHS}] SOH={sm:.4f} RUL={rm:.4f}")
    torch.save(model.state_dict(), os.path.join(dirs["models"],"model.pth"))
    sr2=r2_score(st,sp); rr2=r2_score(rt,rp)
    md={"soh_mae":sm,"rul_mae":rm,"soh_r2":sr2,"rul_r2":rr2,"history_loss":hl,"history_soh_val":hsoh,"history_rul_val":hrul}
    with open(os.path.join(dirs["metrics"],"metrics.pkl"),"wb") as f: pickle.dump(md,f)
    pd.DataFrame({"true_soh":st,"pred_soh":sp,"true_rul":rt,"pred_rul":rp}).to_csv(os.path.join(dirs["metrics"],"predictions.csv"),index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4),facecolor="white")
    for ax in axes: ax.set_facecolor("white")
    axes[0].plot(hl,label="Loss"); axes[0].plot(hsoh,label="SOH MAE"); axes[0].plot(hrul,label="RUL MAE")
    axes[0].set_title(f"{name} Training"); axes[0].legend()
    axes[1].scatter(st,sp,alpha=0.5); axes[1].plot([0,1],[0,1],"r--"); axes[1].set_title(f"SOH R²={sr2:.3f}")
    axes[2].scatter(rt,rp,alpha=0.5,color="orange"); axes[2].plot([0,1],[0,1],"r--"); axes[2].set_title(f"RUL R²={rr2:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(dirs["plots"],"performance_summary.png"),dpi=150); plt.close()
    print("\n--- Bayesian Optimization ---")
    vi = ~np.isnan(PHYS_te).any(1) & ~np.isinf(PHYS_te).any(1)
    Pc=PHYS_te[vi]; rc=np.array(rp)[vi]
    sur=RandomForestRegressor(50,random_state=42); sur.fit(Pc,rc)
    sp2=[Real(Pc[:,i].min(),Pc[:,i].max(),name=n) for i,n in enumerate(["voltage","current","dod"])]
    @use_named_args(sp2)
    def obj(voltage,current,dod): return -sur.predict([[voltage,current,dod]])[0]
    res=gp_minimize(obj,sp2,n_calls=30,random_state=42); bv,bc,bd=res.x; br=-res.fun
    txt=f"BAYESIAN OPTIMIZATION RESULT\n{'='*30}\nVoltage:{bv:.4f}V\nCurrent:{bc:.4f}A\nDoD:{bd:.4f}\nMax RUL:{br:.4f}\n"
    print(txt); open(os.path.join(dirs["metrics"],"optimal_charging_parameters.txt"),"w").write(txt)
    print(f"[DONE] {name} → {os.path.abspath(dirs['models'])}"); return md

if __name__ == "__main__":
    import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
    train_wavenet()
