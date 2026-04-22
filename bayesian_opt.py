import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

def extract_physical_data():
    """Extract physical features if full_analysis doesn't exist."""
    print("Extracting physical data from raw files...")
    meta_path = os.path.join("Batteries", "master_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found.")
        
    df_meta = pd.read_csv(meta_path)
    results = []
    
    for _, row in df_meta.iterrows():
        try:
            raw_path = os.path.join("Batteries", row["raw_path"])
            raw = np.load(raw_path, allow_pickle=True)
            if isinstance(raw, np.ndarray) and raw.dtype == object:
                raw = raw.item()
            if "OCVdc" not in raw:
                continue
            data = raw["OCVdc"]
            v = data["v"]
            q = data["q"]
            t = data["t"]
            if len(q) < 5:
                continue
            i = np.gradient(q, t)
            p = v * i
            dod = np.max(q) - np.min(q)
            results.append({
                "raw_path": row["raw_path"],
                "voltage": np.mean(v),
                "current": np.mean(i),
                "dod": dod
            })
        except Exception:
            pass
            
    df_phys = pd.DataFrame(results)
    
    # Try to merge with any available results.csv
    res_path = os.path.join("outputs", "results", "results.csv")
    if os.path.exists(res_path):
        df_pred = pd.read_csv(res_path)
        df_merged = pd.merge(df_phys, df_pred, on="raw_path", how="inner")
        if "pred_rul" in df_merged.columns:
            return df_merged

    print("Could not find results.csv with pred_rul. Cannot train surrogate model.")
    return None


def run_bayesian_optimization():
    print("?? Starting Bayesian Optimization for Optimal Charge/Discharge Values")
    
    data_path = os.path.join("outputs", "optimization", "full_analysis.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print("Loaded data from outputs/optimization/full_analysis.csv")
    else:
        df = extract_physical_data()
        if df is None or len(df) == 0:
            print("Insufficient data for optimization.")
            return

    # Filter out outliers or weird values
    df = df[(df["voltage"] >= 2.5) & (df["voltage"] <= 4.2)]
    
    # We want to optimize Voltage, Current, and DoD to MAXIMIZE pred_rul.
    X = df[['voltage', 'current', 'dod']].values
    y = df['pred_rul'].values
    
    # 1. Surrogate Model
    surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
    surrogate.fit(X, y)
    print("? Surrogate model trained on historical data.")
    
    # 2. Define search space
    v_min, v_max = df['voltage'].min(), df['voltage'].max()
    c_min, c_max = df['current'].min(), df['current'].max()
    d_min, d_max = df['dod'].min(), df['dod'].max()
    
    space  = [
        Real(v_min, v_max, name='voltage'),
        Real(c_min, c_max, name='current'),
        Real(d_min, d_max, name='dod')
    ]
    
    # 3. Objective function (scikit-optimize minimizes, so return negative RUL)
    @use_named_args(space)
    def objective(voltage, current, dod):
        rul_pred = surrogate.predict([[voltage, current, dod]])[0]
        return -rul_pred 
        
    print(f"?? Searching optimal parameters over 50 iterations...")
    res = gp_minimize(objective, space, n_calls=50, random_state=42)
    
    best_v, best_c, best_dod = res.x
    best_rul = -res.fun
    
    print("\n=========================================")
    print("      BAYESIAN OPTIMIZATION RESULT       ")
    print("=========================================")
    print(f" Optimal Voltage:  {best_v:.4f} V")
    print(f" Optimal Current:  {best_c:.4f} A")
    print(f" Optimal DoD:      {best_dod:.4f}")
    print(f" Predicted Max RUL:{best_rul:.4f}")
    print("=========================================\n")
    
    # 4. Save Outputs
    out_dir = os.path.join("final_outputs", "bayesian_optimization")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "optimal_parameters.txt"), "w") as f:
        f.write("BAYESIAN OPTIMIZATION RESULT\n")
        f.write("==============================\n")
        f.write(f"Optimal Voltage:  {best_v:.4f} V\n")
        f.write(f"Optimal Current:  {best_c:.4f} A\n")
        f.write(f"Optimal DoD:      {best_dod:.4f}\n")
        f.write(f"Predicted Max RUL: {best_rul:.4f}\n")
        
    print(f"? Saved optimization results to {out_dir}")

if __name__ == '__main__':
    run_bayesian_optimization()
