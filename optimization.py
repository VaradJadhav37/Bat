import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_optimization(data_dir, metadata_csv, save_dir):

    print("?? Running optimization analysis...")

    df_meta = pd.read_csv(metadata_csv)

    results = []

    # ---------------- EXTRACT PHYSICS ----------------
    for _, row in df_meta.iterrows():

        try:
            raw_path = os.path.join(data_dir, row["raw_path"])
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
                "power": np.mean(p),
                "dod": dod
            })

        except Exception as e:
            print(f"?? Skipping file: {e}")

    df_phys = pd.DataFrame(results)

    # ---------------- LOAD MODEL RESULTS ----------------
    results_path = os.path.join(os.path.dirname(save_dir), "results", "results.csv")

    if not os.path.exists(results_path):
        raise FileNotFoundError("results.csv not found. Run evaluation first.")

    df_pred = pd.read_csv(results_path)

    # merge using raw_path
    df = pd.merge(df_pred, df_phys, on="raw_path", how="inner")

    # ---------------- CONSTRAINTS ----------------
    df_filtered = df[
        (df["voltage"] >= 2.5) &
        (df["voltage"] <= 4.2) &
        (df["current"].abs() >= 0.1) &
        (df["current"].abs() <= 2.0) &
        (df["power"] > 0.5)
    ]

    print(f"? Total valid points after constraints: {len(df_filtered)}")

    # ---------------- FIND OPTIMAL REGION ----------------
    best = df_filtered.sort_values("pred_rul", ascending=False).head(50)

    best_path = os.path.join(save_dir, "best_operating_points.csv")
    best.to_csv(best_path, index=False)

    print(f"? Best operating points saved at: {best_path}")

    # ---------------- DOD vs RUL ----------------
    plt.figure()
    plt.scatter(df_filtered["dod"], df_filtered["pred_rul"])
    plt.xlabel("Depth of Discharge (DoD)")
    plt.ylabel("Predicted RUL")
    plt.title("RUL vs DoD")
    plt.savefig(os.path.join(save_dir, "dod_vs_rul.png"))
    plt.close()

    # ---------------- POWER vs RUL ----------------
    plt.figure()
    plt.scatter(df_filtered["power"], df_filtered["pred_rul"])
    plt.xlabel("Power")
    plt.ylabel("Predicted RUL")
    plt.title("RUL vs Power")
    plt.savefig(os.path.join(save_dir, "power_vs_rul.png"))
    plt.close()

    # ---------------- VOLTAGE-CURRENT MAP ----------------
    plt.figure()
    plt.scatter(df_filtered["current"], df_filtered["voltage"], c=df_filtered["pred_rul"])
    plt.colorbar(label="RUL")
    plt.xlabel("Current")
    plt.ylabel("Voltage")
    plt.title("Operating Region (RUL colored)")
    plt.savefig(os.path.join(save_dir, "operating_region.png"))
    plt.close()

    # ---------------- SAVE FULL DATA ----------------
    full_path = os.path.join(save_dir, "full_analysis.csv")
    df_filtered.to_csv(full_path, index=False)

    print(f"? Full analysis saved at: {full_path}")
    print(f"?? Optimization completed successfully")