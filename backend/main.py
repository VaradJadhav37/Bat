"""
main.py — Enerlytics Battery Diagnostics API
============================================
Flask entry point for Render deployment.
Loads pre-trained artifacts and serves predictions via REST API.
"""

import os, pickle, io
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scipy.interpolate import interp1d
import warnings; warnings.filterwarnings("ignore")

from run_idtcn import IDTCNModel
from run_wavenet import WaveNetCNN

app = Flask(__name__)
CORS(app)

ARTIFACTS = "artifacts"
SEQ_LEN   = 200
DEVICE    = torch.device("cpu")

# ── Artifact store ─────────────────────────────────────────────────────────
_store = {}

def load_artifacts():
    required = [
        "idtcn_model.pth", "wavenet_model.pth",
        "metrics_idtcn.pkl", "metrics_wavenet.pkl",
        "bayesian_idtcn.pkl", "bayesian_wavenet.pkl"
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ARTIFACTS,f))]
    if missing:
        print(f"[WARN] Missing artifacts: {missing}")
        return False

    pcs_path = os.path.join(ARTIFACTS,"per_cell_scalers.pkl")
    if os.path.exists(pcs_path):
        with open(pcs_path,"rb") as f:
            _store["scalers"] = pickle.load(f)
    else:
        _store["scalers"] = {}
    with open(os.path.join(ARTIFACTS,"metrics_idtcn.pkl"),"rb") as f:
        _store["metrics_idtcn"] = pickle.load(f)
    with open(os.path.join(ARTIFACTS,"metrics_wavenet.pkl"),"rb") as f:
        _store["metrics_wavenet"] = pickle.load(f)
    with open(os.path.join(ARTIFACTS,"bayesian_idtcn.pkl"),"rb") as f:
        _store["bayesian_idtcn"] = pickle.load(f)
    with open(os.path.join(ARTIFACTS,"bayesian_wavenet.pkl"),"rb") as f:
        _store["bayesian_wavenet"] = pickle.load(f)

    # Load cell cycle metadata for un-normalization
    cc_path = os.path.join(ARTIFACTS, "cell_cycles.pkl")
    if os.path.exists(cc_path):
        with open(cc_path, "rb") as f:
            _store["cell_cycles"] = pickle.load(f)

    m_i = IDTCNModel()
    m_i.load_state_dict(torch.load(os.path.join(ARTIFACTS,"idtcn_model.pth"), map_location=DEVICE))
    m_i.eval(); _store["model_idtcn"] = m_i

    m_w = WaveNetCNN()
    m_w.load_state_dict(torch.load(os.path.join(ARTIFACTS,"wavenet_model.pth"), map_location=DEVICE))
    m_w.eval(); _store["model_wavenet"] = m_w

    print("[OK] All artifacts loaded successfully.")
    return True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def _safe_resample(arr, length=SEQ_LEN):
    arr = np.array(arr, dtype=np.float32)
    if len(arr) < 2: return np.zeros(length, dtype=np.float32)
    x_old = np.linspace(0,1,len(arr)); x_new = np.linspace(0,1,length)
    return interp1d(x_old, arr, kind="linear", fill_value="extrapolate")(x_new).astype(np.float32)

def _self_normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _preprocess(v_raw, q_raw, T_raw, cell_name=None):
    scalers = _store.get("scalers", {}).get(cell_name) if cell_name else None
    
    if scalers:
        v_min, v_max = scalers['v_min'], scalers['v_max']
        q_min, q_max = scalers['q_min'], scalers['q_max']
        T_min, T_max = scalers['T_min'], scalers['T_max']
        
        v = _safe_resample(v_raw)
        q = _safe_resample(q_raw)
        T = _safe_resample(T_raw)
        
        v = (v - v_min) / max(v_max - v_min, 1e-8)
        q = (q - q_min) / max(q_max - q_min, 1e-8)
        T = (T - T_min) / max(T_max - T_min, 1e-8)
    else:
        v = _self_normalize(_safe_resample(v_raw))
        q = _self_normalize(_safe_resample(q_raw))
        T = _self_normalize(_safe_resample(T_raw))

    seq = np.stack([v, q, T], axis=1)
    return torch.tensor(seq[np.newaxis], dtype=torch.float32)

def _run_model(model, tensor):
    with torch.no_grad():
        soh, rul = model(tensor)
    return float(soh.item()), float(rul.item())

def _clamp01(v): return max(0.0, min(1.0, v))

def _to_cycles(rul_norm, cell_name=None, total_cycles=None):
    t = total_cycles
    if t is None and "scalers" in _store and cell_name in _store["scalers"]:
        t = _store["scalers"][cell_name].get("total_cycles")
    if t is None: t = 8200 # Default to 8200 for Oxford
    return int(round(_clamp01(rul_norm) * t))

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return jsonify({"message": "Enerlytics API is running", "docs": "/health"}), 200

@app.route("/health")
def health():
    ready = "model_idtcn" in _store and "model_wavenet" in _store
    return jsonify({"status": "ok" if ready else "loading", "models_loaded": ready}), 200

def _safe_metrics(m):
    import math
    def _f(v):
        try:
            f = float(v)
            return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 6)
        except: return 0.0
    return {
        "soh_mae": _f(m.get("soh_mae", 0)), "rul_mae": _f(m.get("rul_mae", 0)),
        "soh_r2": _f(m.get("soh_r2", 0)), "rul_r2": _f(m.get("rul_r2", 0)),
        "history_loss": [_f(x) for x in m.get("history_loss", [])],
        "history_soh_val": [_f(x) for x in m.get("history_soh_val", [])],
        "history_rul_val": [_f(x) for x in m.get("history_rul_val", [])],
    }

@app.route("/metrics")
def metrics():
    if "metrics_idtcn" not in _store: return jsonify({"error": "No data"}), 503
    return jsonify({"idtcn": _safe_metrics(_store["metrics_idtcn"]), "wavenet": _safe_metrics(_store["metrics_wavenet"])})

@app.route("/bayesian")
def bayesian():
    if "bayesian_idtcn" not in _store: return jsonify({"error": "No data"}), 503
    cell = request.args.get('cell')
    res = {
        "idtcn": dict(_store["bayesian_idtcn"]),
        "wavenet": dict(_store["bayesian_wavenet"])
    }
    if cell and "scalers" in _store and cell in _store["scalers"]:
        total_cyc = _store["scalers"][cell]["total_cycles"]
        res["idtcn"]["best_rul_cycles"] = int(res["idtcn"]["best_rul"] * total_cyc)
        res["wavenet"]["best_rul_cycles"] = int(res["wavenet"]["best_rul"] * total_cyc)
    return jsonify(res)

@app.route("/samples")
def get_samples():
    path = os.path.join(ARTIFACTS, "test_samples.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f: return jsonify(pickle.load(f))
    return jsonify([])

@app.route("/upload", methods=["POST"])
def upload_file():
    if not request.content_type or "multipart" not in request.content_type:
        return jsonify({"error": "Expected multipart/form-data"}), 400
    f = request.files.get("file")
    if f is None: return jsonify({"error": "No file uploaded"}), 400
    try:
        data = pickle.load(io.BytesIO(f.read()))
        if not isinstance(data, list):
            data = [data]
        # Return the parsed samples so the frontend can populate the dropdown
        return jsonify(data)
    except Exception as e:
        print(f"Error parsing upload: {e}")
        return jsonify({"error": "Failed to parse file"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "model_idtcn" not in _store: return jsonify({"error": "Loading"}), 503
    v_raw = q_raw = T_raw = None
    soh_true = rul_true_norm = rul_true_cycles = cell_name = total_cycles = None
    model_sel = "both"

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("file")
        if f is None: return jsonify({"error": "No file"}), 400
        data = pickle.load(io.BytesIO(f.read()))
        if isinstance(data, list) and len(data) > 0: data = data[0]
        v_raw, q_raw, T_raw = data["v"], data["q"], data["T"]
        soh_true = data.get("soh_true")
        rul_true_norm = data.get("rul_true_norm") or data.get("rul_true")
        rul_true_cycles = data.get("rul_true_cycles")
        cell_name = data.get("cell"); total_cycles = data.get("total_cycles")
        model_sel = request.form.get("model", "both")
    else:
        body = request.get_json(silent=True)
        if not body: return jsonify({"error": "No JSON"}), 400
        v_raw, q_raw, T_raw = body.get("v"), body.get("q"), body.get("T")
        model_sel = body.get("model", "both")
        soh_true = body.get("soh_true")
        rul_true_norm = body.get("rul_true_norm") or body.get("rul_true")
        rul_true_cycles = body.get("rul_true_cycles")
        cell_name = body.get("cell"); total_cycles = body.get("total_cycles")

    tensor = _preprocess(v_raw, q_raw, T_raw, cell_name)
    result = {}
    def _get_res(k):
        s, r = _run_model(_store[k], tensor)
        return {"soh": round(_clamp01(s),4), "rul_norm": round(_clamp01(r),4), "rul_cycles": _to_cycles(r, cell_name, total_cycles)}

    if model_sel in ("idtcn", "both"): result["idtcn"] = _get_res("model_idtcn")
    if model_sel in ("wavenet", "both"): result["wavenet"] = _get_res("model_wavenet")
    if model_sel == "both":
        s_e = (result["idtcn"]["soh"] + result["wavenet"]["soh"]) / 2
        r_e = (result["idtcn"]["rul_norm"] + result["wavenet"]["rul_norm"]) / 2
        result["ensemble"] = {"soh": round(s_e,4), "rul_norm": round(r_e,4), "rul_cycles": _to_cycles(r_e, cell_name, total_cycles)}

    # Compute current physical parameters for the sample
    mv = float(np.nanmean(v_raw))
    ta = np.array(T_raw, dtype=np.float64)
    qa = np.array(q_raw, dtype=np.float64)
    tr = ta[-1] - ta[0] if len(ta) > 1 else 0.0
    mi = float(np.nanmean(np.gradient(qa, ta) if tr > 1e-8 else np.gradient(qa)))
    dod = float(np.max(qa) - np.min(qa))
    
    result["current_phys"] = {
        "voltage": round(mv, 4),
        "current": round(mi, 4),
        "dod": round(dod, 4)
    }

    result.update({"soh_true": soh_true, "rul_true_norm": rul_true_norm, "rul_true_cycles": rul_true_cycles, "cell": cell_name, "n_points": len(v_raw)})
    return jsonify(result)

@app.route("/test_sample/<int:idx>")
def test_sample(idx):
    with open(os.path.join(ARTIFACTS, "test_samples.pkl"),"rb") as f: samples = pickle.load(f)
    return jsonify(samples[idx])

@app.route("/test_samples/count")
def test_samples_count():
    tp = os.path.join(ARTIFACTS,"test_samples.pkl")
    if not os.path.exists(tp): return jsonify({"count":0})
    with open(tp,"rb") as f: samples = pickle.load(f)
    return jsonify({"count": len(samples), "samples": [{"cell":s["cell"],"cycle":s["cycle"],"soh_true":s["soh_true"],"rul_true_norm":s.get("rul_true_norm") or s.get("rul_true"), "rul_true_cycles": s.get("rul_true_cycles")} for s in samples]})

with app.app_context(): load_artifacts()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
