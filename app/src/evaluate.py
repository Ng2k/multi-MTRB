import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    fbeta_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import calibration_curve

from src.model.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils import settings, get_logger

logger = get_logger().bind(module="evaluate")

def mtrb_collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    x_batch = torch.stack(xs)
    y_batch = torch.stack(ys).unsqueeze(-1)
    mask = (x_batch.abs().sum(dim=-1, keepdim=True) != 0).float()
    return {"x": x_batch, "y": y_batch, "mask": mask}

def log_metrics_to_csv(metrics: dict, model_path: Path, output_dir: Path):
    """
    Appends the current run metrics to a centralized CSV tracker.
    Uses a standard naming convention: experiment_history.csv
    """
    csv_path = output_dir / "experiment_history.csv"

    row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_file": model_path.name,
        **metrics
    }

    df_new = pd.DataFrame([row])

    # Append to existing file or create new one
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    logger.info(f"Metrics logged", path=str(csv_path))

def create_interactive_plots(labels, probs, avg_weights):
    """Generates modern interactive Plotly charts with improved aesthetics."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)

    fig_perf = make_subplots(rows=1, cols=2, subplot_titles=("Precision-Recall (PR)", "Receiver Operating Characteristic (ROC)"))
    fig_perf.add_trace(go.Scatter(x=recall, y=precision, name=f"PR (AUC={auc(recall, precision):.2f})", 
                                 fill='tozeroy', line=dict(color='#6366f1', width=3)), row=1, col=1)
    fig_perf.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc_score(labels, probs):.2f})", 
                                 line=dict(color='#f59e0b', width=3)), row=1, col=2)
    fig_perf.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#94a3b8"), row=1, col=2)
    fig_perf.update_layout(height=450, template="plotly_white", margin=dict(t=50, b=50, l=50, r=50))

    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=5)
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='markers+lines', name='Model Confidence', line=dict(color='#10b981', width=3)))
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated', line=dict(dash='dash', color='#64748b')))
    fig_cal.update_layout(title="Probability Calibration (Reliability)", height=400, template="plotly_white")

    mean_attn = np.mean(avg_weights, axis=0)
    fig_attn = go.Figure()
    fig_attn.add_trace(go.Scatter(x=list(range(len(mean_attn))), y=mean_attn, fill='tozeroy', 
                                  line=dict(color='#8b5cf6', width=2), name="Attention Intensity"))
    fig_attn.update_layout(title="Temporal Attention Distribution", height=400, template="plotly_white")

    return {
        "perf": fig_perf.to_html(full_html=False, include_plotlyjs='cdn'),
        "cal": fig_cal.to_html(full_html=False, include_plotlyjs=False),
        "attn": fig_attn.to_html(full_html=False, include_plotlyjs=False)
    }

@torch.no_grad()
def run_evaluation(model_path: Path, dev_loader: DataLoader, device: str):
    best_params_path = settings.artifacts / "best_params.json"
    overrides = json.load(open(best_params_path)) if best_params_path.exists() else {}

    model = MultiMTRBClassifier(
        input_dim=settings.input_dim,
        hidden_dim=int(overrides.get("hidden_dim", 512)),
        n_heads=int(overrides.get("n_heads", 8))
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_logits, all_labels, all_weights = [], [], []
    for batch in dev_loader:
        x, mask, y = batch["x"].to(device), batch["mask"].to(device), batch["y"].to(device)
        logits, weights, _ = model(x, mask)
        all_logits.append(logits.cpu()); all_labels.append(y.cpu()); all_weights.append(weights.cpu())

    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_labels).numpy()
    avg_weights = torch.cat(all_weights).numpy()
    preds = (probs > 0.5).astype(float)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0.0), #type: ignore
        "Recall": recall_score(labels, preds),
        "F1-Score": f1_score(labels, preds),
        "F2-Score": fbeta_score(labels, preds, beta=2.0),
        "AUROC": roc_auc_score(labels, probs),
        "AUPRC": average_precision_score(labels, probs),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0
    }
    return metrics, labels, probs, preds, avg_weights

def generate_html_report(metrics, labels, probs, avg_weights, output_path: Path):
    plots = create_interactive_plots(labels, probs, avg_weights)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Clinical Intelligence Report | MTRB-AI</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{ --primary: #6366f1; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #10b981; }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; display: flex; }}
            .sidebar {{ width: 320px; background: #020617; height: 100vh; padding: 2rem; border-right: 1px solid #334155; position: fixed; overflow-y: auto; }}
            .content {{ margin-left: 380px; padding: 3rem; max-width: 1100px; width: 100%; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0; }}
            .card {{ background: var(--card); padding: 1.5rem; border-radius: 1rem; border: 1px solid #334155; position: relative; }}
            .val {{ font-size: 2.2rem; font-weight: 800; color: var(--primary); }}
            .lab {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
            .desc {{ font-size: 0.7rem; color: #64748b; margin-top: 8px; line-height: 1.2; }}
            .chart-box {{ background: white; padding: 1.5rem; border-radius: 1rem; margin-bottom: 2rem; color: #0f172a; }}
            .interpretation {{ background: #f1f5f9; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; border-left: 4px solid var(--primary); font-size: 0.9rem; }}
            .status-tag {{ background: var(--accent); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; color: #052e16; }}
            h2 {{ color: var(--primary); font-weight: 800; margin-top: 2.5rem; }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2 style="color: var(--primary); margin-top: 0;">MTRB-AI</h2>
            <p style="color: #64748b; font-size: 0.85rem;">Clinical Evaluation Dashboard for Depression Screening</p>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 1.5rem 0;">
            <div class="lab">Model Status</div>
            <p><span class="status-tag">Ready for Clinical Review</span></p>
            <div class="lab" style="margin-top: 1.5rem;">Data Export</div>
            <p style="font-size: 0.85rem; color: #cbd5e1;">Results are automatically logged to <code>experiment_history.csv</code> for tracking.</p>
        </div>
        <div class="content">
            <h1>Clinical Performance Insights</h1>
            <p style="color: #94a3b8;">Evaluation Results for the DAIC-WOZ Validation Set | {metrics['timestamp']}</p>
            <div class="metric-grid">
                <div class="card"><div class="lab">Accuracy</div><div class="val">{metrics['Accuracy']:.1%}</div><div class="desc">Overall correctness across both Healthy and Depressed classes.</div></div>
                <div class="card"><div class="lab">Recall (Sensitivity)</div><div class="val">{metrics['Recall']:.1%}</div><div class="desc">Critical for screening: What % of depressed patients did we identify?</div></div>
                <div class="card"><div class="lab">F2-Score</div><div class="val">{metrics['F2-Score']:.3f}</div><div class="desc">Medical-grade metric prioritizing Recall over Precision.</div></div>
                <div class="card"><div class="lab">Precision</div><div class="val">{metrics['Precision']:.1%}</div><div class="desc">Reliability of a positive flag.</div></div>
                <div class="card"><div class="lab">NPV</div><div class="val">{metrics['npv']:.1%}</div><div class="desc">Negative Predictive Value: Confidence when model says "Healthy".</div></div>
                <div class="card"><div class="lab">F1-Score</div><div class="val">{metrics['F1-Score']:.3f}</div><div class="desc">Harmonic mean of Precision and Recall.</div></div>
            </div>
            <h2>1. Discrimination Performance</h2>
            <div class="chart-box"> {plots['perf']} </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <div><h2>2. Model Reliability</h2><div class="chart-box">{plots['cal']}</div></div>
                <div><h2>3. Attention Saliency</h2><div class="chart-box">{plots['attn']}</div></div>
            </div>
        </div>
    </body>
    </html>
    """
    output_path.write_text(html_template)
    logger.info("Saved html report", path=str(output_path))

def main():
    best_model_path = settings.model_path / "mtrb_model.pt"
    dev_dataset = MultiMTRBDataset(settings.features, settings.dev_csv, settings.clean_data, settings.token_size)
    dev_loader = DataLoader(dev_dataset, batch_size=settings.batch_size, collate_fn=mtrb_collate_fn, shuffle=False)

    if not best_model_path.exists(): return logger.error("Model not found")

    metrics, labels, probs, preds, weights = run_evaluation(best_model_path, dev_loader, settings.device)

    # 1. Console Output
    print(f"\\n{' CLINICAL CONSOLE SUMMARY ':=^50}")
    print(f"Date: {metrics['timestamp']}")
    for m in ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "AUROC"]:
        print(f"{m:20}: {metrics[m]:.4f}")
    print(f"{'':-^50}")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))

    # 2. Log to CSV (History Tracking)
    log_metrics_to_csv(metrics, best_model_path, settings.artifacts)

    # 3. Generate HTML Report
    report_path = settings.artifacts / "clinical_report.html"
    generate_html_report(metrics, labels, probs, weights, report_path)
    logger.info(f"[SUCCESS] Modern Report and History Update complete.")

if __name__ == "__main__":
    main()

