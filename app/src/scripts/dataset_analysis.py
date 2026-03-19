import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from src.utils import configure_logger, get_logger, settings

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(enable_json=False, log_level="INFO")
logger = get_logger().bind(module="dataset_study")


def analyze_split(name, csv_path, feat_dir):
    """
    Performs deep analysis on a split, capturing global stats, 
    per-session metrics, and feature-level variance.
    """
    if not csv_path.exists():
        logger.error(f"{name} CSV not found", path=str(csv_path))
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Identify key columns
    id_col = next(c for c in df.columns if c.lower() in ['participant_id', 'id'])
    label_col = next(c for c in df.columns if 'phq' in c.lower() and 'binary' in c.lower())

    def process_row(row):
        """Worker to extract metadata and stats from a single participant session."""
        pid = str(int(row[id_col]))
        label = int(row[label_col])

        patterns = [f"{pid}_P.pt", f"{pid}_FEATURES.pt", f"{pid}.pt"]
        feat_file = next((feat_dir / p for p in patterns if (feat_dir / p).exists()), None)

        if not feat_file:
            feat_file = next(feat_dir.glob(f"{pid}_*.pt"), None)

        if feat_file and feat_file.exists():
            try:
                # Get disk size in MB
                file_size_mb = os.path.getsize(feat_file) / (1024 * 1024)
                # Load features (CPU only)
                features = torch.load(feat_file, map_location='cpu', weights_only=True)

                # Improvement: Calculate sparsity (percentage of zeros)
                sparsity = (features == 0).float().mean().item()

                return {
                    "pid": pid,
                    "label": "Depressed" if label == 1 else "Healthy",
                    "lines": features.shape[0],
                    "dim": features.shape[1],
                    "size_mb": round(file_size_mb, 2),
                    "mean": round(features.mean().item(), 4),
                    "std": round(features.std().item(), 4),
                    "sparsity": round(sparsity, 4),
                    "raw_features": features, # Kept for global variance analysis
                    "found": True
                }
            except Exception as e:
                logger.error(f"Corrupted file detected for PID {pid}: {e}")
                return {"pid": pid, "found": False}
        return {"pid": pid, "found": False}

    # Parallel processing for performance
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

    valid_sessions = [res for res in results if res["found"]]
    line_counts = [s["lines"] for s in valid_sessions]

    # Improvement: Global Feature Variance Analysis
    all_feats = torch.cat([s["raw_features"] for s in valid_sessions], dim=0)
    feat_variance = torch.var(all_feats, dim=0).numpy()
    dead_features = np.sum(feat_variance == 0)

    stats = {
        "name": name,
        "pids": set(df[id_col].astype(str)),
        "total": len(df),
        "raw_counts": df[label_col].value_counts().to_dict(),
        "avg_lines": int(np.mean(line_counts)) if valid_sessions else 0,
        "line_percentiles": np.percentile(line_counts, [25, 50, 75]) if valid_sessions else [0,0,0],
        "avg_size": round(float(np.mean([s["size_mb"] for s in valid_sessions])), 2) if valid_sessions else 0,
        "avg_sparsity": round(float(np.mean([s["sparsity"] for s in valid_sessions])), 4) if valid_sessions else 0,
        "dead_features": int(dead_features),
        "missing": len(results) - len(valid_sessions),
        "sessions": valid_sessions
    }
    return stats


def create_dashboard(reports, output_path=settings.artifacts/"dataset_report.html"):
    """Generates a modern Glassmorphism HTML dashboard with interactive charts."""
    train, dev = reports[0], reports[1]

    # Check for Cross-Split Leakage
    overlap = train['pids'].intersection(dev['pids'])
    leakage_status = f"CLEAN (0 Overlap)" if not overlap else f"WARNING: {len(overlap)} Shared IDs"

    # 1. Visualization Setup
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Class Balance (Train)", "Sequence Length Distribution", 
                        "Feature Mean vs. Sparsity", "Missing Data Analysis"),
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )

    # Pie Chart for Balance
    fig.add_trace(go.Pie(
        labels=['Healthy', 'Depressed'],
        values=[train['raw_counts'].get(0, 0), train['raw_counts'].get(1, 0)],
        hole=.4, marker=dict(colors=['#10b981', '#f43f5e'])
    ), 1, 1)

    # Histogram for Lines
    fig.add_trace(go.Histogram(
        x=[s['lines'] for s in train['sessions']], 
        marker_color='#6366f1', name="Lines"
    ), 1, 2)

    # Mean vs Sparsity Scatter
    fig.add_trace(go.Scatter(
        x=[s['mean'] for s in train['sessions']], 
        y=[s['sparsity'] for s in train['sessions']],
        mode='markers', marker=dict(color='#8b5cf6', size=8, opacity=0.7), name="Stability"
    ), 2, 1)

    # Bar chart for missing files
    fig.add_trace(go.Bar(
        x=['Train Missing', 'Dev Missing'], 
        y=[train['missing'], dev['missing']],
        marker_color=['#f43f5e', '#fbbf24']
    ), 2, 2)

    fig.update_layout(height=700, template="plotly_white", margin=dict(t=50, b=50, l=20, r=20), showlegend=False)

    # 2. Table Generation
    table_rows = ""
    for s in train['sessions'] + dev['sessions']:
        label_style = "color:#f43f5e; font-weight:bold" if s['label'] == "Depressed" else "color:#10b981"
        table_rows += f"""
        <tr>
            <td>{s['pid']}</td>
            <td style='{label_style}'>{s['label']}</td>
            <td>{s['lines']}</td>
            <td>{s['sparsity']:.2%}</td>
            <td>{s['size_mb']} MB</td>
            <td>{s['mean']}</td>
        </tr>"""

    # 3. HTML Assembly with Modern CSS
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Dataset Intelligence | DAIC-WOZ</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{ --primary: #6366f1; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #10b981; }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; display: flex; }}
            .sidebar {{ width: 300px; background: #020617; height: 100vh; padding: 2rem; border-right: 1px solid #334155; position: fixed; overflow-y: auto; }}
            .content {{ margin-left: 350px; padding: 3rem; max-width: 1100px; width: 100%; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
            .card {{ background: var(--card); padding: 1.2rem; border-radius: 1rem; border: 1px solid #334155; }}
            .val {{ font-size: 1.8rem; font-weight: 800; color: var(--primary); }}
            .lab {{ font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
            .chart-box {{ background: white; padding: 1rem; border-radius: 1rem; margin-bottom: 2rem; color: #0f172a; }}
            table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 1rem; overflow: hidden; }}
            th {{ text-align: left; padding: 1rem; background: #334155; font-size: 0.8rem; color: #94a3b8; }}
            td {{ padding: 1rem; border-top: 1px solid #334155; font-size: 0.85rem; }}
            .badge {{ padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: bold; background: #334155; }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2 style="color: var(--primary)">DATA-INSIGHT</h2>
            <p style="color: #64748b; font-size: 0.85rem;">DAIC-WOZ Corpus Analysis</p>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 1.5rem 0;">
            <div class="lab">Leakage Status</div>
            <p><span class="badge">{leakage_status}</span></p>
            <div class="lab" style="margin-top:1.5rem">Feature Health</div>
            <p style="font-size:0.85rem">Dead Dimensions: <b style="color:#f43f5e">{train['dead_features']}</b></p>
            <div class="lab" style="margin-top:1.5rem">Line Percentiles (Train)</div>
            <p style="font-size:0.85rem; color:#94a3b8">P25: {train['line_percentiles'][0]:.0f} | P50: {train['line_percentiles'][1]:.0f} | P75: {train['line_percentiles'][2]:.0f}</p>
        </div>
        <div class="content">
            <h1>Dataset Characteristics</h1>
            <div class="metric-grid">
                <div class="card"><div class="lab">Avg Lines</div><div class="val">{train['avg_lines']}</div></div>
                <div class="card"><div class="lab">Avg Sparsity</div><div class="val">{train['avg_sparsity']:.1%}</div></div>
                <div class="card"><div class="lab">Avg Size</div><div class="val">{train['avg_size']}MB</div></div>
                <div class="card"><div class="lab">Missing Files</div><div class="val" style="color:#f43f5e">{train['missing']}</div></div>
            </div>
            <div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            <h3>Session-Level Details</h3>
            <table>
                <thead><tr><th>ID</th><th>Status</th><th>Lines</th><th>Sparsity</th><th>Size</th><th>Mean Act.</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </body>
    </html>
    """
    Path(output_path).write_text(html_content)


def run_full_study():
    """Main execution entry point."""
    logger.info("Starting Enhanced Dataset Study...")
    train_report = analyze_split("Train", settings.train_csv, settings.features)
    dev_report = analyze_split("Dev", settings.dev_csv, settings.features)
    output_path = settings.artifacts/"dataset_report.html"

    if train_report and dev_report:
        create_dashboard([train_report, dev_report], output_path)
        logger.info("Detailed dashboard generated", path=str(output_path))


if __name__ == "__main__":
    run_full_study()

