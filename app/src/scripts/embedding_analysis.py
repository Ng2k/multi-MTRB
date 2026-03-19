import os
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from src.utils import configure_logger, get_logger, settings

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(enable_json=False, log_level="INFO")
logger = get_logger().bind(module="embeddings_analysis")

class EmbeddingsAnalyzer:
    def __init__(self, feature_dir: str, train_csv: str, dev_csv: str):
        self.feature_dir = Path(feature_dir)
        self.splits = {"Train": Path(train_csv), "Dev": Path(dev_csv)}
        # Dimensions: RoBERTa(768) + mT5(512) = 1280
        self.rob_dim = 768
        self.mt5_dim = 512

    def _calculate_advanced_metrics(self, rob, mt5):
        """Advanced analysis: Inter-stream correlation and redundancy."""
        # Ensure same row count for correlation (trimming if necessary)
        min_dim = min(rob.shape[1], mt5.shape[1])

        # 1. Cosine Similarity between streams per segment
        cos = torch.nn.functional.cosine_similarity(rob[:, :min_dim], mt5[:, :min_dim], dim=1)
        avg_cos = cos.mean().item()

        # 2. Variance Analysis (Dead dimensions)
        combined = torch.cat([rob, mt5], dim=1)
        variances = torch.var(combined, dim=0)
        dead_dims = torch.sum(variances < 1e-6).item()

        return round(avg_cos, 4), int(dead_dims)

    def _analyze_single_file(self, pid: str, label: int):
        """Worker to analyze dual-stream features with advanced diagnostics."""
        feat_path = self.feature_dir / f"{pid}_FEATURES.pt"
        if not feat_path.exists():
            return {"pid": pid, "found": False}

        try:
            feats = torch.load(feat_path, map_location="cpu", weights_only=True)
            rob_part = feats[:, :self.rob_dim]
            mt5_part = feats[:, self.rob_dim:]

            avg_cos, dead_dims = self._calculate_advanced_metrics(rob_part, mt5_part)

            return {
                "pid": pid,
                "label": "Depressed" if label == 1 else "Healthy",
                "lines": feats.shape[0],
                "size_mb": round(os.path.getsize(feat_path) / (1024 * 1024), 2),
                "rob_mean": round(rob_part.mean().item(), 4),
                "mt5_mean": round(mt5_part.mean().item(), 4),
                "sparsity": round((feats == 0).float().mean().item(), 4),
                "inter_stream_sim": avg_cos,
                "dead_dims": dead_dims,
                "found": True
            }
        except Exception as e:
            logger.error("Analysis failed", pid=pid, error=str(e))
            return {"pid": pid, "found": False}

    def run_analysis(self):
        reports = {}
        for name, csv_path in self.splits.items():
            if not csv_path.exists(): continue
            df = pd.read_csv(csv_path)
            id_col = next(c for c in df.columns if c.lower() in ['participant_id', 'id'])
            label_col = next(c for c in df.columns if 'phq' in c.lower() and 'binary' in c.lower())

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda x: self._analyze_single_file(str(int(x[0])), x[1]), 
                    df[[id_col, label_col]].values
                ))

            valid = [r for r in results if r["found"]]
            reports[name] = {
                "sessions": valid,
                "missing": len(results) - len(valid),
                "avg_sim": np.mean([r["inter_stream_sim"] for r in valid]) if valid else 0,
                "avg_sparsity": np.mean([r["sparsity"] for r in valid]) if valid else 0,
                "total_dead": sum([r["dead_dims"] for r in valid]),
                "avg_lines": int(np.mean([r["lines"] for r in valid])) if valid else 0,
                "total_size_mb": round(sum([r["size_mb"] for r in valid]), 2)
            }
        return reports

    def create_dashboard(self, reports, output_path=settings.artifacts/"embeddings_report.html"):
        train = reports.get("Train", {})
        all_sessions = train.get("sessions", []) + reports.get("Dev", {}).get("sessions", [])

        # --- Visualization ---
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Stream Magnitude (RoBERTa vs mT5)", "Inter-Stream Cosine Similarity", 
                            "Sequence Depth Distribution", "Dimensional Health (Dead Dims)"),
            vertical_spacing=0.15, horizontal_spacing=0.1
        )

        fig.add_trace(go.Scatter(x=[s['rob_mean'] for s in all_sessions], y=[s['mt5_mean'] for s in all_sessions], 
                                 mode='markers', marker=dict(color='#6366f1', size=9, opacity=0.6)), 1, 1)
        fig.add_trace(go.Violin(y=[s['inter_stream_sim'] for s in all_sessions], line_color='#10b981', name="Similarity"), 1, 2)
        fig.add_trace(go.Histogram(x=[s['lines'] for s in all_sessions], marker_color='#f43f5e'), 2, 1)
        fig.add_trace(go.Box(y=[s['dead_dims'] for s in all_sessions], marker_color='#fbbf24', name="Dead Dims"), 2, 2)

        fig.update_layout(template="plotly_white", height=750, showlegend=False, margin=dict(t=60, b=20, l=20, r=20))

        # --- Report Table ---
        table_rows = "".join([f"""
            <tr>
                <td><span class='badge'>{s['pid']}</span></td>
                <td style="color: {'#f43f5e' if s['label'] == 'Depressed' else '#10b981'}; font-weight:800;">{s['label']}</td>
                <td>{s['lines']}</td>
                <td>{s['inter_stream_sim']:.4f}</td>
                <td>{s['dead_dims']}</td>
                <td>{s['sparsity']*100:.1f}%</td>
            </tr>""" for s in all_sessions])

        # --- UI Construction ---
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Embedding Intelligence | Multi-MTRB</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
            <style>
                :root {{ --primary: #6366f1; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #10b981; }}
                body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; display: flex; }}
                .sidebar {{ width: 320px; background: #020617; height: 100vh; padding: 2rem; border-right: 1px solid #334155; position: fixed; overflow-y: auto; }}
                .content {{ margin-left: 380px; padding: 3rem; max-width: 1200px; width: 100%; }}
                .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }}
                .card {{ background: var(--card); padding: 1.2rem; border-radius: 1rem; border: 1px solid #334155; }}
                .val {{ font-size: 1.8rem; font-weight: 800; color: var(--primary); }}
                .lab {{ font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
                .chart-box {{ background: white; padding: 1.5rem; border-radius: 1rem; margin-bottom: 2rem; color: #0f172a; }}
                .legend-item {{ margin-bottom: 1.5rem; border-left: 3px solid var(--primary); padding-left: 10px; }}
                .legend-title {{ font-weight: 800; font-size: 0.8rem; color: var(--text); display: block; }}
                .legend-desc {{ font-size: 0.75rem; color: #64748b; line-height: 1.4; }}
                table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 1rem; overflow: hidden; }}
                th {{ text-align: left; padding: 1rem; background: #334155; font-size: 0.75rem; color: #94a3b8; }}
                td {{ padding: 1rem; border-top: 1px solid #334155; font-size: 0.85rem; }}
                .badge {{ background: #334155; padding: 2px 8px; border-radius: 4px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2 style="color: var(--primary)">EMBED-STUDY</h2>
                <div class="legend-item">
                    <span class="legend-title">Cosine Similarity</span>
                    <span class="legend-desc">Measures redundancy between RoBERTa & mT5. Low similarity (<0.3) is ideal as it means streams are providing different perspectives.</span>
                </div>
                <div class="legend-item">
                    <span class="legend-title">Dead Dimensions</span>
                    <span class="legend-desc">Count of features with zero variance. High counts indicate the embedding model is "collapsed" or over-regularized.</span>
                </div>
                <div class="legend-item">
                    <span class="legend-title">Sparsity</span>
                    <span class="legend-desc">The ratio of zero-values in the tensor. High sparsity (>50%) might suggest signal loss during extraction.</span>
                </div>
            </div>
            <div class="content">
                <h1>Feature Extraction Intelligence</h1>
                <div class="kpi-grid">
                    <div class="card"><div class="lab">Stream Overlap (Sim)</div><div class="val">{train['avg_sim']:.3f}</div></div>
                    <div class="card"><div class="lab">Global Sparsity</div><div class="val">{train['avg_sparsity'] * 100:.1f}%</div></div>
                    <div class="card"><div class="lab">Dead Dimensions</div><div class="val" style="color:#f43f5e">{train['total_dead']}</div></div>
                    <div class="card"><div class="lab">Avg Sequence Depth</div><div class="val">{train['avg_lines']}</div></div>
                    <div class="card"><div class="lab">Total Corpus Size</div><div class="val">{train['total_size_mb']}MB</div></div>
                    <div class="card"><div class="lab">Missing Sessions</div><div class="val" style="color:#fbbf24">{train['missing']}</div></div>
                </div>
                <div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                <h3>Session-Level Dual-Stream Breakdown</h3>
                <table>
                    <thead><tr><th>PID</th><th>Status</th><th>Lines</th><th>Stream Sim</th><th>Dead Dims</th><th>Sparsity</th></tr></thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </body>
        </html>
        """
        Path(output_path).write_text(html_content)
        logger.info("Embeddings report saved", path=output_path)

    def print_console_report(self, reports):
        """Standardized console output for quick auditing."""
        print(f"\n{' EMBEDDING ANALYSIS SUMMARY ':=^50}")
        for split, data in reports.items():
            print(f"[{split}]")
            print(f"  Sessions Processed : {len(data['sessions'])}")
            print(f"  Avg Inter-Stream Sim: {data['avg_sim']:.4f}")
            print(f"  Avg Sparsity       : {data['avg_sparsity'] * 100:.1f}%")
            print(f"  Total Dead Dims    : {data['total_dead']}")
            print(f"  Missing Files      : {data['missing']}")
        print("="*50)


if __name__ == "__main__":
    analyzer = EmbeddingsAnalyzer(feature_dir=str(settings.features),
                                  train_csv=str(settings.train_csv),
                                  dev_csv=str(settings.dev_csv))
    results = analyzer.run_analysis()
    analyzer.print_console_report(results)
    analyzer.create_dashboard(results)

