import os
import re
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from scipy import stats
import plotly.express as px

# Load environment variables
load_dotenv("../.env", override=True)

def generate_visual_report(raw_dir: Path, labels_csv: Path, output_html: str):
    # 1. Load Labels
    if not labels_csv.exists():
        print(f"Error: Labels file not found at {labels_csv}")
        return

    labels_df = pd.read_csv(labels_csv)
    id_col = 'Participant_ID' if 'Participant_ID' in labels_df.columns else labels_df.columns[0]
    target_col = 'PHQ8_Binary' if 'PHQ8_Binary' in labels_df.columns else 'PHQ_Binary'
    labels_map = labels_df.set_index(id_col)[target_col].to_dict()

    data = []

    # 2. Extract Data from Transcripts
    print(f"Analyzing transcripts in {raw_dir}...")
    for f_path in raw_dir.glob("*_TRANSCRIPT.csv"):
        p_id = f_path.name.split('_')[0]
        try:
            df_raw = pd.read_csv(f_path, sep='\t')
        except Exception as e:
            print(f"Skipping {f_path.name}: {e}")
            continue

        p_lines = df_raw[df_raw['speaker'] == 'Participant'].copy()
        i_lines = df_raw[df_raw['speaker'] != 'Participant'].copy()

        p_lines['value'] = p_lines['value'].fillna('').astype(str)
        text = " ".join(p_lines['value'])
        words = text.split()

        # --- Advanced Metrics ---
        # Vocabulary Richness: Type-Token Ratio (Unique words / Total words)
        unique_words = len(set(words))
        ttr = unique_words / len(words) if len(words) > 0 else 0

        # DAIC-WOZ Specific Tag Detection
        laughter_count = len(re.findall(r'<laughter>', text, re.I))
        scr_tags = len(re.findall(r'<[a-z_]+>', text, re.I)) - laughter_count 

        # Pause detection (Empty lines or explicit pause markers)
        pause_markers = len(re.findall(r'\(\.+\)', text)) 
        empty_lines = len(p_lines[p_lines['value'].str.strip() == ""])

        data.append({
            "ID": p_id,
            "Label": "Depressed" if labels_map.get(int(p_id)) == 1 else "Healthy",
            "Participant Utterances": len(p_lines),
            "Interviewer Utterances": len(i_lines),
            "Engagement Ratio (P/I)": round(len(p_lines) / len(i_lines), 2) if len(i_lines) > 0 else 0,
            "Avg Words/Line": round(len(words) / len(p_lines), 2) if len(p_lines) > 0 else 0,
            "Vocab Richness (TTR)": round(ttr, 3),
            "Laughter Tags": laughter_count,
            "Vocalizations (<scr_>)": scr_tags,
            "Empty/Pause Lines": empty_lines + pause_markers
        })

    if not data:
        print("No data extracted. Check your paths.")
        return

    df = pd.DataFrame(data).sort_values("ID")

    # --- 3. Statistical Analysis (T-Tests) ---
    metrics_to_test = ["Participant Utterances", "Avg Words/Line", "Vocab Richness (TTR)", "Laughter Tags"]
    p_values = {}
    for metric in metrics_to_test:
        group1 = df[df["Label"] == "Healthy"][metric]
        group2 = df[df["Label"] == "Depressed"][metric]
        _, p = stats.ttest_ind(group1, group2, nan_policy='omit')
        p_values[metric] = round(p, 4)

    # --- 4. Clinical Statistics Summary ---
    summary_stats = df.groupby("Label").agg({
        "Participant Utterances": "mean",
        "Avg Words/Line": "mean",
        "Vocab Richness (TTR)": "mean",
        "Laughter Tags": "mean",
        "Engagement Ratio (P/I)": "mean"
    }).round(3).reset_index()

    # --- 5. Visualizations ---
    fig_vocab = px.box(df, x="Label", y="Vocab Richness (TTR)", color="Label",
                       title="Vocabulary Diversity (Type-Token Ratio)", points="all",
                       template="plotly_white")

    fig_laugh = px.violin(df, x="Label", y="Laughter Tags", color="Label", box=True,
                          title="Laughter Distribution (Healthy vs Depressed)",
                          template="plotly_white")

    # --- 6. Generate HTML Report ---
    html_content = f"""
    <html>
    <head>
        <title>Advanced Clinical Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background-color: #f4f7f6; }}
            .container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .stat-box {{ border-left: 5px solid #2c3e50; background: #fafafa; padding: 15px; margin-bottom: 25px; }}
            .p-value-tag {{ font-size: 0.8em; color: #7f8c8d; }}
            .sig {{ color: #27ae60; font-weight: bold; }} /* Significant */
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background-color: #2c3e50; color: white; padding: 10px; }}
            td {{ border-bottom: 1px solid #eee; padding: 10px; text-align: center; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clinical Investigation Report: DAIC-WOZ</h1>

            <div class="stat-box">
                <h2>Group Mean Comparison & Statistical Significance</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Healthy (Avg)</th>
                        <th>Depressed (Avg)</th>
                        <th>P-Value (T-Test)</th>
                    </tr>
                    {"".join([f"<tr><td>{m}</td><td>{summary_stats.iloc[1][m]}</td><td>{summary_stats.iloc[0][m]}</td><td class='{'sig' if p_values.get(m, 1) < 0.05 else ''}'>{p_values.get(m, 'N/A')}</td></tr>" for m in metrics_to_test])}
                </table>
                <p><small>*P-values < 0.05 (green) indicate statistically significant differences between groups.</small></p>
            </div>

            <div class="grid">
                <div>{fig_vocab.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                <div>{fig_laugh.to_html(full_html=False, include_plotlyjs=False)}</div>
            </div>

            <h2>Individual Participant Data</h2>
            <div style="overflow-x:auto;">{df.to_html(index=False, classes='table')}</div>
        </div>
    </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report saved to {Path(output_html).absolute()}")

if __name__ == "__main__":
    generate_visual_report(
        raw_dir=Path(os.getenv("DATASET_RAW_DIR", "../dataset/raw")),
        labels_csv=Path(os.getenv("TRAIN_SPLIT", "../dataset/full_labels.csv")),
        output_html="clinical_data_report.html"
    )

