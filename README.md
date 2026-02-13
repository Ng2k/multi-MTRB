# Multi-MTRB: Depression Detection in Clinical Interviews

Multi-Instance Learning with mT5 & RoBERTa Fusion on the DAIC-WOZ Dataset
----------------------------------------------------------------------------------------------------

This repository contains a production-grade implementation of a Multi-Instance Learning (MIL) framework for diagnosing Depression (MDD) from clinical interview transcripts. Inspired by the [multi-MTRB architecture](https://pubmed.ncbi.nlm.nih.gov/39994325/), the system leverages an ensemble of [mT5](https://arxiv.org/pdf/2010.11934) and [RoBERTa encoders](https://arxiv.org/pdf/1907.11692) to capture both semantic nuance and structural dialogue cues.

The project is designed for the DAIC-WOZ (Distress Analysis Interview Corpus), treating each interview as a "bag" of utterances.

## Table of Content

- [Multi-MTRB: Depression Detection in Clinical Interviews](#multi-mtrb-depression-detection-in-clinical-interviews)
	- [Multi-Instance Learning with mT5 \& RoBERTa Fusion on the DAIC-WOZ Dataset](#multi-instance-learning-with-mt5--roberta-fusion-on-the-daic-woz-dataset)
	- [Table of Content](#table-of-content)
	- [Repository Structure \& Design Rationale](#repository-structure--design-rationale)
	- [Technical Architecture](#technical-architecture)
		- [Model Design](#model-design)
		- [System Flow](#system-flow)
	- [Engineering \& Reproducibility](#engineering--reproducibility)
	- [Metrics \& Evaluation](#metrics--evaluation)
	- [Extensions (Experimental)](#extensions-experimental)
	- [Responsible AI \& Bioinformatics Standards](#responsible-ai--bioinformatics-standards)

## Repository Structure & Design Rationale
The repository is architected following modular separation of concerns, ensuring that the high-stakes logic of model architecture is decoupled from volatile data-handling and experimental configurations.

```
├── configs/                # Centralized YAML files for hyperparameter versioning
├── data/                   # Local storage (Git-ignored) for DAIC-WOZ transcripts
├── models/                 
│   ├── encoder_branch.py   # Specialized Transformer wrappers (mT5/RoBERTa)
│   ├── mil_head.py         # Reusable Gated Attention MIL pooling logic
│   └── fusion.py           # Multi-modal fusion and classification heads
├── utils/                  
│   ├── preprocessing.py    # Automated cleaning and participant diarization
│   └── logger.py           # Observability
├── environment.yaml        # Deterministic Conda environment specification
├── train.py                # Main execution engine for remote training
|-- evaluate.py				# Evaluation script
├── mock_data_gen.py        # Pipeline validation utility for pre-dataset access
└── README.md               # Technical specification and operational manual
```

## Technical Architecture

### Model Design

The architecture is bifurcated into two specialized feature extraction branches to maximize representational diversity:

- **RoBERTa Branch**: Optimized for discriminative semantic features and sentiment-laden language patterns.
- **mT5 Encoder Branch**: Utilizes the encoder segment of the T5 architecture to capture structural and contextual dependencies within multi-lingual or complex dialogue schemas.
- **Attention-based MIL Pooling**: Rather than simple averaging, a gated attention mechanism learns a scoring function ai​ for each utterance, effectively identifying "High-Diagnostic Value" segments.
- **Late Fusion**: Final classification is performed on the concatenated, attention-weighted embeddings of both branches.

### System Flow
1. **Preprocessing**: Tokenization and diarization filter (Focus on Participant turns).
2. **Embedding**: Utterance-level feature extraction via dual transformers.
3. **Aggregation**: Attention MIL maps N instances → 1 Bag representation.
4. **Classification**: Binary Cross-Entropy (BCE) with Logits for depression probability.

## Engineering & Reproducibility
To ensure binary compatibility for CUDA kernels and deep learning dependencies, we use Conda environments.

```bash
# Setup environment
conda env create -f environment.yaml
conda activate depression-mil

# Verify GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

## Metrics & Evaluation
In alignment with the reference paper, the model's performance is evaluated on:

- **Primary Metric**: Macro F1-Score (due to class imbalance in DAIC-WOZ).
- **Secondary Metrics**: Precision, Recall, and Area Under the ROC Curve (AUC).
- **Explainability**: Heatmaps of attention weights mapped back to patient transcripts to identify clinical markers.

## Extensions (Experimental)
- [ ] **XAI**: Use of SHAP or LIME for explainability.
- [ ] **Mental-RoBERTa**: Replacing RoBERTa with domain-specific encoders.
- [ ] **Temporal Gating**: Exploring RNN-MIL to account for the chronological progression of interviews.
- [ ] **Ordinal Regression**: Predicting raw PHQ-8 scores instead of binary labels.

## Responsible AI & Bioinformatics Standards
- **Bias Mitigation**: The model evaluation pipeline includes a check for performance parity across demographic subsets (if available in DAIC-WOZ metadata).
- **Interpretability**: Attention weights are exported to identify which phrases (e.g., "lack of sleep," "loss of interest") the model prioritizes, allowing for clinical validation.
- **Reproducibility**: All experiments are versioned with fixed random seeds and logged hyperparameters.