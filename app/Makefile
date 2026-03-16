# Ensure the python path includes the current directory so 'src' can be found
export PYTHONPATH := $(shell pwd)

.PHONY: clean-data  data_analysis extract rnd_search train evaluate xai

# Clean the raw transcripts
clean-data:
	python -m src.scripts.clean

# Execute data data analysis
data_analysis:
	python -m src.scripts.dataset_analysis

# Extract features (RoBERTa + mT5)
extract:
	python -m src.scripts.extract_features

# Execute the tuning of hyperparameters
rnd_search:
	python -m src.scripts.random_search

# Start the training pipeline
train:
	python -m src.scripts.train

# Run the clinical evaluation
evaluate:
	python -m src.scripts.evaluate

# Run script for the XAI
xai:
	python -m src.scripts.xai
