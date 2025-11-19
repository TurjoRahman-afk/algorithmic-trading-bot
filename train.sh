#!/bin/zsh
# Shortcut script to train ML models for selected symbols
PYTHONPATH=$(pwd) python3 -m scripts.train_ml_model "$@"
