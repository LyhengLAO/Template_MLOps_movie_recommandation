#!/bin/bash

dvc stage add -n import \
    -d src/data/import_raw_data.py \
    -o data/raw/ \
    python src/data/import_raw_data.py

dvc stage add -n prepare \
    -d src/data/make_dataset.py -d data/raw/ \
    -o data/processed/ \
    python src/data/make_dataset.py data/raw data/processed

dvc stage add -n train \
    -d src/models/train_model.py -d data/processed/ \
    -o models/model.pkl \
    python src/models/train_model.py

dvc stage add -n evaluate \
    -d src/models/evalue.py -d models/ -d data/processed/ \
    -M metrics/metrics.json \
    python src/models/evalue.py

dvc repro
git add metrics/metrics.json dvc.lock dvc.yaml .gitignore
git commit -m "DVC pipeline"
dvc push
git push
