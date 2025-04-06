#!/bin/bash

dvc repro
dvc push

git add metrics/metrics.json dvc.lock dvc.yaml .gitignore
git commit -m "Updated DVC pipeline"
dvc push
git push