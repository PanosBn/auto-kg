#!/bin/bash


CONFIG_PATH="need to add this path"

# Run the Python script for Relation Extraction training
python -c "
from relation_classification import FlairRelationTrainer

relation_trainer = FlairRelationTrainer('$CONFIG_PATH')
relation_trainer.train_models()
"

echo "Relation extraction model training complete."
