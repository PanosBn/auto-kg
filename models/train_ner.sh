#!/bin/bash

CONFIG_PATH="please add this first!"

python -c "
from sequence_labeling import FlairNERTrainer

ner_trainer = FlairNERTrainer('$CONFIG_PATH')
ner_trainer.train_models()
"

echo "Entity Recognition model training complete :D !"
