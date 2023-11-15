#!/bin/bash

CONFIG_PATH="./config.ini"
DATA_FILE_PATH="data_for_word2vec.gz"

python -c "
from your_python_script import Word2VecModelCreator

model_creator = Word2VecModelCreator('$DATA_FILE_PATH', '$CONFIG_PATH')
model_creator.load_data()
model_creator.clean_data()
model_creator.lemmatize_and_remove_stopwords()
model_creator.create_bigrams()
model_creator.train_word2vec_model()
"

echo "Word2Vec training is complete!! Yay!"