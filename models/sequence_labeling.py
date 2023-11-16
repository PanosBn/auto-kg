import os
import configparser
import logging
import flair
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import OneCycleLR

logging.basicConfig(filename="trainNER.log", encoding="utf-8", level=logging.DEBUG)


class FlairNERTrainer:
    def __init__(self, config_path):
        self.config = self.read_config(config_path)
        self.corpus = self.setup_corpus()

    def read_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def setup_corpus(self):
        columns = {0: "text", 1: "ner"}
        data_folder = os.getcwd()
        corpus = ColumnCorpus(
            data_folder,
            columns,
            train_file="TRAINING_FILE_SHOULD_GO_HERE",
            dev_file="DEV_FILE_SHOULD_GO_HERE",
            test_file="TEST_FILE_SHOULD_GO_HERE.bioes",
            document_separator_token="-DOCSTART-",
        )
        corpus.filter_empty_sentences()
        return corpus

    def train_models(self):
        models = self.config["NER_Training"]["models"].split(",")
        learning_rate = float(self.config["NER_Training"]["learning_rate"])
        max_epochs = int(self.config["NER_Training"]["max_epochs"])
        mini_batch_size = int(self.config["NER_Training"]["mini_batch_size"])
        output_folder = self.config["NER_Training"]["output_folder"]
        fine_tune = self.config["NER_Training"].getboolean("fine_tune")
        fine_tune_layers = self.config["NER_Training"]["fine_tune_layers"]

        for model_name in models:
            try:
                embeddings = TransformerWordEmbeddings(
                    model=str(model_name),
                    layers=fine_tune_layers,
                    fine_tune=fine_tune,
                    model_max_length=512,
                )
                tag_dictionary = self.corpus.make_label_dictionary("ner")
                model = SequenceTagger(
                    hidden_size=256,
                    embeddings=embeddings,
                    tag_dictionary=tag_dictionary,
                    tag_type="ner",
                    use_crf=False,
                    use_rnn=False,
                    reproject_embeddings=False,
                )
                trainer = ModelTrainer(model=model, corpus=self.corpus)

                model_output_dir = os.path.join(
                    output_folder, f"trained_model_{model_name.replace('/', '_')}"
                )
                os.makedirs(model_output_dir, exist_ok=True)

                trainer.fine_tune(
                    model_output_dir,
                    max_epochs=max_epochs,
                    learning_rate=learning_rate,
                    mini_batch_size=mini_batch_size,
                    scheduler=OneCycleLR,
                    main_evaluation_metric=("macro avg", "f1-score"),
                )

                logging.info(f"Model trained and saved in {model_output_dir}")
            except Exception as e:
                logging.error(f"Error during training {model_name}: {str(e)}")
