import os
import configparser
import logging
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier
from flair.models.relation_classifier_model import TypedEntityMarker
from flair.trainers import ModelTrainer
import pickle

logging.basicConfig(filename="trainREL.log", encoding="utf-8", level=logging.DEBUG)


class FlairRelationTrainer:
    def __init__(self, config_path: str, entity_label_map_path=None):
        self.config = self.read_config(config_path)
        self.corpus = self.setup_corpus()
        if entity_label_map_path:
            self.entity_label_map = self.load_entity_label_map(entity_label_map_path)

    def read_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def setup_corpus(self):
        # Even though we are doing relation classification, we need to have a column for NER.
        # This is because the relation classifier uses the NER labels to identify the entities.
        # The relations themselves are located in a comment row in the CoNLL-U file.
        # P.S Check the README.md for more information on the CoNLL-U format.
        columns = {1: "text", 2: "ner"}
        data_folder = os.getcwd()
        corpus = ColumnCorpus(
            data_folder,
            columns,
            train_file="TRAINING_FILE.conllu",
            dev_file="DEV_FILE.conllu",
            test_file="TEST_FILE.conllu",
        )
        return corpus

    def load_entity_label_map(self, entity_label_map_path: str):
        with open(entity_label_map_path, "rb") as file:
            return pickle.load(file)

    def train_models(self):
        if self.entity_label_map is None:
            entity_label_map_path = self.config["RelationExtraction"][
                "entity_label_map_path"
            ]
            self.entity_label_map = self.load_entity_label_map(entity_label_map_path)

        models = self.config["RelationExtraction"]["models"].split(",")
        learning_rate = float(self.config["RelationExtraction"]["learning_rate"])
        max_epochs = int(self.config["RelationExtraction"]["max_epochs"])
        mini_batch_size = int(self.config["RelationExtraction"]["mini_batch_size"])
        output_folder = self.config["RelationExtraction"]["output_folder"]
        fine_tune = self.config["RelationExtraction"].getboolean("fine_tune")
        fine_tune_layers = self.config["RelationExtraction"]["fine_tune_layers"]

        for model_name in models:
            try:
                embeddings = TransformerDocumentEmbeddings(
                    model=str(model_name),
                    layers=fine_tune_layers,
                    fine_tune=fine_tune,
                    model_max_length=512,
                )
                label_dictionary = self.corpus.make_label_dictionary("relation")
                model = RelationClassifier(
                    embeddings=embeddings,
                    label_dictionary=label_dictionary,
                    label_type="relation",
                    entity_label_types="ner",
                    entity_pair_labels=self.entity_label_map,
                    allow_unk_tag=True,
                    cross_augmentation=True,
                    encoding_strategy=TypedEntityMarker(),
                )
                trainer = ModelTrainer(
                    model=model, corpus=model.transform_corpus(self.corpus)
                )

                model_output_dir = os.path.join(
                    output_folder, f"relation_model_{model_name.replace('/', '_')}"
                )
                os.makedirs(model_output_dir, exist_ok=True)

                trainer.fine_tune(
                    model_output_dir,
                    max_epochs=max_epochs,
                    learning_rate=learning_rate,
                    mini_batch_size=mini_batch_size,
                    main_evaluation_metric=("macro avg", "f1-score"),
                    reduce_transformer_vocab=False,
                )

                logging.info(
                    f"Relation classification model trained and saved in {model_output_dir}"
                )
            except Exception as e:
                logging.error(f"Error during training {model_name}: {str(e)}")
