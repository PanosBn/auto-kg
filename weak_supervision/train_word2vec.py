import pandas as pd
import numpy as np
import spacy
import re
import itertools
import string
import multiprocessing
import configparser
import itertools
import logging
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from string import punctuation
from time import time

logging.basicConfig(filename="trainw2vec.log", encoding="utf-8", level=logging.DEBUG)


class Word2VecWrapper:
    def __init__(self, file_path: str, config_path: str):
        self.file_path = file_path
        self.config = self.read_config(config_path)
        self.df = None
        self.nlp = spacy.load("nl_core_news_md", disable=["ner", "parser"])
        self.model = None

    @staticmethod
    def read_config(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def load_data(self):
        self.df = pd.read_csv(
            self.file_path, compression="gzip", delimiter=",", header=None
        )
        self.df.columns = ["CONTACT_TEXT"]

    def clean_data(self):
        t = time()
        cleaned_texts = [
            re.sub("[\(\[].*?[\)\]]", "", str(conversation))
            for conversation in self.df["CONTACT_TEXT"].values
        ]
        cleaned_texts = [re.sub("\.\.\.", "", str(sent)) for sent in cleaned_texts]
        cleaned_texts = [
            re.split(r"(?<=\w[?\.])\s", str(call)) for call in cleaned_texts
        ]
        cleaned_texts = list(itertools.chain(*cleaned_texts))
        cleaned_texts = [call.strip() for call in cleaned_texts]

        self.df = pd.DataFrame({"clean": cleaned_texts})
        self.df["Pass"] = self.df.progress_apply(
            lambda x: self.remove_short(x.clean), axis=1
        )
        self.df = (
            self.df[self.df["Pass"] != False]
            .drop("Pass", axis=1)
            .reset_index(drop=True)
        )
        self.df["clean"] = self.df["clean"].progress_apply(
            lambda x: "".join([i.lower() for i in x if i not in string.punctuation])
        )

        logging.debug(
            "Time to clean up everything: {} mins".format(round((time() - t) / 60, 2))
        )

    @staticmethod
    def remove_short(x: str) -> str:
        sentences = x.split(" ")
        if len(sentences) < 5:
            return False

    def lemmatize_and_remove_stopwords(self):
        self.df["clean"] = self.df.clean.progress_apply(
            lambda text: " ".join(
                token.text for token in self.nlp(text) if not token.is_stop
            )
        )

    def create_bigrams(self):
        sent = [row.split() for row in self.df["clean"][:1000000]]
        phrases = Phrases(sent, min_count=30, progress_per=10000)
        bigram = Phraser(phrases)
        self.sentences = bigram[sent]

    def train_word2vec_model(self):
        epochs_list = [int(e) for e in self.config["Word2Vec"]["epochs"].split(",")]
        window_sizes = [
            int(w) for w in self.config["Word2Vec"]["window_sizes"].split(",")
        ]
        vector_sizes = [
            int(v) for v in self.config["Word2Vec"]["vector_sizes"].split(",")
        ]

        for epochs, window_size, vector_size in itertools.product(
            epochs_list, window_sizes, vector_sizes
        ):
            logging.info(
                f"Training model with epochs={epochs}, window_size={window_size}, vector_size={vector_size}"
            )
            cores = multiprocessing.cpu_count()
            self.model = Word2Vec(
                min_count=20,
                window=window_size,
                vector_size=vector_size,
                sample=6e-5,
                alpha=0.03,
                min_alpha=0.0007,
                negative=20,
                workers=cores - 1,
            )

            t = time()
            self.model.build_vocab(self.sentences, progress_per=10000)
            logging.info(
                "Time to build vocab: {} mins".format(round((time() - t) / 60, 2))
            )

            t = time()
            self.model.train(
                self.sentences,
                total_examples=self.model.corpus_count,
                epochs=epochs,
                report_delay=1,
            )
            logging.info(
                "Time to train the model: {} mins".format(round((time() - t) / 60, 2))
            )

            model_filename = f"w2v_model_e{epochs}_w{window_size}_v{vector_size}.model"
            self.model.save(model_filename)
            logging.info(f"Model saved as {model_filename}")
