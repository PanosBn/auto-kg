import spacy
import csv
import logging
import yaml
import os
import random
import pandas as pd
import datetime
from spacy.tokens import Doc, Span
from spacy.training import iob_to_biluo
from typing import Dict, List, Tuple
from itertools import combinations

logging.basicConfig(filename="parseLabelbox.log", encoding="utf-8", level=logging.DEBUG)


class EntityRelationshipExtractor:
    def __init__(
        self,
        nlp_model: str,
        entity_mapping: Dict[str, str],
        relationship_mapping: Dict[str, str],
        entity_locations_file: str,
        relationships_file: str,
    ):
        self.nlp = spacy.blank(nlp_model)
        self.entity_mapping = entity_mapping
        self.relationship_mapping = relationship_mapping
        self.entity_locations_file = entity_locations_file
        self.relationships_file = relationships_file
        self.annotated_spacy_docs = list()

        Doc.set_extension("rel", default=list(), force=True)
        Doc.set_extension("rel_ready", default=list(), force=True)
        Span.set_extension("span_rels", default=list(), force=True)

        self.entity_lv1_lv2 = self.load_mapping(
            "path_to_entity_naming_map", delimiter="\t"
        )
        self.rel_lv1_lv2 = self.load_mapping(
            "path_To_relationship_naming_map", delimiter=","
        )

        self.set_of_rels = set()

        self.rel_ontology_mapping = {}
        with open("data/jaap_mapping.txt") as file:
            lines = file.readlines()
            for line in lines:
                line = line.split("-")
                subclass = line[0].strip()
                new_subclass = (" ").join(subclass.split(" ")).lower()
                main_class = line[1].strip().lower()
                # print(f'From {new_subclass} to -> {main_class}')
                self.rel_ontology_mapping[new_subclass.lower()] = main_class.lower()
                logging.debug(
                    f"Number of rels in ontology mapping: {len(self.rel_ontology_mapping)}"
                )

    @staticmethod
    def gather_entity_info(ent_id: str, df: pd.DataFrame):
        """
        A dictionary of sorts, that holds info about the entities in our dataset. Careful tho, you have to instatiate and populate the dataframe first.
        """
        ent = df[df["id"] == ent_id]

        start = ent.start.values[0]
        end = ent.end.values[0]
        type = ent.type.values[0]

        return start, end, type

    @staticmethod
    def apply_doc_level_annotations(
        self, annotated_docs: List[Doc], ent_data: pd.DataFrame
    ) -> List[Doc]:
        """
        Spacy is super nice and allows us to add custom attributes to the Doc object. We can use this as an additional annotatio layer to store the relationships in the doc object itself.
        There are many annotation levels for Spacy Tokens, Spans and (in our case) Docs. We can use the doc level annotations to store the relationships.
        See here for more: https://spacy.io/api/doc
        """
        if ent_data:
            for doc in annotated_docs:
                ents = list(doc.ents)
                ent_ids = [ent.id_ for ent in ents]
                rels = doc._.rel

                ent_pair_combinations = list(combinations(ent_ids, 2))
                count = 0
                rel_ready_dict = {}
                for key in rels:
                    subject, object, relation = rels.get(key)
                    s_o_tuple = tuple((subject, object))

                    for pair in ent_pair_combinations:
                        if pair == s_o_tuple:
                            count += 1

                            (
                                subject_start,
                                subject_end,
                                _,
                            ) = self.__class__.gather_entity_info(subject, ent_data)
                            (
                                object_start,
                                object_end,
                                _,
                            ) = self.__class__.gather_entity_info(object, ent_data)

                            relation = "_".join(relation.split(" "))
                            rel_ready_dict[
                                key
                            ] = f"{subject_start+1};{subject_end+1};{object_start+1};{object_end+1};{relation}"

                doc._.rel_ready = rel_ready_dict
                logging.debug(f"Number of relationships in doc: {len(doc._.rel_ready)}")
        else:
            ValueError("No entity data provided")

    def load_mapping(self, file_path: str, delimiter: str) -> dict:
        mapping = {}
        with open(file_path, "r", encoding="utf-8") as file:
            if delimiter == "\t":
                for line in file:
                    to, frm = line.strip().split(delimiter)
                    mapping[frm] = to
            else:  # Assuming CSV format
                reader = csv.reader(file, delimiter=delimiter)
                for line in reader:
                    to, frm = line[5].lower().strip(), line[0].strip()
                    mapping[frm] = to
        return mapping

    def _load_yaml_config() -> dict:
        config_file = os.getenv("CONFIG_YAML_PATH")
        if not config_file:
            raise ValueError("CONFIG_YAML_PATH environment variable not set")
        with open(config_file, "r") as file:
            return yaml.safe_load(file)

    def extract_ent_rel_pairs(
        self, dataset: List[Dict]
    ) -> List[Tuple[str, List[Tuple], List[Tuple]]]:
        prepared_dataset = list()  # List[str,List,List]
        rel_tuples = (
            list()
        )  # store relationships here momentarily and use to populate [entity pair]-[relation] in doc._.rel
        ent_spans_locations = (
            list()
        )  # store entity id and entity span locations s.t. we can retrieve them later
        total_rels = list()

        for idx, item in enumerate(dataset):
            total_ents = list()

            doc_id = item["ID"]
            text = item["Labeled Data"]
            ents = item["Label"]["objects"]
            doc = self.nlp(text)
            ent_spans = []
            for ent in ents:
                start_pos = ent["data"]["location"]["start"]
                end_pos = ent["data"]["location"]["end"]
                name = ent["title"]

                repl_name = self.entity_lv1_lv2.get(
                    name
                )  # apply lvl1 to lvl2 entity mapping as provided by Jaap

                if not isinstance(repl_name, type(None)):
                    name = repl_name
                ent_id = ent["featureId"]
                ent_text = text[start_pos : end_pos + 1]

                total_ents.append((ent_id, name, start_pos, end_pos))

                span = doc.char_span(
                    start_pos,
                    end_pos,
                    label=name,
                    alignment_mode="expand",
                    kb_id=ent_id,
                )
                span.id_ = ent_id
                ent_spans.append(
                    Span(doc, span.start, span.end, name, span_id=span.id_)
                )

                ent_spans_locations.append(
                    [span.id_, span.start, span.end - 1, name, ent_text]
                )

            try:
                doc.set_ents(ent_spans)
            except:
                print(f"Item {idx} failed")
                print(text)

            self.annotated_spacy_docs.append(doc)

            with open(self.entity_locations_file, "w", newline="") as ent1:
                csv_writer = csv.writer(ent1)
                csv_writer.writerows(ent_spans_locations)

            relationships = item["Label"]["relationships"]
            rel_dict = {}
            for rel in relationships:
                contents = rel["data"].keys()

                subject = rel["data"]["source"]
                object = rel["data"]["target"]

                if "label" in contents:
                    name = rel["data"]["label"]
                    self.set_of_rels.add(name.lower())
                    # print(name.lower())
                repl_name = self.rel_ontology_mapping.get(name.lower())
                if not isinstance(repl_name, type(None)):
                    name = repl_name  # Change relationship name to a higher one using Jaaps mapping scheme
                else:
                    name = "action_misc"

                rel_id = rel["featureId"]
                total_rels.append((rel_id, name, subject, object, text, doc_id))

                rel_dict[rel_id] = (subject, object, name)

            doc._.rel = rel_dict

            for rel in total_rels:
                rel_tuples.append(tuple((rel[2], rel[3])))

            prepared_dataset.append((text, total_ents, total_rels))

        with open(self.relationships_file, "w", newline="") as rel1:
            csv_writer = csv.writer(rel1)
            csv_writer.writerows(total_rels)


def generate_training_data(annotated_spacy_docs: List[Doc]) -> List[str]:
    random.seed(42).shuffle(annotated_spacy_docs)

    split_1 = int(0.65 * len(annotated_spacy_docs))
    split_2 = int(0.80 * len(annotated_spacy_docs))
    train_filenames = annotated_spacy_docs[:split_1]
    dev_filenames = annotated_spacy_docs[split_1:split_2]
    test_filenames = annotated_spacy_docs[split_2:]

    logging.info(
        f"Share of data, Training: {len(train_filenames)}, Dev: {len(dev_filenames)}, Test: {len(test_filenames)}."
    )

    current_time = datetime.datetime.now()
    today = str(current_time.day) + "_" + str(current_time.month)

    export_training_data(train_filenames, f"train_{today}", mode="BIO")
    export_training_data(dev_filenames, f"dev_{today}", mode="BIO")
    export_training_data(test_filenames, f"test_{today}", mode="BIO")
    logging.info(f"Exported training")


def export_training_data(annotated_spacy_docs: List[Doc], split: str, mode="conllu"):
    """
    Exports the training data in the (BIOES)[https://stackoverflow.com/questions/17116446/what-do-the-bilou-tags-mean-in-named-entity-recognition]
    or [CONLLU](https://universaldependencies.org/format.html) format.
    """

    document = list()

    if mode == "conllu":
        logging.info(f"Output type: {mode}")
        document.append("# global.columns = " + ("\t").join(["id", "text", "ner"]))

    for doc in annotated_spacy_docs:
        word_positions_dict = dict()
        current_document = []

        tags = [
            token.ent_iob_ + "-" + "_".join((token.ent_type_.split(" ")))
            if token.ent_iob_ != "O"
            else "O"
            for token in doc
        ]

        biluo_tags = iob_to_biluo(tags)

        for i, tag in enumerate(biluo_tags):
            if tag.startswith("U-"):
                biluo_tags[i] = tag.replace("U-", "S-")
            if tag.startswith("L-"):
                biluo_tags[i] = tag.replace("L-", "E-")

        tags = biluo_tags

        if mode == "bio":
            document.append("-DOCSTART-" + "\t" + "-X-")
            document.append("\n\n")

        elif mode == "conllu":
            document.append("")
            text = "# text = " + doc.text
            rels = "# relations = " + ("|").join([*doc._.rel_ready.values()])

            rels_flat = list([doc._.rel_ready.values()][0])

        sentence = "# text = "
        tag_sanity_check = set()

        start_position, end_position = 0, 0
        end_of_prev_sentence = 0

        ner_lines = list()
        rel_lines = list()
        counter = 1
        for i, word_tag in enumerate(zip(doc, tags)):
            if mode == "conllu":
                word_positions_dict[i] = word_tag[0]
                line = ("\t").join([str(counter), str(word_tag[0]), str(tags[i])])
                counter += 1

                tag_sanity_check.add(str(word_tag[1]))
                ner_lines.append(line)

                if str(word_tag[0]) == "." or str(word_tag[0]) == "?":
                    end_of_prev_sentence = i - end_of_prev_sentence
                    start_position = list(word_positions_dict.keys())[0]
                    end_position = list(word_positions_dict.keys())[-1]

                    current_document = doc[start_position : end_position + 1]
                    rel_flag = False  # whether relations exist for this sentence
                    for rel in rels_flat:
                        # Get the positions of the entities {h: head, t: tail} in the relation and the relation itself.
                        h1, h2, t1, t2, relation = rel.split(";")[:5]

                        if (
                            (int(h1) >= int(start_position))
                            and (int(h2) <= int(end_position))
                        ) and ((int(t2) <= int(end_position))):
                            h1 = int(h1) - start_position
                            h2 = int(h2) - start_position
                            t1 = int(t1) - start_position
                            t2 = int(t2) - start_position
                            rel = (";").join(
                                [str(h1), str(h2), str(t1), str(t2), relation]
                            )

                            rel_lines.append(rel)

                            rel_flag = True

                    if rel_flag:
                        document.append("\n")
                        document.append("# text = " + current_document.text)
                        document.append("\n")
                        number_of_rels = len(rel_lines)
                        if number_of_rels == 1:
                            document.append("# relations = " + rel_lines[0])
                        else:
                            relations = ("|").join([rel for rel in rel_lines[:-1]])
                            document.append(
                                "# relations = " + relations + "|" + rel_lines[-1]
                            )
                        document.append("\n")
                        for line in ner_lines:
                            document.append(line)
                            document.append("\n")
                        document.append("\n")
                        sentence = "# text = "
                        word_positions_dict = (
                            dict()
                        )  # RE INITIALIZE DICT TO SAVE THE POSITIONS OF THE NEW SENTENCE

                        ner_lines = list()
                        rel_lines = list()
                        counter = 1
            else:
                document.append(
                    ("\t").join([str(i + 1), str(word_tag[0]), str(word_tag[1])])
                )
                if str(word_tag[0]) == "." or str(word_tag[0]) == "?":
                    document.append("\n")

                document.append("\n")
        document.append("\n\n\n")

    if mode == "conllu":
        output = f"data/from_json/{split}.{mode}"
    else:
        output = f"data/from_json/{split}.txt"
    logging.info(f"Writing {mode} to {output}")
    with open(output, "w", encoding="utf-8") as outfile:
        outfile.write("".join(document))
