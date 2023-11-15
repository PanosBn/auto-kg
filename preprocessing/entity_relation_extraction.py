import spacy
import csv
import logging
import yaml
import os
from spacy.tokens import Doc, Span
from typing import Dict, List, Tuple

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
                # creating writer object
                csv_writer = csv.writer(ent1)
                # appending data
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
