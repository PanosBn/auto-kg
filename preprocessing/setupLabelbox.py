import labelbox
import os
import re
import uuid
import logging
import yaml
from labelbox import LabelImport, LabelboxError
from labelbox.data.annotation_types import Label, TextData, ObjectAnnotation, TextEntity
from labelbox.data.serialization import NDJsonConverter
from spacy.tokens import Doc
from typing import List, Dict, Any

logging.basicConfig(filename='uploadJob.log', encoding='utf-8', level=logging.DEBUG)

class DataLabellingService:
    def __init__(self, api_key: str, ontology_id: str = None):
        try:
            api_key = os.environ.get('LABELBOX_API_KEY', api_key)
            if not api_key:
                raise ValueError("LABELBOX_API_KEY environment variable not set or provided as argument")
            
            self.client = labelbox.Client(api_key=api_key)

            if ontology_id:
                self.ontology = self.client.get_ontology(ontology_id=ontology_id)
            self.dataset = None

            with open('config.yaml', 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def create_dataset(self, name: str):
        try:
            self.dataset = self.client.create_dataset(name=name)
            logging.info(f"Dataset '{name}' created")
            return self.dataset
        except LabelboxError as e:
            logging.error(f"Error creating dataset: {e}")
            raise

    def create_data_rows_and_wait(self, assets: List[Dict[str, Any]]):
        try:
            task = self.dataset.create_data_rows(assets)
            task.wait_till_done()
            if task.errors:
                logging.error(f"Data row creation errors: {task.errors}")
            else:
                logging.info("Data rows created and task completed")
            return [dr.uid for dr in list(self.dataset.export_data_rows())]
        except LabelboxError as e:
            logging.error(f"Error creating data rows: {e}")
            raise

    def create_asset(self, doc: Doc):
        try:
            asset_data = {"row_data": doc.text, "global_key": f"TEST-ID-{uuid.uuid1()}"}
            return asset_data
        except Exception as e:
            logging.error(f"Error creating asset: {e}")
            raise

    @staticmethod
    def get_word_span(text: str, word: str):
        p = re.compile(word)
        for m in p.finditer(text):
            logging.debug((m.start(), m.end(), m.group()))
        return m.start(), m.end()

    def gather_labels(self, label_payload: list, data_row):
        text_data = TextData(uid=data_row)
        return Label(data=text_data, annotations=label_payload)

    def create_ner_annotation(self, start: int, end: int, type_of_entity: str):

        self.entity_mapping = self.config['entity_mapping']
        named_entity = TextEntity(start=start, end=end-1)
        entity_name = self.entity_mapping.get(type_of_entity, type_of_entity)
        named_entity_annotation = ObjectAnnotation(value=named_entity, name=entity_name)
        return named_entity_annotation
    
    def process_documents_and_create_labels(self, documents: List[Doc], data_rows):
        labels = []
        for doc in documents:
            labels_payload = []
            doc_uid = None
            for data_row in data_rows:
                if doc.text == data_row.row_data:
                    doc_ents = doc
                    doc_uid = data_row.uid
            if doc_uid:  # Proceed only if a matching data row is found
                for ent in doc_ents.ents:
                    ner_annotation = self.create_ner_annotation(ent.start_char, ent.end_char, ent.label_)
                    labels_payload.append(ner_annotation)
                labels.append(self.gather_labels(labels_payload, doc_uid))
        return labels

    def upload_labels(self, labels: list):
        try:
            label_ndjson = list(NDJsonConverter.serialize(labels))
            upload_job = LabelImport.create_from_objects(
                client=self.client,
                project_id=self.dataset.uid,
                name="lbm2" + str(uuid.uuid1()),
                labels=label_ndjson)
            upload_job.wait_until_done()
            if upload_job.errors:
                logging.error(f"Label upload errors: {upload_job.errors}")
            else:
                logging.info("Labels uploaded successfully")
        except LabelboxError as e:
            logging.error(f"Error uploading labels: {e}")
            raise

    def signing_function_batch(self, obj_bytes: bytes) -> str:
        try:
            url = self.client.upload_data(content=obj_bytes, sign=True)
            logging.info(f'Uploaded batch data at {url}')
            return url
        except LabelboxError as e:
            logging.error(f"Error in signing batch function: {e}")
            raise
