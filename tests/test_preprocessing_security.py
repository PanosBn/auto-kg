import logging

import pytest

for _required in ("spacy", "pandas", "labelbox"):
    pytest.importorskip(_required)


@pytest.fixture
def labelling_service():
    from preprocessing.setup_labelbox import DataLabellingService

    return DataLabellingService


@pytest.fixture
def extractor_class():
    from preprocessing.entity_relation_extraction import EntityRelationshipExtractor

    return EntityRelationshipExtractor


SIGNED_URL = "https://storage.example.invalid/uploads/row.ndjson?X-Goog-Signature=SYNTHETICSIG&Expires=1"


def test_get_word_span_matches_literally(labelling_service):
    text = "prijs is 9.99 (incl) en a+b"
    assert labelling_service.get_word_span(text, "9.99") == (9, 13)
    assert labelling_service.get_word_span(text, "(incl)") == (14, 20)
    assert labelling_service.get_word_span(text, "a+b") == (24, 27)
    with pytest.raises(ValueError):
        labelling_service.get_word_span("prijs 9x99", "9.99")
    with pytest.raises(ValueError):
        labelling_service.get_word_span(text, "")


def test_get_word_span_no_catastrophic_backtracking(labelling_service):
    text = "a" * 5000 + "!"
    with pytest.raises(ValueError):
        labelling_service.get_word_span(text, "(a+)+$x")


class _FakeClient:
    def upload_data(self, content, sign):
        assert sign is True
        return SIGNED_URL


def test_signed_upload_url_is_not_logged(labelling_service, caplog):
    service = object.__new__(labelling_service)
    service.client = _FakeClient()
    with caplog.at_level(logging.DEBUG):
        assert service.signing_function_batch(b"{}") == SIGNED_URL
    assert "SYNTHETICSIG" not in caplog.text
    assert "row.ndjson" not in caplog.text
    assert "storage.example.invalid" in caplog.text


def test_failed_entity_assignment_does_not_leak_dialogue(extractor_class, tmp_path, caplog, capsys):
    import spacy
    from spacy.tokens import Doc, Span

    Doc.set_extension("rel", default=list(), force=True)
    Doc.set_extension("rel_ready", default=list(), force=True)
    Span.set_extension("span_rels", default=list(), force=True)

    extractor = object.__new__(extractor_class)
    extractor.nlp = spacy.blank("nl")
    extractor.entity_lv1_lv2 = {}
    extractor.rel_ontology_mapping = {}
    extractor.set_of_rels = set()
    extractor.annotated_spacy_docs = []
    extractor.entity_locations_file = str(tmp_path / "ents.csv")
    extractor.relationships_file = str(tmp_path / "rels.csv")

    dialogue = "Synthetisch geheim gesprek over modem en factuur"
    item = {
        "ID": "synthetic-row-1",
        "Labeled Data": dialogue,
        "Label": {
            "objects": [
                {"data": {"location": {"start": 0, "end": 18}}, "title": "PRODUCT", "featureId": "e1"},
                {"data": {"location": {"start": 12, "end": 25}}, "title": "SERVICE", "featureId": "e2"},
            ],
            "relationships": [],
        },
    }
    with caplog.at_level(logging.DEBUG):
        extractor.extract_ent_rel_pairs([item])
    out = capsys.readouterr()
    assert "Item 0" in caplog.text and "synthetic-row-1" in caplog.text
    for chunk in ("geheim gesprek", "Synthetisch", "factuur"):
        assert chunk not in caplog.text
        assert chunk not in out.out and chunk not in out.err
