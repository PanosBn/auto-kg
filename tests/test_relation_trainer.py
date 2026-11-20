"""Relation trainer uses the JSON loader (requires Flair; skipped otherwise)."""

import json

import pytest

pytest.importorskip("flair")

from models.entity_label_map import EntityLabelMapError  # noqa: E402
from models.relation_classfication import FlairRelationTrainer  # noqa: E402


def test_trainer_loads_json_map(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"schema_version": 1, "entity_pairs": [["PERSON", "PRODUCT"]]}))
    trainer = object.__new__(FlairRelationTrainer)
    assert trainer.load_entity_label_map(str(path)) == {("PERSON", "PRODUCT")}


def test_trainer_refuses_legacy_pickle(malicious_pickle):
    path, sentinel = malicious_pickle
    trainer = object.__new__(FlairRelationTrainer)
    with pytest.raises(EntityLabelMapError):
        trainer.load_entity_label_map(str(path))
    assert not sentinel.exists()
