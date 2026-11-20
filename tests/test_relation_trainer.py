import json

import pytest

from models.entity_label_map import EntityLabelMapError


@pytest.fixture
def trainer():
    pytest.importorskip("flair")
    from models.relation_classfication import FlairRelationTrainer

    return object.__new__(FlairRelationTrainer)


def test_trainer_loads_json_map(tmp_path, trainer):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"schema_version": 1, "entity_pairs": [["PERSON", "PRODUCT"]]}))
    assert trainer.load_entity_label_map(str(path)) == {("PERSON", "PRODUCT")}


def test_trainer_refuses_legacy_pickle(malicious_pickle, trainer):
    path, sentinel = malicious_pickle
    with pytest.raises(EntityLabelMapError):
        trainer.load_entity_label_map(str(path))
    assert not sentinel.exists()
