import json
import pathlib
import re
import subprocess
import sys

import pytest

from models.entity_label_map import (
    EntityLabelMapError,
    dump_entity_label_map,
    load_entity_label_map,
    pairs_from_conllu,
)
from tests.conftest import REPO_ROOT


def write(tmp_path, document, name="map.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_loads_valid_map_as_set_of_tuples(tmp_path):
    path = write(
        tmp_path,
        {
            "schema_version": 1,
            "description": "synthetic",
            "entity_pairs": [["PERSON", "PRODUCT"], ["PERSON", "SERVICE"], ["PERSON", "PRODUCT"]],
        },
    )
    assert load_entity_label_map(path) == {("PERSON", "PRODUCT"), ("PERSON", "SERVICE")}


def test_example_file_in_repo_is_valid():
    example = pathlib.Path(REPO_ROOT) / "models" / "entity_label_map.example.json"
    assert load_entity_label_map(example)


def test_rejects_pickle_without_unpickling(malicious_pickle):
    path, sentinel = malicious_pickle
    with pytest.raises(EntityLabelMapError, match="pickle"):
        load_entity_label_map(path)
    assert not sentinel.exists(), "legacy pickle must never be loaded"


@pytest.mark.parametrize(
    "document, message",
    [
        ([], "JSON object"),
        ({"schema_version": 2, "entity_pairs": [["A", "B"]]}, "schema_version"),
        ({"schema_version": True, "entity_pairs": [["A", "B"]]}, "schema_version"),
        ({"schema_version": 1, "entity_pairs": []}, "non-empty"),
        ({"schema_version": 1, "entity_pairs": [["A"]]}, "exactly two"),
        ({"schema_version": 1, "entity_pairs": [["A", 3]]}, "string"),
        ({"schema_version": 1, "entity_pairs": [["A", "has space"]]}, "unsupported characters"),
        ({"schema_version": 1, "entity_pairs": [["A", ""]]}, "characters long"),
        ({"schema_version": 1, "entity_pairs": [["A", "B"]], "py/object": "os.system"}, "unknown keys"),
    ],
)
def test_rejects_invalid_documents(tmp_path, document, message):
    with pytest.raises(EntityLabelMapError, match=message):
        load_entity_label_map(write(tmp_path, document))


def test_rejects_non_json_and_missing_and_oversized(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(EntityLabelMapError, match="not valid UTF-8 JSON"):
        load_entity_label_map(bad)
    with pytest.raises(EntityLabelMapError, match="not found"):
        load_entity_label_map(tmp_path / "missing.json")
    big = tmp_path / "big.json"
    big.write_bytes(b" " * 1_000_001)
    with pytest.raises(EntityLabelMapError, match="limit"):
        load_entity_label_map(big)


def test_dump_round_trip(tmp_path):
    pairs = {("PERSON", "PRODUCT"), ("ORG", "SERVICE")}
    path = tmp_path / "out.json"
    path.write_text(dump_entity_label_map(pairs), encoding="utf-8")
    assert load_entity_label_map(path) == pairs


SYNTHETIC_CONLLU = """\
# text = Ik heb een synthetisch abonnement.
# relations = 1;1;5;5;heeft|1;1;5;5;heeft
1\tIk\tS-PERSON
2\theb\tO
3\teen\tO
4\tsynthetisch\tB-PRODUCT
5\tabonnement\tE-PRODUCT
6\t.\tO

# text = Router werkt niet.
# relations = 1;1;3;3;status
1\tRouter\tS-DEVICE
2\twerkt\tO
3\tniet\tS-STATUS
"""


def test_pairs_from_conllu_uses_relation_heads_and_tails():
    assert pairs_from_conllu(SYNTHETIC_CONLLU.splitlines(True)) == {
        ("PERSON", "PRODUCT"),
        ("DEVICE", "STATUS"),
    }


def test_cli_from_conllu_and_validate(tmp_path):
    conllu = tmp_path / "train.conllu"
    conllu.write_text(SYNTHETIC_CONLLU, encoding="utf-8")
    out = tmp_path / "map.json"
    run = subprocess.run(
        [sys.executable, "-m", "models.entity_label_map", "from-conllu", str(conllu), "-o", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert load_entity_label_map(out) == {("PERSON", "PRODUCT"), ("DEVICE", "STATUS")}
    check = subprocess.run(
        [sys.executable, "-m", "models.entity_label_map", "validate", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert check.returncode == 0 and "valid" in check.stdout


def test_cli_validate_rejects_pickle(malicious_pickle):
    path, sentinel = malicious_pickle
    run = subprocess.run(
        [sys.executable, "-m", "models.entity_label_map", "validate", str(path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 1 and "pickle" in run.stderr
    assert not sentinel.exists()


def test_helper_import_does_not_pull_in_flair():
    code = "import sys, models.entity_label_map; print('flair' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "False"


def test_no_pickle_deserialization_in_application_code():
    pattern = re.compile(r"\b(?:pickle|cPickle|dill|joblib)\.loads?\b|import\s+pickle|from\s+pickle\s+import")
    offenders = []
    for path in pathlib.Path(REPO_ROOT).rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in {"tests", ".venv", "venv"} or ".git" in rel.parts:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(rel))
    assert offenders == []
