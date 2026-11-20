from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterable, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = 1
MAX_FILE_BYTES = 1_000_000
MAX_PAIRS = 10_000
MAX_LABEL_LENGTH = 128

_LABEL_RE = re.compile(r"^[^\W][\w.\-:/]*$", re.UNICODE)
_ALLOWED_KEYS = {"schema_version", "entity_pairs", "description"}
_BIOES_PREFIX_RE = re.compile(r"^(?:[BIESLU])-")

EntityPair = Tuple[str, str]


class EntityLabelMapError(ValueError):
    pass


PICKLE_PROTOCOL_2_PLUS_MARKER = b"\x80"


def _looks_like_pickle(head: bytes) -> bool:
    return head[:1] == PICKLE_PROTOCOL_2_PLUS_MARKER


def validate_label(label: object, where: str) -> str:
    if not isinstance(label, str):
        raise EntityLabelMapError(f"{where}: label must be a string, got {type(label).__name__}")
    if not label or len(label) > MAX_LABEL_LENGTH:
        raise EntityLabelMapError(
            f"{where}: label must be 1-{MAX_LABEL_LENGTH} characters long"
        )
    if not _LABEL_RE.match(label):
        raise EntityLabelMapError(
            f"{where}: label {label!r} contains unsupported characters "
            "(use letters, digits, '_', '-', '.', ':' or '/', no whitespace)"
        )
    return label


def parse_entity_label_map(document: object) -> Set[EntityPair]:
    if not isinstance(document, dict):
        raise EntityLabelMapError("entity label map must be a JSON object")
    unknown = set(document) - _ALLOWED_KEYS
    if unknown:
        raise EntityLabelMapError(f"unknown keys in entity label map: {sorted(unknown)}")
    version = document.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        raise EntityLabelMapError(
            f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}"
        )
    if "description" in document and not isinstance(document["description"], str):
        raise EntityLabelMapError("description must be a string")
    pairs = document.get("entity_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise EntityLabelMapError("entity_pairs must be a non-empty list")
    if len(pairs) > MAX_PAIRS:
        raise EntityLabelMapError(f"entity_pairs has more than {MAX_PAIRS} entries")

    result: Set[EntityPair] = set()
    for index, pair in enumerate(pairs):
        where = f"entity_pairs[{index}]"
        if not isinstance(pair, list) or len(pair) != 2:
            raise EntityLabelMapError(f"{where}: must be a list of exactly two labels")
        head = validate_label(pair[0], where)
        tail = validate_label(pair[1], where)
        result.add((head, tail))
    return result


def load_entity_label_map(path: str | os.PathLike) -> Set[EntityPair]:
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise EntityLabelMapError(f"entity label map not found: {path}")
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise EntityLabelMapError(
            f"entity label map is {size} bytes; limit is {MAX_FILE_BYTES}"
        )
    with open(path, "rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    if _looks_like_pickle(raw):
        raise EntityLabelMapError(
            "entity label map looks like a Python pickle. Pickle maps are no longer "
            "supported because loading them can execute code. Convert it to JSON as "
            "described in docs/entity_label_map_migration.md."
        )
    try:
        text = raw.decode("utf-8")
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntityLabelMapError(
            f"entity label map is not valid UTF-8 JSON: {exc.__class__.__name__}"
        ) from None
    return parse_entity_label_map(document)


def dump_entity_label_map(pairs: Iterable[EntityPair], description: Optional[str] = None) -> str:
    document = {
        "schema_version": SCHEMA_VERSION,
        "entity_pairs": [list(pair) for pair in sorted(set(pairs))],
    }
    if description:
        document["description"] = description
    parse_entity_label_map(document)
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _strip_bioes(tag: str) -> Optional[str]:
    tag = tag.strip()
    if not tag or tag == "O":
        return None
    return _BIOES_PREFIX_RE.sub("", tag)


def pairs_from_conllu(lines: Iterable[str]) -> Set[EntityPair]:
    pairs: Set[EntityPair] = set()
    relations: List[Tuple[int, int]] = []
    tags: dict = {}

    def flush() -> None:
        for head_id, tail_id in relations:
            head = tags.get(head_id)
            tail = tags.get(tail_id)
            if head and tail:
                pairs.add((validate_label(head, "conllu"), validate_label(tail, "conllu")))
        relations.clear()
        tags.clear()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            body = stripped[1:].strip()
            if body.startswith("relations") and "=" in body:
                spec = body.split("=", 1)[1].strip()
                for rel in filter(None, spec.split("|")):
                    fields = rel.split(";")
                    if len(fields) < 5:
                        continue
                    try:
                        relations.append((int(fields[0]), int(fields[2])))
                    except ValueError:
                        continue
            continue
        columns = line.split("\t")
        if len(columns) >= 3:
            try:
                token_id = int(columns[0])
            except ValueError:
                continue
            label = _strip_bioes(columns[2])
            if label:
                tags[token_id] = label
    flush()
    return pairs


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create or validate a JSON entity label map.")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser(
        "from-conllu", help="derive an entity label map from CoNLL-U training files"
    )
    convert.add_argument("conllu", nargs="+", help="CoNLL-U file(s) with '# relations' comments")
    convert.add_argument("-o", "--output", required=True, help="JSON file to write")

    check = sub.add_parser("validate", help="validate an entity label map JSON file")
    check.add_argument("path")

    args = parser.parse_args(argv)
    try:
        if args.command == "from-conllu":
            pairs: Set[EntityPair] = set()
            for path in args.conllu:
                with open(path, encoding="utf-8") as handle:
                    pairs |= pairs_from_conllu(handle)
            if not pairs:
                raise EntityLabelMapError("no entity pairs found in the given CoNLL-U files")
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(dump_entity_label_map(pairs))
            print(f"wrote {len(pairs)} entity pairs to {args.output}")
        else:
            pairs = load_entity_label_map(args.path)
            print(f"{args.path}: valid, {len(pairs)} entity pairs")
    except EntityLabelMapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
