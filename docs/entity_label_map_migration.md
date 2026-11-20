# Entity label map: pickle → JSON migration

**Breaking change.** `FlairRelationTrainer` no longer accepts a pickled entity label
map (`entity_label_map_path` / `[RelationExtraction] entity_label_map_path`). Loading a
pickle runs whatever code the file contains, so a tampered or untrusted map file could
take over the training machine. The trainer now accepts only a small, validated JSON
document and rejects anything that looks like a pickle **without** opening it as one.

## New format (`schema_version` 1)

```json
{
  "schema_version": 1,
  "description": "optional free text",
  "entity_pairs": [["PERSON", "PRODUCT"], ["PERSON", "SERVICE"]]
}
```

- `entity_pairs`: non-empty list of `[head_label, tail_label]` pairs, with labels as they
  appear in the NER column (no `B-`/`I-`/`E-`/`S-` prefix). This is the
  `Set[Tuple[str, str]]` Flair's `RelationClassifier(entity_pair_labels=...)` expects.
- Labels: 1–128 characters, letters/digits/`_`/`-`/`.`/`:`/`/`, no whitespace.
- Unknown keys, other schema versions, files over 1 MB or over 10,000 pairs are rejected.

A synthetic example lives at `models/entity_label_map.example.json`.

## How to migrate (no pickle is loaded)

**Option A, recommended: regenerate from your training data.** The pairs the classifier
needs are the entity-type pairs that occur in your relation annotations:

```bash
python -m models.entity_label_map from-conllu TRAIN.conllu DEV.conllu -o entity_label_map.json
python -m models.entity_label_map validate entity_label_map.json
```

Review the generated list, then point `entity_label_map_path` at the JSON file. If your
old map deliberately allowed pairs that never occur in the data, add them by hand.

**Option B: write the JSON by hand** from your ontology / documentation of allowed
entity-type pairs, then run the `validate` command above.

This repository intentionally ships **no** pickle-reading converter, and the tests and
ingestion code never unpickle legacy files. If you must recover pairs from an old pickle
you created yourself, do it only for a file whose origin and hash you have verified, in a
disposable offline environment with no credentials or customer data, and then discard the
pickle. Do not accept pickle maps from anyone else.

## Other behaviour changes in the same branch

- `FlairRelationTrainer` initialises `entity_label_map` to `None`, so the
  config-file fallback in `train_models()` works instead of raising `AttributeError`.
- `models/__init__.py` imports trainers lazily; the previously exported but non-existent
  `FlairSequenceLabelingTrainer` is now an alias of `FlairNERTrainer`.
