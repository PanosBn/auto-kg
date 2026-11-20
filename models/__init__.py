"""Model training entry points.

Trainers are imported lazily so lightweight helpers (for example
``models.entity_label_map``) can be used and tested without importing Flair
and PyTorch.
"""

__all__ = ["FlairRelationTrainer", "FlairNERTrainer", "FlairSequenceLabelingTrainer"]


def __getattr__(name):
    if name == "FlairRelationTrainer":
        from .relation_classfication import FlairRelationTrainer

        return FlairRelationTrainer
    if name in ("FlairNERTrainer", "FlairSequenceLabelingTrainer"):
        # ``FlairSequenceLabelingTrainer`` was exported previously but never
        # existed; keep the name as an alias of the real class.
        from .sequence_labeling import FlairNERTrainer

        return FlairNERTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
