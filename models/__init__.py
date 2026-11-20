__all__ = ["FlairRelationTrainer", "FlairNERTrainer", "FlairSequenceLabelingTrainer"]

_LAZY = {
    "FlairRelationTrainer": (".relation_classfication", "FlairRelationTrainer"),
    "FlairNERTrainer": (".sequence_labeling", "FlairNERTrainer"),
    "FlairSequenceLabelingTrainer": (".sequence_labeling", "FlairNERTrainer"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_name, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
