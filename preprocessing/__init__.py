__all__ = [
    "DataLabellingService",
    "EntityRelationshipExtractor",
    "export_training_data",
    "generate_training_data",
    "read_json",
    "load_mapping",
]

_LAZY = {
    "DataLabellingService": ".setup_labelbox",
    "EntityRelationshipExtractor": ".entity_relation_extraction",
    "export_training_data": ".entity_relation_extraction",
    "generate_training_data": ".entity_relation_extraction",
    "read_json": ".utils",
    "load_mapping": ".utils",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
