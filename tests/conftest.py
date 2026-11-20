import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _PickleThatCreatesSentinelFileWhenLoaded:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def __reduce__(self):
        return (open, (self.sentinel, "w"))


@pytest.fixture
def malicious_pickle(tmp_path):
    import pickle

    sentinel = tmp_path / "PWNED"
    path = tmp_path / "entity_label_map.pkl"
    path.write_bytes(pickle.dumps(_PickleThatCreatesSentinelFileWhenLoaded(str(sentinel)), protocol=4))
    return path, sentinel
