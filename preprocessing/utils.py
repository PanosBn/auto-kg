import json
from typing import Dict, List


def read_json(file_path: str) -> List[Dict]:
    with open(file_path) as f:
        return json.load(f)


def load_mapping(file_path: str, delimiter: str = "\t") -> Dict[str, str]:
    mapping = {}
    with open(file_path) as file:
        for line in file:
            key, value = line.strip().split(delimiter)
            mapping[key.strip()] = value.strip()
    return mapping
