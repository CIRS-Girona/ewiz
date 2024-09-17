"""General core utilities.
"""
import numpy as np
import json
import os

from typing import Any, Dict, List, Tuple, Callable, Union


def get_children(parent: Callable) -> List[Callable]:
    """Gets class inheritors.
    """
    parents = [parent]
    children = set()
    while parents:
        parent = parents.pop()
        for child in parent.__subclasses__():
            if child not in children:
                children.add(child)
                parents.append(child)
    return children

def get_inheritors(parent: Callable) -> Dict[str, Callable]:
    """Gets class inheritors.
    """
    children = {child.name: child for child in get_children(parent)}
    children.update({"base": parent})
    return children

def save_json(data: Dict, path: str) -> None:
    """Saves JSON file to disk.
    """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(data, indent=2))

def read_json(path: str) -> None:
    """Reads JSON file.
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data

def create_dir(dir_path: str) -> str:
    """Creates directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path
