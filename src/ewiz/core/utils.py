"""General core utilities.
"""
import numpy as np

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
