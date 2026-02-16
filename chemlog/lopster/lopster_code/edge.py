from typing import Set

class Edge:
    """Labeled directed edge from `from_node` to `to_node`."""
    def __init__(self, from_node: int, to_node: int, label: Set[str]):
        self.from_node = int(from_node)
        self.to_node = int(to_node)
        self.label = set(label)

    def get_from_node(self) -> int:
        return self.from_node

    def get_to_node(self) -> int:
        return self.to_node

    def get_label(self) -> Set[str]:
        return self.label
