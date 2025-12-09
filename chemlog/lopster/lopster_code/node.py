from typing import Set

class Node:
    """Labeled node with a number and a set of labels."""
    def __init__(self, number: int, label: Set[str]):
        self.number = int(number)
        self.label = set(label)

    def get_number(self) -> int:
        return self.number

    def get_label(self) -> Set[str]:
        return self.label
    
    def __repr__(self):
        label = ','.join(self.label)
        return f"Node({self.number}: {{{label}}})"
