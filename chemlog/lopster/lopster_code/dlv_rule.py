from typing import List
from .atom import Atom

class DLVRule:
    """DLV rule with head atom and optional body atoms."""
    def __init__(self, head_atom: Atom, body_atoms: List[Atom]):
        self.head_atom = head_atom
        self.body_atoms = list(body_atoms)

    def get_head_atom(self) -> Atom:
        return self.head_atom

    def get_body_atoms(self) -> List[Atom]:
        return self.body_atoms

    def dlv_syntax_rule(self) -> str:
        head = self.head_atom.text_representation()
        if not self.body_atoms:
            return head + "."
        body = ','.join(a.text_representation() for a in self.body_atoms)
        return f"{head} :- {body}."
