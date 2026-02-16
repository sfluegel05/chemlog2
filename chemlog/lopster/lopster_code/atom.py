from typing import List

class Atom:
    """Data structure for first-order logic atoms."""
    def __init__(self, predicate: str, terms: List[str]):
        self.predicate = predicate
        self.terms = list(terms)

    def get_predicate(self) -> str:
        return self.predicate

    def get_terms(self) -> List[str]:
        return self.terms

    def text_representation(self) -> str:
        if not self.terms:
            return self.predicate
        return f"{self.predicate}({','.join(self.terms)})"
