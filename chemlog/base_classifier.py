import abc
import enum
from enum import auto
from typing import Optional, Any

from rdkit import Chem

class Classifier(abc.ABC):
    """
    Base class for classifiers.
    """

    @abc.abstractmethod
    def classify(self, mol: Chem.Mol, *args, **kwargs) -> (Any, Optional[dict]):
        pass

    def on_finish(self):
        pass


class ChargeCategories(enum.Enum):
    ANION = auto()
    CATION = auto()
    ZWITTERION = auto()
    SALT = auto()
    NEUTRAL = auto()
    UNKNOWN = auto()
