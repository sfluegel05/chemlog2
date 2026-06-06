"""
Tests for FastModelChecker.

Validates that FastModelChecker produces the same results as the original
ModelChecker on a range of formula shapes relevant to ChEBI classification.
"""

import itertools
import time

import numpy as np
import pytest
from gavel.logic import logic
from gavel.logic.logic_utils import convert_to_nnf

from chemlog.fol_classification.model_checking import (
    ModelChecker as OriginalModelChecker,
    ModelCheckerOutcome,
)
from chemlog.fol_classification.fast_model_checking import (
    FastModelChecker,
    compile_formula,
)
from chemlog.fol_classification.fol_utils import normalize_fol_formula


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain_structure(n_atoms: int):
    """Create a linear chain molecule-like structure: 0-1-2-..-(n-1).

    Unary predicates: 'carbon' on even atoms, 'nitrogen' on odd atoms.
    Binary predicate: 'bond' as single-bond adjacency.
    """
    universe = n_atoms
    extensions = {
        logic.BinaryConnective.EQ.name: np.eye(universe, dtype=np.bool_),
        "carbon": np.array([i % 2 == 0 for i in range(universe)], dtype=np.bool_),
        "nitrogen": np.array([i % 2 == 1 for i in range(universe)], dtype=np.bool_),
        "atom": np.ones(universe, dtype=np.bool_),
        "bond": np.zeros((universe, universe), dtype=np.bool_),
    }
    for i in range(n_atoms - 1):
        extensions["bond"][i][i + 1] = True
        extensions["bond"][i + 1][i] = True
    return universe, extensions


def _make_ring_structure(n_atoms: int):
    """Create a ring: 0-1-2-..(n-1)-0."""
    universe, extensions = _make_chain_structure(n_atoms)
    extensions["bond"][0][n_atoms - 1] = True
    extensions["bond"][n_atoms - 1][0] = True
    return universe, extensions


def _outcome_matches(a: ModelCheckerOutcome, b: ModelCheckerOutcome) -> bool:
    """Check that outcomes are equivalent (treating inferred as same)."""
    pos = {ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED}
    neg = {ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED}
    return (a in pos and b in pos) or (a in neg and b in neg)


# ---------------------------------------------------------------------------
# Tests: basic predicate evaluation
# ---------------------------------------------------------------------------

class TestBasicEvaluation:

    def test_unary_positive(self):
        universe, ext = _make_chain_structure(4)
        # ∃x: carbon(x)  — should be true (atoms 0, 2)
        x = logic.Variable("x")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x],
            logic.PredicateExpression("carbon", [x]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, witness = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND

    def test_unary_negative(self):
        universe, ext = _make_chain_structure(4)
        # ∃x: oxygen(x)  — should be false (no oxygen in structure)
        x = logic.Variable("x")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x],
            logic.PredicateExpression("oxygen", [x]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, _ = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.NO_MODEL

    def test_binary_bond(self):
        universe, ext = _make_chain_structure(4)
        # ∃x ∃y: bond(x, y) ∧ carbon(x) ∧ nitrogen(y)
        x, y = logic.Variable("x"), logic.Variable("y")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("bond", [x, y]),
                logic.PredicateExpression("carbon", [x]),
                logic.PredicateExpression("nitrogen", [y]),
            ]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, witness = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND
        # Witness should map x to an even atom and y to an adjacent odd atom
        witness_dict = dict(witness)
        assert witness_dict["x"] % 2 == 0
        assert witness_dict["y"] % 2 == 1
        assert abs(witness_dict["x"] - witness_dict["y"]) == 1

    def test_inequality(self):
        universe, ext = _make_chain_structure(4)
        # ∃x ∃y: carbon(x) ∧ carbon(y) ∧ x ≠ y
        x, y = logic.Variable("x"), logic.Variable("y")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("carbon", [x]),
                logic.PredicateExpression("carbon", [y]),
                logic.BinaryFormula(x, logic.BinaryConnective.NEQ, y),
            ]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, witness = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND
        witness_dict = dict(witness)
        assert witness_dict["x"] != witness_dict["y"]


# ---------------------------------------------------------------------------
# Tests: quantifier nesting
# ---------------------------------------------------------------------------

class TestQuantifiers:

    def test_forall_unary(self):
        universe, ext = _make_chain_structure(4)
        # ∀x: atom(x)  — true, all are atoms
        x = logic.Variable("x")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.UNIVERSAL, [x],
            logic.PredicateExpression("atom", [x]),
        )
        # Wrap in dummy existential so find_model works
        checker = FastModelChecker(universe, ext)
        outcome, _ = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND

    def test_forall_negative(self):
        universe, ext = _make_chain_structure(4)
        # ∀x: carbon(x)  — false (odd atoms are nitrogen)
        x = logic.Variable("x")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.UNIVERSAL, [x],
            logic.PredicateExpression("carbon", [x]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, _ = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.NO_MODEL

    def test_exists_forall_pattern(self):
        universe, ext = _make_chain_structure(6)
        # ∃x: carbon(x) ∧ ∀y: (bond(x,y) → nitrogen(y))
        # = ∃x: carbon(x) ∧ ∀y: (¬bond(x,y) ∨ nitrogen(y))
        x, y = logic.Variable("x"), logic.Variable("y")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("carbon", [x]),
                logic.QuantifiedFormula(
                    logic.Quantifier.UNIVERSAL, [y],
                    logic.NaryFormula(logic.BinaryConnective.DISJUNCTION, [
                        logic.UnaryFormula(
                            logic.UnaryConnective.NEGATION,
                            logic.PredicateExpression("bond", [x, y]),
                        ),
                        logic.PredicateExpression("nitrogen", [y]),
                    ]),
                ),
            ]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, witness = checker.find_model(formula)
        # Carbon atoms at 0, 2, 4. Their neighbors:
        # 0: [1] (nitrogen) ✓
        # 2: [1, 3] (both nitrogen) ✓
        # 4: [3, 5] (both nitrogen) ✓
        assert outcome == ModelCheckerOutcome.MODEL_FOUND


# ---------------------------------------------------------------------------
# Tests: defined predicates
# ---------------------------------------------------------------------------

class TestDefinedPredicates:

    def test_simple_definition(self):
        universe, ext = _make_chain_structure(6)
        # Define: cn_pair(a, b) <=> carbon(a) ∧ nitrogen(b) ∧ bond(a, b)
        a, b = logic.Variable("a"), logic.Variable("b")
        cn_body = logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
            logic.PredicateExpression("carbon", [a]),
            logic.PredicateExpression("nitrogen", [b]),
            logic.PredicateExpression("bond", [a, b]),
        ])
        definitions = {"cn_pair": ([a, b], cn_body)}

        # ∃x ∃y: cn_pair(x, y)
        x, y = logic.Variable("x"), logic.Variable("y")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.PredicateExpression("cn_pair", [x, y]),
        )

        checker = FastModelChecker(universe, ext, predicate_definitions=definitions)
        outcome, witness = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND

    def test_definition_caching(self):
        """Verify that defined predicate results are cached across calls."""
        universe, ext = _make_chain_structure(6)
        a, b = logic.Variable("a"), logic.Variable("b")
        cn_body = logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
            logic.PredicateExpression("carbon", [a]),
            logic.PredicateExpression("nitrogen", [b]),
            logic.PredicateExpression("bond", [a, b]),
        ])
        definitions = {"cn_pair": ([a, b], cn_body)}

        checker = FastModelChecker(universe, ext, predicate_definitions=definitions)

        # Call twice — second should hit cache
        x, y = logic.Variable("x"), logic.Variable("y")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.PredicateExpression("cn_pair", [x, y]),
        )
        r1, _ = checker.find_model(formula)

        x2, y2 = logic.Variable("x"), logic.Variable("y")
        formula2 = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x2, y2],
            logic.PredicateExpression("cn_pair", [x2, y2]),
        )
        r2, _ = checker.find_model(formula2)
        assert _outcome_matches(r1, r2)
        assert len(checker._pred_cache) > 0


# ---------------------------------------------------------------------------
# Tests: comparison with original ModelChecker
# ---------------------------------------------------------------------------

class TestEquivalenceWithOriginal:
    """Run the same formulas through both checkers, verify same outcome."""

    @staticmethod
    def _check_both(universe, ext, formula, definitions=None):
        """Run formula through both checkers, assert same outcome."""
        # Normalize for original checker
        normalized = normalize_fol_formula(formula)

        pred_defs_orig = None
        pred_defs_fast = None
        if definitions:
            pred_defs_orig = {k: (v[0], normalize_fol_formula(v[1])) for k, v in definitions.items()}
            pred_defs_fast = definitions

        orig = OriginalModelChecker(universe, ext, predicate_definitions=pred_defs_orig)
        fast = FastModelChecker(universe, ext, predicate_definitions=pred_defs_fast)

        # Original expects normalized formula
        t0 = time.perf_counter()
        orig_result, orig_witness = orig.find_model(normalized, timeout=10)
        t_orig = time.perf_counter() - t0

        t0 = time.perf_counter()
        fast_result, fast_witness = fast.find_model(formula, timeout=10)
        t_fast = time.perf_counter() - t0

        assert _outcome_matches(orig_result, fast_result), (
            f"Mismatch: original={orig_result}, fast={fast_result} "
            f"for formula={formula}"
        )
        return t_orig, t_fast

    def test_chain_substructure(self):
        """Find a C-N-C chain in a 10-atom molecule."""
        universe, ext = _make_chain_structure(10)
        x, y, z = logic.Variable("x"), logic.Variable("y"), logic.Variable("z")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y, z],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("carbon", [x]),
                logic.PredicateExpression("nitrogen", [y]),
                logic.PredicateExpression("carbon", [z]),
                logic.PredicateExpression("bond", [x, y]),
                logic.PredicateExpression("bond", [y, z]),
                logic.BinaryFormula(x, logic.BinaryConnective.NEQ, z),
            ]),
        )
        t_orig, t_fast = self._check_both(universe, ext, formula)
        print(f"chain_substructure: orig={t_orig:.4f}s, fast={t_fast:.4f}s")

    def test_no_match(self):
        """Search for a pattern that doesn't exist."""
        universe, ext = _make_chain_structure(10)
        x, y = logic.Variable("x"), logic.Variable("y")
        # Look for oxygen — doesn't exist
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("oxygen", [x]),
                logic.PredicateExpression("bond", [x, y]),
            ]),
        )
        t_orig, t_fast = self._check_both(universe, ext, formula)
        print(f"no_match: orig={t_orig:.4f}s, fast={t_fast:.4f}s")

    def test_deeper_nesting(self):
        """5-variable existential pattern on a 20-atom chain."""
        universe, ext = _make_chain_structure(20)
        vs = [logic.Variable(f"v{i}") for i in range(5)]
        conjuncts = []
        for i in range(4):
            conjuncts.append(logic.PredicateExpression("bond", [vs[i], vs[i + 1]]))
        for i in range(5):
            pred = "carbon" if i % 2 == 0 else "nitrogen"
            conjuncts.append(logic.PredicateExpression(pred, [vs[i]]))
        # All different
        for i in range(5):
            for j in range(i + 1, 5):
                conjuncts.append(logic.BinaryFormula(vs[i], logic.BinaryConnective.NEQ, vs[j]))

        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, vs,
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, conjuncts),
        )
        t_orig, t_fast = self._check_both(universe, ext, formula)
        print(f"deeper_nesting (5 vars, 20 atoms): orig={t_orig:.4f}s, fast={t_fast:.4f}s")

    def test_larger_molecule(self):
        """6-variable pattern on a 50-atom chain — this is where speedup matters."""
        universe, ext = _make_chain_structure(50)
        vs = [logic.Variable(f"v{i}") for i in range(6)]
        conjuncts = []
        for i in range(5):
            conjuncts.append(logic.PredicateExpression("bond", [vs[i], vs[i + 1]]))
        for i in range(6):
            pred = "carbon" if i % 2 == 0 else "nitrogen"
            conjuncts.append(logic.PredicateExpression(pred, [vs[i]]))
        for i in range(6):
            for j in range(i + 1, 6):
                conjuncts.append(logic.BinaryFormula(vs[i], logic.BinaryConnective.NEQ, vs[j]))

        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, vs,
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, conjuncts),
        )
        t_orig, t_fast = self._check_both(universe, ext, formula)
        print(f"larger_molecule (6 vars, 50 atoms): orig={t_orig:.4f}s, fast={t_fast:.4f}s")
        # The fast checker should be significantly faster due to adjacency pruning
        assert t_fast < t_orig or t_fast < 1.0, (
            f"Fast checker ({t_fast:.3f}s) not faster than original ({t_orig:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Tests: witness format compatibility
# ---------------------------------------------------------------------------

class TestWitnessFormat:

    def test_witness_is_list_of_tuples(self):
        universe, ext = _make_chain_structure(6)
        x, y = logic.Variable("x"), logic.Variable("y")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("carbon", [x]),
                logic.PredicateExpression("nitrogen", [y]),
                logic.PredicateExpression("bond", [x, y]),
            ]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, witness = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND
        assert isinstance(witness, list)
        for item in witness:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], int)

    def test_witness_variable_names_preserved(self):
        universe, ext = _make_chain_structure(6)
        x, y = logic.Variable("MyVar"), logic.Variable("Other")
        formula = logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, [x, y],
            logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, [
                logic.PredicateExpression("carbon", [x]),
                logic.PredicateExpression("nitrogen", [y]),
            ]),
        )
        checker = FastModelChecker(universe, ext)
        outcome, witness = checker.find_model(formula)
        assert outcome == ModelCheckerOutcome.MODEL_FOUND
        names = {item[0] for item in witness}
        assert "MyVar" in names
        assert "Other" in names


# ---------------------------------------------------------------------------
# Benchmarks (not assertions, just timing)
# ---------------------------------------------------------------------------

class TestBenchmarks:

    def test_scaling_comparison(self):
        """Show how both checkers scale with molecule size."""
        print("\n--- Scaling comparison ---")
        print(f"{'N atoms':>10} {'Vars':>6} {'Original':>12} {'Fast':>12} {'Speedup':>10}")

        for n_atoms in [10, 20, 30, 50]:
            universe, ext = _make_chain_structure(n_atoms)
            n_vars = min(6, n_atoms // 3)
            vs = [logic.Variable(f"v{i}") for i in range(n_vars)]
            conjuncts = []
            for i in range(n_vars - 1):
                conjuncts.append(logic.PredicateExpression("bond", [vs[i], vs[i + 1]]))
            for i in range(n_vars):
                pred = "carbon" if i % 2 == 0 else "nitrogen"
                conjuncts.append(logic.PredicateExpression(pred, [vs[i]]))
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    conjuncts.append(logic.BinaryFormula(vs[i], logic.BinaryConnective.NEQ, vs[j]))

            formula = logic.QuantifiedFormula(
                logic.Quantifier.EXISTENTIAL, vs,
                logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, conjuncts),
            )
            normalized = normalize_fol_formula(formula)

            orig = OriginalModelChecker(universe, ext)
            fast = FastModelChecker(universe, ext)

            t0 = time.perf_counter()
            orig.find_model(normalized, timeout=10)
            t_orig = time.perf_counter() - t0

            t0 = time.perf_counter()
            fast.find_model(formula, timeout=10)
            t_fast = time.perf_counter() - t0

            speedup = t_orig / t_fast if t_fast > 0 else float("inf")
            print(f"{n_atoms:>10} {n_vars:>6} {t_orig:>11.4f}s {t_fast:>11.4f}s {speedup:>9.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
