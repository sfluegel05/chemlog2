"""Basic unit tests for ``ModelChecker.find_model``.

These tests construct ``ModelChecker`` directly from small numpy extensions and
gavel formulas (rather than going through the RDKit/parser pipeline), so they
exercise ``find_model`` in isolation and document the expected outcome for a
handful of simple extension/formula combinations.

``find_model`` expects a normalised formula in prenex CNF with only existential
quantifiers: the matrix is a conjunction (``NaryFormula`` with ``CONJUNCTION``)
of clauses, and every clause is a disjunction (``NaryFormula`` with
``DISJUNCTION``) of literals.
"""

import numpy as np
import pytest
from gavel.logic import logic

from chemlog.fol_classification.model_checking import (
    ModelChecker,
    ModelCheckerOutcome,
)


@pytest.fixture
def simple_checker():
    """Universe of 3 with one unary and one binary predicate.

    p = {0, 1}; edge = {(0, 1), (1, 2)}.
    """
    universe = 3
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "p": np.array([True, True, False]),
        "edge": np.array(
            [
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ]
        ),
    }
    return ModelChecker(universe, extensions)


# --- ground (variable-free) formulas ----------------------------------------------


def test_ground_literal_true(simple_checker):
    formula = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("p", [0])],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND


def test_ground_literal_false(simple_checker):
    formula = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("p", [2])],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.NO_MODEL


def test_ground_negated_literal(simple_checker):
    # ~p(2) holds because 2 is not in the extension of p.
    holds = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [
                    logic.UnaryFormula(
                        logic.UnaryConnective.NEGATION,
                        logic.PredicateExpression("p", [2]),
                    )
                ],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(holds)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND

    # ~p(0) fails because 0 is in the extension of p.
    fails = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [
                    logic.UnaryFormula(
                        logic.UnaryConnective.NEGATION,
                        logic.PredicateExpression("p", [0]),
                    )
                ],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(fails)
    assert outcome == ModelCheckerOutcome.NO_MODEL


def test_ground_disjunction(simple_checker):
    # One true disjunct is enough to satisfy the clause: p(2) is false but p(0) is true.
    satisfiable = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [
                    logic.PredicateExpression("p", [2]),
                    logic.PredicateExpression("p", [0]),
                ],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(satisfiable)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND

    # Both disjuncts false -> no model.
    unsatisfiable = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [
                    logic.PredicateExpression("p", [2]),
                    logic.PredicateExpression("edge", [2, 2]),
                ],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(unsatisfiable)
    assert outcome == ModelCheckerOutcome.NO_MODEL


def test_ground_binary_predicate(simple_checker):
    present = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("edge", [0, 1])],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(present)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND

    # edge is directed; (1, 0) is not in the extension.
    absent = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("edge", [1, 0])],
            )
        ],
    )
    outcome, _ = simple_checker.find_model(absent)
    assert outcome == ModelCheckerOutcome.NO_MODEL


def test_ground_conjunction_of_clauses(simple_checker):
    # p(0) & edge(0, 1) both hold.
    satisfiable = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("p", [0])],
            ),
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("edge", [0, 1])],
            ),
        ],
    )
    outcome, _ = simple_checker.find_model(satisfiable)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND

    # p(0) holds but edge(0, 2) does not -> the conjunction fails.
    unsatisfiable = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("p", [0])],
            ),
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("edge", [0, 2])],
            ),
        ],
    )
    outcome, _ = simple_checker.find_model(unsatisfiable)
    assert outcome == ModelCheckerOutcome.NO_MODEL


# --- existentially quantified formulas --------------------------------------------


def test_existential_single_variable(simple_checker):
    x = logic.Variable("X")
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("p", [x])],
                )
            ],
        ),
    )
    outcome, _ = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND


def test_existential_no_witness():
    universe = 3
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "p": np.array([False, False, False]),
    }
    checker = ModelChecker(universe, extensions)
    x = logic.Variable("X")
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("p", [x])],
                )
            ],
        ),
    )
    outcome, _ = checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.NO_MODEL


def test_existential_two_variables(simple_checker):
    x, y = logic.Variable("X"), logic.Variable("Y")
    # There exist X, Y with edge(X, Y).
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x, y],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("edge", [x, y])],
                )
            ],
        ),
    )
    outcome, _ = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND


def test_existential_returns_allocations(simple_checker):
    x, y = logic.Variable("X"), logic.Variable("Y")
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x, y],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("edge", [x, y])],
                )
            ],
        ),
    )
    outcome, allocations = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND
    assignment = dict(allocations)
    # The only edges are (0, 1) and (1, 2); whichever is picked must be a real edge.
    assert (assignment["X"], assignment["Y"]) in {(0, 1), (1, 2)}


def test_existential_with_inequality(simple_checker):
    x, y = logic.Variable("X"), logic.Variable("Y")
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x, y],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("edge", [x, y])],
                ),
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.BinaryFormula(x, logic.BinaryConnective.NEQ, y)],
                ),
            ],
        ),
    )
    outcome, _ = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND


def test_existential_with_negated_literal(simple_checker):
    x = logic.Variable("X")
    # exists X: p(X) & ~edge(X, 0). p = {0, 1}; neither has an edge into 0, so any holds.
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("p", [x])],
                ),
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [
                        logic.UnaryFormula(
                            logic.UnaryConnective.NEGATION,
                            logic.PredicateExpression("edge", [x, 0]),
                        )
                    ],
                ),
            ],
        ),
    )
    outcome, _ = simple_checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND


def test_existential_shared_variable_unsatisfiable():
    # p and q have no common witness, so "exists X: p(X) & q(X)" has no model.
    universe = 3
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "p": np.array([True, False, False]),
        "q": np.array([False, True, False]),
    }
    checker = ModelChecker(universe, extensions)
    x = logic.Variable("X")
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("p", [x])],
                ),
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("q", [x])],
                ),
            ],
        ),
    )
    outcome, _ = checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.NO_MODEL


# --- all_different ----------------------------------------------------------------


def test_all_different_blocks_reused_individual():
    # Only individual 0 satisfies p, so two *distinct* witnesses cannot both satisfy it.
    universe = 2
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "p": np.array([True, False]),
    }
    x, y = logic.Variable("X"), logic.Variable("Y")
    formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [x, y],
        logic.NaryFormula(
            logic.BinaryConnective.CONJUNCTION,
            [
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("p", [x])],
                ),
                logic.NaryFormula(
                    logic.BinaryConnective.DISJUNCTION,
                    [logic.PredicateExpression("p", [y])],
                ),
            ],
        ),
    )

    # Without all_different the same individual can fill both variables.
    outcome, _ = ModelChecker(universe, extensions).find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND

    # With all_different each variable needs its own individual -> no model.
    outcome, _ = ModelChecker(universe, extensions, all_different=True).find_model(formula)
    assert outcome == ModelCheckerOutcome.NO_MODEL


# --- predicate definitions --------------------------------------------------------


def test_definition_satisfied():
    # Definition hasP <=> ?[X]: p(X) (a 0-ary derived predicate).
    universe = 3
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "p": np.array([False, True, False]),
    }
    x = logic.Variable("X")
    definitions = {
        "hasP": (
            [],
            logic.QuantifiedFormula(
                logic.Quantifier.EXISTENTIAL,
                [x],
                logic.NaryFormula(
                    logic.BinaryConnective.CONJUNCTION,
                    [
                        logic.NaryFormula(
                            logic.BinaryConnective.DISJUNCTION,
                            [logic.PredicateExpression("p", [x])],
                        )
                    ],
                ),
            ),
        )
    }
    checker = ModelChecker(universe, extensions, definitions)
    formula = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("hasP", [])],
            )
        ],
    )
    outcome, _ = checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.MODEL_FOUND


def test_definition_unsatisfied():
    universe = 3
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "p": np.array([False, False, False]),
    }
    x = logic.Variable("X")
    definitions = {
        "hasP": (
            [],
            logic.QuantifiedFormula(
                logic.Quantifier.EXISTENTIAL,
                [x],
                logic.NaryFormula(
                    logic.BinaryConnective.CONJUNCTION,
                    [
                        logic.NaryFormula(
                            logic.BinaryConnective.DISJUNCTION,
                            [logic.PredicateExpression("p", [x])],
                        )
                    ],
                ),
            ),
        )
    }
    checker = ModelChecker(universe, extensions, definitions)
    formula = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("hasP", [])],
            )
        ],
    )
    outcome, _ = checker.find_model(formula)
    assert outcome == ModelCheckerOutcome.NO_MODEL


# --- inferred outcomes (proven/disproven cache) -----------------------------------


def test_repeated_formula_is_inferred(simple_checker):
    formula = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("p", [0])],
            )
        ],
    )
    first, _ = simple_checker.find_model(formula)
    second, _ = simple_checker.find_model(formula)
    assert first == ModelCheckerOutcome.MODEL_FOUND
    # The second call short-circuits via the proven-formulae cache.
    assert second == ModelCheckerOutcome.MODEL_FOUND_INFERRED


def test_repeated_unsatisfiable_formula_is_inferred(simple_checker):
    formula = logic.NaryFormula(
        logic.BinaryConnective.CONJUNCTION,
        [
            logic.NaryFormula(
                logic.BinaryConnective.DISJUNCTION,
                [logic.PredicateExpression("p", [2])],
            )
        ],
    )
    first, _ = simple_checker.find_model(formula)
    second, _ = simple_checker.find_model(formula)
    assert first == ModelCheckerOutcome.NO_MODEL
    assert second == ModelCheckerOutcome.NO_MODEL_INFERRED
