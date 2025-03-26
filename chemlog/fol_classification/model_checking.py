import itertools
import logging
import queue
import time
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np
from gavel.logic import logic
from gavel.logic.logic import NaryFormula
from gavel.logic.logic_utils import substitute_var_in_formula, get_vars_in_formula, binary_to_nary


class ModelCheckerOutcome(Enum):
    MODEL_FOUND = 0
    NO_MODEL = 1
    TIMEOUT = 2
    ERROR = 3
    MODEL_FOUND_INFERRED = 4
    NO_MODEL_INFERRED = 5
    UNKNOWN = 6



class AbstractModelChecker:

    def __init__(self, universe: int, predicate_extensions, predicate_definitions=None):
        if predicate_definitions is None:
            predicate_definitions = {}
        self.universe = universe
        self.extensions = predicate_extensions
        self.definitions = predicate_definitions

    def find_model(
            self, formula, timeout=0
    ) -> (ModelCheckerOutcome, Optional[Tuple[str, int]]):
        raise NotImplementedError


class ModelChecker(AbstractModelChecker):
    """Model checker for first-order logic formulas. Expects input in gavels internal representation.
    Expects normalised formulas in PNF, CNF with only existential quantifiers. Not build for n-ary predicates
     where n > 2"""

    def __init__(
            self,
            universe: int,
            predicate_extensions: Dict,
            predicate_definitions: Optional[
                Dict[str, Tuple[List[logic.Variable], logic.QuantifiedFormula]]
            ] = None,
            precalculate_predicates: Optional[List[Tuple[str, int]]] = None,
            all_different: bool = False,
    ):
        super().__init__(universe, predicate_extensions, predicate_definitions)
        # if all_different, assign each instance to max. 1 variable
        self.all_different = all_different

        extensions_str = ""
        def_str = "\n\t\t".join(
            [
                f"{logic.PredicateExpression(name, vs)} <=> {formula}"
                for name, (vs, formula) in self.definitions.items()
            ]
        )
        if logging.getLogger().level == logging.DEBUG:
            for key, values in self.extensions.items():
                extensions_str += f"\t\t{key} "
                if len(values.shape) in [1, 2]:
                    if len(values.shape) == 1:
                        values_sparse_str = [
                            str(i) for i in range(values.shape[0]) if values[i] != 0
                        ]
                    else:
                        values_sparse_str = [
                            f"({i},{j})"
                            for i in range(values.shape[0])
                            for j in range(values.shape[1])
                            if values[i][j] != 0
                        ]
                    extensions_str += f"(total: {len(values_sparse_str)}): "
                    extensions_str += ", ".join(values_sparse_str)
                else:
                    extensions_str += str(values)
                extensions_str += "\n"
                # extensions_str += f"{', '.join(['(' + ','.join([str(c).split('_')[-1] if '_' in str(c) else str(c) for c in v]) + ')' for v in values])}\n"
                # except TypeError:
                #    extensions_str += f"{', '.join([str(v).split('_')[-1] if '_' in str(v) else str(v) for v in values])}\n"
            logging.debug(
                f"Initialised ModelChecker with:\n"
                f"\tuniverse: {', '.join([str(u) for u in range(universe)])},"
                f"\n\textensions:\n{extensions_str}"
                f"\tdefinitions:\n\t\t{def_str}"
            )

        self.calculated_extensions = {
            key: np.empty(
                (
                    tuple(universe for _ in range(len(self.definitions[key][0])))
                    if len(self.definitions[key][0]) > 0
                    else 1
                ),
                dtype=np.bool_,
            )
                 * np.nan
            for key in self.definitions.keys()
        }

        if precalculate_predicates is not None:
            for predicate, arity in precalculate_predicates:
                self.precalculate_extension(predicate, arity)

        self.proven_formulae = (
            []
        )  # store already proven / disproven formulae to avoid doing it twice
        self.disproven_formulae = []

    def precalculate_extension(self, predicate: str, arity: int):
        """Recursively find all elements of the predicate extension, using the definition"""
        logging.info(f"Precalculating {arity}-ary predicate {predicate}")
        for inds in itertools.combinations_with_replacement(
                range(self.universe), arity
        ):
            formula = logic.PredicateExpression(predicate, list(inds))
            self.calculated_extensions[predicate][tuple(inds)] = self.is_true(formula)

    def get_possible_substitutes(
            self, substituted_literals: List, substitutes: List[int]
    ) -> set:
        """For a list of individuals, check which can replace a given variable in a given literal
        (without violating the extensions)"""
        return {
            ind
            for ind, sub in zip(substitutes, substituted_literals)
            if self.is_true(sub)
        }

    def is_true(self, literal: logic.LogicExpression) -> bool:
        """For ~P(...), P(...), a=b, b=a without variables"""
        # assert is_literal(literal)
        orig_literal = literal
        negated = False
        if (
                isinstance(literal, logic.UnaryFormula)
                and literal.connective == logic.UnaryConnective.NEGATION
        ):
            negated = True
            literal = literal.formula
        if isinstance(literal, logic.PredicateExpression):
            if literal.predicate in self.extensions:
                if len(literal.arguments) > 1:
                    res = self.extensions[literal.predicate][tuple(literal.arguments)]
                elif len(literal.arguments) == 1:
                    res = self.extensions[literal.predicate][literal.arguments[0]]
                else:
                    res = self.extensions[literal.predicate]
            elif literal.predicate in self.definitions:
                if np.isnan(
                        self.calculated_extensions[literal.predicate][
                            tuple(literal.arguments)
                        ]
                ):
                    definition = self.definitions[literal.predicate]
                    # take definition formula, replace variables with literal arguments
                    # e.g. for literal abc(k) and definition abc(x) <=> \exists y: p(x, y) replace x with k,
                    # run model checking on \exists y: p(k, y)
                    def_formula = deepcopy(definition[1])
                    for ind, def_var in zip(literal.arguments, definition[0]):
                        def_formula = substitute_var_in_formula(
                            def_formula, def_var, ind
                        )
                    logging.debug(
                        f">>> Starting definition model finding for {literal.predicate}, substituting "
                        f"{', '.join([str(ind) + '|->' + str(def_var) for ind, def_var in zip(literal.arguments, definition[0])])}"
                    )
                    model_found = (
                            self.find_model(def_formula)[0]
                            == ModelCheckerOutcome.MODEL_FOUND
                    )
                    logging.debug(
                        f"<<< Adding {', '.join(str(arg) for arg in literal.arguments)} as "
                        f"{'positive' if model_found else 'negative'} to extension of {literal.predicate}"
                    )
                    self.calculated_extensions[literal.predicate][
                        tuple(literal.arguments)
                    ] = model_found
                res = self.calculated_extensions[literal.predicate][tuple(literal.arguments)]
            else:
                res = False
            return not res if negated else res
        if isinstance(literal, logic.BinaryFormula):
            if literal.operator == logic.BinaryConnective.NEQ:
                return not self.extensions[logic.BinaryConnective.EQ.name][
                    literal.left, literal.right
                ]
            elif literal.operator == logic.BinaryConnective.EQ:
                return self.extensions[logic.BinaryConnective.EQ.name][
                    literal.left, literal.right
                ]
        raise NotImplementedError(
            f"literal {literal} is of type {type(literal)} - original input: {orig_literal} "
            f"of type {type(orig_literal)} with connective {orig_literal.connective}, "
            f"negated: {negated}, check-for-negation instance: {isinstance(literal, logic.UnaryFormula)}"
            f" and connective: {literal.connective == logic.UnaryConnective.NEGATION}"
        )

    def find_model(
            self, formula, timeout=30
    ) -> (ModelCheckerOutcome, Optional[Tuple[str, int]]):
        """Recursive strategy, insert one individual in the formula at a time, assume formula in PNF, CNF with
        only existential quantifiers"""
        q = queue.LifoQueue()
        if not isinstance(formula, logic.QuantifiedFormula):
            formula = logic.QuantifiedFormula(
                logic.Quantifier.EXISTENTIAL, [], formula
            )
        assert formula.quantifier == logic.Quantifier.EXISTENTIAL
        # no free variables
        assert all(
            var in formula.variables for var in get_vars_in_formula(formula.formula)
        ), (
            f"Formula contains free variables, namely "
            f"{set(str(var) for var in get_vars_in_formula(formula.formula) if var not in formula.variables)}"
        )
        # convert (chain of) binary conjunctions or single-clause cnf formula into n-ary conjunction
        if not (
                isinstance(formula.formula, NaryFormula)
                and formula.formula.operator == logic.BinaryConnective.CONJUNCTION
        ):
            formula.formula = binary_to_nary(
                formula.formula, logic.BinaryConnective.CONJUNCTION
            )
        clauses = list(formula.formula.formulae)
        for i, clause in enumerate(clauses):
            if not (
                    isinstance(clause, NaryFormula)
                    and clause.operator == logic.BinaryConnective.DISJUNCTION
            ):
                clauses[i] = binary_to_nary(clause, logic.BinaryConnective.DISJUNCTION)
        # TODO check how efficient this mechanism is
        if formula in self.proven_formulae:
            logging.debug(
                f"Skipping formula {formula} because it has already been proven"
            )
            return ModelCheckerOutcome.MODEL_FOUND_INFERRED, None
        elif formula in self.disproven_formulae:
            logging.debug(
                f"Skipping formula {formula} because it has already been disproven"
            )
            return ModelCheckerOutcome.NO_MODEL_INFERRED, None

        logging.debug(
            f"Starting find_model with sanitized formula {' & '.join(str(c) for c in clauses)}"
        )
        q.put((clauses, formula.variables, []))
        start_time = time.perf_counter()

        while not q.empty():
            if timeout != 0 and (time.perf_counter() - start_time) > timeout:
                logging.warning(
                    f"Timed out after {(time.perf_counter() - start_time):.2f} seconds (timeout set to {timeout})"
                )
                return ModelCheckerOutcome.TIMEOUT, None
            clauses, variables, allocations = q.get()
            assert all(
                isinstance(clause, NaryFormula)
                and clause.operator == logic.BinaryConnective.DISJUNCTION
                for clause in clauses
            )
            assert all(isinstance(var, logic.Variable) for var in variables)

            n_clauses_old = len(clauses)
            # in each clause, remove false literals
            for clause in clauses:
                literals = [
                    literal
                    for literal in clause.formulae
                    if len(get_vars_in_formula(literal)) > 0 or self.is_true(literal)
                ]
                if len(literals) == 0:
                    logging.debug(
                        f"Found contradiction: Clause '{str(clause)}' contradicts extensions"
                    )
                    self.disproven_formulae.append(formula)
                    return ModelCheckerOutcome.NO_MODEL, None
                clause.formulae = literals

            if len(variables) == 0:
                logging.debug(
                    f"Model found ({(time.perf_counter() - start_time):.2f}s): "
                    f"Variable assignments: {', '.join([f'{var} |-> {ind}' for var, ind in allocations])}"
                )
                self.proven_formulae.append(formula)
                return ModelCheckerOutcome.MODEL_FOUND, allocations
            logging.debug(
                f"Checking formula with{'out' if len(allocations) == 0 else ''} allocations "
                f"{', '.join([f'{var} |-> {ind}' for var, ind in allocations])}"
            )

            # remove clauses with at least one true literal
            clauses = [
                clause
                for clause in clauses
                if all(len(get_vars_in_formula(lit)) > 0 for lit in clause.formulae)
            ]

            logging.debug(
                f"Using {len(clauses)} clauses, discarding {n_clauses_old - len(clauses)} which are already fulfilled"
            )

            possible_substitutes = {
                str(var): [
                    u
                    for u in range(self.universe)
                    if u not in [allocation[1] for allocation in allocations]
                       or not self.all_different
                ]
                for var in variables
            }

            clauses_one_var = [
                (clause, get_vars_in_formula(clause))
                for clause in clauses
                if len(get_vars_in_formula(clause)) == 1
            ]

            logging.debug(
                f"Using clauses with one variable: "
                f"{', '.join([str(pred) for pred, _ in clauses_one_var])}"
            )
            clauses_one_var_by_var = {}
            for clause_idx, (clause, var) in enumerate(clauses_one_var):
                var = str(var.pop())
                if var not in clauses_one_var_by_var:
                    clauses_one_var_by_var[var] = []
                clauses_one_var_by_var[var].append([replace_vars_in_clause(clause, const) for const in range(self.universe)])
            for var, clauses_var in clauses_one_var_by_var.items():
                for clauses_v in clauses_var:
                    possible_substitutes_for_clause = [
                        self.get_possible_substitutes(
                            [
                                clauses_v[sub].formulae[i]
                                for sub in possible_substitutes[var]
                            ],
                            possible_substitutes[str(var)],
                        )
                        for i in range(len(clauses_v[0].formulae))
                    ]
                    possible_substitutes[var] = list(
                        set.union(*possible_substitutes_for_clause)
                    )
            logging.debug(
                f"Found possible assignments based on clauses with one variable: \n\t"
                + "\n\t".join(
                    [
                        f'{key}: {", ".join([str(elem) for elem in value])}'
                        for key, value in possible_substitutes.items()
                    ]
                )
            )

            variables = sorted(
                variables, key=lambda var: len(possible_substitutes[str(var)])
            )

            for ind in possible_substitutes[str(variables[0])]:
                new_vars = variables[1:]
                new_clauses = [
                    substitute_var_in_formula(clause, variables[0], ind)
                    for clause in clauses
                ]
                new_allocations = [(a[0], a[1]) for a in allocations]
                new_allocations.append((str(variables[0]), ind))

                q.put((new_clauses, new_vars, new_allocations))
            if len(possible_substitutes[str(variables[0])]) > 0:
                logging.debug(
                    f"Putting {variables[0]} |-> "
                    f"{', '.join(str(ind) for ind in possible_substitutes[str(variables[0])])} in queue"
                )

        self.disproven_formulae.append(formula)
        return ModelCheckerOutcome.NO_MODEL, None

def replace_vars_in_clause(clause, const):
    return NaryFormula(
                        logic.BinaryConnective.DISJUNCTION,
                        [
                            (
                                logic.UnaryFormula(
                                    logic.UnaryConnective.NEGATION,
                                    logic.PredicateExpression(
                                        literal.formula.predicate,
                                        [
                                            (
                                                const
                                                if isinstance(arg, logic.Variable)
                                                else arg
                                            )
                                            for arg in literal.formula.arguments
                                        ],
                                    ),
                                )
                                if isinstance(literal, logic.UnaryFormula)
                                else (
                                    logic.PredicateExpression(
                                        literal.predicate,
                                        [
                                            (
                                                const
                                                if isinstance(arg, logic.Variable)
                                                else arg
                                            )
                                            for arg in literal.arguments
                                        ],
                                    )
                                    if isinstance(literal, logic.PredicateExpression)
                                    else logic.BinaryFormula(
                                        (
                                            const
                                            if isinstance(literal.left, logic.Variable)
                                            else literal.left
                                        ),
                                        literal.operator,
                                        (
                                            const
                                            if isinstance(literal.right, logic.Variable)
                                            else literal.right
                                        ),
                                    )
                                )
                            )
                            for literal in clause.formulae
                        ],
                    )