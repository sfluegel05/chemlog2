import itertools
import logging
import queue
import time
from enum import Enum
from functools import wraps
from typing import Dict, List, Optional, Tuple

import numpy as np
from gavel.logic import logic
from gavel.logic.logic_utils import (
    get_vars_in_formula,
    substitute_var_in_formula
)


def _ensure_bool(func):
    """Wrapper that converts numpy array return values to bool via any()."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, np.ndarray):
            res = bool(res.any())
        return res
    return wrapper


class ModelCheckerOutcome(Enum):
    MODEL_FOUND = 0
    NO_MODEL = 1
    TIMEOUT = 2
    ERROR = 3
    MODEL_FOUND_INFERRED = 4
    NO_MODEL_INFERRED = 5
    UNKNOWN = 6


class ModelCheckerInputError(ValueError):
    """Raised when the input formula or predicate definitions are malformed."""



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

    @staticmethod
    def _format_name_list(names: List[str]) -> str:
        quoted = [f"'{name}'" for name in names]
        if len(quoted) == 1:
            return quoted[0]
        if len(quoted) == 2:
            return f"{quoted[0]} and {quoted[1]}"
        return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"

    def _predicate_arity(self, predicate_name: str) -> Optional[int]:
        if predicate_name in self.extensions:
            extension = self.extensions[predicate_name]
            if isinstance(extension, np.ndarray):
                return extension.ndim
            return 0
        if predicate_name in self.definitions:
            return len(self.definitions[predicate_name][0])
        return None

    def _validate_literal_arguments(self, literal: logic.PredicateExpression) -> Tuple[int, ...]:
        validated_arguments = []
        for argument in literal.arguments:
            if isinstance(argument, (int, np.integer)):
                validated_arguments.append(int(argument))
                continue
            if isinstance(argument, logic.Constant):
                constant_name = str(argument)
                if constant_name in self.extensions or constant_name in self.definitions:
                    raise ModelCheckerInputError(
                        f"Predicate '{constant_name}' is being used as a constant in the formula. "
                        f"Please check the formula and ensure that predicates are not used as constants."
                    )
                raise ModelCheckerInputError(
                    f"Invalid constant '{constant_name}' used as argument in predicate '{literal.predicate}'. "
                    f"Expected a bound variable or integer individual index."
                )
            if isinstance(argument, logic.Variable):
                raise ModelCheckerInputError(
                    f"Variable '{argument}' in predicate '{literal.predicate}' is not bound "
                    f"at evaluation time. Please add a quantifier or include it in the predicate definition head."
                )
            raise ModelCheckerInputError(
                f"Unsupported argument type '{type(argument).__name__}' in predicate '{literal.predicate}'."
            )
        return tuple(validated_arguments)

    def _validate_formula_inputs(self, formula: logic.LogicElement):
        known_predicates = set(self.extensions.keys()).union(self.definitions.keys())
        missing_predicates = set()
        predicate_constants = set()
        arity_errors: List[str] = []
        seen_arity_errors = set()
        unbound_definition_variables = set()

        def walk_predicates(expr: logic.LogicElement, bound_variables: Optional[set] = None, definition_name: Optional[str] = None):
            if bound_variables is None:
                bound_variables = set()

            if isinstance(expr, logic.PredicateExpression):
                predicate_name = str(expr.predicate)
                expected_arity = self._predicate_arity(predicate_name)
                actual_arity = len(expr.arguments)

                if expected_arity is None:
                    # Unknown predicates with arguments are treated as empty relations
                    # because atom-level predicates can be absent for specific molecules.
                    if actual_arity == 0:
                        missing_predicates.add(predicate_name)
                elif expected_arity != actual_arity:
                    allow_global_shortcut = (
                        predicate_name in self.extensions
                        and expected_arity == 1
                        and actual_arity == 0
                    )
                    if allow_global_shortcut:
                        expected_arity = actual_arity

                if expected_arity is not None and expected_arity != actual_arity:
                    key = (predicate_name, expected_arity, actual_arity)
                    if key not in seen_arity_errors:
                        seen_arity_errors.add(key)
                        arity_errors.append(
                            f"Predicate `{predicate_name}` is defined with arity {expected_arity} "
                            f"but called with {actual_arity} arguments"
                        )

                for argument in expr.arguments:
                    if isinstance(argument, logic.Constant):
                        constant_name = str(argument)
                        if constant_name in known_predicates:
                            predicate_constants.add(constant_name)
                    elif isinstance(argument, logic.Variable) and definition_name is not None:
                        variable_name = str(argument)
                        if variable_name not in bound_variables and predicate_name in known_predicates:
                            unbound_definition_variables.add((variable_name, definition_name))
                return

            if isinstance(expr, logic.QuantifiedFormula):
                quantifier_bound_vars = set(bound_variables)
                quantifier_bound_vars.update(str(v) for v in expr.variables)
                walk_predicates(expr.formula, quantifier_bound_vars, definition_name)
                return

            if isinstance(expr, logic.UnaryFormula):
                walk_predicates(expr.formula, bound_variables, definition_name)
                return

            if isinstance(expr, logic.BinaryFormula):
                walk_predicates(expr.left, bound_variables, definition_name)
                walk_predicates(expr.right, bound_variables, definition_name)
                return

            if isinstance(expr, logic.NaryFormula):
                for sub_formula in expr.formulae:
                    walk_predicates(sub_formula, bound_variables, definition_name)

        walk_predicates(formula)

        predicates_to_validate = {
            str(expr.predicate)
            for expr in get_predicate_expressions(formula)
            if str(expr.predicate) in self.definitions
        }
        visited_definitions = set()
        while predicates_to_validate:
            predicate_name = predicates_to_validate.pop()
            if predicate_name in visited_definitions or predicate_name not in self.definitions:
                continue
            visited_definitions.add(predicate_name)
            definition_vars, definition_formula = self.definitions[predicate_name]
            bound_vars = {str(v) for v in definition_vars}
            walk_predicates(definition_formula, bound_vars, predicate_name)

            for predicate_expr in get_predicate_expressions(definition_formula):
                sub_predicate_name = str(predicate_expr.predicate)
                if sub_predicate_name in self.definitions and sub_predicate_name not in visited_definitions:
                    predicates_to_validate.add(sub_predicate_name)

        error_messages = []
        if predicate_constants:
            names = sorted(predicate_constants)
            if len(names) == 1:
                error_messages.append(
                    f"Predicate {self._format_name_list(names)} is being used as a constant in the formula. "
                    f"Please check the formula and ensure that predicates are not used as constants."
                )
            else:
                error_messages.append(
                    f"Predicates {self._format_name_list(names)} are being used as constants in the formula. "
                    f"Please check the formula and ensure that predicates are not used as constants."
                )

        if missing_predicates:
            names = sorted(missing_predicates)
            if len(names) == 1:
                error_messages.append(f"Predicate {self._format_name_list(names)} is not defined")
            else:
                error_messages.append(f"Predicates {self._format_name_list(names)} are not defined")

        error_messages.extend(arity_errors)

        for variable_name, definition_name in sorted(unbound_definition_variables):
            error_messages.append(
                f"Variable '{variable_name}' is used in the definition of predicate '{definition_name}' "
                f"but is not bound by predicate arguments or quantifiers."
            )

        if error_messages:
            raise ModelCheckerInputError("\n".join(error_messages))

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

    @_ensure_bool
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
                expected_arity = self._predicate_arity(literal.predicate)
                allow_global_shortcut = (
                    expected_arity == 1 and len(tuple(literal.arguments)) == 0
                )
                if expected_arity is not None and len(tuple(literal.arguments)) != expected_arity and not allow_global_shortcut:
                    raise ValueError(
                        f"Predicate `{literal.predicate}` is defined with arity"
                        f" {expected_arity} but called with"
                        f" {len(literal.arguments)} arguments"
                    )
                validated_arguments = self._validate_literal_arguments(literal)
                if len(literal.arguments) > 1:
                    res = self.extensions[literal.predicate][validated_arguments]
                elif len(literal.arguments) == 1:
                    res = self.extensions[literal.predicate][validated_arguments[0]]
                else:
                    res = self.extensions[literal.predicate]
            elif literal.predicate in self.definitions:
                expected_arity = self._predicate_arity(literal.predicate)
                if expected_arity is not None and len(tuple(literal.arguments)) != expected_arity:
                    raise ValueError(
                        f"Predicate `{literal.predicate}` is defined with arity"
                        f" {expected_arity} but called with"
                        f" {len(literal.arguments)} arguments"
                    )
                validated_arguments = self._validate_literal_arguments(literal)

                if np.isnan(
                        self.calculated_extensions[literal.predicate][
                            validated_arguments
                        ]
                ):
                    definition = self.definitions[literal.predicate]
                    # take definition formula, replace variables with literal arguments
                    # e.g. for literal abc(k) and definition abc(x) <=> \exists y: p(x, y) replace x with k,
                    # run model checking on \exists y: p(k, y)
                    def_formula = definition[1]
                    def_formula = substitute_n_vars_in_formula(
                        def_formula, {def_var: ind for def_var, ind in zip(definition[0], literal.arguments)}
                    )
                    logging.debug(
                        f">>> Starting definition model finding for {literal.predicate}, substituting "
                        f"{', '.join([str(ind) + '|->' + str(def_var) for ind, def_var in zip(literal.arguments, definition[0])])}"
                    )
                    model_found = (
                            self.find_model(def_formula, validate_input=False)[0]
                            == ModelCheckerOutcome.MODEL_FOUND
                    )
                    logging.debug(
                        f"<<< Adding {', '.join(str(arg) for arg in literal.arguments)} as "
                        f"{'positive' if model_found else 'negative'} to extension of {literal.predicate}"
                    )
                    self.calculated_extensions[literal.predicate][
                        validated_arguments
                    ] = model_found
                res = self.calculated_extensions[literal.predicate][validated_arguments]
            else:
                if len(literal.arguments) == 0:
                    raise ModelCheckerInputError(
                        f"Predicate '{literal.predicate}' is not defined"
                    )
                res = False
            return not res if negated else res
        elif isinstance(literal, logic.BinaryFormula):
            if literal.operator == logic.BinaryConnective.NEQ:
                return not self.extensions[logic.BinaryConnective.EQ.name][
                    literal.left, literal.right
                ]
            elif literal.operator == logic.BinaryConnective.EQ:
                return self.extensions[logic.BinaryConnective.EQ.name][
                    literal.left, literal.right
                ]
        elif isinstance(literal, logic.QuantifiedFormula) and (literal.quantifier == logic.Quantifier.UNIVERSAL):
            # for universal quantifiers, check if the formula is true for all individuals in the universe
            for assignment in itertools.product(
                    range(self.universe), repeat=len(list(literal.variables))
            ):
                if self.all_different and len(set(assignment)) != len(assignment):
                    continue
                substituted_formula = literal.formula
                substituted_formula = substitute_n_vars_in_formula(
                        substituted_formula, {var: ind for var, ind in zip(literal.variables, assignment)}
                    )
                res = self.find_model(substituted_formula)
                if res[0] in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                    return False if not negated else True
            return True if not negated else False
        raise NotImplementedError(
            f"literal {literal} is of type {type(literal)} - original input: {orig_literal} "
            f"of type {type(orig_literal)} with connective {orig_literal.connective}, "
            f"negated: {negated}, check-for-negation instance: {isinstance(literal, logic.UnaryFormula)}"
            f" and connective: {literal.connective == logic.UnaryConnective.NEGATION}"
        )



    def find_model_quantified(self, formula, timeout=30) -> (ModelCheckerOutcome, Optional[Tuple[str, int]]):
        # find model for PNF formula with mixed universal and existential quantifiers
        if isinstance(formula, logic.QuantifiedFormula):
            if formula.quantifier == logic.Quantifier.UNIVERSAL:
                for assignment in itertools.product(
                        range(self.universe), repeat=len(list(formula.variables))
                ):
                    if self.all_different and len(set(assignment)) != len(assignment):
                        continue
                    substituted_formula = formula.formula
                    substituted_formula = substitute_n_vars_in_formula(
                            substituted_formula, {var: ind for var, ind in zip(formula.variables, assignment)}
                        )
                    res = self.find_model(substituted_formula, timeout, validate_input=False)
                    if res[0] in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                        return ModelCheckerOutcome.NO_MODEL, None
                return ModelCheckerOutcome.MODEL_FOUND, dict()
            else:
                if not isinstance(formula.formula, logic.QuantifiedFormula):
                    # innermost quantifier
                    return self.find_model(formula, timeout, validate_input=False)
                for assignment in itertools.product(
                        range(self.universe), repeat=len(list(formula.variables))
                ):
                    if self.all_different and len(set(assignment)) != len(assignment):
                        continue
                    substituted_formula = formula.formula
                    substituted_formula = substitute_n_vars_in_formula(
                            substituted_formula, {var: ind for var, ind in zip(formula.variables, assignment)}
                    )
                    res = self.find_model(substituted_formula, timeout, validate_input=False)
                    if res[0] in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                        return ModelCheckerOutcome.MODEL_FOUND, {**{var: ind for var, ind in zip(formula.variables, assignment)},
                                                                 **(res[1] if res[1] is not None else {})}
                return ModelCheckerOutcome.NO_MODEL, None
        else:
            return self.find_model(formula, timeout, validate_input=False)


    def find_model(
            self, formula, timeout=30, validate_input=True
    ) -> (ModelCheckerOutcome, Optional[Tuple[str, int]]):
        """Recursive strategy, insert one individual in the formula at a time, assume formula in PNF, CNF with
        only existential quantifiers"""
        if validate_input:
            self._validate_formula_inputs(formula)

        q = queue.LifoQueue()

        if isinstance(formula, logic.QuantifiedFormula):
            clauses = list(formula.formula.formulae)
            init_variables = formula.variables
        else:
            clauses = formula.formulae
            init_variables = []

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
            f"Starting find_model_existential with sanitized formula {formula}"
        )
        q.put((clauses, init_variables, []))
        start_time = time.perf_counter()

        while not q.empty():
            if timeout != 0 and (time.perf_counter() - start_time) > timeout:
                logging.warning(
                    f"Timed out after {(time.perf_counter() - start_time):.2f} seconds (timeout set to {timeout})"
                )
                return ModelCheckerOutcome.TIMEOUT, None
            clauses, variables, allocations = q.get()
            assert all(
                isinstance(clause, logic.NaryFormula)
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
                    if len(get_vars_in_formula(literal).intersection(variables)) > 0 or self.is_true(literal)
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
                if all(len(get_vars_in_formula(lit).intersection(variables)) > 0 for lit in clause.formulae)
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
                (clause, get_vars_in_formula(clause).intersection(variables))
                for clause in clauses
                if len(get_vars_in_formula(clause).intersection(variables)) == 1
            ]

            logging.debug(
                f"Using clauses with one variable: "
                f"{', '.join([str(pred) for pred, _ in clauses_one_var])}"
            )
            clauses_one_var_by_var = {}
            for clause_idx, (clause, vars) in enumerate(clauses_one_var):
                var = vars.pop()
                var_str = str(var)
                if var_str not in clauses_one_var_by_var:
                    clauses_one_var_by_var[var_str] = []
                clauses_one_var_by_var[var_str].append([substitute_var_in_formula(clause, var, const) for const in range(self.universe)])
            for var_str, clauses_var in clauses_one_var_by_var.items():
                for clauses_v in clauses_var:
                    possible_substitutes_for_clause = [
                        self.get_possible_substitutes(
                            [
                                clauses_v[sub].formulae[i]
                                for sub in possible_substitutes[var_str]
                            ],
                            possible_substitutes[str(var_str)],
                        )
                        for i in range(len(clauses_v[0].formulae))
                    ]
                    possible_substitutes[var_str] = list(
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

def substitute_n_vars_in_formula(
    formula: logic.LogicElement, substitutions: Dict
):
    """Replace every occurrence of the key variables with the given value. If var is None, replace all variables with ind"""
    if isinstance(formula, logic.NaryFormula):
        return logic.NaryFormula(
            formula.operator,
            [substitute_n_vars_in_formula(f, substitutions) for f in formula.formulae],
        )
    elif isinstance(formula, logic.PredicateExpression):
        return logic.PredicateExpression(
            formula.predicate,
            [substitutions.get(arg, arg) for arg in formula.arguments
            ],
        )
    elif isinstance(formula, logic.UnaryFormula):
        return logic.UnaryFormula(
            formula.connective, substitute_n_vars_in_formula(formula.formula, substitutions)
        )
    elif isinstance(formula, logic.QuantifiedFormula):
        variables = [arg for arg in formula.variables if arg not in substitutions]
        return logic.QuantifiedFormula(
            formula.quantifier,
            variables,
            substitute_n_vars_in_formula(formula.formula, substitutions),
        )
    elif isinstance(formula, logic.BinaryFormula):
        left = substitute_n_vars_in_formula(formula.left, substitutions)
        right = substitute_n_vars_in_formula(formula.right, substitutions)
        return logic.BinaryFormula(left, formula.operator, right)
    elif isinstance(formula, logic.Variable):
        return substitutions.get(formula, formula)
    return formula


def get_predicate_expressions(formula: logic.LogicElement) -> List[logic.PredicateExpression]:
    """Collect all predicate expressions occurring in a formula tree."""
    if isinstance(formula, logic.PredicateExpression):
        return [formula]
    if isinstance(formula, logic.QuantifiedFormula):
        return get_predicate_expressions(formula.formula)
    if isinstance(formula, logic.UnaryFormula):
        return get_predicate_expressions(formula.formula)
    if isinstance(formula, logic.BinaryFormula):
        return get_predicate_expressions(formula.left) + get_predicate_expressions(formula.right)
    if isinstance(formula, logic.NaryFormula):
        return [expr for sub_formula in formula.formulae for expr in get_predicate_expressions(sub_formula)]
    return []

def replace_vars_in_clause(clause, const):
    return logic.NaryFormula(
        logic.BinaryConnective.DISJUNCTION, [
            logic.UnaryFormula(
                logic.UnaryConnective.NEGATION,
                logic.PredicateExpression(
                    literal.formula.predicate, [
                        const if isinstance(arg, logic.Variable) else arg for arg in literal.formula.arguments
                    ],
                ),
            ) if isinstance(literal, logic.UnaryFormula)
            else logic.PredicateExpression(literal.predicate, [
                const if isinstance(arg, logic.Variable) else arg for arg in literal.arguments
                ]) if isinstance(literal, logic.PredicateExpression)
                else logic.BinaryFormula(
                    const if isinstance(literal.left, logic.Variable) else literal.left,
                    literal.operator,
                    const if isinstance(literal.right, logic.Variable) else literal.right,
                )
            for literal in clause.formulae
        ],
    )