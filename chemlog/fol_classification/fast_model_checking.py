"""
Fast FOL model checker for molecular graph structures.

Optimized replacement for the recursive/queue-based checker in model_checking.py.
Targets the specific structure of ChEBI classification queries:
  - Domain: molecule atoms (10-100 elements)
  - Predicates: unary (element type, charge, H-count) + binary (bonds, degree <= 6)
  - Formulas: existential-conjunctive core, with optional forall/negation
  - Quantifier depth: 5-10

Key optimizations:
  1. Compile formula to integer-indexed ops — no AST walks during search
  2. Adjacency-indexed domain restriction for binary predicates
  3. Forward checking with constraint propagation
  4. Unary pre-filtering (carbon(x) ∧ nitrogen(y) restricts domains before search)
  5. Work in NNF directly — skip exponential CNF blowup
  6. Cache defined predicate evaluations per argument tuple
  7. Fail-first variable ordering (smallest domain first)
"""

import itertools
import logging
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np
from gavel.logic import logic

from chemlog.fol_classification.model_checking import (
    ModelCheckerOutcome,
    AbstractModelChecker,
)


# ---------------------------------------------------------------------------
# Compiled formula representation
# ---------------------------------------------------------------------------

class _Op(Enum):
    ATOM = 0
    NEG = 1
    AND = 2
    OR = 3
    FORALL = 4
    EXISTS = 5
    EQ = 6
    NEQ = 7


class _CNode:
    """Compiled formula node — flat, integer-indexed.

    Slots-based for memory efficiency (many nodes per formula).
    """

    __slots__ = ("op", "pred_name", "args", "children", "qvars", "qvar_names", "free_vars")

    def __init__(
        self,
        op,
        pred_name="",
        args=(),
        children=(),
        qvars=(),
        qvar_names=(),
        free_vars=frozenset(),
    ):
        self.op = op
        self.pred_name = pred_name
        self.args = args          # tuple of (is_var: bool, index: int)
        self.children = children  # tuple of _CNode
        self.qvars = qvars        # tuple of int (variable IDs)
        self.qvar_names = qvar_names  # tuple of str (original variable names for witnesses)
        self.free_vars = free_vars


def _var_key(v):
    if isinstance(v, logic.Variable):
        return ("var", v.symbol)
    return ("var", str(v))


def _var_name(v):
    if isinstance(v, logic.Variable):
        return v.symbol
    return str(v)


def _compile_arg(arg, var_map):
    """Returns (is_variable: bool, index_or_const: int)."""
    if isinstance(arg, logic.Variable):
        key = _var_key(arg)
        if key in var_map:
            return (True, var_map[key])
        raise ValueError(f"Unbound variable: {arg} (symbol={arg.symbol})")
    elif isinstance(arg, (int, np.integer)):
        return (False, int(arg))
    else:
        raise ValueError(f"Unexpected argument type: {type(arg)} = {arg}")


def compile_formula(formula, var_map=None, next_var_id=None):
    """Compile a gavel logic formula into a _CNode tree.

    Parameters
    ----------
    formula : gavel LogicElement
    var_map : dict, maps _var_key(v) -> int ID
    next_var_id : list of [int], mutable counter for fresh variable IDs
    """
    if var_map is None:
        var_map = {}
    if next_var_id is None:
        next_var_id = [0]

    if isinstance(formula, logic.QuantifiedFormula):
        qvar_ids = []
        qvar_names = []
        for v in formula.variables:
            key = _var_key(v)
            vid = next_var_id[0]
            next_var_id[0] += 1
            var_map[key] = vid
            qvar_ids.append(vid)
            qvar_names.append(_var_name(v))

        child = compile_formula(formula.formula, var_map, next_var_id)
        op = _Op.EXISTS if formula.quantifier == logic.Quantifier.EXISTENTIAL else _Op.FORALL
        free = child.free_vars - frozenset(qvar_ids)
        return _CNode(
            op=op,
            qvars=tuple(qvar_ids),
            qvar_names=tuple(qvar_names),
            children=(child,),
            free_vars=free,
        )

    elif isinstance(formula, logic.NaryFormula):
        op = _Op.AND if formula.operator == logic.BinaryConnective.CONJUNCTION else _Op.OR
        children = tuple(compile_formula(f, var_map, next_var_id) for f in formula.formulae)
        free = frozenset().union(*(c.free_vars for c in children))
        return _CNode(op=op, children=children, free_vars=free)

    elif isinstance(formula, logic.BinaryFormula):
        if formula.operator in (logic.BinaryConnective.EQ, logic.BinaryConnective.NEQ):
            left = _compile_arg(formula.left, var_map)
            right = _compile_arg(formula.right, var_map)
            op = _Op.EQ if formula.operator == logic.BinaryConnective.EQ else _Op.NEQ
            free = frozenset()
            if left[0]:
                free = free | {left[1]}
            if right[0]:
                free = free | {right[1]}
            return _CNode(op=op, args=(left, right), free_vars=free)
        elif formula.operator == logic.BinaryConnective.CONJUNCTION:
            op = _Op.AND
        elif formula.operator == logic.BinaryConnective.DISJUNCTION:
            op = _Op.OR
        else:
            raise NotImplementedError(
                f"Binary operator {formula.operator} not supported — convert to NNF first"
            )
        children = tuple(
            compile_formula(f, var_map, next_var_id) for f in [formula.left, formula.right]
        )
        free = frozenset().union(*(c.free_vars for c in children))
        return _CNode(op=op, children=children, free_vars=free)

    elif isinstance(formula, logic.UnaryFormula):
        if formula.connective == logic.UnaryConnective.NEGATION:
            child = compile_formula(formula.formula, var_map, next_var_id)
            return _CNode(op=_Op.NEG, children=(child,), free_vars=child.free_vars)
        raise NotImplementedError(f"Unary connective {formula.connective}")

    elif isinstance(formula, logic.PredicateExpression):
        args = tuple(_compile_arg(a, var_map) for a in formula.arguments)
        free = frozenset(a[1] for a in args if a[0])
        return _CNode(op=_Op.ATOM, pred_name=formula.predicate, args=args, free_vars=free)

    elif isinstance(formula, (int, np.integer)):
        return _CNode(op=_Op.ATOM, pred_name="__const__", args=((False, int(formula)),))

    else:
        raise NotImplementedError(f"Cannot compile {type(formula)}: {formula}")


# ---------------------------------------------------------------------------
# Indexed structure for fast predicate lookup
# ---------------------------------------------------------------------------

class _IndexedStructure:
    """Wraps numpy extension arrays with pre-computed adjacency indices."""

    __slots__ = ("universe", "extensions", "domain", "neighbors", "members")

    def __init__(self, universe: int, extensions: Dict[str, np.ndarray]):
        self.universe = universe
        self.extensions = extensions
        self.domain = list(range(universe))

        # Adjacency lists for binary predicates
        self.neighbors: Dict[str, Dict[int, List[int]]] = {}
        for pred, ext in extensions.items():
            if isinstance(ext, np.ndarray) and ext.ndim == 2:
                adj: Dict[int, List[int]] = {}
                for i in range(universe):
                    js = np.where(ext[i])[0]
                    if len(js):
                        adj[i] = js.tolist()
                self.neighbors[pred] = adj

        # Members of unary predicates
        self.members: Dict[str, List[int]] = {}
        for pred, ext in extensions.items():
            if isinstance(ext, np.ndarray) and ext.ndim == 1:
                self.members[pred] = np.where(ext)[0].tolist()

    def eval_pred(self, pred_name: str, args: tuple) -> bool:
        ext = self.extensions.get(pred_name)
        if ext is None:
            return False
        if isinstance(ext, np.ndarray):
            ndim = ext.ndim
            if ndim == 0:
                return bool(ext)
            elif ndim == 1:
                return bool(ext[args[0]])
            elif ndim == 2:
                return bool(ext[args[0], args[1]])
            else:
                return bool(ext[args])
        return False


# ---------------------------------------------------------------------------
# Domain restriction analysis
# ---------------------------------------------------------------------------

def _restrict_domains(node, structure, domains, defined_preds_set):
    """Restrict variable domains by walking positive conjunctive spine.

    Only looks at base predicates (not defined ones) for speed.
    """
    op = node.op
    if op == _Op.AND:
        for child in node.children:
            _restrict_domains(child, structure, domains, defined_preds_set)
    elif op == _Op.EXISTS:
        _restrict_domains(node.children[0], structure, domains, defined_preds_set)
    elif op == _Op.ATOM:
        pred = node.pred_name
        if pred in defined_preds_set:
            return  # skip defined predicates — too expensive to pre-evaluate
        args = node.args
        if len(args) == 1 and args[0][0]:
            var_id = args[0][1]
            if var_id in domains and pred in structure.members:
                allowed = set(structure.members[pred])
                domains[var_id] = domains[var_id] & allowed
        elif len(args) == 2:
            a0, a1 = args
            if a0[0] and not a1[0] and a0[1] in domains:
                var_id, const = a0[1], a1[1]
                if pred in structure.neighbors:
                    adj = structure.neighbors[pred]
                    possible = set()
                    for elem in domains[var_id]:
                        if const in adj.get(elem, []):
                            possible.add(elem)
                    domains[var_id] = possible
            elif not a0[0] and a1[0] and a1[1] in domains:
                const, var_id = a0[1], a1[1]
                if pred in structure.neighbors:
                    nbrs = set(structure.neighbors[pred].get(const, []))
                    domains[var_id] = domains[var_id] & nbrs


# ---------------------------------------------------------------------------
# Binary constraint extraction for forward checking
# ---------------------------------------------------------------------------

class _BinConstraint:
    __slots__ = ("var1", "var2", "pred_name", "negated")

    def __init__(self, var1, var2, pred_name, negated):
        self.var1 = var1
        self.var2 = var2
        self.pred_name = pred_name
        self.negated = negated


def _extract_bin_constraints(node, constraints, negated=False, skip_preds=frozenset()):
    """Extract binary constraints, skipping defined predicates (not in base extensions)."""
    op = node.op
    if op == _Op.AND:
        for child in node.children:
            _extract_bin_constraints(child, constraints, negated, skip_preds)
    elif op == _Op.EXISTS:
        _extract_bin_constraints(node.children[0], constraints, negated, skip_preds)
    elif op == _Op.NEG:
        _extract_bin_constraints(node.children[0], constraints, not negated, skip_preds)
    elif op == _Op.ATOM and len(node.args) == 2:
        if node.pred_name in skip_preds:
            return
        a0, a1 = node.args
        if a0[0] and a1[0]:
            constraints.append(_BinConstraint(a0[1], a1[1], node.pred_name, negated))


# ---------------------------------------------------------------------------
# Core model checker
# ---------------------------------------------------------------------------

class FastModelChecker(AbstractModelChecker):
    """Fast FOL model checker optimized for molecular graph queries.

    Drop-in replacement for ``ModelChecker``.  Uses the same constructor
    signature and ``find_model`` / ``find_model_quantified`` return format.
    """

    def __init__(
        self,
        universe: int,
        predicate_extensions: Dict,
        predicate_definitions: Optional[
            Dict[str, Tuple[List[logic.Variable], logic.QuantifiedFormula]]
        ] = None,
        precalculate_predicates=None,   # accepted for API compat, ignored
        all_different: bool = False,
    ):
        super().__init__(universe, predicate_extensions, predicate_definitions)
        self.structure = _IndexedStructure(universe, predicate_extensions)
        self.all_different = all_different

        # Compile defined predicates once
        self._compiled_defs: Dict[str, Tuple[tuple, _CNode, tuple]] = {}
        # Also keep a set of defined predicate names for domain restriction
        self._defined_pred_names: set = set()
        if predicate_definitions:
            for pred_name, (vars_list, formula) in predicate_definitions.items():
                self._defined_pred_names.add(pred_name)
                var_map = {}
                next_var_id = [0]
                param_ids = []
                param_names = []
                for v in vars_list:
                    key = _var_key(v)
                    vid = next_var_id[0]
                    next_var_id[0] += 1
                    var_map[key] = vid
                    param_ids.append(vid)
                    param_names.append(_var_name(v))
                compiled = compile_formula(formula, var_map, next_var_id)
                self._compiled_defs[pred_name] = (
                    tuple(param_ids),
                    compiled,
                    tuple(param_names),
                )

        # Cache for evaluated defined predicates: (pred_name, args) -> bool
        self._pred_cache: Dict[Tuple, bool] = {}

        # Caches for proven/disproven formulae (string-keyed for compat)
        self.proven_formulae = []
        self.disproven_formulae = []

        if precalculate_predicates is not None:
            for predicate, arity in precalculate_predicates:
                self._precalculate(predicate, arity)

    def _precalculate(self, predicate: str, arity: int):
        if predicate not in self._compiled_defs:
            return
        param_ids, def_body, _ = self._compiled_defs[predicate]
        for inds in itertools.combinations_with_replacement(range(self.universe), arity):
            assignment = {pid: val for pid, val in zip(param_ids, inds)}
            result = self._eval_compiled(def_body, assignment, 0, 0)
            self._pred_cache[(predicate, inds)] = result

    # ------------------------------------------------------------------
    # Public API (compatible with the original ModelChecker)
    # ------------------------------------------------------------------

    def find_model(
        self, formula, timeout=30
    ) -> Tuple[ModelCheckerOutcome, Optional[List[Tuple[str, int]]]]:
        """Check if *formula* is satisfiable in the structure.

        Returns ``(outcome, witness)`` where *witness* is a list of
        ``(variable_name, element)`` tuples — same format as the original.
        """
        start_time = time.perf_counter()

        try:
            var_map: Dict[tuple, int] = {}
            next_var_id = [0]
            compiled = compile_formula(formula, var_map, next_var_id)
            # Build reverse map: int ID -> original variable name
            id_to_name: Dict[int, str] = {}
            for key, vid in var_map.items():
                # key is ("var", symbol_str)
                id_to_name[vid] = key[1]

            assignment: Dict[int, int] = {}
            result = self._eval_compiled(compiled, assignment, timeout, start_time)

            if result:
                # Format witness as list of (name, value) tuples
                witness = [(id_to_name[vid], val) for vid, val in sorted(assignment.items()) if vid in id_to_name]
                self.proven_formulae.append(formula)
                return ModelCheckerOutcome.MODEL_FOUND, witness
            else:
                self.disproven_formulae.append(formula)
                return ModelCheckerOutcome.NO_MODEL, None

        except TimeoutError:
            logging.warning(f"Timed out after {time.perf_counter() - start_time:.2f}s")
            return ModelCheckerOutcome.TIMEOUT, None
        except Exception as e:
            logging.error(f"FastModelChecker error: {e}")
            raise

    def find_model_quantified(
        self, formula, timeout=30
    ) -> Tuple[ModelCheckerOutcome, Optional[List[Tuple[str, int]]]]:
        """Same as find_model — handles mixed quantifiers natively."""
        return self.find_model(formula, timeout)

    # ------------------------------------------------------------------
    # Compiled evaluator
    # ------------------------------------------------------------------

    def _eval_compiled(self, node, assignment, timeout, start_time):
        """Evaluate compiled formula node under *assignment*."""
        if timeout > 0 and (time.perf_counter() - start_time) > timeout:
            raise TimeoutError("Model checking timed out")

        op = node.op

        if op == _Op.ATOM:
            return self._eval_atom(node, assignment, timeout, start_time)

        elif op == _Op.NEG:
            return not self._eval_compiled(node.children[0], assignment, timeout, start_time)

        elif op == _Op.AND:
            return all(
                self._eval_compiled(c, assignment, timeout, start_time) for c in node.children
            )

        elif op == _Op.OR:
            return any(
                self._eval_compiled(c, assignment, timeout, start_time) for c in node.children
            )

        elif op == _Op.EQ:
            a0, a1 = node.args
            v0 = assignment[a0[1]] if a0[0] else a0[1]
            v1 = assignment[a1[1]] if a1[0] else a1[1]
            return v0 == v1

        elif op == _Op.NEQ:
            a0, a1 = node.args
            v0 = assignment[a0[1]] if a0[0] else a0[1]
            v1 = assignment[a1[1]] if a1[0] else a1[1]
            return v0 != v1

        elif op == _Op.EXISTS:
            return self._search_exists(node, assignment, timeout, start_time)

        elif op == _Op.FORALL:
            return self._search_forall(node, assignment, timeout, start_time)

        return False

    def _eval_atom(self, node, assignment, timeout, start_time):
        """Evaluate a predicate atom, resolving defined predicates with caching."""
        concrete_args = tuple(
            assignment[a[1]] if a[0] else a[1] for a in node.args
        )
        pred = node.pred_name

        # Base extension lookup
        if pred in self.structure.extensions:
            return self.structure.eval_pred(pred, concrete_args)

        # Defined predicate
        if pred in self._compiled_defs:
            cache_key = (pred, concrete_args)
            cached = self._pred_cache.get(cache_key)
            if cached is not None:
                return cached

            param_ids, def_body, _ = self._compiled_defs[pred]
            def_assignment = {pid: val for pid, val in zip(param_ids, concrete_args)}
            result = self._eval_compiled(def_body, def_assignment, timeout, start_time)
            self._pred_cache[cache_key] = result
            return result

        # Unknown predicate — empty extension
        return False

    # ------------------------------------------------------------------
    # Existential search with forward checking
    # ------------------------------------------------------------------

    def _search_exists(self, node, assignment, timeout, start_time):
        qvars = list(node.qvars)
        qvar_names = node.qvar_names
        body = node.children[0]

        # Initial domains: restrict via unary predicates and adjacency
        domains = {v: set(self.structure.domain) for v in qvars}
        _restrict_domains(body, self.structure, domains, self._defined_pred_names)

        # If all_different, exclude already-assigned values
        if self.all_different:
            used = set(assignment.values())
            for v in qvars:
                domains[v] -= used

        # Extract binary constraints for forward checking
        constraints: List[_BinConstraint] = []
        _extract_bin_constraints(body, constraints, skip_preds=self._defined_pred_names)

        # Build per-variable constraint index
        var_constraints: Dict[int, List[_BinConstraint]] = {v: [] for v in qvars}
        for c in constraints:
            if c.var1 in var_constraints:
                var_constraints[c.var1].append(c)
            if c.var2 in var_constraints:
                var_constraints[c.var2].append(c)

        found = self._backtrack(
            qvars, body, assignment, domains, var_constraints, timeout, start_time
        )
        if not found:
            # Clean up any partial assignments
            for v in qvars:
                assignment.pop(v, None)
        return found

    def _backtrack(self, remaining, body, assignment, domains, var_constraints, timeout, start_time):
        if timeout > 0 and (time.perf_counter() - start_time) > timeout:
            raise TimeoutError("Model checking timed out")

        if not remaining:
            return self._eval_compiled(body, assignment, timeout, start_time)

        # Fail-first: pick variable with smallest remaining domain
        remaining_sorted = sorted(remaining, key=lambda v: len(domains.get(v, set())))
        var = remaining_sorted[0]
        rest = remaining_sorted[1:]

        domain = domains.get(var, set())
        if not domain:
            return False

        for value in sorted(domain):  # deterministic ordering
            assignment[var] = value

            # Forward check: propagate to neighbors
            saved = {}
            feasible = True

            for c in var_constraints.get(var, []):
                other = c.var2 if c.var1 == var else c.var1
                if other not in domains or other in assignment:
                    continue

                old_dom = domains[other]
                saved[other] = old_dom
                nbrs = self.structure.neighbors.get(c.pred_name, {})

                if not c.negated:
                    if c.var1 == var:
                        allowed = set(nbrs.get(value, []))
                    else:
                        allowed = {e for e in old_dom if value in nbrs.get(e, [])}
                    new_dom = old_dom & allowed
                else:
                    if c.var1 == var:
                        forbidden = set(nbrs.get(value, []))
                    else:
                        forbidden = {e for e in old_dom if value in nbrs.get(e, [])}
                    new_dom = old_dom - forbidden

                # all_different: exclude currently assigned values
                if self.all_different:
                    new_dom -= set(assignment.values())

                domains[other] = new_dom
                if not new_dom:
                    feasible = False
                    break

            if feasible and self._backtrack(rest, body, assignment, domains, var_constraints, timeout, start_time):
                return True

            # Restore domains
            for other, old_dom in saved.items():
                domains[other] = old_dom

        del assignment[var]
        return False

    # ------------------------------------------------------------------
    # Universal quantifier
    # ------------------------------------------------------------------

    def _search_forall(self, node, assignment, timeout, start_time):
        qvars = list(node.qvars)
        body = node.children[0]

        for combo in itertools.product(range(self.structure.universe), repeat=len(qvars)):
            if timeout > 0 and (time.perf_counter() - start_time) > timeout:
                raise TimeoutError("Model checking timed out")

            if self.all_different and len(set(combo)) != len(combo):
                continue

            for var, val in zip(qvars, combo):
                assignment[var] = val

            if not self._eval_compiled(body, assignment, timeout, start_time):
                for var in qvars:
                    assignment.pop(var, None)
                return False

        for var in qvars:
            assignment.pop(var, None)
        return True
