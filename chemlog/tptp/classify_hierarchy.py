import os
from chemlog.tptp.leo_theorem_prover import prove

from chemlog.msol import peptide_size, msol, peptide_charge, peptide_substructure
from gavel.logic.problem import AnnotatedFormula, FormulaRole

from chemlog.msol.peptide_size import Peptide
from chemlog.tptp.tptp_compiler import TPTPMSOLCompiler

SIZE_BACKGROUND = "peptide_size_background.tptp"
CHARGE_BACKGROUND = "peptide_charge_background.tptp"
SUBSTRUCT_BACKGROUND = "substruct_background.tptp"

def build_peptide_structure(min_aars=2, max_aars=None):
    """MSOL formula for peptide structure with at least `min_aars` amino acid residues (and at most `max_aars` amino acid residues if `max_aars` is not None)."""
    # build peptide structure using the internal MSOL representation
    peptide_size_def_min = peptide_size.Peptide(min_aars)
    peptide_size_formula = peptide_size_def_min()
    if max_aars is not None:
        peptide_size_def_max = peptide_size.Peptide(max_aars + 1)
        peptide_size_formula = peptide_size_formula & ~peptide_size_def_max()
    return peptide_size_formula

def build_size_based_background():
    definitions = [peptide_size.HasOverlap(), peptide_size.IsConnected(), peptide_size.CarbonConnected(),
                   peptide_size.CarbonFragment(), peptide_size.AmideBondFO(),
                   peptide_size.AminoGroupFO(), peptide_size.CarboxyResidueFO(),
                   peptide_size.BuildingBlock(), peptide_size.AAR(), peptide_size.Peptide(2), peptide_size.Peptide(3),
                   peptide_size.Peptide(4), peptide_size.Peptide(5), peptide_size.Peptide(6), peptide_size.Peptide(10)]
    predictates_defined_in_structure = [
        ("bSINGLE", 2),
        ("bDOUBLE", 2),
        ("has_bond_to", 2),
        ("has1Hs", 1),
        ("chargeN", 1),
        ("c", 1),
        ("o", 1),
        ("n", 1)
    ]
    compiler = TPTPMSOLCompiler(given_predicate_names=[name for name, _ in predictates_defined_in_structure])
    axioms = [
        compiler.visit(AnnotatedFormula("thf", f"type_{name}", FormulaRole.TYPE,
                                        f"{name}: ({''.join('$i > ' for _ in range(arity))}$o)"))
        for name, arity in predictates_defined_in_structure
    ]
    for definition in definitions:
        axioms.extend(compiler.visit(definition))

    peptide_structures = [
        msol.BinaryFormula("peptide_structure", msol.BinaryConnective.EQ, msol.PredicateExpression(Peptide(2).name(), [])),
        msol.BinaryFormula("dipeptide_structure", msol.BinaryConnective.EQ, msol.BinaryFormula(
            msol.PredicateExpression(Peptide(2).name(), []),
            msol.BinaryConnective.CONJUNCTION,
            msol.UnaryFormula(msol.UnaryConnective.NEGATION, msol.PredicateExpression(Peptide(3).name(), []))
        )),
        msol.BinaryFormula("tripeptide_structure", msol.BinaryConnective.EQ, msol.BinaryFormula(
            msol.PredicateExpression(Peptide(3).name(), []),
            msol.BinaryConnective.CONJUNCTION,
            msol.UnaryFormula(msol.UnaryConnective.NEGATION, msol.PredicateExpression(Peptide(4).name(), []))
        )),
        msol.BinaryFormula("tetrapeptide_structure", msol.BinaryConnective.EQ, msol.BinaryFormula(
            msol.PredicateExpression(Peptide(4).name(), []),
            msol.BinaryConnective.CONJUNCTION,
            msol.UnaryFormula(msol.UnaryConnective.NEGATION, msol.PredicateExpression(Peptide(5).name(), []))
        )),
        msol.BinaryFormula("pentapeptide_structure", msol.BinaryConnective.EQ, msol.BinaryFormula(
            msol.PredicateExpression(Peptide(5).name(), []),
            msol.BinaryConnective.CONJUNCTION,
            msol.UnaryFormula(msol.UnaryConnective.NEGATION, msol.PredicateExpression(Peptide(6).name(), []))
        )),
        msol.BinaryFormula("oligopeptide_structure", msol.BinaryConnective.EQ, msol.BinaryFormula(
            msol.PredicateExpression(Peptide(2).name(), []),
            msol.BinaryConnective.CONJUNCTION,
            msol.UnaryFormula(msol.UnaryConnective.NEGATION, msol.PredicateExpression(Peptide(10).name(), []))
        )),
        msol.BinaryFormula("polypeptide_structure", msol.BinaryConnective.EQ, msol.PredicateExpression(Peptide(10).name(), []))
    ]
    for structure in peptide_structures:
        axioms.append(compiler.visit(AnnotatedFormula("thf", f"{structure.left}_type", FormulaRole.TYPE, f"{structure.left}: $o")))
        axioms.append(compiler.visit(AnnotatedFormula("thf", structure.left, FormulaRole.AXIOM, structure)))

    return "\n".join(axioms)

def build_charge_based_background(max_charge_level=3):
    definitions = [peptide_charge.HasChargeComponent(-i) for i in range(1, max_charge_level + 1)] \
                        + [peptide_charge.HasChargeComponent(i) for i in range(1, max_charge_level + 1)]\
                        + [peptide_charge.IsConnected(), peptide_charge.ConnectedComponent(),
                           peptide_charge.Salt(max_charge_level), peptide_charge.Zwitterion(),
                           peptide_charge.OrganicAnion(), peptide_charge.OrganicCation(), peptide_charge.Neutral()]
    predictates_defined_in_structure = [
        ("has_bond_to", 2),
        ("chargeP", 1),
        ("chargeN", 1),
        ("netChargeN", 0),
        ("netCharge0", 0),
        ("netChargeP", 0),
    ] + [(f"chargeM{-i}" if i < 0 else f"charge{i}", 1) for i in range(-max_charge_level, max_charge_level + 1)]

    compiler = TPTPMSOLCompiler(given_predicate_names=[name for name, _ in predictates_defined_in_structure])
    axioms = [
        compiler.visit(AnnotatedFormula("thf", f"{name}_type", FormulaRole.TYPE,
                                        f"{name}: ({''.join('$i > ' for _ in range(arity))}$o)"))
        for name, arity in predictates_defined_in_structure
    ]
    for definition in definitions:
        axioms.extend(compiler.visit(definition))

    return "\n".join(axioms)

def build_substruct_based_background():
    definitions = [peptide_substructure.Diketopiperazines(), peptide_substructure.Emericellamide()]
    predictates_defined_in_structure = [
        ("has_at_least_1_hs", 1),
        ("has_at_least_2_hs", 1),
        ("has_at_least_3_hs", 1),
        ("n", 1),
        ("c", 1),
        ("o", 1),
        ("has_bond_to", 2),
        ("charge0", 1),
        ("cip_code_S", 1)
    ]


    compiler = TPTPMSOLCompiler(given_predicate_names=[name for name, _ in predictates_defined_in_structure])
    axioms = [
        compiler.visit(AnnotatedFormula("thf", f"{name}_type", FormulaRole.TYPE,
                                        f"{name}: ({''.join('$i > ' for _ in range(arity))}$o)"))
        for name, arity in predictates_defined_in_structure
    ]
    for definition in definitions:
        axioms.extend(compiler.visit(definition))

    return "\n".join(axioms)

def build_subclass_superclass_problem(subclass_formula, superclass_formula, negated=False, imports=None):
    """Build and compile a TPTP problem with the standard background axioms and the conjecture subclass -> superclass
    (if negated: ~(subclass -> superclass))."""

    if imports is None:
        axioms = [f"include('{os.path.join(os.getcwd(), SIZE_BACKGROUND)}').",
              f"include('{os.path.join(os.getcwd(), CHARGE_BACKGROUND)}')."]
    else:
        axioms = [f"include('{os.path.join(os.getcwd(), import_path)}')." for import_path in imports]

    compiler = TPTPMSOLCompiler()

    if negated:
        conjecture = AnnotatedFormula("thf", "conjecture", FormulaRole.NEGATED_CONJECTURE,
                                      ~msol.BinaryFormula(subclass_formula, msol.BinaryConnective.IMPLICATION, superclass_formula))
    else:
        conjecture = AnnotatedFormula("thf", "conjecture", FormulaRole.CONJECTURE,
                                  msol.BinaryFormula(subclass_formula, msol.BinaryConnective.IMPLICATION, superclass_formula))
    axioms.append(compiler.visit(conjecture))

    return "\n".join(axioms)


def run_sizebased_hierarchy():
    # add background axioms for peptide size
    if not os.path.exists(SIZE_BACKGROUND):
        tptp_problem = build_size_based_background()
        with open(SIZE_BACKGROUND, "w") as f:
            f.write(tptp_problem)
    # check relations between peptide structure classes
    # oligopeptide -> peptide
    # polypeptide -> peptide
    # dipeptide -> oligopeptide
    # tripeptide -> oligopeptide
    # tetrapeptide -> oligopeptide
    # pentapeptide -> oligopeptide
    for subclass, superclass in [
        ("oligopeptide_structure", "peptide_structure"),
        ("polypeptide_structure", "peptide_structure"),
        ("dipeptide_structure", "oligopeptide_structure"),
        ("tripeptide_structure", "oligopeptide_structure"),
        ("tetrapeptide_structure", "oligopeptide_structure"),
        ("pentapeptide_structure", "oligopeptide_structure")]:
        print(f"Checking relation {subclass} -> {superclass}")
        tptp_problem = build_subclass_superclass_problem(subclass, superclass, negated=False, imports=[SIZE_BACKGROUND])
        #print(tptp_problem)
        # save to file
        with open(f"{subclass}_implies_{superclass}.tptp", "w") as f:
            f.write(tptp_problem)

        print(prove(tptp_problem, timeout=10))

def run_substruct_hierarchy():
    # add background axioms for peptide size and charge
    if not os.path.exists(SIZE_BACKGROUND):
        tptp_problem = build_size_based_background()
        with open(SIZE_BACKGROUND, "w") as f:
            f.write(tptp_problem)
    if not os.path.exists(CHARGE_BACKGROUND):
        tptp_problem = build_charge_based_background()
        with open(CHARGE_BACKGROUND, "w") as f:
            f.write(tptp_problem)
    if not os.path.exists(SUBSTRUCT_BACKGROUND):
        tptp_problem = build_substruct_based_background()
        with open(SUBSTRUCT_BACKGROUND, "w") as f:
            f.write(tptp_problem)

    compiler = TPTPMSOLCompiler()
    diketo_vars = [msol.Var1(f"n{i}") for i in range(8)]
    diketo_exists = msol.QuantifiedFormula(
        msol.Quantifier.EXISTENTIAL, diketo_vars,
        msol.PredicateExpression("diketopiperazines", diketo_vars)
    )
    diketo_exists = compiler.visit(diketo_exists)
    for subclass, superclass in [
        (diketo_exists, "dipeptide_structure"),
        #("emericellamide", "pentapeptide_structure"),
        (diketo_exists, "neutral"),
        #("emericellamide", "neutral"),
    ]:
        print(f"Checking relation {subclass} -> {superclass}")
        tptp_problem = build_subclass_superclass_problem(
            subclass, superclass,
            negated=False,
            imports=[SIZE_BACKGROUND, CHARGE_BACKGROUND, SUBSTRUCT_BACKGROUND]
        )
        #print(tptp_problem)
        # save to file
        with open(f"{subclass}_implies_{superclass}.tptp", "w") as f:
            f.write(tptp_problem)

        print(prove(tptp_problem, timeout=10))


if __name__ == "__main__":
    run_substruct_hierarchy()