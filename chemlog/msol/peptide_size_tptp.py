from chemlog.msol import peptide_size, msol
from gavel.logic.problem import Problem, AnnotatedFormula, FormulaRole

from chemlog.msol.tptp_compiler import TPTPMSOLCompiler


def build_peptide_structure(n):
    compiler = TPTPMSOLCompiler()
    # build peptide structure using the internal MSOL representation
    peptide_size_def = peptide_size.Peptide(n)
    peptide_size_formula = peptide_size_def()
    return AnnotatedFormula("thf", f"peptide_size_{n}", FormulaRole.AXIOM, compiler.visit(peptide_size_formula))

def build_peptide_size_problem():
    # todo: problem for subclass relations between peptide classes
    pass