import abc

from chemlog.msol import msol


class MSOLDefinition(abc.ABC):
    """
    forall x1, ..., xn: self.name(x1, ..., xn) <=> self()
    """

    def name(self):
        pass

    @staticmethod
    def __call__(*args, **kwargs) -> msol.QuantifiedFormula:
        """
        Returns the right-hand side of the MSOL definition as a quantified formula.
        """
        pass


class HasOverlap(MSOLDefinition):

    def name(self):
        return "HasOverlap"

    @staticmethod
    def __call__(x1: msol.Var2, x2: msol.Var2) -> msol.QuantifiedFormula:
        # pred HasOverlap(var2 X, var2 Y) = ex1 a: a in X & a in Y;
        a = msol.Var1("a")
        return msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [a],
                                      msol.InSetFormula(a, x1) & msol.InSetFormula(a, x2))


class IsConnected(MSOLDefinition):

    def name(self):
        return "IsConnected"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred IsConnected(var2 X) = all2 A: all2 B: (ex1 u: u in A & ex1 v: v in B & A union B = X) => ex1 u: ex1 v: (u in A & v in B & has_bond_to(u, v));
        a = msol.Var2("A")
        b = msol.Var2("B")
        u, v = msol.Var1("u"), msol.Var1("v")
        t, w = msol.Var1("t"), msol.Var1("w")
        return msol.QuantifiedFormula(
            msol.Quantifier.UNIVERSAL, [a, b],
            msol.BinaryFormula(
                msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [u], msol.InSetFormula(u, a))
                & msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [v], msol.InSetFormula(v, b))
                & msol.SetSetFormula(msol.SetSetFunctorExpression(a, msol.SetSetFunctor.UNION, b),
                                     msol.SetSetOperator.SET_EQ, x),
                msol.BinaryConnective.IMPLICATION,
                msol.QuantifiedFormula(
                    msol.Quantifier.EXISTENTIAL, [t, w],
                    msol.InSetFormula(t, a) & msol.InSetFormula(w, b) & msol.PredicateExpression("has_bond_to", [t, w])
                )
            )
        )


class CarbonConnected(MSOLDefinition):

    def name(self):
        return "CarbonConnected"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred CarbonConnected(var2 X) = X sub C & IsConnected(X);
        return msol.SetSetFormula(x, msol.SetSetOperator.SUBSET_EQ, msol.Var2("C")) & msol.PredicateExpression(IsConnected().name(), [x])


class CarbonFragment(MSOLDefinition):

    def name(self):
        return "CarbonFragment"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred CarbonFragment(var2 X) = CarbonConnected(X) & ~ex2 Y: (X sub Y & X ~= Y & CarbonConnected(Y));
        y = msol.Var2("Y")
        return msol.PredicateExpression(CarbonConnected().name(), [x]) & ~msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [y],
            msol.SetSetFormula(x, msol.SetSetOperator.SUBSET, y)
            & msol.SetSetFormula(x, msol.SetSetOperator.SET_NEQ, y)
            & msol.PredicateExpression(CarbonConnected().name(), [y])
        )


class AmideBond(MSOLDefinition):

    def name(self):
        return "AmideBond"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred AmideBond(var2 X) = ex1 a_c: ex1 a_o: ex1 a_n: X = {a_n, a_c, a_o}
        #     & a_c in C & a_o in O & a_n in N & ((bSINGLE(a_c, a_o) & bDOUBLE(a_c, a_n)
        #     & (a_o in Has1Hs union ChargeN)) | (bDOUBLE(a_c, a_o) & bSINGLE(a_c,a_n)));
        a_c = msol.Var1("a_c")
        a_o = msol.Var1("a_o")
        a_n = msol.Var1("a_n")
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [a_c, a_o, a_n],
            msol.SetSetFormula(x, msol.SetSetOperator.SET_EQ, msol.SetOf([a_c, a_o, a_n])) &
            msol.InSetFormula(a_c, msol.Var2("C")) &
            msol.InSetFormula(a_o, msol.Var2("O")) &
            msol.InSetFormula(a_n, msol.Var2("N")) &
            (
                    (msol.PredicateExpression("bSINGLE", [a_c, a_o]) &
                     msol.PredicateExpression("bDOUBLE", [a_c, a_n]) &
                     (msol.InSetFormula(a_o, msol.Var2("Has1Hs")) | msol.InSetFormula(a_o, msol.Var2("ChargeN")))) |
                    (msol.PredicateExpression("bDOUBLE", [a_c, a_o]) &
                     msol.PredicateExpression("bSINGLE", [a_c, a_n]))
            )
        )


class AminoGroup(MSOLDefinition):

    def name(self):
        return "AminoGroup"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred AminoGroup(var2 X) = ex1 a_n: (X = {a_n} & a_n in N
        #     & all1 a_x: (has_bond_to(a_n, a_x) => (a_x in C & (bSINGLE(a_n, a_x) | (ex2 B: AmideBond(B) & a_n in B & a_x in B)))));
        a_n = msol.Var1("a_n")
        a_x = msol.Var1("a_x")
        b = msol.Var2("B")
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [a_n],
            msol.SetSetFormula(x, msol.SetSetOperator.SET_EQ, msol.SetOf([a_n])) &
            msol.InSetFormula(a_n, msol.Var2("N")) &
            msol.QuantifiedFormula(
                msol.Quantifier.UNIVERSAL,[a_x],
                msol.BinaryFormula(
                    msol.PredicateExpression("has_bond_to", [a_n, a_x]),
                    msol.BinaryConnective.IMPLICATION,
                    msol.InSetFormula(a_x, msol.Var2("C")) &
                        (msol.PredicateExpression("bSINGLE", [a_n, a_x]) |
                            msol.QuantifiedFormula(
                                msol.Quantifier.EXISTENTIAL, [b],
                                msol.PredicateExpression(AmideBond().name(), [b]) & msol.InSetFormula(a_n, b) & msol.InSetFormula(a_x, b)
                            )
                         )
                )
            )
        )


class CarboxyResidue(MSOLDefinition):

    def name(self):
        return "CarboxyResidue"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred CarboxyResidue(var2 X) = ex1 a_c: ex1 a_o: ex1 a_s: X = {a_c, a_o, a_s} & a_c in C & a_o in O
        #     & bDOUBLE(a_c, a_o) & bSINGLE(a_c, a_s);
        a_c = msol.Var1("a_c")
        a_o = msol.Var1("a_o")
        a_s = msol.Var1("a_s")
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [a_c, a_o, a_s],
            msol.SetSetFormula(x, msol.SetSetOperator.SET_EQ, msol.SetOf([a_c, a_o, a_s])) &
            msol.InSetFormula(a_c, msol.Var2("C")) &
            msol.InSetFormula(a_o, msol.Var2("O")) &
            msol.PredicateExpression("bDOUBLE", [a_c, a_o]) &
            msol.PredicateExpression("bSINGLE", [a_c, a_s])
        )


class BuildingBlock(MSOLDefinition):
    """
    A building block is a carbon fragment with at least one amide bond and one carboxy residue.
    """

    def name(self):
        return "BuildingBlock"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred BuildingBlock(var2 X) = ex2 Y: CarbonFragment(Y) & Y sub X & all1 x: (x in X => (x in Y | ex1 y: (y in Y & has_bond_to(x, y) & (x in N => ~ ex2 B: (AmideBond(B) & x in B & y in B)))));
        y = msol.Var2("Y")
        u = msol.Var1("u")
        v = msol.Var1("v")
        b = msol.Var2("B")
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [y],
            msol.PredicateExpression(CarbonFragment().name(), [y]) &
            msol.SetSetFormula(y, msol.SetSetOperator.SUBSET, x) &
            msol.QuantifiedFormula(
                msol.Quantifier.UNIVERSAL, [u],
                msol.BinaryFormula(
                    msol.InSetFormula(u, x),
                    msol.BinaryConnective.IMPLICATION,
                    msol.InSetFormula(u, y) |
                    msol.QuantifiedFormula(
                        msol.Quantifier.EXISTENTIAL, [v],
                        # (y in Y & has_bond_to(x, y) & (x in N => ~ ex2 B: (AmideBond(B) & x in B & y in B)))
                        msol.InSetFormula(v, y) & msol.PredicateExpression("has_bond_to", [u, v])
                        & msol.BinaryFormula(
                            msol.InSetFormula(u, msol.Var2("N")),
                            msol.BinaryConnective.IMPLICATION,
                            ~msol.QuantifiedFormula(
                                msol.Quantifier.EXISTENTIAL, [b],
                                msol.PredicateExpression(AmideBond().name(), [b]) &
                                msol.InSetFormula(u, b) & msol.InSetFormula(v, b)
                            )
                        )
                    )
                )
            )
        )

class AAR(MSOLDefinition):

    def name(self):
        return "AAR"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred AAR(var2 X) = BuildingBlock(X) & ex2 AG: ex2 CG: AminoGroup(AG) & CarboxyResidue(CG) & AG sub X & CG sub X;
        ag = msol.Var2("AG")
        cg = msol.Var2("CG")
        return msol.PredicateExpression(BuildingBlock().name(), [x]) & msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [ag, cg],
            msol.PredicateExpression(AminoGroup().name(), [ag]) & msol.PredicateExpression(CarboxyResidue().name(), [cg]) &
            msol.SetSetFormula(ag, msol.SetSetOperator.SUBSET_EQ, x) &
            msol.SetSetFormula(cg, msol.SetSetOperator.SUBSET_EQ, x)
        )

class Peptide(MSOLDefinition):

    def __init__(self, n_amino_acid_residues):
        self.n_amino_acid_residues = n_amino_acid_residues

    def name(self):
        return f"Peptide{self.n_amino_acid_residues}plus"

    def __call__(self) -> msol.QuantifiedFormula:
        aars = [msol.Var2(f"A{i}") for i in range(self.n_amino_acid_residues)]
        bonds = [msol.Var2(f"B{i}") for i in range(self.n_amino_acid_residues - 1)]
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, aars + bonds,
            msol.NaryFormula(
                msol.BinaryConnective.CONJUNCTION,
                # AAR(a_i)
                [msol.PredicateExpression(AAR().name(), [aars[i]]) for i in range(self.n_amino_acid_residues)]
                # ~HasOverlap(a_i, a_j) for i < j
                + [~msol.PredicateExpression(HasOverlap().name(), [aars[i], aars[j]])
                   for i in range(self.n_amino_acid_residues - 1) for j in range(i + 1, self.n_amino_acid_residues)]
                # AmideBond(b_i)
                + [msol.PredicateExpression(AmideBond().name(), [bonds[i]]) for i in range(self.n_amino_acid_residues - 1)]
                # HasOverlap(b_i, a_{i+1})
                + [msol.PredicateExpression(HasOverlap().name(), [bonds[i], aars[i + 1]])
                   for i in range(self.n_amino_acid_residues - 1)]
                # HasOverlap(b_i, a_j) for some j <= i
                + [msol.NaryFormula(msol.BinaryConnective.DISJUNCTION,
                                    [msol.PredicateExpression(HasOverlap().name(), [bonds[i], aars[j]])
                                     for j in range(i + 1)]) for i in range(self.n_amino_acid_residues - 1)]
            )
        )

if __name__ == "__main__":
    # Example usage
    deff = AmideBond()
    print(deff.name())
    f = deff(msol.Var2("X"))
    print(f)
    from chemlog.msol_classification.mona_compiler import MONACompiler
    compiler = MONACompiler()
    print(f"MONA translation: {compiler.visit(f)}")