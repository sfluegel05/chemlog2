from chemlog.msol import msol
from chemlog.msol.peptide_size import IsConnected


def partitions(n, max_value=None):
    # partitions of positive integers that sum up to n
    # max value ensures that the values are ordered descendingly
    if max_value is None:
        max_value = n
    if n == 0:
        return [[]]
    result = []
    for i in range(1, min(max_value, n) + 1):
        for tail in partitions(n - i, i):
            result.append([i] + tail)
    return result


class HasChargeComponent(msol.MSOLDefinition):

    def __init__(self, charge_level=1):
        """This predicate defines that a second order entity has at least this many positive / negative charges."""
        super().__init__()
        self.charge_level = charge_level

    def name(self):
        return f"HasChargeComponent{'M' if self.charge_level < 0 else ''}{abs(self.charge_level)}"

    def __call__(self, x: msol.Var2):
        disj = []
        for p in partitions(abs(self.charge_level)):
            atom_vars = [msol.Var1(f"a_{i}") for i, pp in enumerate(p)]
            disj.append(
                msol.QuantifiedFormula(
                    msol.Quantifier.EXISTENTIAL,
                    atom_vars,
                    msol.NaryFormula(
                        msol.BinaryConnective.CONJUNCTION,
                        [msol.InSetFormula(a_i, msol.Var2(f"Charge{'M' if self.charge_level < 0 else ''}{pp}"))
                            for a_i, pp in zip(atom_vars, p)]
                        + [msol.BinaryFormula(atom_vars[i], msol.BinaryConnective.NEQ, atom_vars[j])
                            for i in range(len(atom_vars)) for j in range(i + 1, len(atom_vars))]
                    )
                )
            )
        return msol.NaryFormula(msol.BinaryConnective.DISJUNCTION, disj)

class ConnectedComponent(msol.MSOLDefinition):

    def name(self):
        return "ConnectedComponent"

    @staticmethod
    def __call__(x: msol.Var2) -> msol.QuantifiedFormula:
        # pred ConnectedComponent(var2 X) = IsConnected(X) & ~ex2 Y: (X sub Y & IsConnected(Y));
        y = msol.Var2("Y")
        return msol.PredicateExpression(IsConnected().name(), [x]) & ~msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [y],
            msol.SetSetFormula(x, msol.SetSetOperator.SUBSET, y)
            & msol.PredicateExpression(IsConnected().name(), [y])
        )

class Salt(msol.MSOLDefinition):
    """
    Salts are defined a molecule that has two connected components, one with a net negative charge and one with a net positive charge.
    Since we cannot count, the net positive and net negative charge are defined up to a maximum charge level
    """

    def __init__(self, max_charge_level=3):
        super().__init__()
        self.max_charge_level = max_charge_level

    def name(self):
        return "Salt"

    def __call__(self):
        pos_component = msol.Var2("Pos")
        neg_component = msol.Var2("Neg")
        # salt <-> connected_component(Pos) & connected_component(Neg) & ((charge1(Pos) & ~chargeM1(Pos)) | (charge2(Pos) & ~chargeM2(Pos)) | ...)
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [pos_component, neg_component],
            msol.NaryFormula(
                msol.BinaryConnective.CONJUNCTION,
                [
                    msol.PredicateExpression(ConnectedComponent().name(), [pos_component]),
                    msol.PredicateExpression(ConnectedComponent().name(), [neg_component]),
                    msol.NaryFormula(
                        msol.BinaryConnective.DISJUNCTION,
                        [msol.PredicateExpression(HasChargeComponent(i).name(), [pos_component])
                         & ~msol.PredicateExpression(HasChargeComponent(-i).name(), [pos_component])
                         for i in range(1, self.max_charge_level + 1)]
                    ),
                    msol.NaryFormula(
                        msol.BinaryConnective.DISJUNCTION,
                        [msol.PredicateExpression(HasChargeComponent(-i).name(), [neg_component])
                         & ~msol.PredicateExpression(HasChargeComponent(i).name(), [neg_component])
                         for i in range(1, self.max_charge_level + 1)]
                    ),
                ]
            )
        )


class Zwitterion(msol.MSOLDefinition):

    def name(self):
        return "Zwitterion"

    def __call__(self):
        f = msol.Var2("F")
        a_plus, a_minus = msol.Var1("A_plus"), msol.Var1("A_minus")
        return msol.QuantifiedFormula(
            msol.Quantifier.EXISTENTIAL, [f, a_plus, a_minus],
            msol.NaryFormula(
                msol.BinaryConnective.CONJUNCTION,
                [
                    msol.PredicateExpression(ConnectedComponent().name(), [f]),
                    msol.PredicateExpression("NetCharge0", []),
                    msol.InSetFormula(a_plus, msol.Var2("ChargeP")),
                    msol.InSetFormula(a_minus, msol.Var2("ChargeN")),
                    msol.InSetFormula(a_plus, f),
                    msol.InSetFormula(a_minus, f),
                    ~msol.PredicateExpression("has_bond_to", [a_plus, a_minus]),
                ]
            )
        )


class OrganicAnion(msol.MSOLDefinition):

    def __call__(self):
        return ~msol.PredicateExpression(Salt().name(), []) & msol.PredicateExpression("NetChargeN", [])

    def name(self):
        return "OrganicAnion"


class OrganicCation(msol.MSOLDefinition):

    def __call__(self):
        return ~msol.PredicateExpression(Salt().name(), []) & msol.PredicateExpression("NetChargeP", [])

    def name(self):
        return "OrganicCation"


class Neutral(msol.MSOLDefinition):

    def __call__(self):
        return msol.NaryFormula(
            msol.BinaryConnective.CONJUNCTION,
            [
                ~msol.PredicateExpression(Salt().name(), []),
                ~msol.PredicateExpression(Zwitterion().name(), []),
                ~msol.PredicateExpression(OrganicAnion().name(), []),
                ~msol.PredicateExpression(OrganicCation().name(), []),
            ]
        )

    def name(self):
        return "Neutral"

if __name__ == "__main__":
    print(HasChargeComponent(-3)(msol.Var2("X")))
    print(Salt()())
