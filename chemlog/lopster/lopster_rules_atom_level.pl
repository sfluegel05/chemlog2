% rules taken from https://github.com/magkades/lopster/blob/1e4ae80367319b6c721e78b2ffd9d44b72eb22c2/src/main/resources/input/chemrules/chemRulesWithCyclic
% Magka, Despoina, Markus Krötzsch, and Ian Horrocks. "A rule-based ontological framework for the classification of molecules." Journal of biomedical semantics 5.1 (2014): 17.

bond(X,Y) :- single(X,Y).
bond(X,Y) :- double(X,Y).
bond(X,Y) :- triple(X,Y).

horc(X) :- h(X).
horc(X) :- c(X).

charged(X) :- plus3(X).
charged(X) :- plus2(X).
charged(X) :- plus1(X).
charged(X) :- minus3(X).
charged(X) :- minus2(X).
charged(X) :- minus1(X).

midOxygen(Y) :- hasAtom(X,Y), o(Y), hasAtom(X,Y1), hasAtom(X,Y2), bond(Y,Y1), bond(Y1,Y), bond(Y,Y2), bond(Y2,Y), Y1!=Y2.

bondAtLeast1(X) :- bond(X,Y1), bond(Y1,X).
bondAtLeast2(X) :- bond(X,Y1), bond(Y1,X), bond(X,Y2), bond(Y2,X), Y1!=Y2.
bondAtLeast3(X) :- bond(X,Y1), bond(Y1,X), bond(X,Y2), bond(Y2,X), bond(X,Y3), bond(Y3,X), Y1!=Y2, Y1!=Y3, Y2!=Y3.
bondAtLeast4(X) :- bond(X,Y1), bond(Y1,X), bond(X,Y2), bond(Y2,X), bond(X,Y3), bond(Y3,X), bond(X,Y4), bond(Y4,X), Y1!=Y2, Y1!=Y3, Y1!=Y4, Y2!=Y3, Y2!=Y4, Y3!=Y4.
bondExactly2(X) :- bondAtLeast2(X), not bondAtLeast3(X).
bondExactly3(X) :- bondAtLeast3(X), not bondAtLeast4(X).
bond1to3(X) :- bondAtLeast1(X), not bondAtLeast4(X).

acylCarbon(Y) :- hasAtom(X,Y), c(Y), hasAtom(X,Z), o(Z), double(Y,Z), double(Z,Y).

%noble gas atoms
noble(X) :- ar(X).
noble(X) :- he(X).
noble(X) :- kr(X).
noble(X) :- ne(X).
noble(X) :- rn(X).
noble(X) :- xe(X).
noble(X) :- og(X).

% cyclic molecules
cleanReachable(X,Y,Z) :- bond(X,Y), bond(Y,X), bond(Y,Z), bond(Z,Y), X != Y, Y != Z, X != Z.
cleanReachable(X,Z,W) :- cleanReachable(X,Y,Z), bond(Z,W), bond(W,Z), X != W, Y != W, W != Z.
closedLoopAtLeast3(X) :- cleanReachable(X,Y,Z), bond(Z,X), bond(X,Z).


% halogens (this is not part of the original rule set)
halogen(X) :- f(X).
halogen(X) :- cl(X).
halogen(X) :- br(X).
halogen(X) :- i(X).
halogen(X) :- at(X).
halogen(X) :- ts(X). 