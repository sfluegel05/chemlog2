%% for these rules, X is a molecule (either explicit or implicit)

% molecules that contain astatine
astatineMolEntity(X) :- molecule(X), hasAtom(X,Y), at(Y).

% molecules that contain bromine
bromineMolEntity(X) :- molecule(X), hasAtom(X,Y), br(Y).

% molecules that contain chlorine
chlorineMolEntity(X) :- molecule(X), hasAtom(X,Y), cl(Y).

% molecules that contain fluorine
fluorineMolEntity(X) :- molecule(X), hasAtom(X,Y), f(Y).

% molecules that contain iodine
iodineMolEntity(X) :- molecule(X), hasAtom(X,Y), i(Y).

tsMolEntity(X) :- molecule(X), hasAtom(X,Y), ts(Y).

% molecules that contain halogens
halogenMolEntity(X) :- bromineMolEntity(X).
halogenMolEntity(X) :- chlorineMolEntity(X).
halogenMolEntity(X) :- fluorineMolEntity(X).
halogenMolEntity(X) :- astatineMolEntity(X).
halogenMolEntity(X) :- iodineMolEntity(X).
halogenMolEntity(X) :- tsMolEntity(X).


% molecules that contain polonium
poloniumMolEntity(X) :- molecule(X), hasAtom(X,Y), po(Y).

% molecules that contain selenium
seleniumMolEntity(X) :- molecule(X), hasAtom(X,Y), se(Y).

% molecules that contain sulfur
sulfurMolEntity(X) :- molecule(X), hasAtom(X,Y), s(Y).

% molecules that contain tellurium
telluriumMolEntity(X) :- molecule(X), hasAtom(X,Y), te(Y).

livermoriumMolEntity(X) :- molecule(X), hasAtom(X,Y), lv(Y).

% molecules that are chalcogens
chalcogenMolEntity(X) :- oxygenMolEntity(X).
chalcogenMolEntity(X) :- poloniumMolEntity(X).
chalcogenMolEntity(X) :- seleniumMolEntity(X).
chalcogenMolEntity(X) :- sulfurMolEntity(X).
chalcogenMolEntity(X) :- telluriumMolEntity(X).
chalcogenMolEntity(X) :- livermoriumMolEntity(X).

% molecules that contain antimony
antimonyMolEntity(X) :- molecule(X), hasAtom(X,Y), sb(Y).

% molecules that contain arsenic
arsenicMolEntity(X) :- molecule(X), hasAtom(X,Y), as(Y).

% molecules that contain bismuth
bismuthMolEntity(X) :- molecule(X), hasAtom(X,Y), bi(Y).

% molecules that contain nitrogen
nitrogenMolEntity(X) :- molecule(X), hasAtom(X,Y), n(Y).

moscoviumMolEntity(X) :- molecule(X), hasAtom(X,Y), mc(Y).

% molecules that are pnictogens
pnictogenMolEntity(X) :- antimonyMolEntity(X).
pnictogenMolEntity(X) :- arsenicMolEntity(X).
pnictogenMolEntity(X) :- bismuthMolEntity(X).
pnictogenMolEntity(X) :- nitrogenMolEntity(X).
pnictogenMolEntity(X) :- phosphorusMolEntity(X).
pnictogenMolEntity(X) :- moscoviumMolEntity(X).

% molecules that contain chromium
chromiumMolEntity(X) :- molecule(X), hasAtom(X,Y), cr(Y).

% molecules that contain molybdenum
molybdenumMolEntity(X) :- molecule(X), hasAtom(X,Y), mo(Y).

% molecules that contain seaborgium
seaborgiumMolEntity(X) :- molecule(X), hasAtom(X,Y), sg(Y).

% molecules that contain tungsten
tungstenMolEntity(X) :- molecule(X), hasAtom(X,Y), w(Y).

% molecules that belong to the chromium group
chromiumGroupMolEntity(X) :- chromiumMolEntity(X).
chromiumGroupMolEntity(X) :- molybdenumMolEntity(X).
chromiumGroupMolEntity(X) :- seaborgiumMolEntity(X).
chromiumGroupMolEntity(X) :- tungstenMolEntity(X).

% molecules that contain noble gas atoms
nobleGasMolEntity(X) :- molecule(X), hasAtom(X,Y), noble(Y).

% molecules that contain carbon
carbonMolEntity(X) :- molecule(X), hasAtom(X,Y), c(Y).

% molecules that contain oxygen
oxygenMolEntity(X) :- molecule(X), hasAtom(X,Y), o(Y).

% molecules that contain hydrogen
hydrogenMolEntity(X) :- molecule(X), hasAtom(X,Y), h(Y).

% molecules that contain phosphorus
phosphorusMolEntity(X) :- molecule(X), hasAtom(X,Y), p(Y).

% molecules that do not contain carbon
inorganic(X) :- molecule(X), not carbonMolEntity(X).

% molecules that contain carbon, hydrogen and consist only of carbon and hydrogen
notHydroCarbon(X) :- hasAtom(X,Y), not c(Y), not h(Y).
hydroCarbon(X) :- carbonMolEntity(X), not notHydroCarbon(X).

% molecules that contain carbon, hydrogen and halogen and consist only of carbon, hydrogen and halogen
notHaloHydroCarbon(X) :- hasAtom(X,Y), not c(Y), not h(Y), not halogen(Y).
haloHydroCarbon(X) :- carbonMolEntity(X), halogenMolEntity(X), not notHaloHydroCarbon(X).

% molecules that contain exactly one carbon
atLeast2Carbons(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), c(Y2), Y1 != Y2.
exactly1Carbon(X) :- carbonMolEntity(X), not atLeast2Carbons(X).

% molecules that contain exactly two carbons
atLeast3Carbons(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), c(Y2), Y1 != Y2, hasAtom(X,Y3), c(Y3), Y1 != Y3, Y2 != Y3.
exactly2Carbons(X) :- atLeast2Carbons(X), not atLeast3Carbons(X).

% molecules that contain exactly three carbons
atLeast4Carbons(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), c(Y2), Y1 != Y2, hasAtom(X,Y3), c(Y3), Y1 != Y3, Y2 != Y3, hasAtom(X,Y4), c(Y4), Y1 != Y4, Y2 != Y4, Y3 != Y4.
exactly3Carbons(X) :- atLeast3Carbons(X), not atLeast4Carbons(X).

% molecules that contain exactly four carbons
atLeast5Carbons(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), c(Y2), Y1 != Y2, hasAtom(X,Y3), c(Y3), Y1 != Y3, Y2 != Y3, hasAtom(X,Y4), c(Y4), Y1 != Y4, Y2 != Y4, Y3 != Y4, hasAtom(X,Y5), c(Y5), Y1 != Y5, Y2 != Y5, Y3 != Y5, Y4 != Y5.
exactly4Carbons(X) :- atLeast4Carbons(X), not atLeast5Carbons(X).

% molecules that contain at least two atoms
polyatomic(X) :- molecule(X), hasAtom(X,Y1), hasAtom(X,Y2), Y1 != Y2.

% molecules that contains exactly one atom
monoatomic(X) :- molecule(X), hasAtom(X,Y1),not polyatomic(X).

% molecules that contain a carboxy group - modified (explicit H atom added)
carboxylicAcid(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), o(Y3), hasAtom(X,Y4), horc(Y4), hasAtom(X,Y5), h(Y5), double(Y1,Y2), double(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y1,Y4), single(Y4,Y1), single(Y3,Y5), single(Y5,Y3).

% molecules that contain exactly one carboxy group (not needed for ChEBI classification)
%atLeast2CarboxyGroups(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), o(Y3), hasAtom(X,Y4), horc(Y4), double(Y1,Y2), double(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y1,Y4), single(Y4,Y1), not midOxygen(Y3), not charged(Y3), hasAtom(X,Y5), c(Y5), hasAtom(X,Y6), o(Y6), hasAtom(X,Y7), o(Y7), hasAtom(X,Y8), horc(Y8), double(Y5,Y6), double(Y6,Y5), single(Y5,Y7), single(Y7,Y5), single(Y5,Y8), single(Y8,Y5), not midOxygen(Y7), not charged(Y7), Y1!=Y5.
%exactly1CarboxyGroup(X) :- carboxylicAcid(X), not atLeast2CarboxyGroups(X).

% molecules that contain exactly two carboxy groups
%atLeast3CarboxyGroups(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), o(Y3), hasAtom(X,Y4), horc(Y4), double(Y1,Y2), double(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y1,Y4), single(Y4,Y1), not midOxygen(Y3), not charged(Y3), hasAtom(X,Y5), c(Y5), hasAtom(X,Y6), o(Y6), hasAtom(X,Y7), o(Y7), hasAtom(X,Y8), horc(Y8), double(Y5,Y6), double(Y6,Y5), single(Y5,Y7), single(Y7,Y5), single(Y5,Y8), single(Y8,Y5), not midOxygen(Y7), not charged(Y7), hasAtom(X,Y9), c(Y9), hasAtom(X,Y10), o(Y10), hasAtom(X,Y11), o(Y11), hasAtom(X,Y12), horc(Y12), double(Y9,Y10), double(Y10,Y9), single(Y9,Y11), single(Y11,Y9), single(Y9,Y12), single(Y12,Y9), not midOxygen(Y11), not charged(Y11), Y1!=Y5, Y9!=Y5, Y9!=Y1.
%exactly2CarboxyGroups(X) :- atLeast2CarboxyGroups(X), not atLeast3CarboxyGroups(X).

% molecules that are carboxylic esters - modified (added inequality)
carboxylicEster(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), o(Y3), hasAtom(X,Y4), c(Y4), hasAtom(X,Y5), horc(Y5), double(Y1,Y2), double(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y1,Y5), single(Y5,Y1), single(Y3,Y4), single(Y4,Y3), Y4 != Y1.

% molecules that contain a benzene ring
hasBenzeneRing(X) :- molecule(X), hasAtom(X,Y1), c(Y1),  hasAtom(X,Y2), c(Y2), single(Y1,Y2), single(Y2,Y1), hasAtom(X,Y3), c(Y3), double(Y2,Y3), double(Y3,Y2), hasAtom(X,Y4), c(Y4), single(Y3,Y4), single(Y4,Y3), hasAtom(X,Y5), double(Y4,Y5), double(Y5,Y4), c(Y5), hasAtom(X,Y6), c(Y6), single(Y5,Y6), single(Y6,Y5), double(Y6,Y1), double(Y1,Y6).

% molecules that contain a four-membered ring
hasFourMemberedRing(X) :- molecule(X), hasAtom(X,Y1), hasAtom(X,Y2), Y1 != Y2, bond(Y1,Y2), bond(Y2,Y1), hasAtom(X,Y3), Y1 != Y3,  Y2 != Y3, bond(Y2,Y3), bond(Y3,Y2), hasAtom(X,Y4), Y1 != Y4, Y2 != Y4, Y3 != Y4, bond(Y3,Y4), bond(Y4,Y3), bond(Y4,Y1), bond(Y1,Y4).

% molecules that are amines
amine(X) :- molecule(X), hasAtom(X,Y1), n(Y1), bond1to3(Y1), hasAtom(X,Y2), horc(Y2), hasAtom(X,Y3), horc(Y3), hasAtom(X,Y4), c(Y4), single(Y1,Y2), single(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y1,Y4), single(Y4,Y1), not acylCarbon(Y2), not acylCarbon(Y3), not acylCarbon(Y4), Y2!=Y3, Y2!=Y4, Y3!=Y4.

% molecules that are aldehydes - modified (explicit H atom added)
aldehyde(X) :- molecule(X), hasAtom(X,Y1), c(Y1), bondExactly3(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), horc(Y3), hasAtom(X,Y4), h(Y4), double(Y1,Y2), double(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y3,Y4), single(Y4,Y3).

cyclic(X) :- hasAtom(X,Y), molecule(X), closedLoopAtLeast3(Y).

% molecules that are ketone
ketone(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), c(Y3), hasAtom(X,Y4), c(Y4), double(Y1,Y2), double(Y2,Y1), single(Y1,Y3), single(Y3,Y1), single(Y1,Y4), single(Y4,Y1), Y3!=Y4.

% organic molecules that are unsaturated
unsaturated(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), c(Y2), double(Y1,Y2), double(Y2,Y1).
unsaturated(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), c(Y2), triple(Y1,Y2), triple(Y2,Y1).

% organic molecules that are saturated
saturated(X) :- molecule(X), hasAtom(X,Y1), c(Y1), not unsaturated(X).

% organophosphorus molecules (direct C-P bond)
organophosphorus(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), p(Y2), bond(Y1,Y2), bond(Y2,Y1).
% ester extension (new): C-O-P linkage
organophosphorus(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), o(Y2), hasAtom(X,Y3), p(Y3), bond(Y1,Y2), bond(Y2,Y1), bond(Y2,Y3), bond(Y3,Y2).
% thioester extension (new): C-S-P linkage
organophosphorus(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), s(Y2), hasAtom(X,Y3), p(Y3), bond(Y1,Y2), bond(Y2,Y1), bond(Y2,Y3), bond(Y3,Y2).

% alkane molecules
alkane(X) :- saturated(X), hydroCarbon(X), not cyclic(X).

% haloalkane molecules
haloAlkane(X) :- saturated(X), haloHydroCarbon(X), not cyclic(X).

% molecules that contain at least one (possibly acyl) carbon connected with a non-carbon atom
heteroOrganic(X) :- molecule(X), hasAtom(X,Y1), c(Y1), hasAtom(X,Y2), not c(Y2),  not h(Y2), bond(Y1,Y2), bond(Y2,Y1).