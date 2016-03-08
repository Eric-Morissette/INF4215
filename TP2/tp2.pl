/* Session 1 */
cours(inf1005).
cours(inf1500).
cours(mth1101).
cours(mth1006).
cours(inf1040).

/* Session 2 */
cours(inf1010).
cours(log1000).
cours(inf1600).
cours(mth1102).
cours(inf1995).


prerequis(inf1005, inf1010).
prerequis(inf1005, log1000).
prerequis([inf1005, inf1500], inf1600).
prerequis(mth1101, mth1102).
prerequis(inf1040, inf1995).

corequis(mth1006, mth1102).
corequis([inf1600, log1000], inf1995).


/* Session 3 */
cours(inf2010).
cours(log2410).
cours(phs1102).
cours(mth1110).
cours(mth1210).
cours(log2810).

prerequis(inf1010, inf2010).
prerequis([inf1010, log1000], log2410).
prerequis(mth1006, mth1110).

corequis(inf2810, inf2010).
corequis(inf2010, inf2810).
corequis(mth1110, mth1210).

/* Session 4 */
cours(inf2705).
cours(inf2610).
cours(ele2302).
cours(mth2302).
cours(inf2990).

prerequis([inf2010, mth1006], inf2705).
prerequis([inf1600, inf1010], inf2610).
prerequis(mth1110, ele2302).
prerequis([inf1995, inf2010, log2410], inf2990).

corequis(inf2705, inf2990).
corequis(inf2990, inf2705).
corequis(mth1101, mth2302).

/* Session 5 */
cours(inf3710).
cours(phs4700).
cours(gbm1610).
cours(ele2305).
cours(inf3500).
cours(inf3405).
cours(ssh5201).

coursOption(5, [phs4700, gbm1610, ele2305]).

prerequis(inf2010, inf3710).
prerequis(mth1210, phs4700).
prerequis(phs1102, ele2305).
prerequis(inf1600, inf3500).
prerequis(27, ssh5201).

corequis(inf2610, inf3710).
corequis(ele2302, inf3500).
corequis(mth2302, inf3405).

/* Session 6 */
cours(inf3610).
cours(inf4420).
cours(ssh5501).
cours(inf3990).
cours(phs1101).

prerequis([inf2610, inf3500], inf3610).
prerequis([inf2610, inf3405], inf4420).
prerequis(27, ssh5501).
prerequis([inf3405, inf3500], inf3990).

corequis(inf3610, inf3990).
corequis(inf3990, inf3610).



coursValides(ListeCours):-
	forall(member(X, ListeCours), cours(X)).

getPrerequis(Requis, Cours) :-
	(prerequis(X, Cours) ->
		getListe(Requis, X)
	;
		Requis = []
	).

getCorequis(Requis, Cours) :-
	(corequis(X, Cours) ->
		getListe(Requis, X)
	;
		Requis = []
	).

getListe(Liste, TempListe):-
	(is_list(TempListe) ->
		append([], TempListe, Liste)
	;
		Liste = [TempListe]
	).

getCreditsRequis(Credits, Cours):-
	Credits = max(0, getCredits(Cours)).

validationChoixCours(Session, TempListeChoix, TempListeFaits):-
	getListe(ListeChoix, TempListeChoix),
	getListe(ListeFaits, TempListeFaits),
	coursValides(ListeChoix),
	coursValides(ListeFaits),
	forall(member(X, ListeChoix), validationPrerequis(X, ListeFaits)),
	append(ListeFaits, ListeChoix, ListeComplete),
	forall(member(X, ListeChoix), validationCorequis(X, ListeComplete)),
	calculerCreditsFaits(NombreCreditsFaits, ListeFaits),
	forall(member(X, ListeChoix), validationCreditsRequis(X, NombreCreditsFaits)).

validationPrerequis(Cours, ListeFaits):-
	getPrerequis(Requis, Cours),
	forall(member(X, Requis), checkDansListe(X, ListeFaits)).

validationCorequis(Cours, ListeComplete):-
	getCorequis(Requis, Cours),
	forall(member(X, Requis), checkDansListe(X, ListeComplete)).

validationCreditsRequis(Cours, NombreCreditsFaits):-
	getCreditsRequis(CreditsRequis, Cours)
	NombreCreditsFaits > CreditsRequis.

checkDansListe(Cours, ListeFaits):-
	member(Cours, ListeFaits).






