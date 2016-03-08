/* Session 1 */
cours(inf1005).
cours(inf1500).
cours(mth1101).
cours(mth1006).
cours(inf1040).

credits(inf1005, 3).
credits(inf1500, 3).
credits(mth1101, 2).
credits(mth1006, 2).
credits(inf1040, 3).

/* Session 2 */
cours(inf1010).
cours(log1000).
cours(inf1600).
cours(mth1102).
cours(inf1995).

credits(inf1010, 3).
credits(log1000, 3).
credits(inf1600, 3).
credits(mth1102, 2).
credits(inf1995, 4).

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

credits(inf2010, 3).
credits(log2410, 3).
credits(phs1102, 3).
credits(mth1110, 2).
credits(mth1210, 1).
credits(log2810, 3).

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

credits(inf2705, 3).
credits(inf2610, 3).
credits(ele2302, 3).
credits(mth2302, 3).
credits(inf2990, 4).

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

credits(inf3710, 3).
credits(phs4700, 3).
credits(gbm1610, 3).
credits(ele2305, 3).
credits(inf3500, 3).
credits(inf3405, 3).
credits(ssh5201, 3).

coursOption(5, [phs4700, gbm1610, ele2305]).

prerequis(inf2010, inf3710).
prerequis(mth1210, phs4700).
prerequis(phs1102, ele2305).
prerequis(inf1600, inf3500).

corequis(inf2610, inf3710).
corequis(ele2302, inf3500).
corequis(mth2302, inf3405).

creditsRequis(27, ssh5201).

/* Session 6 */
cours(inf3610).
cours(inf4420).
cours(ssh5501).
cours(inf3990).
cours(phs1101).

credits(inf3610, 3).
credits(inf4420, 3).
credits(ssh5501, 2).
credits(inf3990, 4).
credits(phs1101, 3).

prerequis([inf2610, inf3500], inf3610).
prerequis([inf2610, inf3405], inf4420).
prerequis([inf3405, inf3500], inf3990).

corequis(inf3610, inf3990).
corequis(inf3990, inf3610).

creditsRequis(27, ssh5501).


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

getCredits(Cours, Credits):-
	(credits(Cours, X) ->
		Credits = X
	;
		Credits = 0
	).

getCreditsRequis(Cours, Credits):-
	(creditsRequis(X, Cours) ->
		Credits = X
	;
		Credits = 0
	).

validationChoixCours(Session, TempListeChoix, TempListeFaits):-
	getListe(ListeChoix, TempListeChoix),
	getListe(ListeFaits, TempListeFaits),
	coursValides(ListeChoix),
	coursValides(ListeFaits),
	forall(member(X, ListeChoix), validationPrerequis(X, ListeFaits)),
	append(ListeFaits, ListeChoix, ListeComplete),
	forall(member(X, ListeChoix), validationCorequis(X, ListeComplete)),
	forall(member(X, ListeChoix), validationCreditsRequis(X, ListeFaits)).

validationPrerequis(Cours, ListeFaits):-
	getPrerequis(Requis, Cours),
	forall(member(X, Requis), checkDansListe(X, ListeFaits)).

validationCorequis(Cours, ListeComplete):-
	getCorequis(Requis, Cours),
	forall(member(X, Requis), checkDansListe(X, ListeComplete)).

validationCreditsRequis(Cours, ListeFaits):-
	aggregate(sum(N), C, (credits(C, N), member(C, ListeFaits)), TotalCredits),
	print(TotalCredits),
	getCreditsRequis(Cours, K),
	TotalCredits >= K.
	
checkDansListe(Cours, ListeFaits):-
	member(Cours, ListeFaits).



language(inf1005, cpp).
language(inf1010, cpp).
language(inf2010, java).
language(inf1995, cpp).
language(log1000, scala).

getLanguage(Language):-
	forall(language(X, Language), print(X)).



inverse(inf3500).
inverse(inf4215).

getInverse(_):-
	forall(inverse(X), print(X)).







