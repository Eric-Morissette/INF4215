/* Themes */
theme(genieInformatique).

sousTheme(programmation, genieInformatique).
sousTheme(materiel, genieInformatique).
sousTheme(reseautique, genieInformatique).
sousTheme(baseDeDonnees, genieInformatique).
sousTheme(securite, genieInformatique).
sousTheme(sciencesHumaines, genieInformatique).
sousTheme(mathematiques, genieInformatique).
sousTheme(physique, genieInformatique).
sousTheme(projetIntegrateur, genieInformatique).

sousTheme(programmationProcedurale, programmation).
sousTheme(programmationOrienteeObjet, programmation).
sousTheme(programmationWeb, programmation).
sousTheme(algorithmie, programmation).
sousTheme(qualiteLogicielle, programmation).
sousTheme(infographie, programmation).
sousTheme(systemeExploitation, programmation).

sousTheme(securiteReseau, reseautique).
sousTheme(informatiqueMobile, reseautique).

sousTheme(economie, sciencesHumaines).
sousTheme(sociologie, sciencesHumaines).
sousTheme(ethique, sciencesHumaines).

sousTheme(calculIntegral, mathematiques).
sousTheme(calculDifferentiel, mathematiques).
sousTheme(algebreLineaire, mathematiques).
sousTheme(mathematiquesDiscretes, mathematiques).
sousTheme(probabilitesStatistiques, mathematiques).

sousTheme(electronique, physique).
sousTheme(electrique, physique).
sousTheme(magnetisme, physique).
sousTheme(mecanique, physique).

theme(X):-
	sousTheme(X, Y),
	theme(Y).

obligatoire(X):-
	cours(X),
	\+ optionnel(X).

echange(Cours):-
	obligatoire(Cours),
	\+ projet(Cours).

/* Session 1 */
cours(inf1005).
cours(inf1500).
cours(mth1101).
cours(mth1006).
cours(inf1040).

equivalence(inf1005).
equivalence(mth1101).
equivalence(mth1006).

aborde(inf1005, programmationProcedurale).
aborde(inf1500, materiel).
aborde(mth1101, calculDifferentiel).
aborde(mth1006, algebreLineaire).
aborde(inf1040, qualiteLogicielle).

credits(inf1005, 3).
credits(inf1500, 3).
credits(mth1101, 2).
credits(mth1006, 2).
credits(inf1040, 3).

annee(inf1005, 1).
annee(inf1500, 1).
annee(mth1101, 1).
annee(mth1006, 1).
annee(inf1040, 1).

language(inf1005, cpp).
language(inf1500, vhdl).

programme(inf1005, [logiciel, informatique, biomedical, chimique]).
programme(inf1500, [logiciel, informatique]).
programme(mth1101, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(mth1006, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(inf1040, [logiciel, informatique]).

/* Session 2 */
cours(inf1010).
cours(log1000).
cours(inf1600).
cours(mth1102).
cours(inf1995).

equivalence(mth1102).

projet(inf1995).

aborde(inf1010, programmationOrienteeObjet).
aborde(log1000, qualiteLogicielle).
aborde(inf1600, materiel).
aborde(mth1102, calculIntegral).
aborde(inf1995, projetIntegrateur). 

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

annee(inf1010, 1).
annee(log1000, 1).
annee(inf1600, 1).
annee(mth1102, 1).
annee(inf1995, 1).

language(inf1010, cpp).
language(log1000, scala).
language(inf1600, assembleur).
language(inf1995, c).

programme(inf1010, [logiciel, informatique]).
programme(log1000, [logiciel, informatique]).
programme(inf1600, [logiciel, informatique]).
programme(mth1102, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(inf1995, [logiciel, informatique]).

/* Session 3 */
cours(inf2010).
cours(log2410).
cours(phs1102).
cours(mth1110).
cours(mth1210).
cours(log2810).

equivalence(mth1110).

aborde(inf2010, algorithmie).
aborde(log2410, qualiteLogicielle).
aborde(phs1102, magnetisme).
aborde(mth1110, calculDifferentiel).
aborde(mth1210, calculDifferentiel).
aborde(log2810, mathematiquesDiscretes).

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

annee(inf2010, 2).
annee(log2410, 2).
annee(phs1102, 2).
annee(mth1110, 2).
annee(mth1210, 2).
annee(log2810, 2).

language(inf2010, java).
language(log2810, cpp).
language(mth1210, matlab).

programme(inf2010, [logiciel, informatique]).
programme(log2410, [logiciel, informatique]).
programme(phs1102, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(mth1110, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(mth1210, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(log2810, [logiciel, informatique]).

/* Session 4 */
cours(inf2705).
cours(inf2610).
cours(ele2302).
cours(mth2302).
cours(inf2990).

equivalence(mth2302).

projet(inf2990).

aborde(inf2705, infographie).
aborde(inf2610, systemeExploitation).
aborde(ele2302, electronique).
aborde(mth2302, probabilitesStatistiques).
aborde(inf2990, projetIntegrateur).

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

annee(inf2705, 2).
annee(inf2610, 2).
annee(ele2302, 2).
annee(mth2302, 2).
annee(inf2990, 2).

language(inf2705, glsl).
language(inf2705, c).
language(inf2610, cpp).
language(inf2990, cpp).
language(inf2990, java).

programme(inf2705, [logiciel, informatique]).
programme(inf2610, [logiciel, informatique]).
programme(ele2302, [logiciel, informatique, electrique]).
programme(mth2302, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(inf2990, [logiciel, informatique]).

/* Session 5 */
cours(inf3710).
cours(phs4700).
cours(gbm1610).
cours(ele2305).
cours(inf3500).
cours(inf3405).
cours(ssh5201).

inverse(inf3500).

equivalence(ssh5201).

optionnel(phs4700).
optionnel(gbm1610).
optionnel(ele2305).

aborde(inf3710, baseDeDonnees).
aborde(phs4700, physique).
aborde(gbm1610, physique).
aborde(ele2305, electronique).
aborde(inf3500, materiel).
aborde(inf3405, reseautique).
aborde(ssh5201, economie).

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

annee(inf3710, 3).
annee(phs4700, 3).
annee(gbm1610, 3).
annee(ele2305, 3).
annee(inf3500, 3).
annee(inf3405, 3).
annee(ssh5201, 3).

language(inf3710, sql).
language(inf3710, java).
language(phs4700, matlab).
language(inf3500, vhdl).
language(inf3405, cpp).

programme(inf3710, [logiciel, informatique]).
programme(phs4700, [logiciel, informatique, biomedical, physique, electrique]).
programme(gbm1610, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(ele2305, [logiciel, informatique, electrique]).
programme(inf3500, [logiciel, informatique]).
programme(inf3405, [logiciel, informatique]).
programme(ssh5201, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).

/* Session 6 */
cours(inf3610).
cours(inf4420).
cours(ssh5501).
cours(inf3990).
cours(phs1101).

inverse(inf4420).

equivalence(ssh5501).
equivalence(phs1101).

projet(inf3990).

aborde(inf3610, materiel).
aborde(inf4420, securite).
aborde(ssh5501, ethique).
aborde(inf3990, projetIntegrateur).
aborde(phs1101, mecanique).

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

annee(inf3610, 3).
annee(inf4420, 3).
annee(ssh5501, 3).
annee(inf3990, 3).
annee(phs1101, 3).

language(inf3610, cpp).
language(inf3610, c).
language(inf4420, c).
language(inf3990, c).
language(inf3990, java).

programme(inf3610, [logiciel, informatique]).
programme(inf4420, [logiciel, informatique]).
programme(ssh5501, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).
programme(inf3990, [informatique]).
programme(phs1101, [logiciel, informatique, biomedical, chimique, physique, minier, electrique]).


/* Getters */
getListe(Liste, TempListe):-
	(is_list(TempListe) ->
		append([], TempListe, Liste)
	;
		Liste = [TempListe]
	).

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

printElement(E):-
	print(E),
	print('\n').

getDiff(A, B, X):-
	Y is A - B,
	X is abs(Y).


/* Question 1 */
coursValides(ListeCours):-
	forall(member(X, ListeCours), cours(X)).

validationChoixCours(TempListeChoix, TempListeFaits):-
	getListe(ListeChoix, TempListeChoix),
	getListe(ListeFaits, TempListeFaits),

	(coursValides(ListeChoix) ->
		printElement('Validation des cours choisis: OK')
	;
		printElement('Validation des cours choisis: Erreur'),
		fail
	),

	(coursValides(ListeFaits) ->
		printElement('Validation des cours faits:   OK')
	;
		printElement('Validation des cours faits:   Erreur'),
		fail
	),

	forall(member(X, ListeChoix), validationPrerequis(X, ListeFaits)),
	printElement('Validation des prerequis:     OK'),
	append(ListeFaits, ListeChoix, ListeComplete),
	forall(member(X, ListeChoix), validationCorequis(X, ListeComplete)),
	printElement('Validation des corequis:      OK'),
	forall(member(X, ListeChoix), validationCreditsRequis(X, ListeFaits)),
	printElement('Validation des credits requis pour suivre les cours: OK').

validationPrerequis(Cours, ListeFaits):-
	getPrerequis(Requis, Cours),
	(forall(member(X, Requis), checkDansListe(X, ListeFaits)) ->
		true
	;
		printElement('Validation des prerequis:     Erreur'),
		print('\t'),
		print(Cours),
		print(' a comme prerequis '),
		printElement(Requis),
		fail
	).

validationCorequis(Cours, ListeComplete):-
	getCorequis(Requis, Cours),
	(forall(member(X, Requis), checkDansListe(X, ListeComplete)) ->
		true
	;
		printElement('Validation des corequis:      Erreur'),
		print('\t'),
		print(Cours),
		print(' a comme corequis '),
		printElement(Requis),
		fail
	).

validationCreditsRequis(Cours, ListeFaits):-
	aggregate(sum(N), C, (credits(C, N), member(C, ListeFaits)), TotalCredits),
	getCreditsRequis(Cours, K),
	(TotalCredits >= K ->
		true
	;
		printElement('Validation des credits requis pour suivre les cours: Erreur'),
		print('\t'),
		print(Cours),
		print(' a comme prerequis d''avoir accumule au moins '),
		print(TotalCredits),
		print(' credits'),
		fail
	).

checkDansListe(Cours, ListeFaits):-
	member(Cours, ListeFaits).

/* Question 2 */
qualiteChoixCours(ListeCours):-
	aggregate(sum(A), C, (member(C, ListeCours), annee(C, A)), SumAnnee),
	length(ListeCours, NbCours),
	Moyenne is SumAnnee / NbCours,
	aggregate_all(sum(Diff), C, (member(C, ListeCours), annee(C, A), getDiff(A, Moyenne, Diff)), SumDiff),
	ScoreTotal is SumDiff / NbCours,
	print('Voici le score de votre choix de cours: '),
	printElement(ScoreTotal),
	printElement('La minimisation de cette valeur represente la localite de vos choix de cours (Les cours choisis devraient etre prevus pour la meme annee)').

/* Question 3 */
coursSelonSujet(Sujet):-
	findall(X, sousTheme(X, Sujet), Liste),
	length(Liste, L),
	(L > 0 ->
		forall(member(Y, Liste), coursSelonSujet(Y))
	;
		forall(aborde(C, Sujet), printElement(C))
	).

/* Question 4 */
getLanguage(Language):-
	forall(language(X, Language), printElement(X)).

/* Question 5 */
getProgrammes(Cours):-
	(programme(Cours, P) ->
		getListe(Programmes, P)
	;
		Programmes = []
	),
	forall(member(X, Programmes), printElement(X)).

/* Question 6 */
getInverse:-
	forall(inverse(X), printElement(X)).

/* Question 7 */
getStatusCours(Cours):-
	cours(Cours),
	(projet(Cours) ->
		printElement('Projet')
	;
		(optionnel(Cours) ->
			printElement('Optionnel')
		;
			printElement('Obligatoire')
		)
	).

/* Question 9 */
getEquivalence(Cours):-
	cours(Cours),
	(equivalence(Cours) -> 
		aborde(Cours, Sujet),
		print('Cours creditable si l''eleve a deja fait un cours similaire de: '),
		printElement(Sujet)
	;
		printElement('Aucune equivalence possible')
	).

/* Question 10 */
getCoursEchangeable:-
	forall(echange(C), printElement(C)).

















