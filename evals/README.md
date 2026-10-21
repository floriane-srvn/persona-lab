# Golden dataset V0 — revue manuelle

`golden-dataset.json` contient quatre scénarios de référence, un par niveau de grounding (`documented`, `inferred`, `speculative`, `not_established`). Les chunks attendus ont été choisis en lisant le corpus v1 (D01 à D06, 29 chunks). Il n'y a ni script d'évaluation ni LLM-juge : la revue est faite par une personne.

## Lancer un scénario

```
npm run ask -- "<question du scénario>" --debug
```

Chaque lancement est **payant** : un appel Voyage et un appel Anthropic. Il faut une autorisation avant tout lancement. Paramètres à garder fixes d'une revue à l'autre : `top_k` 5, aucun seuil de similarité, corpus v1, et le modèle indiqué dans `ANTHROPIC_MODEL`, à noter dans la revue. Chaque question se lance une fois, sans historique : la reformulation n'est pas encore branchée sur `npm run ask`.

`--debug` affiche les 5 chunks récupérés, leur score et ceux que Claude a cités.

## Grille de revue

Pour chaque chunk du top 5, le relecteur indique son étiquette selon le dataset : `essential`, `acceptable`, `near_miss`, ou `hors liste`. Une étiquette `hors liste` est discutée et notée, puis la liste est éventuellement mise à jour pour la version suivante.

| Mesure | Comment la calculer | Échelle |
|---|---|---|
| **Retrieval relevance** | Étiquette de chaque chunk récupéré | qualitative |
| **precision@5** | (chunks `essential` + `acceptable` dans le top 5) / nombre de chunks récupérés | 0 à 1 |
| **recall@5** | (chunks `essential` dans le top 5) / nombre de chunks `essential` | 0 à 1, sans objet si la liste est vide (EVAL-04) |
| **Faithfulness** | Chaque affirmation de `persona_response` est soutenue par un chunk cité, ou présentée comme une inférence ou une hypothèse | 2 = entièrement fidèle ; 1 = détail non soutenu sans incidence ; 0 = fait inventé |
| **Answer relevance** | La réponse traite la question posée, sans dériver vers un sujet voisin | 2 = répond ; 1 = partiellement ; 0 = hors sujet |
| **Niveau de grounding** | Niveau obtenu comparé au niveau attendu | conforme / alternatif accepté (à signaler) / non conforme |

## Hard gates

Un seul échec suffit à faire échouer le scénario, quels que soient les scores.

- Un **critère bloquant** du scénario est rencontré (`blocking_criteria`).
- La sortie ne respecte pas le contrat JSON, ou `source_ids` cite un chunk non récupéré. Le code le vérifie déjà : un échec à ce niveau est une erreur, pas une réponse.
- Julie parle d'elle à la troisième personne, se présente comme une IA ou mentionne documents, corpus ou passages.
- Le niveau de grounding est plus fort que les preuves : par exemple `documented` sans source de Julie, ou une spéculation présentée comme un fait.
- Faithfulness à 0.

## Fiche de revue (une ligne par scénario)

| Scénario | Date · modèle | Top 5 (étiquettes) | P@5 | R@5 | Faithfulness | Relevance | Niveau | Hard gates | Commentaire |
|---|---|---|---|---|---|---|---|---|---|
| EVAL-01 | | | | | | | | | |
| EVAL-02 | | | | | | | | | |
| EVAL-03 | | | | | | | | | |
| EVAL-04 | | | | | | | | | |

## Limites

- **Quatre scénarios** : c'est un point de contrôle, pas une mesure statistique. Un bon résultat ne prouve pas que le système est fiable.
- **Un seul lancement** : les réponses du modèle varient d'un appel à l'autre. Un résultat isolé peut être un hasard.
- **Les listes de chunks reflètent un jugement**, surtout pour EVAL-02 et EVAL-03, où les indices sont indirects. Le recall@5 dépend directement de la taille de la liste `essential`.
- **Ce que la precision@5 ne mesure pas** : avec `top_k` 5 et une ou deux réponses vraiment pertinentes, elle reste basse, même quand le retrieval est bon. Elle ne dit pas non plus si le bon chunk est en première position.
- **Revue humaine** : faithfulness et answer relevance restent subjectives. Une double relecture limite ce biais.
- **Hors périmètre** : la conversation sur plusieurs tours, la reformulation, les demandes de précision et l'interface. Le corpus est fictif et propre à Julie.
