import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { buildClarificationQuestion, detectUnresolvedReferences, normalizeWord } from "./reference-check.ts";
import type { ConversationMessage } from "./reformulation.ts";

const pm = (content: string): ConversationMessage => ({ role: "pm", content });
const julie = (content: string): ConversationMessage => ({ role: "julie", content });
const MEAL_KITS = [
  pm("Que pensez-vous des paniers-repas livrés ?"),
  julie("Si ça remplace vraiment une bonne partie des courses, je pourrais regarder. Mais 60 € en plus des courses, certainement pas."),
];

const detect = (question: string, window: ConversationMessage[] = MEAL_KITS, excluded = 0) =>
  detectUnresolvedReferences(question, window, excluded).map((d) => [d.kind, d.expression, d.question_to_pm]);

describe("démonstratifs sans antécédent : détectés", () => {
  const cases: [string, string, string][] = [
    ["Et cette application, vous l'utiliseriez ?", "cette application", "De quelle application parlez-vous ?"],
    ["Et cet abonnement, vous le garderiez longtemps ?", "cet abonnement", "De quel abonnement parlez-vous ?"],
    ["Et ce robot, il vous ferait gagner du temps ?", "ce robot", "De quel robot parlez-vous ?"],
    ["Vous prendriez cette application-là ?", "cette application-là", "De quelle application parlez-vous ?"],
    ["Et ce petit robot ?", "ce petit robot", "De quel petit robot parlez-vous ?"],
    ["Ce genre de service vous tenterait ?", "Ce genre de service", "De quel genre de service parlez-vous ?"],
    ["Et ce type d'appareil ?", "ce type d'appareil", "De quel type d'appareil parlez-vous ?"],
    ["Ces applications vous plaisent ?", "Ces applications", "Pouvez-vous préciser ce que vous entendez par « Ces applications » ?"],
    ["Et cette application gratuite, vous l'utiliseriez ?", "cette application", "De quelle application parlez-vous ?"],
  ];
  for (const [question, expression, clarification] of cases) {
    test(question, () => assert.deepEqual(detect(question), [["demonstrative", expression, clarification]]));
  }

  test("première question sans historique", () => {
    assert.deepEqual(detect("Et cette application, vous l'utiliseriez ?", []), [["demonstrative", "cette application", "De quelle application parlez-vous ?"]]);
  });

  test("objet voisin : « paniers-repas » ne fournit pas d'antécédent à « cette application »", () => {
    assert.equal(detect("Et cette application, vous l'utiliseriez ?").length, 1);
  });
});

describe("faux positifs évités", () => {
  const cases: [string, string, ConversationMessage[]?][] = [
    ["décrite par un complément", "Et cette application de recettes qui permet de cuisiner les restes, vous l'utiliseriez ?"],
    ["décrite par une relative", "Et cette application qui calcule les portions ?"],
    ["décrite par « pour »", "Et cette application pour les listes, vous la paieriez ?"],
    ["décrite par « d' »", "Et ce service d'épicerie en ligne ?"],
    ["expression temporelle (semaine)", "Et cette semaine, qui a fait les courses chez vous ?"],
    ["expression temporelle (soir)", "Ce soir, vous cuisinez quoi ?"],
    ["expression temporelle (week-end)", "Et ce week-end, vous faites le marché ?"],
    ["expression temporelle (année)", "Cette année, votre budget a augmenté ?"],
    ["nom de discours (point)", "Sur ce point, vous êtes sûre ?"],
    ["nom de discours (prix)", "Et à ce prix-là, vous les prendriez ?"],
    ["« ce » pronom (ce que)", "C'est ce que vous voulez ?"],
    ["« ce » pronom (ce serait)", "Ce serait trop cher ?"],
    ["« ce » pronom (ce dont)", "Pour revenir à ce dont on parlait au début : qu'est-ce qui vous ferait essayer ?"],
    ["antécédent nommé", "Et cette application, vous la paieriez ?", [pm("Avez-vous une application de listes ?"), julie("Oui.")]],
    ["synonyme appli", "Et cette application, vous la paieriez ?", [pm("Avez-vous une appli pour vos listes de courses ?"), julie("Oui.")]],
    ["synonyme abo", "Et cet abonnement, vous le garderiez ?", [pm("Vous avez un abo de streaming ?"), julie("Oui.")]],
    ["pluriel et majuscules", "Et cette application ?", [pm("Utilisez-vous des Applications de cuisine ?"), julie("Non.")]],
    ["accents", "Et ce réfrigérateur connecté ?", [pm("Votre refrigerateur est-il récent ?"), julie("Oui.")]],
    ["mot composé", "Et ce panier, il vous suffirait ?"],
    ["antécédent plus tôt dans la même question", "J'ai vu une application de menus. Cette application vous plairait ?"],
  ];
  for (const [name, question, window] of cases) {
    test(name, () => assert.deepEqual(detect(question, window ?? MEAL_KITS), []));
  }
});

describe("début de conversation hors de la fenêtre", () => {
  test("messages exclus : « le tout premier sujet » est introuvable", () => {
    assert.deepEqual(detect("Revenons au tout premier sujet : qu'en pensez-vous finalement ?", MEAL_KITS, 2), [
      ["conversation_start", "tout premier sujet", "De quel sujet parlez-vous ?"],
    ]);
  });

  test("messages exclus : « au début » en fin de proposition", () => {
    assert.deepEqual(detect("Ce dont on parlait au début : vous en pensez quoi ?", MEAL_KITS, 4), [
      ["conversation_start", "au début", "De quel sujet parlez-vous ?"],
    ]);
  });

  test("aucun historique : le premier sujet n'existe pas encore", () => {
    assert.equal(detect("Revenons au premier sujet.", []).length, 1);
  });

  test("fenêtre complète : le premier sujet y figure, le modèle s'en charge", () => {
    assert.deepEqual(detect("Revenons au tout premier sujet : qu'en pensez-vous finalement ?", MEAL_KITS, 0), []);
  });

  test("« au début de la semaine » n'est pas le début de la conversation", () => {
    assert.deepEqual(detect("Vous faites les courses au début de la semaine ?", MEAL_KITS, 3), []);
  });
});

describe("questions de clarification", () => {
  test("modèles de phrase selon le déterminant, sans opinion de Julie", () => {
    const questions = [
      buildClarificationQuestion("cette application"),
      buildClarificationQuestion("cet abonnement"),
      buildClarificationQuestion("ce forfait-là"),
      buildClarificationQuestion("tout premier sujet"),
      buildClarificationQuestion("celle-là"),
    ];
    assert.deepEqual(questions, [
      "De quelle application parlez-vous ?",
      "De quel abonnement parlez-vous ?",
      "De quel forfait parlez-vous ?",
      "De quel sujet parlez-vous ?",
      "Pouvez-vous préciser ce que vous entendez par « celle-là » ?",
    ]);
    for (const q of questions) {
      assert.match(q, /\?$/);
      assert.ok(q.length <= 120);
      assert.doesNotMatch(q, /\bje\b|j'|aime|préf|pense|envie/i);
    }
  });

  test("normalisation commune aux deux côtés de la comparaison", () => {
    assert.equal(normalizeWord("Applications"), normalizeWord("application"));
    assert.equal(normalizeWord("réfrigérateur"), normalizeWord("refrigerateur"));
    assert.equal(normalizeWord("l'abonnement"), normalizeWord("abonnement"));
    assert.equal(normalizeWord("application-là"), normalizeWord("application"));
  });
});

describe("limites connues de l'heuristique (documentées, non corrigées)", () => {
  test("les pronoms ne sont pas détectés : le modèle doit les signaler", () => {
    assert.deepEqual(detect("Et celle-là, vous l'utiliseriez ?"), []);
    assert.deepEqual(detect("Et ça, ça vous tenterait ?"), []);
  });

  test("un synonyme hors liste déclenche une demande de précision inutile", () => {
    assert.equal(detect("Et ce logiciel, vous le paieriez ?", [pm("Avez-vous une application de budget ?"), julie("Oui.")]).length, 1);
  });

  test("un même nom désignant un autre objet passe pour un antécédent", () => {
    assert.deepEqual(detect("Et cette application, vous la paieriez ?", [pm("Utilisez-vous l'application de votre banque ?"), julie("Oui.")]), []);
  });

  test("un adjectif après le nom n'est pas une description suffisante", () => {
    assert.equal(detect("Et cette application gratuite ?").length, 1);
  });
});
