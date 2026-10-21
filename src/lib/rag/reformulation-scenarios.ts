import type { ContextType, ConversationMessage, DetectionSource, ReformulationInput, ReformulationOutcome } from "./reformulation.ts";

/**
 * Conversation scenarios shared by the tests and `npm run reformulate`.
 * Julie's lines are simulated conversation turns, not corpus evidence.
 */
export interface ReadyExpectation {
  status: "ready";
  context_type: ContextType;
  history_needed_for_query: boolean;
  /** True when the question must be copied verbatim (autonomous question). */
  query_unchanged?: boolean;
  /** Terms (case-insensitive) that must not appear in the query, e.g. a neighbouring topic. */
  forbidden?: string[];
  /** Terms the query must contain: each group is satisfied by any of its alternatives (case-insensitive). */
  required?: string[][];
  /** What a good query looks like, for a human reading the Terminal output. */
  note: string;
}

export interface ClarificationExpectation {
  status: "clarification_needed";
  unresolved_expression: string;
  question_to_pm: string;
  /** Accepted detection sources ("both" only appears in comparison mode). */
  detected_by: DetectionSource[];
  note: string;
}

export interface ReformulationScenario {
  name: string;
  description: string;
  input: ReformulationInput;
  expected: ReadyExpectation | ClarificationExpectation;
}

export interface ScenarioCheck {
  label: string;
  ok: boolean;
}

/** Mechanical checks of a real outcome against a scenario; the note still needs a human read. */
export function evaluateScenario(scenario: ReformulationScenario, outcome: ReformulationOutcome): ScenarioCheck[] {
  const e = scenario.expected;
  const checks: ScenarioCheck[] = [{ label: `status ${e.status}`, ok: outcome.status === e.status }];
  if (e.status === "clarification_needed") {
    const c = outcome.status === "clarification_needed" ? outcome.clarification : null;
    checks.push(
      { label: `expression « ${e.unresolved_expression} »`, ok: c?.unresolved_expression === e.unresolved_expression },
      { label: `question « ${e.question_to_pm} »`, ok: c?.question_to_pm === e.question_to_pm },
      { label: `détection ${e.detected_by.join(" ou ")}`, ok: !!c && e.detected_by.includes(c.detected_by) },
    );
    return checks;
  }
  if (outcome.status !== "ready") return checks;
  const r = outcome.reformulation;
  checks.push(
    { label: `context_type ${e.context_type}`, ok: r.context_type === e.context_type },
    { label: `history_needed_for_query ${e.history_needed_for_query}`, ok: r.history_needed_for_query === e.history_needed_for_query },
  );
  if (e.query_unchanged) {
    checks.push({ label: "query identique à la question", ok: r.query === scenario.input.latest_question.trim() });
  }
  for (const group of e.required ?? []) {
    checks.push({
      label: `query avec « ${group.join(" » ou « ")} »`,
      ok: group.some((term) => r.query.toLowerCase().includes(term.toLowerCase())),
    });
  }
  for (const term of e.forbidden ?? []) {
    checks.push({ label: `query sans « ${term} »`, ok: !r.query.toLowerCase().includes(term.toLowerCase()) });
  }
  return checks;
}

const pm = (content: string): ConversationMessage => ({ role: "pm", content });
const julie = (content: string): ConversationMessage => ({ role: "julie", content });

const MEAL_KITS = [
  pm("Que pensez-vous des paniers-repas livrés ?"),
  julie("Si ça remplace vraiment une bonne partie des courses, je pourrais regarder. Mais 60 € en plus des courses, certainement pas."),
];

/** Six exchanges: the first one (meal kits) falls outside a five-exchange window. */
const LONG_HISTORY = [
  ...MEAL_KITS,
  pm("Comment organisez-vous les dîners du mardi ?"),
  julie("Les mardis, avec les activités des enfants, on rentre tard : il faut que ce soit rapide."),
  pm("Qui fait les courses chez vous ?"),
  julie("C'est surtout moi, le samedi matin en général."),
  pm("Vous utilisez une liste de courses ?"),
  julie("Une liste sur mon téléphone, mais je l'oublie souvent."),
  pm("Achetez-vous des produits bio ?"),
  julie("Parfois des pommes bio et des œufs plein air."),
  pm("Et au marché ?"),
  julie("Je prends des produits locaux quand c'est raisonnable."),
];

export const REFORMULATION_SCENARIOS: ReformulationScenario[] = [
  {
    name: "premiere-question",
    description: "Première question, sans historique : aucun appel Anthropic.",
    input: { latest_question: "Quel budget vous fixez-vous pour les courses ?", history: [] },
    expected: {
      status: "ready",
      context_type: "new_topic",
      history_needed_for_query: false,
      query_unchanged: true,
      note: "Question recopiée telle quelle.",
    },
  },
  {
    name: "relance-dependante",
    description: "Relance qui ne se comprend qu'avec l'historique, avec de nouveaux paramètres.",
    input: { latest_question: "Et à 45 € par semaine pour 4 repas ?", history: MEAL_KITS },
    expected: {
      status: "ready",
      context_type: "follow_up",
      history_needed_for_query: true,
      note: "Mentionne les paniers-repas livrés et conserve 45 € par semaine et 4 repas.",
    },
  },
  {
    name: "relance-autonome",
    description: "Relance sur le même sujet, déjà compréhensible seule.",
    input: { latest_question: "Seriez-vous prête à payer 45 € par semaine pour des paniers-repas livrés ?", history: MEAL_KITS },
    expected: {
      status: "ready",
      context_type: "follow_up",
      history_needed_for_query: false,
      query_unchanged: true,
      note: "Question recopiée telle quelle.",
    },
  },
  {
    name: "nouveau-sujet",
    description: "Question autonome sur un autre sujet.",
    input: { latest_question: "Combien de temps passez-vous en cuisine les soirs de semaine ?", history: MEAL_KITS },
    expected: {
      status: "ready",
      context_type: "new_topic",
      history_needed_for_query: false,
      forbidden: ["panier"],
      note: "Aucune mention des paniers-repas.",
    },
  },
  {
    name: "retour-sujet",
    description: "Retour aux paniers-repas, abordés deux échanges plus tôt.",
    input: {
      latest_question: "Pour revenir à ce dont on parlait au début : qu'est-ce qui vous ferait essayer ?",
      history: [
        ...MEAL_KITS,
        pm("Avez-vous des abonnements en ce moment ?"),
        julie("Oui, une plateforme de streaming, mais je fais attention à ne pas en accumuler."),
      ],
    },
    expected: {
      status: "ready",
      context_type: "topic_return",
      history_needed_for_query: true,
      note: "Porte sur ce qui pourrait faire essayer des paniers-repas livrés, sans parler d'abonnements.",
    },
  },
  {
    name: "retour-sujet-implicite",
    description: "Retour aux paniers-repas deux échanges plus tard, sans marqueur comme « revenons à ».",
    input: {
      latest_question: "Et trois repas par semaine, ça vous semblerait suffisant ?",
      history: [
        pm("Que penseriez-vous d'un service de paniers-repas livrés, avec un nombre de repas fixe chaque semaine ?"),
        julie("Pourquoi pas, si ça remplace vraiment une partie des courses et que les enfants mangent ce qu'on reçoit."),
        pm("Comment organisez-vous vos courses ?"),
        julie("Un gros passage au supermarché le week-end, puis quelques achats en semaine."),
        pm("Utilisez-vous une application de listes de courses ?"),
        julie("Une liste sur mon téléphone, mais je l'oublie souvent."),
      ],
    },
    expected: {
      status: "ready",
      context_type: "topic_return",
      history_needed_for_query: true,
      required: [["panier"], ["trois", "3"], ["repas"]],
      forbidden: ["application", "liste", "supermarché", "week-end", "téléphone"],
      note: "Porte sur le nombre de repas (trois par semaine) du service de paniers-repas livrés, sans courses, supermarché ni application.",
    },
  },
  {
    name: "reference-ambigue",
    description: "« Cette application » n'a jamais été mentionnée : demande de précision.",
    input: { latest_question: "Et cette application, vous l'utiliseriez ?", history: MEAL_KITS },
    expected: {
      status: "clarification_needed",
      unresolved_expression: "cette application",
      question_to_pm: "De quelle application parlez-vous ?",
      detected_by: ["rule", "both"],
      note: "Aucune application dans l'historique : demande de précision par le code, sans appel.",
    },
  },
  {
    name: "objet-voisin-abonnement",
    description: "« Cet abonnement » jamais nommé ; les paniers-repas rendent un lien plausible mais non établi.",
    input: { latest_question: "Et cet abonnement, vous le garderiez longtemps ?", history: MEAL_KITS },
    expected: {
      status: "clarification_needed",
      unresolved_expression: "cet abonnement",
      question_to_pm: "De quel abonnement parlez-vous ?",
      detected_by: ["rule", "both"],
      note: "Aucun abonnement dans l'historique : demande de précision, sans « abonnement de paniers-repas ».",
    },
  },
  {
    name: "objet-voisin-robot",
    description: "« Ce robot » jamais nommé ; le temps en cuisine rend un robot cuiseur plausible mais non établi.",
    input: {
      latest_question: "Et ce robot, il vous ferait gagner du temps ?",
      history: [pm("Combien de temps cuisinez-vous le soir en semaine ?"), julie("Une demi-heure au maximum, souvent moins.")],
    },
    expected: {
      status: "clarification_needed",
      unresolved_expression: "ce robot",
      question_to_pm: "De quel robot parlez-vous ?",
      detected_by: ["rule", "both"],
      note: "Aucun robot dans l'historique : demande de précision, sans robot cuiseur inventé.",
    },
  },
  {
    name: "reference-explicite",
    description: "Témoin : l'application est nommée dans l'historique, la référence doit être résolue.",
    input: {
      latest_question: "Et cette application, vous l'utiliseriez si elle était gratuite ?",
      history: [pm("Avez-vous déjà utilisé une application anti-gaspillage ?"), julie("Non, je n'en ai jamais essayé.")],
    },
    expected: {
      status: "ready",
      context_type: "follow_up",
      history_needed_for_query: true,
      note: "Mentionne l'application anti-gaspillage et conserve la condition « gratuite ».",
    },
  },
  {
    name: "sujet-hors-fenetre",
    description: "Retour au premier sujet (paniers-repas), sorti de la fenêtre de cinq échanges.",
    input: { latest_question: "Revenons au tout premier sujet : qu'en pensez-vous finalement ?", history: LONG_HISTORY },
    expected: {
      status: "clarification_needed",
      unresolved_expression: "tout premier sujet",
      question_to_pm: "De quel sujet parlez-vous ?",
      detected_by: ["rule", "both"],
      note: "Le premier sujet est sorti de la fenêtre : demande de précision plutôt que question recopiée.",
    },
  },
  {
    name: "reference-decrite",
    description: "« Cette application de recettes qui… » est décrite dans la question : pas de demande de précision.",
    input: { latest_question: "Et cette application de recettes qui permet de cuisiner les restes, vous l'utiliseriez ?", history: MEAL_KITS },
    expected: {
      status: "ready",
      context_type: "new_topic",
      history_needed_for_query: false,
      query_unchanged: true,
      forbidden: ["panier"],
      note: "Question recopiée telle quelle : l'objet est décrit, sans lien inventé avec les paniers-repas.",
    },
  },
  {
    name: "synonyme-appli",
    description: "« Cette application » désigne l'« appli » nommée juste avant.",
    input: {
      latest_question: "Et cette application, vous la paieriez ?",
      history: [pm("Avez-vous une appli pour vos listes de courses ?"), julie("Oui, une liste sur mon téléphone.")],
    },
    expected: {
      status: "ready",
      context_type: "follow_up",
      history_needed_for_query: true,
      required: [["liste"]],
      note: "Porte sur le paiement d'une application de listes de courses.",
    },
  },
  {
    name: "expression-temporelle",
    description: "« Cette semaine » désigne le présent, pas un objet antérieur.",
    input: { latest_question: "Et cette semaine, qui a fait les courses chez vous ?", history: MEAL_KITS },
    expected: {
      status: "ready",
      context_type: "new_topic",
      history_needed_for_query: false,
      query_unchanged: true,
      forbidden: ["panier"],
      note: "Question recopiée telle quelle.",
    },
  },
  {
    name: "premier-tour-ambigu",
    description: "« Cette application » dès la première question : demande de précision, sans appel.",
    input: { latest_question: "Et cette application, vous l'utiliseriez ?", history: [] },
    expected: {
      status: "clarification_needed",
      unresolved_expression: "cette application",
      question_to_pm: "De quelle application parlez-vous ?",
      detected_by: ["rule"],
      note: "Aucun historique : demande de précision par le code.",
    },
  },
  {
    name: "reponse-clarification",
    description: "Le PM précise l'application après la demande de Julie : la précision rejoint la question initiale.",
    input: {
      latest_question: "Celle qui propose des recettes avec les restes du frigo.",
      history: [...MEAL_KITS, pm("Et cette application, vous l'utiliseriez ?"), julie("De quelle application parlez-vous ?")],
    },
    expected: {
      status: "ready",
      context_type: "follow_up",
      history_needed_for_query: true,
      required: [["application"], ["restes"], ["utilise"]],
      forbidden: ["panier"],
      note: "« Utiliseriez-vous une application qui propose des recettes avec les restes du frigo ? », sans paniers-repas.",
    },
  },
  {
    name: "inference-julie",
    description: "Julie a émis une hypothèse ; la relance ne doit pas la transformer en fait.",
    input: {
      latest_question: "Pourquoi ?",
      history: [
        pm("Testeriez-vous des paniers-repas livrés ?"),
        julie("Je pense que je testerais peut-être une semaine, pour voir si les enfants mangent ce qu'on reçoit."),
      ],
    },
    expected: {
      status: "ready",
      context_type: "follow_up",
      history_needed_for_query: true,
      note: "Reste une hypothèse (« envisageriez-vous peut-être… »), pas « pourquoi testez-vous ».",
    },
  },
];
