import type { ConversationMessage } from "./reformulation.ts";

/**
 * Deterministic guard run before the reformulation model, for two narrow cases:
 * - a bare demonstrative noun phrase (« cette application ») whose noun appears nowhere in the window
 *   nor earlier in the question;
 * - a reference to the start of the conversation (« le tout premier sujet ») when that start is not in the window.
 *
 * Known limits, on purpose: it reads words, not meaning. It ignores pronouns (« ça », « celle-là »), unusual
 * phrasings and synonyms outside a short list, and it cannot tell whether a noun found in the history really
 * designates the same object. It only adds clarifications; it never resolves anything by itself.
 */

export type RuleDetectionKind = "demonstrative" | "conversation_start";

export interface RuleDetection {
  kind: RuleDetectionKind;
  /** Exact text of the question, e.g. "cette application". */
  expression: string;
  question_to_pm: string;
}

const DETERMINERS = new Set(["ce", "cet", "cette", "ces"]);

/** After « ce », these words make it a pronoun (« ce que », « ce serait »), not a determiner. */
const CE_PRONOUN_NEXT = new Set(["que", "qu", "qui", "dont", "quoi", "sont", "serait", "seraient", "sera", "seront", "fut", "soit", "doit", "devrait", "peut", "pourrait", "était", "n"]);

/** Adjectives that may sit between the determiner and the noun (« ce petit robot »). */
const PRE_ADJECTIVES = ["petit", "petite", "grand", "grande", "nouveau", "nouvel", "nouvelle", "fameux", "fameuse", "même", "autre", "dernier", "dernière", "gros", "grosse", "bon", "bonne", "vieux", "vieil", "vieille", "seul", "seule", "beau", "bel", "belle", "joli", "jolie"];

/** « ce genre de service » designates « service ». */
const KIND_NOUNS = ["genre", "type", "sorte", "style"];

/** Time expressions (« cette semaine », « ce soir ») designate the present, not an earlier object. */
const TEMPORAL = ["semaine", "soir", "soirée", "matin", "matinée", "midi", "nuit", "jour", "journée", "mois", "année", "an", "week-end", "weekend", "moment", "instant", "fois", "heure", "période", "époque", "saison", "été", "hiver", "printemps", "automne", "trimestre", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "après-midi", "temps"];

/** Discourse nouns (« ce point », « ce prix-là ») usually refer to the exchange itself: left to the model. */
const ABSTRACT = ["sujet", "point", "question", "idée", "cas", "aspect", "problème", "raison", "chose", "propos", "thème", "histoire", "situation", "prix", "tarif", "montant", "niveau", "rythme", "façon", "manière"];

/** Words that start a complement describing the object inside the question itself. */
const COMPLEMENT_STARTERS = new Set(["de", "du", "des", "qui", "que", "dont", "où", "pour", "avec", "sans", "à", "au", "aux", "sur", "en", "contre", "chez", "entre"]);

/** Short colloquial forms, mapped both ways through `canonical`. */
const SYNONYMS: Record<string, string> = { appli: "application", app: "application", abo: "abonnement", resto: "restaurant", frigo: "refrigerateur" };

const WORD_RE = /[\p{L}\p{N}]+(?:[-'’][\p{L}\p{N}]+)*/gu;
const POINTED_RE = /-(là|ci)$/iu;
const ELISION_RE = /^(?:l|d|j|m|t|s|n|c|qu|jusqu|lorsqu|puisqu)'/;

/** Lowercase, no accents, no elision, no -là/-ci, crude singular. Used on both sides of every comparison. */
export function normalizeWord(word: string): string {
  let w = word.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "").replace(/’/g, "'");
  w = w.replace(ELISION_RE, "").replace(/-(la|ci)$/, "");
  if (w.length > 3 && /[sx]$/.test(w)) w = w.slice(0, -1);
  return w;
}

const canonical = (word: string) => {
  const n = normalizeWord(word);
  return SYNONYMS[n] ?? n;
};
const normalizedSet = (words: string[]) => new Set(words.map(canonical));
const PRE_ADJECTIVE_SET = normalizedSet(PRE_ADJECTIVES);
const KIND_SET = normalizedSet(KIND_NOUNS);
const TEMPORAL_SET = normalizedSet(TEMPORAL);
const ABSTRACT_SET = normalizedSet(ABSTRACT);

interface Word {
  text: string;
  start: number;
  end: number;
}

const words = (text: string): Word[] => [...text.matchAll(WORD_RE)].map((m) => ({ text: m[0], start: m.index, end: m.index + m[0].length }));

/** Every word of a text, normalised, with the parts of hyphenated compounds (« paniers-repas » → paniers, repas). */
function vocabulary(text: string): Set<string> {
  const vocab = new Set<string>();
  for (const { text: w } of words(text)) {
    vocab.add(canonical(w));
    if (w.includes("-")) for (const part of w.split("-")) vocab.add(canonical(part));
  }
  return vocab;
}

/** Template questions: no model call, nothing said about Julie's opinions. */
export function buildClarificationQuestion(expression: string): string {
  const expr = expression.trim();
  const demonstrative = /^(ce|cet|cette)\s+(.+)$/iu.exec(expr);
  if (demonstrative) {
    const which = demonstrative[1].toLowerCase() === "cette" ? "quelle" : "quel";
    return `De ${which} ${demonstrative[2].replace(POINTED_RE, "")} parlez-vous ?`;
  }
  if (CONVERSATION_START_RE.test(expr)) return "De quel sujet parlez-vous ?";
  return `Pouvez-vous préciser ce que vous entendez par « ${expr} » ?`;
}

const CONVERSATION_START_RE =
  /(?<![\p{L}])(?:(?:tout\s+)?premi(?:er|ère)\s+(?:sujet|question|point|thème|échange|discussion)|(?:tout\s+)?début\s+de\s+(?:la|notre)\s+(?:conversation|discussion|entretien)|au\s+(?:tout\s+)?début(?=\s*(?:[,.:;!?)]|$)))/iu;

function demonstrativeDetections(question: string, window: ConversationMessage[]): RuleDetection[] {
  const ws = words(question);
  const onlySpaceBetween = (a: Word, b: Word) => question.slice(a.end, b.start).trim() === "";
  const detections: RuleDetection[] = [];

  for (let i = 0; i < ws.length; i++) {
    const determiner = ws[i].text.toLowerCase();
    if (!DETERMINERS.has(determiner)) continue;

    let j = i + 1;
    while (j < ws.length && onlySpaceBetween(ws[j - 1], ws[j]) && PRE_ADJECTIVE_SET.has(canonical(ws[j].text))) j++;
    if (j >= ws.length || !onlySpaceBetween(ws[j - 1], ws[j])) continue;
    if (determiner === "ce" && CE_PRONOUN_NEXT.has(ws[j].text.toLowerCase().split(/[-'’]/)[0])) continue;

    // « ce genre de service », « ce type d'application »: the noun is after « de ».
    let nounIndex = j;
    let noun = ws[j].text;
    const next = ws[j + 1];
    if (KIND_SET.has(canonical(noun)) && next && onlySpaceBetween(ws[j], next)) {
      if (next.text.toLowerCase() === "de" && ws[j + 2]) {
        nounIndex = j + 2;
        noun = ws[j + 2].text;
      } else if (/^d['’]/i.test(next.text)) {
        nounIndex = j + 1;
        noun = next.text.slice(2);
      }
    }

    const head = canonical(noun);
    if (TEMPORAL_SET.has(head) || ABSTRACT_SET.has(head)) continue;

    // A complement right after the noun describes the object within the question itself.
    const after = ws[nounIndex + 1];
    const pointed = POINTED_RE.test(noun);
    const described =
      !pointed && after && onlySpaceBetween(ws[nounIndex], after) && (COMPLEMENT_STARTERS.has(after.text.toLowerCase()) || /^(?:d|qu)['’]/i.test(after.text));
    if (described) continue;

    const antecedents = vocabulary([...window.map((m) => m.content), question.slice(0, ws[i].start)].join("\n"));
    if (antecedents.has(head)) continue;

    const expression = question.slice(ws[i].start, ws[nounIndex].end);
    detections.push({ kind: "demonstrative", expression, question_to_pm: buildClarificationQuestion(expression) });
  }
  return detections;
}

/**
 * Returns the references the code can tell are unresolvable, in question order. Empty does NOT mean
 * every reference is resolved: the model still checks the rest.
 */
export function detectUnresolvedReferences(question: string, window: ConversationMessage[], excludedMessages: number): RuleDetection[] {
  const detections = demonstrativeDetections(question, window);
  // The start of the conversation is unknown when nothing was exchanged or older messages were left out.
  if (excludedMessages > 0 || window.length === 0) {
    const start = CONVERSATION_START_RE.exec(question);
    if (start) detections.push({ kind: "conversation_start", expression: start[0], question_to_pm: buildClarificationQuestion(start[0]) });
  }
  return detections;
}
