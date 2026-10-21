// Usage :
//   npm run reformulate -- --list
//   npm run reformulate -- --scenario=relance-dependante [--dry-run] [--max-exchanges=5] [--compare-model]
//   npm run reformulate -- --all [--dry-run]
//   npm run reformulate -- --file=conversation.json [--dry-run]   ({ "latest_question": "…", "history": [{ "role": "pm", "content": "…" }, …] })
// Teste la reformulation seule : ni retrieval, ni génération. Un appel Anthropic par scénario, sauf sans historique
// ou quand le contrôle du code demande déjà une précision. --dry-run n'appelle rien et montre ce qui serait envoyé.
// --compare-model appelle quand même le modèle quand le code a détecté une référence non résolue (évaluation, payant).
import { readFileSync } from "node:fs";

import { detectUnresolvedReferences } from "../src/lib/rag/reference-check.ts";
import {
  DEFAULT_MAX_EXCHANGES,
  buildReformulationUserMessage,
  createReformulatorFromEnv,
  selectHistoryWindow,
} from "../src/lib/rag/reformulation.ts";
import type { ReformulationInput } from "../src/lib/rag/reformulation.ts";
import { REFORMULATION_SCENARIOS, evaluateScenario } from "../src/lib/rag/reformulation-scenarios.ts";
import type { ReformulationScenario } from "../src/lib/rag/reformulation-scenarios.ts";

const args = process.argv.slice(2);
const option = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const known = /^--(list|all|dry-run|compare-model|scenario=.+|file=.+|max-exchanges=\d+)$/;
const unknown = args.filter((a) => !known.test(a));
const usage = "Usage : npm run reformulate -- --list | --scenario=<nom> | --all | --file=<conversation.json> [--dry-run] [--max-exchanges=5] [--compare-model]";

function exit(message: string): never {
  console.error(message);
  console.error(usage);
  process.exit(1);
}

if (unknown.length) exit(`Argument(s) inconnu(s) : ${unknown.join(", ")}`);

if (args.includes("--list")) {
  for (const s of REFORMULATION_SCENARIOS) console.log(`${s.name.padEnd(24)} ${s.description}`);
  process.exit(0);
}

type Run = { name: string; input: ReformulationInput; scenario?: ReformulationScenario };
let runs: Run[];
const scenarioName = option("scenario");
const file = option("file");
if (args.includes("--all")) runs = REFORMULATION_SCENARIOS.map((s) => ({ ...s, scenario: s }));
else if (scenarioName) {
  const found = REFORMULATION_SCENARIOS.find((s) => s.name === scenarioName);
  if (!found) exit(`Scénario inconnu : ${scenarioName} (voir --list)`);
  runs = [{ ...found, scenario: found }];
} else if (file) {
  try {
    runs = [{ name: file, input: JSON.parse(readFileSync(file, "utf8")) as ReformulationInput }];
  } catch (error) {
    exit(`Fichier illisible : ${error instanceof Error ? error.message : "erreur inconnue"}`);
  }
} else exit("Indiquez --scenario, --all ou --file.");

const maxExchanges = option("max-exchanges") ? Number(option("max-exchanges")) : DEFAULT_MAX_EXCHANGES;
const dryRun = args.includes("--dry-run");
let reformulator: ReturnType<typeof createReformulatorFromEnv> | null = null;
try {
  if (!dryRun) reformulator = createReformulatorFromEnv(process.env, { maxExchanges, callModelOnRuleDetection: args.includes("--compare-model") });
} catch (error) {
  console.error(`Configuration invalide — ${error instanceof Error ? `${error.name} : ${error.message}` : "erreur inconnue"}`);
  process.exit(1);
}
let failures = 0;

for (const run of runs) {
  console.log(`=== ${run.name}`);
  console.log(`Question : ${run.input.latest_question}`);
  try {
    if (!reformulator) {
      const { latestQuestion, window, excludedMessages } = selectHistoryWindow(run.input, maxExchanges);
      console.log(`Fenêtre : ${window.length} message(s) envoyés · ${excludedMessages} exclu(s)`);
      const [rule] = detectUnresolvedReferences(latestQuestion, window, excludedMessages);
      if (rule) console.log(`Contrôle du code : demande de précision — « ${rule.expression} » → ${rule.question_to_pm} Aucun appel.`);
      else console.log(window.length ? buildReformulationUserMessage(latestQuestion, window, excludedMessages) : "Aucun appel : pas d'historique.");
    } else {
      const result = await reformulator.reformulate(run.input);
      console.log(`Statut : ${result.status}`);
      console.log(JSON.stringify(result.status === "ready" ? result.reformulation : result.clarification, null, 2));
      console.log(`Fenêtre : ${result.window.length} message(s) envoyés · ${result.excludedMessages} exclu(s)`);
      if (result.ruleDetections.length) console.log(`Contrôle du code : ${result.ruleDetections.map((d) => `« ${d.expression} »`).join(", ")}`);
      if (result.modelUnresolved.length) console.log(`Signalé par le modèle : ${result.modelUnresolved.map((e) => `« ${e} »`).join(", ")}`);
      const c = result.call;
      const noCall = result.ruleDetections.length ? "aucun (contrôle du code)" : "aucun (pas d'historique)";
      console.log(c.called ? `Appel : ${c.model} · ${c.inputTokens} tokens en entrée · ${c.outputTokens} en sortie` : `Appel : ${noCall}`);
      if (run.scenario) {
        console.log(`Attendu : ${evaluateScenario(run.scenario, result).map((c) => `${c.label} ${c.ok ? "✓" : "✗"}`).join(" · ")}`);
        console.log(`À vérifier : ${run.scenario.expected.note}`);
      }
    }
  } catch (error) {
    failures++;
    console.error(`Reformulation interrompue — ${error instanceof Error ? `${error.name} : ${error.message}` : "erreur inconnue"}`);
  }
  console.log("");
}
if (failures) process.exit(1);
