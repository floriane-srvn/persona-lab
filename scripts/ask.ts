// Usage : npm run ask -- "question" [--persona=julie] [--top-k=5] [--min-similarity=0.4] [--debug]
// Un appel Voyage (embedding de la question), un appel Supabase en lecture et, si des passages
// sont retenus, un appel Anthropic. --debug détaille ensuite les passages déjà récupérés, sans appel supplémentaire.
import { formatDebugReport } from "../src/lib/rag/debug.ts";
import { createPersonaResponderFromEnv } from "../src/lib/rag/generation.ts";

const args = process.argv.slice(2);
const option = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];
const known = /^--(persona=.+|top-k=\d+|min-similarity=-?[\d.]+|debug)$/;
const unknown = args.filter((a) => a.startsWith("--") && !known.test(a));
const question = args.filter((a) => !a.startsWith("--")).join(" ");

if (unknown.length || !question) {
  console.error(unknown.length ? `Option(s) inconnue(s) : ${unknown.join(", ")}` : "Question manquante.");
  console.error('Usage : npm run ask -- "question" [--persona=julie] [--top-k=5] [--min-similarity=0.4] [--debug]');
  process.exit(1);
}

try {
  const minSimilarity = option("min-similarity");
  const reply = await createPersonaResponderFromEnv().respond(option("persona") ?? "julie", question, {
    topK: option("top-k") ? Number(option("top-k")) : undefined,
    minSimilarity: minSimilarity === undefined ? null : Number(minSimilarity),
  });

  console.log(JSON.stringify(reply.answer, null, 2));
  console.log("");
  const r = reply.retrieval;
  console.log(`Retrieval : top_k ${r.topK} · seuil ${r.minSimilarity ?? "aucun"} · ${r.passages.length}/${r.candidates} passage(s) · ${r.queryTokens} token(s) Voyage`);
  const g = reply.generation;
  console.log(g.called ? `Génération : ${g.model} · ${g.inputTokens} tokens en entrée · ${g.outputTokens} en sortie` : "Génération : aucun appel Anthropic");
  for (const p of reply.citedPassages) {
    console.log(`  ${p.id}  ${p.similarity.toFixed(4)}  ${p.provenance.docTitle} · ${p.provenance.evidenceStatus} · l.${p.provenance.lineStart}-${p.provenance.lineEnd}`);
  }
  if (args.includes("--debug")) console.log(`\n${formatDebugReport(reply)}`);
} catch (error) {
  console.error(`Réponse interrompue — ${error instanceof Error ? `${error.name} : ${error.message}` : "erreur inconnue"}`);
  process.exit(1);
}
