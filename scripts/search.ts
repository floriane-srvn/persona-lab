// Usage : npm run search -- "question" [--persona=julie] [--top-k=5] [--min-similarity=0.4]
// Un seul appel Voyage (embedding de la question) et un appel Supabase en lecture.
import { createRetrieverFromEnv } from "../src/lib/rag/retrieval.ts";

const args = process.argv.slice(2);
const option = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];
const known = /^--(persona=.+|top-k=\d+|min-similarity=-?[\d.]+)$/;
const unknown = args.filter((a) => a.startsWith("--") && !known.test(a));
const question = args.filter((a) => !a.startsWith("--")).join(" ");

if (unknown.length || !question) {
  console.error(unknown.length ? `Option(s) inconnue(s) : ${unknown.join(", ")}` : "Question manquante.");
  console.error('Usage : npm run search -- "question" [--persona=julie] [--top-k=5] [--min-similarity=0.4]');
  process.exit(1);
}

try {
  const minSimilarity = option("min-similarity");
  const result = await createRetrieverFromEnv().retrieve(option("persona") ?? "julie", question, {
    topK: option("top-k") ? Number(option("top-k")) : undefined,
    minSimilarity: minSimilarity === undefined ? null : Number(minSimilarity),
  });

  console.log(`Question : ${result.question}`);
  console.log(`Persona : ${result.personaId} · top_k ${result.topK} · seuil ${result.minSimilarity ?? "aucun"} · ${result.queryTokens} token(s) Voyage`);
  console.log(`Candidats : ${result.candidates} · sous le seuil : ${result.belowThreshold} · meilleur score : ${result.bestSimilarity?.toFixed(4) ?? "—"}`);
  console.log("");
  if (result.passages.length === 0) console.log("Aucun passage retenu.");
  for (const p of result.passages) {
    const { provenance: s } = p;
    console.log(`#${p.rank}  ${p.id}  ${p.similarity.toFixed(4)}  ${s.docTitle} · ${s.evidenceStatus} · l.${s.lineStart}-${s.lineEnd}`);
    console.log(`    ${p.content.replace(/\s+/g, " ").slice(0, 160)}${p.content.length > 160 ? "…" : ""}`);
  }
} catch (error) {
  console.error(`Recherche interrompue — ${error instanceof Error ? `${error.name} : ${error.message}` : "erreur inconnue"}`);
  process.exit(1);
}
