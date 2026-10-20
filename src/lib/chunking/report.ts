import type { ChunkingResult } from "./types.ts";

const STATUS_LABELS = { declared: "déclaratif", self_reported: "auto-rapporté", interpretation: "interprétation" };

export function formatReport(result: ChunkingResult): string {
  const out: string[] = [];
  const { chunks, documents, anomalies } = result;
  const tokens = chunks.map((c) => c.approxTokens).sort((a, b) => a - b);
  const total = tokens.reduce((a, b) => a + b, 0);

  out.push(`Persona Lab — chunking « ${result.personaId} » (corpus_version ${result.corpusVersion})`);
  out.push(`Source : ${result.sourceDir}`);
  out.push(`Fichiers ignorés : ${result.ignoredFiles.join(", ") || "aucun"}`);
  out.push("");

  for (const doc of documents) {
    const docChunks = chunks.filter((c) => c.metadata.sourceId === doc.sourceId);
    const caveatLabels = doc.caveats.map((c) => c.label).join(", ");
    out.push(`${doc.sourceId}  ${doc.docTitle}`);
    out.push(`     ${doc.sourceType} · ${STATUS_LABELS[doc.evidenceStatus]} · ${doc.sourceDateStart ?? "date ?"} · ${docChunks.length} chunks · avertissements : ${caveatLabels}`);
    for (const c of docChunks) {
      const lines = `l.${c.metadata.lineStart}-${c.metadata.lineEnd}`;
      const label = c.metadata.question ?? c.metadata.sectionPath ?? "";
      out.push(`     ${c.id.padEnd(10)} ${lines.padEnd(9)} ${String(c.text.length).padStart(5)} car.  ≈${String(c.approxTokens).padStart(3)} tok  ${label}`);
    }
    out.push("");
  }

  const median = tokens.length ? tokens[Math.floor(tokens.length / 2)] : 0;
  out.push(`Total : ${documents.length} documents, ${chunks.length} chunks, ≈${total} tokens à embedder (min ${tokens[0] ?? 0}, médiane ${median}, max ${tokens[tokens.length - 1] ?? 0})`);
  out.push("");

  const errors = anomalies.filter((a) => a.severity === "error");
  out.push(`Anomalies : ${errors.length} erreur(s), ${anomalies.length - errors.length} avertissement(s)`);
  for (const a of anomalies) {
    out.push(`  [${a.severity === "error" ? "ERREUR" : "attention"}] ${a.chunkId ?? a.sourceFile} — ${a.message}`);
  }
  return out.join("\n");
}
