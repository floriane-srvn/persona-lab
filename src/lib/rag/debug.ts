import type { PersonaReply } from "./generation.ts";

export const DEBUG_EXCERPT_CHARS = 160;

const excerpt = (text: string) => {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > DEBUG_EXCERPT_CHARS ? `${flat.slice(0, DEBUG_EXCERPT_CHARS)}…` : flat;
};

/**
 * Terminal-only diagnostic of every passage sent to the model, cited or not.
 * Built from the reply already computed: no extra call. Never part of the answer contract.
 */
export function formatDebugReport(reply: PersonaReply): string {
  const r = reply.retrieval;
  const cited = new Set(reply.answer.grounding.source_ids);
  const lines = [
    "── Diagnostic retrieval (--debug) ──",
    `top_k ${r.topK} · seuil ${r.minSimilarity ?? "aucun"} · ${r.candidates} candidat(s) · ${r.belowThreshold} sous le seuil (non détaillés) · ${r.passages.length} passage(s) transmis · ${cited.size} cité(s)`,
  ];
  if (!reply.generation.called) lines.push("Aucun appel Anthropic : aucun passage transmis au modèle.");
  for (const p of r.passages) {
    const s = p.provenance;
    const where = s.lineStart > 0 ? `l.${s.lineStart}-${s.lineEnd}` : "lignes inconnues";
    const source = [s.sourceFile || s.sourceId, s.docTitle].filter(Boolean).join(" · ");
    lines.push(
      "",
      `#${p.rank}  ${p.id}  similarité ${p.similarity.toFixed(4)}  ${cited.has(p.id) ? "[cité]" : "[non cité]"}`,
      `    source : ${source} · ${where} · ${s.evidenceStatus}`,
      `    extrait : ${excerpt(p.content)}`,
    );
  }
  return lines.join("\n");
}
