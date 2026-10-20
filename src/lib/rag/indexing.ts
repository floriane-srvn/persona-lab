import { createHash } from "node:crypto";

import type { Chunk, ChunkingResult } from "../chunking/types.ts";
import { VOYAGE_DIMENSIONS, VOYAGE_MODEL, VOYAGE_OUTPUT_DTYPE } from "./voyage.ts";
import type { VoyageClient, VoyageInputType } from "./voyage.ts";

/** Parameters that determine an embedding: changing any of them requires re-embedding. */
export interface EmbeddingConfig {
  model: string;
  dimensions: number;
  inputType: VoyageInputType;
  dtype: string;
}

export const DOCUMENT_EMBEDDING: EmbeddingConfig = {
  model: VOYAGE_MODEL,
  dimensions: VOYAGE_DIMENSIONS,
  inputType: "document",
  dtype: VOYAGE_OUTPUT_DTYPE,
};

/** Default ceiling on estimated tokens before any paid call (the Julie corpus is ≈3,300). */
export const DEFAULT_MAX_ESTIMATED_TOKENS = 10_000;

/** A `persona_chunks` row without its embedding (the vector is never read back). */
export interface ChunkRow {
  persona_id: string;
  id: string;
  content: string;
  metadata: Record<string, unknown>;
}

export interface ChunkRowWithEmbedding extends ChunkRow {
  embedding: number[];
}

/** Storage used by the indexer. No delete: the database grants none (migration 0001). */
export interface ChunkStore {
  listByPersona(personaId: string): Promise<ChunkRow[]>;
  upsert(rows: ChunkRowWithEmbedding[]): Promise<void>;
  /** Updates content and metadata of existing rows, keeping their embedding. */
  updateWithoutEmbedding(rows: ChunkRow[]): Promise<void>;
}

export class IndexingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IndexingError";
  }
}

export function embeddingInputHash(embeddingText: string, config: EmbeddingConfig = DOCUMENT_EMBEDDING): string {
  const payload = JSON.stringify([config.model, config.dimensions, config.inputType, config.dtype, embeddingText]);
  return createHash("sha256").update(payload).digest("hex");
}

export function chunkToRow(chunk: Chunk, config: EmbeddingConfig = DOCUMENT_EMBEDDING): ChunkRow {
  return {
    persona_id: chunk.metadata.personaId,
    id: chunk.id,
    content: chunk.text,
    metadata: {
      ...chunk.metadata,
      embeddingText: chunk.embeddingText,
      embeddingModel: config.model,
      embeddingDimensions: config.dimensions,
      embeddingInputType: config.inputType,
      embeddingDtype: config.dtype,
      embeddingInputHash: embeddingInputHash(chunk.embeddingText, config),
    },
  };
}

/** JSON with sorted object keys: JSONB does not preserve key order, so compare canonically. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export interface IndexPlan {
  create: ChunkRow[];
  reembed: ChunkRow[];
  update: ChunkRow[];
  unchanged: ChunkRow[];
  /** Rows present in the database but no longer produced by the pipeline. Reported, never deleted. */
  obsolete: ChunkRow[];
}

export function planIndexing(desired: ChunkRow[], existing: ChunkRow[], options: { force?: boolean } = {}): IndexPlan {
  const plan: IndexPlan = { create: [], reembed: [], update: [], unchanged: [], obsolete: [] };
  const personaIds = new Set(desired.map((r) => r.persona_id));
  if (personaIds.size > 1) throw new IndexingError("planification limitée à un seul persona à la fois");
  const [personaId] = personaIds;

  const stored = new Map<string, ChunkRow>();
  for (const row of existing) {
    if (personaId && row.persona_id !== personaId) {
      throw new IndexingError(`ligne ${row.persona_id}/${row.id} hors du persona ${personaId}`);
    }
    stored.set(row.id, row);
  }
  const seen = new Set<string>();
  for (const row of desired) {
    if (seen.has(row.id)) throw new IndexingError(`identifiant de chunk dupliqué : ${row.id}`);
    seen.add(row.id);

    const current = stored.get(row.id);
    if (!current) plan.create.push(row);
    else if (options.force || current.metadata.embeddingInputHash !== row.metadata.embeddingInputHash) plan.reembed.push(row);
    else if (current.content !== row.content || canonicalJson(current.metadata) !== canonicalJson(row.metadata)) plan.update.push(row);
    else plan.unchanged.push(row);
  }
  plan.obsolete = existing.filter((row) => !seen.has(row.id));
  return plan;
}

/** Same estimate as the chunking report (~4 characters per token). */
export const estimateTokens = (text: string) => Math.ceil(text.length / 4);

export interface IndexOptions {
  result: ChunkingResult;
  store: ChunkStore;
  /** Created lazily: never instantiated in dry-run or when nothing needs embedding. */
  getEmbedder: () => VoyageClient;
  dryRun?: boolean;
  force?: boolean;
  maxEstimatedTokens?: number;
  /** Chunks embedded then written together, so a later failure does not lose paid embeddings. */
  writeBatchSize?: number;
}

export interface IndexReport {
  personaId: string;
  corpusVersion: string;
  dryRun: boolean;
  force: boolean;
  chunkCount: number;
  plan: Record<keyof IndexPlan, string[]>;
  estimatedTokens: number;
  maxEstimatedTokens: number;
  voyageTokens: number;
  voyageRequests: number;
  upserted: number;
  updated: number;
}

export async function indexPersona(options: IndexOptions): Promise<IndexReport> {
  const { result, store } = options;
  const dryRun = options.dryRun ?? false;
  const force = options.force ?? false;
  const maxEstimatedTokens = options.maxEstimatedTokens ?? DEFAULT_MAX_ESTIMATED_TOKENS;
  const writeBatchSize = options.writeBatchSize ?? 100;

  const errors = result.anomalies.filter((a) => a.severity === "error");
  if (errors.length) {
    throw new IndexingError(`chunking en erreur (${errors.length} anomalie(s) bloquante(s)) : indexation annulée`);
  }
  if (result.chunks.length === 0) throw new IndexingError("aucun chunk produit : indexation annulée");

  const desired = result.chunks.map((chunk) => chunkToRow(chunk));
  const byId = new Map(result.chunks.map((c) => [c.id, c]));
  const plan = planIndexing(desired, await store.listByPersona(result.personaId), { force });
  const toEmbed = [...plan.create, ...plan.reembed];
  const estimatedTokens = toEmbed.reduce((sum, row) => sum + estimateTokens(byId.get(row.id)!.embeddingText), 0);

  const report: IndexReport = {
    personaId: result.personaId,
    corpusVersion: result.corpusVersion,
    dryRun,
    force,
    chunkCount: desired.length,
    plan: {
      create: plan.create.map((r) => r.id),
      reembed: plan.reembed.map((r) => r.id),
      update: plan.update.map((r) => r.id),
      unchanged: plan.unchanged.map((r) => r.id),
      obsolete: plan.obsolete.map((r) => r.id),
    },
    estimatedTokens,
    maxEstimatedTokens,
    voyageTokens: 0,
    voyageRequests: 0,
    upserted: 0,
    updated: 0,
  };
  if (dryRun) return report;

  if (estimatedTokens > maxEstimatedTokens) {
    throw new IndexingError(
      `≈${estimatedTokens} tokens estimés dépassent le plafond de ${maxEstimatedTokens} : aucun appel Voyage effectué`,
    );
  }

  if (toEmbed.length) {
    const embedder = options.getEmbedder();
    for (let i = 0; i < toEmbed.length; i += writeBatchSize) {
      const batch = toEmbed.slice(i, i + writeBatchSize);
      const texts = batch.map((row) => byId.get(row.id)!.embeddingText);
      const { embeddings, totalTokens, requests } = await embedder.embed(texts, DOCUMENT_EMBEDDING.inputType);
      report.voyageTokens += totalTokens;
      report.voyageRequests += requests;
      if (embeddings.length !== batch.length) {
        throw new IndexingError(`${embeddings.length} embeddings reçus pour ${batch.length} chunks`);
      }
      await store.upsert(batch.map((row, j) => ({ ...row, embedding: embeddings[j] })));
      report.upserted += batch.length;
    }
  }

  if (plan.update.length) {
    await store.updateWithoutEmbedding(plan.update);
    report.updated = plan.update.length;
  }
  return report;
}

const LABELS: Record<keyof IndexPlan, string> = {
  create: "À créer",
  reembed: "À ré-embedder",
  update: "Mise à jour sans embedding",
  unchanged: "Inchangés",
  obsolete: "Obsolètes (non supprimés)",
};

export function formatIndexReport(report: IndexReport): string {
  const out: string[] = [];
  const mode = report.dryRun ? "SIMULATION (--dry-run) : aucun appel Voyage, aucune écriture" : "EXÉCUTION";
  out.push(`Persona Lab — indexation « ${report.personaId} » (corpus_version ${report.corpusVersion})`);
  out.push(`Mode : ${mode}${report.force ? " · --force : tous les chunks sont ré-embeddés" : ""}`);
  out.push(`Embeddings : ${DOCUMENT_EMBEDDING.model}, ${DOCUMENT_EMBEDDING.dimensions} dimensions, input_type ${DOCUMENT_EMBEDDING.inputType}`);
  out.push(`Chunks produits par le pipeline : ${report.chunkCount}`);
  out.push("");
  for (const key of Object.keys(LABELS) as (keyof IndexPlan)[]) {
    const ids = report.plan[key];
    out.push(`${LABELS[key].padEnd(28)} ${String(ids.length).padStart(3)}${ids.length && key !== "unchanged" ? `  ${ids.join(", ")}` : ""}`);
  }
  out.push("");
  out.push(`Tokens estimés à embedder : ≈${report.estimatedTokens} (plafond ${report.maxEstimatedTokens})`);
  if (report.dryRun) {
    out.push("Tokens Voyage consommés : 0 (simulation)");
  } else {
    out.push(`Tokens Voyage consommés : ${report.voyageTokens} en ${report.voyageRequests} requête(s)`);
    out.push(`Lignes écrites : ${report.upserted} upsert(s) avec embedding, ${report.updated} mise(s) à jour sans embedding`);
  }
  if (report.plan.obsolete.length) {
    out.push("");
    out.push(`⚠ ${report.plan.obsolete.length} chunk(s) obsolète(s) restent en base et peuvent encore être retrouvés par la recherche.`);
    out.push("  La clé serveur n'a pas le droit DELETE : retrait manuel par un administrateur si nécessaire.");
  }
  return out.join("\n");
}
