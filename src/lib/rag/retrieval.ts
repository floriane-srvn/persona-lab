import type { SupabaseClient } from "@supabase/supabase-js";

import type { Caveat, EvidenceStatus, SourceType } from "../chunking/types.ts";
import { createSupabaseAdminClient } from "./supabase-store.ts";
import { VOYAGE_DIMENSIONS, VOYAGE_MODEL, createVoyageClientFromEnv } from "./voyage.ts";
import type { VoyageClient } from "./voyage.ts";

export const MATCH_FUNCTION = "match_persona_chunks";
export const DEFAULT_TOP_K = 5;
/** Same cap as `match_persona_chunks` (migration 0001). */
export const MAX_TOP_K = 50;
export const MAX_QUESTION_CHARS = 2000;
/** Same format as the `persona_chunks_persona_id_format` constraint. */
const PERSONA_ID_RE = /^[a-z0-9_-]+$/;
const SIMILARITY_EPSILON = 1e-6;

export interface RetrievalOptions {
  topK?: number;
  /**
   * Minimum cosine similarity, between -1 and 1. No default on purpose: the threshold must be
   * calibrated on real questions. Undefined or null keeps every candidate.
   */
  minSimilarity?: number | null;
}

export interface PassageProvenance {
  sourceId: string;
  sourceFile: string;
  docTitle: string;
  sourceType: SourceType;
  sourceDateStart: string | null;
  sourceDateEnd: string | null;
  evidenceStatus: EvidenceStatus;
  statusLabel: string;
  containsVerbatim: boolean;
  sectionPath: string | null;
  question: string | null;
  lineStart: number;
  lineEnd: number;
  citedSources: string[];
  contextChunkIds: string[];
  caveats: Caveat[];
  corpusVersion: string;
}

export interface RetrievedPassage {
  id: string;
  personaId: string;
  /** Original chunk text (`persona_chunks.content`). */
  content: string;
  similarity: number;
  /** 1 = most similar. */
  rank: number;
  provenance: PassageProvenance;
  /** Full stored metadata, including embeddingText and embedding parameters. */
  metadata: Record<string, unknown>;
}

export interface RetrievalResult {
  personaId: string;
  question: string;
  topK: number;
  minSimilarity: number | null;
  passages: RetrievedPassage[];
  /** Rows returned by Supabase before the threshold, useful to calibrate it. */
  candidates: number;
  belowThreshold: number;
  bestSimilarity: number | null;
  queryTokens: number;
}

/** Raw row returned by `match_persona_chunks`. */
export interface MatchRow {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
}

export interface ChunkMatcher {
  match(personaId: string, queryEmbedding: number[], matchCount: number): Promise<unknown>;
}

export interface Retriever {
  retrieve(personaId: string, question: string, options?: RetrievalOptions): Promise<RetrievalResult>;
}

export class RetrievalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RetrievalError";
  }
}

/** Calls the SQL function with its exact argument names. */
export function createSupabaseChunkMatcher(client: SupabaseClient): ChunkMatcher {
  return {
    async match(personaId, queryEmbedding, matchCount) {
      const { data, error } = await client.rpc(MATCH_FUNCTION, {
        p_persona_id: personaId,
        p_query_embedding: queryEmbedding,
        p_match_count: matchCount,
      });
      if (error) {
        throw new RetrievalError(`Supabase (${MATCH_FUNCTION}) : ${error.message}${error.code ? ` [${error.code}]` : ""}`);
      }
      return data;
    },
  };
}

function validateInput(personaId: unknown, question: unknown, options: RetrievalOptions) {
  if (typeof personaId !== "string" || !PERSONA_ID_RE.test(personaId)) {
    throw new RetrievalError("identifiant de persona invalide (minuscules, chiffres, « _ » ou « - »)");
  }
  if (typeof question !== "string" || question.trim() === "") throw new RetrievalError("question vide");
  const trimmed = question.trim();
  if (trimmed.length > MAX_QUESTION_CHARS) {
    throw new RetrievalError(`question trop longue (${trimmed.length} caractères, maximum ${MAX_QUESTION_CHARS})`);
  }
  const topK = options.topK ?? DEFAULT_TOP_K;
  if (!Number.isInteger(topK) || topK < 1 || topK > MAX_TOP_K) {
    throw new RetrievalError(`top_k doit être un entier entre 1 et ${MAX_TOP_K}`);
  }
  const minSimilarity = options.minSimilarity ?? null;
  if (minSimilarity !== null && (typeof minSimilarity !== "number" || !Number.isFinite(minSimilarity) || minSimilarity < -1 || minSimilarity > 1)) {
    throw new RetrievalError("le seuil de similarité doit être un nombre entre -1 et 1");
  }
  return { question: trimmed, topK, minSimilarity };
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** Checks every row: shape, persona, embedding parameters, score range and ordering. */
export function validateMatchRows(data: unknown, personaId: string, topK: number): MatchRow[] {
  const invalid = (message: string) => new RetrievalError(`réponse Supabase invalide : ${message}`);
  if (!Array.isArray(data)) throw invalid("un tableau de lignes était attendu");
  if (data.length > topK) throw invalid(`${data.length} lignes reçues pour top_k = ${topK}`);

  const seen = new Set<string>();
  const rows = data.map((raw, i): MatchRow => {
    if (!isObject(raw)) throw invalid(`ligne ${i} n'est pas un objet`);
    const { id, content, metadata, similarity } = raw;
    if (typeof id !== "string" || id === "") throw invalid(`ligne ${i} sans identifiant`);
    if (seen.has(id)) throw invalid(`identifiant ${id} dupliqué`);
    seen.add(id);
    if (typeof content !== "string" || content.trim() === "") throw invalid(`${id} sans contenu`);
    if (!isObject(metadata)) throw invalid(`${id} sans métadonnées`);
    if (typeof similarity !== "number" || !Number.isFinite(similarity)) throw invalid(`${id} : score non numérique`);
    if (similarity < -1 - SIMILARITY_EPSILON || similarity > 1 + SIMILARITY_EPSILON) {
      throw invalid(`${id} : score ${similarity} hors de [-1, 1]`);
    }
    // Defence in depth: the SQL already filters on persona_id.
    if (metadata.personaId !== personaId) throw invalid(`${id} appartient au persona « ${String(metadata.personaId)} »`);
    if (metadata.embeddingModel !== VOYAGE_MODEL || metadata.embeddingDimensions !== VOYAGE_DIMENSIONS) {
      throw invalid(`${id} indexé avec ${String(metadata.embeddingModel)}/${String(metadata.embeddingDimensions)}, incompatible avec la question`);
    }
    if (typeof metadata.sourceId !== "string") throw invalid(`${id} sans sourceId`);
    return { id, content, metadata, similarity };
  });

  for (let i = 1; i < rows.length; i++) {
    if (rows[i].similarity > rows[i - 1].similarity + SIMILARITY_EPSILON) throw invalid("lignes non triées par similarité");
  }
  return rows;
}

function toPassage(row: MatchRow, personaId: string, rank: number): RetrievedPassage {
  const m = row.metadata;
  const str = (v: unknown) => (typeof v === "string" ? v : null);
  const num = (v: unknown) => (typeof v === "number" ? v : null);
  const strings = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
  return {
    id: row.id,
    personaId,
    content: row.content,
    similarity: row.similarity,
    rank,
    provenance: {
      sourceId: m.sourceId as string,
      sourceFile: str(m.sourceFile) ?? "",
      docTitle: str(m.docTitle) ?? "",
      sourceType: m.sourceType as SourceType,
      sourceDateStart: str(m.sourceDateStart),
      sourceDateEnd: str(m.sourceDateEnd),
      evidenceStatus: m.evidenceStatus as EvidenceStatus,
      statusLabel: str(m.statusLabel) ?? "",
      containsVerbatim: m.containsVerbatim === true,
      sectionPath: str(m.sectionPath),
      question: str(m.question),
      lineStart: num(m.lineStart) ?? 0,
      lineEnd: num(m.lineEnd) ?? 0,
      citedSources: strings(m.citedSources),
      contextChunkIds: strings(m.contextChunkIds),
      caveats: Array.isArray(m.caveats) ? (m.caveats as Caveat[]) : [],
      corpusVersion: str(m.corpusVersion) ?? "",
    },
    metadata: m,
  };
}

export function createRetriever(deps: { embedder: VoyageClient; matcher: ChunkMatcher }): Retriever {
  return {
    async retrieve(personaId, rawQuestion, options = {}) {
      // Validation happens before any paid or network call.
      const { question, topK, minSimilarity } = validateInput(personaId, rawQuestion, options);

      const { embeddings, totalTokens } = await deps.embedder.embed([question], "query");
      const [queryEmbedding] = embeddings;
      if (embeddings.length !== 1 || queryEmbedding?.length !== VOYAGE_DIMENSIONS) {
        throw new RetrievalError("embedding de la question invalide");
      }

      const rows = validateMatchRows(await deps.matcher.match(personaId, queryEmbedding, topK), personaId, topK);
      const kept = minSimilarity === null ? rows : rows.filter((r) => r.similarity >= minSimilarity);
      return {
        personaId,
        question,
        topK,
        minSimilarity,
        passages: kept.map((row, i) => toPassage(row, personaId, i + 1)),
        candidates: rows.length,
        belowThreshold: rows.length - kept.length,
        bestSimilarity: rows[0]?.similarity ?? null,
        queryTokens: totalTokens,
      };
    },
  };
}

/** Server-side only: uses VOYAGE_API_KEY and SUPABASE_SECRET_KEY. Never import from client components. */
export function createRetrieverFromEnv(env: Record<string, string | undefined> = process.env): Retriever {
  return createRetriever({
    embedder: createVoyageClientFromEnv(env),
    matcher: createSupabaseChunkMatcher(createSupabaseAdminClient(env)),
  });
}
