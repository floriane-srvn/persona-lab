import assert from "node:assert/strict";
import { inspect } from "node:util";
import path from "node:path";
import { describe, test } from "node:test";

import { chunkPersona } from "../chunking/pipeline.ts";
import { JULIE } from "../chunking/personas.ts";
import { chunkToRow } from "./indexing.ts";
import type { ChunkRow } from "./indexing.ts";
import {
  DEFAULT_TOP_K,
  RetrievalError,
  createRetriever,
  createSupabaseChunkMatcher,
  validateMatchRows,
} from "./retrieval.ts";
import type { ChunkMatcher, MatchRow } from "./retrieval.ts";
import { createSupabaseAdminClient } from "./supabase-store.ts";
import type { VoyageClient, VoyageInputType } from "./voyage.ts";

// Garde-fou : aucun test ne doit atteindre le réseau.
let networkCalls = 0;
globalThis.fetch = (() => {
  networkCalls++;
  throw new Error("Appel réseau interdit dans les tests");
}) as typeof fetch;

const DIMS = 1024;
const ROOT = path.resolve(import.meta.dirname, "../../..");
const julieRows: ChunkRow[] = chunkPersona(JULIE, ROOT).chunks.map((c) => chunkToRow(c));
const julieIds = julieRows.map((r) => r.id);

/** Chunk i gets the unit vector e_i: the cosine with a query is its normalised weight on axis i. */
const axisVector = (axis: number) => Array.from({ length: DIMS }, (_, k) => (k === axis ? 1 : 0));

/** Query vector with chosen weights on chunk axes, e.g. { "D04-q4": 0.9, "D04-q3": 0.5 }. */
function queryVector(weights: Record<string, number>, background = 0.01) {
  const v = Array.from({ length: DIMS }, () => background);
  for (const [id, w] of Object.entries(weights)) v[julieIds.indexOf(id)] = w;
  return v;
}

const cosine = (a: number[], b: number[]) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] ** 2;
    nb += b[i] ** 2;
  }
  return dot / Math.sqrt(na * nb);
};

/** Mimics match_persona_chunks: persona filter, cosine ordering (ties by id) and the 1..50 cap. */
class FakeMatcher implements ChunkMatcher {
  calls: { personaId: string; dims: number; matchCount: number }[] = [];
  rows: (ChunkRow & { embedding: number[] })[] = julieRows.map((r, i) => ({ ...r, embedding: axisVector(i) }));
  override: unknown = undefined;

  async match(personaId: string, queryEmbedding: number[], matchCount: number) {
    this.calls.push({ personaId, dims: queryEmbedding.length, matchCount });
    if (this.override !== undefined) return this.override;
    return this.rows
      .filter((r) => r.persona_id === personaId)
      .map((r) => ({ id: r.id, content: r.content, metadata: r.metadata, similarity: cosine(r.embedding, queryEmbedding) }))
      .sort((a, b) => b.similarity - a.similarity || a.id.localeCompare(b.id))
      .slice(0, Math.min(Math.max(matchCount ?? 5, 1), 50));
  }
}

class FakeEmbedder implements VoyageClient {
  calls: { texts: string[]; inputType: VoyageInputType }[] = [];
  vector: number[];
  constructor(vector: number[] = queryVector({ "D04-q4": 0.9, "D04-q3": 0.6, "D04-q1": 0.3 })) {
    this.vector = vector;
  }
  async embed(texts: string[], inputType: VoyageInputType) {
    this.calls.push({ texts, inputType });
    return { embeddings: [this.vector], totalTokens: 12, requests: 1 };
  }
}

function setup(vector?: number[]) {
  const embedder = new FakeEmbedder(vector);
  const matcher = new FakeMatcher();
  return { embedder, matcher, retriever: createRetriever({ embedder, matcher }) };
}

const QUESTION = "Julie paierait-elle 60 € par semaine pour des paniers-repas ?";

describe("embedding de la question", () => {
  test("un seul appel Voyage, input_type « query », question nettoyée", async () => {
    const { embedder, matcher, retriever } = setup();
    const result = await retriever.retrieve("julie", `  ${QUESTION}  `);
    assert.deepEqual(embedder.calls, [{ texts: [QUESTION], inputType: "query" }]);
    assert.equal(result.question, QUESTION);
    assert.equal(result.queryTokens, 12);
    assert.deepEqual(matcher.calls, [{ personaId: "julie", dims: DIMS, matchCount: DEFAULT_TOP_K }]);
  });

  test("embedding de mauvaise dimension : erreur avant l'appel Supabase", async () => {
    const { matcher, retriever } = setup([0.1, 0.2]);
    await assert.rejects(retriever.retrieve("julie", QUESTION), /embedding de la question invalide/);
    assert.equal(matcher.calls.length, 0);
  });
});

describe("classement, top_k et seuil", () => {
  test("passages classés par similarité décroissante, rangs 1..n", async () => {
    const { retriever } = setup();
    const { passages, candidates, bestSimilarity } = await retriever.retrieve("julie", QUESTION);
    assert.equal(passages.length, 5);
    assert.equal(candidates, 5);
    assert.deepEqual(passages.slice(0, 3).map((p) => p.id), ["D04-q4", "D04-q3", "D04-q1"]);
    assert.deepEqual(passages.map((p) => p.rank), [1, 2, 3, 4, 5]);
    for (let i = 1; i < passages.length; i++) assert.ok(passages[i].similarity <= passages[i - 1].similarity);
    assert.equal(bestSimilarity, passages[0].similarity);
  });

  test("top_k configurable et transmis à match_persona_chunks", async () => {
    const { matcher, retriever } = setup();
    const result = await retriever.retrieve("julie", QUESTION, { topK: 2 });
    assert.equal(result.topK, 2);
    assert.deepEqual(result.passages.map((p) => p.id), ["D04-q4", "D04-q3"]);
    assert.equal(matcher.calls[0].matchCount, 2);
  });

  test("seuil configurable : filtre après classement et compte les candidats écartés", async () => {
    const { retriever } = setup();
    const all = await retriever.retrieve("julie", QUESTION);
    const threshold = (all.passages[1].similarity + all.passages[2].similarity) / 2;
    const result = await retriever.retrieve("julie", QUESTION, { minSimilarity: threshold });
    assert.deepEqual(result.passages.map((p) => p.id), ["D04-q4", "D04-q3"]);
    assert.equal(result.minSimilarity, threshold);
    assert.equal(result.candidates, 5);
    assert.equal(result.belowThreshold, 3);
    assert.ok(result.passages.every((p) => p.similarity >= threshold));
  });

  test("seuil égal au score : passage conservé", async () => {
    const { retriever } = setup();
    const best = (await retriever.retrieve("julie", QUESTION)).passages[0].similarity;
    const result = await retriever.retrieve("julie", QUESTION, { minSimilarity: best });
    assert.deepEqual(result.passages.map((p) => p.id), ["D04-q4"]);
  });

  test("aucun seuil par défaut : rien n'est écarté", async () => {
    const { retriever } = setup();
    const result = await retriever.retrieve("julie", QUESTION, { minSimilarity: null });
    assert.equal(result.minSimilarity, null);
    assert.equal(result.belowThreshold, 0);
    assert.equal(result.passages.length, 5);
  });
});

describe("zéro résultat", () => {
  test("aucun passage au-dessus du seuil : résultat vide, sans erreur", async () => {
    const { retriever } = setup();
    const result = await retriever.retrieve("julie", QUESTION, { minSimilarity: 0.99 });
    assert.deepEqual(result.passages, []);
    assert.equal(result.candidates, 5);
    assert.equal(result.belowThreshold, 5);
    assert.ok(result.bestSimilarity! < 0.99);
  });

  test("persona sans aucun chunk : résultat vide, sans erreur", async () => {
    const { retriever } = setup();
    const result = await retriever.retrieve("inconnu", QUESTION);
    assert.deepEqual(result.passages, []);
    assert.equal(result.candidates, 0);
    assert.equal(result.bestSimilarity, null);
  });
});

describe("filtrage par persona", () => {
  test("un chunk d'un autre persona, même identique à la question, n'est jamais renvoyé", async () => {
    const { embedder, matcher, retriever } = setup();
    matcher.rows.push({
      persona_id: "autre",
      id: "D01-p1",
      content: "autre persona",
      metadata: { ...julieRows[0].metadata, personaId: "autre" },
      embedding: embedder.vector,
    });
    const result = await retriever.retrieve("julie", QUESTION);
    assert.ok(result.passages.every((p) => p.personaId === "julie" && p.metadata.personaId === "julie"));
    assert.ok(!result.passages.some((p) => p.content === "autre persona"));
    assert.equal(matcher.calls[0].personaId, "julie");
  });

  test("défense en profondeur : une ligne d'un autre persona renvoyée par Supabase est rejetée", async () => {
    const { matcher, retriever } = setup();
    matcher.override = [{ id: "D01-p1", content: "x", metadata: { ...julieRows[0].metadata, personaId: "autre" }, similarity: 0.9 }];
    await assert.rejects(retriever.retrieve("julie", QUESTION), /D01-p1 appartient au persona « autre »/);
  });
});

describe("provenance", () => {
  test("texte original, score, identifiant et métadonnées de provenance", async () => {
    const { retriever } = setup();
    const [first] = (await retriever.retrieve("julie", QUESTION)).passages;
    const source = julieRows.find((r) => r.id === "D04-q4")!;

    assert.equal(first.id, "D04-q4");
    assert.equal(first.personaId, "julie");
    assert.equal(first.content, source.content);
    assert.ok(first.similarity > 0 && first.similarity <= 1);
    assert.deepEqual(first.metadata, source.metadata);
    assert.deepEqual(
      {
        sourceId: first.provenance.sourceId,
        sourceFile: first.provenance.sourceFile,
        evidenceStatus: first.provenance.evidenceStatus,
        question: first.provenance.question,
        lines: [first.provenance.lineStart, first.provenance.lineEnd],
        contextChunkIds: first.provenance.contextChunkIds,
        corpusVersion: first.provenance.corpusVersion,
        sourceDateStart: first.provenance.sourceDateStart,
      },
      {
        sourceId: "D04",
        sourceFile: "data/personas/julie/D04_entretien_budget.md",
        evidenceStatus: "declared",
        question: "Seriez-vous prête à payer 60 € par semaine ?",
        lines: [13, 14],
        contextChunkIds: ["D04-q3"],
        corpusVersion: "v1",
        sourceDateStart: "2026-09-29",
      },
    );
    assert.ok(first.provenance.caveats.some((c) => c.label === "Limite" && /pas accepté par Julie/.test(c.text)));
  });

  test("statut épistémique des notes de recherche conservé", async () => {
    const { retriever } = setup(queryVector({ "D06-s2": 1 }));
    const [first] = (await retriever.retrieve("julie", "Quelles hypothèses restent à vérifier ?", { topK: 1 })).passages;
    assert.equal(first.id, "D06-s2");
    assert.equal(first.provenance.evidenceStatus, "interpretation");
    assert.equal(first.provenance.containsVerbatim, false);
    assert.equal(first.provenance.sectionPath, "Hypothèses à vérifier, non établies");
  });
});

describe("validation des entrées (aucun appel si invalide)", () => {
  const cases: [string, string, string, object, RegExp][] = [
    ["persona vide", "", QUESTION, {}, /identifiant de persona invalide/],
    ["persona avec majuscules", "Julie", QUESTION, {}, /identifiant de persona invalide/],
    ["persona avec injection", "julie' or 1=1--", QUESTION, {}, /identifiant de persona invalide/],
    ["question vide", "julie", "   ", {}, /question vide/],
    ["question trop longue", "julie", "x".repeat(2001), {}, /question trop longue/],
    ["top_k = 0", "julie", QUESTION, { topK: 0 }, /top_k doit être un entier entre 1 et 50/],
    ["top_k = 51", "julie", QUESTION, { topK: 51 }, /top_k/],
    ["top_k décimal", "julie", QUESTION, { topK: 2.5 }, /top_k/],
    ["seuil > 1", "julie", QUESTION, { minSimilarity: 1.5 }, /seuil de similarité/],
    ["seuil NaN", "julie", QUESTION, { minSimilarity: Number.NaN }, /seuil de similarité/],
  ];
  for (const [name, persona, question, options, pattern] of cases) {
    test(name, async () => {
      const { embedder, matcher, retriever } = setup();
      await assert.rejects(retriever.retrieve(persona, question, options), (e: unknown) => {
        assert.ok(e instanceof RetrievalError);
        assert.match(e.message, pattern);
        return true;
      });
      assert.equal(embedder.calls.length, 0, "aucun appel Voyage");
      assert.equal(matcher.calls.length, 0, "aucun appel Supabase");
    });
  }
});

describe("validation des réponses Supabase", () => {
  const meta = julieRows[0].metadata;
  const row = (overrides: Partial<MatchRow> & Record<string, unknown> = {}) => ({ id: "D01-p1", content: "texte", metadata: meta, similarity: 0.5, ...overrides });
  const cases: [string, unknown, RegExp][] = [
    ["pas un tableau", { data: [] }, /tableau de lignes était attendu/],
    ["trop de lignes", Array.from({ length: 6 }, (_, i) => row({ id: `X${i}` })), /6 lignes reçues pour top_k = 5/],
    ["ligne non objet", [42], /ligne 0 n'est pas un objet/],
    ["identifiant absent", [row({ id: "" })], /sans identifiant/],
    ["identifiant dupliqué", [row(), row()], /D01-p1 dupliqué/],
    ["contenu vide", [row({ content: " " })], /sans contenu/],
    ["métadonnées absentes", [row({ metadata: null as unknown as Record<string, unknown> })], /sans métadonnées/],
    ["score texte", [row({ similarity: "0.5" as unknown as number })], /score non numérique/],
    ["score hors bornes", [row({ similarity: 1.5 })], /hors de \[-1, 1\]/],
    ["autre modèle d'embedding", [row({ metadata: { ...meta, embeddingModel: "voyage-4-lite" } })], /incompatible avec la question/],
    ["autres dimensions", [row({ metadata: { ...meta, embeddingDimensions: 1536 } })], /incompatible avec la question/],
    ["sans sourceId", [row({ metadata: { ...meta, sourceId: undefined } })], /sans sourceId/],
    ["non triées", [row({ id: "A", similarity: 0.2 }), row({ id: "B", similarity: 0.8 })], /non triées/],
  ];
  for (const [name, data, pattern] of cases) {
    test(name, () => assert.throws(() => validateMatchRows(data, "julie", 5), pattern));
  }

  test("liste vide acceptée", () => assert.deepEqual(validateMatchRows([], "julie", 5), []));
});

describe("appel Supabase réel simulé (faux fetch)", () => {
  const ENV = { SUPABASE_URL: "https://abcdefghijklmnopqrst.supabase.co", SUPABASE_SECRET_KEY: "sb_secret_TEST-ONLY-not-a-real-key" };

  function fake(response: Response) {
    const calls: { method: string; url: URL; body: Record<string, unknown> }[] = [];
    const impl = (async (input: string | URL | Request, init: RequestInit = {}) => {
      calls.push({ method: init.method ?? "GET", url: new URL(String(input)), body: JSON.parse(String(init.body)) });
      return response;
    }) as typeof fetch;
    return { calls, matcher: createSupabaseChunkMatcher(createSupabaseAdminClient(ENV, { fetch: impl })) };
  }

  test("RPC match_persona_chunks avec exactement les arguments de la signature SQL", async () => {
    const { calls, matcher } = fake(new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } }));
    const embedding = Array.from({ length: DIMS }, () => 0.1);
    assert.deepEqual(await matcher.match("julie", embedding, 3), []);

    const [call] = calls;
    assert.equal(call.method, "POST");
    assert.equal(call.url.pathname, "/rest/v1/rpc/match_persona_chunks");
    assert.deepEqual(Object.keys(call.body).sort(), ["p_match_count", "p_persona_id", "p_query_embedding"]);
    assert.equal(call.body.p_persona_id, "julie");
    assert.equal(call.body.p_match_count, 3);
    assert.deepEqual(call.body.p_query_embedding, embedding);
  });

  test("erreur Supabase : RetrievalError claire, sans la clé", async () => {
    const response = new Response(JSON.stringify({ message: "permission denied for function match_persona_chunks", code: "42501" }), {
      status: 403,
      headers: { "Content-Type": "application/json" },
    });
    const { matcher } = fake(response);
    await assert.rejects(matcher.match("julie", [0.1], 5), (e: unknown) => {
      assert.ok(e instanceof RetrievalError);
      assert.match(e.message, /permission denied for function match_persona_chunks \[42501\]/);
      assert.ok(!inspect(e, { depth: 10, showHidden: true }).includes(ENV.SUPABASE_SECRET_KEY));
      return true;
    });
  });
});

test("aucun appel réseau réel pendant ces tests", () => {
  assert.equal(networkCalls, 0);
});
