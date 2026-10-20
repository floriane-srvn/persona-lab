import assert from "node:assert/strict";
import path from "node:path";
import { describe, test } from "node:test";

import { chunkPersona } from "../chunking/pipeline.ts";
import { JULIE } from "../chunking/personas.ts";
import type { ChunkingResult } from "../chunking/types.ts";
import {
  DOCUMENT_EMBEDDING,
  IndexingError,
  canonicalJson,
  chunkToRow,
  embeddingInputHash,
  formatIndexReport,
  indexPersona,
} from "./indexing.ts";
import type { ChunkRow, ChunkRowWithEmbedding, ChunkStore, IndexOptions } from "./indexing.ts";
import type { VoyageClient, VoyageInputType } from "./voyage.ts";

// Garde-fou : aucun test ne doit atteindre le réseau.
globalThis.fetch = (() => {
  throw new Error("Appel réseau interdit dans les tests");
}) as typeof fetch;

const ROOT = path.resolve(import.meta.dirname, "../../..");
const freshResult = (): ChunkingResult => structuredClone(chunkPersona(JULIE, ROOT));

/** In-memory `persona_chunks`, mimicking JSONB (key order not preserved) and the NOT NULL embedding. */
class MemoryStore implements ChunkStore {
  rows = new Map<string, ChunkRowWithEmbedding>();
  lists = 0;
  upserts: string[][] = [];
  updates: string[][] = [];

  private key = (personaId: string, id: string) => `${personaId}/${id}`;
  private jsonb = (value: Record<string, unknown>) => {
    // Reverse key order to prove that comparisons do not depend on it.
    const reordered = JSON.parse(canonicalJson(value)) as Record<string, unknown>;
    return Object.fromEntries(Object.entries(reordered).reverse());
  };

  async listByPersona(personaId: string): Promise<ChunkRow[]> {
    this.lists++;
    return [...this.rows.values()]
      .filter((r) => r.persona_id === personaId)
      .map((r) => structuredClone({ persona_id: r.persona_id, id: r.id, content: r.content, metadata: r.metadata }));
  }

  async upsert(rows: ChunkRowWithEmbedding[]) {
    for (const row of rows) {
      assert.equal(row.embedding?.length, 1024, "embedding NOT NULL de 1024 dimensions");
      this.rows.set(this.key(row.persona_id, row.id), { ...structuredClone(row), metadata: this.jsonb(row.metadata) });
    }
    this.upserts.push(rows.map((r) => r.id));
  }

  async updateWithoutEmbedding(rows: ChunkRow[]) {
    for (const row of rows) {
      const current = this.rows.get(this.key(row.persona_id, row.id));
      assert.ok(current, `ligne ${row.id} introuvable`);
      this.rows.set(this.key(row.persona_id, row.id), { ...current, content: row.content, metadata: this.jsonb(row.metadata) });
    }
    this.updates.push(rows.map((r) => r.id));
  }

  get writes() {
    return this.upserts.length + this.updates.length;
  }
}

/** Fake Voyage client: deterministic vectors, 1 token per 4 characters, every call recorded. */
class FakeEmbedder implements VoyageClient {
  calls: { texts: string[]; inputType: VoyageInputType }[] = [];
  failOnCall: number | null = null;

  async embed(texts: string[], inputType: VoyageInputType) {
    this.calls.push({ texts, inputType });
    if (this.failOnCall === this.calls.length) throw new Error("panne simulée");
    return {
      embeddings: texts.map((t) => Array.from({ length: 1024 }, (_, i) => ((t.length + i) % 97) / 97)),
      totalTokens: texts.reduce((sum, t) => sum + Math.ceil(t.length / 4), 0),
      requests: 1,
    };
  }

  get tokens() {
    return this.calls.flatMap((c) => c.texts).reduce((sum, t) => sum + Math.ceil(t.length / 4), 0);
  }
}

function setup() {
  const store = new MemoryStore();
  const embedder = new FakeEmbedder();
  let embedderCreated = 0;
  const run = (options: Partial<IndexOptions> = {}) =>
    indexPersona({
      result: freshResult(),
      store,
      getEmbedder: () => {
        embedderCreated++;
        return embedder;
      },
      ...options,
    });
  return { store, embedder, run, embedderCreated: () => embedderCreated };
}

describe("empreinte et lignes", () => {
  test("SHA-256 de embeddingText, du modèle, des dimensions et de input_type", () => {
    const hash = embeddingInputHash("texte");
    assert.match(hash, /^[0-9a-f]{64}$/);
    assert.equal(embeddingInputHash("texte"), hash, "déterministe");
    assert.notEqual(embeddingInputHash("texte modifié"), hash);
    assert.notEqual(embeddingInputHash("texte", { ...DOCUMENT_EMBEDDING, model: "voyage-4-lite" }), hash);
    assert.notEqual(embeddingInputHash("texte", { ...DOCUMENT_EMBEDDING, dimensions: 512 }), hash);
    assert.notEqual(embeddingInputHash("texte", { ...DOCUMENT_EMBEDDING, inputType: "query" }), hash);
  });

  test("une ligne conserve le texte original, embeddingText et toutes les métadonnées du chunk", () => {
    const chunk = freshResult().chunks.find((c) => c.id === "D04-q4")!;
    const row = chunkToRow(chunk);
    assert.equal(row.persona_id, "julie");
    assert.equal(row.id, "D04-q4");
    assert.equal(row.content, chunk.text);
    for (const [key, value] of Object.entries(chunk.metadata)) assert.deepEqual(row.metadata[key], value, key);
    assert.equal(row.metadata.embeddingText, chunk.embeddingText);
    assert.equal(row.metadata.embeddingModel, "voyage-4");
    assert.equal(row.metadata.embeddingDimensions, 1024);
    assert.equal(row.metadata.embeddingInputType, "document");
    assert.equal(row.metadata.embeddingInputHash, embeddingInputHash(chunk.embeddingText));
  });
});

describe("indexation de Julie", () => {
  test("premier passage : 29 chunks créés en un seul appel Voyage « document »", async () => {
    const { store, embedder, run } = setup();
    const report = await run();
    assert.equal(report.chunkCount, 29);
    assert.equal(report.plan.create.length, 29);
    assert.equal(embedder.calls.length, 1);
    assert.equal(embedder.calls[0].inputType, "document");
    assert.equal(embedder.calls[0].texts.length, 29);
    assert.equal(report.voyageTokens, embedder.tokens);
    assert.ok(report.voyageTokens > 0);
    assert.equal(report.upserted, 29);
    assert.equal(store.rows.size, 29);
    const stored = store.rows.get("julie/D04-q4")!;
    assert.match(stored.metadata.embeddingText as string, /^\[D04 · .*\nContexte : .*paniers-repas livrés/);
    assert.doesNotMatch(stored.content, /Contexte :/, "content reste le texte original");
  });

  test("deuxième passage sans changement : aucun token, aucun appel, aucune écriture", async () => {
    const { store, embedder, run, embedderCreated } = setup();
    await run();
    const writesAfterFirst = store.writes;
    const report = await run();
    assert.equal(report.plan.unchanged.length, 29);
    assert.deepEqual([report.plan.create, report.plan.reembed, report.plan.update, report.plan.obsolete], [[], [], [], []]);
    assert.equal(report.voyageTokens, 0);
    assert.equal(report.voyageRequests, 0);
    assert.equal(embedder.calls.length, 1, "seul le premier passage a appelé Voyage");
    assert.equal(embedderCreated(), 1, "le client Voyage n'est même pas instancié au second passage");
    assert.equal(store.writes, writesAfterFirst);
    assert.match(formatIndexReport(report), /Tokens Voyage consommés : 0 en 0 requête/);
  });

  test("métadonnées modifiées sans changement d'embeddingText : mise à jour sans embedding", async () => {
    const { store, embedder, run } = setup();
    await run();
    const before = store.rows.get("julie/D04-q4")!.embedding;

    const result = freshResult();
    const chunk = result.chunks.find((c) => c.id === "D04-q4")!;
    chunk.metadata.caveats = [...chunk.metadata.caveats, { label: "Test", text: "nouvel avertissement", lineStart: 1, lineEnd: 1 }];
    const report = await run({ result });

    assert.deepEqual(report.plan.update, ["D04-q4"]);
    assert.equal(report.plan.unchanged.length, 28);
    assert.equal(report.voyageTokens, 0);
    assert.equal(embedder.calls.length, 1);
    assert.deepEqual(store.updates.at(-1), ["D04-q4"]);
    assert.deepEqual(store.rows.get("julie/D04-q4")!.embedding, before, "embedding conservé");
    assert.equal((store.rows.get("julie/D04-q4")!.metadata.caveats as unknown[]).length, 3);
  });

  test("embeddingText modifié : seul ce chunk est ré-embeddé", async () => {
    const { embedder, run } = setup();
    await run();
    const result = freshResult();
    const chunk = result.chunks.find((c) => c.id === "D03-s2")!;
    chunk.embeddingText += "\nligne ajoutée";
    const report = await run({ result });

    assert.deepEqual(report.plan.reembed, ["D03-s2"]);
    assert.equal(embedder.calls.length, 2);
    assert.deepEqual(embedder.calls[1].texts, [chunk.embeddingText]);
    assert.equal(report.voyageTokens, Math.ceil(chunk.embeddingText.length / 4));
    assert.equal(report.upserted, 1);
  });

  test("chunk obsolète : signalé, jamais supprimé ; les autres personas sont ignorés", async () => {
    const { store, run } = setup();
    await run();
    const old = { ...store.rows.get("julie/D01-p1")!, id: "D99-old" };
    store.rows.set("julie/D99-old", old);
    store.rows.set("autre/D01-p1", { ...old, persona_id: "autre", id: "D01-p1", metadata: { ...old.metadata, personaId: "autre" } });

    const report = await run();
    assert.deepEqual(report.plan.obsolete, ["D99-old"]);
    assert.ok(store.rows.has("julie/D99-old"), "la ligne obsolète est toujours là");
    assert.ok(store.rows.has("autre/D01-p1"));
    assert.equal(report.plan.unchanged.length, 29);
    assert.match(formatIndexReport(report), /Obsolètes \(non supprimés\)\s+1\s+D99-old[\s\S]*retrait manuel/);
    assert.equal("delete" in store, false);
  });

  test("--dry-run : plan et estimation, sans client Voyage ni écriture", async () => {
    const { store, embedder, run, embedderCreated } = setup();
    const report = await run({ dryRun: true });
    assert.equal(report.dryRun, true);
    assert.equal(report.plan.create.length, 29);
    assert.ok(report.estimatedTokens > 3000 && report.estimatedTokens < 4000, `≈${report.estimatedTokens}`);
    assert.equal(embedderCreated(), 0);
    assert.equal(embedder.calls.length, 0);
    assert.equal(store.writes, 0);
    assert.equal(store.lists, 1, "le dry-run lit l'état existant");
    assert.match(formatIndexReport(report), /SIMULATION[\s\S]*Tokens Voyage consommés : 0 \(simulation\)/);
  });

  test("--dry-run ignore le plafond (aucun appel payant) mais l'affiche", async () => {
    const { run } = setup();
    const report = await run({ dryRun: true, maxEstimatedTokens: 10 });
    assert.match(formatIndexReport(report), /plafond 10\)/);
  });

  test("plafond de tokens dépassé : arrêt avant tout appel Voyage et toute écriture", async () => {
    const { store, embedder, run, embedderCreated } = setup();
    await assert.rejects(run({ maxEstimatedTokens: 100 }), (e: unknown) => {
      assert.ok(e instanceof IndexingError);
      assert.match(e.message, /dépassent le plafond de 100 : aucun appel Voyage/);
      return true;
    });
    assert.equal(embedderCreated(), 0);
    assert.equal(embedder.calls.length, 0);
    assert.equal(store.writes, 0);
  });

  test("le plafond ne compte que les chunks à embedder", async () => {
    const { run } = setup();
    await run();
    const result = freshResult();
    result.chunks.find((c) => c.id === "D01-p1")!.embeddingText += " modifié";
    const report = await run({ result, maxEstimatedTokens: 200 });
    assert.deepEqual(report.plan.reembed, ["D01-p1"]);
  });

  test("--force : tout est ré-embeddé", async () => {
    const { embedder, run } = setup();
    await run();
    const report = await run({ force: true });
    assert.equal(report.plan.reembed.length, 29);
    assert.equal(embedder.calls.length, 2);
  });

  test("écriture par lot : une panne Voyage ne fait pas perdre les embeddings déjà payés", async () => {
    const { store, embedder, run } = setup();
    embedder.failOnCall = 3;
    await assert.rejects(run({ writeBatchSize: 10 }), /panne simulée/);
    assert.equal(store.rows.size, 20);

    embedder.failOnCall = null;
    const report = await run({ writeBatchSize: 10 });
    assert.equal(report.plan.create.length, 9);
    assert.equal(report.plan.unchanged.length, 20);
    assert.equal(store.rows.size, 29);
  });

  test("chunking en erreur : rien n'est lu ni écrit", async () => {
    const { store, run } = setup();
    const result = freshResult();
    result.anomalies.push({ severity: "error", sourceFile: "x", message: "test" });
    await assert.rejects(run({ result }), /chunking en erreur/);
    assert.equal(store.lists, 0);
    assert.equal(store.writes, 0);
  });

  test("aucun chunk : indexation refusée", async () => {
    const { store, run } = setup();
    const result = freshResult();
    result.chunks = [];
    await assert.rejects(run({ result }), /aucun chunk produit/);
    assert.equal(store.lists, 0);
  });
});

describe("planification", () => {
  test("ordre des clés JSONB sans effet sur la comparaison", () => {
    assert.equal(canonicalJson({ b: 1, a: { d: [1, { y: 2, x: 1 }], c: null } }), canonicalJson({ a: { c: null, d: [1, { x: 1, y: 2 }] }, b: 1 }));
    assert.notEqual(canonicalJson({ a: [1, 2] }), canonicalJson({ a: [2, 1] }));
  });

  test("ligne d'un autre persona renvoyée par le stockage : erreur", async () => {
    const store = new MemoryStore();
    store.listByPersona = async () => [{ persona_id: "autre", id: "D01-p1", content: "x", metadata: {} }];
    await assert.rejects(
      indexPersona({ result: freshResult(), store, getEmbedder: () => new FakeEmbedder(), dryRun: true }),
      /hors du persona julie/,
    );
  });

  test("identifiant dupliqué dans le pipeline : erreur", async () => {
    const result = freshResult();
    result.chunks.push(structuredClone(result.chunks[0]));
    await assert.rejects(
      indexPersona({ result, store: new MemoryStore(), getEmbedder: () => new FakeEmbedder(), dryRun: true }),
      /identifiant de chunk dupliqué : D01-p1/,
    );
  });
});
