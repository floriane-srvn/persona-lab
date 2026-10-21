import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import path from "node:path";
import { describe, test } from "node:test";
import { promisify } from "node:util";

import { chunkPersona } from "../chunking/pipeline.ts";
import { JULIE } from "../chunking/personas.ts";
import type { AnthropicClient } from "./anthropic.ts";
import { DEBUG_EXCERPT_CHARS, formatDebugReport } from "./debug.ts";
import { createPersonaResponder } from "./generation.ts";
import { chunkToRow } from "./indexing.ts";
import type { ChunkRow } from "./indexing.ts";
import { createRetriever } from "./retrieval.ts";
import type { ChunkMatcher } from "./retrieval.ts";
import type { VoyageClient } from "./voyage.ts";

// Garde-fou : aucun test ne doit atteindre le réseau.
globalThis.fetch = (() => {
  throw new Error("Appel réseau interdit dans les tests");
}) as typeof fetch;

const ROOT = path.resolve(import.meta.dirname, "../../..");
const julieRows = new Map<string, ChunkRow>(chunkPersona(JULIE, ROOT).chunks.map((c) => [c.id, chunkToRow(c)]));
const IDS = ["D04-q4", "D04-q3", "D05-q4", "D06-s2"];

/** Counts every paid or network call, so the diagnostic can be shown to add none. */
function setup(citedIds: string[], level = "documented") {
  const calls = { voyage: 0, supabase: 0, anthropic: 0 };
  const embedder: VoyageClient = {
    async embed() {
      calls.voyage++;
      return { embeddings: [Array.from({ length: 1024 }, () => 0.01)], totalTokens: 14, requests: 1 };
    },
  };
  const matcher: ChunkMatcher = {
    async match(_p, _e, matchCount) {
      calls.supabase++;
      return IDS.slice(0, matchCount).map((id, i) => {
        const row = julieRows.get(id)!;
        return { id, content: row.content, metadata: row.metadata, similarity: 0.8123 - i * 0.05 };
      });
    },
  };
  const llm: AnthropicClient = {
    model: "claude-test",
    async callTool() {
      calls.anthropic++;
      const input = {
        persona_response: "Je ne sais pas, ça dépend du nombre de repas.",
        grounding: { level, source_ids: citedIds, reasoning: "Julie le dit en D04-q4.", limitations: [] },
      };
      return { input, model: "claude-test", messageId: "msg_1", inputTokens: 1500, outputTokens: 180, attempts: 1 };
    },
  };
  const responder = createPersonaResponder({ retriever: createRetriever({ embedder, matcher }), llm });
  return { calls, responder };
}

const QUESTION = "Paieriez-vous 60 € par semaine pour des paniers-repas ?";

describe("diagnostic --debug", () => {
  test("liste tous les passages récupérés, cités ou non, sans appel supplémentaire", async () => {
    const { responder, calls } = setup(["D04-q4", "D04-q3"]);
    const reply = await responder.respond("julie", QUESTION);
    const before = { ...calls };
    const report = formatDebugReport(reply);

    assert.deepEqual(calls, before, "le diagnostic ne refait aucun appel");
    assert.deepEqual(before, { voyage: 1, supabase: 1, anthropic: 1 });
    assert.match(report, /top_k 5 · seuil aucun · 4 candidat\(s\) · 0 sous le seuil \(non détaillés\) · 4 passage\(s\) transmis · 2 cité\(s\)/);
    assert.match(report, /#1 {2}D04-q4 {2}similarité 0\.8123 {2}\[cité\]/);
    assert.match(report, /#2 {2}D04-q3 {2}similarité 0\.7623 {2}\[cité\]/);
    assert.match(report, /#3 {2}D05-q4 {2}similarité 0\.7123 {2}\[non cité\]/);
    assert.match(report, /#4 {2}D06-s2 {2}similarité 0\.6623 {2}\[non cité\]/);
  });

  test("affiche le document source, les lignes d'origine, le statut et un extrait court", async () => {
    const { responder } = setup(["D04-q4"]);
    const report = formatDebugReport(await responder.respond("julie", QUESTION));
    const q4 = julieRows.get("D04-q4")!.metadata as { sourceFile: string; docTitle: string; lineStart: number; lineEnd: number };

    assert.ok(report.includes(`source : ${q4.sourceFile} · ${q4.docTitle} · l.${q4.lineStart}-${q4.lineEnd} · declared`));
    assert.match(report, /source : .*D06.* · interpretation/);
    const excerpts = report.split("\n").filter((l) => l.startsWith("    extrait : "));
    assert.equal(excerpts.length, 4);
    for (const line of excerpts) {
      const text = line.slice("    extrait : ".length);
      assert.ok(text.length <= DEBUG_EXCERPT_CHARS + 1, "extrait borné");
      assert.ok(!text.includes("\n"));
    }
    assert.match(report, /extrait : \*\*Chercheuse : Seriez-vous prête à payer 60 € par semaine \?\*\*/);
  });

  test("n'expose ni texte d'embedding ni métadonnées techniques", async () => {
    const { responder } = setup(["D04-q4"]);
    const reply = await responder.respond("julie", QUESTION);
    const report = formatDebugReport(reply);
    const embeddingText = String(reply.retrieval.passages[0].metadata.embeddingText);
    assert.ok(!report.includes(embeddingText));
    assert.ok(!/embedding|contentHash|SECRET|API_KEY/i.test(report));
  });

  test("ne modifie pas la réponse", async () => {
    const { responder } = setup(["D04-q4"]);
    const reply = await responder.respond("julie", QUESTION);
    const snapshot = structuredClone(reply);
    formatDebugReport(reply);
    assert.deepEqual(reply, snapshot);
    assert.ok(!reply.answer.persona_response.includes("Diagnostic"));
  });

  test("sans appel Anthropic : le signale et compte les candidats écartés par le seuil", async () => {
    const { responder, calls } = setup([]);
    const reply = await responder.respond("julie", QUESTION, { minSimilarity: 0.99 });
    const report = formatDebugReport(reply);
    assert.equal(calls.anthropic, 0);
    assert.match(report, /seuil 0\.99 · 4 candidat\(s\) · 4 sous le seuil \(non détaillés\) · 0 passage\(s\) transmis · 0 cité\(s\)/);
    assert.match(report, /Aucun appel Anthropic/);
    assert.ok(!report.includes("#1"));
  });
});

describe("scripts/ask.ts", () => {
  const run = (...args: string[]) =>
    promisify(execFile)(process.execPath, [path.join(ROOT, "scripts/ask.ts"), ...args], {
      cwd: ROOT,
      // No --env-file and no key: the script stops at configuration, before any network call.
      env: { PATH: process.env.PATH, NODE_ENV: "test" },
    }).then(
      () => assert.fail("échec de configuration attendu"),
      (e: { stderr: string }) => e.stderr,
    );

  test("--debug est une option reconnue, exclue de la question", async () => {
    const stderr = await run(QUESTION, "--debug");
    assert.doesNotMatch(stderr, /Option\(s\) inconnue\(s\)/);
    assert.match(stderr, /ANTHROPIC_API_KEY manquante ou vide/);
  });

  test("une option voisine reste refusée", async () => {
    assert.match(await run(QUESTION, "--debug=1"), /Option\(s\) inconnue\(s\) : --debug=1/);
  });
});
