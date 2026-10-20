import assert from "node:assert/strict";
import { inspect } from "node:util";
import { describe, test } from "node:test";

import type { ChunkRow } from "./indexing.ts";
import { SupabaseStoreError, createSupabaseAdminClient, createSupabaseChunkStore } from "./supabase-store.ts";

// Garde-fou : aucun test ne doit atteindre le réseau. Seul le faux fetch injecté répond.
globalThis.fetch = (() => {
  throw new Error("Appel réseau interdit dans les tests");
}) as typeof fetch;

const ENV = { SUPABASE_URL: "https://abcdefghijklmnopqrst.supabase.co", SUPABASE_SECRET_KEY: "sb_secret_TEST-ONLY-not-a-real-key" };

interface Call {
  method: string;
  url: URL;
  headers: Headers;
  body: unknown;
}

function fakeFetch(...responses: (Response | (() => Response))[]) {
  const calls: Call[] = [];
  const impl = (async (input: string | URL | Request, init: RequestInit = {}) => {
    const text = typeof init.body === "string" ? init.body : null;
    calls.push({
      method: init.method ?? "GET",
      url: new URL(String(input)),
      headers: new Headers(init.headers),
      body: text ? JSON.parse(text) : null,
    });
    const next = responses.shift();
    if (!next) throw new Error("réponse non prévue");
    return typeof next === "function" ? next() : next;
  }) as typeof fetch;
  return { impl, calls };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const storeWith = (impl: typeof fetch) => createSupabaseChunkStore(createSupabaseAdminClient(ENV, { fetch: impl }));

const row = (id: string): ChunkRow => ({ persona_id: "julie", id, content: `texte ${id}`, metadata: { personaId: "julie", sourceId: "D01" } });

describe("client administrateur", () => {
  test("variables manquantes refusées sans rien révéler", () => {
    assert.throws(() => createSupabaseAdminClient({}), /SUPABASE_URL manquante/);
    assert.throws(() => createSupabaseAdminClient({ SUPABASE_URL: ENV.SUPABASE_URL }), /SUPABASE_SECRET_KEY manquante/);
    assert.throws(() => createSupabaseAdminClient({ SUPABASE_URL: ENV.SUPABASE_URL, SUPABASE_SECRET_KEY: "  " }), /SUPABASE_SECRET_KEY manquante/);
  });
});

describe("lecture", () => {
  test("seulement le persona demandé, sans la colonne embedding", async () => {
    const { impl, calls } = fakeFetch(json([row("D01-p1")]));
    const rows = await storeWith(impl).listByPersona("julie");
    assert.deepEqual(rows, [row("D01-p1")]);

    const [call] = calls;
    assert.equal(call.method, "GET");
    assert.equal(call.url.origin + call.url.pathname, `${ENV.SUPABASE_URL}/rest/v1/persona_chunks`);
    assert.equal(call.url.searchParams.get("select"), "persona_id,id,content,metadata");
    assert.equal(call.url.searchParams.get("persona_id"), "eq.julie");
    assert.equal(call.headers.get("apikey"), ENV.SUPABASE_SECRET_KEY);
  });

  test("pagination au-delà de 500 lignes", async () => {
    const page = Array.from({ length: 500 }, (_, i) => row(`D01-p${i}`));
    const { impl, calls } = fakeFetch(json(page), json([row("D02-q1")]));
    const rows = await storeWith(impl).listByPersona("julie");
    assert.equal(rows.length, 501);
    assert.equal(calls.length, 2);
    assert.deepEqual([calls[0].url.searchParams.get("offset"), calls[1].url.searchParams.get("offset")], ["0", "500"]);
  });
});

describe("écriture", () => {
  test("upsert sur (persona_id, id) avec l'embedding", async () => {
    const { impl, calls } = fakeFetch(new Response(null, { status: 201 }));
    const embedding = Array.from({ length: 1024 }, () => 0.5);
    await storeWith(impl).upsert([{ ...row("D04-q4"), embedding }]);

    const [call] = calls;
    assert.equal(call.method, "POST");
    assert.equal(call.url.searchParams.get("on_conflict"), "persona_id,id");
    assert.match(call.headers.get("prefer") ?? "", /resolution=merge-duplicates/);
    assert.deepEqual(call.body, [{ ...row("D04-q4"), embedding }]);
  });

  test("upsert découpé par lots de 100", async () => {
    const { impl, calls } = fakeFetch(new Response(null, { status: 201 }), new Response(null, { status: 201 }));
    const rows = Array.from({ length: 150 }, (_, i) => ({ ...row(`D01-p${i}`), embedding: [0.1] }));
    await storeWith(impl).upsert(rows);
    assert.deepEqual(calls.map((c) => (c.body as unknown[]).length), [100, 50]);
  });

  test("mise à jour sans embedding : PATCH ciblé, embedding jamais envoyé", async () => {
    const { impl, calls } = fakeFetch(json([{ id: "D04-q4" }]));
    await storeWith(impl).updateWithoutEmbedding([row("D04-q4")]);

    const [call] = calls;
    assert.equal(call.method, "PATCH");
    assert.equal(call.url.searchParams.get("persona_id"), "eq.julie");
    assert.equal(call.url.searchParams.get("id"), "eq.D04-q4");
    assert.deepEqual(call.body, { content: "texte D04-q4", metadata: row("D04-q4").metadata });
    assert.equal(Object.hasOwn(call.body as object, "embedding"), false);
  });

  test("mise à jour d'une ligne absente : erreur explicite", async () => {
    const { impl } = fakeFetch(json([]));
    await assert.rejects(storeWith(impl).updateWithoutEmbedding([row("D99-x")]), /ligne julie\/D99-x introuvable/);
  });

  test("le stockage n'expose aucune suppression", () => {
    const store = storeWith(fakeFetch().impl);
    assert.deepEqual(Object.keys(store).sort(), ["listByPersona", "updateWithoutEmbedding", "upsert"]);
  });
});

describe("erreurs", () => {
  test("permission refusée : erreur claire sans la clé", async () => {
    const { impl } = fakeFetch(json({ message: "permission denied for table persona_chunks", code: "42501" }, 403));
    await assert.rejects(storeWith(impl).listByPersona("julie"), (e: unknown) => {
      assert.ok(e instanceof SupabaseStoreError);
      assert.match(e.message, /Supabase \(lecture\) : permission denied for table persona_chunks \[42501\]/);
      assert.ok(!inspect(e, { depth: 10, showHidden: true }).includes(ENV.SUPABASE_SECRET_KEY));
      return true;
    });
  });

  test("échec d'upsert remonté", async () => {
    const { impl } = fakeFetch(json({ message: "expected 1024 dimensions, not 3", code: "22000" }, 400));
    await assert.rejects(storeWith(impl).upsert([{ ...row("D01-p1"), embedding: [1, 2, 3] }]), /Supabase \(upsert\) : expected 1024 dimensions/);
  });
});
