import assert from "node:assert/strict";
import { inspect } from "node:util";
import { beforeEach, describe, test } from "node:test";

import {
  VOYAGE_DIMENSIONS,
  VOYAGE_ENDPOINT,
  VoyageError,
  createVoyageClient,
  createVoyageClientFromEnv,
  splitBatches,
} from "./voyage.ts";
import type { VoyageClientOptions } from "./voyage.ts";

// Garde-fou : aucun test ne doit atteindre le réseau. Le vrai fetch est neutralisé.
globalThis.fetch = (() => {
  throw new Error("Appel réseau interdit dans les tests");
}) as typeof fetch;

const API_KEY = "test-voyage-key-SECRET-123";

const vector = (seed: number) => Array.from({ length: VOYAGE_DIMENSIONS }, (_, i) => (seed + i) / 10_000);

const okBody = (count: number, overrides: Record<string, unknown> = {}) => ({
  object: "list",
  data: Array.from({ length: count }, (_, i) => ({ object: "embedding", embedding: vector(i), index: i })),
  model: "voyage-4",
  usage: { total_tokens: count * 10 },
  ...overrides,
});

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

interface Call {
  url: string;
  init: RequestInit;
  body: Record<string, unknown>;
}

/** Fake fetch: replays the queued responses and records every request. */
function fakeFetch(...responses: (Response | ((call: Call) => Response) | Error)[]) {
  const calls: Call[] = [];
  const impl = (async (url: string | URL | Request, init?: RequestInit) => {
    const call = { url: String(url), init: init!, body: JSON.parse(String(init!.body)) };
    calls.push(call);
    const next = responses.shift();
    if (!next) throw new Error("réponse non prévue");
    if (next instanceof Error) throw next;
    return typeof next === "function" ? next(call) : next;
  }) as typeof fetch;
  return { impl, calls };
}

/** Echoes one valid vector per input, whatever the batch size. */
const echo = (call: Call) => json(okBody((call.body.input as string[]).length));

let sleeps: number[];
beforeEach(() => {
  sleeps = [];
});

const client = (fetchImpl: typeof fetch, options: Partial<VoyageClientOptions> = {}) =>
  createVoyageClient({
    apiKey: API_KEY,
    fetch: fetchImpl,
    sleep: async (ms) => void sleeps.push(ms),
    ...options,
  });

const rejectsWith = async (promise: Promise<unknown>, check: (e: VoyageError) => void) => {
  await assert.rejects(promise, (error: unknown) => {
    assert.ok(error instanceof VoyageError, `VoyageError attendue, reçu ${String(error)}`);
    check(error);
    return true;
  });
};

describe("requête envoyée", () => {
  test("documents : modèle, dimensions, input_type, truncation et authentification", async () => {
    const { impl, calls } = fakeFetch(echo);
    const result = await client(impl).embed(["chunk A", "chunk B"], "document");

    assert.equal(calls.length, 1);
    const [{ url, init, body }] = calls;
    assert.equal(url, VOYAGE_ENDPOINT);
    assert.equal(init.method, "POST");
    assert.deepEqual(init.headers, { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" });
    assert.ok(init.signal instanceof AbortSignal, "délai d'attente absent");
    assert.deepEqual(body, {
      input: ["chunk A", "chunk B"],
      model: "voyage-4",
      input_type: "document",
      output_dimension: 1024,
      output_dtype: "float",
      truncation: false,
    });
    assert.equal(result.embeddings.length, 2);
    assert.equal(result.totalTokens, 20);
    assert.equal(result.requests, 1);
  });

  test("questions : input_type query", async () => {
    const { impl, calls } = fakeFetch(echo);
    await client(impl).embed(["Julie paierait-elle 60 € ?"], "query");
    assert.equal(calls[0].body.input_type, "query");
  });
});

describe("cas nominaux", () => {
  test("les vecteurs sont remis dans l'ordre des textes, même si l'API les renvoie dans le désordre", async () => {
    const body = okBody(3);
    body.data.reverse();
    const { impl } = fakeFetch(json(body));
    const { embeddings } = await client(impl).embed(["a", "b", "c"], "document");
    assert.deepEqual(embeddings, [vector(0), vector(1), vector(2)]);
  });

  test("liste vide : aucun appel", async () => {
    const { impl, calls } = fakeFetch();
    assert.deepEqual(await client(impl).embed([], "document"), { embeddings: [], totalTokens: 0, requests: 0 });
    assert.equal(calls.length, 0);
  });

  test("plus de 1 000 textes : plusieurs requêtes, ordre et tokens conservés", async () => {
    const texts = Array.from({ length: 1001 }, (_, i) => `texte ${i}`);
    const { impl, calls } = fakeFetch(echo, echo);
    const result = await client(impl).embed(texts, "document");
    assert.deepEqual(calls.map((c) => (c.body.input as string[]).length), [1000, 1]);
    assert.equal((calls[1].body.input as string[])[0], "texte 1000");
    assert.equal(result.embeddings.length, 1001);
    assert.equal(result.totalTokens, 10_010);
    assert.equal(result.requests, 2);
  });

  test("découpage par volume estimé de tokens", () => {
    const big = "x".repeat(400_000); // ≈200 000 tokens en estimation prudente
    assert.deepEqual(splitBatches([big, big, "petit"]).map((b) => b.length), [1, 2]);
    assert.deepEqual(splitBatches(Array(29).fill("chunk")).map((b) => b.length), [29]);
  });
});

describe("validation des entrées", () => {
  test("texte vide refusé avant tout appel", async () => {
    const { impl, calls } = fakeFetch();
    await rejectsWith(client(impl).embed(["ok", "  "], "document"), (e) => assert.match(e.message, /texte 1 vide/));
    assert.equal(calls.length, 0);
  });

  test("input_type inconnu refusé", async () => {
    const { impl, calls } = fakeFetch();
    // @ts-expect-error valeur volontairement invalide
    await rejectsWith(client(impl).embed(["a"], "passage"), (e) => assert.match(e.message, /input_type invalide/));
    assert.equal(calls.length, 0);
  });

  test("clé absente ou vide refusée", () => {
    assert.throws(() => createVoyageClient({ apiKey: "  " }), /VOYAGE_API_KEY manquante/);
    assert.throws(() => createVoyageClientFromEnv({}), /VOYAGE_API_KEY manquante/);
    assert.doesNotThrow(() => createVoyageClientFromEnv({ VOYAGE_API_KEY: API_KEY }));
  });
});

describe("validation des réponses (jamais de nouvelle tentative)", () => {
  const cases: [string, unknown, RegExp][] = [
    ["nombre de vecteurs incorrect", okBody(1), /1 vecteurs reçus pour 2 textes/],
    ["dimension incorrecte", { ...okBody(2), data: [{ index: 0, embedding: [0.1, 0.2] }, { index: 1, embedding: vector(1) }] }, /dimension 2 au lieu de 1024/],
    ["valeur non numérique", { ...okBody(2), data: [{ index: 0, embedding: [...vector(0).slice(1), "0.1"] }, { index: 1, embedding: vector(1) }] }, /valeurs non numériques/],
    ["valeur nulle (NaN sérialisé)", { ...okBody(2), data: [{ index: 0, embedding: [null, ...vector(0).slice(1)] }, { index: 1, embedding: vector(1) }] }, /valeurs non numériques/],
    ["index dupliqué", { ...okBody(2), data: [{ index: 0, embedding: vector(0) }, { index: 0, embedding: vector(1) }] }, /index 0 dupliqué/],
    ["index hors limites", { ...okBody(2), data: [{ index: 0, embedding: vector(0) }, { index: 5, embedding: vector(1) }] }, /index 5 hors limites/],
    ["embedding absent", { ...okBody(2), data: [{ index: 0 }, { index: 1, embedding: vector(1) }] }, /dimension aucune/],
    ["autre modèle", okBody(2, { model: "voyage-4-lite" }), /modèle « voyage-4-lite »/],
    ["usage absent", okBody(2, { usage: undefined }), /usage.total_tokens/],
    ["data absent", { model: "voyage-4", usage: { total_tokens: 1 } }, /champ data absent/],
  ];

  for (const [name, body, pattern] of cases) {
    test(name, async () => {
      const { impl, calls } = fakeFetch(json(body));
      await rejectsWith(client(impl).embed(["a", "b"], "document"), (e) => assert.match(e.message, pattern));
      assert.equal(calls.length, 1);
    });
  }

});

describe("réponses 2xx illisibles : diagnostic précis, sans nouvelle tentative ni corps brut", () => {
  const MARKER = "CONTENU-BRUT-À-NE-JAMAIS-AFFICHER";
  const valid = JSON.stringify(okBody(1));
  const enc = new TextEncoder();

  /** Body that sends `head`, then fails with `error` (connection cut, timeout…). */
  const brokenStream = (head: string, error: unknown) =>
    new ReadableStream({
      start(controller) {
        controller.enqueue(enc.encode(head));
        controller.error(error);
      },
    });

  const cases: [string, () => Response, string, RegExp][] = [
    ["corps vide", () => new Response("", { status: 200, headers: { "content-type": "application/json", "content-length": "0" } }), "empty_body", /Réponse Voyage AI vide — statut 200, content-type application\/json, content-length 0, 0 octets reçus/],
    ["corps composé d'espaces", () => new Response("  \n ", { status: 200 }), "empty_body", /Réponse Voyage AI vide/],
    [
      "JSON tronqué, taille annoncée supérieure à la taille reçue",
      () => new Response(valid.slice(0, 500) + MARKER, { status: 200, headers: { "content-type": "application/json", "content-length": String(valid.length) } }),
      "json_parse",
      /JSON tronqué \(\d+ octets reçus sur \d+ annoncés\)/,
    ],
    ["JSON incomplet sans Content-Length", () => new Response(`{"data": [${MARKER}`, { status: 200 }), "json_parse", /JSON incomplet ou mal formé — statut 200, content-type text\/plain;charset=UTF-8, content-length absent/],
    ["JSON compressé : pas de comparaison de taille trompeuse", () => new Response(`{"x":${MARKER}`, { status: 200, headers: { "content-encoding": "gzip", "content-length": "99999" } }), "json_parse", /JSON incomplet ou mal formé .*content-encoding gzip/],
    ["page HTML renvoyée avec un statut 200", () => new Response(`<html><body>${MARKER}</body></html>`, { status: 200, headers: { "content-type": "text/html" } }), "json_parse", /contenu HTML ou XML au lieu de JSON .*content-type text\/html/],
    ["texte non JSON", () => new Response(`OK ${MARKER}`, { status: 200 }), "json_parse", /contenu non JSON/],
    ["connexion coupée pendant la lecture", () => new Response(brokenStream(valid.slice(0, 300), new TypeError("terminated")), { status: 200 }), "body_read", /interrompue pendant la lecture du corps : flux interrompu \(TypeError\)/],
    ["délai dépassé pendant la lecture", () => new Response(brokenStream(valid.slice(0, 300), new DOMException("timeout", "TimeoutError")), { status: 200 }), "body_read", /interrompue pendant la lecture du corps : délai de 30000 ms dépassé/],
  ];

  for (const [name, makeResponse, stage, pattern] of cases) {
    test(name, async () => {
      const { impl, calls } = fakeFetch(makeResponse());
      await rejectsWith(client(impl).embed(["a"], "query"), (e) => {
        assert.equal(e.stage, stage);
        assert.equal(e.status, 200);
        assert.equal(e.retryable, false);
        assert.match(e.message, pattern);
        assert.ok(!e.message.includes(MARKER), "le corps brut ne doit jamais apparaître");
        assert.ok(!e.message.includes(API_KEY));
      });
      assert.equal(calls.length, 1, "un 2xx a déjà été traité par Voyage : pas de nouvel appel facturé");
      assert.deepEqual(sleeps, []);
    });
  }

  test("JSON valide mais structure inattendue : étape « validation »", async () => {
    const { impl } = fakeFetch(json({ unexpected: true }));
    await rejectsWith(client(impl).embed(["a"], "query"), (e) => assert.equal(e.stage, "validation"));
  });
});

describe("étapes d'échec et corps d'erreur HTTP", () => {
  test("erreur réseau avant toute réponse : étape « request »", async () => {
    const { impl } = fakeFetch(new TypeError("fetch failed"));
    await rejectsWith(client(impl).embed(["a"], "query"), (e) => {
      assert.equal(e.stage, "request");
      assert.match(e.message, /avant toute réponse/);
    });
  });

  test("erreur HTTP : étape « http », détail structuré de Voyage conservé", async () => {
    const { impl } = fakeFetch(json({ detail: "Invalid input:\n bad request" }, 400));
    await rejectsWith(client(impl).embed(["a"], "query"), (e) => {
      assert.equal(e.stage, "http");
      assert.match(e.message, /Voyage AI a répondu 400 : Invalid input: bad request/);
    });
  });

  test("erreur HTTP avec corps non JSON : taille seulement, jamais le contenu", async () => {
    const MARKER = "PAGE-PROXY-NE-PAS-AFFICHER";
    const { impl } = fakeFetch(new Response(`<html>${MARKER}</html>`, { status: 403, headers: { "content-type": "text/html" } }));
    await rejectsWith(client(impl).embed(["a"], "query"), (e) => {
      assert.match(e.message, /Voyage AI a répondu 403 : corps non JSON de \d+ octets/);
      assert.ok(!e.message.includes(MARKER));
    });
  });
});

describe("erreurs HTTP et nouvelles tentatives", () => {
  test("429 puis succès : une nouvelle tentative avec délai croissant", async () => {
    const { impl, calls } = fakeFetch(json({ detail: "Rate limit" }, 429), echo);
    const result = await client(impl).embed(["a"], "document");
    assert.equal(calls.length, 2);
    assert.deepEqual(sleeps, [500]);
    assert.equal(result.embeddings.length, 1);
  });

  test("Retry-After est respecté", async () => {
    const { impl } = fakeFetch(json({ detail: "Rate limit" }, 429, { "Retry-After": "2" }), echo);
    await client(impl).embed(["a"], "document");
    assert.deepEqual(sleeps, [2000]);
  });

  test("5xx répétés : arrêt après 3 nouvelles tentatives (4 appels)", async () => {
    const { impl, calls } = fakeFetch(...Array.from({ length: 4 }, () => json({ detail: "boom" }, 503)));
    await rejectsWith(client(impl).embed(["a"], "document"), (e) => {
      assert.equal(e.status, 503);
      assert.equal(e.retryable, true);
      assert.equal(e.attempts, 4);
      assert.match(e.message, /503 après 4 tentatives : boom/);
    });
    assert.equal(calls.length, 4);
    assert.deepEqual(sleeps, [500, 1000, 2000]);
  });

  test("délai plafonné", async () => {
    const { impl } = fakeFetch(json({}, 500), json({}, 500), echo);
    await client(impl, { baseDelayMs: 20_000, maxDelayMs: 25_000 }).embed(["a"], "document");
    assert.deepEqual(sleeps, [20_000, 25_000]);
  });

  for (const status of [400, 401, 403, 404, 422]) {
    test(`${status} : aucune nouvelle tentative`, async () => {
      const { impl, calls } = fakeFetch(json({ detail: "refusé" }, status));
      await rejectsWith(client(impl).embed(["a"], "document"), (e) => {
        assert.equal(e.status, status);
        assert.equal(e.retryable, false);
        assert.equal(e.attempts, 1);
      });
      assert.equal(calls.length, 1);
      assert.deepEqual(sleeps, []);
    });
  }

  test("erreur réseau ou délai dépassé : aucune nouvelle tentative", async () => {
    const timeout = Object.assign(new Error("The operation was aborted due to timeout"), { name: "TimeoutError" });
    const { impl, calls } = fakeFetch(timeout);
    await rejectsWith(client(impl).embed(["a"], "document"), (e) => {
      assert.equal(e.status, null);
      assert.match(e.message, /TimeoutError/);
    });
    assert.equal(calls.length, 1);
  });
});

describe("aucun secret dans les erreurs", () => {
  const leaks = (error: unknown) => {
    const text = [String(error), JSON.stringify(error), inspect(error, { depth: 10, showHidden: true })].join("\n");
    return text.includes(API_KEY);
  };

  test("clé renvoyée par l'API dans le détail d'une erreur 401", async () => {
    const { impl } = fakeFetch(json({ detail: `Invalid API key: ${API_KEY}` }, 401));
    await rejectsWith(client(impl).embed(["a"], "document"), (e) => {
      assert.ok(!leaks(e), "la clé apparaît dans l'erreur");
      assert.match(e.message, /\[clé masquée\]/);
    });
  });

  test("clé présente dans un corps d'erreur non JSON", async () => {
    const { impl } = fakeFetch(...Array.from({ length: 4 }, () => new Response(`upstream error ${API_KEY}`, { status: 502 })));
    await rejectsWith(client(impl).embed(["a"], "document"), (e) => assert.ok(!leaks(e)));
  });

  test("clé présente dans le message d'une erreur réseau", async () => {
    const { impl } = fakeFetch(new TypeError(`fetch failed for Bearer ${API_KEY}`));
    await rejectsWith(client(impl).embed(["a"], "document"), (e) => assert.ok(!leaks(e)));
  });

  test("le corps d'erreur est tronqué", async () => {
    const { impl } = fakeFetch(json({ detail: "x".repeat(5000) }, 400));
    await rejectsWith(client(impl).embed(["a"], "document"), (e) => assert.ok(e.message.length < 400));
  });
});
