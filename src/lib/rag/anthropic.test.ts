import assert from "node:assert/strict";
import { inspect } from "node:util";
import { beforeEach, describe, test } from "node:test";

import {
  ANTHROPIC_DEFAULT_MODEL,
  ANTHROPIC_ENDPOINT,
  ANTHROPIC_VERSION,
  AnthropicError,
  createAnthropicClient,
  createAnthropicClientFromEnv,
} from "./anthropic.ts";
import type { AnthropicClientOptions, ToolCallRequest } from "./anthropic.ts";

// Garde-fou : aucun test ne doit atteindre le réseau. Le vrai fetch est neutralisé.
globalThis.fetch = (() => {
  throw new Error("Appel réseau interdit dans les tests");
}) as typeof fetch;

const API_KEY = "test-anthropic-key-SECRET-456";
const MARKER = "CORPS-BRUT-NE-DOIT-PAS-FUIR";

const REQUEST: ToolCallRequest = {
  system: "Consignes système",
  messages: [{ role: "user", content: "Bonjour" }],
  tool: { name: "persona_answer", description: "Réponse", input_schema: { type: "object" } },
  maxTokens: 512,
};

const INPUT = { persona_response: "Je ne sais pas.", grounding: { level: "not_established", source_ids: [], reasoning: "r", limitations: [] } };

const okMessage = (overrides: Record<string, unknown> = {}) => ({
  id: "msg_123",
  type: "message",
  role: "assistant",
  model: ANTHROPIC_DEFAULT_MODEL,
  content: [{ type: "tool_use", id: "toolu_1", name: "persona_answer", input: INPUT }],
  stop_reason: "tool_use",
  usage: { input_tokens: 900, output_tokens: 120 },
  ...overrides,
});

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

const apiError = (status: number, type: string, message: string, headers: Record<string, string> = {}) =>
  json({ type: "error", error: { type, message } }, status, headers);

interface Call {
  url: string;
  init: RequestInit;
  body: Record<string, unknown>;
}

/** Fake fetch: replays the queued responses and records every request. */
function fakeFetch(...responses: (Response | Error)[]) {
  const calls: Call[] = [];
  const impl = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init!, body: JSON.parse(String(init!.body)) });
    const next = responses.shift();
    if (!next) throw new Error("réponse non prévue");
    if (next instanceof Error) throw next;
    return next;
  }) as typeof fetch;
  return { impl, calls };
}

let sleeps: number[];
beforeEach(() => {
  sleeps = [];
});

const client = (fetchImpl: typeof fetch, options: Partial<AnthropicClientOptions> = {}) =>
  createAnthropicClient({ apiKey: API_KEY, fetch: fetchImpl, sleep: async (ms) => void sleeps.push(ms), ...options });

/** Asserts the error and that neither the key nor a raw body leaks through it. */
async function rejects(promise: Promise<unknown>, expected: Partial<Omit<AnthropicError, "message">> & { message?: RegExp }) {
  const error = await promise.then(
    () => assert.fail("une erreur était attendue"),
    (e: unknown) => e,
  );
  assert.ok(error instanceof AnthropicError, `AnthropicError attendue, reçu ${inspect(error)}`);
  const { message, ...fields } = expected;
  if (message) assert.match(error.message, message);
  for (const [k, v] of Object.entries(fields)) assert.deepEqual(error[k as keyof AnthropicError], v, k);
  const dump = inspect(error, { depth: 5 }) + error.message;
  assert.ok(!dump.includes(API_KEY), "la clé ne doit jamais apparaître");
  assert.ok(!dump.includes(MARKER), "le corps brut ne doit jamais apparaître");
  return error;
}

describe("configuration", () => {
  test("clé absente ou vide : erreur avant tout appel", () => {
    for (const env of [{}, { ANTHROPIC_API_KEY: "" }, { ANTHROPIC_API_KEY: "   " }]) {
      assert.throws(() => createAnthropicClientFromEnv(env), (e: unknown) => e instanceof AnthropicError && e.stage === "configuration");
    }
  });

  test("modèle par défaut, surchargé par ANTHROPIC_MODEL (vide = défaut)", () => {
    assert.equal(createAnthropicClientFromEnv({ ANTHROPIC_API_KEY: API_KEY }).model, ANTHROPIC_DEFAULT_MODEL);
    assert.equal(createAnthropicClientFromEnv({ ANTHROPIC_API_KEY: API_KEY, ANTHROPIC_MODEL: "" }).model, ANTHROPIC_DEFAULT_MODEL);
    assert.equal(createAnthropicClientFromEnv({ ANTHROPIC_API_KEY: API_KEY, ANTHROPIC_MODEL: "claude-haiku-5-5" }).model, "claude-haiku-5-5");
  });
});

describe("requête et réponse", () => {
  test("envoie une requête Messages avec l'outil forcé, la clé uniquement dans l'en-tête", async () => {
    const { impl, calls } = fakeFetch(json(okMessage(), 200, { "request-id": "req_1" }));
    const result = await client(impl).callTool(REQUEST);

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, ANTHROPIC_ENDPOINT);
    assert.equal(calls[0].init.method, "POST");
    const headers = calls[0].init.headers as Record<string, string>;
    assert.equal(headers["x-api-key"], API_KEY);
    assert.equal(headers["anthropic-version"], ANTHROPIC_VERSION);
    assert.ok(calls[0].init.signal, "un délai maximal doit être appliqué");
    assert.deepEqual(calls[0].body, {
      model: ANTHROPIC_DEFAULT_MODEL,
      max_tokens: 512,
      system: "Consignes système",
      messages: REQUEST.messages,
      tools: [REQUEST.tool],
      tool_choice: { type: "tool", name: "persona_answer" },
    });
    assert.ok(!String(calls[0].init.body).includes(API_KEY));

    assert.deepEqual(result, {
      input: INPUT,
      model: ANTHROPIC_DEFAULT_MODEL,
      messageId: "msg_123",
      inputTokens: 900,
      outputTokens: 120,
      attempts: 1,
    });
  });

  test("ignore les blocs texte qui accompagnent l'appel d'outil", async () => {
    const content = [{ type: "text", text: "Voici" }, ...okMessage().content];
    const { impl } = fakeFetch(json(okMessage({ content })));
    assert.deepEqual((await client(impl).callTool(REQUEST)).input, INPUT);
  });
});

describe("réponses 2xx invalides : une seule tentative (appel facturé)", () => {
  const cases: [string, Record<string, unknown>, RegExp][] = [
    ["réponse tronquée", { stop_reason: "max_tokens" }, /tronquée \(max_tokens atteint\)/],
    ["refus du modèle", { stop_reason: "refusal" }, /refusé de répondre/],
    ["fin sans outil", { stop_reason: "end_turn", content: [{ type: "text", text: "Bonjour" }] }, /stop_reason « end_turn »/],
    ["aucun appel d'outil", { content: [] }, /0 appels d'outil/],
    ["deux appels d'outil", { content: [...okMessage().content, ...okMessage().content] }, /2 appels d'outil/],
    ["mauvais outil", { content: [{ type: "tool_use", id: "t", name: "autre", input: {} }] }, /outil « autre »/],
    ["entrée absente", { content: [{ type: "tool_use", id: "t", name: "persona_answer", input: "texte" }] }, /entrée de l'outil absente/],
    ["usage invalide", { usage: { input_tokens: -1, output_tokens: 3 } }, /usage/],
    ["pas un message", { type: "error" }, /objet message attendu/],
  ];
  for (const [name, overrides, message] of cases) {
    test(name, async () => {
      const { impl, calls } = fakeFetch(json(okMessage(overrides)));
      await rejects(client(impl).callTool(REQUEST), { stage: "validation", status: 200, retryable: false, message });
      assert.equal(calls.length, 1);
      assert.deepEqual(sleeps, []);
    });
  }

  test("corps vide", async () => {
    const { impl, calls } = fakeFetch(new Response("", { status: 200 }));
    await rejects(client(impl).callTool(REQUEST), { stage: "empty_body", message: /vide — statut 200/ });
    assert.equal(calls.length, 1);
  });

  test("JSON mal formé, sans recopier le corps", async () => {
    const { impl, calls } = fakeFetch(new Response(`<html>${MARKER}</html>`, { status: 200, headers: { "content-type": "text/html" } }));
    await rejects(client(impl).callTool(REQUEST), { stage: "json_parse", message: /JSON mal formé\) — statut 200, content-type text\/html/ });
    assert.equal(calls.length, 1);
  });

  test("flux interrompu pendant la lecture", async () => {
    const body = new ReadableStream({
      start(controller) {
        controller.error(new TypeError("terminated"));
      },
    });
    const { impl } = fakeFetch(new Response(body, { status: 200 }));
    await rejects(client(impl).callTool(REQUEST), { stage: "body_read", message: /flux interrompu \(TypeError\)/ });
  });
});

describe("erreurs HTTP et nouvelles tentatives", () => {
  test("429 puis succès : nouvelle tentative en respectant Retry-After", async () => {
    const { impl, calls } = fakeFetch(apiError(429, "rate_limit_error", "Trop de requêtes", { "retry-after": "2" }), json(okMessage()));
    const result = await client(impl).callTool(REQUEST);
    assert.equal(calls.length, 2);
    assert.equal(result.attempts, 2);
    assert.deepEqual(sleeps, [2000]);
  });

  test("529 (surcharge) et 500 : délai croissant puis abandon", async () => {
    const { impl, calls } = fakeFetch(
      apiError(529, "overloaded_error", "Overloaded"),
      apiError(500, "api_error", "Erreur interne"),
      apiError(529, "overloaded_error", "Overloaded"),
      apiError(529, "overloaded_error", "Overloaded", { "request-id": "req_42" }),
    );
    await rejects(client(impl).callTool(REQUEST), {
      stage: "http",
      status: 529,
      retryable: true,
      attempts: 4,
      requestId: "req_42",
      message: /529 après 4 tentatives : overloaded_error : Overloaded \(request-id req_42\)/,
    });
    assert.equal(calls.length, 4);
    assert.deepEqual(sleeps, [1000, 2000, 4000]);
  });

  for (const [status, type] of [[400, "invalid_request_error"], [401, "authentication_error"], [403, "permission_error"], [404, "not_found_error"]] as const) {
    test(`${status} : pas de nouvelle tentative, détail structuré conservé`, async () => {
      const { impl, calls } = fakeFetch(apiError(status, type, "Détail de l'API"));
      await rejects(client(impl).callTool(REQUEST), {
        stage: "http",
        status,
        retryable: false,
        message: new RegExp(`${status} : ${type} : Détail de l'API`),
      });
      assert.equal(calls.length, 1);
      assert.deepEqual(sleeps, []);
    });
  }

  test("la clé renvoyée par l'API est masquée", async () => {
    const { impl } = fakeFetch(apiError(401, "authentication_error", `invalid x-api-key ${API_KEY}`));
    const error = await rejects(client(impl).callTool(REQUEST), { stage: "http", status: 401 });
    assert.match(error.message, /\[clé masquée\]/);
  });

  test("corps d'erreur non JSON : seule sa taille est indiquée", async () => {
    const { impl } = fakeFetch(new Response(`<html>${MARKER}</html>`, { status: 502 }), new Response("", { status: 400 }));
    await rejects(client(impl, { maxRetries: 0 }).callTool(REQUEST), { status: 502, message: /502 après 1 tentatives : corps non JSON de \d+ octets/ });
  });

  test("échec réseau avant toute réponse : pas de nouvelle tentative", async () => {
    const { impl, calls } = fakeFetch(new TypeError("fetch failed"));
    await rejects(client(impl).callTool(REQUEST), { stage: "request", status: null, message: /avant toute réponse \(TypeError: fetch failed\)/ });
    assert.equal(calls.length, 1);
  });
});
