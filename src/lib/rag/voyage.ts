export const VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings";
export const VOYAGE_MODEL = "voyage-4";
export const VOYAGE_DIMENSIONS = 1024;
export const VOYAGE_OUTPUT_DTYPE = "float";
/** API limit: at most 1,000 inputs per request. */
export const VOYAGE_MAX_BATCH_SIZE = 1000;
/** API limit for voyage-4 is 320K tokens per request; we stay well below it. */
export const VOYAGE_MAX_BATCH_TOKENS = 300_000;

export type VoyageInputType = "document" | "query";

export interface VoyageClientOptions {
  apiKey: string;
  fetch?: typeof fetch;
  /** Retries after the first attempt, for 429 and 5xx responses only. */
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  timeoutMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

export interface EmbeddingResult {
  embeddings: number[][];
  /** Tokens billed by Voyage, summed over all requests. */
  totalTokens: number;
  requests: number;
}

export interface VoyageClient {
  embed(texts: string[], inputType: VoyageInputType): Promise<EmbeddingResult>;
}

/** Where a call failed: before any response, on an HTTP error, or while reading/decoding a 2xx body. */
export type VoyageErrorStage = "configuration" | "request" | "http" | "body_read" | "empty_body" | "json_parse" | "validation";

export class VoyageError extends Error {
  readonly status: number | null;
  readonly retryable: boolean;
  readonly attempts: number;
  readonly stage: VoyageErrorStage;

  constructor(
    message: string,
    options: { status?: number | null; retryable?: boolean; attempts?: number; stage?: VoyageErrorStage } = {},
  ) {
    super(message);
    this.name = "VoyageError";
    this.status = options.status ?? null;
    this.retryable = options.retryable ?? false;
    this.attempts = options.attempts ?? 1;
    this.stage = options.stage ?? "configuration";
  }
}

/**
 * Deliberately conservative (~2 characters per token) so that batches never exceed the API limit.
 * Not used for billing: the real count comes from `usage.total_tokens`.
 */
const conservativeTokens = (text: string) => Math.ceil(text.length / 2);

export function splitBatches(texts: string[]): string[][] {
  const batches: string[][] = [];
  let current: string[] = [];
  let tokens = 0;
  for (const text of texts) {
    const t = conservativeTokens(text);
    if (current.length && (current.length >= VOYAGE_MAX_BATCH_SIZE || tokens + t > VOYAGE_MAX_BATCH_TOKENS)) {
      batches.push(current);
      current = [];
      tokens = 0;
    }
    current.push(text);
    tokens += t;
  }
  if (current.length) batches.push(current);
  return batches;
}

const isRetryableStatus = (status: number) => status === 429 || status >= 500;

const byteLength = (text: string) => new TextEncoder().encode(text).length;

/** Describes why a body is not JSON from its shape only (first character, declared vs received size). */
function classifyUnparsable(text: string, response: Response, bytes: number): string {
  const first = text.trimStart()[0];
  if (first === "<") return "contenu HTML ou XML au lieu de JSON";
  if (first !== "{" && first !== "[") return "contenu non JSON";
  const declared = Number(response.headers.get("content-length"));
  // Content-Length is the compressed size when the body is encoded: only compare uncompressed bodies.
  if (!response.headers.get("content-encoding") && Number.isFinite(declared) && declared > bytes) {
    return `JSON tronqué (${bytes} octets reçus sur ${declared} annoncés)`;
  }
  return "JSON incomplet ou mal formé";
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function retryAfterMs(response: Response): number | null {
  const header = response.headers.get("retry-after");
  if (!header) return null;
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : null;
}

export function createVoyageClient(options: VoyageClientOptions): VoyageClient {
  const apiKey = options.apiKey?.trim();
  if (!apiKey) throw new VoyageError("VOYAGE_API_KEY manquante ou vide");

  const fetchImpl = options.fetch ?? globalThis.fetch;
  const maxRetries = options.maxRetries ?? 3;
  const baseDelayMs = options.baseDelayMs ?? 500;
  const maxDelayMs = options.maxDelayMs ?? 30_000;
  const timeoutMs = options.timeoutMs ?? 30_000;
  const sleep = options.sleep ?? defaultSleep;

  // Last line of defence: no error message may carry the key, even if the API echoes it back.
  const redact = (text: string) => text.split(apiKey).join("[clé masquée]");
  const fail = (message: string, extra?: ConstructorParameters<typeof VoyageError>[1]) =>
    new VoyageError(redact(message), extra);

  /** Safe technical facts about a response, never its content. */
  function describe(response: Response, receivedBytes?: number): string {
    const parts = [
      `statut ${response.status}`,
      `content-type ${response.headers.get("content-type") ?? "absent"}`,
      `content-length ${response.headers.get("content-length") ?? "absent"}`,
    ];
    const encoding = response.headers.get("content-encoding");
    if (encoding) parts.push(`content-encoding ${encoding}`);
    if (receivedBytes !== undefined) parts.push(`${receivedBytes} octets reçus`);
    return parts.join(", ");
  }

  /** Only Voyage's structured `detail` field is kept; a raw (non-JSON) body is never echoed. */
  async function errorDetail(response: Response): Promise<string> {
    let body: string;
    try {
      body = await response.text();
    } catch {
      return "corps illisible";
    }
    try {
      const detail = (JSON.parse(body) as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail.replace(/\s+/g, " ").slice(0, 300);
    } catch {
      // Not JSON: described below without its content.
    }
    return `corps non JSON de ${byteLength(body)} octets`;
  }

  /** Reads a 2xx body and tells apart interrupted reads, empty bodies and malformed JSON. */
  async function readJson(response: Response, attempt: number): Promise<unknown> {
    const info = { status: response.status, attempts: attempt };
    let text: string;
    try {
      text = await response.text();
    } catch (error) {
      const name = error instanceof Error ? error.name : "erreur inconnue";
      const cause = name === "TimeoutError" ? `délai de ${timeoutMs} ms dépassé` : `flux interrompu (${name})`;
      throw fail(`Réponse Voyage AI interrompue pendant la lecture du corps : ${cause} — ${describe(response)}`, {
        ...info,
        stage: "body_read",
      });
    }

    const bytes = byteLength(text);
    if (text.trim() === "") {
      throw fail(`Réponse Voyage AI vide — ${describe(response, bytes)}`, { ...info, stage: "empty_body" });
    }
    try {
      return JSON.parse(text);
    } catch {
      throw fail(`Réponse Voyage AI illisible : ${classifyUnparsable(text, response, bytes)} — ${describe(response, bytes)}`, {
        ...info,
        stage: "json_parse",
      });
    }
  }

  async function post(batch: string[], inputType: VoyageInputType): Promise<unknown> {
    const body = JSON.stringify({
      input: batch,
      model: VOYAGE_MODEL,
      input_type: inputType,
      output_dimension: VOYAGE_DIMENSIONS,
      output_dtype: VOYAGE_OUTPUT_DTYPE,
      truncation: false,
    });

    for (let attempt = 1; ; attempt++) {
      let response: Response;
      try {
        response = await fetchImpl(VOYAGE_ENDPOINT, {
          method: "POST",
          headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
          body,
          signal: AbortSignal.timeout(timeoutMs),
        });
      } catch (error) {
        // Network failures and timeouts are not retried: the request may already have been billed.
        const reason = error instanceof Error ? `${error.name}: ${error.message}` : "erreur inconnue";
        throw fail(`Échec de l'appel à Voyage AI avant toute réponse (${reason})`, { attempts: attempt, stage: "request" });
      }

      // A 2xx means Voyage processed (and likely billed) the request: body failures are not retried.
      if (response.ok) return readJson(response, attempt);

      const retryable = isRetryableStatus(response.status);
      const detail = await errorDetail(response);
      if (!retryable || attempt > maxRetries) {
        const suffix = retryable ? ` après ${attempt} tentatives` : "";
        throw fail(`Voyage AI a répondu ${response.status}${suffix}${detail ? ` : ${detail}` : ""}`, {
          status: response.status,
          retryable,
          attempts: attempt,
          stage: "http",
        });
      }
      const backoff = Math.min(baseDelayMs * 2 ** (attempt - 1), maxDelayMs);
      await sleep(Math.min(retryAfterMs(response) ?? backoff, maxDelayMs));
    }
  }

  function parseResponse(json: unknown, expected: number): { embeddings: number[][]; totalTokens: number } {
    const invalid = (message: string) => fail(`Réponse Voyage AI invalide : ${message}`, { stage: "validation" });
    if (!json || typeof json !== "object") throw invalid("corps vide");
    const { data, model, usage } = json as { data?: unknown; model?: unknown; usage?: { total_tokens?: unknown } };

    if (model !== VOYAGE_MODEL) throw invalid(`modèle « ${String(model)} » au lieu de « ${VOYAGE_MODEL} »`);
    if (!Array.isArray(data)) throw invalid("champ data absent");
    if (data.length !== expected) throw invalid(`${data.length} vecteurs reçus pour ${expected} textes`);
    const totalTokens = usage?.total_tokens;
    if (typeof totalTokens !== "number" || !Number.isInteger(totalTokens) || totalTokens < 0) {
      throw invalid("usage.total_tokens absent ou invalide");
    }

    const embeddings: number[][] = new Array(expected);
    for (const item of data as { index?: unknown; embedding?: unknown }[]) {
      const { index, embedding } = item ?? {};
      if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= expected) {
        throw invalid(`index ${String(index)} hors limites`);
      }
      if (embeddings[index]) throw invalid(`index ${index} dupliqué`);
      if (!Array.isArray(embedding) || embedding.length !== VOYAGE_DIMENSIONS) {
        const size = Array.isArray(embedding) ? embedding.length : "aucune";
        throw invalid(`vecteur ${index} de dimension ${size} au lieu de ${VOYAGE_DIMENSIONS}`);
      }
      if (!embedding.every((v) => typeof v === "number" && Number.isFinite(v))) {
        throw invalid(`vecteur ${index} contient des valeurs non numériques`);
      }
      embeddings[index] = embedding as number[];
    }
    return { embeddings, totalTokens };
  }

  return {
    async embed(texts, inputType) {
      if (inputType !== "document" && inputType !== "query") {
        throw new VoyageError(`input_type invalide : ${String(inputType)}`);
      }
      const empty = texts.findIndex((t) => typeof t !== "string" || t.trim() === "");
      if (empty !== -1) throw new VoyageError(`texte ${empty} vide : Voyage AI refuse les entrées vides`);
      if (texts.length === 0) return { embeddings: [], totalTokens: 0, requests: 0 };

      const result: EmbeddingResult = { embeddings: [], totalTokens: 0, requests: 0 };
      for (const batch of splitBatches(texts)) {
        const { embeddings, totalTokens } = parseResponse(await post(batch, inputType), batch.length);
        result.embeddings.push(...embeddings);
        result.totalTokens += totalTokens;
        result.requests += 1;
      }
      return result;
    },
  };
}

/** Builds a client from `VOYAGE_API_KEY`. Server-side only: never import from client components. */
export function createVoyageClientFromEnv(
  env: Record<string, string | undefined> = process.env,
  options: Omit<VoyageClientOptions, "apiKey"> = {},
): VoyageClient {
  return createVoyageClient({ ...options, apiKey: env.VOYAGE_API_KEY ?? "" });
}
