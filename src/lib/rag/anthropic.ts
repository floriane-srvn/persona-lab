export const ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages";
export const ANTHROPIC_VERSION = "2023-06-01";
/** Default generation model; `ANTHROPIC_MODEL` overrides it. */
export const ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5-5";

export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface ToolCallRequest {
  system: string;
  messages: { role: "user" | "assistant"; content: string }[];
  /** The model is forced to call this tool: its input is the structured answer. */
  tool: ToolDefinition;
  maxTokens: number;
}

export interface ToolCallResult {
  /** Tool input as sent by the model, not validated against the schema yet. */
  input: unknown;
  model: string;
  messageId: string;
  inputTokens: number;
  outputTokens: number;
  attempts: number;
}

export interface AnthropicClient {
  readonly model: string;
  callTool(request: ToolCallRequest): Promise<ToolCallResult>;
}

export interface AnthropicClientOptions {
  apiKey: string;
  model?: string;
  fetch?: typeof fetch;
  /** Retries after the first attempt, for 429 and 5xx responses only (529 = overloaded). */
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  timeoutMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

/** Same stages as the Voyage client: before any response, HTTP error, or while reading/decoding a 2xx body. */
export type AnthropicErrorStage = "configuration" | "request" | "http" | "body_read" | "empty_body" | "json_parse" | "validation";

export class AnthropicError extends Error {
  readonly status: number | null;
  readonly retryable: boolean;
  readonly attempts: number;
  readonly stage: AnthropicErrorStage;
  /** `request-id` header, safe to share with Anthropic support. */
  readonly requestId: string | null;

  constructor(
    message: string,
    options: { status?: number | null; retryable?: boolean; attempts?: number; stage?: AnthropicErrorStage; requestId?: string | null } = {},
  ) {
    super(message);
    this.name = "AnthropicError";
    this.status = options.status ?? null;
    this.retryable = options.retryable ?? false;
    this.attempts = options.attempts ?? 1;
    this.stage = options.stage ?? "configuration";
    this.requestId = options.requestId ?? null;
  }
}

const isRetryableStatus = (status: number) => status === 429 || status >= 500;
const byteLength = (text: string) => new TextEncoder().encode(text).length;
const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const isCount = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v >= 0;
const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function retryAfterMs(response: Response): number | null {
  const header = response.headers.get("retry-after");
  if (!header) return null;
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : null;
}

export function createAnthropicClient(options: AnthropicClientOptions): AnthropicClient {
  const apiKey = options.apiKey?.trim();
  if (!apiKey) throw new AnthropicError("ANTHROPIC_API_KEY manquante ou vide");
  const model = options.model?.trim() || ANTHROPIC_DEFAULT_MODEL;

  const fetchImpl = options.fetch ?? globalThis.fetch;
  const maxRetries = options.maxRetries ?? 3;
  const baseDelayMs = options.baseDelayMs ?? 1000;
  const maxDelayMs = options.maxDelayMs ?? 30_000;
  const timeoutMs = options.timeoutMs ?? 60_000;
  const sleep = options.sleep ?? defaultSleep;

  // Last line of defence: no error message may carry the key, even if the API echoes it back.
  const redact = (text: string) => text.split(apiKey).join("[clé masquée]");
  const fail = (message: string, extra?: ConstructorParameters<typeof AnthropicError>[1]) =>
    new AnthropicError(redact(message), extra);

  /** Safe technical facts about a response, never its content. */
  function describe(response: Response, receivedBytes?: number): string {
    const parts = [
      `statut ${response.status}`,
      `content-type ${response.headers.get("content-type") ?? "absent"}`,
      `content-length ${response.headers.get("content-length") ?? "absent"}`,
    ];
    if (receivedBytes !== undefined) parts.push(`${receivedBytes} octets reçus`);
    return parts.join(", ");
  }

  /** Only the structured `error.type` and `error.message` are kept; a raw (non-JSON) body is never echoed. */
  async function errorDetail(response: Response): Promise<string> {
    let body: string;
    try {
      body = await response.text();
    } catch {
      return "corps illisible";
    }
    try {
      const error = (JSON.parse(body) as { error?: { type?: unknown; message?: unknown } }).error;
      if (typeof error?.message === "string") {
        const type = typeof error.type === "string" ? `${error.type} : ` : "";
        return `${type}${error.message}`.replace(/\s+/g, " ").slice(0, 300);
      }
    } catch {
      // Not JSON: described below without its content.
    }
    return `corps non JSON de ${byteLength(body)} octets`;
  }

  async function readJson(response: Response, info: { status: number; attempts: number; requestId: string | null }) {
    let text: string;
    try {
      text = await response.text();
    } catch (error) {
      const name = error instanceof Error ? error.name : "erreur inconnue";
      const cause = name === "TimeoutError" ? `délai de ${timeoutMs} ms dépassé` : `flux interrompu (${name})`;
      throw fail(`Réponse Anthropic interrompue pendant la lecture du corps : ${cause} — ${describe(response)}`, {
        ...info,
        stage: "body_read",
      });
    }
    const bytes = byteLength(text);
    if (text.trim() === "") throw fail(`Réponse Anthropic vide — ${describe(response, bytes)}`, { ...info, stage: "empty_body" });
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw fail(`Réponse Anthropic illisible (JSON mal formé) — ${describe(response, bytes)}`, { ...info, stage: "json_parse" });
    }
  }

  function parseMessage(json: unknown, toolName: string, info: { status: number; attempts: number; requestId: string | null }) {
    const invalid = (message: string) => fail(`Réponse Anthropic invalide : ${message}`, { ...info, stage: "validation" });
    if (!isObject(json) || json.type !== "message") throw invalid("objet message attendu");
    const { id, content, stop_reason: stopReason, usage } = json;
    if (!Array.isArray(content)) throw invalid("champ content absent");
    if (stopReason === "max_tokens") throw invalid("réponse tronquée (max_tokens atteint)");
    if (stopReason === "refusal") throw invalid("le modèle a refusé de répondre");
    if (stopReason !== "tool_use") throw invalid(`stop_reason « ${String(stopReason)} » au lieu de « tool_use »`);

    const calls = content.filter((block) => isObject(block) && block.type === "tool_use");
    if (calls.length !== 1) throw invalid(`${calls.length} appels d'outil au lieu d'un`);
    const call = calls[0] as Record<string, unknown>;
    if (call.name !== toolName) throw invalid(`outil « ${String(call.name)} » au lieu de « ${toolName} »`);
    if (!isObject(call.input)) throw invalid("entrée de l'outil absente");
    if (!isObject(usage) || !isCount(usage.input_tokens) || !isCount(usage.output_tokens)) {
      throw invalid("usage.input_tokens ou usage.output_tokens absent ou invalide");
    }
    return {
      input: call.input,
      model: typeof json.model === "string" ? json.model : model,
      messageId: typeof id === "string" ? id : "",
      inputTokens: usage.input_tokens,
      outputTokens: usage.output_tokens,
      attempts: info.attempts,
    };
  }

  return {
    model,
    async callTool(request) {
      const body = JSON.stringify({
        model,
        max_tokens: request.maxTokens,
        system: request.system,
        messages: request.messages,
        tools: [request.tool],
        tool_choice: { type: "tool", name: request.tool.name },
      });

      for (let attempt = 1; ; attempt++) {
        let response: Response;
        try {
          response = await fetchImpl(ANTHROPIC_ENDPOINT, {
            method: "POST",
            headers: { "x-api-key": apiKey, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json" },
            body,
            signal: AbortSignal.timeout(timeoutMs),
          });
        } catch (error) {
          // Network failures and timeouts are not retried: the request may already have been billed.
          const reason = error instanceof Error ? `${error.name}: ${error.message}` : "erreur inconnue";
          throw fail(`Échec de l'appel à Anthropic avant toute réponse (${reason})`, { attempts: attempt, stage: "request" });
        }

        const info = { status: response.status, attempts: attempt, requestId: response.headers.get("request-id") };
        // A 2xx means the generation ran (and was billed): failures past this point are not retried.
        if (response.ok) return parseMessage(await readJson(response, info), request.tool.name, info);

        const retryable = isRetryableStatus(response.status);
        const detail = await errorDetail(response);
        if (!retryable || attempt > maxRetries) {
          const suffix = retryable ? ` après ${attempt} tentatives` : "";
          const id = info.requestId ? ` (request-id ${info.requestId})` : "";
          throw fail(`Anthropic a répondu ${response.status}${suffix} : ${detail}${id}`, { ...info, retryable, stage: "http" });
        }
        const backoff = Math.min(baseDelayMs * 2 ** (attempt - 1), maxDelayMs);
        await sleep(Math.min(retryAfterMs(response) ?? backoff, maxDelayMs));
      }
    },
  };
}

/** Builds a client from `ANTHROPIC_API_KEY` and optional `ANTHROPIC_MODEL`. Server-side only. */
export function createAnthropicClientFromEnv(
  env: Record<string, string | undefined> = process.env,
  options: Omit<AnthropicClientOptions, "apiKey" | "model"> = {},
): AnthropicClient {
  return createAnthropicClient({ ...options, apiKey: env.ANTHROPIC_API_KEY ?? "", model: env.ANTHROPIC_MODEL });
}
