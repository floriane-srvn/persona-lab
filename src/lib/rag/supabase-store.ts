import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";

import type { ChunkRow, ChunkRowWithEmbedding, ChunkStore } from "./indexing.ts";

export const CHUNKS_TABLE = "persona_chunks";
const PAGE_SIZE = 500;
const WRITE_BATCH_SIZE = 100;

export class SupabaseStoreError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SupabaseStoreError";
  }
}

const fail = (action: string, error: { message: string; code?: string }) =>
  new SupabaseStoreError(`Supabase (${action}) : ${error.message}${error.code ? ` [${error.code}]` : ""}`);

/**
 * Server-side admin client (secret key, bypasses RLS). Never import from client components.
 * `fetch` can be injected for offline tests.
 */
export function createSupabaseAdminClient(
  env: Record<string, string | undefined> = process.env,
  options: { fetch?: typeof fetch } = {},
): SupabaseClient {
  const url = env.SUPABASE_URL?.trim();
  const key = env.SUPABASE_SECRET_KEY?.trim();
  if (!url) throw new SupabaseStoreError("SUPABASE_URL manquante ou vide");
  if (!key) throw new SupabaseStoreError("SUPABASE_SECRET_KEY manquante ou vide");
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
    global: options.fetch ? { fetch: options.fetch } : undefined,
  });
}

export function createSupabaseChunkStore(client: SupabaseClient): ChunkStore {
  return {
    async listByPersona(personaId) {
      const rows: ChunkRow[] = [];
      for (let from = 0; ; from += PAGE_SIZE) {
        // The embedding column is deliberately not read: comparison relies on embeddingInputHash.
        const { data, error } = await client
          .from(CHUNKS_TABLE)
          .select("persona_id, id, content, metadata")
          .eq("persona_id", personaId)
          .order("id")
          .range(from, from + PAGE_SIZE - 1);
        if (error) throw fail("lecture", error);
        rows.push(...(data as ChunkRow[]));
        if (data.length < PAGE_SIZE) return rows;
      }
    },

    async upsert(rows: ChunkRowWithEmbedding[]) {
      for (let i = 0; i < rows.length; i += WRITE_BATCH_SIZE) {
        const { error } = await client
          .from(CHUNKS_TABLE)
          .upsert(rows.slice(i, i + WRITE_BATCH_SIZE), { onConflict: "persona_id,id" });
        if (error) throw fail("upsert", error);
      }
    },

    async updateWithoutEmbedding(rows) {
      // An upsert must carry the NOT NULL embedding; a plain UPDATE keeps the stored vector.
      for (const row of rows) {
        const { data, error } = await client
          .from(CHUNKS_TABLE)
          .update({ content: row.content, metadata: row.metadata })
          .eq("persona_id", row.persona_id)
          .eq("id", row.id)
          .select("id");
        if (error) throw fail("mise à jour", error);
        if (!data?.length) throw new SupabaseStoreError(`mise à jour : ligne ${row.persona_id}/${row.id} introuvable`);
      }
    },
  };
}
