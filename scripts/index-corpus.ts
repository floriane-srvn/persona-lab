// Usage : npm run index -- [persona] [--dry-run] [--force] [--max-tokens=N]
import { chunkPersona } from "../src/lib/chunking/pipeline.ts";
import { PERSONAS } from "../src/lib/chunking/personas.ts";
import { DEFAULT_MAX_ESTIMATED_TOKENS, formatIndexReport, indexPersona } from "../src/lib/rag/indexing.ts";
import { createSupabaseAdminClient, createSupabaseChunkStore } from "../src/lib/rag/supabase-store.ts";
import { createVoyageClientFromEnv } from "../src/lib/rag/voyage.ts";

const args = process.argv.slice(2);
const flag = (name: string) => args.includes(`--${name}`);
const option = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];

const known = /^--(dry-run|force|max-tokens=\d+)$/;
const unknown = args.filter((a) => a.startsWith("--") && !known.test(a));
if (unknown.length) {
  console.error(`Option(s) inconnue(s) : ${unknown.join(", ")}`);
  process.exit(1);
}

const personaId = args.find((a) => !a.startsWith("--")) ?? "julie";
const config = PERSONAS[personaId];
if (!config) {
  console.error(`Persona inconnue : ${personaId}. Disponibles : ${Object.keys(PERSONAS).join(", ")}`);
  process.exit(1);
}

try {
  const report = await indexPersona({
    result: chunkPersona(config, process.cwd()),
    store: createSupabaseChunkStore(createSupabaseAdminClient()),
    // Only built if something must be embedded: a dry-run or an unchanged corpus needs no Voyage key.
    getEmbedder: () => createVoyageClientFromEnv(),
    dryRun: flag("dry-run"),
    force: flag("force"),
    maxEstimatedTokens: Number(option("max-tokens") ?? DEFAULT_MAX_ESTIMATED_TOKENS),
  });
  console.log(formatIndexReport(report));
} catch (error) {
  console.error(`Indexation interrompue — ${error instanceof Error ? `${error.name} : ${error.message}` : "erreur inconnue"}`);
  process.exit(1);
}
