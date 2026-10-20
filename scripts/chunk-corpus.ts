// Usage : npm run chunk [-- <persona>] [-- --json]
import { chunkPersona } from "../src/lib/chunking/pipeline.ts";
import { PERSONAS } from "../src/lib/chunking/personas.ts";
import { formatReport } from "../src/lib/chunking/report.ts";

const args = process.argv.slice(2);
const personaId = args.find((a) => !a.startsWith("--")) ?? "julie";
const config = PERSONAS[personaId];
if (!config) {
  console.error(`Persona inconnue : ${personaId}. Disponibles : ${Object.keys(PERSONAS).join(", ")}`);
  process.exit(1);
}

const result = chunkPersona(config, process.cwd());
console.log(args.includes("--json") ? JSON.stringify(result.chunks, null, 2) : formatReport(result));
if (result.anomalies.some((a) => a.severity === "error")) process.exit(1);
