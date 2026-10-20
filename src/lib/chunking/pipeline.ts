import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import {
  extractCaveats,
  parseDurationMin,
  parseFrenchDateRange,
  parseHeader,
  splitBlocks,
  stripBold,
  toLines,
} from "./markdown.ts";
import { STRATEGIES } from "./strategies.ts";
import type { ChunkDraft } from "./strategies.ts";
import { CORPUS_VERSION } from "./types.ts";
import type {
  Anomaly,
  Caveat,
  Chunk,
  ChunkingResult,
  Line,
  PersonaCorpusConfig,
  SourceDocument,
  SourceFile,
} from "./types.ts";

const SOURCE_FILE_RE = /^(D\d{2})_[\w-]+\.md$/;
const MIN_TOKENS = 25;
const MAX_TOKENS = 500;

/** Rough estimate (~4 characters per token for French prose); good enough for size checks. */
export const approxTokens = (text: string) => Math.ceil(text.length / 4);

export function loadSourceFiles(dir: string): SourceFile[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .sort()
    .map((fileName) => ({ fileName, content: readFileSync(path.join(dir, fileName), "utf8") }));
}

/** Keeps blank-line separation between non-contiguous source lines. */
function linesToText(lines: Line[]): string {
  return lines
    .map((l, i) => (i > 0 && l.n > lines[i - 1].n + 1 ? `\n${l.text}` : l.text))
    .join("\n");
}

function embeddingHeader(doc: SourceDocument, sectionPath: string | null): string {
  const date = doc.sourceDateStart === doc.sourceDateEnd
    ? doc.sourceDateStart
    : `${doc.sourceDateStart} → ${doc.sourceDateEnd}`;
  const head = `[${[doc.sourceId, doc.docTitle, date, doc.statusLabel].filter(Boolean).join(" · ")}]`;
  return sectionPath ? `${head}\nSection : ${sectionPath}` : head;
}

interface EmbeddingContext {
  chunkId: string;
  text: string;
}

function buildChunk(
  doc: SourceDocument,
  personaId: string,
  draft: ChunkDraft,
  index: number,
  context?: EmbeddingContext,
): Chunk {
  const text = linesToText(draft.lines);
  // The heading is already carried by the "Section :" line of the header.
  const bodyLines = draft.sectionPath && draft.lines[0].text.startsWith("#") ? draft.lines.slice(1) : draft.lines;
  const embeddingText = [
    embeddingHeader(doc, draft.sectionPath),
    context && `Contexte : cette question fait suite à la question « ${context.text} » (${context.chunkId}).`,
    stripBold(linesToText(bodyLines)),
  ].filter(Boolean).join("\n");
  const citedSources = [...new Set(text.match(/\bD\d{2}\b/g) ?? [])].filter((id) => id !== doc.sourceId).sort();
  return {
    id: `${doc.sourceId}-${draft.suffix}`,
    text,
    embeddingText,
    approxTokens: approxTokens(embeddingText),
    metadata: {
      personaId,
      sourceId: doc.sourceId,
      sourceFile: doc.sourceFile,
      corpusVersion: CORPUS_VERSION,
      contentHash: createHash("sha256").update(text).digest("hex").slice(0, 16),
      docTitle: doc.docTitle,
      sourceType: doc.sourceType,
      sourceDescription: doc.sourceDescription,
      sourceDateStart: doc.sourceDateStart,
      sourceDateEnd: doc.sourceDateEnd,
      durationMin: doc.durationMin,
      evidenceStatus: doc.evidenceStatus,
      statusLabel: doc.statusLabel,
      // Quotes inside researcher notes are paraphrases or counter-examples, not verbatims.
      containsVerbatim: doc.evidenceStatus !== "interpretation" && /«[^»]+»/.test(text),
      chunkIndex: index,
      sectionPath: draft.sectionPath,
      lineStart: draft.lines[0].n,
      lineEnd: draft.lines[draft.lines.length - 1].n,
      speakers: draft.speakers ?? [],
      question: draft.question ?? null,
      week: draft.week ?? null,
      grocerySpendEur: draft.grocerySpendEur ?? null,
      citedSources,
      contextChunkIds: context ? [context.chunkId] : [],
      caveats: doc.caveats,
    },
  };
}

export function chunkCorpus(config: PersonaCorpusConfig, files: SourceFile[]): ChunkingResult {
  const documents: SourceDocument[] = [];
  const chunks: Chunk[] = [];
  const ignoredFiles: string[] = [];
  const anomalies: Anomaly[] = [];
  const seen = new Set<string>();

  for (const file of files) {
    const sourceFile = `${config.sourceDir}/${file.fileName}`;
    const report = (severity: Anomaly["severity"], message: string, chunkId?: string) =>
      anomalies.push({ severity, sourceFile, chunkId, message });

    const idFromName = file.fileName.match(SOURCE_FILE_RE)?.[1];
    if (!idFromName) {
      ignoredFiles.push(file.fileName);
      if (file.fileName.toLowerCase() !== "readme.md") report("warning", "fichier ignoré : nom hors convention Dxx_nom.md");
      continue;
    }
    const sourceConfig = config.sources[idFromName];
    if (!sourceConfig) {
      ignoredFiles.push(file.fileName);
      report("error", `identifiant ${idFromName} absent du registre des sources`);
      continue;
    }

    const lines = toLines(file.content);
    const header = parseHeader(lines);
    if (typeof header === "string") {
      report("error", header);
      continue;
    }
    if (header.sourceId !== idFromName) {
      report("error", `identifiant du titre (${header.sourceId}) différent du nom de fichier (${idFromName})`);
      continue;
    }
    seen.add(idFromName);

    const dates = parseFrenchDateRange(header.description);
    if (!dates) report("warning", "date de la source non reconnue");

    const { caveats, content } = extractCaveats(splitBlocks(header.body));
    const inlineCaveats: Caveat[] = [];
    for (const { label, startsWith } of sourceConfig.inlineCaveats ?? []) {
      const line = header.body.find((l) => l.text.includes(startsWith));
      if (!line) {
        report("error", `avertissement en ligne introuvable : « ${startsWith}… »`);
        continue;
      }
      const text = stripBold(line.text.slice(line.text.indexOf(startsWith))).trim();
      inlineCaveats.push({ label, text, lineStart: line.n, lineEnd: line.n });
    }
    if (caveats.length + inlineCaveats.length === 0) {
      report("warning", "aucun avertissement (Attention, Limite…) : seul le statut est rattaché aux chunks");
    }

    const doc: SourceDocument = {
      sourceId: idFromName,
      sourceFile,
      docTitle: header.title,
      sourceType: sourceConfig.sourceType,
      evidenceStatus: sourceConfig.evidenceStatus,
      statusLabel: header.statusLabel,
      sourceDescription: header.description,
      sourceDateStart: dates?.start ?? null,
      sourceDateEnd: dates?.end ?? null,
      durationMin: parseDurationMin(header.description),
      // The status line is itself a warning ("non vérifiées"…) and always travels with the chunks.
      caveats: [
        { label: "Statut", text: header.statusLabel, lineStart: header.statusLine.n, lineEnd: header.statusLine.n },
        ...caveats,
        ...inlineCaveats,
      ],
    };
    documents.push(doc);

    const { drafts, issues } = STRATEGIES[sourceConfig.strategy](content);
    for (const issue of issues) {
      report(issue.severity, issue.message, issue.suffix && `${idFromName}-${issue.suffix}`);
    }
    if (drafts.length === 0) report("error", "aucun chunk produit");

    const contexts = new Map<string, EmbeddingContext>();
    for (const suffix of sourceConfig.previousQuestionContext ?? []) {
      const i = drafts.findIndex((d) => d.suffix === suffix);
      const previous = drafts.slice(0, Math.max(i, 0)).findLast((d) => d.question);
      if (i === -1 || !drafts[i].question || !previous) {
        report("error", `contexte de question précédente impossible pour ${idFromName}-${suffix}`);
        continue;
      }
      contexts.set(suffix, { chunkId: `${idFromName}-${previous.suffix}`, text: previous.question! });
    }

    const docChunks = drafts.map((draft, i) => buildChunk(doc, config.personaId, draft, i, contexts.get(draft.suffix)));
    chunks.push(...docChunks);

    // Every non-empty body line must land in exactly one chunk or caveat.
    const covered = new Map<number, number>();
    const mark = (from: number, to: number) => {
      for (let n = from; n <= to; n++) covered.set(n, (covered.get(n) ?? 0) + 1);
    };
    for (const d of drafts) for (const l of d.lines) mark(l.n, l.n);
    for (const c of caveats) mark(c.lineStart, c.lineEnd);
    for (const line of header.body) {
      if (line.text.trim() === "") continue;
      const count = covered.get(line.n) ?? 0;
      if (count === 0) report("error", `ligne ${line.n} absente de tout chunk`);
      if (count > 1) report("error", `ligne ${line.n} présente dans ${count} chunks`);
    }

    for (const chunk of docChunks) {
      if (chunk.approxTokens < MIN_TOKENS) report("warning", `chunk très court (≈${chunk.approxTokens} tokens)`, chunk.id);
      if (chunk.approxTokens > MAX_TOKENS) report("warning", `chunk long (≈${chunk.approxTokens} tokens)`, chunk.id);
    }
  }

  for (const id of Object.keys(config.sources)) {
    if (!seen.has(id)) {
      anomalies.push({ severity: "error", sourceFile: config.sourceDir, message: `source ${id} attendue mais absente ou illisible` });
    }
  }
  const ids = chunks.map((c) => c.id);
  for (const id of new Set(ids.filter((id, i) => ids.indexOf(id) !== i))) {
    anomalies.push({ severity: "error", sourceFile: config.sourceDir, chunkId: id, message: "identifiant de chunk dupliqué" });
  }

  return {
    personaId: config.personaId,
    corpusVersion: CORPUS_VERSION,
    sourceDir: config.sourceDir,
    documents,
    chunks,
    ignoredFiles,
    anomalies,
  };
}

/** Reads `config.sourceDir` relative to `rootDir` and chunks it. */
export function chunkPersona(config: PersonaCorpusConfig, rootDir: string): ChunkingResult {
  return chunkCorpus(config, loadSourceFiles(path.join(rootDir, config.sourceDir)));
}
