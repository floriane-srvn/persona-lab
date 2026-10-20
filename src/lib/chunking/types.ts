export const CORPUS_VERSION = "v1";

export type SourceType = "profile" | "interview" | "journal" | "research_notes";

/** How much a source can be trusted as a statement of fact about the persona. */
export type EvidenceStatus = "declared" | "self_reported" | "interpretation";

export type ChunkStrategy = "paragraphs" | "qa_pairs" | "journal_weeks" | "sections";

export interface SourceConfig {
  sourceType: SourceType;
  evidenceStatus: EvidenceStatus;
  strategy: ChunkStrategy;
  /**
   * Warnings written inside regular content rather than in a labelled paragraph.
   * The text from `startsWith` to the end of its line is attached to every chunk of the document
   * (it also stays in its own content chunk).
   */
  inlineCaveats?: { label: string; startsWith: string }[];
  /**
   * Interview chunks (by suffix, e.g. "q4") whose question only makes sense with the previous one.
   * The previous question is quoted in the embedding text; the chunk text is unchanged.
   */
  previousQuestionContext?: string[];
}

export interface PersonaCorpusConfig {
  personaId: string;
  /** Repo-relative directory, stored in chunk metadata for traceability. */
  sourceDir: string;
  /** Indexed sources keyed by stable id (D01, D02…). Any other file is ignored. */
  sources: Record<string, SourceConfig>;
}

export interface SourceFile {
  fileName: string;
  content: string;
}

export interface Line {
  /** 1-based line number in the source file. */
  n: number;
  text: string;
}

export interface Caveat {
  label: string;
  text: string;
  lineStart: number;
  lineEnd: number;
}

export interface SourceDocument {
  sourceId: string;
  sourceFile: string;
  docTitle: string;
  sourceType: SourceType;
  evidenceStatus: EvidenceStatus;
  statusLabel: string;
  sourceDescription: string;
  sourceDateStart: string | null;
  sourceDateEnd: string | null;
  durationMin: number | null;
  caveats: Caveat[];
}

export interface ChunkMetadata {
  // Identity
  personaId: string;
  sourceId: string;
  sourceFile: string;
  corpusVersion: typeof CORPUS_VERSION;
  contentHash: string;
  // Provenance
  docTitle: string;
  sourceType: SourceType;
  sourceDescription: string;
  sourceDateStart: string | null;
  sourceDateEnd: string | null;
  durationMin: number | null;
  // Epistemic status
  evidenceStatus: EvidenceStatus;
  statusLabel: string;
  containsVerbatim: boolean;
  // Traceability
  chunkIndex: number;
  sectionPath: string | null;
  lineStart: number;
  lineEnd: number;
  // Source-specific
  speakers: string[];
  question: string | null;
  week: number | null;
  grocerySpendEur: number | null;
  citedSources: string[];
  /** Chunks whose content was quoted as context in the embedding text. */
  contextChunkIds: string[];
  // Document-level warnings, attached to every chunk of the document
  caveats: Caveat[];
}

export interface Chunk {
  id: string;
  /** Exact source lines covered by the chunk. */
  text: string;
  /** Contextual header + cleaned text, the string meant to be embedded. */
  embeddingText: string;
  approxTokens: number;
  metadata: ChunkMetadata;
}

export interface Anomaly {
  severity: "error" | "warning";
  sourceFile: string;
  chunkId?: string;
  message: string;
}

export interface ChunkingResult {
  personaId: string;
  corpusVersion: typeof CORPUS_VERSION;
  sourceDir: string;
  documents: SourceDocument[];
  chunks: Chunk[];
  ignoredFiles: string[];
  anomalies: Anomaly[];
}
