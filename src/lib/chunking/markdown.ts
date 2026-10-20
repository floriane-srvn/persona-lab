import type { Caveat, Line } from "./types.ts";

export function toLines(content: string): Line[] {
  return content
    .replace(/^﻿/, "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((text, i) => ({ n: i + 1, text: text.trimEnd() }));
}

/** Groups consecutive non-empty lines; blank lines separate blocks and headings are blocks of their own. */
export function splitBlocks(lines: Line[]): Line[][] {
  const blocks: Line[][] = [];
  let current: Line[] = [];
  for (const line of lines) {
    if (line.text.trim() === "") {
      if (current.length) blocks.push(current);
      current = [];
    } else if (line.text.startsWith("#")) {
      if (current.length) blocks.push(current);
      blocks.push([line]);
      current = [];
    } else {
      current.push(line);
    }
  }
  if (current.length) blocks.push(current);
  return blocks;
}

export const stripBold = (s: string) => s.replace(/\*\*/g, "");

export const blockText = (block: Line[]) => block.map((l) => l.text).join("\n");

export const isHeading = (block: Line[], level = 2) =>
  block[0].text.startsWith("#".repeat(level) + " ");

export const isTable = (block: Line[]) => block.every((l) => l.text.startsWith("|"));

const TITLE_RE = /^#\s+(D\d{2})\s+[—–-]\s+(.+)$/;
const SOURCE_LINE_RE = /^\*\*Source fictive\*\*\s*:\s*(.+?)\s*\*\*Statut\*\*\s*:\s*(.+)$/;

export interface DocumentHeader {
  sourceId: string;
  title: string;
  description: string;
  statusLabel: string;
  statusLine: Line;
  body: Line[];
}

export function parseHeader(lines: Line[]): DocumentHeader | string {
  const nonEmpty = lines.filter((l) => l.text.trim() !== "");
  const [titleLine, statusLine] = nonEmpty;
  const title = titleLine?.text.match(TITLE_RE);
  if (!title) return "titre « # Dxx — … » introuvable en première ligne";
  const source = statusLine?.text.match(SOURCE_LINE_RE);
  if (!source) return "ligne « **Source fictive** : … **Statut** : … » introuvable";
  return {
    sourceId: title[1],
    title: title[2].trim(),
    description: source[1].replace(/\.$/, ""),
    statusLabel: stripBold(source[2]).replace(/\.$/, "").trim(),
    statusLine,
    body: lines.filter((l) => l.n > statusLine.n),
  };
}

const MONTHS: Record<string, number> = {
  janvier: 1, février: 2, mars: 3, avril: 4, mai: 5, juin: 6,
  juillet: 7, août: 8, septembre: 9, octobre: 10, novembre: 11, décembre: 12,
};
const MONTH_RE = Object.keys(MONTHS).join("|");
const DATE_RE = new RegExp(
  `(?:du\\s+(\\d{1,2})(?:er)?\\s+(?:(${MONTH_RE})\\s+)?au\\s+)?(\\d{1,2})(?:er)?\\s+(${MONTH_RE})\\s+(\\d{4})`,
  "i",
);

const iso = (y: string, m: number, d: string) =>
  `${y}-${String(m).padStart(2, "0")}-${d.padStart(2, "0")}`;

/** Parses "5 septembre 2026", "1er octobre 2026" or "du 7 au 27 septembre 2026". */
export function parseFrenchDateRange(text: string): { start: string; end: string } | null {
  const m = text.match(DATE_RE);
  if (!m) return null;
  const [, fromDay, fromMonth, day, month, year] = m;
  const end = iso(year, MONTHS[month.toLowerCase()], day);
  if (!fromDay) return { start: end, end };
  const startMonth = MONTHS[(fromMonth ?? month).toLowerCase()];
  return { start: iso(year, startMonth, fromDay), end };
}

export function parseDurationMin(text: string): number | null {
  const m = text.match(/(\d+)\s*minutes?\b/);
  return m ? Number(m[1]) : null;
}

const CAVEAT_RE = /^\*\*(Attention|Note méthodologique|Limites?|Règle d'usage|Précautions?)\*\*\s*:\s*/;

/** Separates labelled warning paragraphs (Attention, Limite…) from the indexable body. */
export function extractCaveats(blocks: Line[][]): { caveats: Caveat[]; content: Line[][] } {
  const caveats: Caveat[] = [];
  const content: Line[][] = [];
  for (const block of blocks) {
    const m = block[0].text.match(CAVEAT_RE);
    if (!m) {
      content.push(block);
      continue;
    }
    caveats.push({
      label: m[1],
      text: stripBold(blockText(block).slice(m[0].length)).trim(),
      lineStart: block[0].n,
      lineEnd: block[block.length - 1].n,
    });
  }
  return { caveats, content };
}
