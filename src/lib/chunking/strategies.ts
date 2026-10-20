import { blockText, isHeading, isTable, stripBold } from "./markdown.ts";
import type { ChunkStrategy, Line } from "./types.ts";

/** A chunk before metadata is attached: the source lines it covers plus parsed fields. */
export interface ChunkDraft {
  suffix: string;
  lines: Line[];
  sectionPath: string | null;
  speakers?: string[];
  question?: string;
  week?: number;
  grocerySpendEur?: number;
}

export interface DraftIssue {
  severity: "error" | "warning";
  suffix?: string;
  message: string;
}

export interface StrategyResult {
  drafts: ChunkDraft[];
  issues: DraftIssue[];
}

type Strategy = (blocks: Line[][]) => StrategyResult;

const headingText = (block: Line[]) => stripBold(block[0].text.replace(/^#+\s+/, "")).trim();

/** Splits on `## ` headings; blocks before the first heading form the intro. */
function groupSections(blocks: Line[][]) {
  const intro: Line[][] = [];
  const sections: { heading: Line[]; blocks: Line[][] }[] = [];
  for (const block of blocks) {
    if (isHeading(block)) sections.push({ heading: block, blocks: [] });
    else if (sections.length) sections[sections.length - 1].blocks.push(block);
    else intro.push(block);
  }
  return { intro, sections };
}

const paragraphs: Strategy = (blocks) => ({
  drafts: blocks.map((block, i) => ({ suffix: `p${i + 1}`, lines: block, sectionPath: null })),
  issues: blocks
    .filter((block) => isHeading(block, 1) || isHeading(block))
    .map((block) => ({ severity: "warning", message: `titre inattendu dans un document découpé par paragraphe (l.${block[0].n})` })),
});

// "**Chercheuse : Comment décidez-vous des repas ?**"
const QUESTION_RE = /^\*\*([^*:]+?)\s*:\s*([^*]+?)\*\*$/;
// "**Julie :** « … »"
const ANSWER_RE = /^\*\*([^*:]+?)\s*:\*\*\s*(.*)$/;

const qaPairs: Strategy = (blocks) => {
  const drafts: ChunkDraft[] = [];
  const issues: DraftIssue[] = [];
  let pair: ChunkDraft | null = null;
  let orphan: ChunkDraft | null = null;

  for (const line of blocks.flat()) {
    const q = line.text.match(QUESTION_RE);
    if (q) {
      pair = { suffix: `q${drafts.filter((d) => d.question).length + 1}`, lines: [line], sectionPath: null, speakers: [q[1].trim()], question: q[2].trim() };
      drafts.push(pair);
      orphan = null;
      continue;
    }
    if (!pair) {
      // Content before any question is kept (never dropped) but flagged.
      if (!orphan) {
        orphan = { suffix: `x${drafts.length + 1}`, lines: [], sectionPath: null, speakers: [] };
        drafts.push(orphan);
        issues.push({ severity: "warning", suffix: orphan.suffix, message: `contenu sans question associée (l.${line.n})` });
      }
      orphan.lines.push(line);
      continue;
    }
    const a = line.text.match(ANSWER_RE);
    if (a && !pair.speakers!.includes(a[1].trim())) pair.speakers!.push(a[1].trim());
    pair.lines.push(line);
  }

  for (const d of drafts) {
    if (d.question && d.speakers!.length < 2) {
      issues.push({ severity: "error", suffix: d.suffix, message: `question sans réponse : « ${d.question} »` });
    }
  }
  return { drafts, issues };
};

// "## Semaine 1 — Courses : 94 €"
const WEEK_RE = /^##\s+Semaine\s+(\d+)(?:\s*[—–-]\s*Courses\s*:\s*(\d+(?:[.,]\d+)?)\s*€)?/i;
const TABLE_SEPARATOR_RE = /^\|(\s*:?-+:?\s*\|)+$/;

const cellCount = (row: string) => row.split("|").length - 2;

function checkTable(table: Line[], suffix: string): DraftIssue[] {
  const issues: DraftIssue[] = [];
  const [header, separator, ...rows] = table;
  if (!separator || !TABLE_SEPARATOR_RE.test(separator.text)) {
    issues.push({ severity: "error", suffix, message: `tableau sans ligne de séparation d'en-tête (l.${header.n})` });
  }
  if (rows.length === 0) issues.push({ severity: "warning", suffix, message: `tableau sans ligne de données (l.${header.n})` });
  const expected = cellCount(header.text);
  for (const row of rows) {
    if (cellCount(row.text) !== expected) {
      issues.push({ severity: "warning", suffix, message: `ligne l.${row.n} : ${cellCount(row.text)} cellules au lieu de ${expected}` });
    }
  }
  return issues;
}

const journalWeeks: Strategy = (blocks) => {
  const drafts: ChunkDraft[] = [];
  const issues: DraftIssue[] = [];
  const { intro, sections } = groupSections(blocks);
  const trailing: Line[][] = [];

  if (intro.length) drafts.push({ suffix: "intro", lines: intro.flat(), sectionPath: "Introduction" });

  sections.forEach((section, i) => {
    const m = section.heading[0].text.match(WEEK_RE);
    const week = m ? Number(m[1]) : undefined;
    const suffix = `s${week ?? i + 1}`;
    if (!m) issues.push({ severity: "warning", suffix, message: `titre de semaine non reconnu : « ${section.heading[0].text} »` });

    const lastTable = section.blocks.findLastIndex(isTable);
    if (lastTable === -1) issues.push({ severity: "warning", suffix, message: "section sans tableau" });
    else issues.push(...checkTable(section.blocks[lastTable], suffix));

    // Paragraphs after the week's table (e.g. "Notes de Julie") are not part of that week.
    const own = lastTable === -1 ? section.blocks : section.blocks.slice(0, lastTable + 1);
    trailing.push(...section.blocks.slice(own.length));
    drafts.push({
      suffix,
      lines: [...section.heading, ...own.flat()],
      sectionPath: headingText(section.heading),
      week,
      grocerySpendEur: m?.[2] ? Number(m[2].replace(",", ".")) : undefined,
    });
  });

  trailing.forEach((block, i) => {
    const label = block[0].text.match(/^\*\*(.+?)\*\*/)?.[1] ?? "Notes";
    drafts.push({ suffix: i === 0 ? "notes" : `notes${i + 1}`, lines: block, sectionPath: label });
  });
  return { drafts, issues };
};

const sections: Strategy = (blocks) => {
  const drafts: ChunkDraft[] = [];
  const { intro, sections } = groupSections(blocks);
  if (intro.length) drafts.push({ suffix: "intro", lines: intro.flat(), sectionPath: "Introduction" });
  sections.forEach((section, i) => {
    drafts.push({ suffix: `s${i + 1}`, lines: [...section.heading, ...section.blocks.flat()], sectionPath: headingText(section.heading) });
  });
  const issues: DraftIssue[] = sections
    .filter((s) => s.blocks.length === 0)
    .map((s) => ({ severity: "warning", message: `section vide : « ${blockText(s.heading)} »` }));
  return { drafts, issues };
};

export const STRATEGIES: Record<ChunkStrategy, Strategy> = {
  paragraphs,
  qa_pairs: qaPairs,
  journal_weeks: journalWeeks,
  sections,
};
