import type { PersonaCorpusConfig } from "./types.ts";

export const JULIE: PersonaCorpusConfig = {
  personaId: "julie",
  sourceDir: "data/personas/julie",
  sources: {
    D01: { sourceType: "profile", evidenceStatus: "declared", strategy: "paragraphs" },
    D02: { sourceType: "interview", evidenceStatus: "declared", strategy: "qa_pairs" },
    D03: {
      sourceType: "journal",
      evidenceStatus: "self_reported",
      strategy: "journal_weeks",
      inlineCaveats: [
        { label: "Périmètre des dépenses", startsWith: "Les dépenses indiquées" },
        { label: "Limite du journal", startsWith: "Une partie des temps inclut" },
      ],
    },
    D04: {
      sourceType: "interview",
      evidenceStatus: "declared",
      strategy: "qa_pairs",
      // "Seriez-vous prête à payer 60 € par semaine ?" refers to the meal kits of the previous question.
      previousQuestionContext: ["q4"],
    },
    D05: { sourceType: "interview", evidenceStatus: "declared", strategy: "qa_pairs" },
    D06: { sourceType: "research_notes", evidenceStatus: "interpretation", strategy: "sections" },
  },
};

export const PERSONAS: Record<string, PersonaCorpusConfig> = { julie: JULIE };
