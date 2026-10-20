import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";

import { parseFrenchDateRange } from "./markdown.ts";
import { chunkCorpus, chunkPersona } from "./pipeline.ts";
import { JULIE } from "./personas.ts";
import type { Chunk, PersonaCorpusConfig } from "./types.ts";

const ROOT = path.resolve(import.meta.dirname, "../../..");
const result = chunkPersona(JULIE, ROOT);
const chunksOf = (sourceId: string) => result.chunks.filter((c) => c.metadata.sourceId === sourceId);
const byId = (id: string) => byIdIn(result.chunks, id);
const sourceLines = (chunk: Chunk) =>
  readFileSync(path.join(ROOT, chunk.metadata.sourceFile), "utf8").split("\n");

describe("corpus Julie : documents", () => {
  test("aucune anomalie bloquante", () => {
    assert.deepEqual(result.anomalies.filter((a) => a.severity === "error"), []);
  });

  test("indexe exactement D01 à D06 et jamais le README", () => {
    assert.deepEqual(result.documents.map((d) => d.sourceId), ["D01", "D02", "D03", "D04", "D05", "D06"]);
    assert.deepEqual(result.ignoredFiles, ["README.md"]);
    for (const chunk of result.chunks) {
      assert.doesNotMatch(chunk.metadata.sourceFile, /README/i);
      assert.doesNotMatch(chunk.text, /Persona Lab — Corpus fictif/);
    }
  });

  test("chaque document produit des chunks, avec des identifiants stables", () => {
    assert.deepEqual(result.chunks.map((c) => c.id), [
      "D01-p1", "D01-p2", "D01-p3",
      "D02-q1", "D02-q2", "D02-q3", "D02-q4", "D02-q5", "D02-q6", "D02-q7",
      "D03-intro", "D03-s1", "D03-s2", "D03-s3", "D03-notes",
      "D04-q1", "D04-q2", "D04-q3", "D04-q4", "D04-q5",
      "D05-q1", "D05-q2", "D05-q3", "D05-q4", "D05-q5",
      "D06-s1", "D06-s2", "D06-s3", "D06-s4",
    ]);
  });

  test("aucune ligne de contenu n'est perdue : chaque ligne est dans un chunk ou un avertissement", () => {
    for (const doc of result.documents) {
      const lines = readFileSync(path.join(ROOT, doc.sourceFile), "utf8").split("\n");
      const covered = new Set<number>([1, 2]); // titre et ligne source → métadonnées
      for (const c of chunksOf(doc.sourceId)) {
        for (let n = c.metadata.lineStart; n <= c.metadata.lineEnd; n++) covered.add(n);
      }
      for (const c of doc.caveats) for (let n = c.lineStart; n <= c.lineEnd; n++) covered.add(n);
      lines.forEach((text, i) => {
        if (text.trim()) assert.ok(covered.has(i + 1), `${doc.sourceFile}:${i + 1} perdue : « ${text} »`);
      });
    }
  });

  test("le texte d'un chunk correspond exactement aux lignes source citées", () => {
    for (const chunk of result.chunks) {
      const { lineStart, lineEnd } = chunk.metadata;
      const expected = sourceLines(chunk).slice(lineStart - 1, lineEnd).join("\n");
      assert.equal(chunk.text, expected, chunk.id);
    }
  });
});

describe("corpus Julie : avertissements", () => {
  const expectedCaveats: Record<string, { label: string; excerpt: string }[]> = {
    D01: [{ label: "Attention", excerpt: "ni budget réel, ni disposition à payer" }],
    D02: [{ label: "Note méthodologique", excerpt: "pas des mesures chronométrées" }],
    D03: [
      { label: "Statut", excerpt: "reçus et durées non vérifiés" },
      { label: "Périmètre des dépenses", excerpt: "ne constituent pas un budget mensuel complet" },
      { label: "Limite du journal", excerpt: "ne permet pas d'estimer précisément" },
    ],
    D04: [{ label: "Limite", excerpt: "présenté par la chercheuse comme scénario, pas accepté par Julie" }],
    D05: [{ label: "Limite", excerpt: "reste hypothétique" }],
    D06: [
      { label: "Statut", excerpt: "pas déclarations directes de Julie" },
      { label: "Règle d'usage", excerpt: "ne doivent pas être attribuées à Julie" },
    ],
  };

  for (const [sourceId, caveats] of Object.entries(expectedCaveats)) {
    test(`${sourceId} : chaque chunk porte ses avertissements`, () => {
      const chunks = chunksOf(sourceId);
      assert.ok(chunks.length > 0);
      for (const chunk of chunks) {
        assert.equal(chunk.metadata.caveats[0].label, "Statut", `${chunk.id} sans statut`);
        for (const { label, excerpt } of caveats) {
          const caveat = chunk.metadata.caveats.find((c) => c.label === label);
          assert.ok(caveat, `${chunk.id} : avertissement « ${label} » manquant`);
          assert.ok(caveat.text.includes(excerpt), `${chunk.id} : « ${excerpt} » absent de « ${label} »`);
        }
      }
    });
  }

  test("les avertissements étiquetés ne deviennent pas des chunks autonomes", () => {
    for (const chunk of result.chunks) {
      assert.doesNotMatch(chunk.text, /^\*\*(Attention|Note méthodologique|Limite|Règle d'usage)\*\*/m, chunk.id);
    }
  });
});

describe("corpus Julie : entretiens", () => {
  test("une paire question/réponse par chunk, dans l'ordre", () => {
    for (const [sourceId, count] of [["D02", 7], ["D04", 5], ["D05", 5]] as const) {
      const chunks = chunksOf(sourceId);
      assert.equal(chunks.length, count, sourceId);
      for (const chunk of chunks) {
        const lines = chunk.text.split("\n");
        assert.match(lines[0], /^\*\*Chercheuse : .+\?\*\*$/, `${chunk.id} ne commence pas par la question`);
        assert.ok(lines.slice(1).some((l) => /^\*\*Julie :\*\* « .+ »$/.test(l)), `${chunk.id} sans réponse de Julie`);
        assert.equal(chunk.text.match(/\*\*Chercheuse :/g)?.length, 1, `${chunk.id} contient plusieurs questions`);
        assert.deepEqual(chunk.metadata.speakers, ["Chercheuse", "Julie"]);
        assert.ok(lines[0].includes(chunk.metadata.question!), `${chunk.id} : question en métadonnée incohérente`);
        assert.equal(chunk.metadata.containsVerbatim, true);
      }
    }
  });

  test("le scénario à 60 € reste lié à la réponse hésitante de Julie", () => {
    const chunk = byId("D04-q4");
    assert.equal(chunk.metadata.question, "Seriez-vous prête à payer 60 € par semaine ?");
    assert.match(chunk.text, /Je ne sais pas/);
    assert.match(chunk.embeddingText, /^\[D04 · .* · 2026-09-29 · déclarations de Julie\]\nContexte : .*\nChercheuse : Seriez-vous/);
  });

  test("D04-q4 : l'embedding précise que les 60 € concernent les paniers-repas livrés", () => {
    const chunk = byId("D04-q4");
    const previous = byId("D04-q3");
    assert.equal(chunk.metadata.question, "Seriez-vous prête à payer 60 € par semaine ?");
    assert.deepEqual(chunk.metadata.contextChunkIds, ["D04-q3"]);
    // Le contexte cite mot pour mot la question précédente du document source.
    assert.ok(chunk.embeddingText.includes(`« ${previous.metadata.question} » (D04-q3)`));
    assert.match(chunk.embeddingText, /paniers-repas livrés/);
    const lines = chunk.embeddingText.split("\n");
    assert.ok(lines.indexOf(lines.find((l) => l.startsWith("Contexte :"))!) < lines.indexOf("Chercheuse : Seriez-vous prête à payer 60 € par semaine ?"));
  });

  test("le contexte ajouté ne modifie ni le texte du chunk ni les verbatims", () => {
    const chunk = byId("D04-q4");
    assert.equal(chunk.text, sourceLines(chunk).slice(12, 14).join("\n"));
    assert.doesNotMatch(chunk.text, /paniers-repas/);
    assert.ok(chunk.embeddingText.includes("Julie : « Je ne sais pas. Ça dépend du nombre de repas, des portions, de ce qui reste à acheter. Il faudrait essayer. »"));
    assert.equal(chunk.metadata.caveats.find((c) => c.label === "Limite")?.lineStart, 19);
  });

  test("seul D04-q4 reçoit un contexte de question précédente", () => {
    const withContext = result.chunks.filter((c) => c.metadata.contextChunkIds.length > 0).map((c) => c.id);
    assert.deepEqual(withContext, ["D04-q4"]);
    for (const chunk of result.chunks) {
      if (chunk.id !== "D04-q4") assert.doesNotMatch(chunk.embeddingText, /^Contexte :/m, chunk.id);
    }
  });

  test("les réponses de D02 restent avec leur question", () => {
    assert.match(byId("D02-q2").text, /temps en cuisine[\s\S]*vingt minutes de préparation/);
    assert.match(byId("D02-q6").text, /liste automatique[\s\S]*trois paquets de riz/);
  });
});

describe("corpus Julie : journal, profil et notes", () => {
  test("chaque semaine du journal garde l'en-tête du tableau et ses 7 jours", () => {
    for (const [week, spend] of [[1, 94], [2, 76], [3, 83]]) {
      const chunk = byId(`D03-s${week}`);
      const lines = chunk.text.split("\n");
      assert.match(lines[0], new RegExp(`^## Semaine ${week} — Courses : ${spend} €$`));
      assert.equal(lines[1], "| Jour | Dîner | Préparation active estimée | Remarque |");
      assert.equal(lines[2], "|---|---|---:|---|");
      assert.equal(lines.filter((l) => /^\| (Lundi|Mardi|Mercredi|Jeudi|Vendredi|Samedi|Dimanche) \|/.test(l)).length, 7);
      assert.equal(chunk.metadata.week, week);
      assert.equal(chunk.metadata.grocerySpendEur, spend);
    }
  });

  test("le titre de section n'apparaît qu'une fois dans le texte d'embedding", () => {
    const headed = result.chunks.filter((c) => c.text.startsWith("## "));
    assert.deepEqual(headed.map((c) => c.id), ["D03-s1", "D03-s2", "D03-s3", "D06-s1", "D06-s2", "D06-s3", "D06-s4"]);
    for (const chunk of headed) {
      const { sectionPath } = chunk.metadata;
      assert.equal(chunk.embeddingText.split(sectionPath!).length - 1, 1, `${chunk.id} : titre dupliqué`);
      assert.doesNotMatch(chunk.embeddingText, /^#/m, `${chunk.id} : titre markdown dans l'embedding`);
    }
    const lines = byId("D03-s2").embeddingText.split("\n");
    assert.equal(lines[1], "Section : Semaine 2 — Courses : 76 €");
    assert.equal(lines[2], "| Jour | Dîner | Préparation active estimée | Remarque |");
    // Le texte original, lui, garde son titre.
    assert.match(byId("D03-s2").text, /^## Semaine 2 — Courses : 76 €\n\| Jour/);
  });

  test("l'introduction et les notes de Julie sont des chunks distincts des semaines", () => {
    assert.match(byId("D03-intro").text, /^Les dépenses indiquées/);
    assert.match(byId("D03-notes").text, /^\*\*Notes de Julie\*\*/);
    assert.doesNotMatch(byId("D03-s3").text, /Notes de Julie/);
  });

  test("le profil est découpé par paragraphe", () => {
    assert.match(byId("D01-p1").text, /^Julie, 38 ans/);
    assert.match(byId("D01-p2").text, /^Les mardis et jeudis/);
    assert.match(byId("D01-p3").text, /^Julie décrit sa famille/);
  });

  test("les notes D06 sont des interprétations, jamais des verbatims", () => {
    const chunks = chunksOf("D06");
    assert.deepEqual(chunks.map((c) => c.metadata.sectionPath), [
      "Observations documentées dans le corpus",
      "Hypothèses à vérifier, non établies",
      "Contradictions et précautions",
      "Questions ouvertes",
    ]);
    for (const chunk of chunks) {
      assert.equal(chunk.metadata.evidenceStatus, "interpretation");
      assert.equal(chunk.metadata.containsVerbatim, false);
      assert.match(chunk.embeddingText, /pas déclarations directes de Julie/);
    }
    assert.deepEqual(byId("D06-s1").metadata.citedSources, ["D02", "D03", "D04", "D05"]);
  });
});

describe("métadonnées", () => {
  test("identité, provenance et traçabilité renseignées", () => {
    for (const chunk of result.chunks) {
      const m = chunk.metadata;
      assert.equal(m.personaId, "julie");
      assert.equal(m.corpusVersion, "v1");
      assert.match(m.sourceFile, /^data\/personas\/julie\/D0\d_[\w]+\.md$/);
      assert.ok(chunk.id.startsWith(`${m.sourceId}-`));
      assert.match(m.contentHash, /^[0-9a-f]{16}$/);
      assert.match(m.sourceDateStart!, /^2026-\d{2}-\d{2}$/);
      assert.ok(m.lineStart <= m.lineEnd);
      assert.ok(chunk.approxTokens > 0);
    }
    assert.equal(byId("D03-s1").metadata.sourceDateEnd, "2026-09-27");
    assert.equal(byId("D02-q1").metadata.durationMin, 42);
    assert.equal(byId("D05-q1").metadata.sourceDateStart, "2026-10-01");
  });

  test("dates françaises", () => {
    assert.deepEqual(parseFrenchDateRange("entretien du 1er octobre 2026"), { start: "2026-10-01", end: "2026-10-01" });
    assert.deepEqual(parseFrenchDateRange("du 28 février au 3 mars 2026"), { start: "2026-02-28", end: "2026-03-03" });
    assert.equal(parseFrenchDateRange("sans date"), null);
  });
});

describe("détection d'anomalies", () => {
  const config: PersonaCorpusConfig = {
    personaId: "test",
    sourceDir: "fixtures",
    sources: {
      D01: { sourceType: "interview", evidenceStatus: "declared", strategy: "qa_pairs" },
      D02: { sourceType: "profile", evidenceStatus: "declared", strategy: "paragraphs" },
    },
  };
  const header = (id: string) => `# ${id} — Test\n**Source fictive** : entretien du 2 mai 2026. **Statut** : test.\n\n`;

  test("question sans réponse, fichier inconnu, source manquante, README ignoré", () => {
    const res = chunkCorpus(config, [
      { fileName: "D01_test.md", content: `${header("D01")}**Chercheuse : Une question ?**\n\n**Chercheuse : Une autre ?**\n**Julie :** « Oui. »\n` },
      { fileName: "D09_inconnu.md", content: header("D09") },
      { fileName: "README.md", content: "# README" },
    ]);
    const messages = res.anomalies.map((a) => `${a.severity}:${a.chunkId ?? a.sourceFile}:${a.message}`);
    assert.ok(messages.some((m) => m.startsWith("error:D01-q1:question sans réponse")), messages.join("\n"));
    assert.ok(messages.some((m) => m.includes("D09 absent du registre")));
    assert.ok(messages.some((m) => m.includes("source D02 attendue mais absente")));
    assert.deepEqual(res.ignoredFiles, ["D09_inconnu.md", "README.md"]);
    assert.equal(byIdIn(res.chunks, "D01-q2").metadata.question, "Une autre ?");
  });

  test("identifiant du titre incohérent avec le nom du fichier", () => {
    const res = chunkCorpus({ ...config, sources: { D02: config.sources.D02 } }, [
      { fileName: "D02_test.md", content: `${header("D03")}Un paragraphe.\n` },
    ]);
    assert.ok(res.anomalies.some((a) => a.message.includes("différent du nom de fichier")));
  });

  test("contexte de question précédente demandé pour la première question", () => {
    const res = chunkCorpus(
      { ...config, sources: { D01: { ...config.sources.D01, previousQuestionContext: ["q1"] } } },
      [{ fileName: "D01_test.md", content: `${header("D01")}**Chercheuse : Première ?**\n**Julie :** « Oui. »\n` }],
    );
    assert.ok(res.anomalies.some((a) => a.severity === "error" && a.message.includes("contexte de question précédente impossible pour D01-q1")));
    assert.deepEqual(byIdIn(res.chunks, "D01-q1").metadata.contextChunkIds, []);
  });

  test("avertissement en ligne déclaré mais introuvable", () => {
    const res = chunkCorpus(
      { ...config, sources: { D02: { ...config.sources.D02, inlineCaveats: [{ label: "X", startsWith: "Introuvable" }] } } },
      [{ fileName: "D02_test.md", content: `${header("D02")}Un paragraphe.\n` }],
    );
    assert.ok(res.anomalies.some((a) => a.severity === "error" && a.message.includes("avertissement en ligne introuvable")));
  });
});

function byIdIn(chunks: Chunk[], id: string) {
  const chunk = chunks.find((c) => c.id === id);
  assert.ok(chunk, `chunk ${id} introuvable`);
  return chunk;
}
