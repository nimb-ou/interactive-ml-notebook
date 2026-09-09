# Authoring standard — the expanded edition

This document governs the rewrite of every section from reference-density prose into a
teaching text. §0.1 (`js/content/10-onramp.js`) is the reference implementation; read it
before writing anything else.

## The problem being fixed

The original content stated conclusions. It was accurate, quotable and dense, and it
assumed the reader already had the scaffolding to hang each claim on. A reader meeting a
topic for the first time got the summary a knowledgeable person would write *after*
understanding it, never the explanation that produces the understanding.

The fix is **not** to wrap existing sentences in `H.intuition` boxes. It is to rewrite the
narrative so that it builds the idea, then let the original crisp sentence stand as the
summary it always was.

## The five rules

**1. Every concept gets motivated before it is named.**
Open with a concrete situation the reader can already reason about, walk to the point where
their existing tools break, and introduce the new idea as the thing that resolves the
tension. Never open a passage with a definition.

> §0.1 opens with writing a spam filter by hand, lets the rules pile up and contradict each
> other, then observes that you can *recognise* spam without being able to *state* the rule.
> Only then does the term "machine learning" appear.

**2. Complete sentences, plain words, one idea per sentence.**
The old voice compressed three claims and two caveats into a single clause joined by
em-dashes. Unpack these. Prefer full stops. Prefer "because" over a dash. A sentence that
needs re-reading to parse has failed, no matter how elegant it is.

Say what a symbol is called and how to read it aloud the first time it appears
("$\theta$, the Greek letter theta"; "read $f_\theta(x)$ as *f, tuned by theta, applied to
x*"). Never assume notation is self-evident.

**3. Formulas are walked through, not dropped in.**
After any non-trivial expression, add a paragraph that reads it back in words: what each
symbol is, what the operation does, and what changes if an input changes. Use `H.deriv` for
anything with more than two algebraic steps — the right-hand "why this line is allowed"
column is the pedagogy, so never leave it empty.

**4. Every lab gets three paragraphs of framing before it.**
A lab with only its inline `note` for context is a puzzle, not a lesson. Before each
`H.lab(...)`, write:

- **What you are looking at** — what each visual element *is*. Name the axes, say what a dot
  represents, say what the dashed lines mean.
- **What to do with it** — the specific interaction to perform, with a small observation the
  reader should make while doing it.
- **The thing genuinely worth noticing** — the pedagogical payload. The one behaviour that,
  if they see it, means the section landed.

Verify the interaction against the lab's actual code before describing it. The original
§0.1 note described the controls backwards.

**5. Quiz explanations teach; cards stay compressed.**
The `why` field is a teaching slot, not an answer key. Explain why the correct answer is
correct, why the most tempting wrong answer is tempting, and what general principle the
question is really testing. Three to six sentences. Name the section that develops it.

Recall cards are the opposite and must stay one or two lines. They are the compression
layer, and compression is their function.

## Length

Sections currently run 250–450 readable words. Target **5× that**, roughly 1,300–2,200
words, reached by adding explanation — never by padding. If a passage cannot be made longer
without repeating itself, it is already finished; move on.

## Voice

Second person, present tense, British spelling (`-ise`, `-isation`). Confident but not
clipped. The existing wit and bluntness are assets — keep them. What changes is the *runway*
before each landing, not the landing.

Two words the project has deliberately removed: "highest-leverage" and "orchestrator". Not
enforced by any test; honour it anyway.

## The layers that were built and never used

The platform ships helpers that were almost entirely unused. Corpus-wide counts before this
rewrite: `H.analogy` 0, `H.practice` 0, `H.history` 0, `H.intuition` 15, `H.more` 6,
`H.deriv` 11 — across 116 sections. They are styled, tested and ready.

| Helper | Use it for |
|---|---|
| `H.analogy` | A physical or everyday system with the same structure. One per hard concept. |
| `H.intuition` | The reframing that makes a result feel inevitable rather than arbitrary. |
| `H.history` | Why the field went this way — what was tried first and why it failed. |
| `H.more` | Depth a first-time reader should skip, and a second-time reader wants. |
| `H.deriv` | Any multi-step algebra. Fill the justification column. |
| `H.practice` | What actually goes wrong when you do this on real data. |
| `H.pitfall` | The specific mistake, stated as the mistake. |
| `H.key` | The one sentence to memorise. At most two per section. |

## What must not change

- **Section `id`, `track` and `num`** — these are routes, sort keys and cross-reference
  targets. Changing one silently breaks links from other sections.
- **Lab keys.** Every key in `labs: {}` needs its `H.lab('key', ...)` placeholder and vice
  versa. A mismatch logs a warning, and warnings fail the build.
- **Existing lab code**, unless it is wrong. Expand the prose around it.
- **The honesty conventions**: `H.key`/`<mark>` for memorise-verbatim, `H.flag` for
  contested or fast-moving claims, the `mistake` argument of `H.probe` for the error that
  costs offers, `H.pitfall` for traps. This culture is the project's signature.
- **The interview layer** (`H.probe`, `H.iq`).

## Verification — required before any commit

```bash
node test/smoke.js       # must end with "errors : 0"
```

The smoke test treats **every** `console.warn` and `console.error` as fatal. It catches
duplicate ids, unknown tracks, missing lab hosts, labs that throw, and template artefacts
(`${undefined}`, `[object Object]`, `NaN%`) leaking into markup. It also reports sections
under 350 words, which after this rewrite should be empty except for the recall pages.

Then read the section in a browser — `python3 -m http.server 8781` — and confirm the maths
typesets, the boxes render, and the lab still mounts.

### Escaping

Content is a JS template literal, so every LaTeX backslash is doubled: `$P(A\\mid B)$`,
`\\frac`, `\\theta`. A single backslash is eaten by the string and never reaches KaTeX.
Maths inside `H.code(...)` is never typeset, so dollar signs in code are safe.
