# Quality audit — bringing every section to publishable-textbook standard

`AUTHORING.md` defines the writing contract. This document defines the *quality bar* a
section must clear to be marked done, and tracks which sections have cleared it.

The rewrite that preceded this audit brought every section to the right shape. This pass is
about **richness**: whether the theory is actually taught, whether the labs are integrated
rather than decorative, and whether anything is missing.

## Why this pass exists

Three defects were found by reading that no test could catch:

- §1.5 was never rewritten at all. It sat at 332 words between neighbours of 5,448 and 7,021.
- §2.16 was skipped on the mistaken belief that it was already good.
- 41 sections had quiz explanations averaging 22 words against a corpus norm of ~120.

All three were invisible to `npm test`, which checks that pages render, not that they teach.
A reviewer who skims will miss the next one too, so this pass requires reading.

## The bar

A section clears when all seven hold.

**1. The theory is derived, not asserted.**
Every non-obvious result either carries its derivation or points to the section that has it.
`H.deriv` ladders must fill the justification column — the right-hand "why this line is
allowed" is the pedagogy. A formula the reader cannot reconstruct has not been taught.

**2. Every concept is motivated before it is named.**
Open from a situation the reader can already reason about, walk to where their tools break,
introduce the idea as the resolution. No passage opens on a definition or a bullet list.

**3. Notation is never assumed.**
Each symbol is named and read aloud on first use. After any non-trivial expression, a
paragraph says it back in words and works a small numeric example the reader can check.

**4. The labs are load-bearing.**
Three paragraphs *before* each lab: what you are looking at, what to do, and the one thing
worth noticing. The lab must demonstrate something the prose claims, and the prose must
reference what the lab shows. A lab that could be deleted without weakening the argument is
decorative, and either the framing or the lab needs work.

**5. The teaching scaffolding is used.**
Corpus average is 4.6 of `H.analogy` / `H.intuition` / `H.history` / `H.practice` / `H.more` /
`H.deriv` / `H.pitfall` / `H.key` / `H.worked` per section. A section well below that is
usually asserting where it should be explaining. Use them where they genuinely help; do not
pad to hit a count.

**6. Every number survives scrutiny.**
Any checkable figure must be recomputed before it ships. Where the text and a lab both
produce a number, they must agree — a note contradicting its own widget has already happened
twice here.

**7. Honesty conventions hold.**
`H.flag` on contested or fast-moving claims, `H.pitfall` on traps, `H.key` on the sentence to
memorise, the `mistake` argument of `H.probe`. Fast-moving material stated flat is a defect.

## Priority

Measured before this pass began.

**Tier 1 — zero teaching helpers (7).**
4.10 scaling-laws · 4.11 distributed · 4.17 safety · 5.3 agents · 5.4 multi-agent ·
5.5 mcp · 5.6 frameworks

**Tier 2 — short bodies, below the 1,300-word floor (11).**
5.12 mlops (849) · 5.9 chunking (858) · 5.11 evals (934) · 5.10 vector-search (1,034) ·
4.21 structured-output (1,143) · 4.22 speculative (1,159) · 5.2 rag-vs-ft (1,162) ·
6.7 vision-tasks (1,200) · 5.6 frameworks (1,247) · 4.20 reasoning (1,274) ·
4.19 multimodal (1,283)

**Tier 3 — fewer than four teaching helpers (35).** Listed in the ledger.

**Tier 4 — everything else.** Still requires a read against the seven points, but these are
expected to pass with minor or no changes.

## Verification, required before any section is marked done

```bash
npm test        # smoke.js (renders every section, mounts every lab) + formulas.js (KaTeX)
```

Both must report `errors : 0`. Note that `smoke.js` discards KaTeX warnings by design, which
is why `formulas.js` exists and why both must run.

Structural invariants that must survive every edit:

- `id`, `track`, `num` are routes and cross-reference targets. Never change them.
- Every key in `labs: {}` keeps its `H.lab('key', ...)` placeholder, and vice versa.
- Quiz `q:`, `options:` and `answer:` are not touched when editing explanations. Answer
  indices must stay byte-identical.
- LaTeX backslashes are doubled inside the template literals.
- British spelling in prose. `regularization` and `optimization` stay American where they are
  section ids or routes.
- Scratch files go in /tmp, never in `test/`. Three have been committed by accident already.
