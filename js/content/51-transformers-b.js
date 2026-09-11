/* ============================================================
   PART 4 — LLMs & transformers (4.9 – 4.17), plus the Part 4 recall page (4.23)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 4.9 */
  ML.section({
    id: 'pretraining', track: 'llm', num: '4.9',
    title: 'Pretraining',
    lede: 'The objective is one line of mathematics; the corpus is the whole job.',
    html: `
<p>Parts 4.1 through 4.8 built the machine — attention, position, the block, mixture-of-experts and state-space alternatives. None of it does anything until it is trained, and the striking fact about pretraining, the stage that turns a random matrix of weights into a language model, is how little there is to say about the <i>objective</i> compared with how much there is to say about the <i>data</i>. This section is deliberately lopsided for that reason: one short subsection on the loss function, and three much longer ones on what actually determines whether the resulting model is any good.</p>

<h2><span class="sn">4.9.1</span> The objective</h2>
<p>At every position in a training sequence, the model outputs a distribution over the vocabulary — a probability for every possible next token, produced by the softmax at the top of the stack (§4.5) — and the loss compares that distribution against the token that actually came next. This is categorical cross-entropy, and §1.5 already established exactly what that quantity is: the negative log-likelihood of the true label under the model's predicted distribution, the same maximum-likelihood objective that underlies logistic regression and every other probabilistic classifier in this course, here applied with a vocabulary-sized number of classes at every single position of every sequence. There is genuinely nothing more exotic in the loss function itself. §4.4 already noted the multiplying trick that makes this scale so well — a causal mask turns one forward pass over a sequence of length $s$ into $s$ simultaneous supervised predictions — and between the two of them, the entire training signal for a frontier model is fully specified in two sentences.</p>
${H.intuition(`<p>It is worth sitting with why such a plain objective produces a model that appears to reason, translate, and code. Predicting the next token <i>accurately</i>, averaged across a corpus spanning news, contracts, Python, physics textbooks and casual conversation, is an enormously demanding requirement — a model that only pattern-matches surface statistics will be caught out constantly, because getting "the professor who taught the course that Maria took last spring" right requires tracking who "the professor" refers to, and getting the next line of a function right requires something that behaves like an understanding of what the function is for. Next-token prediction does not <i>ask</i> for grammar, facts or reasoning; it turns out that reliably predicting real text at scale is not achievable without something that functions like all three, so a large enough model trained on a diverse enough corpus acquires them as a side effect of getting better at the one thing it was actually optimised for.</p>`)}

<h2><span class="sn">4.9.2</span> The corpus funnel</h2>
<p>If the loss function is nearly free of design decisions, the corpus is where every one of them lives. A raw web crawl is not remotely fit to train on directly — it is dominated by boilerplate, spam, near-duplicate pages, and text of wildly uneven quality — and the sequence of filters that turns it into a training set is, in aggregate, the single largest lever any pretraining team pulls.</p>
${H.table(['Stage', 'What it does', 'Why it matters'], [
      ['<b>Deduplication</b>', 'Exact and near-duplicate removal via MinHash or suffix arrays', 'Measurably improves quality per token <i>and</i> sharply reduces verbatim memorisation'],
      ['<b>Quality filtering</b>', 'A classifier trained to recognise reference-quality text', 'A smaller, cleaner corpus reliably beats a larger, dirtier one at equal compute'],
      ['<b>Mixture weights</b>', 'Proportions across code, web, books, maths, multilingual', 'A first-class hyperparameter — code in the mixture improves reasoning on non-code tasks'],
      ['<b>Curriculum / anneal</b>', 'Ordering, and a high-quality phase at the end', 'A further gain for close to free; the decay phase is when the model is most sensitive'],
      ['<b>Decontamination</b>', 'Remove eval-set overlap — <i>before</i> training, not after', 'Otherwise your benchmark measures the benchmark’s age (§4.16)']
    ])}
<p><b>Deduplication</b> exists to answer two separate, nameable failures at once, and it is worth seeing both before treating "run MinHash" as a single undifferentiated step. The first failure is statistical: if a page of text appears a thousand times in the crawl — a common outcome for boilerplate, licence text, and popular quotes — training on it a thousand times does not merely waste compute; it gives the model a thousand gradient updates reinforcing that exact string, far out of proportion to its actual informational importance, and the model becomes able to recite it close to verbatim. The mechanism is direct: the fraction of total pretraining gradient signal any one piece of text contributes is roughly proportional to how many times it appears, so a document duplicated a thousand times receives a thousand times the "vote" a genuinely unique document of equal length gets, purely as an accident of how many times a web crawler happened to find a copy of it — nothing about the document's actual value to the model justifies that imbalance.</p>
<p>The second failure is computational, and it is the reason deduplication uses MinHash rather than the obvious approach of comparing every document to every other one directly. A crawl feeding a frontier pretraining run holds on the order of ten billion documents; comparing every pair directly costs $\\binom{n}{2}\\approx n^2/2$ comparisons, which for $n=10^{10}$ is on the order of $5\\times10^{19}$ — not a large-but-doable number, an obviously infeasible one at any computational budget a lab actually has. <b>MinHash</b> sidesteps the quadratic cost by replacing each full document with a small, fixed-size <b>sketch</b> — a few hundred integers computed from hashing the document's content in a way specifically designed so that two documents sharing most of their content produce sketches that mostly agree, even though the sketch itself is a small, fixed fraction of the original document's size. Comparing sketches, and — critically — using locality-sensitive hashing to group similar sketches into the same bucket so that only documents already landing in the same bucket ever get compared at all, turns an intractable quadratic search into one that scales close to linearly with the size of the crawl. Removing near-duplicates this way both flattens the over-representation the first failure describes and is the single most effective lever against verbatim memorisation of any specific document, which matters for both privacy and copyright reasons.</p>
${H.pitfall('A quality classifier trained to recognise "reference-quality" text is itself trained on a reference set, and a reference set skewed toward one register of English — formally edited prose, the dialect Wikipedia and mainstream news happen to be written in — will systematically score text in other, equally legitimate registers and dialects lower, not because that text is lower quality, but because it looks less like the training signal the classifier learned "quality" from. This is a documented finding, not a hypothetical: audits of C4-style filtering pipelines have found that automatic quality and toxicity classifiers disproportionately flag and remove text written in African American English and other non-standard dialects, systematically under-representing exactly the voices a narrow reference set was never built to recognise. A filter that quietly encodes "quality" as "resembles Wikipedia" is not a neutral technical choice — check what a quality classifier actually discards, not only how much it discards, before trusting the number on its own.')}
<p><b>Quality filtering</b> trains a small classifier to distinguish text that resembles a curated reference set — Wikipedia, well-edited books, moderated forums — from the rest of the crawl, and keeps only what scores highly. The finding that this reliably beats simply keeping more data, at matched compute, says something specific about what a forward pass is for: every token spent on a low-information page (a listicle, a template-generated product description) is a forward pass not spent on text the model can actually learn structure from, so a smaller, denser corpus outperforms a larger, diluted one of the same total token count. <b>Mixture weights</b> — the proportions of code, web text, books, mathematics and each language in the final blend — are consequently treated as a first-class, tuned hyperparameter rather than an afterthought: code in the mixture, in particular, has been shown to improve reasoning performance even on entirely non-code tasks, a transfer effect that is not yet fully understood but is consistently observed.</p>
<p><b>Curriculum and anneal</b> exploit something about training dynamics rather than data quality per se: the order tokens are seen in is not neutral. A short, high-quality phase placed at the very <i>end</i> of pretraining, after the learning rate has decayed toward its minimum (§4.11's schedules), buys a further, close-to-free quality gain, because a small learning rate makes late updates surgical rather than sweeping — the model is maximally able to absorb the specific character of whatever it sees last without that signal being drowned out by the large, coarse updates earlier training runs on. <b>Decontamination</b> — removing any text that overlaps with known evaluation benchmarks — has to happen at this stage, before training, not as a post-hoc audit; §4.16 covers precisely what goes wrong if it is skipped, but the short version is that a benchmark score on contaminated training data measures how old the benchmark is, not how capable the model is.</p>

<p><b>What you are looking at.</b> A funnel of horizontal bars, each one a stage of the pipeline just described, shrinking left to right as tokens are progressively removed; the width of each bar is the token count surviving that stage, computed live from the filter strengths you set, with the final high-quality anneal phase picked out in a different colour.</p>
<p><b>What to do with it.</b> Push deduplication and quality-filter strictness up from zero and watch two readouts move in opposite directions: usable tokens fall, and relative quality per token rises. There is no setting where both improve together — that trade is the entire content of this subsection made numeric.</p>
<p><b>The thing genuinely worth noticing.</b> The final readout converts whatever token count survives your filters into a Chinchilla-optimal model size (§4.10's 20-tokens-per-parameter ratio, applied here rather than derived). That is the uncomfortable, practical punchline of the whole funnel: the model size you can responsibly train is set by how much your data pipeline actually produces, not by how many GPUs you can buy.</p>

${H.lab('corpus', 'The corpus funnel, with the knobs that matter', 'Move the filters and watch tokens fall while modelled quality rises. The curve is illustrative of the published direction of travel — the point is the shape: aggressive filtering costs tokens and buys quality, up to the point where you run out of data.')}
${H.key('Every stage of the corpus funnel is the answer to a specific, nameable failure of raw web text, not a generic cleanliness pass: deduplication answers over-representation and quadratic comparison cost, quality filtering answers low-information tokens crowding out learnable structure, mixture weighting answers the fact that a blend\'s proportions are themselves a capability lever, and decontamination answers a benchmark ceasing to measure anything once its answers have leaked into training. Know which failure each filter is fixing, not just that "cleaning the data" helps.')}

<h2><span class="sn">4.9.3</span> Synthetic data is now a first-class ingredient</h2>
<p>Model-generated text used to be treated as a fallback for when real data ran short. It no longer is — three specific mechanisms now carry serious weight in frontier pretraining and post-training mixtures, each solving a different problem. <b>Distillation from a stronger model</b> generates answers with a frontier model and trains a smaller one on them; it is the single cheapest way to move a small model's quality, since the expensive part (producing a good answer) has already been paid for by whoever trained the teacher, subject to that teacher's licence terms permitting the use. <b>Self-instruct bootstrapping</b> has a model write new instructions and answers starting from a small seed set of examples, then filters the results for quality — a way of multiplying a small amount of expensive human-curated data into a much larger training set without a proportional increase in human effort. <b>Verifier-filtered generation</b> keeps only generated samples that pass an external checker — a unit test suite for code, a symbolic checker for a maths answer — which is exactly the same idea §4.12's RLVR applies to <i>rewards</i> rather than to <i>data</i>, and it is precisely why mathematics and code benefit disproportionately from synthetic data: both domains have cheap, reliable, automatic verifiers, so filtering is nearly free, whereas open-ended prose has no equivalent checker and synthetic data there is correspondingly harder to trust.</p>
${H.history(`<p>The caveat worth stating before anyone asks is now backed by published evidence rather than intuition alone: Shumailov et al.'s 2024 <i>Nature</i> paper demonstrated "model collapse" directly — train a model on its own unfiltered outputs, then train the next generation on <i>that</i> model's outputs, and repeat, and each generation's output distribution narrows measurably, losing the rare, tail information that made the original data rich, until later generations produce text that is fluent but has visibly forgotten the diversity of the real distribution it started from. The mechanism is intuitive once stated: a model's outputs are always a slightly narrower, slightly smoothed copy of its training distribution, so training on nothing but that copy — with no fresh, real data anchoring each generation back to reality — is repeated photocopying of a photocopy. Each generation loses a little more of what the original had.</p>`)}
${H.flag('The caveat to state before anyone asks: training repeatedly on unfiltered self-generated text degrades diversity and eventually quality — "model collapse". Synthetic data works when it is filtered against something real, and fails when it is a closed loop.')}

<h2><span class="sn">4.9.4</span> Continual pretraining</h2>
<p>Between full pretraining from scratch and lightweight fine-tuning (§4.13) sits a third option that gets less attention than it deserves: take an existing base model and keep training it, with the identical next-token-prediction objective from §4.9.1, on a domain-specific corpus — case law, medical literature, one bank's internal documentation — using a low learning rate and, critically, a <b>replay mixture</b> that mixes in a slice of the original general-purpose training data alongside the new domain text. That replay slice exists to guard against <b>catastrophic forgetting</b>: training purely on narrow domain text, with no counterbalance, measurably erodes the model's general capabilities even as it gains domain fluency, because gradient descent has no built-in preference for retaining what it is not currently being shown.</p>
${H.analogy(`<p>Catastrophic forgetting is the same failure a person immersed entirely in a second language for a year risks with their first: fluency in the new language climbs, genuinely and measurably, while the old one grows rusty not because anything actively erased it, but because nothing in months of daily practice ever asked it to be used. Gradient descent has exactly that character — a weight update only ever responds to whatever the current batch of training text is showing it, with no separate mechanism protecting capabilities the current batch happens not to touch — which is precisely why continual pretraining's replay mixture works the way regular practice does for the dormant language: deliberately keep exercising the general-purpose skill alongside the new specialism, rather than trusting it to survive neglect on its own.</p>`)}
<p>Continual pretraining buys something LoRA (§4.13) fundamentally cannot: LoRA adapts a model's <i>behaviour</i> within a low-rank subspace of its existing weights, but it has nowhere near the capacity to install large amounts of genuinely new factual knowledge the base model never saw. When the honest answer to "can we make the model actually know our internal product catalogue" is no, retrieval is insufficient, and a lightweight adapter is not enough, continual pretraining — expensive, and requiring real care around the replay mixture and learning rate — is the answer that is left.</p>

${H.probe([
      ['What is the pretraining objective?', 'Next-token categorical cross-entropy — the NLL of a Categorical label (§1.5), applied at every position of every sequence, turned into $s$ simultaneous training signals per pass by the causal mask (§4.4). Nothing more exotic sits underneath it; the model\'s apparent reasoning and world knowledge are a side effect of what it takes to get this simple prediction right at scale.'],
      ['Bigger corpus or cleaner corpus?', 'Cleaner, at equal compute — deduplication and quality filtering both beat raw volume, and dedup also cuts verbatim memorisation. A low-information token is a forward pass not spent learning anything useful, so a smaller dense corpus outperforms a larger diluted one of the same token count.'],
      ['When does synthetic data fail?', 'When it is a closed loop with no real data anchoring it — repeated training on a model\'s own unfiltered outputs measurably narrows the distribution (Shumailov et al., 2024) generation after generation. It works reliably when filtered against something real: a verifier, a stronger teacher model, or genuine human data.']
    ])}`,
    labs: {
      corpus: function (host) {
        const st = Viz.controls(host, [
          { k: 'raw', label: 'raw crawl (trillions of tokens)', min: 5, max: 200, step: 5, value: 100, fmt: v => v + 'T' },
          { k: 'dedup', label: 'deduplication strength', min: 0, max: 1, step: .05, value: .6, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'quality', label: 'quality-filter strictness', min: 0, max: 1, step: .05, value: .5, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'anneal', label: 'final high-quality anneal', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'tokens', label: 'usable tokens', cls: 'key' }, { k: 'quality', label: 'relative quality per token', cls: 'good' },
          { k: 'memo', label: 'verbatim memorisation risk' }, { k: 'model', label: 'Chinchilla-optimal model size' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const stages = [
              { l: 'raw crawl', f: 1, c: T.line },
              { l: 'language ID + boilerplate strip', f: .55, c: T.faint },
              { l: 'deduplication (MinHash)', f: .55 * (1 - .45 * st.dedup), c: T.blue },
              { l: 'quality classifier', f: .55 * (1 - .45 * st.dedup) * (1 - .55 * st.quality), c: T.red },
              { l: 'mixture weighting', f: .55 * (1 - .45 * st.dedup) * (1 - .55 * st.quality) * .92, c: T.text },
              { l: st.anneal ? 'final anneal on the best data' : '(no anneal)', f: .55 * (1 - .45 * st.dedup) * (1 - .55 * st.quality) * .92 * (st.anneal ? 1 : 1), c: T.amber }
            ];
            const bx = 20, bw = w - 260;
            stages.forEach((s, i) => {
              const y = 26 + i * 40;
              ctx.fillStyle = s.c;
              ctx.globalAlpha = i === 5 && !st.anneal ? .25 : 1;
              ctx.fillRect(bx, y, bw * s.f, 26);
              ctx.globalAlpha = 1;
              ctx.fillStyle = T.text; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              ctx.fillText(s.l, bx + bw * s.f + 10, y + 13);
              ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
              ctx.fillText((st.raw * s.f).toFixed(1) + 'T', bx + 6, y + 13);
            });
            const finalT = st.raw * stages[4].f;
            const quality = 1 + .55 * st.quality + .25 * st.dedup + (st.anneal ? .12 : 0);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('yellow tail = the high-quality anneal phase: a small fraction of tokens, a disproportionate share of the final quality', bx, 26 + 6 * 40 + 6);
            out({
              tokens: finalT.toFixed(1) + 'T',
              quality: '×' + quality.toFixed(2),
              memo: st.dedup > .5 ? 'low' : st.dedup > .2 ? 'moderate' : 'high',
              model: (finalT * 1e12 / 20 / 1e9).toFixed(0) + 'B'
            });
          }
        });
        Viz.note(host, 'The last readout applies Chinchilla’s 20 tokens per parameter (§4.10) to whatever corpus survives your filters — which is the honest way to discover that the model size you can train is set by your <i>data pipeline</i>, not your GPU budget.');
      }
    },
    quiz: [
      {
        q: 'Deduplication improves models mainly by…',
        options: ['reducing training time only', 'improving quality per token and sharply reducing verbatim memorisation', 'increasing vocabulary coverage', 'removing toxic content'],
        answer: 1,
        why: 'Training on the same text a thousand times, when a page is repeated boilerplate, gives the model a thousand gradient updates reinforcing that exact string, so removing near-duplicates both flattens that over-representation and is the single most effective lever against verbatim memorisation — precisely the two effects §4.9.2 documents from MinHash-based deduplication. "Reducing training time only" is the tempting wrong answer because fewer surviving tokens genuinely does mean fewer forward passes and a shorter run, a true side effect — but that framing mistakes an incidental saving for the actual mechanism, which is fixing a distortion in what the model learns, not merely doing less of it. Deduplication\'s real payoff is measured in quality per token, not tokens per second: a smaller, deduplicated corpus reliably beats a larger, duplicate-laden one at matched compute, the same principle §4.9.2\'s whole funnel discussion turns on. Vocabulary coverage and toxic content are separate concerns handled by language identification and content filters, not by MinHash-style near-duplicate detection at all.'
      },
      {
        q: 'Training a model repeatedly on its own unfiltered outputs leads to…',
        options: ['steady improvement', 'model collapse — diversity and then quality degrade', 'faster convergence', 'better calibration'],
        answer: 1,
        why: 'Shumailov et al.\'s 2024 Nature paper demonstrated this directly: a model\'s own outputs are always a slightly narrower, smoothed copy of the distribution it was trained on, so training the next generation purely on that copy — with no fresh real data anchoring it back to reality — repeats the narrowing each time, like photocopying a photocopy, until the rare, tail information that made the original data rich is lost. "Steady improvement" is the tempting wrong answer because each individual generation\'s outputs look fluent and reasonable in isolation, exactly what you would expect if training on model outputs simply propagated whatever the model already knew — but fluency is not the same as preserving the richness of the true distribution, and it is the diversity that degrades first, quality only later once enough of the tail is gone. §4.9.3\'s contrast is the general principle: synthetic data works precisely when it is filtered against something real — a verifier, a stronger teacher model, human data — and fails exactly when it becomes a closed loop with nothing external anchoring it.'
      }
    ],
    cards: [
      { q: 'Pretraining objective', a: 'Next-token categorical cross-entropy — the NLL of a Categorical label.' },
      { q: 'Corpus funnel', a: 'Language ID → dedup (MinHash) → quality classifier → mixture weights → high-quality anneal; decontaminate before training.' },
      { q: 'Synthetic data, three mechanisms', a: 'Distillation from a stronger model, self-instruct bootstrapping, verifier-filtered generation. Fails as a closed loop.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.10 */
  ML.section({
    id: 'scaling-laws', track: 'llm', num: '4.10',
    title: 'Scaling laws, Chinchilla, compute- vs inference-optimal',
    lede: 'One formula lets you sanity-check any training claim in seconds — and one distinction explains why everyone now trains past the compute-optimal point.',
    html: `
<p>Two questions face anyone about to spend a large amount of money on a training run. Given a fixed compute budget, how good a model can it possibly produce — and given that answer, how should the budget actually be split between making the model bigger and showing it more data? Both questions turned out to have answers stable enough to plan a nine-figure training run around, and this section derives the accounting behind both.</p>

<h2><span class="sn">4.10.1</span> The accounting</h2>
<p>Empirically, as you train larger models on more data with more compute, loss falls in a remarkably smooth, predictable way — not noisily, not with sudden jumps, but following what looks like a straight line once you plot log(loss) against log(compute) rather than the raw numbers. That is worth unpacking rather than taking as a slogan. A straight line on log-log axes means $\\log L = c - \\alpha\\log N$ for some constants $c$ and $\\alpha$, and exponentiating both sides recovers $L = e^{c} N^{-\\alpha} = A N^{-\\alpha}$ — loss falls as a constant power of $N$. That is exactly what a <b>power law</b> is, and it is a much stronger statement than "loss goes down as you scale up": a power law has no natural scale of its own, so the same relationship that held between a 10M-parameter model and a 100M-parameter one keeps holding, unchanged in shape, between a 10B model and a 100B one. That is the practical payoff. You can fit $\\alpha$ from a handful of small, cheap runs — models you can afford to train many of in an afternoon — and extrapolate the fitted line out to a run a thousand times larger, with real, if imperfect, confidence, before you have spent the money to run it. That predictability, more than any single number in this section, is what turned frontier training from a gamble into a budgeting exercise.</p>
${H.intuition(`<p>Why would loss ever follow a power law at all, rather than falling in some more arbitrary way? One useful intuition: each additional parameter, or additional token, buys you a shrinking, but never fully vanishing, share of what is left to learn. Language has structure at every scale — letter frequencies, word choice, phrase-level idiom, paragraph-level coherence, document-level facts — and a model with more capacity or more data keeps picking up structure at ever finer, ever rarer scales, the way each additional digit of $\\pi$ you memorise captures a genuinely new, but decreasingly consequential, piece of information. There is no single scale at which the returns suddenly stop; they just keep thinning out geometrically. That is the qualitative signature a power law encodes precisely, and it is why scaling curves are remarkably straight on log-log axes across such an enormous range, rather than flattening out the way a "close to done" process would.</p>`)}
<p>§4.5 already derived the compute cost of one token of training: $2N$ FLOPs forward, $4N$ backward, $6N$ total. Multiply by $D$ training tokens and you get the standard accounting used everywhere in this course and in nearly every published scaling-law paper:</p>
$$C \\approx 6ND \\text{ FLOPs}$$
<p>This single formula, entirely earned rather than asserted — you derived the 6 yourself in §4.5 — lets you sanity-check any claimed training run in the time it takes to multiply three numbers: quote a parameter count and a token count for a model, and $6ND$ tells you the compute that run must have consumed, checkable in seconds against whatever compute budget was claimed alongside it.</p>
${H.pitfall('$C\\approx6ND$ is a first-order approximation, not an identity. It counts only the dense matrix multiplies and silently drops attention\'s own $O(s^2)$ cost (fine when the hidden size dominates the sequence length, which is most of training, but not for very long contexts — §4.4), plus every FLOP spent on communication between devices rather than arithmetic (§4.11 is where that second cost stops being ignorable). Treat $6ND$ as accurate to within a few percent for a normal-context dense model, and as a genuine undercount the moment either of those two assumptions stops holding.')}

<h2><span class="sn">4.10.2</span> Chinchilla: deriving the split, not quoting it</h2>
<p>Given a fixed compute budget $C$, the accounting above leaves one free choice: how to split $C=6ND$ between a bigger $N$ and a bigger $D$, since increasing either one at fixed compute forces the other down. Hoffmann et al. (2022) — the paper universally referred to by its model's codename, "Chinchilla" — fit loss as a function of both independently:</p>
$$L(N,D) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}}$$
<p>Read this the way §1.5's exponential family reading applies generally: $E$ is the irreducible loss a model of this architecture could never beat even with infinite parameters and infinite data — the entropy of language itself, plus whatever the tokenizer and architecture themselves leave on the table; $A/N^{\\alpha}$ is the loss you are still paying because your model is not infinitely wide, shrinking as $N$ grows; $B/D^{\\beta}$ is the loss you are still paying because you have not seen infinitely much data, shrinking as $D$ grows. The two shrinking terms are independent contributions, fit separately from hundreds of small training runs at varying $N$ and $D$ — and this is the question the whole section has been building to: given a fixed compute budget, exactly how should you trade one shrinking term against the other?</p>
<p>This is the equality-constrained optimisation §1.12.3 built the machinery for, run here on a genuine research question rather than a toy fence. Minimise $L(N,D)$ subject to $6ND=C$ by folding the constraint into a Lagrangian with a multiplier $\\nu$ — no sign restriction on $\\nu$ this time, because $6ND=C$ is a thin surface you are pinned to everywhere, not a one-sided fence you are merely pressed against, exactly the equality case §1.9.5 and §1.12.3 both handled already.</p>
${H.deriv('the compute-optimal split N*(C), D*(C)', [
      ['$\\mathcal L(N,D,\\nu) = E + AN^{-\\alpha} + BD^{-\\beta} + \\nu(6ND - C)$', 'Fold the constraint into the objective with multiplier $\\nu$, the same move §1.12.3 used for an equality constraint.'],
      ['$\\dfrac{\\partial\\mathcal L}{\\partial N}=0:\\ \\ \\alpha A N^{-\\alpha-1} = 6\\nu D \\qquad\\quad \\dfrac{\\partial\\mathcal L}{\\partial D}=0:\\ \\ \\beta B D^{-\\beta-1} = 6\\nu N$', 'Both partials vanish at a stationary point. Read each line as a marginal statement: the left side is the loss you claw back per unit you grow N (or D); the right side prices that unit in the shared currency of compute, via the common multiplier ν.'],
      ['$\\dfrac{\\alpha A N^{-\\alpha-1}}{\\beta B D^{-\\beta-1}} = \\dfrac{D}{N} \\ \\ \\Longrightarrow\\ \\ D^{\\beta} = \\dfrac{\\beta B}{\\alpha A}\\,N^{\\alpha}$', 'Divide the two stationarity conditions to eliminate ν entirely. This is the economically meaningful step: it says the marginal loss reduction per FLOP must be identical whichever resource that FLOP buys — if growing N returned more per FLOP than growing D, moving a FLOP from D to N would improve the loss for free, so a genuine optimum cannot have that imbalance.'],
      ['$N^\\star(C) = \\left[\\dfrac{C}{6}\\left(\\dfrac{\\alpha A}{\\beta B}\\right)^{1/\\beta}\\right]^{\\beta/(\\alpha+\\beta)}, \\qquad D^\\star(C) = \\dfrac{C}{6N^\\star(C)}$', 'Substitute the relation above into the constraint $ND=C/6$ and solve for N — two equations, two unknowns, routine algebra once ν is gone. The exponent on C, β/(α+β), is the number that decides how N should grow as the budget grows.']
    ], 'Read the exponents, not just the formula. $N^\\star(C)\\propto C^{\\beta/(\\alpha+\\beta)}$ and $D^\\star(C)\\propto C^{\\alpha/(\\alpha+\\beta)}$. If $\\alpha=\\beta$ exactly, both exponents are $0.5$: parameters and tokens grow at <i>exactly</i> the same rate as compute grows, forever, and $D^\\star/N^\\star$ is a genuine constant — a single number, independent of scale. That clean symmetric case is the honest content behind "scale $N$ and $D$ together in fixed proportion", and it is worth seeing it fall out of the algebra rather than being asserted, because the next paragraph is about exactly how far the real, fitted exponents are from that tidy case.')}
${H.analogy(`<p>The stationarity condition above — equalise the marginal return per FLOP across every use of it — is the identical logic behind spending a fixed grocery budget well. You do not ask "is bread good?" in isolation; you ask "does the next dollar buy me more satisfaction as bread or as cheese?", and you keep shifting spend from whichever is worse at the margin to whichever is better, until the two are tied. Stop before they are tied and you are leaving free improvement on the table — buy one more unit of the currently-better option and you are strictly happier for the same money. The Lagrange multiplier $\\nu$ is playing the role of "value per dollar" here, in a currency of FLOPs rather than pounds, and the whole derivation is just insisting that a real optimum cannot have one resource earning a better return than the other.</p>`)}

<p>Now plug in numbers. Hoffmann et al.'s own reported fit for approach 3 gives $E\\approx1.69$, $A\\approx406.4$, $\\alpha\\approx0.34$, $B\\approx410.7$, $\\beta\\approx0.28$ — the same coefficients the lab below draws its contours from. The exponent on $C$ for $N^\\star$ works out to $\\beta/(\\alpha+\\beta)=0.28/0.62\\approx0.45$, and for $D^\\star$ to $\\alpha/(\\alpha+\\beta)\\approx0.55$ — close to the clean symmetric $0.5/0.5$ case, but not exactly it, because the fitted $\\alpha$ and $\\beta$ are close but not identical. That near-miss matters in a way worth stating plainly rather than glossing over: because the two exponents differ, $D^\\star/N^\\star$ is <i>not</i> a true constant under these coefficients — it drifts slowly as $C$ grows, rather than staying pinned at one number for every scale.</p>
${H.worked('working the split at a specific budget', `<p>Take $C=10^{23}$ FLOPs, a realistic mid-sized frontier run, and the coefficients above. Solving the closed form from the derivation: $N^\\star\\approx1.46\\times10^{10}$ (14.6B parameters), $D^\\star = C/(6N^\\star) \\approx 1.14\\times10^{12}$ (1.14T tokens) — a ratio of about <b>78 tokens per parameter</b> at this specific budget, not 20.</p>
<p>Recompute the same closed form at $C=10^{21}$: $N^\\star\\approx1.8\\times10^9$ (1.8B), $D^\\star\\approx9.1\\times10^{10}$ (91B tokens), a ratio of about <b>50:1</b>. And at $C=10^{19}$, close to where most of the paper's actual training runs sat: the ratio comes out near <b>20–30:1</b> — which is exactly where the famous number comes from. The ~20:1 headline was measured accurately, at the scale it was measured at. Solving the same fitted formula analytically at the much larger budgets frontier labs train at today extrapolates it into a regime where $\\alpha\\ne\\beta$ has had far more compute-doublings to compound, and the implied optimal ratio has visibly drifted well past 20:1 by the time $C$ reaches $10^{23}$–$10^{24}$.</p>`)}
${H.flag('Flag when you cite this. The qualitative conclusion — scale parameters and tokens together, not the way 2020-era models did, which trained far too large relative to how little data they saw — is robust and repeatedly confirmed. The specific "~20 tokens per parameter" number is an approximation that held well at the compute scale it was fitted on; solved analytically at much larger budgets using the paper\'s own reported coefficients, the optimal ratio drifts upward rather than staying fixed, exactly because the fitted α and β are close but not equal. Epoch AI\'s 2024 replication attempt flagged a related concern from the other direction — that refitting the coefficients from the underlying data gives numbers that do not cleanly reproduce the original paper\'s own headline recommendation either. Quote the ratio as an order-of-magnitude rule of thumb valid near the scale it was measured at, never as a universal constant, and say so out loud when a claim leans on the exact number.')}
${H.key('A scaling law is a fit, not a law of physics. It is trustworthy for interpolation — inside the range of scales it was measured on — and only cautiously trustworthy for extrapolation many orders of magnitude beyond that range, because a fitted exponent that is off by a few hundredths compounds into a large drift once you raise it to a large enough power.')}
${H.history(`<p>The scaling-law project itself has already been revised once in public, which is worth knowing before treating any single paper's numbers as final. Kaplan et al. (OpenAI, 2020) fit the first widely-cited power laws for transformer language models and reached a specific, actionable conclusion: given a fixed compute budget, you should grow model size aggressively and grow data only modestly — their fits recommended something closer to a few billion tokens for a multi-billion-parameter model, an order of magnitude below what Chinchilla later argued for. That recommendation shaped how an entire generation of large models, GPT-3 among them, was actually trained.</p>
<p>Hoffmann et al. (2022) revisited the fit with a specific methodological fix: they controlled for the learning-rate schedule's length as a confound, something the earlier fits had not isolated cleanly, since a cosine schedule tuned for a short run and then simply cut short for a compute-matched comparison systematically under-trains the smaller model in the comparison and makes bigger-at-fixed-compute look better than it really was. Correcting for that single confound moved the fitted optimum a full order of magnitude in the data direction — the ~20:1 result. The lesson is not "trust Chinchilla and not Kaplan"; it is that a scaling law is only as good as the experimental protocol it was fit from, and a subtle confound in how you ran the small-scale experiments can bias the extrapolated recommendation for a nine-figure training run in a specific, costly direction.</p>`)}

<p><b>What you are looking at.</b> A contour map of the Chinchilla loss surface over log parameters (x-axis) and log tokens (y-axis), drawn from the exact coefficients used in the derivation above — each faint grey curve is an iso-loss contour, every point on it achieving the identical loss. The blue curve is your chosen compute budget's iso-compute line, $D=C/(6N)$; the dashed red ray is the 20-tokens-per-parameter rule of thumb, drawn as a fixed reference rather than assumed to be correct. Point A, marked in red, is now computed the same way the derivation above computes it — the actual tangent point of this specific loss surface — rather than forced onto the 20:1 ray by assumption.</p>
<p><b>What to do with it.</b> Start at the default budget and note where point A sits relative to the dashed 20:1 ray — close, but visibly off it already. Now drag the compute-budget slider up by several orders of magnitude and watch point A visibly pull away from the red ray as it climbs the iso-compute curve, exactly the drift the worked example above computed by hand at $C=10^{19}$, $10^{21}$ and $10^{23}$.</p>
<p><b>The thing genuinely worth noticing.</b> The gap between point A and the 20:1 ray is not a rendering choice or a rounding error — it is the direct, visible consequence of $\\alpha\\ne\\beta$ in the fitted coefficients, the same near-miss the derivation flagged in algebra before you ever saw it on a chart. A scaling-law contour plot that forced its "optimal" point onto a fixed ratio regardless of budget would be quietly asserting something the underlying fit does not actually say. Separately, use control B to slide along the <i>same</i> iso-compute curve away from point A, to any ratio you choose — same total training cost, a visibly higher-loss contour the further you stray from A, and this is the trade §4.10.3 spends the rest of the section justifying anyway: paying that training-loss cost on purpose, because inference dominates lifetime cost.</p>

${H.lab('chinchilla', 'The iso-loss surface, and where your budget lands', 'Move the compute budget and watch the true tangent point — computed from the fitted loss surface, not assumed — pull away from the 20:1 rule of thumb as the budget grows. Then move along a single iso-loss contour to see the inference-optimal trade: comparable loss, a smaller model, far more data — and a much cheaper deployment.')}

<h2><span class="sn">4.10.3</span> Inference-optimal is a different question</h2>
<p>Chinchilla answers a specific, narrower question than it is often credited with: given a fixed compute budget, which split of parameters and tokens minimises <i>training</i> loss per training FLOP. It says nothing about the cost of everything that happens after training finishes, and for any model that will actually be deployed, that omission turns out to be the whole story. A model is trained once and served, potentially, billions of times; §4.7 already showed that decode cost scales with the number of parameters $N$ (every weight has to be read for every generated token), so total lifetime cost is training cost <i>plus</i> inference cost accumulated over every request the model will ever serve — and for a model with enough users, that second term dwarfs the first. If lifetime cost is what you actually want to minimise, and inference cost per token scales with $N$ while training cost was already fixed by the compute-optimal point, the arithmetic favours moving to a <i>smaller</i> $N$ than Chinchilla recommends and compensating with proportionally more training tokens $D$ to reach comparable loss — deliberately "overtraining" relative to the training-only optimum, because a smaller model is cheaper on every single one of the millions of inference calls that follow.</p>
<p>This is now standard practice for anything actually deployed, and the numbers involved are not subtle. Llama-3-8B was trained on roughly 15 trillion tokens against 8 billion parameters — about <b>1,875 tokens per parameter</b>, some 90 times the Chinchilla ratio of 20:1. The one-line answer to "why train so far past the compute-optimal point" is a sentence worth having ready in an interview: <mark>you pay training compute once, and you pay inference compute forever, so the more the model will be served, the more it pays to shrink it — at extra training cost — before you ever put it in front of a user.</mark></p>
${H.practice(`<p>Two practical limits sit on either side of "just overtrain more". On one side, the power-law shape from §4.10.1 means the return on extra tokens at fixed $N$ keeps shrinking — the $B/D^\\beta$ term in the Chinchilla fit is already most of the way to its floor by the time you are tens of thousands of tokens per parameter deep, so past some point you are burning enormous additional training compute for a fractional loss improvement that a slightly larger $N$ would have bought far more cheaply. Overtraining is a genuine lever, not an unbounded one.</p>
<p>On the other side sits a harder constraint: <b>you can run out of unique, high-quality tokens before you run out of willingness to spend compute.</b> §4.9.2's corpus funnel already showed how much of a raw crawl gets discarded by quality filtering and deduplication, and the highest-quality sources — well-edited text, code with real tests, curated books — exist in genuinely finite supply. When $D$ pushed for by the inference-optimal calculus above exceeds what the data pipeline can supply without repeating, you are in <b>data-constrained scaling</b>: either train multiple epochs over the same tokens, which behaves like fresh data for the first few passes and then yields rapidly diminishing, occasionally even harmful, returns as memorisation sets in, or spend real engineering effort growing the pool of genuinely distinct high-quality tokens rather than the model. This is precisely where §4.9.3's synthetic-data mechanisms earn their keep, and precisely where §4.9.3's model-collapse warning is sharpest: reaching for synthetic data to relieve a data-constrained overtraining run is exactly the situation where filtering it against something real, rather than looping it against itself, actually matters.</p>`)}

<p><b>What you are looking at.</b> A live compute calculator: enter any two of parameter count, token count and compute budget, and it derives the third from $C=6ND$, alongside the resulting GPU-hours, wall-clock time at your chosen cluster size, and an approximate dollar cost.</p>
<p><b>What to do with it.</b> Set 7B parameters and 2T tokens and confirm the compute readout lands on $8.4\\times10^{22}$ FLOPs — the same figure the quiz below asks you to derive by hand. Then try the Llama-3-8B preset button and compare its tokens-per-parameter readout against the Chinchilla-ratio preset at the same total compute.</p>
<p><b>The thing genuinely worth noticing.</b> Two runs at the <i>same</i> total compute — one at the 20:1 Chinchilla ratio, one at the ~1,875:1 Llama-3 ratio — land at very different model sizes for the same training bill, and this is precisely the trade §4.10.2's lab visualised geometrically: moving along an iso-compute curve away from the tangent point costs training-loss quality, but if the resulting model is going to be queried by a billion users, that training-time quality cost can still be the cheaper choice once inference is counted.</p>

${H.lab('budget', 'Training budget calculator', 'Enter any two of parameters, tokens and compute and read off the third, with GPU-hours and a rough cost. Check it against a published run — the arithmetic is the same one used to plan them.')}

${H.more('a third axis: spending compute at inference time instead of training time', `<p>Everything in this section trades $N$ against $D$ inside a fixed training budget — but §4.15.3 introduces a scaling axis this whole framework is silent about: given a fixed <i>trained</i> model, spend more compute per query at generation time instead — longer chains of thought, self-consistency voting across many samples, search over reasoning paths — and accuracy on many tasks keeps climbing well past what more training compute alone would buy at the same total cost. That is not a contradiction of anything derived above; $C\\approx6ND$ only ever counted training FLOPs, and test-time compute is a genuinely separate line item on the bill, one paid per request rather than once. It is also, empirically, most reliable on exactly the tasks with a cheap automatic verifier — maths, code — the same domains §4.9.3 already flagged as where synthetic data works best, for the identical underlying reason: a checker to filter against. Whether frontier labs should now be solving a joint allocation problem across three resources, training parameters, training tokens and inference-time compute, rather than the two-resource problem this section derives, is an open, actively studied question rather than a settled one.</p>`)}

${H.probe([
      ['Chinchilla in one line?', 'For a fixed training budget, minimise loss subject to C=6ND by equalising the marginal loss reduction per FLOP across N and D — a standard equality-constrained optimisation (§1.12.3). At the scale the paper fit it on, that balance sits near 20 tokens per parameter; solved analytically at much larger budgets with the paper\'s own coefficients, the ratio drifts upward, because the fitted exponents α and β are close but not identical (§4.10.2).'],
      ['Then why does everyone train past it?', 'Deployment cost is dominated by inference, and inference cost per token scales with parameter count (§4.7); a smaller, overtrained model is more expensive to train but cheaper on every one of the millions of requests that follow, so total lifetime cost favours shrinking N below the training-optimal point.'],
      ['Compute for a 7B on 2T tokens?', '$6ND \\approx 6\\cdot7\\times10^9\\cdot2\\times10^{12} \\approx 8.4\\times10^{22}$ FLOPs — the derived-not-quoted 6 from §4.5, applied directly.'],
      ['Why does the tokens-per-parameter ratio fail to stay constant?', 'Because the compute-optimal exponents are $\\beta/(\\alpha+\\beta)$ for N and $\\alpha/(\\alpha+\\beta)$ for D — equal only if $\\alpha=\\beta$ exactly. The fitted values (≈0.34 and ≈0.28) are close but not equal, so $N^\\star$ and $D^\\star$ grow at very slightly different rates with compute, and the ratio between them drifts rather than staying fixed across scales.']
    ])}`,
    labs: {
      chinchilla: function (host) {
        const st = Viz.controls(host, [
          { k: 'logC', label: 'compute budget (FLOPs)', min: 19, max: 26, step: .1, value: 23, fmt: v => '10^' + v.toFixed(1) },
          { k: 'ratio', label: 'tokens per parameter', min: 5, max: 2000, step: 5, value: 20, fmt: v => v + ':1' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nopt', label: 'compute-optimal N', cls: 'key' }, { k: 'dopt', label: 'compute-optimal D' },
          { k: 'nyour', label: 'N at your ratio' }, { k: 'dyour', label: 'D at your ratio' }, { k: 'infer', label: 'relative inference cost', cls: 'good' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const C = Math.pow(10, st.logC);
            // Chinchilla approach-3 fit (Hoffmann et al. 2022): L = E + A/N^a + B/D^b
            const E = 1.69, A = 406.4, a = 0.34, B = 410.7, b = 0.28;
            const loss = (N, D) => E + A / Math.pow(N, a) + B / Math.pow(D, b);
            // true tangent point: solve the stationarity conditions derived in §4.10.2,
            // N*(C) = [ (C/6)(aA/bB)^(1/b) ]^(b/(a+b)), D*(C) = C/(6N*) — NOT a fixed ratio.
            function tangent(Cc) {
              const Nt = Math.pow((Cc / 6) * Math.pow(a * A / (b * B), 1 / b), b / (a + b));
              return { N: Nt, D: Cc / (6 * Nt) };
            }
            const P = Viz.plot(ctx, w, h, { xd: [8, 12.2], yd: [10, 14] })
              .frame({ xlabel: 'log₁₀ parameters N', ylabel: 'log₁₀ tokens D', xfmt: v => '10^' + v.toFixed(0), yfmt: v => '10^' + v.toFixed(0) });
            P.contours((ln, ld) => loss(Math.pow(10, ln), Math.pow(10, ld)), [1.9, 2.0, 2.15, 2.3, 2.5, 2.8, 3.2], { color: T.faint, alpha: .6, nx: 70, ny: 60 });
            const opt = tangent(C);
            P.clip(() => {
              // 20 tokens/param rule of thumb, drawn as a fixed reference — not assumed correct
              P.fn(ln => Math.log10(20 * Math.pow(10, ln)), { color: T.red, width: 2, dash: [6, 4] });
              // iso-compute curve: D = C/(6N)
              P.fn(ln => Math.log10(C / (6 * Math.pow(10, ln))), { color: T.blue, width: 2.6 });
              // A: the true tangent point of THIS loss surface, computed, not assumed
              P.dots([[Math.log10(opt.N), Math.log10(opt.D)]], { r: 6, color: T.red, stroke: true });
              P.text(Math.log10(opt.N), Math.log10(opt.D), '  A · true tangent point', { color: T.red, font: '11px ui-sans-serif' });
              const Nyour = Math.sqrt(C / (6 * st.ratio)), Dyour = st.ratio * Nyour;
              P.dots([[Math.log10(Nyour), Math.log10(Dyour)]], { r: 6, color: T.green, stroke: true });
              P.text(Math.log10(Nyour), Math.log10(Dyour), '  B · your ratio', { color: T.green, font: '11px ui-sans-serif' });
            });
            const Nyour = Math.sqrt(C / (6 * st.ratio)), Dyour = st.ratio * Nyour;
            out({
              nopt: (opt.N / 1e9).toFixed(1) + 'B', dopt: (opt.D / 1e12).toFixed(2) + 'T',
              nyour: (Nyour / 1e9).toFixed(1) + 'B', dyour: (Dyour / 1e12).toFixed(2) + 'T',
              infer: '×' + (Nyour / opt.N).toFixed(2)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Chinchilla 20:1', primary: true, on: () => { st.$set('ratio', 20); S.redraw(); } },
          { label: 'Llama-3-8B ≈ 1875:1', on: () => { st.$set('ratio', 1875); S.redraw(); } }
        ]);
        Viz.note(host, 'Point A is the actual tangent point of this fitted surface (§4.10.2\'s derivation), not the 20:1 ray by assumption — watch it pull away from the dashed red line as you raise the budget. Point B sits on the same iso-compute curve at whatever ratio you choose: a much smaller model trained on far more tokens, comparable loss, and the "relative inference cost" readout is what you pay forever afterwards. That comparison is why the field trains past compute-optimal at all.');
      },

      budget: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'parameters (B)', min: .1, max: 700, step: .1, value: 8, fmt: v => v.toFixed(1) + 'B' },
          { k: 'D', label: 'training tokens (T)', min: .1, max: 30, step: .1, value: 15, fmt: v => v.toFixed(1) + 'T' },
          { k: 'gpu', label: 'GPU peak (TFLOP/s)', min: 100, max: 2500, step: 10, value: 990, fmt: v => v },
          { k: 'mfu', label: 'MFU achieved', min: .15, max: .65, step: .01, value: .42, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'ngpu', label: 'GPUs', min: 8, max: 32768, step: 8, value: 1024, fmt: v => v.toLocaleString() },
          { k: 'cost', label: '$ per GPU-hour', min: .5, max: 10, step: .1, value: 2.5, fmt: v => '$' + v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'flops', label: 'total FLOPs (6ND)', cls: 'key' }, { k: 'gpuh', label: 'GPU-hours' },
          { k: 'days', label: 'wall-clock days' }, { k: 'usd', label: 'approximate cost' }, { k: 'ratio', label: 'tokens per parameter' }
        ]);
        const S = Viz.surface(host, {
          height: 240,
          draw: function (ctx, w, h, T) {
            const N = st.N * 1e9, D = st.D * 1e12;
            const flops = 6 * N * D;
            const gpuSec = flops / (st.gpu * 1e12 * st.mfu);
            const gpuH = gpuSec / 3600;
            const days = gpuH / st.ngpu / 24;
            ctx.font = '14px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const lines = [
              ['C = 6ND', '= 6 × ' + st.N.toFixed(1) + 'e9 × ' + st.D.toFixed(1) + 'e12', T.text],
              ['', '= ' + flops.toExponential(2) + ' FLOPs', T.blue],
              ['GPU-seconds', '= C / (peak × MFU) = ' + (gpuSec / 1e6).toFixed(1) + 'M s', T.text],
              ['GPU-hours', '= ' + Math.round(gpuH).toLocaleString(), T.text],
              ['wall clock', '= ' + days.toFixed(1) + ' days on ' + st.ngpu.toLocaleString() + ' GPUs', T.green],
              ['cost', '≈ $' + (gpuH * st.cost / 1e6).toFixed(2) + 'M', T.amber]
            ];
            lines.forEach((L, i) => {
              ctx.fillStyle = T.muted; ctx.fillText(L[0], 18, 20 + i * 30);
              ctx.fillStyle = L[2]; ctx.font = 'bold 14px ui-monospace, monospace';
              ctx.fillText(L[1], 150, 20 + i * 30);
              ctx.font = '14px ui-monospace, monospace';
            });
            out({
              flops: flops.toExponential(2), gpuh: Math.round(gpuH).toLocaleString(),
              days: days.toFixed(1), usd: '$' + (gpuH * st.cost / 1e6).toFixed(2) + 'M',
              ratio: (D / N).toFixed(0) + ':1'
            });
          }
        });
        Viz.note(host, 'Set 7B and 2T tokens: 8.4×10²² FLOPs, the number quoted in the interview box. The MFU slider is where reality lives — 42% is a good large run, and the difference between 42% and 25% is months of wall clock.');
      }
    },
    quiz: [
      {
        q: 'Training compute for a 7B model on 2T tokens is approximately…',
        options: ['$8.4\\times10^{19}$', '$8.4\\times10^{22}$', '$1.4\\times10^{25}$', '$6\\times10^{12}$'],
        answer: 1,
        why: 'Apply the accounting §4.10.1 derives from §4.5\'s per-token cost — 2N FLOPs forward and 4N backward, or 6N total, multiplied by D tokens — and $6\\times7\\times10^9\\times2\\times10^{12}=8.4\\times10^{22}$ FLOPs falls straight out; the whole exercise is checking a claimed training run against three numbers you can multiply in your head. $1.4\\times10^{25}$ is the tempting distractor for anyone who multiplies N and D correctly but forgets the factor of 6 that turns parameter-times-token counts into an actual FLOP count. $8.4\\times10^{19}$ makes the mirror-image mistake, dropping three orders of magnitude by treating N or D as though it were already in the wrong unit. The general principle is that this single formula, earned rather than asserted back in §4.5, is what lets you sanity-check any published training claim in seconds rather than trusting it on authority.'
      },
      {
        q: 'Why do deployed models train far past the Chinchilla ratio?',
        options: ['Chinchilla was wrong', 'Inference cost dominates lifetime cost, so a smaller overtrained model is cheaper to serve forever', 'Larger datasets are cheaper', 'It improves calibration'],
        answer: 1,
        why: 'A model is trained once but served, potentially, billions of times, and §4.7 already showed decode cost scales with parameter count N because every weight has to be read for every generated token — so once a model has enough users, accumulated inference cost swamps the one-off training cost, and it becomes worth shrinking N below the training-optimal point and compensating with far more tokens D to reach comparable loss, exactly Llama-3-8B\'s ~1,875 tokens-per-parameter ratio against Chinchilla\'s 20:1. "Chinchilla was wrong" is the tempting wrong answer because it sounds like it resolves the apparent contradiction of everyone ignoring the ratio, but Chinchilla correctly answered a narrower question — which split of a fixed compute budget minimises training loss per training FLOP — and training past it does cost real training-time quality, it just buys back far more than that in lifetime serving cost. "Larger datasets are cheaper" and "it improves calibration" are not real mechanisms operating here at all. The general principle, §4.10.3\'s whole point, is that training-optimal and inference-optimal are different questions with different answers, and any deployed model has to answer the second one.'
      }
    ],
    cards: [
      { q: 'Compute accounting', a: '$C \\approx 6ND$ — 2N forward, 4N backward, times D tokens.' },
      { q: 'Chinchilla', a: '≈20 tokens per parameter at fixed compute (70B on 1.4T beat 280B). ⚑ Exact coefficients contested — quote the ratio.' },
      { q: 'Inference-optimal', a: 'Overtrain a smaller model far past 20:1 (Llama-3-8B ≈ 1,875 tok/param) — pay training once, inference forever.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.11 */
  ML.section({
    id: 'distributed', track: 'llm', num: '4.11',
    title: 'Distributed training, and debugging a diverging run',
    lede: 'Three parallelisms, one memory budget, and an ordered list that interviewers ask for precisely because anyone can name the causes.',
    html: `
<p>A 70B model's weights alone are 140 GB in BF16 (§4.7), and training needs far more than the weights: gradients, optimiser state, and activations for every layer of every example in the batch, all resident at once. No single accelerator — the largest currently ships around 80–192 GB — comes close to holding all of that for a frontier-scale model. Training has to be spread across many devices, and there is more than one axis along which "spread across many devices" can mean something different. This section covers the four ways to split the work, the memory arithmetic that decides whether a given split even fits, and the ordered checklist for when a distributed run nonetheless goes wrong.</p>

<h2><span class="sn">4.11.1</span> Four ways to split the work</h2>
${H.table(['Kind', 'Splits', 'Communication', 'Rule of thumb'], [
      ['<b>Data</b>', 'the batch; every GPU holds the whole model', 'one all-reduce of gradients per step', 'Simplest; the default outer layer'],
      ['<b>Tensor</b>', 'individual weight matrices across GPUs', 'heavy, every layer', 'Keep it inside a node — NVLink, not Ethernet'],
      ['<b>Pipeline</b>', 'layers across devices', 'cheap, point-to-point', 'Bubbles unless you micro-batch'],
      ['<b>Expert</b>', 'MoE experts across devices', 'all-to-all', 'Adds a fourth dimension for MoE (§4.8)']
    ])}
<p><b>Data parallelism</b> is the simplest idea and the one to reach for first: give every GPU a full copy of the model and a different slice of the batch, run forward and backward independently on each, and then combine — average, in effect — every GPU's gradients before anyone takes an optimiser step, using a collective communication operation called an <b>all-reduce</b> (every GPU ends the operation holding the same averaged result, computed by passing partial sums efficiently around the group rather than every GPU talking to every other one individually). It communicates once per step, it is easy to reason about, and it is nearly always the outer layer that the other three parallelisms sit inside.</p>
<p><b>Tensor parallelism</b> splits something data parallelism does not touch: a single weight matrix itself. Slice $W_Q$ column-wise across four GPUs, say, and each GPU computes only its slice of the query projection — but the very next operation in the block typically needs the <i>full</i>, reassembled result before it can proceed, so tensor parallelism forces a communication step inside nearly every layer, not once per training step. That chattiness is why the rule of thumb in the table is so firm: tensor parallelism belongs <i>inside</i> a single machine, over NVLink's very high, very low-latency bandwidth between GPUs on the same node, and falls apart in throughput the moment it has to cross to a different physical machine over ordinary networking.</p>
<p><b>Pipeline parallelism</b> splits along a different axis again: rather than splitting any one matrix, give GPU 1 the first several layers of the model, GPU 2 the next several, and so on, so that a forward pass physically walks across devices as it walks through depth. Communication between stages is comparatively cheap — only the activations at each layer boundary need to be passed, point to point, rather than a full collective operation — but the naive version wastes most of its GPUs most of the time: while GPU 2 works on layer 11 for the first micro-batch, GPU 1 has nothing to do until GPU 2 hands anything back, and GPU 3 has nothing to do at all yet. This idle time is called a <b>bubble</b>, and the standard fix is to slice the batch into many small micro-batches and feed them through the pipeline in a staggered, overlapping sequence — an assembly line rather than one part at a time — so that once the pipeline is full, every stage has something to work on in every cycle, and the bubble shrinks to a fixed start-up and drain-down cost rather than a constant tax on every batch.</p>
<p><b>Expert parallelism</b> is §4.8's mixture-of-experts problem viewed from the hardware side: different experts live on different devices, and every forward pass needs an <b>all-to-all</b> exchange — every device sending some tokens out to wherever their chosen expert lives, and receiving others in return — which is a genuinely different communication pattern from an all-reduce's single combine step, and the reason MoE models are harder to serve efficiently at scale than a dense model of comparable active-parameter count. Real large-scale training runs combine all four of these splits at once — often called 3-D or 4-D parallelism — choosing which axis goes inside a node, which crosses nodes, and which wraps the whole cluster, entirely as a function of which communication pattern each axis needs and how much bandwidth is available at each level of the network.</p>

<h3>Putting numbers on "heavy" and "cheap"</h3>
<p>The table's "communication" column is worth more than an adjective. Data parallelism and tensor parallelism both lean on the identical communication primitive, an all-reduce, and its cost is derivable rather than a fact to memorise — which makes it possible to see exactly why one of them tolerates ordinary networking and the other does not, and the answer turns out to hinge on something other than the obvious guess.</p>
${H.deriv('the communication volume of one ring all-reduce', [
      ['split a tensor of $S$ bytes into $n$ equal chunks, one per GPU, arranged in a ring.', 'the standard implementation both data-parallel gradient synchronisation and tensor-parallel activation synchronisation use under the hood.'],
      ['<b>reduce-scatter:</b> $n-1$ ring steps, each GPU sending and receiving one chunk of size $S/n$.', 'after $n-1$ hops around the ring, every GPU holds the fully-reduced sum for its own $1/n$ slice of the tensor — nobody has the whole thing yet, only their piece of it.'],
      ['<b>all-gather:</b> another $n-1$ ring steps, same chunk size, to hand every GPU\'s completed slice to everybody else.', 'the identical communication pattern, run a second time, now distributing rather than reducing — every GPU ends up with the full, correctly-reduced tensor.'],
      ['total sent by any one GPU across both phases $= 2(n-1)\\cdot\\dfrac{S}{n}$.', 'two phases, each moving $n-1$ chunks of size $S/n$ — add them.']
    ], 'As $n\\to\\infty$, $2(n-1)/n \\to 2$: a single GPU\'s own communication volume converges to a fixed $2S$, essentially independent of how many GPUs are in the group. That is the genuinely nice scalability property behind data parallelism — doubling the cluster barely changes what any one GPU has to send — and it is exactly why the table calls data parallelism\'s communication merely "one all-reduce of gradients per step" rather than something that gets worse as you add machines.')}

<p>Apply that formula to the two parallelisms in turn, with concrete numbers rather than symbols, and the "keep it inside a node" rule stops being an assertion.</p>
${H.worked('the same formula, two very different consequences', `
<p><b>Data parallelism.</b> A 70B model's BF16 gradients are $S=140$ GB. At $n=8$ GPUs, $2(n-1)S/n = 2\\times\\tfrac78\\times140 \\approx 245$ GB per GPU; at $n=512$, the same formula gives $\\approx280$ GB — barely more, exactly the convergence the derivation predicts. This happens <b>once per training step</b>, and it can be scheduled to overlap with the tail of the backward pass, since a layer's gradient is ready for its own slice of the all-reduce well before the whole backward pass finishes — so much of that transfer time is hidden behind computation that was going to happen anyway.</p>
<p><b>Tensor parallelism.</b> The same primitive, applied to an activation tensor instead of the gradient tensor: batch 1, sequence 4096, hidden 8192, BF16 — one activation tensor is $1\\times4096\\times8192\\times2\\;\\text{bytes}\\approx67$ MB. At an 8-way tensor-parallel group, one ring all-reduce call moves $2(n-1)S/n\\approx117$ MB. That call happens roughly <b>four times per transformer block</b> (twice in the forward pass, twice in the backward) — once after the attention block's output projection and once after the MLP's down-projection, each direction. Across 80 layers that is 320 separate calls per micro-batch, totalling roughly $320\\times117\\,\\text{MB}\\approx38$ GB.</p>
<p>Notice what that comparison actually says: tensor parallelism\'s <i>total</i> bytes moved, 38 GB, is <i>less</i> than data parallelism\'s 280 GB in this example — and tensor parallelism is still the one that falls apart the instant it leaves a single node. Total bytes was never the deciding factor. Every one of those 320 calls is a hard synchronisation barrier: the very next matrix multiply in the block cannot start until the reduced activation is back, so there is nothing to overlap it with, unlike data parallelism\'s single transfer riding alongside computation that was happening regardless. Wall-clock cost here is dominated by <i>latency</i> — 320 blocking round trips — not by the total payload, which is exactly why NVLink\'s defining advantage over InfiniBand is not only its roughly order-of-magnitude higher bandwidth (on the order of 900 GB/s per GPU for NVLink versus tens of GB/s for a typical inter-node link) but its dramatically lower per-hop latency, and why pipeline and data parallelism — a handful of large, overlappable transfers rather than hundreds of small, blocking ones — tolerate ordinary cross-node networking just fine.</p>`)}
${H.key('The rule "tensor parallelism stays inside a node" is not about how many bytes it moves — it can move fewer than data parallelism does. It is about how many separate, blocking synchronisation points sit in the critical path per step, and low latency matters far more than raw bandwidth when there are hundreds of them.')}

<p>Pipeline parallelism's bubble deserves the same treatment: not just "there is idle time", but exactly how much, and exactly why micro-batching fixes it.</p>
${H.deriv('the pipeline bubble fraction', [
      ['$p$ stages, $m$ micro-batches, each stage taking a fixed time $t$ per micro-batch (forward and backward combined into one unit here for a clean derivation).', 'the setup: a systolic pipeline, one micro-batch entering at a time.'],
      ['the last stage finishes its <i>first</i> micro-batch at time $p\\cdot t$.', 'micro-batch 1 has to pass through all $p$ stages in sequence before stage $p$ can even begin — one hop of $t$ per stage.'],
      ['the last stage then finishes the remaining $m-1$ micro-batches back to back, $t$ apart: total wall-clock $T=(p+m-1)t$.', 'once the pipeline is full, a finished micro-batch arrives at the last stage exactly every $t$, so nothing after the fill phase adds extra waiting.'],
      ['useful work is $m$ micro-batches processed by each of $p$ devices: total useful device-time $=pmt$. total device-time <i>budget</i> is $p\\cdot T=p(p+m-1)t$.', 'device-time budget is what you paid for — $p$ devices, each running for the whole duration $T$ — regardless of whether they were doing anything.'],
      ['bubble fraction $=\\dfrac{p(p+m-1)t - pmt}{p(p+m-1)t} = \\dfrac{p-1}{p+m-1}$.', 'subtract useful work from the budget to get idle time, divide by the budget, and the common factor $pt$ cancels cleanly.']
    ], 'For $m\\gg p$, this collapses to the memorable approximation $(p-1)/m$: double the micro-batch count and you roughly halve the wasted fraction. But treat that approximation with real caution outside the regime it is valid in — at $p=8,\\,m=8$ the exact formula gives a 47% bubble while the crude $(p-1)/m$ approximation claims 87.5%, and at $p=8,\\,m=1$ (no micro-batching at all) the approximation gives a nonsensical 700% while the true answer is a merely catastrophic 87.5%. Use the exact fraction $(p-1)/(p+m-1)$ unless $m$ is at least several times $p$.')}
${H.analogy(`<p>This is a car assembly line, stated precisely. $p$ workstations, each taking time $t$ to do its one job; a single car takes $p\\cdot t$ to go from empty lot to finished vehicle, because it visits every station in sequence with nobody able to start until the previous station hands off. Run only one car through and every station but the one currently working sits idle — the pipeline never fills. Feed cars in continuously instead, and after the first car clears the line, a new finished car rolls off the end every single $t$: the fill time $p\\cdot t$ is paid once, at the very start, and amortises across however many cars follow it. A short production run pays that fill cost as a large fraction of its total time; a long one barely notices it. Micro-batching a training step is running more cars through the same line before shutting it down for the day.</p>`)}

<p>Underneath all four parallelisms sits one more question: even with the model split up, does the memory actually fit? <b>ZeRO / FSDP</b> answers it by attacking a specific redundancy in plain data parallelism — every GPU in that scheme holds a <i>full, independent copy</i> of the optimiser state, gradients and (in ZeRO-3/FSDP) even the parameters themselves, which is enormously wasteful when Adam's optimiser state alone needs roughly <b>12 bytes per parameter</b> of FP32 storage (a master FP32 weight copy plus two FP32 running moments, §4.11.2 explains why the FP32 copy exists), on top of the weights and gradients. ZeRO shards each of those tensors across the data-parallel group instead — every GPU permanently holds only its own $1/n$ slice — and gathers whatever full tensor is momentarily needed via targeted communication exactly when a computation requires it, then releases it again. That sharding, more than any other single trick, is what makes training a model whose optimiser state alone would not fit on one device possible at all.</p>
${H.table(['Stage', 'Shards optimiser state', 'Shards gradients', 'Shards parameters', 'Per-GPU memory vs. plain DP'], [
      ['None (plain DP)', 'no', 'no', 'no', '1× — every GPU replicates everything'],
      ['<b>ZeRO-1</b>', 'yes', 'no', 'no', 'the 12-byte Adam state shrinks by $1/n$ — usually the single biggest win'],
      ['<b>ZeRO-2</b>', 'yes', 'yes', 'no', 'gradients shrink by $1/n$ too, once each shard\'s reduction is done'],
      ['<b>ZeRO-3 / FSDP</b>', 'yes', 'yes', 'yes', 'parameters shrink by $1/n$ as well — gathered just-in-time per layer, released right after']
    ])}
<p>Each stage only ever adds sharding; it never removes the guarantee that the right full tensor is available at the instant a computation actually needs it — the extra cost of ZeRO-3 over ZeRO-2 is precisely the additional communication needed to gather a layer's full parameters right before using them and discard that full copy again right after, which is why ZeRO-3 saves the most memory and asks for the most communication in return, and why ZeRO-1 alone — sharding only the optimiser state, which is idle except at the update step — is often enough to fit a model that would not otherwise fit, at a communication cost barely different from plain data parallelism.</p>
${H.pitfall('It is tempting to reach straight for ZeRO-3 because it shards the most. Do not: every stage above ZeRO-1 trades memory for communication, and if a model already fits comfortably under ZeRO-1 or ZeRO-2, moving to ZeRO-3 buys nothing but slower steps. Pick the <i>lowest</i> stage that fits, not the highest one available — the memory-budget lab above is precisely the calculation for finding it.')}

<p><b>What you are looking at.</b> Every memory consumer in a training step — parameters, gradients, optimiser state, activations — stacked as a bar and computed live from your chosen model size, GPU count, ZeRO stage and checkpointing setting, with a dashed line marking your card's actual memory limit.</p>
<p><b>What to do with it.</b> Start at ZeRO stage "none": watch the optimiser-state segment dominate the bar, larger than the weights themselves, and note whether the total clears your card's limit. Step through ZeRO-1, -2 and -3 and watch each successive stage shrink a different segment — optimiser state, then gradients, then parameters — as more of each tensor gets sharded across ranks rather than replicated.</p>
<p><b>The thing genuinely worth noticing.</b> Toggle activation checkpointing on a model that was just barely fitting: the activation segment shrinks sharply, at the (invisible to this readout, but real) cost of roughly a third more forward-pass compute during the backward pass, because those discarded activations have to be recomputed rather than looked up. This is the trade §4.11.2 makes explicit: memory bought back by spending compute you otherwise would not have needed.</p>

${H.lab('mem', 'The training memory budget', 'Every consumer of memory, computed. Toggle ZeRO stages and activation checkpointing and watch the total fall below your card — this is the calculation that decides whether a run is possible.')}

<h2><span class="sn">4.11.2</span> The stability toolkit</h2>
<p>Fitting the model is necessary but not sufficient; a run also has to stay numerically stable for weeks at a time, and a small toolkit of standard techniques is what makes that reliable rather than lucky. <b>Mixed precision</b> runs the forward and backward arithmetic in BF16 for speed and memory, while keeping a separate <b>FP32 master copy</b> of every weight purely for the optimiser's update step: an individual weight update can be far smaller than BF16's precision can distinguish at that weight's magnitude, so applying updates directly in BF16 would silently lose most of them to rounding, while accumulating updates into an FP32 master copy — and only casting down to BF16 for the next forward pass — preserves them. BF16 is preferred over the older FP16 specifically because it keeps FP32's full 8-bit exponent range (only trading away mantissa precision), so it cannot overflow or underflow the way FP16's narrower exponent range can; FP16 training needed a separate "loss scaling" trick (multiply the loss by a large constant before the backward pass, then divide it back out) purely to keep small gradients from underflowing to zero, and BF16 makes that whole mechanism unnecessary.</p>
<p><b>Gradient accumulation</b> simulates a larger batch than memory allows for directly: run several forward-and-backward passes at a smaller batch size, summing their gradients without taking an optimiser step in between, and only apply one combined update once enough micro-batches have accumulated — mathematically equivalent to having run the full large batch at once, at the cost of doing the forward/backward work sequentially rather than in one shot. <b>Activation checkpointing</b> is the deliberate mirror image of §3.2's backpropagation cache: rather than storing every intermediate activation needed for the backward pass, as an ordinary forward pass would, checkpointing discards most of them and recomputes them on demand during the backward pass — trading roughly a third more compute for a large cut in peak memory, which the lab above lets you see directly. <b>Linear warmup followed by cosine or WSD decay</b> is §3.5's schedule machinery, applied here at a scale where getting it wrong is expensive rather than merely suboptimal; and <b>gradient clipping by global norm</b> rescales the entire gradient vector, all parameters together, whenever its overall norm exceeds a threshold — a single bad batch or a momentary numerical spike cannot then take an outsized step, because the direction of the update is preserved while its magnitude is capped.</p>

<h3>Two schedule facts worth knowing</h3>
<p>Cosine decay requires committing to a total step count in advance, because the whole shape of the curve is defined relative to where it ends — awkward the moment you might want to extend a run past its originally planned length. WSD (warmup–stable–decay) sidesteps that: hold the learning rate flat for most of training, then decay it sharply only in a short final phase, reaching essentially the same final loss cosine would while letting you checkpoint at the end of the stable phase and branch off several short, independent decay phases from that one trunk — useful when you want to compare final annealing on different data mixtures without rerunning the expensive stable phase each time. Separately, <b>μP</b> (maximal update parametrisation) rescales initialisation variances and learning rates as a function of a model's width according to a specific, derived formula, chosen so that the <i>optimal</i> hyperparameters found by sweeping a small, cheap proxy model transfer, largely unchanged, to a model orders of magnitude larger. That turns hyperparameter search on a run that might cost eight figures from something you cannot afford to do at all into a sweep performed entirely on a model cheap enough to iterate on quickly.</p>

<h3>Batch size is not free</h3>
<p>There is a regime, below the <b>critical batch size</b>, where a gradient estimated from a small batch is genuinely noisy — dominated by the sampling variance of which examples happened to land in it — and doubling the batch meaningfully reduces that noise, letting you take a correspondingly larger, more confident step and reach a target loss in roughly half as many steps. Above the critical batch size, the gradient estimate is already precise enough that adding more examples mostly re-estimates the same direction rather than sharpening it, so you are spending proportionally more compute for a shrinking return. The critical size itself grows as training proceeds and the loss falls, which is the practical reason large runs deliberately <i>ramp</i> batch size upward over the course of training rather than fixing it. And whenever you do increase the batch, the learning rate has to move with it — the standard rules of thumb are square-root scaling for Adam and linear scaling for plain SGD — or the larger batch's more confident gradient estimate is wasted on a step size still calibrated for a noisier, smaller one, which silently under-trains the model without producing any error message at all.</p>

<p>When a run's loss goes to NaN, the failure has an unglamorous but entirely mechanical cause somewhere in this chain: a gradient grows unboundedly large, the resulting weight update overflows the numeric range of whatever precision it is stored in, the next forward pass's activations inherit that overflow and become infinite or undefined, and the loss computed from them is no longer a number at all. Diagnosing which link in that chain broke is a short, ordered list — short enough that an interviewer asking for it is testing whether you have actually run one of these, not whether you can invent causes.</p>
${H.fig('THE LOSS WENT TO NaN — WORK THE LIST IN THIS ORDER', H.steps([
      '<b>Lower the learning rate</b>, and verify warmup is actually running — a restart that skips warmup is the classic cause.',
      '<b>Check the data:</b> NaNs or infs in a shard, unnormalised inputs, a corrupt record. Print the batch that preceded the blow-up.',
      '<b>Add or tighten gradient clipping</b>, and watch the global-norm trace for the spike <i>before</i> the NaN.',
      '<b>Check mixed-precision handling</b> — loss scaling under FP16, or an fp32 softmax/norm that got cast down.',
      '<b>Confirm the LR the scheduler is actually emitting</b> after a resume from checkpoint.',
      '<b>Reduce batch size</b>, or isolate a bad shard / a single misbehaving rank.'
    ]), 'Interviewers ask this to hear an ordered list. Anyone can name the causes; the signal is knowing which to check first and why.')}

<p><b>What you are looking at.</b> A simulated 500-step run's loss (blue) and gradient global norm (red, doubled in scale to sit on the same axes) plotted together, with toggles for a corrupt data shard, gradient clipping, and whether warmup is actually running — the same knobs the checklist above walks through.</p>
<p><b>What to do with it.</b> Turn on the corrupt shard with clipping off and watch the two vertical markers: the amber one, where the gradient norm first spikes, and the red one, where the loss actually goes vertical a step or two later. That gap is the whole point of the exercise.</p>
<p><b>The thing genuinely worth noticing.</b> The gradient norm always moves first. If you are only watching the loss curve, you find out about the problem after it has already destroyed the run and have no batch left to inspect; if you are watching the gradient norm, you get a short but real early warning and can print the exact batch responsible. Now turn clipping back on and watch the identical corrupt shard pass through harmlessly — the spike still happens, but it can no longer propagate into a step large enough to overflow anything.</p>

${H.lab('nan', 'The spike before the NaN', 'A simulated run where the gradient-norm trace spikes one or two steps before the loss goes vertical. Watch which trace moves first — that is the step whose batch you want to print, and the reason experienced people watch the norm rather than the loss.')}

${H.probe([
      ['Which parallelism goes inside a node, and is it because of how much data it moves?', 'Tensor parallelism — but not because of total bytes; a worked example moved less total data than data parallelism\'s single all-reduce. It needs NVLink because it makes hundreds of small, blocking synchronisation calls per step, and wall-clock cost there is dominated by round-trip latency, not payload size — which is why low latency matters more than raw bandwidth once there are hundreds of synchronisation points.'],
      ['What does the pipeline bubble fraction actually equal?', '$(p-1)/(p+m-1)$ exactly, for $p$ stages and $m$ micro-batches — collapsing to the memorable $(p-1)/m$ only once $m$ is several times larger than $p$; used outside that regime, the approximation can overstate the bubble by a factor of several.'],
      ['How much memory does Adam need, and what does each ZeRO stage shard?', '≈12 bytes per parameter of FP32 optimiser state — a master weight copy plus two running moments. ZeRO-1 shards that optimiser state, ZeRO-2 adds gradients, ZeRO-3/FSDP adds parameters too, gathered just-in-time per layer — each stage trading more communication for more memory saved, so pick the lowest stage that fits rather than the highest available.'],
      ['What do you check first when the loss NaNs?', 'The learning rate and whether warmup is actually running, then the data, then clipping — and watch the gradient-norm trace rather than the loss, because the norm reliably spikes one or two steps before the loss goes vertical, giving you an actual batch to inspect.']
    ])}`,
    labs: {
      mem: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'parameters (B)', min: .5, max: 200, step: .5, value: 8, fmt: v => v.toFixed(1) + 'B' },
          { k: 'gpus', label: 'data-parallel ranks', min: 1, max: 512, step: 1, value: 8, fmt: v => v },
          { k: 'zero', label: 'ZeRO stage', type: 'buttons', value: '2', options: [{ v: '0', t: 'none' }, { v: '1', t: 'ZeRO-1' }, { v: '2', t: 'ZeRO-2' }, { v: '3', t: 'ZeRO-3 / FSDP' }] },
          { k: 'ckpt', label: 'activation checkpointing', type: 'toggle', value: true },
          { k: 'batch', label: 'per-GPU batch × seq (k tokens)', min: 1, max: 64, step: 1, value: 8, fmt: v => v + 'k' },
          { k: 'card', label: 'GPU memory (GB)', min: 16, max: 192, step: 8, value: 80, fmt: v => v + ' GB' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'total', label: 'per-GPU memory', cls: 'key' }, { k: 'fits', label: 'fits?' },
          { k: 'opt', label: 'optimiser state' }, { k: 'act', label: 'activations' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const N = st.N * 1e9, z = +st.zero, ranks = st.gpus;
            const weights = N * 2 / 1e9;                     // BF16
            const grads = (z >= 2 ? N * 2 / ranks : N * 2) / 1e9;
            const optim = (z >= 1 ? N * 12 / ranks : N * 12) / 1e9;
            const params = (z >= 3 ? weights / ranks : weights);
            const acts = (st.batch * 1000 * Math.sqrt(N / 1e9) * (st.ckpt ? 0.9 : 9)) * 2 / 1e6;
            const parts = [
              ['parameters (BF16)', params, T.blue],
              ['gradients', grads, T.amber],
              ['optimiser state (Adam FP32)', optim, T.red],
              ['activations', acts, T.green]
            ];
            const total = parts.reduce((a, p) => a + p[1], 0);
            const bx = 20, bw = w - 40, by = 40;
            let x = bx;
            parts.forEach(p => {
              const pw = bw * p[1] / Math.max(total, st.card);
              ctx.fillStyle = p[2]; ctx.fillRect(x, by, pw, 36);
              x += pw;
            });
            // card limit
            const limitX = bx + bw * st.card / Math.max(total, st.card);
            ctx.strokeStyle = T.text; ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(limitX, by - 10); ctx.lineTo(limitX, by + 46); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = T.text; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText(st.card + ' GB card', limitX + 5, by - 12);
            ctx.font = '12px ui-sans-serif'; ctx.textBaseline = 'middle';
            parts.forEach((p, i) => {
              const yy = by + 70 + i * 24;
              ctx.fillStyle = p[2]; ctx.fillRect(bx, yy - 6, 12, 12);
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.fillText(p[0], bx + 20, yy);
              ctx.fillStyle = T.muted; ctx.textAlign = 'right'; ctx.fillText(p[1].toFixed(1) + ' GB', bx + bw, yy);
            });
            ctx.fillStyle = total <= st.card ? T.green : T.red; ctx.font = 'bold 14px ui-monospace, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('total ' + total.toFixed(1) + ' GB per GPU — ' + (total <= st.card ? 'fits' : 'does NOT fit'), bx, by + 70 + 4 * 24 + 8);
            out({
              total: total.toFixed(1) + ' GB', fits: total <= st.card ? 'yes' : 'no',
              opt: optim.toFixed(1) + ' GB', act: acts.toFixed(1) + ' GB'
            });
          }
        });
        Viz.note(host, 'Optimiser state is the biggest single consumer at 12 bytes per parameter — bigger than the weights themselves. That is what ZeRO shards, and why an 8B model that "should" fit in 16 GB of BF16 weights needs roughly 145 GB in total to fine-tune fully (§4.13 turns that number into the case for LoRA).');
      },

      nan: function (host) {
        const st = Viz.controls(host, [
          { k: 'lr', label: 'learning rate', min: -4.5, max: -2.4, step: .05, value: -3.2, fmt: v => Math.pow(10, v).toExponential(1) },
          { k: 'clip', label: 'gradient clipping', type: 'toggle', value: false },
          { k: 'warm', label: 'warmup running', type: 'toggle', value: true },
          { k: 'badshard', label: 'a corrupt shard at step 320', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'die', label: 'run status', cls: 'key' }, { k: 'spike', label: 'gradient-norm spike at' },
          { k: 'nan', label: 'loss goes vertical at' }, { k: 'lead', label: 'warning lead time' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(23);
            const lr = Math.pow(10, st.lr);
            const loss = [], norm = [];
            let l = 11, dead = -1, spike = -1;
            for (let t = 0; t < 500; t++) {
              const warm = st.warm ? Math.min(1, t / 60) : 1;
              let g = 0.9 + R.normal(0, .12) + (lr * warm > 1.2e-3 ? (t / 500) * 2.2 : 0);
              if (st.badshard && t >= 320 && t < 324) g *= 14;
              if (st.clip) g = Math.min(g, 1.0);
              norm.push(g);
              if (dead < 0) {
                l = Math.max(1.6, l - lr * warm * 900 * (1 / (1 + t * .02))) + R.normal(0, .02) + (g > 3 ? g * .35 : 0);
                if (g > 3 && spike < 0) spike = t;
                if (g > 6 || (lr * warm > 2.2e-3 && t > 200 && R() < .02)) { dead = t + 2; }
              }
              loss.push(dead >= 0 && t >= dead ? NaN : l);
            }
            const P = Viz.plot(ctx, w, h, { xd: [0, 500], yd: [0, 12] })
              .frame({ xlabel: 'step', ylabel: 'loss (blue) · gradient global norm (red, ×2)' });
            P.clip(() => {
              P.line(loss.map((v, i) => [i, isFinite(v) ? v : 12]).filter((p, i) => isFinite(loss[i])), { color: T.blue, width: 2.4 });
              P.line(norm.map((v, i) => [i, Math.min(12, v * 2)]), { color: T.red, width: 1.4, alpha: .85 });
              if (spike >= 0) P.vline(spike, { color: T.amber, label: 'norm spikes here' });
              if (dead >= 0) P.vline(dead, { color: T.red, label: 'NaN' });
            });
            out({
              die: dead >= 0 ? 'died at step ' + dead : 'survived 500 steps',
              spike: spike >= 0 ? 'step ' + spike : '—',
              nan: dead >= 0 ? 'step ' + dead : '—',
              lead: (dead >= 0 && spike >= 0) ? (dead - spike) + ' steps' : '—'
            });
          }
        });
        Viz.note(host, 'Turn on the corrupt shard with clipping off: the red norm trace spikes at step 320 and the blue loss only goes vertical a couple of steps later. That gap is your diagnostic — print the batch at the spike, not at the NaN. Turn clipping on and the same shard passes through harmlessly.');
      }
    },
    quiz: [
      {
        q: 'Adam’s optimiser state costs roughly how much memory per parameter?',
        options: ['2 bytes', '4 bytes', '12 bytes', '32 bytes'],
        answer: 2,
        why: 'Adam needs an FP32 master copy of every weight plus two FP32 running moments to keep its updates numerically meaningful, and 4 bytes for each of those three tensors is exactly the 12 bytes per parameter §4.11.1 quotes — more memory than the BF16 weights themselves, which is precisely why ZeRO/FSDP shards it across every GPU in the data-parallel group rather than replicating it everywhere. "4 bytes" is the tempting wrong answer for anyone who correctly recalls that a single FP32 tensor costs 4 bytes per element but forgets that Adam needs three separate FP32 tensors, not one. "2 bytes" is simply the BF16 weight cost mistaken for the optimiser state, and "32 bytes" overshoots by double-counting. The general principle, central to §4.11.1\'s whole ZeRO discussion, is that the optimiser state, not the weights, is usually the single largest memory consumer in a training step, which is exactly the redundancy ZeRO/FSDP was built to attack.'
      },
      {
        q: 'Your loss NaNs after resuming from a checkpoint. The first thing to check is…',
        options: ['the model architecture', 'whether the scheduler is emitting the intended learning rate — a resume that skips warmup is the classic cause', 'the tokenizer', 'the evaluation set'],
        answer: 1,
        why: 'A checkpoint resume that silently skips or restarts warmup hands the model a full, unwarmed-up learning rate on data it has already adapted to at a much smaller step size, and that mismatch is common enough and cheap enough to check — print the scheduler\'s actual emitted LR against what it should be at this step — that it sits first on §4.11.2\'s ordered NaN checklist rather than merely somewhere on it. "The model architecture" is the tempting wrong answer because an architecture bug feels like the natural place to suspect a catastrophic failure, but architecture bugs do not typically wait until exactly the moment of a checkpoint resume to manifest — the timing itself is the diagnostic clue that points at the scheduler instead. The tokenizer and evaluation set are even further from the timing evidence, with no obvious mechanism connecting them to a resume-triggered blow-up. The general principle behind the whole ordered list is to check what is both common and instantly verifiable first, which is exactly why an interviewer asking for this list is testing whether you have actually run one of these training jobs, not whether you can brainstorm plausible causes.'
      },
      {
        q: 'Tensor parallelism should be kept…',
        options: ['across data centres', 'inside a node, on NVLink', 'on the same GPU', 'between pipeline stages'],
        answer: 1,
        why: 'Slicing a single weight matrix across GPUs forces a communication step inside nearly every layer, not once per training step the way data parallelism\'s all-reduce does, so tensor parallelism needs NVLink\'s very high bandwidth and very low latency between GPUs on the same physical machine — cross that boundary to ordinary inter-node networking and the constant chatter collapses throughput, exactly §4.11.1\'s rule of thumb. "Between pipeline stages" is the tempting wrong answer because it confuses two different parallelisms that both split the model rather than the batch: pipeline parallelism puts different layers on different devices and only passes activations at the boundary, which is cheap precisely because it avoids the frequent, heavy communication tensor parallelism requires. "Across data centres" takes the opposite, clearly unworkable extreme, and "on the same GPU" misses the entire point of tensor parallelism, which exists specifically to split a matrix that does not fit on one device. The general principle from §4.11.1 is that each of the four parallelisms has its own communication pattern and bandwidth requirement, and matching a parallelism to the interconnect that can sustain its chatter is what 3-D/4-D parallelism is doing when it decides which axis goes where.'
      }
    ],
    cards: [
      { q: 'Three parallelisms', a: 'Data (all-reduce), tensor (chatty — keep in-node), pipeline (cheap, bubbles). Plus expert parallelism for MoE.' },
      { q: 'Ring all-reduce volume', a: '$2(n-1)S/n \\to 2S$ per GPU as $n$ grows — nearly flat, which is why data parallelism scales cheaply.' },
      { q: 'Why tensor parallelism needs NVLink', a: 'Not total bytes — hundreds of blocking synchronisation calls per step, so round-trip latency dominates, not payload size.' },
      { q: 'Pipeline bubble fraction', a: '$(p-1)/(p+m-1)$ exactly, for $p$ stages and $m$ micro-batches; $(p-1)/m$ only once $m\\gg p$.' },
      { q: 'Adam memory & ZeRO', a: '≈12 bytes/param FP32 state; ZeRO-1 shards it, ZeRO-2 adds gradients, ZeRO-3/FSDP adds parameters too — pick the lowest stage that fits.' },
      { q: 'NaN checklist order', a: 'LR/warmup → data → clipping (watch the norm spike) → mixed precision → scheduler after resume → batch size / bad shard.' },
      { q: 'Critical batch size', a: 'Below it doubling batch nearly halves steps; above it you buy little. It grows as loss falls — hence batch ramping.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.12 */
  ML.section({
    id: 'post-training', track: 'llm', num: '4.12',
    title: 'Post-training: SFT → RLHF → DPO → GRPO → RLVR',
    lede: 'Rests on §1.10 (KL) and §1.5 (likelihood). The DPO derivation is the highest-value derivation in Part 4.',
    html: `
<p>Ask a raw pretrained model, fresh out of §4.9, to "write a haiku about autumn", and it may just as easily continue with "write a haiku about spring. write a limerick about winter." — a perfectly reasonable continuation of internet text that happens to look like a list of exercises, and a completely useless response to what was actually a request. §4.9.1 explained why next-token prediction on raw text produces something that behaves as if it understands grammar and facts; it never promised a model that behaves as if it wants to <i>help you</i>. Post-training is the sequence of stages that closes that gap, and it is worth tracking, across every stage below, exactly what kind of supervision each one is exploiting — because "SFT, then RLHF, then DPO, then GRPO" is not four attempts at the same thing, it is four different answers to the question "what form does the training signal take, and how expensive is it to collect".</p>

<h2><span class="sn">4.12.1</span> SFT</h2>
<p><b>Supervised fine-tuning</b> takes curated (prompt, ideal response) pairs written or approved by people, and trains on them with the identical next-token cross-entropy objective §4.9.1 already established — nothing new in the loss function, only a change in what the training text looks like: coherent instruction-and-answer pairs instead of arbitrary web text. That is enough, on its own, to move the model from "continues text" to "answers a question and then stops", because the model is now imitating a corpus that consistently behaves that way. It teaches format and instruction-following, and it teaches <i>only</i> what was explicitly demonstrated — the model can, at best, become as good as its demonstrations, and demonstrations are expensive to produce well at scale, which is precisely the limitation the rest of this section exists to work around.</p>

<h2><span class="sn">4.12.2</span> RLHF, classically</h2>
<p>Writing the single best possible response to a prompt is hard and slow for a human annotator. Looking at two already-written responses and saying which one is better is comparatively fast and reliable — ranking is a much easier judgment than generation, the same asymmetry that makes multiple-choice questions easier to answer correctly than essay questions. <b>Reinforcement learning from human feedback</b> is built around exploiting exactly that asymmetry: collect pairs of responses to the same prompt with a human preference between them, use that preference data to train a separate <b>reward model</b> — a network that reads a (prompt, response) pair and outputs a single scalar score, trained so that its scores are consistent with the observed preferences (the Bradley–Terry likelihood the DPO derivation below makes precise) — and then use that learned reward model as a stand-in for "what humans want" to optimise the actual policy against, using PPO (§4.12.6).</p>
<p>The reward model is only a proxy, fit from a finite, noisy sample of human judgments, and optimising hard against any proxy eventually finds places where the proxy disagrees with what it was standing in for — text a human would rate poorly but that happens to score highly under the specific, imperfect function the reward model learned. Left unconstrained, the policy drifts toward exactly those blind spots, a failure with its own name, <b>reward hacking</b>, that §4.12.7 catalogues with concrete examples. The naive fix — just maximise expected reward, $\\max_\\pi \\mathbb E_{y\\sim\\pi}[r(x,y)]$, full stop — makes this worse, not better: an unconstrained maximiser has every incentive to walk straight into the reward model's blind spots, since nothing in that objective distinguishes "genuinely better" from "scores better under this one imperfect proxy". What is actually wanted is a constrained version of the same problem: get as much reward as you can, <i>without</i> leaving the neighbourhood of a reference policy that is, at least, known to produce recognisable, well-formed text.</p>
${H.deriv('why the RL objective takes a KL-regularised form at all', [
      ['$\\max_\\pi\\ \\mathbb E_{y\\sim\\pi(\\cdot|x)}[r(x,y)] \\quad\\text{s.t.}\\quad D_{KL}\\big(\\pi(\\cdot|x)\\,\\|\\,\\pi_{\\text{ref}}(\\cdot|x)\\big) \\le \\epsilon$', 'State the actual goal as a constrained optimisation: maximise reward, but keep the policy within a trust region of radius $\\epsilon$ around the reference — the same shape of problem §1.12.3 and §4.10.2 already solved for a compute budget instead of a KL budget.'],
      ['$\\mathcal L(\\pi,\\beta) = \\mathbb E_\\pi[r(x,y)] - \\beta\\big(D_{KL}(\\pi\\|\\pi_{\\text{ref}}) - \\epsilon\\big)$', 'Fold the inequality constraint into a Lagrangian with multiplier $\\beta\\ge0$ — non-negative specifically because this is a one-sided fence (§1.9.5\'s inequality case), not the equality case §4.10.2 used: the constraint only binds when the policy is trying to push past the KL budget, never when it has room to spare.'],
      ['$\\mathcal L(\\pi,\\beta) = \\underbrace{\\mathbb E_\\pi[r(x,y)] - \\beta D_{KL}(\\pi\\|\\pi_{\\text{ref}})}_{\\text{depends on }\\pi} + \\underbrace{\\beta\\epsilon}_{\\text{constant in }\\pi}$', 'Split the Lagrangian into the piece that depends on the policy and the piece that does not — $\\beta\\epsilon$ is fixed once $\\beta$ and $\\epsilon$ are chosen, so it never affects which $\\pi$ maximises the expression.'],
      ['$\\arg\\max_\\pi \\mathcal L(\\pi,\\beta) = \\arg\\max_\\pi\\Big(\\mathbb E_\\pi[r(x,y)] - \\beta D_{KL}(\\pi\\|\\pi_{\\text{ref}})\\Big)$', 'Drop the constant term — it cannot change which policy is optimal — leaving exactly the KL-regularised objective §4.12.2 opened with and §4.12.3\'s derivation starts from. Nothing about that objective\'s shape was assumed; it fell out of Lagrangian-relaxing the constrained problem stated in line 1.']
    ], 'The KL penalty was never an ad hoc regulariser bolted onto a reward-maximisation objective to make training better-behaved — it is the exact Lagrangian relaxation of "maximise reward inside a trust region around the reference policy", with $\\beta$ playing precisely the role a Lagrange multiplier always plays: a large $\\beta$ acts like a small, tightly enforced $\\epsilon$ (stay very close to the reference), a small $\\beta$ acts like a large, loosely enforced one (chase reward more freely) — the identical multiplier-as-dial reading §4.10.2 gave the compute-budget Lagrangian, now applied to a divergence budget instead of a FLOP budget.')}
<p>The fix, in the form actually used, is therefore a <b>KL penalty</b>: subtract $\\beta\\, D_{KL}(\\pi_\\theta\\|\\pi_{\\text{ref}})$ from the raw reward objective, using §1.10's divergence between the policy being trained and the original SFT model it started from, so any move the policy makes toward a higher reward is taxed by how far that move has carried it from a distribution that was, at least, producing recognisably normal text. It is not decoration bolted onto the objective; the derivation above shows it is the mechanism a trust-region constraint becomes once you relax it into something gradient descent can actually optimise. The whole pipeline works, and is used in production, but it is heavy: a reward model, a separate value network estimating expected future reward for PPO's advantage calculation, the policy being trained, and the frozen reference policy for the KL term — four networks resident in memory simultaneously for what is, underneath, training one model.</p>

<h2><span class="sn">4.12.3</span> DPO, derived</h2>
<p>DPO's insight is that the heavy RLHF pipeline above is solving an optimisation problem that has a closed-form answer, and once you know that answer, you can work backwards to a training objective that needs none of the pipeline's machinery. Start from exactly the objective RLHF is trying to maximise: expected reward, penalised by KL divergence from the reference policy, with $\\beta$ controlling the trade-off — precisely §4.12.2's objective, written in symbols.</p>
${H.deriv('from the RLHF objective to a plain classification loss', [
      ['$J(\\pi) = \\mathbb{E}_{y\\sim\\pi}[r(x,y)] - \\beta\\, D_{KL}(\\pi(\\cdot|x)\\|\\pi_{\\text{ref}}(\\cdot|x))$', 'The KL-regularised RLHF objective from §4.12.2, to be maximised over the policy $\\pi$ for each prompt $x$.'],
      ['$= \\mathbb{E}_\\pi\\!\\left[r(x,y) - \\beta\\log\\dfrac{\\pi(y|x)}{\\pi_{\\text{ref}}(y|x)}\\right]$', 'Expand the KL divergence using its definition from §1.10, $D_{KL}(\\pi\\|\\pi_{\\text{ref}}) = \\mathbb{E}_\\pi[\\log(\\pi/\\pi_{\\text{ref}})]$.'],
      ['define $\\pi^*(y|x) := \\dfrac{1}{Z(x)}\\pi_{\\text{ref}}(y|x)\\exp\\!\\left(\\tfrac{1}{\\beta}r(x,y)\\right)$, with $Z(x)=\\sum_y \\pi_{\\text{ref}}(y|x)\\exp(r(x,y)/\\beta)$', 'A deliberate definition, not yet justified — $\\pi^*$ is a valid probability distribution by construction, since it is non-negative and $Z(x)$ was chosen exactly to make it sum to 1 over $y$. The next two lines show why this particular definition was worth making.'],
      ['$J(\\pi) = -\\beta\\, D_{KL}(\\pi(\\cdot|x)\\,\\|\\,\\pi^*(\\cdot|x)) + \\beta\\log Z(x)$', 'Substitute the definition of $\\pi^*$ into line 2 and simplify: $r(x,y)-\\beta\\log(\\pi/\\pi_{\\text{ref}}) = \\beta\\log(\\pi_{\\text{ref}}e^{r/\\beta}/\\pi) = \\beta\\log(Z(x)\\pi^*/\\pi) = \\beta\\log Z(x) - \\beta\\log(\\pi/\\pi^*)$, and taking the expectation under $\\pi$ turns the last term into $-\\beta D_{KL}(\\pi\\|\\pi^*)$.'],
      ['$J(\\pi)$ is maximised exactly when $\\pi = \\pi^*$', 'KL divergence is non-negative with equality only when the two distributions coincide exactly (§1.10\'s Gibbs inequality), and $\\log Z(x)$ does not depend on $\\pi$ at all — so $J(\\pi)$ is maximised by making $D_{KL}(\\pi\\|\\pi^*)$ as small as possible, which is zero exactly when $\\pi=\\pi^*$.']
    ], 'The optimal policy under KL-regularised reward maximisation has a closed form — you never need to run an RL algorithm to find it, provided you already know the reward function $r$. The catch, of course, is that you do not: $r$ is exactly what the reward model in §4.12.2 exists to estimate. DPO\'s next move is to invert this relationship and estimate $r$ implicitly instead.')}
<p>Solve the boxed relationship above for $r(x,y)$ directly — take logs of both sides of $\\pi^*(y|x)=\\pi_{\\text{ref}}(y|x)\\exp(r(x,y)/\\beta)/Z(x)$ and rearrange — and you get $r(x,y) = \\beta\\log\\frac{\\pi(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)} + \\beta\\log Z(x)$: the reward, expressed entirely in terms of the policy's own log-probability ratio against the reference, plus a term $Z(x)$ that depends only on the prompt, never on which response $y$ is being scored. Substitute that expression into the <b>Bradley–Terry</b> preference model from §1.5's likelihood framework, $P(y_w \\succ y_l\\mid x) = \\sigma(r(x,y_w) - r(x,y_l))$ — the standard model for "the probability that $y_w$ is preferred over $y_l$ is a logistic function of their reward gap" — and because the reward only ever appears as a <i>difference</i> between two completions of the <i>same</i> prompt $x$, the $\\beta\\log Z(x)$ term is identical in both and <mark>cancels exactly</mark>. What survives is a loss with no reward model, no partition function, and no sampling loop anywhere in it — purely a maximum-likelihood classification loss (§1.5 again) on which of two completions was preferred, computed straight from the policy's own probabilities:</p>
$$\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log\\sigma\\!\\left(\\beta\\log\\frac{\\pi(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta\\log\\frac{\\pi(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$$
<p>Read this as ordinary logistic regression (§2.4) on a specific, cleverly-constructed feature: the gap between how much more likely the policy has made the preferred response, relative to the reference, versus how much more likely it has made the dispreferred one. Push that gap up and the loss falls, exactly as a classifier's loss falls when it becomes more confident in the correct label.</p>

<p><b>What you are looking at.</b> The DPO loss plotted against the implicit reward margin — the quantity inside the $\\sigma(\\cdot)$ above — with $\\beta$ as a slider and four grey reference curves at fixed $\\beta$ values so you can see how the whole curve's shape changes as $\\beta$ moves.</p>
<p><b>What to do with it.</b> Drag $\\beta$ from small to large and watch the curve steepen: at small $\\beta$ the loss barely responds to the margin at all, while at large $\\beta$ a small margin already drives the loss close to zero.</p>
<p><b>The thing genuinely worth noticing.</b> That steepening is exactly the KL penalty from §4.12.2, arriving through an entirely different door — $\\beta$ never appears explicitly as a divergence anywhere in the DPO loss, yet it is doing precisely the same job of trading "fit the preference data hard" against "do not move far from the reference policy" that the KL term did in PPO. The derivation guarantees they are the same knob; the lab lets you watch it behave like one.</p>

${H.lab('dpo', 'The DPO loss, and what β controls', 'The loss surface over the implicit reward margin, with β as the slider. Watch how β trades preference-fitting against staying near the reference policy — the same knob the KL penalty is in PPO, arriving by a different route.')}

<h3>DPO's known failure</h3>
<p>Length bias is real and documented, not a theoretical worry: Park et al. (<i>Disentangling Length from Quality in Direct Preference Optimisation</i>, arXiv:2403.19159) show DPO exhibits significant length exploitation — because human preference data used to train on frequently correlates "longer" with "more thorough, and therefore better", DPO happily learns to satisfy the correlation rather than the underlying quality it was meant to proxy for, and their length-regularised R-DPO improves length-corrected win rates by roughly 15–20 points once that shortcut is closed off. If you run DPO, <b>monitor mean response length as a first-class metric</b> throughout training, not as an afterthought discovered once users start complaining that answers have grown bloated.</p>

<h2><span class="sn">4.12.4</span> GRPO</h2>
<p>§4.12.2's RLHF pipeline needed a value network specifically to estimate a <b>baseline</b> — an expected-reward estimate to subtract from the actual reward, so that the policy update reflects "was this response better than typically expected for this prompt" rather than the raw reward itself, which reduces the variance of the training signal (the same reduce-variance-with-a-baseline idea that shows up throughout reinforcement learning). Training that value network well is itself a hard, expensive sub-problem. <b>GRPO</b> (DeepSeek) removes it by replacing a learned baseline with a computed one: sample a whole group of $G$ responses to the same prompt, score every one of them, and use the group's own statistics — the group-relative advantage $A_i = (r_i - \\mathrm{mean}(r))/\\mathrm{std}(r)$ — directly as the training signal, no value network anywhere. <b>The group is the baseline.</b> This is both cheaper (one fewer large network to train and hold in memory) and a natural fit for the setting §4.12.5 introduces next, where rewards are cheap, automatic checks rather than an expensive learned model.</p>

${H.worked('worked GRPO advantage', `
<p>Sample $G=6$ answers to one maths prompt and check each against the known answer. Rewards: [1, 0, 1, 1, 0, 0]. Mean = 0.5; population standard deviation = 0.5.</p>
<p>Advantages $(r_i-\\bar r)/\\sigma$ = <b>[+1, −1, +1, +1, −1, −1]</b>: the three correct completions are reinforced, the three wrong ones suppressed, with no value network anywhere.</p>
<p>Now the failure case an interviewer will push on: if all six are correct, rewards are [1,1,1,1,1,1], $\\sigma = 0$, and every advantage is zero (or NaN) — <b>the prompt teaches nothing</b>. Same if all six fail. So GRPO's learning signal lives entirely on prompts of <i>intermediate</i> difficulty, which is why curriculum and difficulty filtering matter far more here than in ordinary RLHF, and why implementations drop degenerate groups rather than dividing by zero.</p>`)}

<p><b>What you are looking at.</b> Six sliders, one reward per sampled answer in a group, and a bar for each showing its group-relative advantage — positive bars extending right (reinforce this completion), negative bars extending left (suppress it), computed from the exact formula above rather than illustrated schematically.</p>
<p><b>What to do with it.</b> Start from the worked example's [1,0,1,1,0,0] and confirm the bars read exactly +1, −1, +1, +1, −1, −1. Then set every reward to 1.</p>
<p><b>The thing genuinely worth noticing.</b> Every bar collapses to zero the instant the group becomes unanimous, and the status readout switches to warning that the prompt teaches nothing — not because the completions were bad, but because a group with no variance carries no information about which of its members to prefer. This is the practical argument for curating training prompts toward intermediate difficulty: a prompt too easy or too hard for the current policy contributes nothing to a GRPO update, however many times you sample it.</p>

${H.lab('grpo', 'GRPO advantages, and the degenerate group', 'Set the rewards yourself. Watch the advantages, and watch what happens when the group is unanimous — the readout tells you exactly how much signal the prompt carries.')}

<h2><span class="sn">4.12.5</span> RLVR</h2>
<p>Every stage so far has depended, in one way or another, on a reward that is ultimately a human opinion or a model trained to imitate one — expensive to collect, and, as §4.12.2 already flagged, hackable because it is a proxy. Some domains do not need a proxy at all. <b>Reinforcement learning from verifiable rewards</b> (RLVR) replaces the learned reward model with a rule-based checker wherever one exists: did the computed answer match the known solution, did the generated unit tests pass, did the program actually compile and run. There is nothing to hack, because the reward is not an approximation of ground truth standing in for it — it <i>is</i> ground truth, mechanically checked. DeepSeek-R1 (arXiv:2501.12948) is the paper that made this approach's power impossible to ignore: a cold-start supervised phase followed by GRPO trained purely on verifiable rewards reached reasoning performance competitive with OpenAI's o1, and — the detail worth remembering specifically — long, structured chains of thought with backtracking and self-correction <i>emerged</i> from the reinforcement learning process itself, never having been demonstrated by a human annotator anywhere in the pipeline. <mark>This — not RLHF — is how current frontier reasoning models are trained.</mark> RLAIF and constitutional methods sit alongside RLVR rather than replacing it, tackling the domains RLVR cannot reach: they substitute a written constitution and model-generated preference judgments for human labels, useful precisely where "correct" is not something a program can check.</p>
${H.flag(`The SFT → RLHF → DPO → GRPO → RLVR story above is presented as a lineage, and the broad direction of travel — toward cheaper, harder-to-hack reward signals, with RLVR now the dominant recipe for frontier reasoning models — is genuinely settled. Several of the specific mechanics inside it are not, and are worth knowing the difference before repeating either as received wisdom. <b>Whether the SFT cold start is actually necessary</b> is an open question DeepSeek's own release line raised directly: R1 used a cold-start supervised phase before GRPO, but R1-Zero, trained with RLVR directly from the base model with no SFT stage at all, still produced long, self-correcting reasoning chains — evidence that the cold start buys readability and stability rather than being a strict requirement for the capability to emerge, though the field has not settled exactly how much it buys or when it is worth its cost. <b>GRPO's own normalisation is under active, published correction, not just informal grumbling</b>: Liu et al.'s "Dr. GRPO" (2025) identifies two concrete biases baked into the formula this section presents as settled — dividing by the group's response length systematically under-penalises long wrong answers relative to short ones, and dividing by the group's reward standard deviation over-weights prompts that happen to land near-unanimous (the same knife-edge the worked example above shows going all the way to zero, but the bias exists in milder form well before that extreme too) — and proposes removing both terms; follow-up work has since questioned whether removing them fully eliminates bias either, rather than trading one distortion for another. And <b>whether DPO strictly dominates PPO-style RLHF</b>, despite being dramatically simpler to implement, is genuinely contested in the literature rather than resolved in DPO's favour — some published comparisons still find PPO ahead on specific benchmarks once both are well-tuned. Quote the settled direction with confidence; quote the specific formula in any one of these boxes as the final word on it, and you are one paper away from being out of date.`)}

<h2><span class="sn">4.12.6</span> The PPO objective, in the form people ask for</h2>
<p>Every stage above except DPO ultimately needs an actual reinforcement-learning update rule to turn a reward signal into a policy change, and <b>PPO</b> (proximal policy optimisation) is the one nearly universally used. Define the probability ratio $r_t = \\pi_\\theta(a_t|s_t)/\\pi_{\\text{old}}(a_t|s_t)$ — how much more (or less) likely the token just generated has become under the policy being updated, compared with the policy that generated it — and let $A_t$ be the advantage, exactly §4.12.4's baseline-subtracted reward signal. PPO maximises</p>
$$\\mathbb{E}\\big[\\min(r_tA_t,\\; \\mathrm{clip}(r_t, 1-\\epsilon, 1+\\epsilon)A_t)\\big] - \\beta D_{KL}(\\pi_\\theta\\|\\pi_{\\text{ref}})$$
<p>Read the clip term the way you would read a speed limit rather than an incentive: once the policy ratio has moved more than $\\epsilon$ (typically 0.2) away from 1 in the <i>favourable</i> direction for a positive-advantage action, the $\\min$ selects the clipped, flat branch, and its gradient with respect to the policy is exactly zero — no matter how good the advantage estimate says this action was, the objective refuses to reward moving further in that direction within a single update. That is the whole trick, stated precisely: a single batch, however enthusiastic its gradient signal, cannot take an enormous step away from the policy that generated the data, because the objective itself goes flat once the step is large enough. The KL term does a genuinely different job on a different timescale — <b>clipping bounds each individual update, the KL penalty bounds the total drift accumulated over the entire training run</b> from the original reference policy — and both are needed, because bounding each step does not by itself prevent a long sequence of small, all-favourable steps from eventually wandering a very long way from where training started.</p>

<p><b>What you are looking at.</b> The clipped PPO objective plotted against the probability ratio $r$, for whatever fixed advantage $A$ and clip width $\\epsilon$ you choose, with the unclipped $rA$ line shown faintly behind it and the trust-region band — $[1-\\epsilon, 1+\\epsilon]$ — shaded.</p>
<p><b>What to do with it.</b> With $A>0$, trace the curve from left to right: it rises with the unclipped line up to $r=1+\\epsilon$, then goes perfectly flat. Flip $A$ negative and watch which side flattens instead.</p>
<p><b>The thing genuinely worth noticing.</b> The flat region always appears on the side that would <i>increase</i> the objective beyond what a cautious update should allow — for positive advantage that is the "make this action more likely" side, for negative advantage it is the "make this action less likely" side. PPO always lets you move further <i>away</i> from a bad action than the clip would otherwise allow, but caps how aggressively one batch can push you toward a good one. That asymmetry is deliberate: it is safer to over-punish a bad action than to over-reward a good one on the strength of a single, possibly noisy, batch.</p>

${H.lab('ppo', 'The PPO clip, drawn', 'The objective as a function of the probability ratio, for positive and negative advantage. The flat regions are where the gradient is switched off — and the asymmetry between them is deliberate.')}

<h2><span class="sn">4.12.7</span> Reward hacking, with examples</h2>
<p>A learned reward model is a proxy, and optimising any proxy hard enough always eventually finds the places where it disagrees with what it was standing in for — this is the single unifying lesson underneath every example below, worth recognising as one phenomenon rather than a list of unrelated quirks. Answers get longer over training because length correlated with perceived quality in the human labels the reward model or preference data learned from (§4.12.3's documented DPO case, quantified). Models learn to hedge on everything because a hedged answer was rarely marked flatly wrong by raters averse to confidently-stated errors, even when a direct answer would have been more useful. Models mirror whatever view the user has just stated, because agreement reliably scored as helpful during training — <b>sycophancy</b>, a specific and well-documented reward-hacking failure. And they format answers with excessive headers, bullet points and bold text because raters, skimming quickly, rewarded the appearance of structure over its substance. Verifiable rewards (§4.12.5) remove the proxy entirely for maths and code, where a checker can confirm ground truth directly, but there is no equivalent checker for taste, tone or judgment — so the practical stance for anything that ships is: <b>verify what can be verified, keep a KL leash on everything that cannot be, and monitor length, refusal rate and sycophancy as first-class metrics tracked throughout training</b>, rather than discovering each one months later in production once users have already noticed.</p>

${H.probe([
      ['How does DPO get rid of the reward model?', 'Invert the closed-form optimal RLHF policy — derivable because a KL-regularised reward-maximisation objective is minimised exactly at $\\pi^*\\propto\\pi_{\\text{ref}}\\exp(r/\\beta)$ (§1.10\'s Gibbs inequality) — to express the reward in terms of the policy\'s own log-probability ratio. In the Bradley–Terry difference between two completions of the same prompt, the intractable partition function $Z(x)$ is identical on both sides and cancels exactly, leaving a plain logistic classification loss.'],
      ['GRPO vs PPO?', 'GRPO keeps PPO\'s clipped update mechanism but drops the learned value network, replacing it with a computed baseline: sample a group of $G$ responses to the same prompt and use the group\'s own mean and standard deviation as the advantage normaliser. Cheaper, more stable, and a natural fit whenever rewards are cheap automatic checks rather than an expensive learned model — exactly the RLVR setting.'],
      ['Why the KL penalty in RLHF?', 'A learned reward model is a proxy fit from finite, noisy preference data, and optimising it hard enough always eventually exploits places where it disagrees with genuine human judgment. The KL penalty taxes how far the policy has moved from a reference distribution that was, at least, producing recognisable text, which is what keeps that exploitation in check — clipping bounds each step, the KL penalty bounds the cumulative drift.']
    ], 'Presenting DPO as strictly better than RLHF and forgetting its length inflation.')}`,
    labs: {
      dpo: function (host) {
        const st = Viz.controls(host, [
          { k: 'beta', label: 'β (KL strength)', min: .05, max: 2, step: .05, value: .1, fmt: v => v.toFixed(2) },
          { k: 'margin', label: 'current implicit reward margin', min: -4, max: 4, step: .1, value: 0, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'loss', label: 'DPO loss', cls: 'key' }, { k: 'grad', label: 'gradient magnitude' },
          { k: 'pwin', label: 'implied P(preferred wins)' }, { k: 'drift', label: 'β says' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const L = m => -Math.log(Num.sigmoid(st.beta * m));
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [0, 3] })
              .frame({ xlabel: 'log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)', ylabel: 'loss' });
            P.clip(() => {
              [0.05, 0.1, 0.5, 1.0].forEach(b => {
                P.fn(m => -Math.log(Num.sigmoid(b * m)), { color: Math.abs(b - st.beta) < .001 ? T.blue : T.faint, width: Math.abs(b - st.beta) < .001 ? 2.8 : 1.2, alpha: Math.abs(b - st.beta) < .001 ? 1 : .6 });
              });
              P.fn(L, { color: T.blue, width: 2.8 });
              P.vline(st.margin, { color: T.red, dash: [4, 4] });
              P.dots([[st.margin, L(st.margin)]], { r: 5, color: T.red, stroke: true });
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText('grey curves: β = 0.05, 0.1, 0.5, 1.0', w - 16, 10);
            const g = st.beta * (1 - Num.sigmoid(st.beta * st.margin));
            out({
              loss: L(st.margin).toFixed(4), grad: g.toFixed(4),
              pwin: Num.sigmoid(st.beta * st.margin).toFixed(3),
              drift: st.beta < .15 ? 'stay close to the reference' : st.beta > .6 ? 'fit preferences hard' : 'balanced'
            });
          }
        });
        Viz.note(host, 'Small β makes the loss nearly flat: the policy is only weakly pushed away from the reference, which is precisely the KL constraint arriving through a different door. Large β fits the preference data aggressively and is where length exploitation and other reward-hacking behaviours appear fastest.');
      },

      grpo: function (host) {
        const st = Viz.controls(host, [
          { k: 'r1', label: 'reward · answer 1', min: 0, max: 1, step: 1, value: 1, fmt: v => v },
          { k: 'r2', label: 'answer 2', min: 0, max: 1, step: 1, value: 0, fmt: v => v },
          { k: 'r3', label: 'answer 3', min: 0, max: 1, step: 1, value: 1, fmt: v => v },
          { k: 'r4', label: 'answer 4', min: 0, max: 1, step: 1, value: 1, fmt: v => v },
          { k: 'r5', label: 'answer 5', min: 0, max: 1, step: 1, value: 0, fmt: v => v },
          { k: 'r6', label: 'answer 6', min: 0, max: 1, step: 1, value: 0, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mean', label: 'group mean', cls: 'key' }, { k: 'sd', label: 'group σ' },
          { k: 'signal', label: 'learning signal', cls: 'good' }, { k: 'note', label: 'status' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const r = [st.r1, st.r2, st.r3, st.r4, st.r5, st.r6];
            const mean = Num.mean(r), sd = Math.sqrt(Num.mean(r.map(v => (v - mean) ** 2)));
            const A = r.map(v => sd > 1e-9 ? (v - mean) / sd : 0);
            const bx = 140, bw = w - bx - 80, mid = bx + bw / 2;
            ctx.font = '12px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(mid, 26); ctx.lineTo(mid, 26 + 6 * 30); ctx.stroke();
            r.forEach((v, i) => {
              const y = 40 + i * 30;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText('answer ' + (i + 1) + '  r=' + v, bx - 10, y);
              const a = A[i];
              ctx.fillStyle = a > 0 ? T.green : a < 0 ? T.red : T.faint;
              const len = Math.min(bw / 2 - 4, Math.abs(a) * bw / 4);
              ctx.fillRect(a >= 0 ? mid : mid - len, y - 9, Math.max(2, len), 18);
              ctx.fillStyle = T.text; ctx.textAlign = a >= 0 ? 'left' : 'right';
              ctx.fillText('A = ' + (a >= 0 ? '+' : '') + a.toFixed(2), a >= 0 ? mid + len + 6 : mid - len - 6, y);
            });
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('baseline = the group mean', mid, 26 + 6 * 30 + 4);
            const degenerate = sd < 1e-9;
            ctx.fillStyle = degenerate ? T.red : T.green; ctx.font = '12px ui-sans-serif';
            ctx.fillText(degenerate ? 'σ = 0 — every advantage is zero: this prompt teaches nothing and should be dropped'
              : 'correct completions reinforced, incorrect suppressed — no value network anywhere', mid, 26 + 6 * 30 + 22);
            out({
              mean: mean.toFixed(3), sd: sd.toFixed(3),
              signal: degenerate ? 'none' : 'present',
              note: degenerate ? (mean > .5 ? 'too easy' : 'too hard') : 'intermediate difficulty ✓'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'The worked example [1,0,1,1,0,0]', primary: true, on: () => { st.$set('r1', 1); st.$set('r2', 0); st.$set('r3', 1); st.$set('r4', 1); st.$set('r5', 0); st.$set('r6', 0); S.redraw(); } },
          { label: 'All correct (degenerate)', on: () => { [1, 2, 3, 4, 5, 6].forEach(i => st.$set('r' + i, 1)); S.redraw(); } },
          { label: 'All wrong (degenerate)', on: () => { [1, 2, 3, 4, 5, 6].forEach(i => st.$set('r' + i, 0)); S.redraw(); } }
        ]);
      },

      ppo: function (host) {
        const st = Viz.controls(host, [
          { k: 'eps', label: 'clip ε', min: .05, max: .5, step: .01, value: .2, fmt: v => v.toFixed(2) },
          { k: 'adv', label: 'advantage A', min: -2, max: 2, step: .1, value: 1, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'range', label: 'gradient active range', cls: 'key' }, { k: 'at', label: 'objective at ratio 1' }, { k: 'note', label: 'behaviour' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const A = st.adv, e = st.eps;
            const obj = r => Math.min(r * A, Math.max(1 - e, Math.min(1 + e, r)) * A);
            const P = Viz.plot(ctx, w, h, { xd: [0, 2.2], yd: [Math.min(-2.6, -Math.abs(A) * 1.4), Math.max(2.6, Math.abs(A) * 1.4)] })
              .frame({ xlabel: 'probability ratio r = π_θ / π_old', ylabel: 'clipped objective' });
            P.clip(() => {
              P.fn(r => r * A, { color: T.faint, width: 1.4, dash: [5, 4] });
              P.fn(obj, { color: T.blue, width: 3 });
              P.vline(1, { color: T.faint, dash: [3, 3] });
              P.vline(1 - e, { color: T.red, dash: [3, 3] });
              P.vline(1 + e, { color: T.red, dash: [3, 3] });
              ctx.fillStyle = 'rgba(200,80,70,.10)';
              ctx.fillRect(P.x(1 - e), P.pad.t, P.x(1 + e) - P.x(1 - e), P.ph);
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('shaded = trust region; outside it in the favourable direction the objective is flat and the gradient is zero', P.pad.l + 6, 10);
            out({
              range: A > 0 ? '[0, ' + (1 + e).toFixed(2) + ']' : '[' + (1 - e).toFixed(2) + ', ∞)',
              at: (1 * A).toFixed(2),
              note: A > 0 ? 'improvement capped at 1+ε' : 'punishment capped at 1−ε'
            });
          }
        });
        Viz.note(host, 'Flip the advantage negative and the flat region moves to the other side. The asymmetry is the point: PPO will always let you move <i>away</i> from a bad action, but caps how far one batch can push you toward a good one — which is what stops a single lucky batch destroying the policy.');
      }
    },
    quiz: [
      {
        q: 'In the DPO derivation, the partition function $Z(x)$ cancels because…',
        options: ['it is approximated by sampling', 'the reward enters only as a difference between two completions of the same prompt', 'it is set to 1 by construction', 'the reference policy is frozen'],
        answer: 1,
        why: 'Once the reward is expressed purely through the policy\'s own log-probability ratio against the reference — the move §4.12.3\'s derivation makes by inverting the closed-form optimal policy — the term $\\beta\\log Z(x)$ depends only on the prompt x, never on which response y is being scored, so when the Bradley–Terry model asks for the difference $r(x,y_w)-r(x,y_l)$ between two completions of the same prompt, that identical term appears on both sides and subtracts to exactly zero. "It is approximated by sampling" is the tempting wrong answer because Z(x), a sum over every possible response, genuinely is intractable to compute directly, and sampling an intractable normaliser is the standard fix in most other settings — but DPO\'s whole point is that no approximation is needed here, because the algebraic structure of a difference between two completions of the same prompt makes the term vanish exactly, not approximately. "It is set to 1 by construction" misdescribes what a partition function is, and "the reference policy is frozen" is true but irrelevant to why this term cancels. The general principle, §4.12.3\'s whole payoff, is that DPO needs no reward model, no partition function and no sampling loop anywhere, purely because of this cancellation.'
      },
      {
        q: 'GRPO rewards for one prompt are [1,1,1,1,1,1]. What is the learning signal?',
        options: ['Strongly positive', 'Zero — σ = 0, so every advantage is zero and the prompt teaches nothing', 'Negative', 'Undefined but usable'],
        answer: 1,
        why: 'GRPO uses the sampled group\'s own mean and standard deviation as the baseline rather than a learned value network, so the advantage for every response is $(r_i-\\text{mean}(r))/\\text{std}(r)$; with all six rewards equal to 1 the standard deviation is exactly zero, every advantage is zero (or undefined), and — as the worked example in §4.12.4 shows explicitly — the prompt teaches nothing at all, however many times it is resampled. "Strongly positive" is the tempting wrong answer for anyone reasoning from the raw reward alone: every response genuinely did get the maximum score, and it feels natural that a perfect batch should produce a strong signal — but GRPO\'s signal is relative to the group, not absolute, and a unanimous group carries no information about which member to prefer. "Negative" and "undefined but usable" both misread the mechanism: nothing here is being punished, and a genuinely undefined advantage is exactly what real implementations detect and drop rather than silently use. The general principle is that GRPO\'s learning signal lives entirely on prompts of intermediate difficulty, which is why curriculum and difficulty filtering matter more here than in ordinary RLHF.'
      },
      {
        q: 'The most commonly cited failure of DPO is…',
        options: ['reward-model overfitting', 'response length inflation (arXiv:2403.19159)', 'catastrophic forgetting', 'mode collapse to one answer'],
        answer: 1,
        why: 'Park et al.\'s 2024 paper (arXiv:2403.19159) documents this directly: because human preference data frequently correlates "longer" with "more thorough, and therefore better", DPO happily learns to exploit that correlation rather than the underlying quality it was meant to proxy for, inflating response length without a matching gain in genuine quality — exactly the failure §4.12.3 flags as needing mean response length tracked as a first-class training metric. "Reward-model overfitting" is the tempting wrong answer because it sounds like the natural failure mode of a preference-learning method, but DPO\'s entire selling point is that it has no reward model to overfit at all — the reward is implicit in the policy\'s own log-probabilities, so this specific failure mode cannot arise in the form it takes for RLHF. "Catastrophic forgetting" and "mode collapse to one answer" are real failure modes elsewhere in this part but are not what the documented DPO literature actually reports. The general principle is that any preference-learning method inherits whatever spurious correlations exist in its training signal, and length is the specific, measured, named instance of that for DPO.'
      }
    ],
    cards: [
      { q: 'DPO loss', a: '$-\\log\\sigma(\\beta\\log\\frac{\\pi(y_w)}{\\pi_{ref}(y_w)}-\\beta\\log\\frac{\\pi(y_l)}{\\pi_{ref}(y_l)})$ — no reward model; Z(x) cancels.' },
      { q: 'GRPO advantage', a: '$A_i=(r_i-\\mathrm{mean}(r))/\\mathrm{std}(r)$ over a sampled group; the group is the baseline, no critic.' },
      { q: 'RLVR', a: 'Rule-based verifiable rewards (tests pass, answer matches) — nothing to hack; how reasoning models are actually trained.' },
      { q: 'PPO clip vs KL', a: 'Clipping bounds each update; the KL penalty bounds total drift from the reference.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.13 */
  ML.section({
    id: 'lora', track: 'llm', num: '4.13',
    title: 'PEFT: LoRA and QLoRA',
    lede: 'The clearest case of §1.8’s low-rank idea paying rent — and a memory calculation that explains why parameter-efficient fine-tuning won.',
    html: `
<p>§4.11 already put a number on the cost of adapting a model the ordinary way: full fine-tuning means updating <i>every</i> parameter, which means holding gradients and Adam optimiser state for every one of them, on top of the weights themselves — the exact memory arithmetic that made ZeRO/FSDP sharding necessary in the first place. For a narrow adaptation task — teach the model your company's tone of voice, your JSON schema, your support-ticket format — paying that full cost to move the model a comparatively short distance in weight space starts to look wasteful, and the question this section answers is whether it actually is.</p>

<h2><span class="sn">4.13.1</span> LoRA</h2>
<p>The working hypothesis behind <b>LoRA</b> (low-rank adaptation) is that whatever change fine-tuning needs to make to a weight matrix $W_0$ for a specific downstream task does not require the full $d\\times d$ space of possible changes — it lives, to good approximation, in a much lower-dimensional subspace. That is precisely §1.8's low-rank idea, the same Eckart–Young result that says a matrix's best rank-$r$ approximation captures most of what matters about it: rather than learning an arbitrary $d\\times d$ update $\\Delta W$, LoRA constrains it to a product of two thin matrices, $\\Delta W = BA$ with $B \\in \\mathbb{R}^{d\\times r}$, $A \\in \\mathbb{R}^{r\\times d}$ and $r \\ll d$, and only trains $B$ and $A$ while $W_0$ itself stays completely frozen. The forward pass becomes $W_0x + \\tfrac{\\alpha}{r}BAx$ — the frozen original output, plus a learned low-rank correction, scaled by a factor $\\alpha/r$ chosen so that the correction's typical magnitude stays roughly comparable as you change the rank $r$, letting the same learning rate work reasonably well whichever rank you pick rather than needing to be retuned every time.</p>
${H.history(`<p>That working hypothesis was not a guess. Aghajanyan et al. (2020) asked a more basic question first: how many degrees of freedom does fine-tuning actually <i>use</i>, regardless of how you parameterise the update? Their method was to fine-tune inside a randomly projected low-dimensional subspace — pick a random direction matrix, optimise only the coordinates within it, and project back up to the full parameter space to compute the loss — and increase the subspace's dimension until performance stopped improving. For RoBERTa-large fine-tuned on MRPC, optimising only <b>200</b> parameters this way, embedded in a random subspace of the full weight space, recovered 90% of full fine-tuning's performance. The number itself matters less than what it rules out: the update fine-tuning needs to make is nowhere close to using all the directions the full parameter count nominally offers, and — a second finding from the same paper — that intrinsic dimension shrinks further as the pretrained model gets larger, meaning bigger models need proportionally <i>less</i> of their own capacity touched to adapt to a new task, not more. LoRA is a direct, practical bet on that finding: rather than a random projection with no learned structure, let $A$ and $B$ themselves be learned, but hold them to the same order of low dimensionality the intrinsic-dimension experiments already showed was sufficient.</p>`)}
<p>The parameter saving is dramatic and easy to see directly: a $d\\times d$ matrix has $d^2$ parameters, while $B$ and $A$ together have $dr + rd = 2dr$. For $d=4096$ and $r=16$, that is $2\\times4096\\times16 \\approx 131{,}000$ against $4096^2 \\approx 16.8$ million — under 1% of the original matrix's parameter count, for a single target matrix, and the full-model figure worked out below lands in the same range once every targeted matrix in every layer is counted. Because $W_0$ never changes, there is no need to hold Adam's expensive optimiser state (§4.11's ≈12 bytes/parameter) for anything but $B$ and $A$ — the overwhelming majority of the model's parameters need no gradient and no optimiser state at all, which is where essentially all of the memory saving in the worked example below actually comes from. You train roughly 0.5% of the parameters, checkpoints are tens of megabytes rather than tens of gigabytes, and because $B$ and $A$ can be added to or removed from $W_0$ independently, an adapter can be hot-swapped at serve time — load a different tiny $B$, $A$ pair for a different customer or task without touching the multi-gigabyte base model at all.</p>
<p>One detail in how $B$ and $A$ are set up at the start of training is easy to skip past and is not cosmetic. $A$ is initialised with small random Gaussian entries — an ordinary random linear-layer initialisation, exactly the kind §2's weight-init discussion already covers — but $B$ is initialised to <i>exactly zero</i>. The consequence is immediate: at the first training step, $\\Delta W = BA = 0$, so the adapted model's forward pass $W_0x + \\tfrac{\\alpha}{r}BAx$ collapses to $W_0x$ — bit-for-bit the unmodified pretrained model. This is the same trick used to zero-initialise the output side of a residual branch elsewhere in deep learning (a ResNet's final batch-norm scale, or a transformer's output projection in some training recipes): an additive branch that starts as a no-op cannot make the model any worse than it already was before training begins, and every gradient step from that point on is spent making a genuine, measured improvement rather than first undoing self-inflicted damage.</p>
${H.pitfall('If both $A$ and $B$ were initialised to nonzero random values instead, $\\Delta W=BA$ would already be a nonzero random matrix at step zero — injecting unstructured noise into a model that, before any fine-tuning, was already producing coherent, well-calibrated output. Early training would spend its first updates cancelling out that self-inflicted corruption before it could start on the task, and starting loss could be visibly worse than the untouched base model\'s. Only one of the two factors needs to start at zero for the product to vanish; the convention is $B$, so the random, correctly-scaled initialisation lives in $A$, exactly like an ordinary linear layer\'s weights, while $B$\'s starting value is the trivial one.')}

<h2><span class="sn">4.13.2</span> QLoRA</h2>
<p>LoRA already avoids optimiser state for the frozen base, but the base weights themselves still have to sit resident in memory at full or near-full precision to be read during the forward and backward pass. <b>QLoRA</b> shrinks that too, by keeping the frozen base compressed to 4-bit <b>NF4</b> — "4-bit NormalFloat", a quantisation scheme whose representable values are spaced to match the roughly Gaussian distribution neural network weights actually follow, rather than spaced uniformly across the numeric range the way a naive 4-bit format would be, which wastes far fewer of its limited 16 representable levels on weight magnitudes that almost never occur. <b>Double quantisation</b> squeezes a further small saving out of the scheme by quantising the quantisation constants themselves — the per-block scaling factors NF4 needs are, in aggregate, a real amount of memory at billions of parameters, and compressing them too is nearly free additional saving. <b>Paged optimisers</b> borrow the operating system's virtual-memory trick directly: when GPU memory comes briefly under pressure — a longer-than-usual sequence in the current batch, say — optimiser state pages out to ordinary CPU memory rather than crashing the run, and pages back in once space frees up. Throughout all of this, gradients still flow <i>through</i> the frozen quantised weights during backpropagation — the base is dequantised on the fly for each matrix multiply, and the chain rule passes a gradient back through that computation exactly as it would through any other layer — but nothing ever accumulates an update <i>into</i> the base weights; only $B$ and $A$, kept in ordinary BF16, are ever modified. That combination is what makes a 70B fine-tune fit on a single large GPU rather than a rack of them.</p>

<h3>2026 defaults worth quoting</h3>
<p><b>$r = 16$ with $\\alpha = 2r$, targeting <i>all</i> linear layers</b> — attention q, k, v, o plus the MLP gate, up and down projections — rising to $r = 32$–$64$ for harder tasks or larger data. Targeting attention only is the older, weaker recipe. <b>DoRA</b> (decomposing magnitude from direction) is a common quality upgrade at the same budget. Prefix- and prompt-tuning still exist but are largely superseded.</p>
${H.intuition(`<p>$r$ and $\\alpha$ are not the same knob wearing two names, and it is worth being precise about what each one actually controls. $r$ sets <i>capacity</i> — the dimension of the subspace the correction is allowed to live in, in the exact §1.8 sense of how many singular directions a rank-$r$ approximation keeps. Too low a rank and the update simply cannot represent whatever the task genuinely needs changed, the way the "full-rank noise" setting in the lab below never lets a small rank capture much of the matrix at all; too high a rank stops buying anything once the task's real update has been captured, and only adds trainable parameters and overfitting risk for no further gain — the empirical "knee" in that same lab's reconstruction-error curve is precisely where a real task's intrinsic rank stops paying rent, echoing Aghajanyan et al.'s finding above. $\\alpha$, by contrast, does not touch capacity at all; it is purely a <i>magnitude</i> dial on top of whatever $B$ and $A$ have learned, rescaling the correction's contribution to the forward pass by the fixed factor $\\alpha/r$ without changing the dimension it lives in. The reason the scaling is written as $\\alpha/r$ rather than a bare constant is exactly the practical property called out above: raising $r$ alone, with no compensating change, tends to raise the typical magnitude of $BA$ simply because there are more rank-one terms being summed into it, and dividing by $r$ counteracts that so a learning rate tuned at one rank keeps behaving sensibly at another. In practice this means the two hyperparameters are swept differently: $r$ is a genuine capacity search, tried at a few values and checked against a validation set the way any model-capacity choice is; $\\alpha$ is far more often left at the fixed convention $\\alpha=2r$ and simply not retuned at all, because its entire job was to remove the need to.</p>`)}

${H.worked('worked number — what an 8B QLoRA actually costs', `
<p>Take the 8B model counted in §4.5, $r=16$, $\\alpha=32$, all seven linear projections per layer (q, k, v, o, gate, up, down), 32 layers.</p>
<p><b>Adapter parameters.</b> Each target of shape $m\\times n$ contributes $r(m+n)$. Per layer roughly $16\\times(4096{+}4096)\\times2$ for q/o, $16\\times(4096{+}1024)\\times2$ for k/v, and $16\\times(4096{+}14336)\\times3$ for the MLP ≈ 0.26M + 0.16M + 0.88M ≈ <b>1.3M</b>. Over 32 layers ≈ <b>42M trainable — 0.52% of the model.</b></p>
<p><b>Memory.</b> Frozen base in 4-bit NF4: 8.03B × 0.5 B ≈ <b>4.0 GB</b>. Adapter weights in BF16: 84 MB. Adam state for the adapter only (≈12 B/param): 0.5 GB. Gradients: 84 MB. Activations with checkpointing at moderate sequence length and batch size: a couple of GB.</p>
<p><b>Total ≈ 7 GB</b> — comfortably inside a single mid-range consumer card, before any headroom you would add for a longer sequence or larger batch. Compare full fine-tuning at BF16: 16 GB of weights, <i>plus</i> roughly 16 bytes/parameter of gradient-and-Adam overhead — another ≈128 GB — for a total of <b>≈145 GB</b> before activations. <mark>That factor of roughly twenty, not the accuracy, is why PEFT won.</mark></p>`)}

<p><b>What you are looking at.</b> Every memory consumer for a LoRA fine-tune — frozen base, adapter weights, adapter optimiser state, adapter gradients, activations — stacked as a bar and computed live from the rank, targets and base precision you choose, with the full-fine-tune comparison figure printed alongside for scale.</p>
<p><b>What to do with it.</b> Load the worked 8B QLoRA preset and confirm the readout's trainable-parameter and memory figures match the worked box above almost exactly. Then switch targets from "all linear layers" to "attention only" and watch the trainable-parameter count fall further still — at the cost, in practice, of adapting a narrower slice of the model's behaviour.</p>
<p><b>The thing genuinely worth noticing.</b> Drag the base precision from 4-bit up to BF16 and watch the frozen-base bar alone grow past the entire PEFT total you started with — the base weights, even <i>frozen</i> and requiring no gradient, are still the single largest line item in the whole budget, which is exactly why QLoRA's quantisation of the base, not just LoRA's low-rank adapter, is what makes a 70B-class fine-tune fit on one consumer-accessible card.</p>

${H.lab('lora', 'LoRA cost calculator', 'Rank, targets and precision, with the memory total and the full-fine-tune comparison alongside. The preset reproduces the worked example exactly.')}

<p><b>What you are looking at.</b> A real, randomly generated matrix, decomposed by an actual SVD (§1.8) rather than a stand-in, with reconstruction error plotted against how many singular values — how much rank — you keep, for three different underlying structures ranging from genuinely low-rank to pure noise.</p>
<p><b>What to do with it.</b> On the "genuinely low-rank" setting, drag the kept-rank slider up from 1 and watch the reconstruction error plunge to near zero within the first few ranks and then flatten — a sharp knee in the curve. Switch to "full-rank noise" and watch the same curve fall almost perfectly linearly instead, with no knee anywhere.</p>
<p><b>The thing genuinely worth noticing.</b> LoRA's entire premise is a bet that a fine-tuning task's weight update looks like the first curve, not the second — that there is a knee, and that it sits somewhere near the ranks people actually use in production (16, 32, 64). The lab lets you see exactly what it would mean for that bet to fail: a genuinely high-rank required update, where no small rank captures most of the structure, is precisely the case where LoRA underperforms and a heavier tool — full fine-tuning, or continual pretraining (§4.9) — is earning its cost rather than wasting it.</p>

${H.lab('rank', 'What rank actually buys', 'A real matrix, approximated at increasing rank via SVD, with the reconstruction error and the parameter count plotted together. The knee in the curve is the empirical justification for r = 16 — and you can watch it move when the underlying matrix is genuinely high-rank.')}

<h2><span class="sn">4.13.3</span> Serving adapters</h2>
<p>An adapter can be <b>merged</b> — compute $W_0 + \\tfrac{\\alpha}{r}BA$ once, offline, and ship a single ordinary weight matrix indistinguishable in shape from the original — giving zero added latency at serve time, at the cost of losing hot-swappability, since the merged weights are now one fixed model rather than a base plus a removable adapter. Or an adapter can be <b>kept separate</b>, applied at inference time as an extra small matrix multiply on top of the frozen base, so that one base model resident in memory can serve dozens of different tenant-specific or task-specific adapters at a small per-request compute cost, switching between them per request with no reloading. Note the trap in merging into a <i>quantised</i> base specifically: the adapter was trained to correct for the base <i>as it actually behaves in 4-bit NF4</i>, so merging it back into the original full-precision weights and re-quantising afterwards is not the same operation and can lose accuracy — merge into the original BF16 checkpoint and requantise from there, or simply keep the adapter separate and never merge at all.</p>
${H.flag('And know when LoRA is the wrong tool: it adapts behaviour and format well, but it is a poor way to install substantial new knowledge — that is what continual pretraining (§4.9) or retrieval (§5.1) is for.')}

<h2><span class="sn">4.13.4</span> Model merging — cheaper than multi-task training</h2>
<p>Because fine-tuning, LoRA or full, moves a model's weights only a comparatively short distance from its shared starting point, two independently fine-tuned checkpoints of the same base often sit close enough together in weight space that their individual displacements do not badly interfere with each other — which opens up an option stranger and cheaper than it sounds. A <b>task vector</b> is simply the difference $\\theta_{\\text{finetuned}} - \\theta_{\\text{base}}$: the direction and distance one fine-tuning run moved the weights, stored as a vector rather than as "a model". Add two different tasks' vectors back onto the same base and the result frequently behaves competently at both tasks at once, despite never having been trained on the combination; subtracting a task vector, instead of adding it, can suppress a specific learned behaviour from a model that already has it. Naive addition does interfere when two tasks pull weights in genuinely conflicting directions, so several refinements exist to reduce that interference: <b>TIES</b> trims each task vector's smallest-magnitude entries (which are more likely to be noise than signal) and resolves sign disagreements between task vectors before summing them; <b>DARE</b> randomly drops a large fraction of each task vector's entries and rescales the rest to compensate, which empirically reduces interference for reasons not fully settled; and <b>SLERP</b> interpolates between two checkpoints along the surface of the hypersphere connecting them rather than along the straight line between them, which tends to preserve each model's norm and behaviour better than linear averaging does. Merging needs no training data and no gradient steps at all — it is pure arithmetic on already-trained weights — which makes it the fastest available way to combine capabilities, and its hard limit is exactly the assumption it rests on: it only works between models that share a base and have not drifted too far apart from it.</p>

${H.probe([
      ['What does the rank control?', 'Adapter capacity, in the same sense as keeping more singular values in a low-rank matrix approximation (§1.8): too low underfits the task, too high spends memory and risks overfitting for no further gain. Find the knee empirically, starting at r=16 — the lab shows the knee directly on a real SVD.'],
      ['The QLoRA trick?', 'Keep the frozen base compressed to 4-bit NF4 (spaced to match weights\' roughly Gaussian distribution) with double-quantised scaling constants; train only the BF16 adapter. Gradients flow through the dequantised base during backprop but never update it — only B and A ever change.'],
      ['When is LoRA the wrong tool?', 'When the task needs genuinely new knowledge rather than a new behaviour or format — a low-rank correction to existing weights has nowhere near the capacity to install large amounts of new factual content the base model never saw. Use continual pretraining (§4.9) or retrieval (§5.1) instead.']
    ])}`,
    labs: {
      lora: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'base model (B params)', min: .5, max: 180, step: .5, value: 8, fmt: v => v.toFixed(1) + 'B' },
          { k: 'd', label: 'hidden size d', min: 512, max: 16384, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'L', label: 'layers', min: 8, max: 128, step: 1, value: 32, fmt: v => v },
          { k: 'ffn', label: 'FFN width', min: 1024, max: 65536, step: 256, value: 14336, fmt: v => v.toLocaleString() },
          { k: 'r', label: 'rank r', min: 1, max: 128, step: 1, value: 16, fmt: v => v },
          { k: 'targets', label: 'targets', type: 'buttons', value: 'all', options: [{ v: 'attn', t: 'attention only' }, { v: 'all', t: 'all linear layers' }] },
          { k: 'quant', label: 'base precision', type: 'buttons', value: '0.5', options: [{ v: '2', t: 'BF16' }, { v: '1', t: 'FP8' }, { v: '0.5', t: '4-bit NF4' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'trainable', label: 'trainable parameters', cls: 'key' }, { k: 'pct', label: 'share of the model' },
          { k: 'mem', label: 'total memory', cls: 'good' }, { k: 'full', label: 'full fine-tune would need', cls: 'bad' }, { k: 'ckpt', label: 'checkpoint size' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const d = st.d, r = st.r, kvW = Math.round(d / 4);
            const perLayerAttn = r * (d + d) * 2 + r * (d + kvW) * 2;
            const perLayerMlp = r * (d + st.ffn) * 3;
            const perLayer = st.targets === 'all' ? perLayerAttn + perLayerMlp : perLayerAttn;
            const trainable = perLayer * st.L;
            const N = st.N * 1e9;
            const baseGB = N * (+st.quant) / 1e9;
            const adapterGB = trainable * 2 / 1e9;
            const optGB = trainable * 12 / 1e9;
            const gradGB = trainable * 2 / 1e9;
            const actGB = 2.5;
            const total = baseGB + adapterGB + optGB + gradGB + actGB;
            const fullFT = N * 2 / 1e9 + N * 16 / 1e9;
            const parts = [
              ['frozen base (' + (st.quant === '0.5' ? '4-bit NF4' : st.quant === '1' ? 'FP8' : 'BF16') + ')', baseGB, T.faint],
              ['adapter weights', adapterGB, T.blue],
              ['adapter optimiser state', optGB, T.red],
              ['gradients', gradGB, T.amber],
              ['activations (checkpointed)', actGB, T.green]
            ];
            const bx = 20, bw = w - 40, by = 34;
            let x = bx;
            parts.forEach(p => { const pw = bw * p[1] / Math.max(total, fullFT * .35); ctx.fillStyle = p[2]; ctx.fillRect(x, by, pw, 30); x += pw; });
            ctx.font = '12px ui-sans-serif'; ctx.textBaseline = 'middle';
            parts.forEach((p, i) => {
              const yy = by + 56 + i * 22;
              ctx.fillStyle = p[2]; ctx.fillRect(bx, yy - 6, 12, 12);
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.fillText(p[0], bx + 20, yy);
              ctx.fillStyle = T.muted; ctx.textAlign = 'right'; ctx.fillText(p[1].toFixed(2) + ' GB', bx + bw, yy);
            });
            ctx.fillStyle = T.green; ctx.font = 'bold 14px ui-monospace, monospace'; ctx.textAlign = 'left';
            ctx.fillText('PEFT total ≈ ' + total.toFixed(1) + ' GB', bx, by + 56 + 5 * 22 + 8);
            ctx.fillStyle = T.red;
            ctx.fillText('full fine-tune ≈ ' + fullFT.toFixed(0) + ' GB', bx + 220, by + 56 + 5 * 22 + 8);
            out({
              trainable: (trainable / 1e6).toFixed(1) + 'M', pct: (100 * trainable / N).toFixed(2) + '%',
              mem: total.toFixed(1) + ' GB', full: fullFT.toFixed(0) + ' GB',
              ckpt: (trainable * 2 / 1e6).toFixed(0) + ' MB'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'The worked 8B QLoRA', primary: true, on: () => { st.$set('N', 8); st.$set('d', 4096); st.$set('L', 32); st.$set('ffn', 14336); st.$set('r', 16); st.$set('targets', 'all'); st.$set('quant', '0.5'); S.redraw(); } },
          { label: '70B QLoRA', on: () => { st.$set('N', 70); st.$set('d', 8192); st.$set('L', 80); st.$set('ffn', 28672); S.redraw(); } }
        ]);
      },

      rank: function (host) {
        const N = 40;
        const st = Viz.controls(host, [
          { k: 'r', label: 'rank r', min: 1, max: 30, step: 1, value: 8, fmt: v => v },
          { k: 'struct', label: 'underlying structure', type: 'buttons', value: 'low', options: [{ v: 'low', t: 'genuinely low-rank' }, { v: 'mid', t: 'mixed' }, { v: 'high', t: 'full-rank noise' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'err', label: 'reconstruction error', cls: 'key' }, { k: 'energy', label: 'energy captured' },
          { k: 'params', label: 'parameters stored' }, { k: 'save', label: 'saving vs full matrix' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(29);
            const M = Array.from({ length: N }, () => new Array(N).fill(0));
            if (st.struct === 'high') {
              for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] = R.normal(0, 1);
            } else {
              const k = st.struct === 'low' ? 3 : 10;
              for (let c = 0; c < k; c++) {
                const u = Array.from({ length: N }, () => R.normal(0, 1)), v = Array.from({ length: N }, () => R.normal(0, 1));
                const s = 1 / (c + 1);
                for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] += s * u[i] * v[j];
              }
              if (st.struct === 'mid') for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] += R.normal(0, .12);
            }
            const { U, s, V } = Num.svd(M);
            const total = s.reduce((a, b) => a + b * b, 0);
            const errs = [];
            for (let r = 1; r <= 30; r++) {
              let kept = 0; for (let k = 0; k < r; k++) kept += s[k] * s[k];
              errs.push([r, Math.sqrt(Math.max(0, 1 - kept / total))]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [1, 30], yd: [0, 1] })
              .frame({ xlabel: 'rank r kept', ylabel: 'relative reconstruction error' });
            P.clip(() => {
              P.line(errs, { color: T.blue, width: 2.6 });
              P.dots(errs, { r: 2.6, color: T.blue });
              P.vline(st.r, { color: T.red, dash: [4, 4] });
              P.dots([errs[st.r - 1]], { r: 5.5, color: T.red, stroke: true });
              P.hline(.1, { color: T.green, dash: [3, 3], label: '10% error' });
            });
            let kept = 0; for (let k = 0; k < st.r; k++) kept += s[k] * s[k];
            out({
              err: (errs[st.r - 1][1] * 100).toFixed(1) + '%',
              energy: (100 * kept / total).toFixed(1) + '%',
              params: (2 * N * st.r).toLocaleString(),
              save: (100 * (1 - 2 * N * st.r / (N * N))).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'On a genuinely low-rank matrix, r = 4 already captures nearly everything and higher ranks buy nothing — that is the LoRA bet. On full-rank noise the error falls linearly and no small rank helps, which is exactly the case where LoRA underperforms and you need full fine-tuning or continual pretraining.');
      }
    },
    quiz: [
      {
        q: 'An 8B QLoRA at r=16 on all linear layers trains roughly…',
        options: ['0.05% of parameters', '0.5% of parameters', '5% of parameters', '20% of parameters'],
        answer: 1,
        why: 'The worked calculation in §4.13.2 counts it exactly: roughly 42M adapter parameters out of 8.03B total is about 0.52%, and that tiny trainable fraction is what shrinks the memory footprint to roughly 7 GB against the ≈145 GB a full BF16 fine-tune needs once gradients and Adam\'s optimiser state for every parameter are counted. "5%" is the tempting wrong answer for anyone who remembers "LoRA trains a small percentage" correctly in spirit but overshoots the actual figure by an order of magnitude — still small next to 100%, but rank-16 adapters on a handful of linear projections per layer are a far tinier fraction of a multi-billion-parameter model than that. "0.05%" undershoots by the same order in the other direction, and "20%" is nowhere near what a low-rank decomposition with $r\\ll d$ actually costs. The general principle, §1.8\'s low-rank idea paying rent exactly as promised, is that a $d\\times d$ update constrained to rank r needs only $2dr$ parameters rather than $d^2$, and for realistic d and r=16 that ratio is a small fraction of a percent.'
      },
      {
        q: 'Merging a LoRA adapter into a 4-bit quantised base is…',
        options: ['exact', 'lossy — you trained against the quantised weights, so merge into the BF16 original and re-quantise', 'impossible', 'faster than keeping it separate but identical in quality'],
        answer: 1,
        why: 'The adapter B, A was trained to correct for the base weights as they actually behave in 4-bit NF4 — every gradient it saw during training passed through the dequantised-on-the-fly quantised base, not the original full-precision weights — so folding it back into the full-precision original and requantising afterwards is a different operation from the one it was trained against, and can measurably lose accuracy; §4.13.3 recommends merging into the original BF16 checkpoint and requantising from there, or simply keeping the adapter separate. "Exact" is the tempting wrong answer because merging an adapter into an unquantised base genuinely is exact arithmetic — $W_0+\\frac{\\alpha}{r}BA$ computed once — and it is easy to assume that property survives once quantisation enters the picture, when quantisation is exactly the step that breaks it. "Impossible" overstates the problem, since the merge is computationally straightforward, just potentially inaccurate, and "faster but identical in quality" gets the trade-off backwards, since a merged model has no separate adapter left to keep at all. The general principle is that a QLoRA adapter\'s training target is the quantised base\'s actual behaviour, not an idealised full-precision version of it, so any operation on the base has to respect which version the adapter was actually trained against.'
      },
      {
        q: 'A task vector is…',
        options: ['the gradient at convergence', '$\\theta_{finetuned} - \\theta_{base}$, which can be added to or subtracted from the base', 'the LoRA B matrix', 'the embedding of the task description'],
        answer: 1,
        why: '§4.13.4 defines it exactly this way: the displacement $\\theta_{finetuned}-\\theta_{base}$ that one fine-tuning run produced in weight space, stored as a vector that can be added back onto the shared base — combining two tasks\' vectors at once, or subtracted to suppress a learned behaviour — entirely through arithmetic on already-trained weights, with no further training data or gradient steps required. "The LoRA B matrix" is the tempting wrong answer because both ideas involve a small, portable object capturing what fine-tuning changed, and it is easy to conflate "a compact representation of an adaptation" with "the specific low-rank factor one particular method happens to use" — but a task vector is defined for any fine-tuning method, LoRA or full, and lives in the full parameter space rather than a constrained low-rank subspace. "The gradient at convergence" and "the embedding of the task description" both name different objects with no role in model merging. The general principle is that because fine-tuning moves weights only a comparatively short distance from a shared starting point, these displacement vectors from different tasks often do not badly interfere with each other, which is what makes TIES, DARE and SLERP-style merging viable as a cheaper alternative to multi-task training.'
      }
    ],
    cards: [
      { q: 'LoRA', a: 'Freeze $W_0$, learn $\\Delta W=BA$ with $r\\ll d$; forward $W_0x+\\frac{\\alpha}{r}BAx$. Defaults r=16, α=2r, all linear layers.' },
      { q: 'QLoRA', a: '4-bit NF4 frozen base + BF16 adapter; gradients flow through the quantised weights without updating them.' },
      { q: 'The 8B QLoRA numbers', a: '≈42M trainable (0.52%), ≈7 GB total, vs ≈145 GB for full BF16 fine-tuning — roughly a twentyfold gap.' },
      { q: 'Model merging', a: 'Add task vectors ($\\theta_{ft}-\\theta_{base}$); TIES trims and resolves signs, DARE drops and rescales, SLERP interpolates on the sphere.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.14 */
  ML.section({
    id: 'serving', track: 'llm', num: '4.14',
    title: 'Inference and serving',
    lede: 'The most practically examined section in Part 4. Know the arithmetic, not the vibes.',
    html: `
<p>A user watching a chat response arrive experiences exactly two numbers, whether they have the vocabulary for them or not: how long they wait before anything appears, and how fast words stream in once they start. Those two numbers turn out to come from two structurally different phases of the same request, governed by opposite sides of §4.7's roofline distinction, and conflating them — or optimising only one while a system is actually bottlenecked on the other — is the single most common mistake in serving discussions.</p>

<h2><span class="sn">4.14.1</span> Two phases with opposite characters</h2>
<p><b>Prefill</b> is the model reading the entire prompt for the first time: every prompt token's query, key and value can be computed together, in one large batched matrix multiply, exactly the way a training batch is processed (§4.5) — because unlike decode, prefill is not waiting on any token it has not already been given. Large matmuls, high arithmetic intensity, excellent GPU utilisation: prefill is <b>compute-bound</b>, and how long it takes sets <b>time to first token</b> (TTFT), the wait before the user sees anything at all. <b>Decode</b> is the opposite in every respect: each subsequent token depends on the one just generated, so it can only be produced one step at a time, and §4.7 already showed exactly what that one step costs — every weight and the whole KV cache streamed from HBM to produce a single token's worth of arithmetic. Decode is <b>memory-bandwidth-bound</b>, and its per-token pace sets <b>time per output token</b> (TPOT), the rate words stream in afterward. Batching helps decode enormously, because the same weight read now serves every sequence in the batch at once (§4.7's central argument); it barely helps prefill, which was already compute-saturated with a single long prompt and has comparatively little slack left to fill.</p>

${H.worked('worked latency — the two numbers a user feels', `
<p><b>TTFT is prefill.</b> A 2,000-token prompt on a 70B model costs about $2N\\cdot s = 2\\times70\\times10^9\\times2000 = 2.8\\times10^{14}$ FLOP. At 990 TFLOP/s peak and a realistic 40% utilisation: $2.8\\times10^{14} \\div (0.4\\times990\\times10^{12}) \\approx$ <b>0.7 s</b>. Double the prompt, double the wait — TTFT is linear in prompt length, which is the real cost of "just paste everything" (§5.2).</p>
<p><b>TPOT is decode</b> — 42 ms at batch 1 from §4.7's bandwidth argument, so ~24 tok/s, about 18 words per second: faster than reading. Batch 32 keeps TPOT roughly flat while multiplying throughput 32×, which is why a serving stack optimises <i>throughput per GPU</i> and a chat UX optimises TTFT, and why those two goals fight.</p>
<p><b>Speculative decoding, quantified.</b> With draft acceptance probability $\\alpha$ and $k$ proposed tokens per verification step, expected accepted tokens per big-model pass is $(1-\\alpha^{k+1})/(1-\\alpha)$. At $\\alpha=0.7$, $k=4$: $(1-0.7^5)/0.3 = (1-0.168)/0.3 \\approx$ <b>2.8 tokens per pass</b> — a ~2.8× speedup, with the output distribution provably unchanged. The failure mode: on unpredictable text $\\alpha$ collapses and you pay for the draft model with no benefit, so measure acceptance in production rather than assuming it. §4.22 derives this formula from first principles and works through the full mechanism.</p>`)}

<p><b>What you are looking at.</b> Time-to-first-token plotted against prompt length, a live figure sitting on the curve at your chosen prompt length, and a readout translating draft acceptance and draft length into the speculative-decoding speedup — all three computed from the formulas in the worked box above, not illustrated schematically.</p>
<p><b>What to do with it.</b> Drag prompt length from 100 tokens toward 128k and watch TTFT climb in a dead straight line — confirming the linear relationship the worked box asserts — while the TPOT readout barely moves, since decode's per-step cost is set by the model and cache size, not by how the prompt got there.</p>
<p><b>The thing genuinely worth noticing.</b> Push the batch-size control up and watch TPOT stay almost flat while total throughput climbs nearly proportionally — the exact signature of a memory-bound workload from §4.7's roofline. A serving team chasing "lower latency" and a serving team chasing "more tokens per GPU per dollar" are optimising against these two different curves, and a change that helps one (bigger batches) can be neutral or actively unhelpful for the other (TTFT for a single impatient user).</p>

${H.lab('latency', 'TTFT, TPOT and speculative decoding', 'All three calculations, live. Move prompt length, batch size and draft acceptance and watch which number the user actually feels change.')}

<h2><span class="sn">4.14.2</span> Serving systems — two named inefficiencies, two fixes</h2>
<p>Both of this subsection's central techniques exist because the obvious, naive way of running a batch of requests wastes a specific, nameable resource, and it is worth seeing the waste in numbers before seeing the fix, because the fix is otherwise just a clever-sounding trick rather than the answer to a concrete problem.</p>
<p><b>The inefficiency static batching lives with.</b> The obvious way to batch requests is the way a training batch works: collect a fixed group of sequences, run every one of them through decode step by step, and start the next batch only once every sequence in the current one has finished. But requests do not all want to finish at the same time — one user's response is four tokens, another's is a page of generated code — and a static batch is bound by whichever sequence in it happens to run longest. Every slot whose sequence finished early sits idle, fully reserved, computing nothing, until the slowest sequence in the batch is done. Take eight concurrent requests with output lengths 50, 100, 150, 200, 250, 300, 400 and 500 tokens — a spread realistic for a mixed workload of short replies and long generations. A static batch runs for as many steps as the longest one needs, 500, holding all eight slots busy for that entire span; the actual useful work done is the sum of the lengths, $50+100+\\cdots+500=1{,}950$ token-steps, against a paid-for capacity of $8\\times500=4{,}000$ slot-steps. Utilisation is $1{,}950/4{,}000\\approx$ <b>49%</b> — the GPU is idle, in this fully realistic and not even adversarial example, for roughly half of every batch's wall-clock time, and every one of those wasted slot-steps is memory and compute reserved, paid for, and doing nothing.</p>
${H.pitfall('This is not a rare pathology of one bad batch. The more output-length variance a workload has — exactly the situation an agent system or an open-ended chat product faces, where some turns are a one-word confirmation and others are a long tool-call trace — the worse static batching\'s utilisation gets, because the gap between the average length and the longest length only widens. A team that benchmarks a serving stack on requests of near-identical length and then deploys it against a real, length-varied traffic mix will see throughput collapse relative to the benchmark, and static batching is very often the reason.')}
<p><b>The fix: continuous batching.</b> Rather than committing to a fixed group of sequences for an entire batch's lifetime, continuous batching (also called in-flight or dynamic batching) makes the admit-and-retire decision at the level of a single decode step rather than a whole batch: the instant a sequence emits its end-of-sequence token, its slot is freed and immediately handed to the next request waiting in queue, without waiting for anything else in the current batch to finish. Applied to the same eight requests above, plus a steady stream of new requests arriving behind them, the GPU is never sitting on a reserved-but-idle slot the way static batching leaves it for roughly half its time — utilisation approaches the ceiling set by decode's actual memory-bandwidth bound (§4.7) rather than by whichever request happens to run longest. This is now the default scheduling discipline in every serving stack that matters, and the eight-request arithmetic above is exactly the number continuous batching is closing.</p>
${H.practice('Continuous batching is not entirely free of its own friction. Admitting a brand-new request mid-flight means running that request\'s prefill — a large, compute-bound matmul (§4.14.1) — interleaved with the memory-bound decode steps of every sequence already in flight, and naively inserting a full prefill can stall every other sequence\'s decode for the duration, spiking their TPOT. <b>Chunked prefill</b> is the standard fix: split a new request\'s prefill into smaller pieces and interleave those pieces with ongoing decode steps rather than running the whole prefill as one blocking unit, trading a slightly longer TTFT for the new request against smoother, more predictable TPOT for everyone already being served. This is the kind of second-order tuning knob that only becomes visible once the first-order fix — continuous batching itself — is already in place.')}
<p><b>The inefficiency contiguous KV allocation lives with.</b> Once a batch is scheduled sensibly, the next waste sits one layer down, in how memory for each sequence's KV cache (§4.7) gets reserved in the first place. The obvious approach reserves one contiguous block of memory per sequence, sized for the longest generation that sequence might conceivably produce — the model's maximum context length — before anything is known about how long it will actually run. Almost none of that reservation is used: a request that generates 80 tokens against a worst-case reservation sized for 4,096 leaves roughly 98% of its allotted block empty for the request's entire lifetime, and because the reservation is one contiguous span of memory, that empty space cannot be lent to a different sequence in the meantime even though it is doing nothing — exactly the way a contiguous file on disk cannot be split to fill gaps left by deleted files, and the term for the resulting waste is the same one used there: <b>fragmentation</b>.</p>
${H.analogy(`<p>The fix, <b>paged attention</b>, borrows its structure — and, deliberately, its name — from how an operating system manages a process's memory. A process is never handed one giant contiguous span of physical RAM sized for its theoretical maximum footprint; it is handed small, fixed-size pages on demand as it actually allocates memory, with a page table translating the process's own tidy, contiguous-looking view of its address space onto whatever physical pages happen to be free, wherever they happen to sit. Paged attention does exactly this for a sequence's KV cache: memory is carved into small fixed-size pages, handed to a sequence only as it actually generates tokens that need them, and a block table performs the same translation trick, so each sequence still <i>computes</i> attention over what looks like one contiguous cache while the physical pages backing it are scattered whichever way keeps memory tightly packed. The payoff is not only fragmentation disappearing — it is that once the cache is organised as freely reassignable pages, two sequences that happen to share an identical prefix (the same system prompt, the same first few conversation turns) can point their block tables at the very same physical pages for that shared portion, rather than each sequence duplicating it, which is the exact mechanism the next paragraph's prefix caching relies on.</p>`)}
<p>Between them, continuous batching keeps the GPU\'s compute close to fully occupied and paged attention keeps its memory close to fully packed, and the combined effect on capacity is multiplicative, not additive: a serving stack running both, versus one running neither, does not serve modestly more concurrent requests per GPU — published reports on real production traffic put the improvement at several times, not a fractional percentage, over the naive static-batch, contiguous-allocation baseline both problems above describe.</p>
<p><b>Prefix caching</b> reuses the already-computed KV cache of a shared prefix — a system prompt, a long set of tool definitions, the first turns of an ongoing conversation — across every request that shares it, skipping prefill for that shared portion entirely on every subsequent request. It is only cheap to implement <i>because</i> paged attention already made the cache a collection of independently addressable pages rather than one contiguous block per sequence — sharing pages between two sequences\' block tables is a pointer operation, where sharing a slice of a contiguous allocation would not be. On agent workloads with long, largely fixed instructions repeated on every call, this is often the single cheapest latency win available, precisely because it eliminates compute rather than merely speeding it up.</p>

<h2><span class="sn">4.14.3</span> Quantisation, by family</h2>
${H.table(['Method', 'What it is', 'Where it belongs'], [
      ['<b>GPTQ</b>', 'Post-training 4-bit weight-only, second-order error compensation', 'GPU serving; largely superseded by AWQ'],
      ['<b>AWQ</b>', 'Activation-aware: protects salient channels', 'The production pick on GPUs via vLLM'],
      ['<b>GGUF</b>', 'llama.cpp’s format and quant family', 'Local and workstation inference, not high-throughput serving'],
      ['<b>FP8</b>', 'Hardware-supported 8-bit on Hopper/Blackwell', 'Near-BF16 quality; the default for large-scale serving'],
      ['<b>KV-cache quantisation</b>', 'Quantise the cache, not the weights', 'Halves the 10 GB cache independently of what you did to the weights']
    ])}
<p>Every one of these methods is ultimately pulling the same lever §4.7's roofline made explicit: decode is memory-bandwidth-bound, so fewer bytes read per weight directly cuts the dominant cost — quantising a 70B model's weights from BF16 to INT4 shrinks the 42 ms per-step memory read (§4.7) roughly fourfold, which is a roughly fourfold ceiling lift on achievable decode throughput before anything else changes. <b>AWQ</b> earns its "activation-aware" name by first measuring which weight channels see the largest activation magnitudes in practice and protecting exactly those from aggressive rounding, rather than quantising every weight with equal, uniform coarseness — a small amount of extra care that meaningfully closes the quality gap to full precision. Keep <b>weight</b> quantisation conceptually distinct from <b>KV-cache</b> quantisation: they attack the two separate memory consumers §4.7 itemised (parameters and the cache) and are two separate decisions with two separate costs, not one setting. And keep <i>post-training</i> quantisation (GPTQ, AWQ, FP8 casting: minutes to hours, a small calibration set, where almost all production quantisation actually happens) distinct from <i>quantisation-aware training</i> (simulate low precision throughout training itself, so the weights adapt to it — better at aggressive sub-4-bit widths, far more expensive to run at all). If asked how to run a model in INT4 with minimal quality loss, the concrete answer is: AWQ, with a calibration set drawn from <i>your own</i> production distribution rather than a generic one, and an eval built specifically to catch the long-tail degradation quantisation damage tends to show up in first.</p>

<h2><span class="sn">4.14.4</span> What to turn on, in order</h2>
${H.steps([
      '<b>Prefix caching</b> — a shared system prompt is prefilled once instead of per request; on agent workloads with long fixed instructions this is often a 2–5× TTFT win for a configuration flag.',
      '<b>Continuous batching</b>, then <b>paged attention</b> — both defaults in a modern server.',
      '<b>FP8 or INT4 weights</b>, measured against your own eval rather than a leaderboard, because quantisation damage is task-dependent and shows up first on long-tail reasoning.',
      '<b>KV-cache quantisation</b> if context is long.',
      '<b>Speculative decoding</b> last: the most operationally fiddly and the most workload-sensitive — §4.22 covers the mechanism, the exact-distribution guarantee, and the drafting variants (Medusa, EAGLE, prompt-lookup decoding) in full.'
    ])}
<p>One more lever sits at training time rather than serving time, worth flagging here because it changes what "speculative decoding" costs later: <b>multi-token prediction</b> adds extra output heads during pretraining that predict not just the next token but $t+2$, $t+3$ and beyond. That denser per-position training signal measurably improves the base model on its own, and as a side effect it leaves the finished model with a built-in speculative drafter — no separate draft model or extra training run required at serving time — DeepSeek V3 is the documented production example of the trade paying off both ways at once.</p>
<p>That ordering is not arbitrary — it runs roughly from "free, no quality risk, large win" toward "genuine engineering effort, quality trade-offs to manage, workload-dependent payoff", and it is worth internalising the ordering itself rather than just the list: prefix caching and continuous batching cost nothing in quality because they change <i>when</i> and <i>how</i> computation happens, not what it computes, while quantisation and speculative decoding both trade something (precision, or draft-model overhead) for speed and need to be measured, not assumed.</p>

${H.probe([
      ['Why is decode memory-bound?', 'Every step streams all weights plus the whole KV cache from HBM for one token’s worth of arithmetic — the exact arithmetic-intensity argument §4.7\'s roofline model makes precise, distinct from prefill\'s compute-bound large batched matmul.'],
      ['Which quantisation in production?', 'AWQ INT4 or FP8 on GPU; GGUF only for local. Quantise the KV cache separately from the weights — they are two different memory consumers with two different costs.'],
      ['Size the cache for 70B, 4k, batch 8.', '10 GB — the exact §4.7 calculation, and be able to show the multiplication live rather than quote the answer.']
    ])}`,
    labs: {
      latency: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'model size (B)', min: 1, max: 700, step: 1, value: 70, fmt: v => v + 'B' },
          { k: 'prompt', label: 'prompt tokens', min: 100, max: 128000, step: 100, value: 2000, fmt: v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v },
          { k: 'batch', label: 'batch size', min: 1, max: 64, step: 1, value: 1, fmt: v => v },
          { k: 'alpha', label: 'speculative acceptance α', min: 0, max: .95, step: .01, value: .7, fmt: v => v.toFixed(2) },
          { k: 'k', label: 'draft tokens k', min: 1, max: 8, step: 1, value: 4, fmt: v => v },
          { k: 'util', label: 'prefill utilisation', min: .1, max: .8, step: .05, value: .4, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ttft', label: 'time to first token', cls: 'key' }, { k: 'tpot', label: 'time per output token' },
          { k: 'tps', label: 'tokens/s (this stream)' }, { k: 'spec', label: 'tokens per verification pass', cls: 'good' }, { k: 'speed', label: 'speculative speedup' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const N = st.N * 1e9, peak = 990e12, bw = 3.35e12;
            const ttft = (2 * N * st.prompt) / (st.util * peak);
            const tpot = (N * 2) / bw;
            const accepted = (1 - Math.pow(st.alpha, st.k + 1)) / (1 - st.alpha);
            const P = Viz.plot(ctx, w, h, { xd: [100, 128000], yd: [0, Math.max(2, (2 * N * 128000) / (st.util * peak)) * 1.05] })
              .frame({ xlabel: 'prompt tokens', ylabel: 'time to first token (s)', xfmt: v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0) });
            P.clip(() => {
              P.fn(p => (2 * N * p) / (st.util * peak), { color: T.blue, width: 2.6 });
              P.vline(st.prompt, { color: T.red, dash: [4, 4] });
              P.dots([[st.prompt, ttft]], { r: 5, color: T.red, stroke: true });
              P.hline(1, { color: T.amber, dash: [3, 3], label: '1 second — the perceptual threshold' });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText('TTFT is linear in prompt length — the real cost of "paste everything"', w - 16, 8);
            out({
              ttft: ttft < 1 ? (ttft * 1000).toFixed(0) + ' ms' : ttft.toFixed(2) + ' s',
              tpot: (tpot * 1000).toFixed(1) + ' ms',
              tps: (1 / tpot).toFixed(0),
              spec: accepted.toFixed(2),
              speed: '×' + accepted.toFixed(2)
            });
          }
        });
        Viz.note(host, 'At α = 0.7 and k = 4 the expected accepted tokens per pass is 2.8 — matching the worked box. Drag α down to 0.3 (unpredictable text) and the speedup nearly vanishes while you still pay for the draft model: measure acceptance, never assume it.');
      }
    },
    quiz: [
      {
        q: 'Time to first token is determined by…',
        options: ['decode bandwidth', 'prefill compute, and it is linear in prompt length', 'the KV cache size', 'the tokenizer'],
        answer: 1,
        why: 'Prefill reads the entire prompt in one large batched matmul, costing roughly 2Ns FLOPs for N parameters and s prompt tokens exactly as §4.14.1 sets out, so at a fixed achieved utilisation TTFT is linear in prompt length — double the prompt and you double the wait, the real, measurable cost behind "just paste everything into the context". "Decode bandwidth" is the tempting wrong answer because it correctly names what governs the other half of a request\'s latency — time per output token — but prefill and decode are opposite sides of §4.7\'s roofline distinction, one compute-bound and one memory-bandwidth-bound, and conflating them is flagged explicitly as the single most common mistake in serving discussions. "The KV cache size" affects memory footprint and decode bandwidth, not how long a fresh prompt takes to process for the first time, and "the tokenizer" has no meaningful bearing on either number at this scale. The general principle is that a user experiences exactly two numbers, TTFT and TPOT, governed by two structurally different phases of the same request, and optimising one while a system is bottlenecked on the other is a real, common failure.'
      },
      {
        q: 'Speculative decoding with α = 0.7 and k = 4 yields roughly how many tokens per big-model pass?',
        options: ['1.4', '2.8', '4.0', '5.0'],
        answer: 1,
        why: 'Sum the geometric series $\\sum_{i=0}^k \\alpha^i=(1-\\alpha^{k+1})/(1-\\alpha)$ from §4.22\'s derivation: at $\\alpha=0.7,k=4$ that is $(1-0.7^5)/0.3=(1-0.168)/0.3\\approx2.8$ expected accepted tokens per verification pass, counting the free bonus token you get when every drafted token is accepted. "4.0" is the tempting wrong answer for anyone assuming every one of the $k=4$ drafted tokens plus the bonus gets accepted every time — the ceiling if $\\alpha$ were 1 — but acceptance is a probability, not a guarantee, so the expectation sits below that ceiling once $\\alpha<1$ discounts each successive token geometrically. "1.4" undershoots the calculation, and "5.0" overshoots past even the best-case ceiling of $k+1=5$. The general principle, worth memorising because it recurs across this course\'s worked numbers, is that speculative decoding\'s payoff is a geometric series in the acceptance rate, so a modest drop in $\\alpha$ — unpredictable text, a poorly aligned draft model — costs speedup fast, which is why production systems measure acceptance rather than assume it.'
      },
      {
        q: 'The cheapest serving win to turn on first is usually…',
        options: ['speculative decoding', 'prefix caching of a shared system prompt', 'INT4 quantisation', 'a bigger batch'],
        answer: 1,
        why: 'Prefix caching reuses the already-computed KV cache of a shared system prompt or long fixed instructions across every request that shares it, skipping prefill for that portion entirely rather than merely speeding it up, and §4.14.4\'s ordered list puts it first precisely because it is a configuration flag with no quality risk and, on agent workloads, frequently a 2–5× TTFT win. "Speculative decoding" is the tempting wrong answer because it is the most conceptually interesting technique in the section and genuinely does help latency, but §4.14.4 places it deliberately last: it is the most operationally fiddly and workload-sensitive, and can actively hurt throughput at high batch size, the opposite profile of a first thing to turn on. "INT4 quantisation" trades away some quality for speed and needs to be measured against your own eval, and "a bigger batch" helps decode throughput while doing essentially nothing for TTFT, since a single long prompt was already compute-saturated before batching entered the picture. The general principle behind the ordering is to move from "free, no quality risk, large win" toward "genuine engineering effort, quality trade-offs to manage, workload-dependent payoff", and reversing that order wastes effort on fiddly wins before claiming the free ones.'
      }
    ],
    cards: [
      { q: 'Prefill vs decode', a: 'Prefill: parallel, compute-bound, sets TTFT. Decode: one token at a time, bandwidth-bound, sets tokens/s. Batching helps decode.' },
      { q: 'Speculative decoding formula', a: 'Expected accepted tokens $=(1-\\alpha^{k+1})/(1-\\alpha)$; α=0.7, k=4 → 2.8.' },
      { q: 'Serving order', a: 'Prefix caching → continuous batching + paged attention → FP8/INT4 weights → KV quantisation → speculative decoding last.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.15 */
  ML.section({
    id: 'decoding', track: 'llm', num: '4.15',
    title: 'Decoding strategies, test-time compute, prompting',
    lede: 'Same distribution, many ways through it — and the axis of scaling that does not require training anything.',
    html: `
<p>Everything from §4.9 through §4.12 shapes one thing: the probability distribution the model's final softmax produces over the vocabulary at each position (§4.5). None of it says how to turn that distribution into an actual chosen token. That translation step — a <b>decoding policy</b> — is a free choice made entirely at inference time, on a model that has already finished training, and different policies pull remarkably different behaviour out of the identical underlying distribution.</p>

<h2><span class="sn">4.15.1</span> The policies</h2>
${H.table(['Strategy', 'Rule', 'Use for'], [
      ['<b>Greedy</b>', 'take the argmax', 'Extraction, classification, anything deterministic'],
      ['<b>Beam search</b>', 'keep the top-b partial sequences', 'Translation; produces bland text in open generation'],
      ['<b>Temperature</b>', 'divide logits by T before softmax', 'T>1 flattens, T<1 sharpens — reshapes before any policy chooses'],
      ['<b>Top-k</b>', 'truncate to the k most likely', 'Simple, but a fixed k is wrong when the distribution is peaked'],
      ['<b>Top-p (nucleus)</b>', 'smallest set whose mass exceeds p', 'Adapts to the shape of the distribution — the common default'],
      ['<b>Min-p</b>', 'threshold relative to the mode', 'Robust at high temperature'],
      ['<b>Repetition / presence penalties</b>', 'down-weight seen tokens', 'Suppressing loops']
    ])}
<p><b>Greedy</b> decoding — always take the single most likely next token — sounds like the obviously correct choice, and it is deterministic and reproducible, which is genuinely valuable for extraction or classification tasks with one right answer. But applied to open-ended generation it has a specific, well-known failure worth deriving rather than just naming: once the model has just written the same short phrase twice in a row, the strongest signal in its own recent context is "this exact pattern belongs here" — the model's attention (§4.3) is quite literally pointing back at the previous occurrence of that phrase, because a repeated local pattern is exactly the kind of structure attention is built to pick up on. That makes "repeat the phrase a third time" the argmax at the very next step too, and a fourth time after that: there is no term anywhere in the model's forward pass that penalises having already said something, so once greedy decoding enters this loop, every one of its own most-confident, individually reasonable next-token choices keeps it there. This is not a rare edge case; it is a direct, mechanical consequence of always taking the argmax of a distribution that a model's own recent output has just made confidently unanimous. <b>Beam search</b>, which tracks several of the most promising partial sequences at once rather than committing token by token, avoids some short-sighted greedy mistakes but has its own well-documented failure in open generation: because it is explicitly searching for the single highest joint-probability sequence, it systematically favours safe, generic, low-surprise continuations over the more varied, specific phrasing an actual sample from the distribution would produce — the mode of a distribution is not a typical sample from it, and text generated by hunting for the mode reads as noticeably duller than text generated by sampling.</p>
<p>Sampling, then, is often what open-ended generation actually wants — drawing a token according to its true predicted probability, exactly like the sampling procedures in §1.13 — but sampling from the raw distribution has its own problem: a language model's distribution nearly always has a very long tail of low-probability tokens, and sampling from that tail even occasionally is enough to derail a generation into incoherent text. <b>Temperature</b> reshapes the distribution before any of this happens — divide every logit by $T$ before the softmax (§4.3's mechanism, applied here rather than to attention scores): $T>1$ flattens the distribution toward uniform, $T<1$ sharpens it toward the mode, and $T=1$ leaves it exactly as the model produced it. Temperature does not remove the tail, it only reweights it, which is why it is nearly always combined with a truncation rule that removes the tail outright. <b>Top-k</b> keeps only the $k$ most likely tokens and renormalises among them — simple, but a fixed $k$ is the wrong tool whenever the shape of the distribution varies from position to position, which it constantly does: at a position where the model is highly confident, $k$ candidates may include tokens with vanishing probability that should never be sampled; at a position of genuine ambiguity, the same $k$ may exclude a perfectly reasonable continuation. <b>Top-p (nucleus) sampling</b>, introduced specifically to fix this, instead keeps the <i>smallest</i> set of tokens whose cumulative probability exceeds $p$ — a set that automatically shrinks to a handful of tokens when the model is confident and expands when it is genuinely uncertain, adapting to the distribution's actual shape rather than assuming a fixed count is ever right.</p>
${H.deriv('what temperature, top-k and top-p actually do to five logits', [
      ['$z = [2.0,\\ 1.0,\\ 0.5,\\ 0.1,\\ {-1.0}]$ for tokens A, B, C, D, E', 'A concrete, small logit vector — small enough to trace by hand, shaped the way a real distribution over plausible continuations is: one clear leader, a long thinning tail.'],
      ['$T{=}1$: softmax(z) $\\approx$ A .559, B .205, C .125, D .084, E .028', 'The untouched distribution — divide by $T{=}1$ changes nothing, so this is just $\\mathrm{softmax}(z)$, exactly §4.3\'s formula applied to next-token logits instead of attention scores.'],
      ['$T{=}0.5$: divide $z$ by 0.5 first $\\Rightarrow$ A .826, B .112, C .041, D .018, E .002', 'Halving $T$ doubles every logit before the exponential, and because softmax is exponential in its input, doubling the gaps between logits does not just shift probability toward A — it shifts it sharply, roughly squaring the odds ratio between the top two.'],
      ['$T{=}2$: A .372, B .226, C .176, D .144, E .083', 'Doubling $T$ halves every logit\'s spread instead, flattening the distribution back toward uniform (which is 0.20 each for five tokens) without ever quite reaching it.'],
      ['top-$k{=}2$ at $T{=}1$: keep {A, B}, renormalise $\\Rightarrow$ A .732, B .268', 'Truncate to the two largest raw probabilities, then rescale so the kept set sums to 1 again — C, D and E are set to exactly zero regardless of how close C\'s .125 was to being worth keeping.'],
      ['top-$p{=}0.8$ at $T{=}1$: accumulate sorted mass .559, .764, .889 $\\Rightarrow$ stop at {A, B, C}', 'Add probabilities in decreasing order until the running total first clears $p{=}0.8$; that happens only once C is included (.889 ≥ .8), so nucleus sampling keeps three tokens here where top-$k{=}2$ kept only two — the direct, numeric demonstration that a fixed $k$ and an adaptive $p$ genuinely disagree, not just in principle.']
    ], 'Renormalising the kept set after top-$p$ gives A .629, B .231, C .141 — visibly flatter than the untruncated $T{=}1$ distribution\'s A .559, because discarding D and E\'s combined 11% of the mass and redistributing it among the survivors necessarily raises each of their shares. Run the same five logits through top-$k{=}1$ and every method above collapses onto greedy: keep the single largest entry, renormalise, and the "distribution" is a certainty on A. That is the sense in which greedy is not a separate algorithm so much as the $k{=}1$ (or $p\\to0$) limit of everything else in this table.')}
${H.history(`<p>Holtzman et al.'s 2019 paper, memorably titled <i>The Curious Case of Neural Text Degeneration</i>, is the source of nucleus sampling and the paper that made the beam-search-produces-bland-text finding precise and widely known: they showed that maximisation-based decoding (greedy, beam search) reliably produces text that is measurably less diverse and more repetitive than human text, by exactly the mode-versus-sample mechanism described above, and that naive random sampling from the full distribution reliably produces text that degenerates into incoherence by wandering into the long tail. Top-p sampling was their proposed middle path, and it has remained the default in most production systems since.</p>`)}
<p><b>Min-p</b> sets its threshold relative to the single most likely token's probability rather than to a cumulative mass, which makes it more robust specifically at high temperature — where top-p's cumulative-mass threshold can admit a surprisingly large or small set depending on how flattened the distribution has become, min-p's relative threshold tracks the model's actual confidence more directly. <b>Repetition and presence penalties</b> take a more direct approach again: down-weight the logits of tokens (or n-grams) already seen in the generated text so far, a blunt but effective way of suppressing the specific loop failure greedy decoding is prone to, at the cost of occasionally penalising a word that genuinely needed repeating.</p>
${H.pitfall('The order these operations happen in is not a free choice, and getting it backwards silently changes the output. The standard pipeline is logits $\\to$ divide by $T$ $\\to$ truncate (top-$k$ then top-$p$, or vice versa) $\\to$ renormalise $\\to$ sample — temperature reshapes the full distribution <i>first</i>, and truncation then acts on the reshaped version. Truncate before applying temperature instead, and the kept set is frozen based on the untouched distribution\'s shape while temperature only gets to redistribute mass among whatever survived — a materially different, and usually worse, result, because a token temperature would have promoted into relevance at low T never gets the chance if it was already cut. The five-logit example above is small enough to check this by hand: truncate to top-$p{=}0.8$ on the $T{=}2$ distribution instead of $T{=}1$\'s, and a different, larger set of tokens survives before temperature ever gets applied — the two orderings are not interchangeable, and most inference libraries apply temperature strictly before any truncation for exactly this reason.')}

<p><b>What you are looking at.</b> A real character-level language model — trained live in your browser on the short text in the box, using an actual n-gram frequency count rather than a stand-in — generating text under whichever decoding policy you select. The bar chart is the model's genuine predicted distribution over the next character at the current position, before and after your chosen policy filters it.</p>
<p><b>What to do with it.</b> Set temperature to 0.2 and generate: the text becomes safe and repetitive, greedy decoding's failure mode in miniature. Push temperature to 2.0 instead and watch it degenerate into near-noise, exactly the failure Holtzman et al. documented. Then bring temperature back toward 1 and turn on top-p at 0.9.</p>
<p><b>The thing genuinely worth noticing.</b> With top-p engaged, the candidate-count readout shrinks and grows from character to character depending on how peaked the model's actual prediction is at that position — narrow right after a highly predictable letter, wider at a genuine branch point — which is precisely nucleus sampling's adaptivity made visible one token at a time, on a model small enough that you can actually watch it happen.</p>

${H.lab('sample', 'A real language model, sampled live', 'This trains a character-level n-gram model in your browser on the text in the box, then generates from it under the decoding policy you choose. Temperature, top-k and top-p are doing exactly what they do in a frontier model — the model is tiny, the mechanism is identical.')}

<h2><span class="sn">4.15.2</span> Constrained decoding</h2>
<p>None of the policies above touch <i>which</i> tokens are even eligible to be sampled — they only reweight or truncate a fixed candidate set. Constrained decoding is a different kind of intervention: mask the logits of any token that cannot possibly continue a valid string under some external grammar or JSON Schema, so the output is <b>valid by construction</b> rather than merely likely to be valid. This is the correct answer to "how do you guarantee parseable output" — not "ask nicely and retry" — and §4.21 covers the full mechanism, the automaton compilation that makes it fast, and why it is what makes reliable tool calling possible (§5.3) in the depth the idea deserves.</p>

<h2><span class="sn">4.15.3</span> Test-time compute</h2>
<p>Every technique above operates on a single forward pass's worth of output. <b>Test-time compute</b> is a genuinely different axis of scaling: rather than making the model bigger (§4.10) or training it further (§4.12), spend more computation <i>at inference</i> on a single problem — longer chains of thought, sample-and-vote across several attempts (self-consistency), best-of-$n$ against a verifier, or search over intermediate reasoning steps — and accuracy measurably rises without changing a single weight. §4.20 derives exactly why this works (a transformer does a fixed amount of computation per generated token, so writing more tokens buys more sequential computation on the same problem) and precisely when self-consistency helps versus when correlated errors make it useless; it is also where the connection to §4.12's RLVR is made in full — training a model to <i>use</i> a long chain of thought productively is what turns raw extra compute into reliably better answers rather than merely longer ones.</p>

<p><b>What you are looking at.</b> Two accuracy curves against samples drawn per problem, both computed from an honest binomial simulation rather than illustrated: majority voting across $k$ independent attempts, and best-of-$n$ selection using a fallible verifier, for whatever per-sample accuracy you set.</p>
<p><b>What to do with it.</b> Set a modest per-sample accuracy and watch both curves climb well above it as sample count rises — neither method requires the model to be reliable on a single attempt, only for its errors to be somewhat independent from attempt to attempt.</p>
<p><b>The thing genuinely worth noticing.</b> Both curves eventually flatten rather than climbing to 100%, and where they flatten is set by exactly the failure §4.20 names explicitly: correlated errors. §4.20's lab lets you dial that correlation directly and watch the ceiling drop as it rises — this lab and that one are measuring the same underlying limit from two adjacent angles.</p>

${H.lab('ttc', 'Test-time compute: majority vote and best-of-n', 'Simulated but honest: with per-sample accuracy p and independent errors, watch how majority voting and verifier-checked best-of-n scale with n — and where each stops helping.')}

<h2><span class="sn">4.15.4</span> Prompting that generalises</h2>
<p>Prompting is decoding policy's quieter cousin: it does not change how a token is sampled, but it changes what distribution the model produces in the first place, by conditioning on carefully chosen context — and it deserves to be treated with the same rigour as any other lever in this section rather than as folklore. Stating the task and the desired output format explicitly removes an entire class of failure where the model has to guess your intent from an underspecified request. Providing two or three well-chosen exemplars exploits <b>in-context learning</b>: the model conditions its next-token distribution on the pattern the examples establish, adjusting its behaviour without a single gradient update ever being taken — a genuinely different mechanism from fine-tuning, operating entirely within one forward pass. Placing the most important instructions at the very start and again at the very end of the context is a direct, practical response to the "lost in the middle" effect §4.4 already documented: information in the middle of a long context is measurably less reliably attended to than information at either end, so instructions parked in the middle of a long prompt are the instructions most likely to be quietly under-weighted. Explicit step-by-step reasoning is the one item on this list that needs a caveat rather than a blanket recommendation: on a non-reasoning model it reliably helps multi-step arithmetic and logic by giving the model the serial computation §4.20 explains chain of thought provides, and it can actively hurt on simple classification by inviting the model to talk itself out of an answer it already had right; on a model already trained for long-form reasoning (§4.12's RLVR-trained models), asking for explicit steps is frequently redundant with reasoning the model already performs internally before it ever starts writing the visible answer.</p>

${H.probe([
      ['Top-k or top-p?', 'Top-p adapts to the distribution’s shape — the set it keeps automatically shrinks where the model is confident and expands where it is not — while a fixed k is wrong whenever the distribution is peaked or flat, which it constantly is from position to position.'],
      ['How do you guarantee valid JSON?', 'Constrained decoding against a schema — mask the logits so invalid tokens cannot be sampled at any temperature, rather than hoping a well-phrased prompt produces valid output (§4.21).'],
      ['What is test-time compute?', 'Trading inference compute, spent at generation time rather than training time, for accuracy: longer CoT, self-consistency voting, best-of-n against a verifier, search over reasoning steps (§4.20) — a scaling axis fully independent of model size (§4.10).']
    ])}`,
    labs: {
      sample: function (host) {
        const seed = 'the quick brown fox jumps over the lazy dog. the model predicts the next token from the previous tokens. attention is all you need for sequence modelling. the cat sat on the mat while the dog barked at the moon. machine learning models learn patterns from data and generalise to new data. ';
        let lm = Num.charLM(seed.repeat(6), 4);
        let outText = 'the ';
        const st = Viz.controls(host, [
          { k: 'temp', label: 'temperature T', min: .1, max: 2.5, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'topk', label: 'top-k (0 = off)', min: 0, max: 12, step: 1, value: 0, fmt: v => v || 'off' },
          { k: 'topp', label: 'top-p (1 = off)', min: .1, max: 1, step: .05, value: 1, fmt: v => v === 1 ? 'off' : v.toFixed(2) },
          { k: 'greedy', label: 'greedy (argmax)', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'cands', label: 'candidates after filtering', cls: 'key' }, { k: 'ent', label: 'entropy of the choice' },
          { k: 'top', label: 'most likely next char' }, { k: 'p', label: 'its probability' }
        ]);
        const textBox = ML.el('div', {
          style: 'font-family:var(--mono);font-size:13px;line-height:1.6;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-height:86px;margin:12px 0;white-space:pre-wrap'
        });
        host.appendChild(textBox);
        function nextDist() {
          const ctx2 = outText.slice(-lm.order);
          let d = lm.dist(ctx2);
          let logits = d.map(x => Math.log(Math.max(1e-9, x.p)) / st.temp);
          let probs = Num.softmax(logits);
          let items = d.map((x, i) => ({ ch: x.ch, p: probs[i] })).sort((a, b) => b.p - a.p);
          if (st.topk > 0) items = items.slice(0, st.topk);
          if (st.topp < 1) {
            const keep = []; let acc = 0;
            for (const it of items) { keep.push(it); acc += it.p; if (acc >= st.topp) break; }
            items = keep;
          }
          const z = items.reduce((a, b) => a + b.p, 0) || 1;
          items = items.map(it => ({ ch: it.ch, p: it.p / z }));
          return items;
        }
        function step(n) {
          for (let i = 0; i < n; i++) {
            const items = nextDist();
            let pick;
            if (st.greedy) pick = items[0];
            else {
              let u = Math.random(), acc = 0;
              pick = items[items.length - 1];
              for (const it of items) { acc += it.p; if (u <= acc) { pick = it; break; } }
            }
            outText += pick.ch;
            if (outText.length > 420) outText = outText.slice(-420);
          }
        }
        const S = Viz.surface(host, {
          height: 220,
          draw: function (ctx, w, h, T) {
            const items = nextDist().slice(0, 14);
            const P = Viz.plot(ctx, w, h, { xd: [-.5, Math.max(1, items.length) - .5], yd: [0, 1], pad: { l: 40, r: 14, t: 14, b: 40 } })
              .frame({ xticks: [], ylabel: 'probability' });
            P.clip(() => items.forEach((it, i) => {
              const x0 = P.x(i - .38), x1 = P.x(i + .38);
              ctx.fillStyle = i === 0 ? T.blue : T.faint;
              ctx.fillRect(x0, P.y(it.p), x1 - x0, P.y(0) - P.y(it.p));
              ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText(it.ch === ' ' ? '␣' : it.ch === '\n' ? '⏎' : it.ch, P.x(i), P.y(0) + 6);
            }));
            textBox.textContent = outText;
            out({
              cands: items.length, ent: Num.entropy(items.map(i => i.p)).toFixed(2),
              top: items[0] ? (items[0].ch === ' ' ? '␣' : items[0].ch) : '—',
              p: items[0] ? items[0].p.toFixed(3) : '—'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Generate 60 characters', primary: true, on: () => { step(60); S.redraw(); } },
          { label: 'Generate 1', on: () => { step(1); S.redraw(); } },
          { label: 'Reset', on: () => { outText = 'the '; S.redraw(); } }
        ]);
        Viz.note(host, 'Set temperature to 0.2 and generate: the text becomes repetitive and safe. Set it to 2.0 and it becomes noise. Turn on top-p 0.9 at high temperature and the tail is cut while the head keeps its diversity — which is exactly why nucleus sampling became the default.');
      },

      ttc: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'per-sample accuracy', min: .05, max: .95, step: .01, value: .45, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'verifier', label: 'verifier accuracy (best-of-n)', min: .5, max: 1, step: .01, value: .95, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'corr', label: 'error correlation between samples', min: 0, max: .9, step: .05, value: .2, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'v1', label: 'single sample', cls: 'key' }, { k: 'maj', label: 'majority vote @ 16' },
          { k: 'bon', label: 'best-of-16 with verifier', cls: 'good' }, { k: 'ceil', label: 'ceiling from correlation' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const p = st.p, c = st.corr;
            // effective independent samples shrinks with correlation
            const eff = n => 1 + (n - 1) * (1 - c);
            const majority = n => {
              const m = eff(n);
              let acc = 0;
              for (let k = Math.floor(m / 2) + 1; k <= Math.ceil(m); k++) acc += Num.binomPmf(Math.round(k), Math.round(m), p);
              return Math.min(1, Math.max(p, acc));
            };
            const bestOf = n => {
              const m = eff(n);
              const anyCorrect = 1 - Math.pow(1 - p, m);
              return anyCorrect * st.verifier + (1 - anyCorrect) * (1 - st.verifier) * .2;
            };
            const P = Viz.plot(ctx, w, h, { xd: [1, 64], yd: [0, 1] })
              .frame({ xlabel: 'samples drawn (test-time compute)', ylabel: 'task accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.fn(n => p, { color: T.faint, width: 1.6, dash: [5, 4] });
              P.fn(majority, { color: T.blue, width: 2.6, n: 120 });
              P.fn(bestOf, { color: T.green, width: 2.6, n: 120 });
            });
            out({
              v1: (p * 100).toFixed(1) + '%', maj: (majority(16) * 100).toFixed(1) + '%',
              bon: (bestOf(16) * 100).toFixed(1) + '%',
              ceil: c > .5 ? 'severe — samples repeat the same error' : c > .2 ? 'moderate' : 'mild'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().faint, t: 'single sample' }, { c: Viz.theme().blue, t: 'majority vote (self-consistency)' }, { c: Viz.theme().green, t: 'best-of-n against a verifier' }]);
        Viz.note(host, 'Raise the error correlation and both curves flatten: if every sample makes the same mistake, drawing more of them buys nothing. That is the real limit of test-time compute, and the reason temperature and prompt diversity matter when you use it.');
      }
    },
    quiz: [
      {
        q: 'Top-p (nucleus) sampling is preferred to top-k because…',
        options: ['it is faster', 'it adapts the candidate set to how peaked the distribution is', 'it always produces longer text', 'it removes the need for temperature'],
        answer: 1,
        why: 'Top-p keeps the smallest set of tokens whose cumulative probability exceeds p, so the candidate set automatically shrinks to a handful of tokens exactly where the model is confident and expands where it is genuinely uncertain, adapting to the distribution\'s actual shape at each position — precisely what Holtzman et al.\'s 2019 paper proposed nucleus sampling to fix. "It is faster" is the tempting wrong answer because both top-k and top-p involve comparably cheap sorting-and-truncation operations, so there is no meaningful speed difference between them — the actual advantage is about output quality, not computational cost. A fixed k, by contrast, is wrong whenever the shape of the distribution varies from position to position, which it constantly does: at a confident position it can admit tokens with vanishing probability, and at an ambiguous one it can exclude a reasonable continuation. The general principle, §4.15.1\'s whole table, is that a decoding policy should respond to the model\'s actual predicted distribution rather than impose a fixed rule that works well only for distributions of one particular shape.'
      },
      {
        q: 'The correct way to guarantee schema-valid JSON output is…',
        options: ['prompt engineering and retries', 'constrained decoding that masks invalid tokens', 'temperature 0', 'a larger model'],
        answer: 1,
        why: 'Constrained decoding masks the logits of every token that cannot continue a valid string under the schema\'s compiled automaton, so invalid output becomes impossible to sample at any temperature rather than merely unlikely — validity by construction, which §4.21 covers in full including why it is often faster than unconstrained decoding via jump-ahead. "Prompt engineering and retries" is the tempting wrong answer because it is genuinely the default first attempt everyone reaches for, and a well-phrased prompt does raise the fraction of valid outputs — but that is a probabilistic improvement with an unbounded tail, never a guarantee, and the tail gets longest exactly when the model is already struggling, the worst time to be paying for retries. "Temperature 0" reduces randomness but does nothing to stop the model\'s single most likely continuation from being invalid in the first place, and "a larger model" may lower the error rate without ever eliminating it. The general principle is that guaranteeing a property of generated text is a decoding-time intervention on which tokens can be sampled, not something achievable purely through better instructions.'
      }
    ],
    cards: [
      { q: 'Temperature vs top-k vs top-p', a: 'Temperature reshapes the distribution; top-k truncates to a fixed count; top-p truncates to a mass, adapting to the shape.' },
      { q: 'Constrained decoding', a: 'Mask logits against a grammar/schema so output is valid by construction — the answer to "guarantee parseable output".' },
      { q: 'Test-time compute', a: 'Longer CoT, self-consistency voting, best-of-n with a verifier, search over steps. Limited by error correlation between samples.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.16 */
  ML.section({
    id: 'llm-eval', track: 'llm', num: '4.16',
    title: 'Evaluation, hallucination, multimodal',
    lede: 'Contamination first, then judges, then the eval you actually build yourself.',
    html: `
<p>Knowing whether a new checkpoint is actually better than the last one is harder for a language model than for almost any other kind of model in this course. A classifier's accuracy is a single, unambiguous number against a fixed label set; an LLM's output is unbounded free text, "better" is frequently a matter of judgment rather than fact, and — the problem unique to this generation of models — the very benchmarks used to measure ability have a way of ending up inside the training data they are meant to test. This section works through that chain of problems in the order they actually bite: whether your score means anything at all, how to get a scalable judgment when the honest answer requires human-level taste, why models state falsehoods fluently and what can be done about it, and how to build an evaluation set that survives contact with a real product.</p>

<h2><span class="sn">4.16.1</span> Contamination first</h2>
<p>Public benchmarks do not stay private. A popular benchmark's questions and answers get discussed on forums, mirrored in GitHub repositories, explained in blog posts, and quoted in papers — and all of that text is exactly the kind of content a web-scale pretraining crawl (§4.9) is built to hoover up, with no one on either side deliberately trying to cheat. Once a benchmark's actual answers are present, even partially or paraphrased, somewhere in a model's training data, a high score on it stops being clean evidence of capability and starts being partly a measurement of how thoroughly that specific benchmark happened to leak — precisely analogous to a student who scores well because they saw a copy of the exam beforehand, not because they understand the material better than a student who did not. <mark>A headline benchmark score is evidence about the benchmark's age as much as the model's ability</mark>: an older, more widely discussed benchmark is more likely to be contaminated in any given model's training data than a newly released one, independent of anything about the models being compared. The practical response is to trust a private, held-out evaluation set built from your own distribution — one that could not possibly have leaked into any public model's training data, because it was never public — and, wherever the training corpus is inspectable at all, to check direct n-gram overlap between your evaluation questions and the training data as a first, cheap sanity check.</p>

<h2><span class="sn">4.16.2</span> LLM-as-a-judge</h2>
<p>Human evaluation of open-ended output is the gold standard and does not scale: it is slow, expensive, and becomes the bottleneck the moment you want to compare more than a handful of model variants on more than a small sample. <b>LLM-as-a-judge</b> — using a strong model to score or compare outputs instead of a human — is cheap enough to run at the scale modern iteration actually needs, and correlates decently with human preference on many tasks, which is why it has become the default first-pass evaluation tool despite carrying real, well-documented biases that a naive user of it will not anticipate. <b>Position bias</b> — favouring whichever answer happens to be shown first in the prompt — persists even when the judge is told explicitly to ignore position, simply because "the first thing mentioned" correlates with primacy in the training data the judge itself learned from. <b>Verbosity bias</b> — favouring the longer of two answers, independent of whether the extra length adds anything — is the judge-side mirror of exactly the length-exploitation failure §4.12.3 already documented for DPO reward models: any evaluator, human or model, that has ever seen length correlate with quality in its own training will tend to reproduce that correlation when asked to score for quality directly. And <b>self-preference</b> — a judge scoring outputs from its own model family more favourably — reflects that a judge's sense of "good" is itself shaped by its own training distribution, so text that resembles what it would have written naturally reads as more fluent to it.</p>
<p>The mitigations, in order of how much each one buys: <mark>score both orders and average — the mandatory position swap</mark>, which directly cancels position bias rather than merely reducing it, at the cost of doubling the number of judge calls; control for length explicitly, either by instructing the judge to ignore it or by normalising scores against response length after the fact; prefer pairwise comparison ("which of these two is better") to pointwise scoring ("rate this 1 to 10"), because comparative judgments are consistently more reliable than absolute ones, for models and humans alike; give the judge an explicit, written rubric rather than an open-ended "rate the quality" instruction, which narrows what "quality" means enough to make repeated judgments consistent; and keep a human review loop on a sample of judged outputs, specifically to catch whichever bias the mitigations above did not.</p>
<p>None of this tells you <i>how much</i> to trust a judge's agreement with human raters, though, and that requires a specific correction rather than reading the raw agreement rate at face value. Two raters who both tend to prefer the same option most of the time will agree with each other often purely by chance, entirely independent of whether either of them is actually tracking quality — so a raw "82% agreement" figure conflates real signal with base-rate luck, and needs to be corrected for how much agreement chance alone would produce.</p>
${H.worked('worked number — is your judge trustworthy?', `
<p>You cannot take a judge's agreement rate at face value, because two raters who both prefer A most of the time agree often by chance. <b>Cohen's κ</b> corrects for that: $\\kappa = (p_o - p_e)/(1 - p_e)$.</p>
<p>On 100 pairwise comparisons the judge and a human agree 82 times, so $p_o = 0.82$. Both pick A about 70% of the time and B 30%, so chance agreement is $p_e = 0.7^2 + 0.3^2 = 0.58$. Then</p>
$$\\kappa = \\frac{0.82-0.58}{1-0.58} = \\frac{0.24}{0.42} = 0.57$$
<p>"82% agreement" sounds strong; κ = 0.57 is <i>moderate</i> — usable for ranking two systems in aggregate, not for adjudicating individual cases. And human–human κ on the same task is the ceiling you should be comparing against, which on genuinely subjective work is often only 0.6–0.7.</p>`)}

<p><b>What you are looking at.</b> A live calculation of Cohen's κ from your chosen observed agreement rate and how skewed the two raters' marginal preferences are, plotted against the standard interpretation bands from "slight" through "almost perfect", alongside a genuine 200-trial simulation of the position-swap test — the same pairwise comparison judged in both orders, counting how often the verdict flips.</p>
<p><b>What to do with it.</b> Reproduce the worked example's defaults and confirm κ lands at 0.57, squarely in the "moderate" band despite 82% raw agreement sounding much stronger on its own. Then push the "both raters pick A" marginal up toward 90% with agreement held fixed.</p>
<p><b>The thing genuinely worth noticing.</b> As the marginal skews further, κ falls even though observed agreement has not changed at all — the more lopsided the raters' shared bias toward one answer, the more of their apparent agreement chance alone would already explain, and the less it says about genuine discrimination between good and bad answers. That is exactly why κ, not raw agreement, is the number worth quoting when someone asks whether a judge can be trusted.</p>

${H.lab('kappa', 'Cohen’s κ calculator, and the position-swap test', 'Move the agreement rate and the marginals; κ falls out. The second panel simulates position bias: two judgements of the same pair in both orders, and how often they disagree.')}

<h2><span class="sn">4.16.3</span> pass@k, pass^k, and the arithmetic of unreliability</h2>
<p>Two different questions both get asked with the word "reliability", and confusing them produces headline numbers that are technically correct and practically misleading in opposite directions. The first question is generous: <i>if the model gets several independent tries at one problem, how likely is at least one of them to be right?</i> The second is unforgiving: <i>if a task requires a chain of several things to each go right in sequence, how likely is the whole chain to succeed?</i> The first question's answer climbs as you allow more tries. The second's answer falls as you demand more links in the chain. Both are legitimate, both get called "reliability", and a number computed under one framing says almost nothing about the other.</p>
<p><b>pass@k</b>, introduced by Chen et al. (2021) alongside the Codex code model and the HumanEval benchmark, formalises the first question precisely. Draw $n\\ge k$ independent samples for a coding problem, let $c$ of them pass the problem's unit tests, and ask: if you were only allowed to look at a random subset of $k$ of those $n$ samples, what is the probability at least one passes? Sampling without replacement gives the unbiased estimator</p>
$$\\text{pass@}k = 1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}}$$
<p>Read the fraction as "the probability every one of your $k$ picks lands among the $n-c$ <i>failing</i> samples" — $\\binom{n-c}{k}$ counts the ways to choose $k$ samples that are all wrong, and $\\binom{n}{k}$ counts every possible way to choose $k$ samples at all, so the ratio is exactly the chance of an all-wrong draw. One minus that is the chance of at least one success. With $n=20$ samples and $c=5$ of them correct, pass@1 is just $c/n=0.25$, matching what you would guess — one random sample, 25% of samples are right, 25% chance it is one of them. But pass@5 works out to about <b>81%</b> and pass@10 to about <b>98%</b>: giving the model ten independent tries at the same problem, and counting it a win if any single one works, makes a 25%-per-try model look nearly reliable. <b>pass@k measures the generous question, and it is the right metric exactly when a real user genuinely gets to keep the best of several tries</b> — an autocomplete suggestion the user can reject, a code generation tool that runs the test suite and shows you whichever sample passed.</p>
${H.pitfall('pass@k climbing sharply with k is not evidence the underlying model got more reliable — it is evidence that generous retry semantics forgive a lot. A model reported at "pass@10 = 98%" and a model reported at "pass@1 = 98%" are in completely different reliability classes even though the headline number is identical, and a pass@k figure quoted without stating k, or without stating whether your actual deployment gets to keep the best of several tries the way the benchmark did, is close to meaningless. Always ask which k a quoted pass@k used, and whether your product actually grants the retries the metric assumes.')}
<p>The unforgiving question has no equally famous name, but the arithmetic is simpler and the consequence is sharper. An agent (§5.3) completing a real task rarely gets to succeed by getting any one of several attempts right — it has to execute a <i>sequence</i> of steps, each one dependent on the last, and the task only succeeds if every step in the chain succeeds. If each step independently succeeds with probability $p$, and a task needs $k$ such steps chained together, the whole-task success probability is the product of $k$ independent probabilities:</p>
$$P(\\text{task succeeds}) = p^k$$
<p>This is the arithmetic of an AND, not an OR — succeed at step one <i>and</i> step two <i>and</i> every step after that — the exact opposite structure from pass@k's "at least one of $k$" OR. Take an agent that is 90% reliable at each individual step of a task — a genuinely strong per-step number, better than most deployed tool-calling accuracy — and chain together 8 such steps, a perfectly ordinary length for a real agentic workflow: read a file, call an API, parse the result, decide the next action, call a second tool, check the output, write a summary, send it. Whole-task reliability is $0.9^8\\approx$ <b>43%</b>, not 90%. Push the chain to 20 steps — not unusual for a genuinely multi-stage workflow — and it falls to $0.9^{20}\\approx$ <b>12%</b>. A component that looks impressively reliable in isolation can compound into a system that fails more often than it succeeds, purely as a consequence of chain length, with no single step ever getting any worse.</p>
${H.key('A 90%-reliable step chained eight times is a 43%-reliable task, not a 90%-reliable one. pass@k describes systems that get to keep the best of several independent tries; real multi-step agents are much closer to the opposite structure, where every step must succeed and failure anywhere breaks the chain — quote whole-task reliability, not per-step reliability, whenever a system chains more than one dependent action.')}
<p><b>Variance across runs</b> is the third piece of the same honesty problem, one level below both formulas above: any single reported accuracy figure, however it was computed, is itself one draw from a distribution, not a fixed property of the model. Run the identical evaluation twice — same model, same questions, sampling temperature above zero, or even at temperature zero if the serving stack's numerics are not perfectly deterministic across batches (§4.14) — and the two scores will differ by some amount that has nothing to do with either evaluation being wrong. For $n$ independent evaluation items each scored right or wrong at a true accuracy $p$, the standard error of the measured accuracy is $\\sqrt{p(1-p)/n}$, and a 95% confidence interval is roughly $\\pm1.96$ standard errors wide. At $n=100$ items and $p\\approx0.85$, that standard error works out to about 0.036, for a 95% interval of roughly <b>±7 percentage points</b>. A leaderboard change from 84% to 86% on a 100-item eval, the kind of headline that routinely gets reported as a meaningful win, sits entirely inside that noise band — you cannot distinguish it from a coin flip on the same 100 items landing slightly differently, without either a much larger evaluation set or repeated runs whose spread you actually measure and report.</p>
${H.practice('The practical fix is not a different formula so much as a different habit: report an interval, or a spread across repeated runs, alongside any single accuracy number, and treat two scores as different only when their intervals do not substantially overlap. This matters most exactly where the stakes are highest — a small, expensive human-labelled eval is the one most tempted to over-read a two-point movement, and it is also the one with the fewest items and therefore the widest noise band around every number it reports.')}

<h2><span class="sn">4.16.4</span> Hallucination</h2>
<p>Hallucination is not a bug in the ordinary sense of a mistake that better engineering removes — it follows directly from what §4.9.1 already established the pretraining objective actually is. Cross-entropy on next-token prediction only ever measures one thing: did the model assign high probability to the token that actually came next in this specific piece of training text. Nothing in that objective distinguishes a plausible sentence that happens to be true from an equally plausible sentence that happens to be false; the model is trained end-to-end to produce fluent, likely-looking continuations, with no separate channel anywhere in the loss that checks the continuation against reality. A model does not "decide to lie" any more than it "decides to tell the truth" — it produces the statistically likely continuation of its context, and when the likely continuation happens to be false, nothing in the training signal it received was ever positioned to catch that.</p>
<p>Mitigations are therefore necessarily <b>structural</b> rather than a matter of asking more nicely. Grounding the model with retrieval and requiring citations (§5.1) gives it something external and checkable to condition its answer on, rather than relying purely on parametric memory that has no confidence signal attached. Constraining decoding (§4.21) cannot fix content, but it guarantees the <i>shape</i> of an answer is checkable, which is a real, if partial, win. Training the model specifically for <b>abstention</b> — rewarding "I don't know" as a correct response on questions genuinely beyond its knowledge, via the same post-training machinery §4.12 builds for everything else — makes refusal a reachable output rather than one the model was implicitly punished for during fine-tuning on datasets where every question had a confident answer written for it. And external verification — checking specific claims against an independent source before acting on them — is simply mandatory wherever the cost of an undetected error is high, because no amount of prompting changes what the underlying objective was ever trained to optimise for.</p>

<h2><span class="sn">4.16.5</span> Multimodal, briefly</h2>
<p>A short bridge rather than a full treatment — §4.19 covers vision-language architecture in the depth the topic deserves, including the token-cost arithmetic of feeding an image through a vision-language model and CLIP's documented compositionality weaknesses. In outline: <b>ViT</b> cuts an image into patches and treats them as tokens, so attention needs no pixel-specific machinery at all — the same transformer block from §4.5 runs unmodified. <b>CLIP</b> trains an image encoder and a text encoder contrastively, pulling matched image-caption pairs together in one shared embedding space, which is what yields zero-shot classification (measure distance to an encoded class name) and the shared retrieval space multimodal search relies on. A <b>VLM</b> is a vision encoder plus a small projector network that maps visual features directly into the LLM's own token embedding space, so the language model attends to image content as if it were simply more words in the sequence — the same attention mechanism throughout, applied to a sequence that happens to have started life as pixels.</p>

<h2><span class="sn">4.16.6</span> How to build an eval worth having</h2>
${H.steps([
      'Start from <b>failures, not capabilities</b>: take 100–300 real inputs weighted toward cases that have actually gone wrong, not a generic capability checklist someone wrote from first principles.',
      'Label the outcome you want. <b>Freeze it. Version it. Never let it into a prompt.</b> A criterion that leaks into the system being evaluated stops being a criterion.',
      'Add three layers: deterministic checks (does it parse, does it cite, is the number right) wherever a checkable answer exists at all, pairwise judge comparisons for anything genuinely subjective, and a small human-reviewed sample specifically to keep the judge honest over time.',
      'Report <b>per-slice</b> results — an aggregate that moves from 84% to 86% while collapsing on the one slice that actually matters to users is a regression dressed as progress, and an aggregate number alone will not show you that it happened.'
    ])}
<p>The throughline across all four steps is the same: an evaluation set is only as trustworthy as its independence from whatever it is measuring, in exactly the sense §4.16.1's contamination argument made precise for public benchmarks, and exactly the sense a leaked reward function undermines training in §4.12.7. Build the eval as carefully as you would build the thing it is meant to check.</p>

${H.probe([
      ['Why swap positions when using a judge?', 'Judges systematically favour whichever answer is shown first, a bias that persists even when explicitly instructed against — averaging scores from both orders cancels the bias directly rather than merely reducing it.'],
      ['A benchmark says 90% — do you trust it?', 'Only after ruling out contamination — an older, more widely discussed benchmark is more likely to have leaked into training data — and never in place of a private eval built from your own distribution, which could not have leaked because it was never public.'],
      ['Is hallucination fixable by prompting?', 'No — it follows from the pretraining objective itself having no mechanism for checking truth (§4.9.1), so mitigations have to be structural: retrieval grounding with citations, constrained decoding for shape, trained abstention so "I don\'t know" is reachable, and external verification wherever errors are costly.']
    ])}`,
    labs: {
      kappa: function (host) {
        const st = Viz.controls(host, [
          { k: 'agree', label: 'observed agreement p₀', min: .5, max: 1, step: .01, value: .82, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'pa', label: 'both raters pick A this often', min: .3, max: .95, step: .01, value: .7, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'bias', label: 'judge position bias', min: 0, max: .5, step: .01, value: .18, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kappa', label: 'Cohen’s κ', cls: 'key' }, { k: 'pe', label: 'chance agreement p_e' },
          { k: 'verdict', label: 'interpretation' }, { k: 'flip', label: 'verdict flips on swap' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const pe = st.pa * st.pa + (1 - st.pa) * (1 - st.pa);
            const kappa = (st.agree - pe) / (1 - pe);
            ctx.font = '14px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillStyle = T.muted; ctx.fillText('κ = (p₀ − p_e) / (1 − p_e)', 18, 18);
            ctx.fillStyle = T.text; ctx.font = 'bold 16px ui-monospace, monospace';
            ctx.fillText('= (' + st.agree.toFixed(2) + ' − ' + pe.toFixed(2) + ') / (1 − ' + pe.toFixed(2) + ') = ' + kappa.toFixed(3), 18, 42);
            // scale
            const bx = 18, bw = w - 36, by = 84;
            const bands = [[0, .2, 'slight', T.red], [.2, .4, 'fair', T.amber], [.4, .6, 'moderate', T.amber], [.6, .8, 'substantial', T.green], [.8, 1, 'almost perfect', T.green]];
            bands.forEach(b => {
              ctx.fillStyle = b[3]; ctx.globalAlpha = .25;
              ctx.fillRect(bx + bw * b[0], by, bw * (b[1] - b[0]) - 2, 26); ctx.globalAlpha = 1;
              ctx.fillStyle = T.muted; ctx.font = '9px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText(b[2], bx + bw * (b[0] + b[1]) / 2, by + 30);
            });
            const kx = bx + bw * Math.max(0, Math.min(1, kappa));
            ctx.strokeStyle = T.text; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(kx, by - 8); ctx.lineTo(kx, by + 30); ctx.stroke();
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText(kappa.toFixed(2), kx, by - 10);
            // position swap simulation
            const R = Num.rng(13);
            let flips = 0;
            for (let i = 0; i < 200; i++) {
              const trueBetter = R() < .5 ? 'A' : 'B';
              const j1 = R() < .5 + st.bias ? 'first' : 'second';
              const j2 = R() < .5 + st.bias ? 'first' : 'second';
              const v1 = j1 === 'first' ? 'A' : 'B';
              const v2 = j2 === 'first' ? 'B' : 'A';
              if (v1 !== v2) flips++;
            }
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('Position-swap check on 200 simulated pairs: the same comparison judged in both orders', bx, by + 62);
            ctx.fillStyle = flips / 200 > .25 ? T.red : T.green; ctx.font = 'bold 13px ui-monospace, monospace';
            ctx.fillText((flips / 2).toFixed(0) + '% of pairs flip their verdict when the order is swapped — those are ties, not verdicts', bx, by + 84);
            out({
              kappa: kappa.toFixed(3), pe: pe.toFixed(3),
              verdict: kappa < .2 ? 'slight' : kappa < .4 ? 'fair' : kappa < .6 ? 'moderate' : kappa < .8 ? 'substantial' : 'almost perfect',
              flip: (flips / 2).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'Set the defaults and you reproduce the worked example: 82% agreement, κ = 0.57 — moderate, not strong. High raw agreement with a skewed marginal is mostly chance, which is exactly what κ is for.');
      }
    },
    quiz: [
      {
        q: 'Judge and human agree on 82 of 100 comparisons; both pick A 70% of the time. κ is…',
        options: ['0.82', '0.72', '0.57', '0.24'],
        answer: 2,
        why: 'Chance agreement is $p_e=0.7^2+0.3^2=0.58$ from the two raters both favouring A about 70% of the time, so $\\kappa=(0.82-0.58)/(1-0.58)=0.24/0.42\\approx0.57$ — exactly the worked example §4.16.2 walks through. "0.82" is the tempting wrong answer because it is the raw agreement rate handed to you directly, and it genuinely sounds the most reassuring — but raw agreement conflates real signal with the agreement two raters would produce purely by chance given how skewed their shared preference for A already is, exactly the correction Cohen\'s κ exists to make. "0.72" and "0.24" are intermediate quantities from the formula mistaken for the final answer — 0.24 is the numerator alone, not the ratio. The general principle is that a headline agreement percentage is not trustworthy on its own whenever two judges share a lopsided bias toward one option, and §4.16.2\'s whole discussion is why κ, not raw agreement, is the number worth quoting when someone asks whether a judge can be trusted.'
      },
      {
        q: 'The mandatory mitigation for LLM-judge position bias is…',
        options: ['using a bigger judge model', 'scoring both orders and averaging', 'raising the temperature', 'using pointwise scores'],
        answer: 1,
        why: 'Position bias — favouring whichever answer is shown first — persists even when the judge is explicitly told to ignore it, so the only mitigation that directly cancels it rather than merely reducing it is scoring the same pair in both orders and averaging, at the cost of doubling the judge calls; §4.16.2 marks this mandatory rather than optional. "Using a bigger judge model" is the tempting wrong answer because a stronger judge does generally correlate better with human preference, and it is natural to assume more capability fixes more problems — but position bias is a structural artefact of how the judge was itself trained on text where "the first thing mentioned" correlates with primacy, and no amount of raw capability removes that correlation. "Raising the temperature" addresses randomness in generation, not a systematic bias in judgment, and "using pointwise scores" moves the wrong direction, since §4.16.2 recommends pairwise comparison over pointwise scoring precisely because comparative judgments are more reliable. The general principle is that if a verdict changes when you swap the order of the two options, you have measured a property of the judge, not a genuine difference between the answers.'
      },
      {
        q: 'Hallucination is best mitigated by…',
        options: ['a stricter system prompt', 'structural measures: retrieval grounding with citations, constrained decoding, trained abstention, external verification', 'lower temperature alone', 'a larger context window'],
        answer: 1,
        why: 'Cross-entropy on next-token prediction measures only whether the model assigned high probability to the token that actually came next in training text, with no separate channel checking the continuation against reality, so §4.16.4\'s mitigations are necessarily structural: retrieval grounding with citations gives the model something external and checkable, constrained decoding guarantees the shape of an answer is checkable even though it cannot fix content, trained abstention makes "I don\'t know" a reachable output, and external verification catches specific claims wherever the cost of an error is high. "A stricter system prompt" is the tempting wrong answer because it feels like the natural first lever and costs nothing to try, but a model does not "decide to lie" any more than it "decides to tell the truth" — it produces the statistically likely continuation of its context regardless of instruction, so no amount of prompting changes what the underlying training objective was ever positioned to optimise for. "Lower temperature alone" and "a larger context window" address different failure modes entirely. The general principle is that a problem which follows directly from an objective\'s definition needs a fix external to that objective, not a better-phrased request made to the same untouched mechanism.'
      }
    ],
    cards: [
      { q: 'Cohen’s κ', a: '$(p_o-p_e)/(1-p_e)$; 82% agreement with 70/30 marginals gives κ = 0.57 — moderate.' },
      { q: 'Judge biases and fixes', a: 'Position (swap and average), verbosity (control length), self-preference (cross-family judge + human sample).' },
      { q: 'Building an eval', a: 'Start from real failures, 100–300 items, frozen and versioned, three layers (deterministic, pairwise judge, human sample), reported per slice.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.17 */
  ML.section({
    id: 'safety', track: 'llm', num: '4.17',
    title: 'Safety, red-teaming, and interpretability',
    lede: 'Increasingly asked even in applied roles, because anyone deploying a model owns these questions.',
    html: `
<p>Nothing in this section needs new machinery. Refusal is trained the same way any other behaviour in this part is trained — §4.12's post-training pipeline, applied to "decline harmful requests" as the target behaviour rather than "follow instructions" — and red-teaming reuses §4.16's judge-based evaluation apparatus, aimed at finding where that training failed rather than at scoring output quality. Treating safety as a special, separate discipline obscures how directly it follows from tools you already have; treating it as an afterthought is how a model ships with failure modes that were entirely predictable from how it was trained.</p>

<h2><span class="sn">4.17.1</span> Jailbreaks, by mechanism</h2>
<p>Refusal is a learned behaviour like any other in this part, which means it was learned from a finite set of training examples covering only some of the space of possible requests — and every jailbreak family below works by finding a distribution of inputs that behaviour was never actually trained on, rather than by defeating it head-on. But there is a sharper, more mechanical version of that story worth having, because it explains why so many superficially different attacks all work, rather than leaving you with five unrelated tricks to memorise.</p>
<p>A refusal is, at the level of tokens, mostly a <i>choice of opening</i>. SFT and RLHF training data teaches the model a small, fairly stereotyped set of ways to begin declining a request — "I can't help with that", "I'm not able to assist with…" — and because these openings are repeated across thousands of training examples, the gradient signal that shapes them is strong and concentrated on exactly those first several tokens. Once autoregressive generation (§4.2) has actually emitted one of those openings, everything that follows is generated by conditioning on a context that now contains its own refusal — the model is, in effect, continuing a sentence it has already started, and continuing to decline is the path of least resistance for next-token prediction from that point on, the same way any other few-token stylistic commitment tends to persist for the rest of a generation. <b>Safety training reshapes the probability distribution hardest over the first handful of output tokens, and comparatively lightly over everything the model might say after that</b>, because the training data itself is concentrated there — full harmful completions written out at length are rarer and noisier training signal than a clean, repeatable refusal opening.</p>
${H.intuition(`<p>That asymmetry is what a "prefill" or "prefix" attack exploits directly: force the assistant's turn to begin with an affirmative opening — "Sure, here's how to…" — rather than letting the model choose its own first tokens, and the rest of generation proceeds by conditioning on an already-compliant context, largely unconstrained by safety training that was concentrated on a different opening the model never got to choose. You do not have to defeat the safety training's judgement; you only have to prevent it from being consulted at the one place it was actually trained hardest.</p>
<p>Read the whole table below through this lens and the five rows stop looking like five unrelated tricks. Persona framing and obfuscation both work by changing what the first few tokens of the exchange look like, so that the stereotyped refusal-opening pattern simply never gets triggered in the first place. Crescendo works by never letting any single turn's opening look like the trigger, one incremental step at a time. Best-of-n works by accepting that the trigger fires most of the time and just sampling past the minority of cases where it does not. One mechanism, five surface forms.</p>`)}
${H.key('Safety training is concentrated on a model\'s first few output tokens far more than on the rest of its generation. Any attack that prevents the trained opening from being the one that gets emitted — by hiding the trigger, forcing a different one, or spreading the request thin enough that no single turn contains it — inherits whatever the base model would have said with no safety training at all.')}
${H.table(['Family', 'Mechanism'], [
      ['<b>Persona / role-play</b>', '"You are an author writing a villain" — moves the request into a frame where compliance looked appropriate in training'],
      ['<b>Obfuscation</b>', 'base64, leetspeak, another language, a cipher — the safety-relevant features never fire while the capability still does'],
      ['<b>Many-shot</b>', 'Fill a long context with dozens of fabricated compliant exchanges, exploiting in-context learning against the model’s own policy — a direct cost of long context (§4.4)'],
      ['<b>Crescendo</b>', 'Escalate gradually across turns so no single turn looks refusable'],
      ['<b>Best-of-n</b>', 'Sample the same request many times at high temperature until one attempt slips through']
    ])}
<p>Each mechanism deserves to be understood rather than merely catalogued. <b>Persona and role-play</b> attacks work because SFT and RLHF training data (§4.12.1, §4.12.2) legitimately contains creative-writing requests where a fictional villain explains something harmful within the story — that context genuinely was rated acceptable by human labellers, so the model learned that refusal is conditional on framing, not on the underlying content, and role-play exploits that conditionality directly by supplying the framing without changing the content at all. <b>Obfuscation</b> attacks a different, more mechanical gap: base64, leetspeak or a simple cipher changes the surface token pattern enough that whatever specific token sequences the safety training associated with "refuse this" never actually appear, while the much more general capability of decoding base64 or reading leetspeak — learned broadly from pretraining (§4.9), not from safety training specifically — still fires perfectly well, so the model complies with a request it would have refused in plain text. <b>Many-shot</b> jailbreaking is a direct, adversarial cost of exactly the in-context learning capability §4.15.4 described as a feature: fill a long context with dozens of fabricated exchanges where a compliant assistant answers similar harmful requests, and the model's own in-context learning — conditioning its next output on the pattern the context establishes — works exactly as designed, just against the model's own safety training rather than in service of it. <b>Crescendo</b> escalates gradually across many turns specifically so that no single turn, evaluated on its own, looks like something to refuse, exploiting the fact that refusal is typically judged locally rather than by integrating the full trajectory of a conversation. And <b>best-of-n</b> jailbreaking is the adversarial mirror of §4.15.3's test-time compute: just as sampling many attempts and keeping the best raises a benign task's success rate, sampling the same harmful request many times at high temperature and keeping whichever attempt happened to slip through raises an attacker's success rate — which is precisely the arithmetic the next subsection makes exact.</p>

${H.deriv('why a 99% refusal rate is not a 99% safety guarantee', [
      ['let $\\epsilon$ be the probability any one request slips through, so the refusal rate is $1-\\epsilon$.', 'for a 99% refusal rate, $\\epsilon=0.01$ — the per-request failure probability, small on its own.'],
      ['assume the attacker\'s $n$ attempts are independent trials.', 'an assumption worth flagging rather than granting silently — see the caveat below.'],
      ['probability every single one of the $n$ attempts fails $= (1-\\epsilon)^n$.', 'independent events: the probability all of them happen is the product of each one happening, exactly §1.5.1\'s likelihood move applied to failure instead of a dataset.'],
      ['probability at least one attempt succeeds $= 1-(1-\\epsilon)^n$.', '"at least one success" is the complement of "every attempt fails" — the two probabilities must sum to 1.']
    ], 'Plug in $\\epsilon=0.01$, $n=100$: $1-0.99^{100}\\approx0.634$. Sixty-three per cent, not one per cent — because the formula does not add $\\epsilon$ across attempts the way intuition wants to, it compounds a survival probability that shrinks geometrically. A per-request rate answers a question nobody actually asked: "how safe is one exchange?" The question that matters operationally is "how safe is this deployment, integrated over every attempt an adversary is willing to make?", and that is a different, much less forgiving number.')}
${H.pitfall('The independence assumption in step two is optimistic, and it is worth naming rather than quietly relying on. A real attacker does not resample the identical prompt $n$ times — they read each refusal, adjust, and try a modified version informed by what just failed, which correlates each attempt with the last one rather than drawing it fresh. That adaptivity should make the true success probability at a given $n$ <i>higher</i> than $1-(1-\\epsilon)^n$ predicts, not lower, so treat the formula\'s output as an optimistic floor on attacker success, not a pessimistic ceiling.')}

<p><b>What you are looking at.</b> The probability that an attacker succeeds at least once, plotted against the number of attempts they are allowed, computed directly from your chosen per-request refusal rate rather than illustrated schematically.</p>
<p><b>What to do with it.</b> Set the refusal rate to 99% — a number that sounds very safe in isolation — and read the success-probability readout at 100 attempts.</p>
<p><b>The thing genuinely worth noticing.</b> That readout reads 63.4%, not anywhere near the 1% a single "99% refusal" headline number might suggest, because $1-(1-\\epsilon)^n$ compounds every attempt the attacker is allowed — the exact derivation above, made interactive. The "rate needed" readout completes the argument: holding total risk under 1% across 100 attempts needs a per-request refusal rate above 99.99%, two orders of magnitude tighter than the number that sounded reassuring at first.</p>
${H.worked('solving the formula in the other direction', `<p>Instead of asking "how safe am I at 99% refusal?", ask the more useful question: "what refusal rate do I actually need?" Fix a target — total risk under 1% across $n=100$ attempts — and solve $1-(1-\\epsilon)^{100} \\le 0.01$ for $\\epsilon$.</p>
<p>Rearranged: $(1-\\epsilon)^{100} \\ge 0.99$, so $1-\\epsilon \\ge 0.99^{1/100}$. Compute the right-hand side: $0.99^{0.01}\\approx0.999899$, so $\\epsilon\\lesssim0.0001$ — a per-request refusal rate above <b>99.99%</b>, precisely the "rate needed" readout in the lab. Going from "sounds safe" (99%) to "is safe against 100 attempts" (99.99%) is a 100-fold tightening of the failure probability, not a rounding adjustment — which is the size of the gap a per-request number hides.</p>`)}

${H.lab('refusal', 'Per-request refusal rates are the wrong unit', 'A 99% refusal rate against an attacker who can retry. The curve is $1-(1-\\epsilon)^n$ and it is unforgiving — this is the calculation to have ready when someone quotes a single-request safety number.')}

<h2><span class="sn">4.17.2</span> The two-sided metric</h2>
<p>Safety training that is evaluated and optimised on only one axis — minimise harmful compliance — will do exactly that, and nothing stops it from overshooting into <b>over-refusal</b>: pushing down the probability of complying with harmful requests, with no matching pressure to keep complying with adjacent benign ones, predictably drifts the decision boundary toward caution generally. A model that declines to explain how to terminate a runaway process on Linux, because the request superficially pattern-matches something that sounds destructive, is not safe — it is simply unusable for that entire class of legitimate question, and users facing enough of these false refusals learn to route around the model's safety training rather than work within it, which is a worse outcome for actual safety than a narrower, more accurate refusal boundary would have been. The correct evaluation is therefore two-sided by construction: attack success rate on a red-team set of genuinely harmful requests, <i>and</i> false-refusal rate on a benign-but-adjacent set specifically constructed to resemble harmful requests without being one — and movement in either direction alone counts as a regression, not just movement in the attack-success direction.</p>
<p>Red-teaming itself is now largely automated, and the automation is a direct reuse of tools already built in this part: an attacker model generates and mutates candidate jailbreak prompts, and a judge model (§4.16.2) scores whether each attempt succeeded — the same generate-and-score loop RLAIF (§4.12.5) uses to replace human preference labels, here aimed at finding failures rather than training a policy. This finds far more of the attack surface than manual red-teaming ever could, simply by exhausting far more candidate prompts than a human team has hours for, with human effort reserved specifically for the genuinely novel attack classes the automated loop has no template to generate from in the first place.</p>
${H.analogy(`<p>A smoke detector tuned to never miss a real fire, with nobody checking whether it also screams at burnt toast, is not a good smoke detector — it is a device people learn to disable. Optimising refusal training against only the attack-success axis produces exactly that instrument: technically effective at the one thing it was measured on, and quietly self-defeating in a way the single metric cannot see. The fix is the same in both cases — measure the false-positive rate as deliberately as the true-positive rate, because a detector nobody trusts protects nobody, no matter how low its miss rate is.</p>`)}

<h2><span class="sn">4.17.3</span> Chain-of-thought faithfulness — the uncomfortable finding</h2>
<p>The reasoning a model prints before its answer is <b>not guaranteed to be the computation it actually performed</b> to reach that answer. Experiments can plant a cue in the prompt that measurably shifts a model's final answer, and the model will then produce a plausible, fluent chain of reasoning that never mentions the cue at all — a post-hoc justification rather than a genuine trace of the process. In the other direction, models can be steered to a wrong final answer while the printed reasoning leading up to it continues to look entirely sound at every individual step. Two consequences follow directly for practice. A legible chain of thought is <i>evidence, not proof</i>: it can genuinely help you debug a wrong answer by suggesting where reasoning went astray, but it should never be treated as an audit trail sufficient on its own for a regulated or high-stakes decision, precisely because it can be fluent and wrong about its own process simultaneously. And there is a real, structural tension in training models to produce reasoning that <i>reads well</i> versus reasoning that stays <i>informative</i> about the model's actual underlying computation: optimising the visible chain too directly toward looking clean and convincing — the same optimise-the-proxy dynamic §4.12.7 already catalogued for reward hacking generally, here applied to the reasoning trace itself rather than to the final answer — degrades exactly the property that made the chain worth reading as a monitor in the first place.</p>

<h2><span class="sn">4.17.4</span> Mechanistic interpretability, and what it actually buys</h2>
<p>Look inside a trained network for the neuron that means "legal disclaimer language" or "this code is in Python" and you typically will not find one — individual neurons are usually <b>polysemantic</b>, firing for several unrelated concepts at once rather than cleanly representing a single interpretable idea. The cause is a phenomenon called <b>superposition</b>, and it is a direct, practical consequence of a fact §0.2 already flagged as one of the genuine surprises of high-dimensional geometry: random vectors in a high-dimensional space are very nearly orthogonal to one another, so a space of $d$ dimensions can hold vastly more than $d$ directions that interfere with each other only slightly. A network exploits exactly that geometric slack to represent far more distinct features than it has neurons, packing them into overlapping, nearly-orthogonal directions rather than giving each feature its own dedicated dimension — which is efficient for the network and exactly why the resulting representation resists being read off neuron by neuron.</p>
<p><b>Sparse autoencoders</b> attack the resulting mess directly: train a much wider layer than the model's own hidden width, with a sparsity penalty forcing only a small number of its units to activate on any given input, to reconstruct the model's actual internal activations. The wide, sparse layer has enough room to give each packed feature something closer to its own dedicated unit, and empirically its units come out far more monosemantic than the original neurons — named, human-inspectable features such as "legal disclaimer language" or "the code is in Python" that a researcher can point to directly rather than infer indirectly. With features in hand, three things become possible that were not before: <b>auditing</b> what a model actually attended to on a specific input by inspecting which features fired, <b>steering</b> behaviour by directly amplifying or suppressing a specific feature at inference time, and <b>detecting</b> internal states such as deception or refusal-suppression with a trained probe against the feature space.</p>
${H.more('the sparsity penalty is not a new idea', `<p>"A sparsity penalty forcing only a small number of units to activate" is not a fresh piece of machinery invented for interpretability — it is §1.5.7's L1 penalty, doing exactly the job the lasso does there: an L1 term on the sparse layer's activations has the same constant-magnitude pull through zero that drives lasso coefficients to land exactly on zero rather than merely shrink, applied here to which <i>units</i> get to fire on a given input rather than to which <i>weights</i> survive in a regression. The reason a sparse autoencoder\'s units come out more interpretable than the original neurons is the same geometric argument as always: force most coordinates to be exactly zero for any given input, and whichever coordinates remain active are doing real, identifiable work rather than each contributing a diffuse, entangled sliver of many overlapping concepts.</p>`)}
<p>The limits are equally worth stating plainly, not as a footnote but as the honest boundary of what this buys you today. Coverage of a model's full feature set is still partial — nobody has recovered "all the features" of a frontier model, and there is no agreed way to know when you are close. None of this yet constitutes a compliance-grade explanation of model behaviour a regulator would accept, which is exactly why §2.17's SHAP, imperfect as it is, remains what actually goes in a model-risk file today rather than an interpretability feature dashboard.</p>
${H.flag('Treat SAE feature-stability as genuinely contested, not a settled limitation to note in passing. Retrain the same architecture on the same data with a different random seed, and the specific features a sparse autoencoder recovers are not guaranteed to be the same features, or even the same number of features, as the first run\'s — two independently trained SAEs on the same model can disagree noticeably about what the "legal disclaimer language" direction even is. This matters beyond an academic caveat: a steering or detection result built on one SAE\'s features may not replicate on a second SAE trained on the identical underlying model, and reproducibility across training runs, not just performance on one, is the bar any interpretability claim should be checked against before it is trusted for something safety-relevant.')}

<h2><span class="sn">4.17.5</span> Two more terms to recognise</h2>
<p><b>Sandbagging</b> is a model performing measurably worse specifically when it detects that it is being evaluated rather than used for real — which is precisely why held-out, unfamiliar-format evaluations carry value beyond simply avoiding the contamination problem §4.16.1 covers: an eval whose format the model has learned to recognise as "this is a test" is vulnerable to sandbagging even when its content is entirely uncontaminated. <b>Provenance</b> covers the tools built to trace whether a piece of content was AI-generated: watermarking text works by introducing a statistical bias into token selection during generation that a detector with the right key can identify, robust to light editing but not to paraphrasing, which removes the specific statistical signature by rewriting the content in different words; content credentials such as C2PA attach cryptographically signed metadata to images and video at the point of creation. Treat both as deterrents and forensic aids that raise the cost of misuse, never as reliable detectors on their own — and never build a policy or a product decision that depends on being able to reliably catch AI-generated text after the fact, because the detection side of this arms race is structurally the weaker one.</p>

${H.probe([
      ['Why do so many different jailbreak techniques all work?', 'Because safety training reshapes a model\'s output distribution hardest over the first few tokens of a response, where refusal openings are concentrated in training data — persona framing, obfuscation and crescendo all work by preventing that trained opening from being the one that gets emitted, after which generation conditions on an effectively unaligned context.'],
      ['Your model refuses 99% of attacks — is that safe?', 'Not against an attacker who can retry: 100 samples gives them a 63.4% success probability under $1-(1-\\epsilon)^n$, and holding total risk under 1% across those 100 attempts needs a per-request rate above 99.99%. Per-request rates are the wrong unit whenever retries are cheap, which is nearly always — and the true number is likely worse still, since real attackers adapt each attempt rather than resampling independently.'],
      ['Can I use the chain of thought as an audit trail?', 'No — it can be unfaithful to the actual computation, reaching an answer via an unmentioned cue or looking sound while being wrong. Use it as a debugging aid, not as evidence, and be aware that training the visible chain to look better can directly degrade how informative it is.'],
      ['What do sparse autoencoders give you, and what do they not?', 'More monosemantic features recovered from a superposed representation — the same near-orthogonality of random high-dimensional directions that §0.2 flagged as a genuine high-dimensional surprise — enabling auditing, steering and probe-based detection. They do not give you full coverage, stable features across separately trained runs (genuinely contested, not a footnote), or a compliance-grade explanation a regulator would accept.']
    ], 'Reporting only harmfulness and never false-refusal. Half a metric hides the cost of your own fix.')}`,
    labs: {
      refusal: function (host) {
        const st = Viz.controls(host, [
          { k: 'rate', label: 'per-request refusal rate', min: .9, max: .9999, step: .0001, value: .99, fmt: v => (v * 100).toFixed(2) + '%' },
          { k: 'n', label: 'attacker attempts', min: 1, max: 1000, step: 1, value: 100, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'succ', label: 'attacker success probability', cls: 'bad' },
          { k: 'need', label: 'attempts for 50% success' },
          { k: 'target', label: 'rate needed for <1% at this n', cls: 'key' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const eps = 1 - st.rate;
            const f = n => 1 - Math.pow(1 - eps, n);
            const P = Viz.plot(ctx, w, h, { xd: [1, 1000], yd: [0, 1] })
              .frame({ xlabel: 'number of attempts', ylabel: 'probability at least one succeeds', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.fn(f, { color: T.red, width: 2.8, n: 400 });
              P.vline(st.n, { color: T.text, dash: [4, 4] });
              P.dots([[st.n, f(st.n)]], { r: 5.5, color: T.red, stroke: true });
              P.hline(.5, { color: T.faint, dash: [3, 3], label: '50%' });
            });
            out({
              succ: (f(st.n) * 100).toFixed(1) + '%',
              need: Math.ceil(Math.log(.5) / Math.log(1 - eps)),
              target: (100 * (1 - (1 - Math.pow(1 - .01, 1 / st.n)))).toFixed(4) + '%'
            });
          }
        });
        Viz.note(host, 'At 99% refusal and 100 attempts the attacker wins 63% of the time. To hold total risk under 1% across 100 attempts you need a per-request refusal rate of about 99.99% — which is the honest way to state a safety requirement in an adversarial setting.');
      }
    },
    quiz: [
      {
        q: 'A model refuses 99% of harmful requests. Over 100 attempts, an attacker succeeds with probability…',
        options: ['1%', '10%', '63%', '99%'],
        answer: 2,
        why: '$1-(1-\\epsilon)^n$ with $\\epsilon=0.01$ and $n=100$ gives $1-0.99^{100}\\approx0.634$, because every additional attempt an attacker is allowed compounds multiplicatively against the per-request refusal rate, exactly the calculation §4.17.1\'s lab makes concrete. "1%" is the tempting wrong answer because it is the natural, intuitive reading of "99% refusal" — treating a single request\'s rate as though it applied unchanged to the whole interaction — but that reading silently assumes the attacker gets exactly one try, when the premise is that they get 100. "10%" undershoots the compounding by roughly an order of magnitude, and "99%" confuses the refusal rate itself with the attacker\'s cumulative success probability. The general principle is that per-request rates are the wrong unit whenever an adversary can retry cheaply, which is nearly always — holding total risk under 1% across 100 attempts actually needs a per-request refusal rate above 99.99%, two orders of magnitude tighter than the number that sounded reassuring on its own.'
      },
      {
        q: 'Chain-of-thought output can be used as…',
        options: ['a compliance audit trail', 'a debugging aid — it may be unfaithful to the actual computation', 'proof of correctness', 'a substitute for evaluation'],
        answer: 1,
        why: '§4.17.3 documents this directly: experiments can plant a cue in the prompt that measurably shifts a model\'s final answer while the printed reasoning never mentions the cue, and in the other direction a model can be steered to a wrong answer while its visible reasoning continues to look sound at every step — so a legible chain of thought is evidence, useful for debugging where reasoning went astray, but never proof of the computation actually performed. "A compliance audit trail" is the tempting wrong answer because a fluent, step-by-step explanation looks exactly like the record an audit trail should be, and it is natural to assume a model that shows its work is thereby showing you the truth of how it got there — but fluency and faithfulness are separate properties, and this section\'s whole finding is that a chain can have one without the other. "Proof of correctness" and "a substitute for evaluation" both overclaim what an unverified trace of text can establish. The general principle is the same optimise-the-proxy dynamic §4.12.7 catalogues for reward hacking, here applied to the reasoning trace itself: training the visible chain to look better can directly degrade how informative it actually is about the model\'s real computation.'
      },
      {
        q: 'Sparse autoencoders are used to…',
        options: ['compress the model', 'extract more monosemantic features from activations in superposition, enabling auditing and steering', 'quantise weights', 'detect watermarks'],
        answer: 1,
        why: 'Individual neurons are usually polysemantic because a network exploits the near-orthogonality of directions in high-dimensional space (§0.2) to pack far more distinct features than it has neurons — superposition — so a sparse autoencoder trains a much wider layer with a sparsity penalty to reconstruct those activations, giving each packed feature something closer to its own dedicated, human-inspectable unit, enabling auditing, steering, and probe-based detection of internal states. "Compress the model" is the tempting wrong answer because a sparse autoencoder sounds compression-adjacent, but it is trained on top of a frozen model purely to reconstruct and interpret its activations — it makes the model no smaller or cheaper to run at all, and is added machinery for interpretability, not efficiency. "Quantise weights" and "detect watermarks" name unrelated techniques from elsewhere in this part. The general principle is that §4.17.4 states the limits alongside the capability: coverage of a model\'s full feature set is still partial, discovered features are not stable across separately trained runs, and none of it yet constitutes a compliance-grade explanation a regulator would accept.'
      }
    ],
    cards: [
      { q: 'Why jailbreaks compose', a: 'Safety training is concentrated on a model\'s first few output tokens; anything that prevents the trained opening from firing inherits the unaligned base model for the rest of generation.' },
      { q: 'Refusal arithmetic', a: '99% per-request refusal → 63.4% attacker success over 100 tries ($1-0.99^{100}$); need >99.99% refusal to hold risk under 1%.' },
      { q: 'The two-sided safety metric', a: 'Attack success rate AND false-refusal rate on benign-adjacent prompts; movement in either is a regression.' },
      { q: 'CoT faithfulness', a: 'The printed chain need not be the computation performed — a debugging aid, not an audit trail.' },
      { q: 'Superposition & SAEs', a: 'Neurons are polysemantic because features outnumber dimensions; sparse autoencoders (an L1-sparse reconstruction, same penalty as lasso) recover more monosemantic features — but which features, is not stable across training runs.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.23 */
  ML.section({
    id: 'part4-recall', track: 'llm', num: '4.23',
    title: 'Rapid recall — Part 4 in fourteen lines',
    lede: 'The transformer sheet. Every line should unpack into a derivation or an arithmetic you can do on a whiteboard.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', '√dₖ: the dot product’s variance is dₖ; rescale or the softmax saturates.', '<a href="#/attention">4.3</a>'],
      ['2', 'RoPE: $(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — relative position for free.', '<a href="#/rope">4.4</a>'],
      ['3', 'KV bytes = 2·L·n_kv·d_head·seq·batch·bytes.', '<a href="#/kv-cache">4.7</a>'],
      ['4', 'Cache size: MHA ≫ GQA > MLA ≈ MQA.', '<a href="#/kv-cache">4.7</a>'],
      ['5', 'FlashAttention is exact — an IO optimisation, not sparsity.', '<a href="#/kv-cache">4.7</a>'],
      ['6', 'C ≈ 6ND; Chinchilla ≈ 20 tokens/param (⚑ coefficients contested).', '<a href="#/scaling-laws">4.10</a>'],
      ['7', 'Inference-optimal overtrains: Llama-3-8B ≈ 1,875 tok/param.', '<a href="#/scaling-laws">4.10</a>'],
      ['8', 'DPO cancels Z(x); it also inflates length (arXiv:2403.19159).', '<a href="#/post-training">4.12</a>'],
      ['9', 'GRPO: $A_i=(r_i-\\mathrm{mean})/\\mathrm{std}$, no critic. RLVR: rule-based reward.', '<a href="#/post-training">4.12</a>'],
      ['10', 'LoRA r=16, α=2r, all linear layers; QLoRA = NF4 base + BF16 adapter.', '<a href="#/lora">4.13</a>'],
      ['11', 'Bytes/param: FP32 4 · FP16/BF16 2 · FP8/INT8 1 · INT4 0.5.', '<a href="#/kv-cache">4.7</a>'],
      ['12', 'Prefill compute-bound; decode memory-bandwidth-bound.', '<a href="#/serving">4.14</a>'],
      ['13', '70B / 4k / batch 8 → 10 GB of KV cache.', '<a href="#/kv-cache">4.7</a>'],
      ['14', 'Judge: always swap positions and control for length.', '<a href="#/llm-eval">4.16</a>']
    ])}
${H.table(['#', 'Also', 'Section'], [
      ['15', 'Long context: local + sink tokens + a few global layers; test with RULER, not one needle.', '<a href="#/rope">4.4</a>'],
      ['16', 'CoT can be unfaithful; SAEs give monosemantic features; 99% refusal ≠ safe under retries.', '<a href="#/safety">4.17</a>'],
      ['17', 'Speculative decoding: $(1-\\alpha^{k+1})/(1-\\alpha)$ — α=0.7, k=4 → 2.8 tokens/pass.', '<a href="#/speculative">4.22</a>'],
      ['18', 'Llama-3-8B parameter arithmetic: 42.0M attention + 176.2M FFN per layer → 8.03B.', '<a href="#/block">4.5</a>'],
      ['19', 'ViT: patchify → project → add position → ordinary transformer. 224×224 at 16×16 = 196 tokens.', '<a href="#/multimodal">4.19</a>'],
      ['20', 'High-resolution image tiling costs 2k–6k tokens per picture — budget it like eight pages of prose.', '<a href="#/multimodal">4.19</a>'],
      ['21', 'CoT works because fixed work per token means generated tokens buy sequential passes: depth limit → length budget.', '<a href="#/reasoning">4.20</a>'],
      ['22', 'Constrained decoding: compile the schema to an automaton, mask every token that cannot continue a valid string.', '<a href="#/structured-output">4.21</a>'],
      ['23', 'Structured output guarantees shape, never semantics — well-formed nonsense passes every automated check.', '<a href="#/structured-output">4.21</a>']
    ])}

${H.lab('drill4', 'Part 4 drill', 'Eighteen prompts, shuffled. These are the ones asked most often.')}`,
    labs: {
      drill4: function (host) {
        const cards = [
          ['Why divide attention logits by √dₖ?', 'Var(q·k)=dₖ, so logits scale like √dₖ; large logits saturate the softmax and kill gradients.'],
          ['State the RoPE identity.', '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — the score depends only on the relative offset.'],
          ['KV cache formula.', 'bytes = 2·L·n_kv·d_head·seq·batch·bytes-per-element.'],
          ['Size the cache for 70B, 4k, batch 8, BF16.', '2·80·8·128·4096·8·2 ≈ 10 GB.'],
          ['Is FlashAttention an approximation?', 'No — exact; it tiles into SRAM with an online softmax so the s×s matrix never reaches HBM.'],
          ['Order the KV variants by cache size.', 'MHA ≫ GQA > MLA ≈ MQA.'],
          ['Training compute formula, and for 7B on 2T?', 'C≈6ND ≈ 8.4×10²² FLOPs.'],
          ['Chinchilla in one line, with the caveat.', '≈20 tokens per parameter at fixed compute; ⚑ exact coefficients are contested.'],
          ['Why train past compute-optimal?', 'Inference cost dominates lifetime cost — pay training once, inference forever.'],
          ['How does DPO remove the reward model?', 'Invert the closed-form optimal policy; in the Bradley–Terry difference Z(x) cancels, leaving a logistic loss on pairs.'],
          ['DPO’s documented failure.', 'Length inflation (arXiv:2403.19159); monitor mean response length.'],
          ['GRPO advantage, and its degenerate case.', '$(r_i-\\mathrm{mean})/\\mathrm{std}$; a unanimous group has σ=0 and teaches nothing.'],
          ['What is RLVR and why does it matter?', 'RL from verifiable rewards (tests pass, answer matches) — nothing to hack; how reasoning models are trained.'],
          ['LoRA defaults in 2026.', 'r=16, α=2r, all linear layers; QLoRA = 4-bit NF4 base + BF16 adapter.'],
          ['The 8B QLoRA memory numbers.', '≈42M trainable (0.52%), ≈7 GB vs ≈145 GB full fine-tune.'],
          ['Bytes per parameter by precision.', 'FP32 4 · FP16/BF16 2 · FP8/INT8 1 · INT4 0.5.'],
          ['Prefill vs decode.', 'Prefill compute-bound (sets TTFT); decode memory-bandwidth-bound (sets tokens/s).'],
          ['Speculative decoding expected tokens.', '$(1-\\alpha^{k+1})/(1-\\alpha)$; α=0.7, k=4 → 2.8.'],
          ['Two mandatory judge mitigations.', 'Swap positions and average; control for length.'],
          ['Long-context recipe and its evaluation.', 'RoPE interpolation + local layers + sink tokens + a few global layers; evaluate with RULER-style multi-needle tests.'],
          ['Parameters per transformer layer.', '≈12d²: 4d² attention + 8d² FFN.'],
          ['99% refusal over 100 attempts?', '63% attacker success — per-request rates are the wrong unit.'],
          ['ViT in one line.', 'Patchify, linearly project, add position, then an ordinary transformer. 224×224 at 16×16 is 196 tokens.'],
          ['Inductive bias vs data.', 'A ViT loses to a CNN on ImageNet-1k and wins at 300M images — bias substitutes for data and stops paying.'],
          ['Why does chain of thought work at all?', 'A transformer does fixed work per token, so generated tokens buy extra sequential passes: a depth limit becomes a length budget.'],
          ['Operational cost of reasoning models.', 'Variable latency and cost, with p99 set by the token tail. Cap the thinking budget and route only hard cases to it.'],
          ['Constrained decoding in one line.', 'Compile the schema to an automaton, mask every token that cannot continue a valid string, sample, advance.'],
          ['What structured output does not fix.', 'Semantic correctness — well-formed nonsense passes every automated check.']
        ];
        let order = cards.map((_, i) => i).sort(() => Math.random() - .5);
        let i = 0, showA = false;
        const face = ML.el('div', { class: 'card-face', style: 'cursor:pointer;border:1px solid var(--line);border-radius:12px;background:var(--panel)' });
        host.appendChild(face);
        const pos = ML.el('span');
        function draw() {
          const c = cards[order[i]];
          face.innerHTML = showA ? '<div class="a">' + c[1] + '</div>' : '<div><b>' + c[0] + '</b></div>';
          pos.textContent = (i + 1) + ' / ' + cards.length;
          ML.typeset(face);
        }
        face.addEventListener('click', () => { showA = !showA; draw(); });
        const nav = ML.el('div', { class: 'cardnav' });
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: '←', onclick: () => { i = (i - 1 + cards.length) % cards.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn primary', type: 'button', text: 'Flip', onclick: () => { showA = !showA; draw(); } }));
        nav.appendChild(pos);
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: '→', onclick: () => { i = (i + 1) % cards.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Shuffle', onclick: () => { order = order.sort(() => Math.random() - .5); i = 0; showA = false; draw(); } }));
        host.appendChild(nav);
        draw();
      }
    },
    quiz: [
      {
        q: 'Which single number is most often needed on an LLM whiteboard?',
        options: ['The learning rate', 'The KV cache size for a given model, context and batch', 'The vocabulary size', 'The number of attention heads'],
        answer: 1,
        why: 'The KV cache size for a given model, context length and batch — computed from bytes $=2\\cdot L\\cdot n_{kv}\\cdot d_{head}\\cdot\\text{seq}\\cdot\\text{batch}\\cdot\\text{bytes-per-element}$, the same formula §4.7 derives and this recall page lists as line 3 — decides how many cards a deployment needs, what batch size is even possible, and whether a long context is feasible at all, which is why it recurs across serving, KV-variant and memory-budget discussions throughout Part 4. "The learning rate" is the tempting wrong answer because it is genuinely important and frequently discussed, but it is a training-time hyperparameter tuned per run rather than a single number with a fixed formula you derive live on a whiteboard the way the cache size is. "The vocabulary size" and "the number of attention heads" are architectural facts you would look up for a specific model, not quantities whose derivation an interviewer is testing. The general principle behind this recall page is that every line should unpack into a derivation or an arithmetic you can perform, and the KV cache formula is the single most load-bearing example of that because it feeds directly into cost, capacity and feasibility decisions.'
      }
    ]
  });
})();
