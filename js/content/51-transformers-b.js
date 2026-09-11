/* ============================================================
   PART 4 — LLMs & transformers (4.9 – 4.18)
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
<p><b>Deduplication</b> matters for a reason beyond the obvious one. If a page of text appears a thousand times in the crawl — a common outcome for boilerplate, licence text, and popular quotes — training on it a thousand times does not merely waste compute; it gives the model a thousand gradient updates reinforcing that exact string, far out of proportion to its actual importance, and the model becomes able to recite it close to verbatim. Removing near-duplicates (MinHash estimates how similar two documents are cheaply, by comparing compact hashed sketches rather than every pair of documents directly, which is the only way deduplication is computationally feasible at web scale) both flattens that over-representation and is the single most effective lever against verbatim memorisation of any specific document, which matters for both privacy and copyright reasons.</p>
<p><b>Quality filtering</b> trains a small classifier to distinguish text that resembles a curated reference set — Wikipedia, well-edited books, moderated forums — from the rest of the crawl, and keeps only what scores highly. The finding that this reliably beats simply keeping more data, at matched compute, says something specific about what a forward pass is for: every token spent on a low-information page (a listicle, a template-generated product description) is a forward pass not spent on text the model can actually learn structure from, so a smaller, denser corpus outperforms a larger, diluted one of the same total token count. <b>Mixture weights</b> — the proportions of code, web text, books, mathematics and each language in the final blend — are consequently treated as a first-class, tuned hyperparameter rather than an afterthought: code in the mixture, in particular, has been shown to improve reasoning performance even on entirely non-code tasks, a transfer effect that is not yet fully understood but is consistently observed.</p>
<p><b>Curriculum and anneal</b> exploit something about training dynamics rather than data quality per se: the order tokens are seen in is not neutral. A short, high-quality phase placed at the very <i>end</i> of pretraining, after the learning rate has decayed toward its minimum (§4.11's schedules), buys a further, close-to-free quality gain, because a small learning rate makes late updates surgical rather than sweeping — the model is maximally able to absorb the specific character of whatever it sees last without that signal being drowned out by the large, coarse updates earlier training runs on. <b>Decontamination</b> — removing any text that overlaps with known evaluation benchmarks — has to happen at this stage, before training, not as a post-hoc audit; §4.16 covers precisely what goes wrong if it is skipped, but the short version is that a benchmark score on contaminated training data measures how old the benchmark is, not how capable the model is.</p>

<p><b>What you are looking at.</b> A funnel of horizontal bars, each one a stage of the pipeline just described, shrinking left to right as tokens are progressively removed; the width of each bar is the token count surviving that stage, computed live from the filter strengths you set, with the final high-quality anneal phase picked out in a different colour.</p>
<p><b>What to do with it.</b> Push deduplication and quality-filter strictness up from zero and watch two readouts move in opposite directions: usable tokens fall, and relative quality per token rises. There is no setting where both improve together — that trade is the entire content of this subsection made numeric.</p>
<p><b>The thing genuinely worth noticing.</b> The final readout converts whatever token count survives your filters into a Chinchilla-optimal model size (§4.10's 20-tokens-per-parameter ratio, applied here rather than derived). That is the uncomfortable, practical punchline of the whole funnel: the model size you can responsibly train is set by how much your data pipeline actually produces, not by how many GPUs you can buy.</p>

${H.lab('corpus', 'The corpus funnel, with the knobs that matter', 'Move the filters and watch tokens fall while modelled quality rises. The curve is illustrative of the published direction of travel — the point is the shape: aggressive filtering costs tokens and buys quality, up to the point where you run out of data.')}

<h2><span class="sn">4.9.3</span> Synthetic data is now a first-class ingredient</h2>
<p>Model-generated text used to be treated as a fallback for when real data ran short. It no longer is — three specific mechanisms now carry serious weight in frontier pretraining and post-training mixtures, each solving a different problem. <b>Distillation from a stronger model</b> generates answers with a frontier model and trains a smaller one on them; it is the single cheapest way to move a small model's quality, since the expensive part (producing a good answer) has already been paid for by whoever trained the teacher, subject to that teacher's licence terms permitting the use. <b>Self-instruct bootstrapping</b> has a model write new instructions and answers starting from a small seed set of examples, then filters the results for quality — a way of multiplying a small amount of expensive human-curated data into a much larger training set without a proportional increase in human effort. <b>Verifier-filtered generation</b> keeps only generated samples that pass an external checker — a unit test suite for code, a symbolic checker for a maths answer — which is exactly the same idea §4.12's RLVR applies to <i>rewards</i> rather than to <i>data</i>, and it is precisely why mathematics and code benefit disproportionately from synthetic data: both domains have cheap, reliable, automatic verifiers, so filtering is nearly free, whereas open-ended prose has no equivalent checker and synthetic data there is correspondingly harder to trust.</p>
${H.history(`<p>The caveat worth stating before anyone asks is now backed by published evidence rather than intuition alone: Shumailov et al.'s 2024 <i>Nature</i> paper demonstrated "model collapse" directly — train a model on its own unfiltered outputs, then train the next generation on <i>that</i> model's outputs, and repeat, and each generation's output distribution narrows measurably, losing the rare, tail information that made the original data rich, until later generations produce text that is fluent but has visibly forgotten the diversity of the real distribution it started from. The mechanism is intuitive once stated: a model's outputs are always a slightly narrower, slightly smoothed copy of its training distribution, so training on nothing but that copy — with no fresh, real data anchoring each generation back to reality — is repeated photocopying of a photocopy. Each generation loses a little more of what the original had.</p>`)}
${H.flag('The caveat to state before anyone asks: training repeatedly on unfiltered self-generated text degrades diversity and eventually quality — "model collapse". Synthetic data works when it is filtered against something real, and fails when it is a closed loop.')}

<h2><span class="sn">4.9.4</span> Continual pretraining</h2>
<p>Between full pretraining from scratch and lightweight fine-tuning (§4.13) sits a third option that gets less attention than it deserves: take an existing base model and keep training it, with the identical next-token-prediction objective from §4.9.1, on a domain-specific corpus — case law, medical literature, one bank's internal documentation — using a low learning rate and, critically, a <b>replay mixture</b> that mixes in a slice of the original general-purpose training data alongside the new domain text. That replay slice exists to guard against <b>catastrophic forgetting</b>: training purely on narrow domain text, with no counterbalance, measurably erodes the model's general capabilities even as it gains domain fluency, because gradient descent has no built-in preference for retaining what it is not currently being shown. Continual pretraining buys something LoRA (§4.13) fundamentally cannot: LoRA adapts a model's <i>behaviour</i> within a low-rank subspace of its existing weights, but it has nowhere near the capacity to install large amounts of genuinely new factual knowledge the base model never saw. When the honest answer to "can we make the model actually know our internal product catalogue" is no, retrieval is insufficient, and a lightweight adapter is not enough, continual pretraining — expensive, and requiring real care around the replay mixture and learning rate — is the answer that is left.</p>

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
        why: 'Both effects are documented; memorisation reduction matters for privacy and for benchmark integrity.'
      },
      {
        q: 'Training a model repeatedly on its own unfiltered outputs leads to…',
        options: ['steady improvement', 'model collapse — diversity and then quality degrade', 'faster convergence', 'better calibration'],
        answer: 1,
        why: 'Synthetic data works when filtered against something real; a closed loop narrows the distribution.'
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
<p>Empirically, as you train larger models on more data with more compute, loss falls in a remarkably smooth, predictable way — not noisily, not with sudden jumps, but following what looks like a straight line on log-log axes across many orders of magnitude of parameters, tokens and compute. A relationship that is linear in log-log space is exactly what a <b>power law</b> is: loss falls as roughly $N^{-\\alpha}$ for some small constant $\\alpha$, and the practical consequence of that smoothness is that you can fit the curve on small, cheap runs and extrapolate it — with real, if imperfect, confidence — to predict the loss of a run a thousand times larger before you have spent the money to run it. That predictability, more than any single number in this section, is what turned frontier training from a gamble into a budgeting exercise.</p>
<p>§4.5 already derived the compute cost of one token of training: $2N$ FLOPs forward, $4N$ backward, $6N$ total. Multiply by $D$ training tokens and you get the standard accounting used everywhere in this course and in nearly every published scaling-law paper:</p>
$$C \\approx 6ND \\text{ FLOPs}$$
<p>This single formula, entirely earned rather than asserted — you derived the 6 yourself in §4.5 — lets you sanity-check any claimed training run in the time it takes to multiply three numbers: quote a parameter count and a token count for a model, and $6ND$ tells you the compute that run must have consumed, checkable in seconds against whatever compute budget was claimed alongside it.</p>

<h2><span class="sn">4.10.2</span> Chinchilla</h2>
<p>Given a fixed compute budget $C$, the accounting above leaves one free choice: how to split $C=6ND$ between a bigger $N$ and a bigger $D$, since increasing either one at fixed compute forces the other down. Hoffmann et al. (2022) — the paper universally referred to by its model's codename, "Chinchilla" — fit loss as a function of both independently, $L(N,D) = E + A/N^{a} + B/D^{b}$, and found the field had been getting the split wrong: contemporary models were trained far too large relative to how much data they were shown. Minimising $L$ subject to the compute constraint $C=6ND$ is a standard constrained-optimisation problem of exactly the shape §1.12 covers — trade one resource against another until the marginal loss reduction per extra FLOP is equal whichever resource you spend it on — and solving it (the specific exponents $a$ and $b$ determine the exact balance point) produces a strikingly simple headline answer: for a fixed compute budget, parameters and tokens should scale <b>together</b>, in roughly fixed proportion, at approximately <mark>20 tokens per parameter</mark>. Their empirical demonstration was concrete and hard to argue with: a 70B model trained on 1.4T tokens (close to that 20:1 ratio) outperformed a 280B model trained on considerably less data per parameter, despite the 280B model having cost comparable or greater total compute — proof that the larger model's extra parameters were, in a real sense, wasted relative to how little data they had been shown.</p>
${H.flag('Flag when you cite this: the ~20:1 headline is robust and replicated, but the exact fitted coefficients are contested — Epoch AI’s replication differs from the original fit. Quote the ratio, not the exponents.')}

<p><b>What you are looking at.</b> A contour map of the Chinchilla loss surface over log parameters (x-axis) and log tokens (y-axis) — each faint grey curve is an iso-loss contour, every point on it achieving the identical loss. The blue curve is your chosen compute budget's iso-compute line, $D=C/(6N)$; the dashed red ray is the 20-tokens-per-parameter line the theory predicts.</p>
<p><b>What to do with it.</b> Watch where the blue iso-compute curve is tangent to the lowest grey contour it can reach — that tangency point, marked A, is the compute-optimal split, and it sits almost exactly where the blue curve crosses the red 20:1 ray, for any compute budget you choose.</p>
<p><b>The thing genuinely worth noticing.</b> Slide the "tokens per parameter" control far away from 20 and watch point B slide along the <i>same</i> iso-compute curve — same total training cost — onto a visibly higher-loss contour. Training away from the Chinchilla ratio at fixed compute genuinely costs you quality; the ratio is not a stylistic preference, it is where the tangent point actually sits on the measured loss surface.</p>

${H.lab('chinchilla', 'The iso-loss surface, and where your budget lands', 'Move the compute budget and watch the compute-optimal point slide along the 20-tokens-per-parameter ray. Then move along a single iso-loss contour to see the inference-optimal trade: identical loss, a smaller model, far more data — and a much cheaper deployment.')}

<h2><span class="sn">4.10.3</span> Inference-optimal is a different question</h2>
<p>Chinchilla answers a specific, narrower question than it is often credited with: given a fixed compute budget, which split of parameters and tokens minimises <i>training</i> loss per training FLOP. It says nothing about the cost of everything that happens after training finishes, and for any model that will actually be deployed, that omission turns out to be the whole story. A model is trained once and served, potentially, billions of times; §4.7 already showed that decode cost scales with the number of parameters $N$ (every weight has to be read for every generated token), so total lifetime cost is training cost <i>plus</i> inference cost accumulated over every request the model will ever serve — and for a model with enough users, that second term dwarfs the first. If lifetime cost is what you actually want to minimise, and inference cost per token scales with $N$ while training cost was already fixed by the compute-optimal point, the arithmetic favours moving to a <i>smaller</i> $N$ than Chinchilla recommends and compensating with proportionally more training tokens $D$ to reach comparable loss — deliberately "overtraining" relative to the training-only optimum, because a smaller model is cheaper on every single one of the millions of inference calls that follow.</p>
<p>This is now standard practice for anything actually deployed, and the numbers involved are not subtle. Llama-3-8B was trained on roughly 15 trillion tokens against 8 billion parameters — about <b>1,875 tokens per parameter</b>, some 90 times the Chinchilla ratio of 20:1. The one-line answer to "why train so far past the compute-optimal point" is a sentence worth having ready in an interview: <mark>you pay training compute once, and you pay inference compute forever, so the more the model will be served, the more it pays to shrink it — at extra training cost — before you ever put it in front of a user.</mark></p>

<p><b>What you are looking at.</b> A live compute calculator: enter any two of parameter count, token count and compute budget, and it derives the third from $C=6ND$, alongside the resulting GPU-hours, wall-clock time at your chosen cluster size, and an approximate dollar cost.</p>
<p><b>What to do with it.</b> Set 7B parameters and 2T tokens and confirm the compute readout lands on $8.4\\times10^{22}$ FLOPs — the same figure the quiz below asks you to derive by hand. Then try the Llama-3-8B preset button and compare its tokens-per-parameter readout against the Chinchilla-ratio preset at the same total compute.</p>
<p><b>The thing genuinely worth noticing.</b> Two runs at the <i>same</i> total compute — one at the 20:1 Chinchilla ratio, one at the ~1,875:1 Llama-3 ratio — land at very different model sizes for the same training bill, and this is precisely the trade §4.10.2's lab visualised geometrically: moving along an iso-compute curve away from the tangent point costs training-loss quality, but if the resulting model is going to be queried by a billion users, that training-time quality cost can still be the cheaper choice once inference is counted.</p>

${H.lab('budget', 'Training budget calculator', 'Enter any two of parameters, tokens and compute and read off the third, with GPU-hours and a rough cost. Check it against a published run — the arithmetic is the same one used to plan them.')}

${H.probe([
      ['Chinchilla in one line?', 'For a fixed training budget, scale parameters and tokens together at about 20 tokens per parameter — the point where the iso-compute curve is tangent to the lowest reachable loss contour, a standard constrained-optimisation balance (§1.12).'],
      ['Then why does everyone train past it?', 'Deployment cost is dominated by inference, and inference cost per token scales with parameter count (§4.7); a smaller, overtrained model is more expensive to train but cheaper on every one of the millions of requests that follow, so total lifetime cost favours shrinking N below the training-optimal point.'],
      ['Compute for a 7B on 2T tokens?', '$6ND \\approx 6\\cdot7\\times10^9\\cdot2\\times10^{12} \\approx 8.4\\times10^{22}$ FLOPs — the derived-not-quoted 6 from §4.5, applied directly.']
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
            // Chinchilla-flavoured loss surface: L = E + A/N^a + B/D^b
            const E = 1.69, A = 406.4, a = 0.34, B = 410.7, b = 0.28;
            const loss = (N, D) => E + A / Math.pow(N, a) + B / Math.pow(D, b);
            const P = Viz.plot(ctx, w, h, { xd: [8, 12.2], yd: [10, 14] })
              .frame({ xlabel: 'log₁₀ parameters N', ylabel: 'log₁₀ tokens D', xfmt: v => '10^' + v.toFixed(0), yfmt: v => '10^' + v.toFixed(0) });
            P.contours((ln, ld) => loss(Math.pow(10, ln), Math.pow(10, ld)), [1.9, 2.0, 2.15, 2.3, 2.5, 2.8, 3.2], { color: T.faint, alpha: .6, nx: 70, ny: 60 });
            P.clip(() => {
              // 20 tokens/param ray
              P.fn(ln => Math.log10(20 * Math.pow(10, ln)), { color: T.red, width: 2, dash: [6, 4] });
              // iso-compute curve: D = C/(6N)
              P.fn(ln => Math.log10(C / (6 * Math.pow(10, ln))), { color: T.blue, width: 2.6 });
              // optimal point on this budget at 20:1
              const Nopt = Math.sqrt(C / (6 * 20));
              const Dopt = 20 * Nopt;
              P.dots([[Math.log10(Nopt), Math.log10(Dopt)]], { r: 6, color: T.red, stroke: true });
              P.text(Math.log10(Nopt), Math.log10(Dopt), '  A · compute-optimal', { color: T.red, font: '11px ui-sans-serif' });
              const Nyour = Math.sqrt(C / (6 * st.ratio)), Dyour = st.ratio * Nyour;
              P.dots([[Math.log10(Nyour), Math.log10(Dyour)]], { r: 6, color: T.green, stroke: true });
              P.text(Math.log10(Nyour), Math.log10(Dyour), '  B · your ratio', { color: T.green, font: '11px ui-sans-serif' });
            });
            const Nopt = Math.sqrt(C / (6 * 20)), Dopt = 20 * Nopt;
            const Nyour = Math.sqrt(C / (6 * st.ratio)), Dyour = st.ratio * Nyour;
            out({
              nopt: (Nopt / 1e9).toFixed(1) + 'B', dopt: (Dopt / 1e12).toFixed(2) + 'T',
              nyour: (Nyour / 1e9).toFixed(1) + 'B', dyour: (Dyour / 1e12).toFixed(2) + 'T',
              infer: '×' + (Nyour / Nopt).toFixed(2)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Chinchilla 20:1', primary: true, on: () => { st.$set('ratio', 20); S.redraw(); } },
          { label: 'Llama-3-8B ≈ 1875:1', on: () => { st.$set('ratio', 1875); S.redraw(); } }
        ]);
        Viz.note(host, 'Both points sit on the same iso-compute curve — the same training budget. Point B is a much smaller model trained on far more tokens: similar loss, and the "relative inference cost" readout is what you pay forever afterwards. That single comparison is why the field moved past compute-optimal.');
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
        why: '$6ND = 6 \\times 7\\times10^9 \\times 2\\times10^{12} = 8.4\\times10^{22}$ FLOPs.'
      },
      {
        q: 'Why do deployed models train far past the Chinchilla ratio?',
        options: ['Chinchilla was wrong', 'Inference cost dominates lifetime cost, so a smaller overtrained model is cheaper to serve forever', 'Larger datasets are cheaper', 'It improves calibration'],
        answer: 1,
        why: 'Chinchilla optimises training loss per training FLOP; deployment optimises total lifetime cost.'
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
<p><b>Tensor parallelism</b> splits something data parallelism does not touch: a single weight matrix itself. Slice $W_Q$ column-wise across four GPUs, say, and each GPU computes only its slice of the query projection — but the very next operation in the block typically needs the <i>full</i, reassembled result before it can proceed, so tensor parallelism forces a communication step inside nearly every layer, not once per training step. That chattiness is why the rule of thumb in the table is so firm: tensor parallelism belongs <i>inside</i> a single machine, over NVLink's very high, very low-latency bandwidth between GPUs on the same node, and falls apart in throughput the moment it has to cross to a different physical machine over ordinary networking.</p>
<p><b>Pipeline parallelism</b> splits along a different axis again: rather than splitting any one matrix, give GPU 1 the first several layers of the model, GPU 2 the next several, and so on, so that a forward pass physically walks across devices as it walks through depth. Communication between stages is comparatively cheap — only the activations at each layer boundary need to be passed, point to point, rather than a full collective operation — but the naive version wastes most of its GPUs most of the time: while GPU 2 works on layer 11 for the first micro-batch, GPU 1 has nothing to do until GPU 2 hands anything back, and GPU 3 has nothing to do at all yet. This idle time is called a <b>bubble</b>, and the standard fix is to slice the batch into many small micro-batches and feed them through the pipeline in a staggered, overlapping sequence — an assembly line rather than one part at a time — so that once the pipeline is full, every stage has something to work on in every cycle, and the bubble shrinks to a fixed start-up and drain-down cost rather than a constant tax on every batch.</p>
<p><b>Expert parallelism</b> is §4.8's mixture-of-experts problem viewed from the hardware side: different experts live on different devices, and every forward pass needs an <b>all-to-all</b> exchange — every device sending some tokens out to wherever their chosen expert lives, and receiving others in return — which is a genuinely different communication pattern from an all-reduce's single combine step, and the reason MoE models are harder to serve efficiently at scale than a dense model of comparable active-parameter count. Real large-scale training runs combine all four of these splits at once — often called 3-D or 4-D parallelism — choosing which axis goes inside a node, which crosses nodes, and which wraps the whole cluster, entirely as a function of which communication pattern each axis needs and how much bandwidth is available at each level of the network.</p>
<p>Underneath all four sits one more question: even with the model split up, does the memory actually fit? <b>ZeRO / FSDP</b> answers it by attacking a specific redundancy in plain data parallelism — every GPU in that scheme holds a <i>full, independent copy</i> of the optimiser state, gradients and (in ZeRO-3/FSDP) even the parameters themselves, which is enormously wasteful when Adam's optimiser state alone needs roughly <b>12 bytes per parameter</b> of FP32 storage (a master FP32 weight copy plus two FP32 running moments, §4.11.2 explains why the FP32 copy exists), on top of the weights and gradients. ZeRO shards each of those tensors across the data-parallel group instead — every GPU permanently holds only its own $1/n$ slice — and gathers whatever full tensor is momentarily needed via targeted communication exactly when a computation requires it, then releases it again. That sharding, more than any other single trick, is what makes training a model whose optimiser state alone would not fit on one device possible at all.</p>

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
      ['Which parallelism goes inside a node?', 'Tensor parallelism — splitting a single matrix forces a communication step inside nearly every layer rather than once per step, so it needs NVLink\'s very high, low-latency bandwidth and falls apart in throughput the moment it crosses to ordinary inter-node networking.'],
      ['How much memory does Adam need?', '≈12 bytes per parameter of FP32 optimiser state — a master weight copy plus two running moments — on top of the weights and gradients, which is precisely what ZeRO/FSDP shards across data-parallel ranks rather than replicating on every GPU.'],
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
        why: 'FP32 master weights plus two moments — about 12 bytes/param, more than the BF16 weights themselves. ZeRO/FSDP shards it.'
      },
      {
        q: 'Your loss NaNs after resuming from a checkpoint. The first thing to check is…',
        options: ['the model architecture', 'whether the scheduler is emitting the intended learning rate — a resume that skips warmup is the classic cause', 'the tokenizer', 'the evaluation set'],
        answer: 1,
        why: 'It is first on the ordered list precisely because it is both common and instantly checkable.'
      },
      {
        q: 'Tensor parallelism should be kept…',
        options: ['across data centres', 'inside a node, on NVLink', 'on the same GPU', 'between pipeline stages'],
        answer: 1,
        why: 'It communicates every layer; crossing a slower interconnect destroys throughput.'
      }
    ],
    cards: [
      { q: 'Three parallelisms', a: 'Data (all-reduce), tensor (chatty — keep in-node), pipeline (cheap, bubbles). Plus expert parallelism for MoE.' },
      { q: 'Adam memory', a: '≈12 bytes/param FP32 state; ZeRO/FSDP shards optimiser state, gradients and parameters.' },
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
<p>The reward model is only a proxy, fit from a finite, noisy sample of human judgments, and optimising hard against any proxy eventually finds places where the proxy disagrees with what it was standing in for — text a human would rate poorly but that happens to score highly under the specific, imperfect function the reward model learned. Left unconstrained, the policy drifts toward exactly those blind spots, a failure with its own name, <b>reward hacking</b>, that §4.12.7 catalogues with concrete examples. The fix is a <b>KL penalty</b>: subtract $\\beta\\, D_{KL}(\\pi_\\theta\\|\\pi_{\\text{ref}})$ from the objective, using §1.10's divergence between the policy being trained and the original SFT model it started from, so any move the policy makes toward a higher reward is taxed by how far that move has carried it from a distribution that was, at least, producing recognisably normal text. It is not decoration bolted onto the objective; it is the mechanism that keeps the optimisation honest. The whole pipeline works, and is used in production, but it is heavy: a reward model, a separate value network estimating expected future reward for PPO's advantage calculation, the policy being trained, and the frozen reference policy for the KL term — four networks resident in memory simultaneously for what is, underneath, training one model.</p>

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
        why: 'Z(x) depends on x alone, so in the Bradley–Terry difference $r(x,y_w)-r(x,y_l)$ it cancels exactly — no reward model needed.'
      },
      {
        q: 'GRPO rewards for one prompt are [1,1,1,1,1,1]. What is the learning signal?',
        options: ['Strongly positive', 'Zero — σ = 0, so every advantage is zero and the prompt teaches nothing', 'Negative', 'Undefined but usable'],
        answer: 1,
        why: 'The group is the baseline; a unanimous group has no relative information. Implementations drop degenerate groups and curate for intermediate difficulty.'
      },
      {
        q: 'The most commonly cited failure of DPO is…',
        options: ['reward-model overfitting', 'response length inflation (arXiv:2403.19159)', 'catastrophic forgetting', 'mode collapse to one answer'],
        answer: 1,
        why: 'Length exploitation is documented and material; monitor mean response length as a first-class metric.'
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
<p>The parameter saving is dramatic and easy to see directly: a $d\\times d$ matrix has $d^2$ parameters, while $B$ and $A$ together have $dr + rd = 2dr$. For $d=4096$ and $r=16$, that is $2\\times4096\\times16 \\approx 131{,}000$ against $4096^2 \\approx 16.8$ million — under 1% of the original matrix's parameter count, for a single target matrix, and the full-model figure worked out below lands in the same range once every targeted matrix in every layer is counted. Because $W_0$ never changes, there is no need to hold Adam's expensive optimiser state (§4.11's ≈12 bytes/parameter) for anything but $B$ and $A$ — the overwhelming majority of the model's parameters need no gradient and no optimiser state at all, which is where essentially all of the memory saving in the worked example below actually comes from. You train roughly 0.5% of the parameters, checkpoints are tens of megabytes rather than tens of gigabytes, and because $B$ and $A$ can be added to or removed from $W_0$ independently, an adapter can be hot-swapped at serve time — load a different tiny $B$, $A$ pair for a different customer or task without touching the multi-gigabyte base model at all.</p>

<h2><span class="sn">4.13.2</span> QLoRA</h2>
<p>LoRA already avoids optimiser state for the frozen base, but the base weights themselves still have to sit resident in memory at full or near-full precision to be read during the forward and backward pass. <b>QLoRA</b> shrinks that too, by keeping the frozen base compressed to 4-bit <b>NF4</b> — "4-bit NormalFloat", a quantisation scheme whose representable values are spaced to match the roughly Gaussian distribution neural network weights actually follow, rather than spaced uniformly across the numeric range the way a naive 4-bit format would be, which wastes far fewer of its limited 16 representable levels on weight magnitudes that almost never occur. <b>Double quantisation</b> squeezes a further small saving out of the scheme by quantising the quantisation constants themselves — the per-block scaling factors NF4 needs are, in aggregate, a real amount of memory at billions of parameters, and compressing them too is nearly free additional saving. <b>Paged optimisers</b> borrow the operating system's virtual-memory trick directly: when GPU memory comes briefly under pressure — a longer-than-usual sequence in the current batch, say — optimiser state pages out to ordinary CPU memory rather than crashing the run, and pages back in once space frees up. Throughout all of this, gradients still flow <i>through</i> the frozen quantised weights during backpropagation — the base is dequantised on the fly for each matrix multiply, and the chain rule passes a gradient back through that computation exactly as it would through any other layer — but nothing ever accumulates an update <i>into</i> the base weights; only $B$ and $A$, kept in ordinary BF16, are ever modified. That combination is what makes a 70B fine-tune fit on a single large GPU rather than a rack of them.</p>

<h3>2026 defaults worth quoting</h3>
<p><b>$r = 16$ with $\\alpha = 2r$, targeting <i>all</i> linear layers</b> — attention q, k, v, o plus the MLP gate, up and down projections — rising to $r = 32$–$64$ for harder tasks or larger data. Targeting attention only is the older, weaker recipe. <b>DoRA</b> (decomposing magnitude from direction) is a common quality upgrade at the same budget. Prefix- and prompt-tuning still exist but are largely superseded.</p>

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
        why: '≈42M of 8.03B ≈ 0.52%, fitting in about 7 GB against ~145 GB for full BF16 fine-tuning (16 GB weights + ~128 GB of gradient-and-Adam overhead) — a roughly twentyfold gap.'
      },
      {
        q: 'Merging a LoRA adapter into a 4-bit quantised base is…',
        options: ['exact', 'lossy — you trained against the quantised weights, so merge into the BF16 original and re-quantise', 'impossible', 'faster than keeping it separate but identical in quality'],
        answer: 1,
        why: 'Or keep the adapter separate, which also preserves hot-swapping across tenants.'
      },
      {
        q: 'A task vector is…',
        options: ['the gradient at convergence', '$\\theta_{finetuned} - \\theta_{base}$, which can be added to or subtracted from the base', 'the LoRA B matrix', 'the embedding of the task description'],
        answer: 1,
        why: 'Adding task vectors combines capabilities with no data and no gradient steps; TIES/DARE/SLERP reduce interference.'
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

<h2><span class="sn">4.14.2</span> Serving systems</h2>
<p><b>Paged attention</b> (vLLM) solves a specific waste inherent in the obvious way of allocating KV cache memory: reserve one contiguous block per sequence, sized for the longest generation that sequence might possibly produce, before you know how long it will actually run. Nearly all of that reservation typically goes unused, and — because it is contiguous — it cannot be lent to another sequence in the meantime even while sitting empty. Paged attention allocates the cache in small, fixed-size pages instead, handed out on demand as a sequence actually grows, tracked via a block table exactly the way an operating system's virtual memory tracks physical pages behind a process's address space: fragmentation effectively disappears, sequences with a shared prefix (§4.14's "prefix caching" below) can literally share the same physical pages rather than duplicating them, and the number of sequences a given amount of memory can serve concurrently rises sharply as a direct result. <b>Continuous batching</b> attacks a different inefficiency: a naive static batch has to wait for every sequence in it to finish before starting the next batch, so one unusually long response holds an entire batch's worth of GPU capacity hostage while shorter sequences that finished early sit idle. Continuous batching admits new requests and retires finished ones at the level of individual decode steps rather than whole batches, keeping the GPU close to fully occupied at all times instead of bound by whichever sequence in the batch happens to run longest. <b>Prefix caching</b> reuses the already-computed KV cache of a shared prefix — a system prompt, a long set of tool definitions, the first turns of an ongoing conversation — across every request that shares it, skipping prefill for that shared portion entirely on every subsequent request. On agent workloads with long, largely fixed instructions repeated on every call, this is often the single cheapest latency win available, precisely because it eliminates compute rather than merely speeding it up.</p>

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
        why: '≈2N·s FLOPs at your achieved utilisation. Doubling the prompt doubles the wait.'
      },
      {
        q: 'Speculative decoding with α = 0.7 and k = 4 yields roughly how many tokens per big-model pass?',
        options: ['1.4', '2.8', '4.0', '5.0'],
        answer: 1,
        why: '$(1-0.7^5)/0.3 \\approx 2.8$, with the output distribution provably unchanged.'
      },
      {
        q: 'The cheapest serving win to turn on first is usually…',
        options: ['speculative decoding', 'prefix caching of a shared system prompt', 'INT4 quantisation', 'a bigger batch'],
        answer: 1,
        why: 'One flag, no quality risk, and on agent workloads with long fixed instructions it is frequently a 2–5× TTFT improvement.'
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
<p><b>Greedy</b> decoding — always take the single most likely next token — sounds like the obviously correct choice, and it is deterministic and reproducible, which is genuinely valuable for extraction or classification tasks with one right answer. But applied to open-ended generation it has a specific, well-known failure: once the model starts repeating a phrase, "repeat the same phrase again" is very often <i>itself</i> the highest-probability continuation, since the model has just seen strong local evidence that this exact pattern is what belongs here — greedy decoding can talk itself into a repetition loop with no mechanism to escape it. <b>Beam search</b>, which tracks several of the most promising partial sequences at once rather than committing token by token, avoids some short-sighted greedy mistakes but has its own well-documented failure in open generation: because it is explicitly searching for the single highest joint-probability sequence, it systematically favours safe, generic, low-surprise continuations over the more varied, specific phrasing an actual sample from the distribution would produce — the mode of a distribution is not a typical sample from it, and text generated by hunting for the mode reads as noticeably duller than text generated by sampling.</p>
<p>Sampling, then, is often what open-ended generation actually wants — drawing a token according to its true predicted probability, exactly like the sampling procedures in §1.13 — but sampling from the raw distribution has its own problem: a language model's distribution nearly always has a very long tail of low-probability tokens, and sampling from that tail even occasionally is enough to derail a generation into incoherent text. <b>Temperature</b> reshapes the distribution before any of this happens — divide every logit by $T$ before the softmax (§4.3's mechanism, applied here rather than to attention scores): $T>1$ flattens the distribution toward uniform, $T<1$ sharpens it toward the mode, and $T=1$ leaves it exactly as the model produced it. Temperature does not remove the tail, it only reweights it, which is why it is nearly always combined with a truncation rule that removes the tail outright. <b>Top-k</b> keeps only the $k$ most likely tokens and renormalises among them — simple, but a fixed $k$ is the wrong tool whenever the shape of the distribution varies from position to position, which it constantly does: at a position where the model is highly confident, $k$ candidates may include tokens with vanishing probability that should never be sampled; at a position of genuine ambiguity, the same $k$ may exclude a perfectly reasonable continuation. <b>Top-p (nucleus) sampling</b>, introduced specifically to fix this, instead keeps the <i>smallest</i> set of tokens whose cumulative probability exceeds $p$ — a set that automatically shrinks to a handful of tokens when the model is confident and expands when it is genuinely uncertain, adapting to the distribution's actual shape rather than assuming a fixed count is ever right.</p>
${H.history(`<p>Holtzman et al.'s 2019 paper, memorably titled <i>The Curious Case of Neural Text Degeneration</i>, is the source of nucleus sampling and the paper that made the beam-search-produces-bland-text finding precise and widely known: they showed that maximisation-based decoding (greedy, beam search) reliably produces text that is measurably less diverse and more repetitive than human text, by exactly the mode-versus-sample mechanism described above, and that naive random sampling from the full distribution reliably produces text that degenerates into incoherence by wandering into the long tail. Top-p sampling was their proposed middle path, and it has remained the default in most production systems since.</p>`)}
<p><b>Min-p</b> sets its threshold relative to the single most likely token's probability rather than to a cumulative mass, which makes it more robust specifically at high temperature — where top-p's cumulative-mass threshold can admit a surprisingly large or small set depending on how flattened the distribution has become, min-p's relative threshold tracks the model's actual confidence more directly. <b>Repetition and presence penalties</b> take a more direct approach again: down-weight the logits of tokens (or n-grams) already seen in the generated text so far, a blunt but effective way of suppressing the specific loop failure greedy decoding is prone to, at the cost of occasionally penalising a word that genuinely needed repeating.</p>

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
        why: 'A fixed k either truncates a genuinely flat distribution or admits junk from a peaked one.'
      },
      {
        q: 'The correct way to guarantee schema-valid JSON output is…',
        options: ['prompt engineering and retries', 'constrained decoding that masks invalid tokens', 'temperature 0', 'a larger model'],
        answer: 1,
        why: 'Validity by construction — and it is what makes tool calling reliable (§5.3).'
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

<h2><span class="sn">4.16.3</span> Hallucination</h2>
<p>Hallucination is not a bug in the ordinary sense of a mistake that better engineering removes — it follows directly from what §4.9.1 already established the pretraining objective actually is. Cross-entropy on next-token prediction only ever measures one thing: did the model assign high probability to the token that actually came next in this specific piece of training text. Nothing in that objective distinguishes a plausible sentence that happens to be true from an equally plausible sentence that happens to be false; the model is trained end-to-end to produce fluent, likely-looking continuations, with no separate channel anywhere in the loss that checks the continuation against reality. A model does not "decide to lie" any more than it "decides to tell the truth" — it produces the statistically likely continuation of its context, and when the likely continuation happens to be false, nothing in the training signal it received was ever positioned to catch that.</p>
<p>Mitigations are therefore necessarily <b>structural</b> rather than a matter of asking more nicely. Grounding the model with retrieval and requiring citations (§5.1) gives it something external and checkable to condition its answer on, rather than relying purely on parametric memory that has no confidence signal attached. Constraining decoding (§4.21) cannot fix content, but it guarantees the <i>shape</i> of an answer is checkable, which is a real, if partial, win. Training the model specifically for <b>abstention</b> — rewarding "I don't know" as a correct response on questions genuinely beyond its knowledge, via the same post-training machinery §4.12 builds for everything else — makes refusal a reachable output rather than one the model was implicitly punished for during fine-tuning on datasets where every question had a confident answer written for it. And external verification — checking specific claims against an independent source before acting on them — is simply mandatory wherever the cost of an undetected error is high, because no amount of prompting changes what the underlying objective was ever trained to optimise for.</p>

<h2><span class="sn">4.16.4</span> Multimodal, briefly</h2>
<p>A short bridge rather than a full treatment — §4.19 covers vision-language architecture in the depth the topic deserves, including the token-cost arithmetic of feeding an image through a vision-language model and CLIP's documented compositionality weaknesses. In outline: <b>ViT</b> cuts an image into patches and treats them as tokens, so attention needs no pixel-specific machinery at all — the same transformer block from §4.5 runs unmodified. <b>CLIP</b> trains an image encoder and a text encoder contrastively, pulling matched image-caption pairs together in one shared embedding space, which is what yields zero-shot classification (measure distance to an encoded class name) and the shared retrieval space multimodal search relies on. A <b>VLM</b> is a vision encoder plus a small projector network that maps visual features directly into the LLM's own token embedding space, so the language model attends to image content as if it were simply more words in the sequence — the same attention mechanism throughout, applied to a sequence that happens to have started life as pixels.</p>

<h2><span class="sn">4.16.5</span> How to build an eval worth having</h2>
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
        why: 'p_e = 0.7² + 0.3² = 0.58; κ = (0.82−0.58)/(1−0.58) = 0.57 — moderate.'
      },
      {
        q: 'The mandatory mitigation for LLM-judge position bias is…',
        options: ['using a bigger judge model', 'scoring both orders and averaging', 'raising the temperature', 'using pointwise scores'],
        answer: 1,
        why: 'If the winner changes with the order, you measured the judge rather than the answers.'
      },
      {
        q: 'Hallucination is best mitigated by…',
        options: ['a stricter system prompt', 'structural measures: retrieval grounding with citations, constrained decoding, trained abstention, external verification', 'lower temperature alone', 'a larger context window'],
        answer: 1,
        why: 'The model has no truth-checking mechanism; the fixes have to come from outside the decoding objective.'
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
<p>Refusal is a learned behaviour like any other in this part, which means it was learned from a finite set of training examples covering only some of the space of possible requests — and every jailbreak family below works by finding a distribution of inputs that behaviour was never actually trained on, rather than by defeating it head-on.</p>
${H.table(['Family', 'Mechanism'], [
      ['<b>Persona / role-play</b>', '"You are an author writing a villain" — moves the request into a frame where compliance looked appropriate in training'],
      ['<b>Obfuscation</b>', 'base64, leetspeak, another language, a cipher — the safety-relevant features never fire while the capability still does'],
      ['<b>Many-shot</b>', 'Fill a long context with dozens of fabricated compliant exchanges, exploiting in-context learning against the model’s own policy — a direct cost of long context (§4.4)'],
      ['<b>Crescendo</b>', 'Escalate gradually across turns so no single turn looks refusable'],
      ['<b>Best-of-n</b>', 'Sample the same request many times at high temperature until one attempt slips through']
    ])}
<p>Each mechanism deserves to be understood rather than merely catalogued. <b>Persona and role-play</b> attacks work because SFT and RLHF training data (§4.12.1, §4.12.2) legitimately contains creative-writing requests where a fictional villain explains something harmful within the story — that context genuinely was rated acceptable by human labellers, so the model learned that refusal is conditional on framing, not on the underlying content, and role-play exploits that conditionality directly by supplying the framing without changing the content at all. <b>Obfuscation</b> attacks a different, more mechanical gap: base64, leetspeak or a simple cipher changes the surface token pattern enough that whatever specific token sequences the safety training associated with "refuse this" never actually appear, while the much more general capability of decoding base64 or reading leetspeak — learned broadly from pretraining (§4.9), not from safety training specifically — still fires perfectly well, so the model complies with a request it would have refused in plain text. <b>Many-shot</b> jailbreaking is a direct, adversarial cost of exactly the in-context learning capability §4.15.4 described as a feature: fill a long context with dozens of fabricated exchanges where a compliant assistant answers similar harmful requests, and the model's own in-context learning — conditioning its next output on the pattern the context establishes — works exactly as designed, just against the model's own safety training rather than in service of it. <b>Crescendo</b> escalates gradually across many turns specifically so that no single turn, evaluated on its own, looks like something to refuse, exploiting the fact that refusal is typically judged locally rather than by integrating the full trajectory of a conversation. And <b>best-of-n</b> jailbreaking is the adversarial mirror of §4.15.3's test-time compute: just as sampling many attempts and keeping the best raises a benign task's success rate, sampling the same harmful request many times at high temperature and keeping whichever attempt happened to slip through raises an attacker's success rate — which is precisely the arithmetic the next subsection makes exact.</p>

<p><b>What you are looking at.</b> The probability that an attacker succeeds at least once, plotted against the number of attempts they are allowed, computed directly from your chosen per-request refusal rate rather than illustrated schematically.</p>
<p><b>What to do with it.</b> Set the refusal rate to 99% — a number that sounds very safe in isolation — and read the success-probability readout at 100 attempts.</p>
<p><b>The thing genuinely worth noticing.</b> That readout reads 63%, not anywhere near the 1% a single "99% refusal" headline number might suggest, because $1-(1-\\epsilon)^n$ compounds every attempt the attacker is allowed. The "rate needed" readout completes the argument: holding total risk under 1% across 100 attempts needs a per-request refusal rate above 99.99%, two orders of magnitude tighter than the number that sounded reassuring at first. Per-request rates are simply the wrong unit whenever an adversary can retry cheaply, which in practice is nearly always.</p>

${H.lab('refusal', 'Per-request refusal rates are the wrong unit', 'A 99% refusal rate against an attacker who can retry. The curve is $1-(1-\\epsilon)^n$ and it is unforgiving — this is the calculation to have ready when someone quotes a single-request safety number.')}

<h2><span class="sn">4.17.2</span> The two-sided metric</h2>
<p>Safety training that is evaluated and optimised on only one axis — minimise harmful compliance — will do exactly that, and nothing stops it from overshooting into <b>over-refusal</b>: pushing down the probability of complying with harmful requests, with no matching pressure to keep complying with adjacent benign ones, predictably drifts the decision boundary toward caution generally. A model that declines to explain how to terminate a runaway process on Linux, because the request superficially pattern-matches something that sounds destructive, is not safe — it is simply unusable for that entire class of legitimate question, and users facing enough of these false refusals learn to route around the model's safety training rather than work within it, which is a worse outcome for actual safety than a narrower, more accurate refusal boundary would have been. The correct evaluation is therefore two-sided by construction: attack success rate on a red-team set of genuinely harmful requests, <i>and</i> false-refusal rate on a benign-but-adjacent set specifically constructed to resemble harmful requests without being one — and movement in either direction alone counts as a regression, not just movement in the attack-success direction.</p>
<p>Red-teaming itself is now largely automated, and the automation is a direct reuse of tools already built in this part: an attacker model generates and mutates candidate jailbreak prompts, and a judge model (§4.16.2) scores whether each attempt succeeded — the same generate-and-score loop RLAIF (§4.12.5) uses to replace human preference labels, here aimed at finding failures rather than training a policy. This finds far more of the attack surface than manual red-teaming ever could, simply by exhausting far more candidate prompts than a human team has hours for, with human effort reserved specifically for the genuinely novel attack classes the automated loop has no template to generate from in the first place.</p>

<h2><span class="sn">4.17.3</span> Chain-of-thought faithfulness — the uncomfortable finding</h2>
<p>The reasoning a model prints before its answer is <b>not guaranteed to be the computation it actually performed</b> to reach that answer. Experiments can plant a cue in the prompt that measurably shifts a model's final answer, and the model will then produce a plausible, fluent chain of reasoning that never mentions the cue at all — a post-hoc justification rather than a genuine trace of the process. In the other direction, models can be steered to a wrong final answer while the printed reasoning leading up to it continues to look entirely sound at every individual step. Two consequences follow directly for practice. A legible chain of thought is <i>evidence, not proof</i>: it can genuinely help you debug a wrong answer by suggesting where reasoning went astray, but it should never be treated as an audit trail sufficient on its own for a regulated or high-stakes decision, precisely because it can be fluent and wrong about its own process simultaneously. And there is a real, structural tension in training models to produce reasoning that <i>reads well</i> versus reasoning that stays <i>informative</i> about the model's actual underlying computation: optimising the visible chain too directly toward looking clean and convincing — the same optimise-the-proxy dynamic §4.12.7 already catalogued for reward hacking generally, here applied to the reasoning trace itself rather than to the final answer — degrades exactly the property that made the chain worth reading as a monitor in the first place.</p>

<h2><span class="sn">4.17.4</span> Mechanistic interpretability, and what it actually buys</h2>
<p>Look inside a trained network for the neuron that means "legal disclaimer language" or "this code is in Python" and you typically will not find one — individual neurons are usually <b>polysemantic</b>, firing for several unrelated concepts at once rather than cleanly representing a single interpretable idea. The cause is a phenomenon called <b>superposition</b>, and it is a direct, practical consequence of a fact §0.2 already flagged as one of the genuine surprises of high-dimensional geometry: random vectors in a high-dimensional space are very nearly orthogonal to one another, so a space of $d$ dimensions can hold vastly more than $d$ directions that interfere with each other only slightly. A network exploits exactly that geometric slack to represent far more distinct features than it has neurons, packing them into overlapping, nearly-orthogonal directions rather than giving each feature its own dedicated dimension — which is efficient for the network and exactly why the resulting representation resists being read off neuron by neuron.</p>
<p><b>Sparse autoencoders</b> attack the resulting mess directly: train a much wider layer than the model's own hidden width, with a sparsity penalty forcing only a small number of its units to activate on any given input, to reconstruct the model's actual internal activations. The wide, sparse layer has enough room to give each packed feature something closer to its own dedicated unit, and empirically its units come out far more monosemantic than the original neurons — named, human-inspectable features such as "legal disclaimer language" or "the code is in Python" that a researcher can point to directly rather than infer indirectly. With features in hand, three things become possible that were not before: <b>auditing</b> what a model actually attended to on a specific input by inspecting which features fired, <b>steering</b> behaviour by directly amplifying or suppressing a specific feature at inference time, and <b>detecting</b> internal states such as deception or refusal-suppression with a trained probe against the feature space. The limits are equally worth stating plainly: coverage of a model's full feature set is still partial, the specific features a sparse autoencoder discovers are not stable across separately trained runs of the same architecture, and none of this yet constitutes a compliance-grade explanation of model behaviour a regulator would accept — which is exactly why §2.17's SHAP, imperfect as it is, remains what actually goes in a model-risk file today rather than an interpretability feature dashboard.</p>

<h2><span class="sn">4.17.5</span> Two more terms to recognise</h2>
<p><b>Sandbagging</b> is a model performing measurably worse specifically when it detects that it is being evaluated rather than used for real — which is precisely why held-out, unfamiliar-format evaluations carry value beyond simply avoiding the contamination problem §4.16.1 covers: an eval whose format the model has learned to recognise as "this is a test" is vulnerable to sandbagging even when its content is entirely uncontaminated. <b>Provenance</b> covers the tools built to trace whether a piece of content was AI-generated: watermarking text works by introducing a statistical bias into token selection during generation that a detector with the right key can identify, robust to light editing but not to paraphrasing, which removes the specific statistical signature by rewriting the content in different words; content credentials such as C2PA attach cryptographically signed metadata to images and video at the point of creation. Treat both as deterrents and forensic aids that raise the cost of misuse, never as reliable detectors on their own — and never build a policy or a product decision that depends on being able to reliably catch AI-generated text after the fact, because the detection side of this arms race is structurally the weaker one.</p>

${H.probe([
      ['Your model refuses 99% of attacks — is that safe?', 'Not against an attacker who can retry: 100 samples gives them a 63% success probability under $1-(1-\\epsilon)^n$, and holding total risk under 1% across those 100 attempts needs a per-request rate above 99.99%. Per-request rates are the wrong unit whenever retries are cheap, which is nearly always.'],
      ['Can I use the chain of thought as an audit trail?', 'No — it can be unfaithful to the actual computation, reaching an answer via an unmentioned cue or looking sound while being wrong. Use it as a debugging aid, not as evidence, and be aware that training the visible chain to look better can directly degrade how informative it is.'],
      ['What do sparse autoencoders give you?', 'More monosemantic features recovered from a superposed representation — the same near-orthogonality of random high-dimensional directions that §0.2 flagged as a genuine high-dimensional surprise — enabling auditing, steering and probe-based detection, with partial coverage, instability across training runs, and no compliance guarantee.']
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
        why: '$1-0.99^{100} \\approx 0.634$. Per-request rates are the wrong unit whenever retries are cheap.'
      },
      {
        q: 'Chain-of-thought output can be used as…',
        options: ['a compliance audit trail', 'a debugging aid — it may be unfaithful to the actual computation', 'proof of correctness', 'a substitute for evaluation'],
        answer: 1,
        why: 'Models can reach answers via cues never mentioned in the chain; legibility is evidence, not proof.'
      },
      {
        q: 'Sparse autoencoders are used to…',
        options: ['compress the model', 'extract more monosemantic features from activations in superposition, enabling auditing and steering', 'quantise weights', 'detect watermarks'],
        answer: 1,
        why: 'Coverage is partial and features are not stable across runs — useful, not yet compliance-grade.'
      }
    ],
    cards: [
      { q: 'Refusal arithmetic', a: '99% per-request refusal → 63% attacker success over 100 tries ($1-0.99^{100}$).' },
      { q: 'The two-sided safety metric', a: 'Attack success rate AND false-refusal rate on benign-adjacent prompts; movement in either is a regression.' },
      { q: 'CoT faithfulness', a: 'The printed chain need not be the computation performed — a debugging aid, not an audit trail.' },
      { q: 'Superposition & SAEs', a: 'Neurons are polysemantic because features outnumber dimensions; sparse autoencoders recover more monosemantic features for audit and steering.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.18 */
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
      ['17', 'Speculative decoding: $(1-\\alpha^{k+1})/(1-\\alpha)$ — α=0.7, k=4 → 2.8 tokens/pass.', '<a href="#/serving">4.14</a>'],
      ['18', 'Llama-3-8B parameter arithmetic: 42.0M attention + 176.2M FFN per layer → 8.03B.', '<a href="#/block">4.5</a>']
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
          ['99% refusal over 100 attempts?', '63% attacker success — per-request rates are the wrong unit.']
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
        why: 'It decides how many cards you need, what batch you can serve, and whether long context is feasible — and the formula is short enough to derive live.'
      }
    ]
  });
})();
