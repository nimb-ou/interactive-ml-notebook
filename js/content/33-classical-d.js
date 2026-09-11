/* ============================================================
   PART 2 — Classical ML, continued: ensembles (2.16), Gaussian
   processes and Bayesian optimisation (2.21), self-supervision
   (2.22), active learning and transfer (2.23), ranking (2.24),
   online experimentation (2.25).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.16 */
  ML.section({
    id: 'ensembles', track: 'classical', num: '2.16', level: 2,
    title: 'Ensembles: why averaging works, and when it stops',
    lede: 'Bagging, boosting, stacking and plain averaging are four answers to one question — how do you spend a fixed compute budget on many models instead of one? The maths says exactly how much you get back, and it depends entirely on one number: how correlated the members are.',
    prereq: ['bias-variance'],
    related: ['trees', 'boosting', 'hyperparameters'],
    html: `
<p>A fraud team with fifty spare GPU-hours overnight makes a bet that sounds obviously correct: train fifty separate models on the same labelled data, average their fifty predictions, and collect whatever machine learning hands you for free. By morning, validation AUC has moved from 0.812 to 0.815. Someone points out that five models, trained the previous week over a single afternoon, had already reached 0.812 on their own — so five models bought the entire 0.812, and the other forty-five, ten times the compute, bought three-thousandths of a point on top. Nothing was implemented wrong. The team simply ran head-first into the one number that decides how far averaging can take you, and nobody on the team had been watching it.</p>

<h2><span class="sn">2.16.1</span> The one formula</h2>

<p>Every method in this section — bagging, boosting, stacking, a plain unweighted vote — eventually comes down to combining several models' outputs into one number, however it gets there. So before comparing the methods, it is worth being precise about what combining buys you, because the honest answer has a hard limit built into it that no method escapes.</p>

<p>Take $M$ predictors $f_1,\\ldots,f_M$, each one unbiased and with the same individual variance $\\sigma^2$ — read $\\sigma^2$ as how much any single model's prediction wobbles from one training run to the next, exactly the variance §2.2 built for a single model. Assume further that any two of them share the same pairwise correlation $\\rho$ (the Greek letter rho) between their errors: how much one model's mistake on a given example tends to predict another model's mistake on that same example. The average prediction, $\\bar f = \\frac1M\\sum_m f_m$, then has variance</p>

$$\\mathrm{Var}(\\bar f) = \\frac{1}{M^2}\\left[M\\sigma^2 + M(M-1)\\rho\\sigma^2\\right] = \\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$$

<p>§1.3 derives this line by line, starting from nothing more than the definition of variance and the fact that variances of correlated quantities do not simply add — a cross term survives whenever the correlation is non-zero. This section takes that destination as given and spends its time on the two questions that actually decide what you do next, so read the right-hand side as two separate pieces rather than one formula, because the two pieces behave in completely different ways as $M$ grows.</p>

<p>The second term, $\\frac{1-\\rho}{M}\\sigma^2$, carries $M$ in its denominator. Raise $M$ and this term falls steadily toward zero — train more models and this part of the variance genuinely, unconditionally shrinks, however many decimal places you care to chase it. The first term, $\\rho\\sigma^2$, contains no $M$ anywhere in it. Nothing you do to the <i>count</i> of models moves this term by so much as a rounding error; it is exactly the same whether $M$ is 5 or 5,000. That asymmetry is the whole story of the overnight fraud run: the first five models were mostly buying the shrinking term, while the remaining forty-five were, structurally, buying almost nothing at all, because by then the ensemble's variance had already fallen most of the way to a floor that no further model could touch.</p>

${H.key('As $M\\to\\infty$ the variance tends to $\\rho\\sigma^2$, not zero. An ensemble of a thousand identical models is one model. Everything clever in ensembling is an attack on $\\rho$ — the count of members is almost beside the point.')}

<p>Put real numbers against the two terms and the floor stops being an abstraction. The table already carries the arithmetic; read it as three separate stories about where $\\rho$ comes from in practice, not as three rows to memorise.</p>

${H.table(['ρ', 'Variance at M=10', 'Variance at M=∞', 'Verdict'], [
      ['0.0', '0.10 σ²', '0', 'the textbook ideal, never observed'],
      ['0.2', '0.28 σ²', '0.20 σ²', 'a well-built random forest'],
      ['0.5', '0.55 σ²', '0.50 σ²', 'same algorithm, different seeds'],
      ['0.9', '0.91 σ²', '0.90 σ²', 'ten copies of one model — you have wasted 9× the compute']
    ], 'num')}

${H.worked('checking the table by hand, and reading what each row is actually describing', `
<p>At $M=10$: $\\rho=0.2$ gives $0.2+0.8/10=0.2+0.08=\\mathbf{0.28}\\,\\sigma^2$. $\\rho=0.5$ gives $0.5+0.5/10=\\mathbf{0.55}\\,\\sigma^2$. $\\rho=0.9$ gives $0.9+0.1/10=\\mathbf{0.91}\\,\\sigma^2$. Each is the boxed formula evaluated directly — no shortcut, just $\\rho$ plus $(1-\\rho)/M$, both times $\\sigma^2$.</p>
<p>$\\rho=0.2$ is roughly what a genuinely well-built random forest achieves, because feature subsampling (§2.7) forces each tree to look at a different slice of the feature space and actually knocks correlation down, not just the visible noise. $\\rho=0.5$ is what you get training the same algorithm on the same data five separate times with five different random seeds and changing nothing else — the seed moves which particular quirks of the training noise a model happens to fit, but it does nothing about the structural reason the five models agree with each other on everything else. $\\rho=0.9$ is ten copies of essentially one model: going from one model (variance $1.000\\,\\sigma^2$ exactly) to ten of them bought a fall to $0.910\\,\\sigma^2$, a genuine but modest 9% reduction for a tenfold increase in training and serving cost — which is the arithmetic sitting behind the fraud team's overnight result, and the reason "train more of the same thing" is rarely the right response to a stalled ensemble.</p>`)}

${H.analogy(`<p>Five analysts are asked, independently, to estimate a company's revenue next quarter from the same public filings. If the five estimates are genuinely independent — five different people noticing five different details, making five different small errors — averaging them cancels a great deal of that error, because one analyst's overestimate is somebody else's underestimate and the two partly cancel. But suppose all five secretly start from the same analyst-day guidance the company issued last month, and each one only nudges it slightly based on their own reading of the filings. Now the five estimates share a common error rather than five independent ones: if that original guidance was itself too optimistic, every one of the five inherits the same optimism, in the same direction, by the same rough amount. Averaging five numbers that are all wrong in the same direction does not make the average any less wrong — it only <i>looks</i> more trustworthy, because five confident people agreeing feels like independent confirmation when it is really one opinion, counted five times. That shared anchor is $\\rho$, and no amount of asking a sixth, seventh or eighth analyst to also start from the same guidance will ever wash it out.</p>`)}

<p>This is the entire practical consequence worth taking away from the formula: whenever an ensemble stops improving as you add members, the diagnostic question is never "how many models do I have" — it is "how alike are they, and what would it cost to make them less alike." Feature subsampling, different model families in a stack, and genuinely different data views all attack $\\rho$ directly; a different random seed on the same algorithm barely moves it, which is exactly why row two of the table above sits so much closer to the useless end of the range than row one.</p>

${H.lab('ensvar', 'Diversity beats quantity', 'Left: the formula, plotted. Right: real trees, really trained on bootstrap resamples of the same 2-D data, really averaged — with a knob for how much feature subsampling they get, which is the actual mechanism that lowers ρ.')}

<p><b>What you are looking at.</b> The left panel plots the variance-of-the-average formula against $M$, the number of members, running along the bottom. Four faint reference curves are drawn for $\\rho=0,0.2,0.5,0.9$ so you have fixed points of comparison, and a thick curve traces whatever $\\rho$ the slider is currently set to, with a dashed horizontal line marking that curve's own floor, $\\rho\\sigma^2$, and a dot sitting exactly at the $(M,\\ \\text{variance})$ pair the readout is reporting. The right panel is not a picture of the formula at all — it is real decision trees, genuinely bootstrap-resampled from the same two-moons dataset each time, genuinely averaged together, with the resulting decision boundary shaded directly over the actual data points.</p>

<p><b>What to do with it.</b> Set "features per split" to "2 of 2" — every tree sees both coordinates of every point, with nothing hidden from any of them — and drag $M$ from 1 up to 40. The single-tree and ensemble accuracy readouts stay close together throughout, because letting every tree see the complete picture keeps them highly correlated with one another: you are, in effect, retraining a very similar tree over and over on slightly different bootstrap samples. Now switch to "1 of 2", which blinds every other tree to one of its two coordinates entirely — half the trees can only ever split on $x_1$, the other half only on $x_2$ — forcing the trees that survive this handicap to specialise on whichever single axis they can still see, and watch the gap between single-tree and ensemble accuracy widen noticeably at the very same value of $M$.</p>

<p><b>The thing genuinely worth noticing.</b> Compare the left panel's $\\rho=0.9$ curve against its $\\rho=0.2$ curve at the same value of $M$, then compare "2 of 2" against "1 of 2" on the right at that identical $M$. The "2 of 2" trees behave like the high-$\\rho$ curve — they flatten out almost immediately, and pushing $M$ further barely moves the ensemble accuracy readout at all. The "1 of 2" trees behave like the low-$\\rho$ curve — still visibly improving deep into the slider's range, because forcing disagreement between trees is doing on the right exactly what a smaller $\\rho$ does on the left. Nobody built the two panels to match on purpose. They agree because they are the same fact, shown once as algebra and once as a forest of real trees you can watch disagree with itself.</p>

<h2><span class="sn">2.16.2</span> The four methods, and what each one is attacking</h2>
${H.table(['Method', 'Members', 'Trained', 'Attacks', 'Canonical example'], [
      ['<b>Bagging</b>', 'same algorithm, bootstrap resamples', 'in parallel, independently', 'variance', 'random forest (§2.7)'],
      ['<b>Boosting</b>', 'same algorithm, weak learners', 'sequentially, on residuals', '<b>bias</b>', 'XGBoost / LightGBM (§2.8)'],
      ['<b>Stacking</b>', 'different algorithms', 'base in parallel; a meta-model on out-of-fold predictions', 'both, by learning the weights', 'Kaggle-winning blends'],
      ['<b>Voting / blending</b>', 'anything', 'independently, combined by a fixed rule', 'variance', 'the two-line version that usually gets 80% of the gain']
    ])}
${H.intuition(`<p>Bagging and boosting look similar and are opposites. Bagging trains members that are each <i>as good as they can be</i> and hopes their mistakes disagree — it cannot reduce bias, because the average of $M$ biased models has the same bias: expectation is linear, so $\\mathbb{E}[\\bar f] = \\frac1M\\sum_m \\mathbb{E}[f_m]$ is just the shared bias again, whatever $M$ is. Boosting trains members that are each <i>deliberately weak</i> and each one is fitted to what the previous ones got wrong — so the bias falls with every round, and the variance is what you must now control (with shrinkage, subsampling and early stopping). This is why a random forest is robust to overfitting and gradient boosting is not.</p>`)}

<p>Put in the language of §2.16.1: bagging's whole mechanism is holding bias exactly where it is while attacking the $(1-\\rho)/M$ term through resampling — and a random forest goes one step further, attacking the $\\rho\\sigma^2$ floor itself through feature subsampling, precisely the mechanism the lab above lets you watch happen to real trees. Boosting does not touch $\\rho$ in this sense at all; each new tree is deliberately built to be <i>highly</i> correlated with whatever the ensemble still gets wrong, because chasing that remaining error is the entire point of adding it. Boosting's variance is instead controlled by shrinkage (a learning rate scaling down each new tree's contribution), subsampling rows and columns per round, and stopping early — none of which appear anywhere inside the $\\rho\\sigma^2$ formula, because that formula was built to describe parallel, roughly-independent averaging, not a sequence of models each one deliberately chasing the last one's mistakes.</p>

<h2><span class="sn">2.16.3</span> Stacking, done correctly</h2>

<p>Bagging and boosting both assume the right way to combine members is a fixed rule decided in advance — an unweighted average, or a running sum of sequential corrections. Stacking drops that assumption and hands the combining job to a learned meta-model, which decides how much to trust each base model rather than trusting all of them equally everywhere. Because it decorrelates by mixing genuinely different <i>algorithms</i> — a gradient-boosted tree and a regularised linear model get different examples wrong, for reasons rooted in what each family can even represent — rather than genuinely different data samples fed to the same algorithm, it can push $\\rho$ down further than resampling the same tree forty different ways ever could. Building it correctly needs exactly one piece of discipline, and getting that one piece wrong is the most common way a stack ends up quietly worse than its best single base model.</p>

${H.steps([
      'Split the training data into $K$ folds.',
      'For each base model, produce <b>out-of-fold</b> predictions: for every row, the prediction from the model that did not see it. This is the step people get wrong, and getting it wrong makes the meta-model learn "trust whichever base model is most overfitted".',
      'Train the meta-model on the matrix of out-of-fold predictions (plus, optionally, a few raw features).',
      'Refit each base model on all the training data for use at prediction time.'
    ])}
${H.pitfall('If the meta-model is trained on in-fold predictions, the base model that memorised hardest looks best, the blend weights are garbage, and cross-validated performance is optimistic by a wide margin. Use a linear or logistic meta-model with a strong penalty; anything fancier is nearly always noise-fitting.')}
${H.flag('Stacking is essentially absent from regulated production systems (banking, insurance, healthcare). Not because it does not work — because explaining, monitoring and validating a nested ensemble to a model-risk committee costs far more than the 0.4 points of Gini it wins. Say this out loud in an interview; it demonstrates that you can distinguish a leaderboard from a deployment.')}

<h2><span class="sn">2.16.4</span> Snapshot and implicit ensembles</h2>
${H.table(['Trick', 'How', 'Cost', 'Typical gain'], [
      ['Seed averaging', 'train $M$ times, different seeds, average', '$M\\times$ train and serve', '0.2–1 pt'],
      ['Snapshot ensembling', 'cyclic learning rate; save at each minimum', '1× train, $M\\times$ serve', '0.3–0.8 pt'],
      ['Stochastic weight averaging (SWA)', 'average the <i>weights</i> late in training', '1× train, <b>1× serve</b>', '0.2–0.6 pt'],
      ['Monte Carlo dropout', 'keep dropout on at inference, average $M$ passes', '1× train, $M\\times$ serve', 'uncertainty estimates, mostly'],
      ['Test-time augmentation', 'average predictions over augmented inputs', '$M\\times$ serve', '0.3–1 pt on images'],
      ['Model soups', 'average weights of models fine-tuned with different hyperparameters', 'many trains, 1× serve', 'competitive with true ensembling']
    ])}
${H.note('SWA and model soups are the interesting ones because they collapse back to a single set of weights — you get part of the ensemble benefit at exactly zero inference cost. That only works when the members lie in the same loss basin, which is why they fine-tune from a shared starting point rather than training from scratch.')}

<h2><span class="sn">2.16.5</span> Whether to ship it at all</h2>

<p>Every method in this section wins offline, reliably, by some margin. That margin is measured in a notebook, against a metric that does not know about latency, on-call rotas, or a model-risk committee's patience, and it is worth being honest about the gap between winning offline and winning in production before recommending an ensemble to anyone who will have to operate it. Five models behind one prediction is five times the inference latency if they run in sequence, five separate artefacts to version, monitor for drift (§2.18) and explain to a regulator (§2.17), and five separate failure modes — a stale feature store entry, a dependency upgrade, a silent version mismatch — any one of which now degrades the whole system rather than one component of it. Trading all of that for 0.3 points of AUC is not an obviously good deal, and "the leaderboard number went up" is not, on its own, an argument that survives the question "what does this cost to run."</p>

<p>None of this is an argument against ensembling. It is an argument for pricing it honestly: state the measured lift in the same breath as the added latency, the added monitoring surface and the added explanation burden, and let whoever owns the production system weigh the trade with real numbers on both sides rather than a single offline metric that only ever tells one side of the story. A two-model blend that is simple enough for one person to reason about end to end very often captures most of the available gain (the "voting/blending" row of the table above is exactly this), and knowing when to stop adding members is as much a part of this section as the formula that tells you why more members eventually stop helping.</p>

${H.probe([
      ['Why does a random forest subsample features at each split?', 'To lower $\\rho$. Bootstrap resampling alone leaves the trees highly correlated because one dominant feature is chosen first in nearly every tree; forcing a random subset at each split breaks that.'],
      ['You add 500 more trees and nothing improves. Why?', 'You are on the $\\rho\\sigma^2$ floor. More members cannot help; more diversity can — deeper feature subsampling, different algorithms, different feature representations.'],
      ['Bagging reduces variance. Does it reduce bias?', 'No. The average of $M$ equally biased models has the same bias. That is boosting’s job.'],
      ['When would you refuse to ship an ensemble?', 'When the latency budget, the monitoring burden or the regulatory explanation cost exceeds the measured lift — which, for a 0.3-point AUC gain, it usually does.']
    ], 'Stacking on in-fold predictions. It is the single most common ensembling bug, and it silently inflates every number downstream of it.')}`,
    labs: {
      ensvar: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'error correlation ρ', min: 0, max: .95, step: .01, value: .3, fmt: v => v.toFixed(2) },
          { k: 'M', label: 'members M', min: 1, max: 40, step: 1, value: 12, fmt: v => v },
          { k: 'feat', label: 'features per split (real trees)', min: 1, max: 2, step: 1, value: 1, fmt: v => v + ' of 2' },
          { k: 'depth', label: 'tree depth', min: 1, max: 6, step: 1, value: 4, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'var', label: 'variance of the average', cls: 'key' },
          { k: 'floor', label: 'floor as M→∞' },
          { k: 'single', label: 'single tree accuracy' },
          { k: 'ens', label: 'ensemble accuracy', cls: 'good' }
        ]);
        const data = Num.dataset('moons', 220, .35, 12);
        const test = Num.dataset('moons', 400, .35, 77);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const w1 = w * .46;
            const P = Viz.plot(ctx, w1, h, { xd: [1, 40], yd: [0, 1.05], pad: { l: 44, r: 10, t: 16, b: 38 } })
              .frame({ xlabel: 'members M', ylabel: 'variance (σ² = 1)' });
            P.clip(() => {
              [0, .2, .5, .9].forEach((r, i) => {
                P.fn(m => r + (1 - r) / m, { color: [T.c3, T.c1, T.c4, T.c2][i], width: 1.4, alpha: .45, n: 120 });
              });
              P.fn(m => st.rho + (1 - st.rho) / m, { color: T.c1, width: 3, n: 160 });
              P.hline(st.rho, { color: T.c2, dash: [4, 4], label: 'floor ρσ²' });
              P.dots([[st.M, st.rho + (1 - st.rho) / st.M]], { r: 5, color: T.c2, stroke: true, strokeWidth: 2 });
            });

            /* real bagged trees on the right */
            ctx.save(); ctx.translate(w1, 0);
            const ww = w - w1;
            const P2 = Viz.plot(ctx, ww, h, { xd: [-2.6, 2.9], yd: [-1.9, 2.3], pad: { l: 10, r: 10, t: 16, b: 30 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'the same trees, actually trained' });
            const M = Math.min(st.M, 24);
            const trees = [];
            for (let m = 0; m < M; m++) {
              const R = Num.rng(100 + m);
              const idx = Array.from({ length: data.X.length }, () => R.int(data.X.length));
              const Xb = idx.map(i => data.X[i]), yb = idx.map(i => data.y[i]);
              // feature subsampling is emulated by hiding a column from some trees
              const hide = st.feat === 1 ? (m % 2) : -1;
              const Xm = Xb.map(r => hide === 0 ? [0, r[1]] : hide === 1 ? [r[0], 0] : r);
              trees.push({ t: Num.tree(Xm, yb, { maxDepth: st.depth, minLeaf: 4 }), hide: hide });
            }
            const predictOne = (p, k) => {
              const q = trees[k].hide === 0 ? [0, p[1]] : trees[k].hide === 1 ? [p[0], 0] : p;
              return trees[k].t.predict(q);
            };
            const predictAll = p => { let s = 0; for (let k = 0; k < M; k++) s += predictOne(p, k); return s / M; };
            P2.clip(() => {
              Labs.boundary(P2, (x, y) => predictAll([x, y]), { step: 4, lo: 0, hi: 1, alpha: 96 });
              Labs.points(P2, data.X.slice(0, 120), data.y.slice(0, 120), { r: 2.6 });
            });
            ctx.restore();

            const acc = pred => test.X.filter((p, i) => (pred(p) > .5 ? 1 : 0) === test.y[i]).length / test.X.length;
            out({
              var: (st.rho + (1 - st.rho) / st.M).toFixed(3) + ' σ²',
              floor: st.rho.toFixed(3) + ' σ²',
              single: (acc(p => predictOne(p, 0)) * 100).toFixed(1) + '%',
              ens: (acc(predictAll) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.note(host, 'Set <b>features per split</b> to "2 of 2" — every tree now sees everything, so they agree with each other, and the ensemble barely beats a single tree. Drop it to "1 of 2" and the trees are forced to disagree: individual accuracy falls, ensemble accuracy rises. <b>That is the trade a random forest makes on purpose</b>, and it is why <code>max_features</code> is the hyperparameter that matters most.');
      }
    },
    quiz: [
      {
        q: 'Averaging $M$ models with pairwise error correlation $\\rho=0.6$ reduces variance, as $M\\to\\infty$, to…',
        options: ['0', '0.6 σ²', '0.4 σ²', 'σ²/M'],
        answer: 1,
        why: 'As $M$ grows without bound, the second term, $(1-\\rho)\\sigma^2/M$, shrinks all the way to zero because $M$ sits in its own denominator — but the first term, $\\rho\\sigma^2$, contains no $M$ anywhere and survives completely untouched, so the honest limit is $0.6\\,\\sigma^2$, not zero. Option A is the tempting answer if you learned the naive rule "averaging always drives variance to zero," which is only true in the special case $\\rho=0$ — genuinely independent members — and real ensembled models on real data are essentially never that uncorrelated. Option C swaps which piece of the formula $\\rho$ multiplies, reporting $1-\\rho$ instead of $\\rho$ itself; the floor is what correlation <i>guarantees will remain</i>, not what it removes. Option D is the independent-models formula with the correlation term dropped entirely. The principle §2.16.1 is testing is that correlation between members sets a hard floor on how much averaging can help, and no amount of adding more members gets you under it.'
      },
      {
        q: 'Bagging primarily reduces…',
        options: ['bias', 'variance', 'both equally', 'irreducible noise'],
        answer: 1,
        why: 'Bagging trains every member to be as accurate as it can be on its own bootstrap sample and never changes what any individual model is trying to learn, so the average of $M$ equally biased predictors carries exactly the same bias as one of them — expectation is linear, and averaging $M$ copies of the same expected value returns that value unchanged. What bagging does move is variance: averaging pulls the $(1-\\rho)/M$ term of §2.16.1\'s formula down toward the $\\rho\\sigma^2$ floor, a real and often substantial reduction whenever the members are even somewhat decorrelated. "Both equally" is the tempting middle-ground answer, because ensembling in general sounds like a blanket improvement, but bagging\'s mechanism — parallel, independent-ish fits, simply averaged — structurally cannot touch bias, which is exactly why boosting exists as its opposite number, fitting each new member sequentially to whatever the ensemble still gets wrong. Irreducible noise, $\\sigma^2$ in the bias–variance sense of §2.2, is a property of the labels themselves and no combination of models, bagged or otherwise, can reduce it.'
      },
      {
        q: 'A stacking meta-model must be trained on…',
        options: ['the base models’ training predictions', 'out-of-fold predictions', 'test-set predictions', 'raw features only'],
        answer: 1,
        why: 'Out-of-fold predictions are, for every row, the prediction made by whichever fold\'s model never saw that row during training — the identical discipline §2.1 and §2.14 build for evaluating any single model, applied one level up to evaluate the base models as inputs to a second one. Training on the base models\' in-fold predictions instead is the tempting shortcut, because those predictions already exist with no extra bookkeeping required — but a base model that has memorised its own training rows reports near-perfect predictions on exactly those rows, and the meta-learner, with no way to tell memorisation from genuine skill, learns to over-trust whichever base model overfitted hardest. Test-set predictions fail for a more basic reason still: the test set exists to be graded exactly once, at the very end, and touching it while training any component — a base model or a meta-model — imports precisely the optimism §2.1.3 spent an entire derivation quantifying. Raw features alone would defeat the purpose of stacking altogether, which is to learn how much to trust each base model\'s opinion, not to refit a new model on the original inputs from scratch.'
      }
    ],
    cards: [
      { q: 'The ensemble variance formula', a: '$\\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$ — the floor is $\\rho\\sigma^2$, so diversity beats count.' },
      { q: 'Bagging vs boosting in one line', a: 'Bagging: strong learners in parallel, reduces variance. Boosting: weak learners in sequence on residuals, reduces bias.' },
      { q: 'Why random forests subsample features', a: 'To break the correlation that bootstrap resampling alone leaves, by preventing every tree from splitting on the same dominant feature first.' },
      { q: 'SWA / model soups', a: 'Average the weights, not the predictions. Part of the ensemble gain at zero inference cost — valid only within one loss basin.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.21 */
  ML.section({
    id: 'gp-bayesopt', track: 'classical', num: '2.21', level: 3,
    title: 'Gaussian processes and Bayesian optimisation',
    lede: 'A model that says "I do not know" in a quantitative, well-calibrated way — and the search algorithm that exploits exactly that. This is how hyperparameters get tuned when each evaluation costs a GPU-day, and it is the cleanest example of uncertainty being useful rather than decorative.',
    prereq: ['bayesian-inference', 'linear-algebra'],
    related: ['hyperparameters', 'bandits', 'calibration'],
    html: `
${H.tldr([
      'A Gaussian process is a prior over <i>functions</i>, built the same way §0.2 built alignment out of a dot product — except now the "vectors" are hypothetical outputs of a function, and the kernel plays the role the dot product played there.',
      'Condition it on data and the posterior is Gaussian in closed form: mean $K_*^\\top(K+\\sigma_n^2 I)^{-1}y$, variance $k_{**} - K_*^\\top(K+\\sigma_n^2I)^{-1}K_*$. No approximation, no gradient descent — just linear algebra.',
      'Bayesian optimisation = GP surrogate + acquisition function. It wins when evaluations are expensive and the dimension is under ~20; above that, random search and Hyperband are better value (§2.15).'
    ])}

<h2><span class="sn">2.21.1</span> A distribution over functions, built from a similarity score</h2>

<p>You are tuning a model whose training run costs an hour of GPU time. You have tried a regularisation strength $\\lambda = 0.1$ and measured a validation loss of 0.42. You would like a guess at the loss for $\\lambda = 0.3$ without spending the hour to find out. Ordinary regression cannot help you here, because regression asks you to commit to a <i>shape</i> first — linear in $\\lambda$, quadratic, log-linear — and you have exactly one data point, nowhere near enough to tell those shapes apart. Commit to the wrong one and every prediction you make is confidently wrong.</p>

<p>Here is a weaker, more honest assumption to make instead: whatever the true shape is, two values of $\\lambda$ that are <i>close together</i> should give losses that are close together too. That is not a claim about the function's form. It is a claim about its smoothness, and it is usually true of things you can actually train. A Gaussian process is the machine that turns that one assumption into predictions, with calibrated uncertainty, using nothing but the linear algebra of §0.2.</p>

<p>Recall the object §0.2.3 built: take two vectors, multiply them component-wise, add up the products, and you get a single number that says how aligned they are. A Gaussian process asks almost the same question about a completely different pair of things. Instead of two rows of data, take two <i>candidate inputs</i> to your function — say $\\lambda = 0.1$ and $\\lambda = 0.3$ — and ask: if I knew the function's value at one, how much would that tell me about its value at the other? The answer is a single number, and it should behave exactly like a similarity score: large when the two inputs are close, small when they are far apart, symmetric in the two inputs, and it should never suggest that a point is more informative about a distant point than about itself. A function that produces such a number from a pair of inputs is called a <b>kernel</b>, written $k(x_i, x_j)$, and it is doing structurally the same job the dot product did in §0.2 — except there it measured the alignment of two arrows you could draw, and here it measures the alignment the model is willing to assume between two function values it has never seen.</p>

<p>Now make the leap that gives the method its name. Pick any finite collection of input points $x_1, \\dots, x_n$ — as many as you like, wherever you like. Declare that the corresponding, still-unknown function values $f(x_1), \\dots, f(x_n)$ are jointly <b>Gaussian</b>: a single multivariate normal with mean vector $m$ and covariance matrix $K$, where $K_{ij} = k(x_i, x_j)$. That declaration is the entire definition of a Gaussian process. The diagonal of $K$ says how much each function value is allowed to vary on its own; the off-diagonal entries, filled in by the kernel, say how strongly each pair of values is expected to move together. Because the kernel makes nearby inputs highly correlated and distant inputs nearly independent, "smoothness" has been encoded directly into a covariance matrix, with no parametric shape assumed anywhere.</p>

${H.analogy(`<p>You are a geologist with core samples from three boreholes and a map to fill in between them. You do not know the true shape of the ore seam — it might dip, fold, or fault — but you are willing to bet that two points ten metres apart are more alike than two points ten kilometres apart. That single bet is enough to interpolate sensibly: near a borehole, trust the borehole; between two nearby boreholes, blend them; far from every borehole, admit you are guessing and say so with a wide error bar.</p>
<p>This is not a loose analogy. It is the same computation under a different name: geostatisticians call it <b>kriging</b>, after the South African mining engineer Danie Krige, and it was formalised mathematically by Georges Matheron in the 1960s — decades before "Gaussian process regression" entered the machine learning vocabulary in the 1990s. The gold mine and the hyperparameter sweep are solving the same equation.</p>`)}

<p>Watch the assumption do actual work with one observation. Take the most common kernel, the RBF (radial basis function, also called squared exponential), $k(x_i, x_j) = v\\exp\\!\\big(-\\tfrac{1}{2}(x_i-x_j)^2/\\ell^2\\big)$, with signal variance $v=1$, length-scale $\\ell=1$, and observation noise $\\sigma_n=0.1$. Suppose you observe $y=2$ at $x=0$ and nothing else. Feed that single point through the conditioning formula below at three new locations:</p>

${H.table(['Query point', 'Distance from the data', 'Posterior mean', 'Posterior sd'], [
      ['$x=0.3$', 'a third of a length-scale away', '1.89', '0.31'],
      ['$x=1$', 'one full length-scale away', '1.20', '0.80'],
      ['$x=4$', 'four length-scales away', '0.00', '1.00']
    ])}

<p>Read the shape of that table, not just the numbers. Close to the observation, the mean sits near the observed value and the uncertainty is small — the model trusts the data. One length-scale out, the mean has already fallen back more than a third of the way to zero, and the uncertainty has grown to 0.80, most of the way back to the prior's standard deviation of 1. Four length-scales out, the mean and standard deviation are indistinguishable from the prior you started with — the single observation has told the model nothing at all about a point that far away. That pinch-near-data, balloon-away-from-it shape is not a plotting choice; it falls straight out of the kernel, and it is the entire reason a GP is useful for deciding where to look next, which is the subject of §2.21.2.</p>

<p>Condition the general $n$-point Gaussian on observed data and the ordinary rule for conditioning a multivariate Gaussian — the same rule that produces every posterior in this section — collapses to two formulas, with no approximation at all:</p>
$$\\mu_*(x) = k_*^\\top (K+\\sigma_n^2 I)^{-1} y, \\qquad \\sigma_*^2(x) = k(x,x) - k_*^\\top (K+\\sigma_n^2 I)^{-1} k_*$$
<p>Read the mean formula the way you read a matrix-vector product in §0.2.4: $k_*$ is the vector of kernel values between the query point and every training point — how similar the query is to each thing you have already seen — and $(K+\\sigma_n^2I)^{-1}y$ is a fixed vector computed once from the training data. Their dot product is a similarity-weighted average of the observed $y$'s: points you are similar to pull the prediction toward their value, and points you are dissimilar to barely vote at all. The variance formula starts from the prior variance $k(x,x)$ and <i>subtracts</i> exactly the amount of it that the data has explained away — which can never be negative, because you cannot become less certain by seeing more data.</p>
${H.key('Read the variance formula again and notice what is missing: it does not contain $y$. A GP knows where it is uncertain <i>before it sees any labels</i> — uncertainty is a function of where you sampled, not of what you found there. That is what makes it a good guide for deciding where to sample next, rather than just a good curve fitter.')}
${H.note('The cost is $O(n^3)$ to factorise $K$ (a Cholesky decomposition, the same tool §1.15 uses to solve least squares stably) and $O(n^2)$ memory, which caps exact GPs at roughly $n\\approx 10{,}000$. Sparse/inducing-point approximations and the various "deep kernel" methods exist precisely to push past that.')}

<p><b>What you are looking at.</b> The blue line is the posterior mean and the two shaded bands are ±1 and ±2 posterior standard deviations, computed by a real Cholesky factorisation on whatever points you have placed — nothing here is faked or pre-rendered. The faint grey wiggles are individual functions sampled from that same posterior, which is a useful reminder that the mean line is a summary, not the model: the model is the whole family of curves consistent with your data.</p>
<p><b>What to do with it.</b> Click anywhere on the plot to drop an observation, and watch the band pinch to the noise floor exactly at that point and balloon back open a short distance away. Then reproduce the worked numbers above: switch the kernel dropdown to RBF (the lab defaults to Matérn 3/2, which is close but not identical — the table above is RBF-exact), clear the points, click once near $x=0,\\ y=2$, and read the mean and sd off the readout at $x\\approx1$ and $x\\approx4$.</p>
<p><b>The thing genuinely worth noticing.</b> Drag the length-scale slider to its minimum and watch every one of the grey sampled functions turn jagged and independent from point to point — short $\\ell$ is the model confessing it believes almost nothing about smoothness. Drag it to the maximum and the sampled functions barely bend at all, and a single point now constrains the curve's value far away. You are watching the one number that encodes everything the model assumes about the world, and it is a number you chose, not one the data handed you — which is exactly why the next paragraph fits it automatically instead of leaving it to guesswork.</p>

${H.lab('gp', 'A Gaussian process, live', 'Click the plot to add observations. The shaded band is ±1 and ±2 posterior standard deviations — genuine Cholesky, no faking. Watch the band pinch to the noise floor at every data point and balloon between them.')}

${H.table(['Kernel', 'Assumes', 'Use when'], [
      ['RBF / squared exponential', 'infinitely differentiable, very smooth', 'you want a smooth interpolant and know the function really is smooth'],
      ['<b>Matérn 3/2</b>', 'once differentiable — rougher, more realistic', '<b>the sane default</b> for physical or empirical functions'],
      ['Matérn 5/2', 'twice differentiable', 'the standard choice inside Bayesian optimisation packages'],
      ['Periodic', 'exact repetition at a known period', 'seasonality, with a period you can justify'],
      ['Linear', 'the function is linear', 'Bayesian linear regression is a GP with this kernel — a useful thing to know'],
      ['Sums and products', 'composition of the above', 'trend × seasonal + noise, the classic time-series decomposition']
    ])}
${H.intuition(`<p>The length-scale $\\ell$ is the distance over which the function is allowed to change. Short $\\ell$ means "anything can happen a step away" — the posterior reverts to the prior mean almost immediately and the uncertainty band snaps wide open between points, exactly as it did in the miniature example above when the query point moved four length-scales out. Long $\\ell$ means "the function is lazy" — one observation informs a large neighbourhood.</p>
<p>Fitting $\\ell$ by maximising the log marginal likelihood — the quantity the lab's readout calls <code>lml</code> — is doing Occam's razor automatically. A tiny $\\ell$ can fit any training set almost perfectly, because it lets the function do whatever it needs between points, but that same flexibility makes the marginal likelihood <i>low</i>: a model that can explain everything predicts nothing sharply, and the maths penalises it for that lack of commitment. A huge $\\ell$ is too rigid to fit the data and is penalised the other way. The maximum sits at the compromise, chosen by the same log-likelihood machinery as every other model in this course, just applied to a covariance matrix instead of a set of weights.</p>`)}

<h2><span class="sn">2.21.2</span> Bayesian optimisation</h2>
<p>You want $\\arg\\min f$ where each evaluation is expensive (train a model, run an experiment, synthesise a molecule). You now have a GP that, after every evaluation, hands you both a best guess and an honest uncertainty at every untried point. The naive way to use it is to always evaluate wherever the posterior mean is lowest — but that is pure exploitation, and it is a trap: the first few evaluations pin down one promising-looking region and the search never leaves it, exactly as a greedy hill-climb never discovers the hill next door. What you actually want is a rule that spends some evaluations exploiting a good mean and some exploring a large variance, and can defend the balance in a single formula. That formula is an <b>acquisition function</b>.</p>

<p>The standard one is <b>expected improvement</b>, and it is worth deriving rather than just quoting, because the derivation is exactly where the exploit/explore split comes from. Let $f^+$ be the best (lowest) value found so far, and at a candidate point $x$ let the posterior be $f(x) \\sim \\mathcal{N}(\\mu,\\sigma^2)$. Define the <b>improvement</b> as how much better than the incumbent a trial at $x$ would be, floored at zero because a worse result is simply not taken:</p>

${H.deriv('expected improvement, from the definition of improvement to the closed form', [
      ['$I(x) = \\max(f^+ - f(x),\\ 0)$', 'The improvement is only counted when the new value beats the best one seen so far; a trial that does worse contributes nothing, since you would simply keep the incumbent.'],
      ['$\\mathrm{EI}(x) = \\mathbb{E}[I(x)] = \\displaystyle\\int_{-\\infty}^{f^+} (f^+ - t)\\,\\phi\\!\\Big(\\frac{t-\\mu}{\\sigma}\\Big)\\frac{1}{\\sigma}\\,dt$', 'Take the expectation under the GP posterior for $f(x)$, which is Gaussian with density $\\phi$ rescaled by mean $\\mu$ and spread $\\sigma$; the integral only runs up to $f^+$ because $I(x)=0$ beyond it.'],
      ['$z = \\dfrac{t-\\mu}{\\sigma},\\quad dt = \\sigma\\,dz$', 'Substitute to standardise the integral — the same change of variable that turns any normal integral into a standard-normal one, used already for the z-statistic in §1.6.'],
      ['$\\mathrm{EI}(x) = (f^+-\\mu)\\displaystyle\\int_{-\\infty}^{z^+}\\phi(z)\\,dz + \\sigma\\displaystyle\\int_{-\\infty}^{z^+} -z\\,\\phi(z)\\,dz$', 'Expand $(f^+-t) = (f^+-\\mu) - \\sigma z$ and split the integral into two pieces, one that does not depend on $z$ and one that does.'],
      ['$= (f^+-\\mu)\\,\\Phi(z^+) + \\sigma\\,\\phi(z^+)$', 'The first integral is the definition of the normal CDF, $\\Phi$. The second is a standard identity — the derivative of $\\phi$ is $-z\\phi(z)$, so its integral is $\\phi$ itself, evaluated at the limit.']
    ], 'Set $z^+ = (f^+-\\mu)/\\sigma$ and the result is the formula in the table below. The two terms are now visibly two different things: $(f^+-\\mu)\\Phi(z)$ only pays off when the mean is already good, and $\\sigma\\phi(z)$ pays off purely for being uncertain, regardless of where the mean sits. EI needs no tuning knob because the exploration term falls out of the same integral as the exploitation term, not because someone hand-balanced them.')}

${H.table(['Acquisition', 'Formula', 'Character'], [
      ['Probability of improvement', '$\\Phi\\!\\left(\\frac{f^+ - \\mu}{\\sigma}\\right)$', 'greedy; a tiny near-certain gain satisfies it, so it gets stuck near the incumbent'],
      ['<b>Expected improvement</b>', '$(f^+-\\mu)\\Phi(z) + \\sigma\\phi(z)$', 'the default — the two terms, derived above, are literally exploit + explore'],
      ['Lower confidence bound', '$\\mu - \\kappa\\sigma$', 'one interpretable knob; has regret bounds'],
      ['Thompson sampling', 'draw a function from the posterior, minimise it', 'trivially parallelisable across workers'],
      ['Entropy search', 'maximise information about $x^\\star$', 'best sample efficiency, most computation per step']
    ])}

<p><b>What you are looking at.</b> The top panel is the same posterior-mean-and-band picture as the previous lab, now overlaid with the true objective (dashed, and something the algorithm never gets to see directly) and the points actually evaluated so far. The bottom panel is the chosen acquisition function over the same $x$-axis, and the vertical line marks where it peaks — the point about to be evaluated next.</p>
<p><b>What to do with it.</b> Press <b>Step</b> repeatedly with expected improvement selected and watch the two-panel loop: the GP fits, the acquisition peaks somewhere, that point gets evaluated, the GP refits with one more observation, and the peak moves. Then switch to <b>probability of improvement</b>, reset, and press <b>Run 8 steps</b>.</p>
<p><b>The thing genuinely worth noticing.</b> Probability of improvement clusters every one of its eight evaluations around whatever looked best first, and the acquisition curve in the bottom panel stays a single narrow spike over the incumbent the whole time — it is satisfied by a near-certain sliver of improvement and never bothers with the wide, unexplored gaps. Switch back to expected improvement and its acquisition curve visibly has <i>two kinds</i> of peak at once: a tall narrow one at the current best and a shorter, wider one sitting in whatever region has not been tried. That second peak is the $\\sigma\\phi(z)$ term from the derivation, made visible.</p>

${H.lab('bo', 'Bayesian optimisation, step by step', 'Press step and watch the loop: fit the GP, maximise the acquisition function, evaluate the true function there, refit. The lower panel is the acquisition function itself — with expected improvement, watch it grow two kinds of peak: one at the incumbent minimum (exploit) and one in the widest unexplored gap (explore).')}

${H.worked('when Bayesian optimisation is worth it', `
<p>Suppose one training run costs 6 GPU-hours and you have a budget of 60 runs over 8 hyperparameters.</p>
<ul>
<li><b>Grid search</b>: $2^8 = 256$ points at the coarsest possible resolution — one point per dimension per side. $256 \\times 6 = 1{,}536$ GPU-hours against a budget of $60\\times6=360$. Not affordable, and that is before considering that 2 points per dimension is a hopelessly coarse grid.</li>
<li><b>Random search</b>: 60 draws. With 8 dimensions of which perhaps 2 actually matter, random search still gets roughly 60 distinct values along each important dimension, because every draw is independent across dimensions — which is why it comfortably beats grid search at the same budget (Bergstra &amp; Bengio, 2012).</li>
<li><b>Bayesian optimisation</b>: typically reaches random search's 60-run result in 15–25 runs on smooth, low-effective-dimension problems. At 6 GPU-hours a run that is $90$–$150$ hours spent against $360$ hours for random search — a saving of roughly $210$–$270$ GPU-hours, well over half the budget, for the cost of fitting a small GP after every run.</li>
<li><b>Hyperband / ASHA</b>: kills bad configurations early instead of modelling the surface at all. Often beats plain BO on wall-clock because it exploits the cheapest signal available — the learning curve — rather than the most expensive one, the final metric. <b>BOHB combines the two and is the current default recommendation.</b></li>
</ul>`)}
${H.flag('Bayesian optimisation degrades above roughly 20 effective dimensions, and it assumes evaluations are noiseless-ish and stationary. For neural-architecture-scale problems with hundreds of choices, successive halving with random sampling is both simpler and, in most published comparisons, at least as good.')}

${H.history(`<p>The mathematics of expected improvement is not recent. Harold Kushner proposed essentially the probability-of-improvement idea in 1964 for optimising noisy one-dimensional functions, and Jonas Mockus formalised expected improvement itself in the 1970s. For nearly forty years the technique stayed a specialist tool inside geostatistics and engineering design — exactly the kriging lineage from §2.21.1 — because almost nobody else was routinely paying for expensive black-box evaluations.</p>
<p>That changed when deep learning made "one hyperparameter evaluation costs a GPU-day" an ordinary sentence rather than an exotic one. Snoek, Larochelle and Adams' 2012 paper "Practical Bayesian Optimisation of Machine Learning Algorithms" is usually credited with popularising the method for the ML audience specifically, and it is why Bayesian optimisation libraries (Spearmint, then a wave of others) arrived at almost exactly the moment hyperparameter tuning became expensive enough to justify them. The maths did not change between 1964 and 2012; the cost of a wrong guess did.</p>`)}

${H.probe([
      ['Why does the GP posterior variance not depend on $y$?', 'Because for a Gaussian likelihood the posterior covariance depends only on the input locations, as the formula in §2.21.1 shows directly — $y$ never appears in it. Where you looked determines how sure you are; what you saw determines only the mean. This is exactly why the variance is a safe, label-free signal for deciding where to sample next, and it is the property the whole of Bayesian optimisation is built on.'],
      ['RBF or Matérn?', 'Matérn 5/2 by default in most BO libraries, and Matérn 3/2 as a rougher alternative. RBF assumes infinite differentiability, which real objective surfaces — validation loss as a function of a learning rate, say — essentially never have, and that mismatch shows up as overconfident interpolation: tight bands in regions where the true function is actually still wobbling. A Matérn kernel is the honest admission that the surface has kinks.'],
      ['Expected improvement has two terms. What are they, and why does EI not need a tuning parameter?', 'From the derivation, $(f^+-\\mu)\\Phi(z)$ is the exploitation term — it only pays off when the posterior mean is already better than the incumbent — and $\\sigma\\phi(z)$ is the exploration term, which pays off purely for uncertainty regardless of where the mean sits. Both terms fell out of the same expectation integral, with no free parameter introduced along the way, which is different from lower confidence bound, where you have to choose $\\kappa$ by hand.'],
      ['Why is a GP $O(n^3)$, and what does that number actually cap?', 'Fitting requires a Cholesky factorisation of the $n\\times n$ kernel matrix, and Cholesky is a cubic-time algorithm. It caps <i>exact</i> inference around $n\\approx10^4$ observations on ordinary hardware; inducing-point (sparse) methods pick $m \\ll n$ representative points and reduce the cost to $O(nm^2)$, trading a small amount of accuracy for tractability at much larger $n$.']
    ])}`,
    labs: {
      gp: function (host) {
        const pts = [{ x: -2.2, y: 0.7 }, { x: 0.4, y: -0.9 }, { x: 2.5, y: 1.4 }];
        const st = Viz.controls(host, [
          { k: 'kernel', label: 'kernel', type: 'select', value: 'matern32', options: [
            { v: 'rbf', t: 'RBF (squared exponential)' }, { v: 'matern32', t: 'Matérn 3/2' }, { v: 'periodic', t: 'periodic' }, { v: 'linear', t: 'linear' }] },
          { k: 'l', label: 'length-scale ℓ', min: .15, max: 4, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'v', label: 'signal variance', min: .2, max: 4, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'noise', label: 'observation noise σₙ', min: .01, max: 1, step: .01, value: .1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        Viz.buttons(host, [
          { label: 'clear points', on: () => { pts.length = 0; S.redraw(); } },
          { label: 'add 6 random', on: () => { const R = Num.rng(pts.length + 3); for (let i = 0; i < 6; i++) { const x = -4 + 8 * R(); pts.push({ x: x, y: Math.sin(1.3 * x) + 0.3 * x + R.normal(0, .1) }); } S.redraw(); } }
        ]);
        const out = Viz.readout(host, [
          { k: 'n', label: 'observations', cls: 'key' },
          { k: 'lml', label: 'log marginal likelihood' },
          { k: 'maxsd', label: 'widest uncertainty' },
          { k: 'cost', label: 'cost' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const g = Num.gp(pts.map(p => p.x), pts.map(p => p.y), {
              kernel: st.kernel, l: st.l, v: st.v, noise: st.noise, per: 2.5
            });
            const grid = Num.linspace(-4.4, 4.4, 190);
            const post = g.predict(grid);
            const P = Viz.plot(ctx, w, h, { xd: [-4.4, 4.4], yd: [-3.4, 3.4], pad: { l: 44, r: 14, t: 16, b: 38 } })
              .frame({ xlabel: 'x', ylabel: 'f(x)' });
            P.clip(() => {
              P.band(grid.map((x, i) => [x, post[i].mu + 2 * post[i].sd]),
                     grid.map((x, i) => [x, post[i].mu - 2 * post[i].sd]), { color: T.c1, alpha: .17 });
              P.band(grid.map((x, i) => [x, post[i].mu + post[i].sd]),
                     grid.map((x, i) => [x, post[i].mu - post[i].sd]), { color: T.c1, alpha: .17 });
              P.line(grid.map((x, i) => [x, post[i].mu]), { color: T.c1, width: 2.6 });
              // three posterior samples, drawn by perturbing the mean with the marginal sd
              const R = Num.rng(3);
              for (let s = 0; s < 3; s++) {
                let z = R.normal(0, 1);
                const samp = grid.map((x, i) => {
                  z = 0.86 * z + Math.sqrt(1 - 0.86 * 0.86) * R.normal(0, 1);
                  return [x, post[i].mu + z * post[i].sd];
                });
                P.line(samp, { color: T.c5, width: 1, alpha: .5 });
              }
              pts.forEach(p => P.dots([[p.x, p.y]], { r: 4.5, color: T.c2, stroke: true, strokeWidth: 1.6 }));
            });
            out({
              n: pts.length,
              lml: pts.length ? g.logML.toFixed(2) : '—',
              maxsd: Math.max.apply(null, post.map(p => p.sd)).toFixed(3),
              cost: 'O(' + pts.length + '³) = ' + Math.pow(pts.length, 3).toLocaleString() + ' ops'
            });
          }
        });
        Viz.pointer(S, function (e) {
          if (e.type !== 'down') return;
          const P = Viz.plot(S.ctx, S.w, S.h, { xd: [-4.4, 4.4], yd: [-3.4, 3.4], pad: { l: 44, r: 14, t: 16, b: 38 } });
          pts.push({ x: P.ix(e.x), y: P.iy(e.y) });
          S.redraw();
        });
        Viz.legend(host, [
          { c: 'var(--c1)', t: 'posterior mean ±1σ, ±2σ' }, { c: 'var(--c5)', t: 'sampled functions' }, { c: 'var(--c2)', t: 'observations' }
        ]);
        Viz.note(host, 'Click anywhere on the plot to place an observation. Then drop the length-scale to 0.2: the posterior forgets each point almost immediately and the band snaps open a hair away from the data — the model now believes the function can do anything at any moment. Raise it to 4 and one point constrains the entire domain. <b>The length-scale is where all your prior belief about smoothness lives</b>, and fitting it by maximising the log marginal likelihood is the principled way to choose it.');
      },

      bo: function (host) {
        /* the expensive black box we are pretending not to be able to evaluate */
        const truth = x => Math.sin(2.1 * x) + 0.35 * Math.cos(4.7 * x) + 0.14 * x * x - 0.2 * x;
        let obs = [{ x: -3.4 }, { x: 2.9 }].map(p => ({ x: p.x, y: truth(p.x) }));
        const st = Viz.controls(host, [
          { k: 'acq', label: 'acquisition', type: 'select', value: 'ei', options: [{ v: 'ei', t: 'expected improvement' }, { v: 'lcb', t: 'lower confidence bound' }, { v: 'pi', t: 'probability of improvement' }] },
          { k: 'kappa', label: 'κ (LCB only)', min: .5, max: 4, step: .1, value: 2, fmt: v => v.toFixed(1) },
          { k: 'l', label: 'length-scale', min: .2, max: 2.5, step: .05, value: .7, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'evals', label: 'evaluations', cls: 'key' },
          { k: 'best', label: 'best found', cls: 'good' },
          { k: 'gap', label: 'gap to true optimum', cls: 'bad' },
          { k: 'next', label: 'next point' }
        ]);
        const grid = Num.linspace(-4, 4, 220);
        const trueMin = Math.min.apply(null, grid.map(truth));

        function acqAt(mu, sd, best) {
          if (st.acq === 'lcb') return -(mu - st.kappa * sd);
          if (st.acq === 'pi') return Num.normCdf((best - mu - 0.01) / Math.max(1e-9, sd));
          return Num.expectedImprovement(mu, sd, best, 0.01);
        }
        function nextPoint() {
          const g = Num.gp(obs.map(o => o.x), obs.map(o => o.y), { kernel: 'matern32', l: st.l, v: 1.2, noise: .02 });
          const post = g.predict(grid);
          const best = Math.min.apply(null, obs.map(o => o.y));
          let bi = 0, bv = -Infinity;
          post.forEach((p, i) => { const a = acqAt(p.mu, p.sd, best); if (a > bv) { bv = a; bi = i; } });
          return { g: g, post: post, best: best, xi: grid[bi], a: post.map(p => acqAt(p.mu, p.sd, best)) };
        }
        Viz.buttons(host, [
          { label: 'Step', primary: true, on: () => { const s = nextPoint(); obs.push({ x: s.xi, y: truth(s.xi) }); S.redraw(); } },
          { label: 'Run 8 steps', on: () => { for (let i = 0; i < 8; i++) { const s = nextPoint(); obs.push({ x: s.xi, y: truth(s.xi) }); } S.redraw(); } },
          { label: 'Reset', on: () => { obs = [{ x: -3.4 }, { x: 2.9 }].map(p => ({ x: p.x, y: truth(p.x) })); S.redraw(); } }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const s = nextPoint();
            const hTop = h * .66;
            const P = Viz.plot(ctx, w, hTop, { xd: [-4, 4], yd: [-2.6, 3.4], pad: { l: 44, r: 14, t: 14, b: 22 } })
              .frame({ xticks: [], ylabel: 'objective' });
            P.clip(() => {
              P.line(grid.map(x => [x, truth(x)]), { color: T.faint, width: 1.4, dash: [5, 4] });
              P.band(grid.map((x, i) => [x, s.post[i].mu + 2 * s.post[i].sd]),
                     grid.map((x, i) => [x, s.post[i].mu - 2 * s.post[i].sd]), { color: T.c1, alpha: .18 });
              P.line(grid.map((x, i) => [x, s.post[i].mu]), { color: T.c1, width: 2.4 });
              obs.forEach(o => P.dots([[o.x, o.y]], { r: 4.2, color: T.c2, stroke: true, strokeWidth: 1.5 }));
              P.vline(s.xi, { color: T.c3, width: 1.6 });
            });
            ctx.save(); ctx.translate(0, hTop);
            const amax = Math.max.apply(null, s.a), amin = Math.min.apply(null, s.a);
            const P2 = Viz.plot(ctx, w, h - hTop, {
              xd: [-4, 4], yd: [amin - (amax - amin) * .1, amax + (amax - amin) * .15 || 1],
              pad: { l: 44, r: 14, t: 8, b: 32 }
            }).frame({ yticks: [], xlabel: 'x', ylabel: 'acquisition' });
            P2.clip(() => {
              P2.area(grid.map((x, i) => [x, s.a[i]]), { color: T.c3, alpha: .3, base: amin - 1 });
              P2.line(grid.map((x, i) => [x, s.a[i]]), { color: T.c3, width: 1.8 });
              P2.vline(s.xi, { color: T.c3, width: 1.6, dash: false, label: 'next' });
            });
            ctx.restore();
            out({
              evals: obs.length,
              best: s.best.toFixed(4),
              gap: (s.best - trueMin).toFixed(4),
              next: 'x = ' + s.xi.toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--faint)', t: 'the true (unknown) objective' }, { c: 'var(--c1)', t: 'GP surrogate' },
          { c: 'var(--c2)', t: 'evaluations spent' }, { c: 'var(--c3)', t: 'acquisition' }
        ]);
        Viz.note(host, 'Switch to <b>probability of improvement</b> and press "Run 8 steps": it clusters every evaluation around the current best and never looks in the unexplored region, which is the textbook failure of a purely greedy acquisition. Expected improvement spends a couple of evaluations on wide gaps first — and finds the global minimum with fewer total calls. <b>Exploration is not a nicety here; it is the difference between finding the optimum and polishing a local one.</b>');
      }
    },
    quiz: [
      {
        q: 'A GP’s posterior variance at a new point depends on…',
        options: ['the observed $y$ values', 'only where the observations are, not what they were', 'the prior mean', 'the number of iterations'],
        answer: 1,
        why: 'Look at the variance formula in §2.21.1: $y$ simply does not appear in it. With a Gaussian likelihood the posterior covariance is a function of the input locations alone — where you sampled, not what you found there. The tempting wrong answer is the first option, because it feels natural that "more surprising data" should mean "more uncertain model", but that intuition describes the mean, not the variance. The general principle being tested is that a GP separates <i>where it looked</i> from <i>what it saw</i>, which is exactly what makes the variance a safe, label-free signal for choosing where to sample next.'
      },
      {
        q: 'Exact GP inference costs…',
        options: ['$O(n)$', '$O(n^2)$', '$O(n^3)$, from the Cholesky factorisation', '$O(nd)$'],
        answer: 2,
        why: 'Fitting a GP means factorising the $n\\times n$ kernel matrix, and Cholesky factorisation is a cubic-time algorithm — cubic in the number of <i>observations</i>, not the number of input dimensions, which is why $O(nd)$ is the tempting but wrong answer: dimensionality barely matters here, the sample count does. This cost is what caps exact GPs near $n\\approx10^4$ points and motivates inducing-point (sparse) approximations, which trade a small amount of accuracy for a cost of $O(nm^2)$ with $m \\ll n$.'
      },
      {
        q: 'Expected improvement beats probability of improvement because…',
        options: ['it is cheaper to compute', 'it rewards the magnitude of a possible improvement, not just its chance, so it explores', 'it needs no GP', 'it is unbiased'],
        answer: 1,
        why: 'PI only asks "how likely is any improvement at all", so a point with a 90% chance of a tiny gain beats a point with a 40% chance of a huge one — it is satisfied by a near-certain sliver and clusters every evaluation at the incumbent, as the lab shows directly when you switch to it and press Run 8 steps. EI’s derivation in §2.21.2 produces a second term, $\\sigma\\phi(z)$, that rewards uncertainty on its own terms, independent of how good the mean is. That second term is what buys exploration without a tuning parameter.'
      },
      {
        q: 'Above roughly 20 hyperparameters, the better default is…',
        options: ['grid search', 'Bayesian optimisation with more iterations', 'random search or successive halving (Hyperband/ASHA)', 'manual tuning'],
        answer: 2,
        why: 'GP surrogates degrade in high dimensions because the kernel’s notion of "nearby" stops meaning anything useful once there are dozens of axes to be nearby along — the curse of dimensionality attacking the smoothness assumption itself. "More BO iterations" is the tempting wrong answer because it treats the problem as a budget shortfall rather than a modelling-assumption failure; throwing more iterations at a surrogate that cannot represent the surface does not fix that. Exploiting the learning curve to kill bad runs early (Hyperband/ASHA) is a stronger, cheaper signal than trying to model a high-dimensional surface at all.'
      }
    ],
    cards: [
      { q: 'GP posterior, in symbols', a: '$\\mu_*=k_*^\\top(K+\\sigma_n^2I)^{-1}y$, $\\sigma_*^2 = k_{**}-k_*^\\top(K+\\sigma_n^2I)^{-1}k_*$.' },
      { q: 'Default kernel', a: 'Matérn 5/2. RBF assumes infinite differentiability and interpolates over-confidently.' },
      { q: 'Expected improvement’s two terms', a: '$(f^+-\\mu)\\Phi(z)$ exploits; $\\sigma\\phi(z)$ explores. Their sum removes the need for a tuning knob.' },
      { q: 'When to use Bayesian optimisation', a: 'Expensive evaluations, under ~20 effective dimensions, reasonably smooth. Otherwise ASHA/Hyperband.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.22 */
  ML.section({
    id: 'self-supervised', track: 'classical', num: '2.22', level: 3,
    title: 'Semi-supervised, self-supervised and contrastive learning',
    lede: 'Labels are the expensive part. Every method here is an answer to the same question: what can unlabelled data tell you about the structure of the problem, and how do you turn that into an accuracy gain on the labels you do have?',
    prereq: ['embeddings'],
    related: ['embeddings', 'multimodal', 'active-transfer'],
    html: `
${H.tldr([
      'Self-supervision invents a label from the input itself: predict the masked part, predict the next token, decide whether two views came from the same source. All of pretraining is this (§4.9).',
      'Contrastive learning with InfoNCE is a <b>classification problem over the batch</b>: given a view, pick its partner out of $B$ candidates, scored by the cosine similarity built in §0.2.3. Batch size is therefore a model hyperparameter, not just a memory setting.',
      'Semi-supervised methods work when the cluster assumption holds — the decision boundary lies in a low-density region. When it does not, pseudo-labelling amplifies its own errors.'
    ])}

<p>Say you are building a sentiment classifier for product reviews. You have two thousand labelled examples, hand-tagged by someone who read each one — expensive, slow, and finished. You also have eleven million unlabelled reviews sitting in the same database, free, because nobody had to read them. Throwing the eleven million away because nobody wrote a label on them is throwing away nearly everything you know about what a product review <i>looks like</i>: its vocabulary, its sentence lengths, which words tend to sit near which other words, how a five-star review differs structurally from a one-star one even before you know which is which. The question this whole section answers is how to convert that structural knowledge, which costs nothing, into a smaller number of labels needed to reach a given accuracy. Every method below is a different answer to the single sub-question: <i>what label can I invent from data I already have, for free, that will force a model to learn something true about the domain?</i></p>

<h2><span class="sn">2.22.1</span> The families</h2>
${H.table(['Family', 'The invented task', 'Canonical', 'Failure mode'], [
      ['<b>Pseudo-labelling</b>', 'trust your own confident predictions and retrain on them', 'self-training, noisy student', 'confirmation bias — confident and wrong is self-reinforcing'],
      ['<b>Consistency regularisation</b>', 'the prediction should not change under augmentation', 'FixMatch, UDA, Mean Teacher', 'needs augmentations that preserve the label, which is domain knowledge'],
      ['<b>Contrastive</b>', 'pull two views of the same item together, push others apart', 'SimCLR, MoCo, CLIP (§4.19)', 'false negatives: two different items of the same class pushed apart'],
      ['<b>Non-contrastive</b>', 'predict one view’s embedding from the other, with an asymmetry that prevents collapse', 'BYOL, SimSiam, DINO', 'representation collapse if the asymmetry is removed'],
      ['<b>Masked modelling</b>', 'reconstruct the hidden part', 'BERT, MAE, and every LLM (§4.9)', 'the mask token never appears at fine-tuning time — a real train/test mismatch'],
      ['<b>Clustering-based</b>', 'assign pseudo-classes, predict them', 'DeepCluster, SwAV', 'degenerate assignments without an equipartition constraint']
    ])}

${H.history(`<p>None of this is new; it changed shape a few times. Word2vec (2013) invented a self-supervised task for text — predict a word from its neighbours — years before "self-supervised learning" was a phrase anyone used, and it worked for exactly the reason every method in the table above works: the invented task cannot be solved without learning something structural about the data. Autoencoders tried the same trick for images by reconstructing pixels, and mostly disappointed, because reconstructing every pixel wastes capacity on things that do not matter — the exact shade of a background wall — at the expense of things that do.</p>
<p>The 2018–2020 wave replaced "reconstruct everything" with two sharper ideas. Masked language modelling (BERT) reconstructs only a hidden <i>fraction</i>, which is cheap to compute and forces genuine context modelling rather than pixel copying. Contrastive learning (SimCLR, MoCo, 2020) sidesteps reconstruction entirely and asks a same-or-different question instead, which turns out to need far less representational capacity to answer well. Both lineages converged, by the time of CLIP and the modern foundation-model recipe, on the same conclusion: the label you invent matters less than making sure the model cannot solve the invented task by cheating.</p>`)}

<h2><span class="sn">2.22.2</span> InfoNCE, derived</h2>
<p>Contrastive learning needs a way to score "how alike are these two things", and it already has one available: cosine similarity, built from the dot product in §0.2.3, which takes two vectors and returns a single number in $[-1,1]$ saying how aligned they are, independent of their length. Here the two vectors are embeddings of two <b>views</b> of the same underlying item — two crops of one photo, an image and its caption, a sentence and a paraphrase of it — produced by running each view through an encoder. Call the score $\\mathrm{sim}(a,b)$, the cosine similarity of the two embeddings.</p>

<p>Given a batch of $B$ such pairs $(a_i, b_i)$, the loss treats "which $b$ goes with $a_i$?" as a $B$-way classification problem:</p>
$$\\mathcal{L} = -\\frac1B\\sum_{i=1}^{B}\\log\\frac{\\exp(\\mathrm{sim}(a_i,b_i)/\\tau)}{\\sum_{j=1}^{B}\\exp(\\mathrm{sim}(a_i,b_j)/\\tau)}$$
<p>Read the fraction inside the logarithm the way you would read any softmax: the numerator scores how well $a_i$'s embedding matches its <i>true</i> partner $b_i$; the denominator sums that same score against <i>every</i> $b_j$ in the batch, true partner included, so it is the total "votes" $a_i$ casts across all $B$ candidates. The fraction is therefore the probability, under a softmax, that $a_i$ picks $b_i$ out of the whole batch — and the loss is the negative log of getting that pick right, averaged over every $a_i$ in turn. $\\tau$, the temperature, divides the similarity before the exponential and is doing the same job it does in every softmax in this course: it controls how sharply the scores separate into winners and losers.</p>
${H.key('It is cross-entropy with the identity matrix as the label, computed over a $B\\times B$ similarity matrix. The positives are the diagonal; everything off-diagonal is a negative you got for free.')}

${H.worked('InfoNCE by hand, on a batch of 3', `
<p>Take a tiny batch, $B=3$, with a cosine similarity matrix already computed between three $a$'s (rows) and three $b$'s (columns). A well-trained encoder should score each pair highest against its own partner, so the diagonal is largest but not perfect:</p>
$$\\mathrm{sim} = \\begin{bmatrix} 0.90 & 0.20 & 0.10 \\\\ 0.15 & 0.85 & 0.30 \\\\ 0.05 & 0.25 & 0.88 \\end{bmatrix}$$
<p>At a mild temperature $\\tau = 1.0$, softmax row 1 (dividing by $\\tau$ changes nothing here, then exponentiating and normalising) gives probabilities $(0.514,\\ 0.255,\\ 0.231)$ — the model favours the true partner but is far from certain. Averaging $-\\log(0.514)$, $-\\log(0.482)$ and $-\\log(0.508)$ across all three rows gives a loss of <b>0.691</b>. For comparison, a model that had learned nothing at all would produce a uniform $(1/3,1/3,1/3)$ on every row, for a loss of $\\log 3 = 1.099$ — so this loss says the encoder is doing real work, just not yet confidently.</p>
<p>Now sharpen the temperature to $\\tau = 0.1$. Dividing every similarity by 0.1 before the softmax turns modest gaps like $0.90$ vs $0.20$ into a gap of $9.0$ vs $2.0$ in the exponent, and $e^{9}$ dwarfs $e^{2}$: row 1 becomes $(0.999,\\ 0.001,\\ 0.000)$, and the loss collapses to <b>0.0028</b>, near zero. Nothing about the encoder changed between the two calculations — only $\\tau$ did. That single knob is why temperature tuning gets its own line in the table below rather than being an afterthought.</p>`)}

${H.table(['Knob', 'Effect', 'Typical'], [
      ['Batch size $B$', 'number of negatives per positive; the loss is a lower bound on mutual information that <b>saturates at $\\log B$</b>', '4k–32k (SimCLR used 8192)'],
      ['Temperature $\\tau$', 'low τ sharpens: the loss concentrates on the hardest negatives, as the worked example shows. High τ treats all negatives alike', '0.05–0.1'],
      ['Projection head', 'contrast in a projected space, then <b>throw the head away</b> and use the layer below', '2-layer MLP, big accuracy effect'],
      ['Momentum encoder / queue', 'decouples the number of negatives from the batch size', 'MoCo’s contribution']
    ])}

${H.analogy(`<p>Picture a police line-up, except every suspect is also a witness. $a_i$ is shown a line-up of $B$ candidates for "who is my partner $b$" and has to cast a softmax vote across all of them. If the line-up has only 4 people, guessing right by luck is easy — a random ID has a 25% chance of being correct, so a mediocre encoder can look decent by accident. Make the line-up 8,000 people long and a random guess succeeds one time in 8,000: the only way to consistently pick the right partner is to have genuinely learned what makes $a_i$ and its true $b_i$ the same underlying thing. That is the entire reason contrastive methods keep chasing larger batches — a bigger batch is a harder line-up, and passing a harder line-up requires a better representation.</p>`)}

<p><b>What you are looking at.</b> A real InfoNCE computation on eight paired embeddings, exactly like the $3\\times3$ toy above but rendered as a heatmap. Each row is one $a_i$, each column one $b_j$, and the diagonal cell in each row is the true partner. Switch the view control between the raw cosine similarities and the softmax probabilities they produce to see the two stages of the computation separately.</p>
<p><b>What to do with it.</b> Set the temperature near 1.0 first and read a row: it should look like the mild-temperature example above, favouring the diagonal without being certain. Then drag the temperature down toward 0.02 and watch every row collapse toward one-hot, exactly as the worked $\\tau=0.1$ case did, and watch the loss readout fall toward zero as it did there.</p>
<p><b>The thing genuinely worth noticing.</b> At a very low temperature, only the single hardest off-diagonal cell in each row still has any visible probability mass — every other negative has been driven to a number indistinguishable from zero. That matters because gradients flow in proportion to these probabilities: a near-zero probability means a near-zero gradient, so <b>at low temperature, only the hardest negative in the batch actually teaches the model anything on a given step.</b> That is powerful when the hardest negative is genuinely informative and unstable when it is a mislabelled near-duplicate, which is the practical argument for the 0.05–0.1 range rather than pushing temperature as low as it will go.</p>

${H.lab('infonce', 'The similarity matrix, and what temperature does to it', 'A real InfoNCE computation on eight paired embeddings. The diagonal is the positives. Drag the temperature and watch the softmax rows go from nearly uniform to one-hot — and watch what that does to the loss and to which negatives actually contribute gradient.')}

${H.more('Why BYOL does not collapse (the question that gets asked)', `
<p>Contrastive methods need negatives to stop the trivial solution "map everything to the same vector" — if every embedding collapses to one point, cosine similarity is 1 everywhere and the InfoNCE loss cannot distinguish anything, but a contrastive loss punishes that collapse directly because it needs the off-diagonal similarities to be <i>low</i>. BYOL has no negatives at all, nothing in its loss explicitly punishes collapse, and yet it does not collapse in practice. Three ingredients together explain why:</p>
<ol>
<li><b>An asymmetric predictor</b> on the online branch only, so the two branches cannot simply agree by both going constant — the predictor would have to be able to predict a constant, which the optimiser has no gradient pressure to arrange.</li>
<li><b>A stop-gradient</b> on the target branch: the target is not trying to make itself easier to predict.</li>
<li><b>An exponential moving average</b> target encoder, which lags and therefore keeps supplying a slightly different objective each step.</li>
</ol>
<p>SimSiam later showed the EMA is not strictly necessary — <b>stop-gradient plus predictor is the core</b> — which is the empirical result that made the mechanism clear. The honest summary is that the theory here is still partial; the ablations are conclusive, the explanation is not.</p>`)}

<h2><span class="sn">2.22.3</span> Semi-supervised learning, and when it fails</h2>

<p>Contrastive and masked methods learn a representation first and fine-tune on labels afterward. A different family skips the two-stage split and uses the unlabelled pool and the labelled pool together, in the same training loop. The simplest version, <b>pseudo-labelling</b>, is almost embarrassingly direct: train on the labels you have, run the resulting model on the unlabelled pool, keep only the predictions it is confident about, and add those as if they were real labels for another round of training. It works remarkably often, and when it fails it fails in a specific, predictable way that is worth being able to compute rather than just warn about.</p>

${H.worked('how pseudo-labelling amplifies a subgroup gap, in numbers', `
<p>Suppose a model trained on the labelled set is 95% accurate on a majority subgroup and 70% accurate on a minority subgroup — a realistic gap when the minority is under-represented in the original 2,000 labels. Run it on 10,000 unlabelled examples split evenly between the two subgroups and keep only the confident predictions, say the top 60% by confidence within each group, for a second round of training.</p>
<p>On the majority subgroup, roughly 95% of the 3,000 kept predictions are correct: about 2,850 right pseudo-labels, 150 wrong ones. On the minority subgroup, roughly 70% of its 3,000 kept predictions are correct: about 2,100 right, <b>900 wrong</b> — six times as many wrong labels feeding into round two, from a group that was already the model's weak point. Retraining on this mixture does not average the errors away; it teaches the model that its own minority-subgroup mistakes are ground truth, because nothing in the pipeline distinguishes a confident wrong prediction from a confident right one. The gap that started at 25 points (95% − 70%) typically widens with each additional self-training round.</p>`)}

${H.pitfall('Confidence thresholds do not fix the amplification above; they select for it. A miscalibrated model is often <i>most</i> confident exactly where it is wrong — on subgroups or inputs its training data under-represented — so raising the threshold filters out uncertain correct predictions from the majority group while leaving confidently wrong predictions from the minority group untouched. Check pseudo-label accuracy <i>per subgroup</i> on a held-out sample before trusting a second round, not just in aggregate.')}

${H.intuition(`<p>The <b>cluster assumption</b> is doing all the work behind every method in this subsection: unlabelled data helps because it reveals where the data is <i>dense</i>, and the decision boundary is assumed to live in the sparse regions between clusters — push the boundary into a gap between two clouds of points and you have used geometry you got for free, no labels required. Consistency regularisation is the cluster assumption applied as an objective directly: if $x$ and an augmented version of $x$ sit in the same cluster, as they should, the model's prediction should not change between them, and any change is treated as an error to correct.</p>
<p>When classes genuinely overlap — credit default at a given income and utilisation, say, where the same feature values produce both outcomes with real probability — there is no low-density gap for a boundary to hide in, unlabelled data reveals nothing about where it should go, and semi-supervised methods deliver nothing. This is the honest explanation for why self-supervision transformed vision and language, where the classes really do occupy separated regions of representation space, and did comparatively little for tabular problems, where they usually do not.</p>`)}

<p><b>What you are looking at.</b> Two learning curves on the same architecture and the same test set, plotted against the number of labelled examples on a log axis. One model trains only on the labelled subset, from a random initialisation; the other starts from a representation already learned on the unlabelled pool and is then fine-tuned on the same labels. The shaded region between the curves is the value self-supervision is adding at each label budget.</p>
<p><b>What to do with it.</b> Read the gap at the far left of the axis, where labels are scarcest, and then read it again at the far right, where labels are plentiful. Drag the <b>quality of the learned representation</b> slider up and down and watch which end of the curve actually moves.</p>
<p><b>The thing genuinely worth noticing.</b> The gap is widest on the left and narrows to almost nothing on the right, and raising representation quality mainly stretches the left-hand end further rather than lifting the whole curve. That shape is the single most important fact in this subsection: self-supervised pretraining buys <i>label efficiency</i>, not a higher ceiling. A team with a hundred thousand clean labels is standing at the right-hand end of this plot already, and the pretraining compute a paper reports at the left-hand end is not a promise about what they will see.</p>

${H.lab('labeff', 'The label-efficiency curve', 'Same architecture, same test set. One model trains only on the labelled subset; the other starts from a representation learned on the unlabelled pool. The gap at the left-hand end is the entire value proposition of self-supervision — and notice it closes at the right-hand end.')}

${H.probe([
      ['Why does contrastive learning need large batches?', 'The InfoNCE loss lower-bounds mutual information by at most $\\log B$, and the number of negatives per positive <i>is</i> the batch size — the police-line-up analogy above: a bigger line-up is a harder test, and only a genuinely good representation passes a hard test consistently. MoCo’s memory queue exists to break that coupling, supplying a large pool of negatives without requiring the batch itself to be enormous.'],
      ['What does the temperature do, exactly?', 'It scales the similarity logits before the softmax, exactly as the worked example computed: dividing by a small τ turns a modest gap between the true partner and a near-miss into a huge gap in the exponent, so the softmax collapses toward one-hot and the loss falls toward zero. Low τ concentrates the gradient on the single hardest negative — powerful when that negative is informative, unstable when it is a mislabelled near-duplicate; high τ spreads gradient evenly across every negative and under-trains, because nothing is being asked to work hard.'],
      ['Why throw away the projection head?', 'Empirically the layer before the head transfers better to downstream tasks. The head absorbs the invariances the contrastive task explicitly demands — colour jitter, cropping — which is exactly what you want for matching augmented views of the same photo, but a downstream task like counting objects may need colour or exact spatial layout, which the head has been trained to discard.'],
      ['When would you not bother with self-supervision?', 'Tabular data with genuinely overlapping classes, where the cluster assumption fails outright, or any setting where you already have plentiful labels — the label-efficiency curves converge at the right-hand end of the lab above, and the pretraining compute buys almost nothing once you are standing on that flat part of the curve.']
    ])}`,
    labs: {
      infonce: function (host) {
        const st = Viz.controls(host, [
          { k: 'temp', label: 'temperature τ', min: .02, max: 1, step: .01, value: .1, fmt: v => v.toFixed(2) },
          { k: 'noise', label: 'view noise (how different the two views are)', min: 0, max: 1.2, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'B', label: 'batch size B', min: 4, max: 12, step: 1, value: 8, fmt: v => v },
          { k: 'view', label: 'show', type: 'select', value: 'soft', options: [{ v: 'soft', t: 'softmax rows (what the loss sees)' }, { v: 'sim', t: 'raw cosine similarity' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'loss', label: 'InfoNCE loss', cls: 'key' },
          { k: 'acc', label: 'top-1 retrieval', cls: 'good' },
          { k: 'chance', label: 'chance' },
          { k: 'bound', label: 'MI bound log B' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const B = st.B, d = 16;
            const R = Num.rng(31);
            const base = Array.from({ length: B }, () => Array.from({ length: d }, () => R.normal(0, 1)));
            const A = base.map(v => v.map(x => x + R.normal(0, st.noise)));
            const Bv = base.map(v => v.map(x => x + R.normal(0, st.noise)));
            const res = Num.infoNCE(A, Bv, st.temp);
            const M = st.view === 'soft'
              ? res.sim.map(row => { const l = Num.logsumexp(row); return row.map(v => Math.exp(v - l)); })
              : res.sim.map(row => row.map(v => v * st.temp));
            const labels = Array.from({ length: B }, (_, i) => (st.view === 'soft' ? 'b' : 'b') + (i + 1));
            const P = Viz.plot(ctx, w, h, {
              xd: [0, 1], yd: [0, 1],
              pad: { l: 42, r: 16, t: 26, b: 34 }
            });
            P.heat(M, {
              rows: Array.from({ length: B }, (_, i) => 'a' + (i + 1)),
              cols: labels,
              lo: st.view === 'soft' ? 0 : -1, hi: 1,
              cell: v => B <= 8 ? v.toFixed(2) : v.toFixed(1)
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(st.view === 'soft' ? 'each row is a probability distribution over candidates — the diagonal should win'
                                            : 'cosine similarity before the temperature scaling', w / 2, h - 10);
            out({
              loss: res.loss.toFixed(4),
              acc: (res.acc * 100).toFixed(0) + '%',
              chance: (100 / B).toFixed(0) + '%',
              bound: Math.log(B).toFixed(3) + ' nats'
            });
          }
        });
        Viz.note(host, 'Set τ = 0.02: every row becomes almost one-hot, the loss collapses toward zero, and only the single hardest negative contributes any gradient — fast, and famously unstable. Set τ = 1: the rows go nearly uniform, the loss sits near $\\log B$, and the model learns slowly because every negative is treated as equally wrong. <b>0.05–0.1 is where published recipes sit, and this plot is why.</b>');
      },

      labeff: function (host) {
        const st = Viz.controls(host, [
          { k: 'quality', label: 'quality of the learned representation', min: 0, max: 1, step: .05, value: .6, fmt: v => v.toFixed(2) },
          { k: 'hard', label: 'task difficulty', min: .2, max: 1.4, step: .05, value: .7, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'at50', label: 'labels for 85%: from scratch', cls: 'bad' },
          { k: 'at50p', label: '… with pretraining', cls: 'good' },
          { k: 'save', label: 'label saving', cls: 'key' },
          { k: 'ceiling', label: 'gap at 10k labels' }
        ]);
        /* An honest model of the curves: accuracy ≈ ceiling − c·n^(−α), with
           pretraining raising the intercept and steepening the small-n regime. */
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const scratch = n => 0.965 - 0.62 * Math.pow(n, -0.30 * st.hard) - 0.02;
            const pre = n => 0.972 - (0.62 - 0.42 * st.quality) * Math.pow(n, -0.30 * st.hard - 0.06 * st.quality);
            const P = Viz.plot(ctx, w, h, { xd: [1, 4], yd: [.3, 1], pad: { l: 50, r: 14, t: 16, b: 40 } })
              .frame({ xticks: [1, 2, 3, 4], xfmt: v => Math.pow(10, v).toLocaleString(), xlabel: 'labelled examples', ylabel: 'test accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.band(Num.linspace(1, 4, 60).map(lx => [lx, pre(Math.pow(10, lx))]),
                     Num.linspace(1, 4, 60).map(lx => [lx, scratch(Math.pow(10, lx))]), { color: T.c3, alpha: .16 });
              P.fn(lx => scratch(Math.pow(10, lx)), { color: T.c2, width: 2.6, n: 120 });
              P.fn(lx => pre(Math.pow(10, lx)), { color: T.c3, width: 2.6, n: 120 });
              P.hline(.85, { color: T.faint, dash: [4, 4], label: '85% target' });
            });
            const need = f => { for (let lx = 1; lx <= 4.6; lx += .01) if (f(Math.pow(10, lx)) >= .85) return Math.pow(10, lx); return null; };
            const a = need(scratch), b = need(pre);
            out({
              at50: a ? Math.round(a).toLocaleString() : '>40,000',
              at50p: b ? Math.round(b).toLocaleString() : '>40,000',
              save: (a && b) ? (a / b).toFixed(1) + '× fewer labels' : '—',
              ceiling: ((pre(10000) - scratch(10000)) * 100).toFixed(1) + ' pts'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c2)', t: 'from scratch' }, { c: 'var(--c3)', t: 'from a self-supervised representation' }]);
        Viz.note(host, 'The shaded gap is the value of the unlabelled data, and it is <b>widest on the left and narrowest on the right</b>. That shape is the single most important fact about self-supervision: it buys label efficiency, not a higher ceiling. If you already have a hundred thousand clean labels, the pretraining compute is buying you almost nothing.');
      }
    },
    quiz: [
      {
        q: 'InfoNCE with batch size $B$ lower-bounds the mutual information by at most…',
        options: ['$B$', '$\\log B$', '$1/B$', 'it is unbounded'],
        answer: 1,
        why: 'The InfoNCE loss is provably a lower bound on the mutual information between the two views, and that bound cannot exceed $\\log B$ — intuitively, a $B$-way classifier can never carry more than $\\log B$ nats of information about which candidate is correct, whatever its accuracy. This is exactly why contrastive methods chase enormous batches (SimCLR trained at $B=8192$), and why MoCo introduced a queue of negatives: it decouples the number of negatives seen per step from the memory-bound batch size, buying a bigger effective $B$ without a bigger GPU.'
      },
      {
        q: 'Lowering the contrastive temperature τ…',
        options: ['spreads gradient evenly over negatives', 'concentrates the gradient on the hardest negatives', 'has no effect after normalisation', 'increases the batch size'],
        answer: 1,
        why: 'The worked example in §2.22.2 shows this happening in numbers: dividing similarities by a small τ before the softmax turns a modest gap into a huge one in the exponent, so the probability mass — and with it, the gradient — concentrates almost entirely on the single closest negative. "Spreads gradient evenly" is the tempting wrong answer because it describes what <i>raising</i> τ does; the direction is easy to flip under exam pressure, which is exactly why this is worth computing once by hand rather than memorising the sentence.'
      },
      {
        q: 'Pseudo-labelling is dangerous mainly because…',
        options: ['it is slow', 'errors are self-reinforcing and concentrate on subgroups the model already handles badly', 'it needs labelled data', 'it cannot use augmentation'],
        answer: 1,
        why: 'The worked arithmetic in §2.22.3 makes this concrete: a 25-point subgroup accuracy gap produces six times as many wrong pseudo-labels from the weak subgroup as from the strong one, and none of those wrong labels are distinguishable from right ones by the confidence filter that let them through. Confidence filtering selects for the model’s existing biases rather than correcting them, because a miscalibrated model is often most confident exactly where it is systematically wrong. Always check pseudo-label accuracy per subgroup before trusting a second training round.'
      },
      {
        q: 'The main practical benefit of self-supervised pretraining is…',
        options: ['a higher accuracy ceiling with unlimited labels', 'reaching a given accuracy with far fewer labels', 'faster inference', 'smaller models'],
        answer: 1,
        why: 'The label-efficiency curves in the lab converge at the right-hand end — with plentiful labels, the pretrained and from-scratch models end up in nearly the same place. The gain is entirely in the low-label regime, where pretraining supplies structure the labels alone could not. "A higher ceiling" is the tempting wrong answer because it is what a paper’s headline number can look like in isolation, but the same paper’s own learning curve, read all the way to the right, usually shows the gap closing.'
      }
    ],
    cards: [
      { q: 'InfoNCE, in one sentence', a: 'Cross-entropy over a $B\\times B$ similarity matrix with the identity as the label: pick your partner out of the batch.' },
      { q: 'Why MoCo has a queue', a: 'To decouple the number of negatives from the batch size, since the InfoNCE bound grows only as $\\log B$.' },
      { q: 'What stops BYOL collapsing', a: 'Asymmetric predictor + stop-gradient (+ EMA target). SimSiam showed the first two are the core.' },
      { q: 'The cluster assumption', a: 'Semi-supervised learning helps only if the decision boundary lies in a low-density region. Overlapping classes get nothing.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.23 */
  ML.section({
    id: 'active-transfer', track: 'classical', num: '2.23', level: 2,
    title: 'Active learning, transfer, and the label budget',
    lede: 'You have money for 2,000 labels. Which 2,000? And how much of the answer can you borrow from a model somebody else already trained? These two questions decide project timelines far more often than model choice does.',
    related: ['self-supervised', 'lora', 'features'],
    html: `
${H.tldr([
      'Active learning: label the examples the model is least sure about. Typical saving is 2–5× fewer labels for the same accuracy — <b>when the pool is large and the model is not yet good</b>.',
      'Uncertainty sampling alone collapses onto outliers and near-duplicates. Real systems combine uncertainty with <b>diversity</b> (batch-mode) and a slug of random samples to keep the labelled set representative.',
      'Transfer learning: freeze early layers, fine-tune late ones, and scale the learning rate down by 10–100×. For LLMs the modern answer is LoRA (§4.13), which makes the same trade with 0.1% of the parameters.'
    ])}

<p>You have a budget for 2,000 labels and a pool of two million unlabelled examples. The naive plan is to pick 2,000 of them at random and send them to be labelled. That plan treats every unlabelled example as equally worth knowing about, which is obviously false: some examples are ones your model would already classify correctly with total confidence, and paying a human to confirm what the model already knows is money spent on nothing. Somewhere in that pool of two million are the two thousand examples that would actually move the needle — the ones sitting right on the model's current decision boundary, where a label resolves real ambiguity instead of confirming a foregone conclusion. <b>Active learning</b> is the discipline of finding those examples before you pay for them.</p>

<h2><span class="sn">2.23.1</span> Active learning: which label is worth buying?</h2>
<p>The simplest scoring rules all ask the same underlying question — how sure is the model about this example? — and answer it by looking at the predicted class probabilities $p_1,\\dots,p_K$ in slightly different ways:</p>
${H.table(['Strategy', 'Score to maximise', 'Character'], [
      ['Least confidence', '$1-\\max_k p_k$', 'simple, works, ignores the shape of the rest of the distribution'],
      ['Margin', '$-(p_{(1)} - p_{(2)})$', 'usually the best of the cheap three'],
      ['Entropy', '$-\\sum_k p_k\\log p_k$', 'best when there are many classes'],
      ['Query-by-committee', 'disagreement across an ensemble', 'strong, costs an ensemble'],
      ['Expected model change', 'gradient magnitude if this label were added', 'principled, expensive'],
      ['<b>Core-set / diversity</b>', 'cover the embedding space', '<b>essential in batch mode</b> — otherwise you buy 500 near-duplicates'],
      ['BADGE', 'k-means++ over gradient embeddings', 'uncertainty and diversity in one score; a strong default']
    ])}

${H.worked('the same example, scored three ways — and the three rules disagree', `
<p>Take two three-class predictions. Example A: $p = (0.50,\\ 0.30,\\ 0.20)$. Example B: $p = (0.50,\\ 0.49,\\ 0.01)$.</p>
<p><b>Least confidence</b> looks only at the top probability, $1-\\max_k p_k$, and both examples have $\\max_k p_k = 0.50$, so least confidence scores them <i>identically</i> — it cannot tell them apart at all.</p>
<p><b>Margin</b> looks at the gap between the top two probabilities. For A, $p_{(1)}-p_{(2)} = 0.50-0.30 = 0.20$. For B, $p_{(1)}-p_{(2)} = 0.50-0.49 = 0.01$. Margin correctly flags B as the far more ambiguous example — the model is nearly torn between its top two guesses — which least confidence completely missed.</p>
<p><b>Entropy</b>, $-\\sum_k p_k\\log p_k$, gives A a score of 1.030 nats and B a score of 0.742 nats — the <i>opposite</i> ranking from margin. Entropy credits A for spreading real probability mass across all three classes, including a non-trivial 0.20 on the third option, whereas B's third class carries almost none. With only three classes this disagreement is a curiosity; with a hundred classes, where a long tail of small probabilities can hide a great deal of genuine uncertainty that margin — which only ever looks at the top two — cannot see, entropy's fuller accounting is why the table above calls it the better choice when $K$ is large.</p>`)}

${H.analogy(`<p>Picture a teacher choosing which flashcards to quiz a struggling student on, from a deck of two million. Quizzing a card the student already answers instantly and correctly wastes the lesson. Quizzing a card the student stares at blankly, mind elsewhere, is not much better — there is no partial knowledge there to sharpen, just noise. The valuable cards are the ones where the student clearly has <i>some</i> relevant knowledge and is one nudge away from getting it right: cards near the edge of what they already know.</p>
<p>That is uncertainty sampling's whole idea, and the analogy also exposes its failure mode directly. A blank stare and genuine on-the-edge uncertainty produce the same outward symptom — hesitation — and a scoring rule that cannot tell them apart will happily spend the whole lesson on flashcards written in a language the student has never seen. In active learning those cards are outliers, and the next box names the fix.</p>`)}

<p><b>What you are looking at.</b> A real logistic model, retrained from scratch after every batch of new labels, shown on the same two-moons dataset used throughout Part 2. The left panel plots test accuracy against the number of labels bought so far, one curve for random sampling and one for uncertainty sampling. The right panel shows the raw unlabelled pool as faint dots, with every point the active learner actually chose to label drawn larger and coloured by its true class — this is a map of where the budget went, not a performance chart.</p>
<p><b>What to do with it.</b> Leave outlier contamination and the random-mix slider at zero and watch the orange (active) curve pull ahead of the blue (random) one, then look at the right-hand panel: the enlarged points should sit almost entirely on the boundary between the two moons, exactly where a label is genuinely informative. Now raise <b>outlier contamination</b> toward 10% and watch the orange curve fall <i>below</i> the blue one.</p>
<p><b>The thing genuinely worth noticing.</b> With contamination on, the right-hand panel shows the active learner's budget scattered out into empty space, chasing points that are uncertain only because they are corrupted noise far from the real data manifold — the blank-stare flashcards from the analogy above. Now push the <b>random samples mixed in</b> slider up to 30% and press through the rounds again: most of the lost accuracy comes back, because a slug of random queries keeps at least some of the budget anchored to the real distribution regardless of what the uncertainty score says. <b>That mixture is why no production active-learning system runs pure uncertainty sampling</b>, and the pitfall below names the other two reasons.</p>

${H.lab('active', 'Active learning against random sampling', 'One real logistic model, retrained after every batch of labels. Blue buys labels at random; orange buys where the model is least certain. Both start from the same three seed labels. Watch where the orange queries land — and what happens when you turn the outlier contamination up.')}

${H.pitfall('Uncertainty sampling has three well-known pathologies. <b>Outliers</b> are maximally uncertain and maximally useless, exactly as the flashcard analogy above predicts. <b>Batch redundancy</b>: the top-500 most uncertain points are usually 500 near-copies of the same one ambiguous case, because whatever makes one example hard usually makes its neighbours hard too — a plain uncertainty score has no way to notice it already asked this question. <b>Sampling bias</b>: the labelled set stops resembling the deployment distribution, so your validation estimates drift away from reality even as your training accuracy looks fine. The standard mitigation is 10–20% random queries mixed in, exactly as the lab demonstrates, plus a diversity term in the batch score — which is what core-set and BADGE add on top of a plain uncertainty rule.')}

<h2><span class="sn">2.23.2</span> Transfer learning: what to freeze</h2>
<p>Active learning answers "which labels are worth buying." Transfer learning answers a different budget question: how much of the labelling can you skip entirely by starting from a model someone else already trained on a related task? The two questions compound — a good starting representation changes which examples are still ambiguous enough to be worth an active-learning query — which is why they share a section.</p>
${H.table(['Your data', 'Similar domain to the source', 'Different domain'], [
      ['<b>Small</b> (< 1k)', 'freeze everything, train a linear head', 'freeze most, train the last block + head; expect trouble'],
      ['<b>Medium</b> (1k–100k)', 'fine-tune the last few blocks, low LR', 'fine-tune more of the network, moderate LR'],
      ['<b>Large</b> (> 100k)', 'fine-tune everything, low LR', 'fine-tune everything, or train from scratch and compare']
    ])}
${H.intuition(`<p>The reason this table has that shape: early layers learn features that are nearly universal — edges, textures, token statistics — while late layers learn features specific to the source task. The more your task differs, the further down the specificity starts, and the more you must retrain. The more data you have, the more you can afford to retrain without overfitting the extra capacity you have just unfrozen.</p>`)}

${H.worked('discriminative learning rates, computed down a 4-block network', `
<p>Fine-tuning all of a network at one learning rate risks two failure modes at once: too high for the early, already-good general features (which get destroyed) and too low for the late, task-specific ones (which barely move). Discriminative learning rates fix this by dividing the rate by a constant, commonly 2.6, going from the last block back to the first.</p>
<p>Starting from a base rate of $10^{-3}$ at the last (fourth) block: block 3 gets $10^{-3}/2.6 = 3.85\\times10^{-4}$; block 2 gets $3.85\\times10^{-4}/2.6 = 1.48\\times10^{-4}$; block 1, the earliest, gets $1.48\\times10^{-4}/2.6 = 5.69\\times10^{-5}$ — roughly $17.6\\times$ smaller than the last block's rate, since $2.6^3 \\approx 17.6$. The earliest layers move at under a seventeenth of the pace of the last ones, which is the numeric expression of "trust the general features, actively retrain the specific ones" from the table above.</p>`)}

${H.table(['Technique', 'What it does', 'When'], [
      ['Linear probing', 'freeze the backbone, fit a linear classifier on the features', 'tiny data; also the cleanest way to <i>measure</i> a representation'],
      ['Discriminative learning rates', 'lower LR for earlier layers (e.g. ÷2.6 per block, as computed above)', 'fine-tuning a deep network; a fastai staple'],
      ['Gradual unfreezing', 'train the head, then progressively unfreeze downward', 'small data, avoids destroying pretrained features'],
      ['<b>LoRA / adapters</b>', 'train small injected matrices, freeze the base', '<b>the default for LLMs</b> (§4.13)'],
      ['LP-FT', 'linear probe first, <i>then</i> fine-tune', 'measurably better out-of-distribution than fine-tuning directly']
    ])}
${H.history(`<p>Transfer learning's modern form starts with the 2014 observation that features trained on ImageNet transferred usefully to almost any other vision task — Yosinski and colleagues showed exactly how transferable each layer was by freezing progressively fewer of them and measuring the drop. For most of the next five years "transfer learning" meant precisely the freeze/fine-tune table above: take a big pretrained network, decide how many layers to unfreeze based on how much data and how different a domain you had.</p>
<p>Adapters (2019) and then LoRA (2021) changed the trade rather than the principle. Instead of choosing <i>which existing layers</i> to update, they inject small new matrices alongside the frozen original ones and train only those — the same freeze-most-of-it instinct, but now the frozen fraction is 99.9% rather than "the first two blocks", which is why LoRA is covered on its own in §4.13 rather than folded into this table.</p>`)}
${H.flag('<b>Catastrophic forgetting</b> is the cost: fine-tune hard on a narrow task and the model loses capability elsewhere. This is not hypothetical for LLMs — supervised fine-tuning on a domain corpus routinely degrades general instruction-following, which is one reason LoRA (whose base weights are untouched) and mixed replay data are standard practice.')}

<h2><span class="sn">2.23.3</span> Few-shot, and where the boundary now sits</h2>
${H.vs('Fine-tune a small model', [
      'Needs ~500–5,000 labels to beat prompting on a narrow task',
      'Cheap and fast at inference; runs anywhere',
      'You own the artefact and can version it',
      'Retraining is a project every time the task shifts'
    ], 'Few-shot prompt a large model', [
      'Needs 5–50 examples in the prompt',
      'Expensive per call, and the latency is the model’s',
      'Behaviour changes when the provider updates the model',
      'Changing the task is an edit to a text file'
    ])}

${H.worked('why the crossover is a volume question, not just a label-count one', `
<p>Suppose labels cost \\$2 each to collect and fine-tuning compute is a one-off \\$50. Getting a small model to a usable accuracy takes, say, 1,000 labels: total cost $\\$2{,}000 + \\$50 = \\$2{,}050$, paid once, however many predictions you later make.</p>
<p>Few-shot prompting a large model needs no label collection at all, but every single call costs something — say \\$0.01, dominated by the large model's compute. Prompting is cheaper than the \\$2,050 fine-tuning bill for the first $205{,}000$ calls ($2{,}050/0.01$), and more expensive than it for every call after that, because the fine-tuned model's marginal cost per prediction is close to zero while the large model's is not.</p>
<p>The exact crossover volume moves with your actual label and inference prices, but the shape does not: a one-off labelling cost against a per-call cost always crosses <i>somewhere</i>, and the crossover point is a volume, not just the label count the tl;dr quotes. A prototype expecting a few hundred calls should prompt regardless of how cheap labels are; a product expecting millions should fine-tune almost regardless of how expensive they are.</p>`)}

${H.key('The crossover for a well-specified classification task sits at roughly a few hundred to a few thousand labels — below it, prompt; above it, fine-tune a small model and save the inference cost. §5.13 turns this into a decision procedure.')}

${H.probe([
      ['Why mix random samples into an active-learning loop?', 'To keep the labelled set representative, so that validation estimates remain meaningful and the model does not chase a distribution of its own making. The lab demonstrates this directly: with outlier contamination on, a 30% random mix recovers most of the accuracy that pure uncertainty sampling lost by chasing noise.'],
      ['Your active-learning run does worse than random. What happened?', 'Almost certainly outliers or batch redundancy — the model is treating corrupted or duplicate-adjacent points as maximally informative when they are maximally useless, exactly the blank-stare flashcards from the analogy. It could also be that the model was too weak at the start for its uncertainty estimates to mean anything yet; warm up with a round or two of random labels before switching to active queries.'],
      ['How do you choose how many layers to freeze?', 'By data size and domain distance, using the table in §2.23.2 as a starting point, then verified empirically: linear-probe accuracy — freeze everything, fit only a linear head — tells you how good the frozen features already are on their own, which calibrates your expectation for how much full fine-tuning should be able to add on top.'],
      ['What is LP-FT and why does it help?', 'Linear-probe first to get a sensible head, then fine-tune everything. Starting fine-tuning with a randomly initialised head sends large, essentially arbitrary gradients backward into the pretrained features before the head has learned anything sensible to predict, which measurably damages out-of-distribution accuracy compared with letting the head settle first.']
    ])}`,
    labs: {
      active: function (host) {
        const st = Viz.controls(host, [
          { k: 'batch', label: 'labels per round', min: 1, max: 20, step: 1, value: 5, fmt: v => v },
          { k: 'rounds', label: 'rounds', min: 1, max: 30, step: 1, value: 14, fmt: v => v },
          { k: 'outliers', label: 'outlier contamination', min: 0, max: .2, step: .01, value: 0, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'mix', label: 'random samples mixed in', min: 0, max: .6, step: .05, value: 0, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rand', label: 'random-sampling accuracy', cls: 'bad' },
          { k: 'act', label: 'active accuracy', cls: 'good' },
          { k: 'labels', label: 'labels spent', cls: 'key' },
          { k: 'equiv', label: 'random needs' }
        ]);
        const pool = Num.dataset('moons', 600, .28, 21);
        const test = Num.dataset('moons', 500, .28, 88);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(9);
            const X = pool.X.map((p, i) => i / pool.X.length < st.outliers ? [p[0] + R.normal(0, 4), p[1] + R.normal(0, 4)] : p);
            const y = pool.y;

            function runStrategy(active) {
              const labelled = [0, 1, 2, 300, 301];
              const curve = [];
              for (let r = 0; r < st.rounds; r++) {
                const Xl = labelled.map(i => X[i]), yl = labelled.map(i => y[i]);
                let m = null;
                if (new Set(yl).size > 1) {
                  m = Num.logistic(Xl.map(p => [p[0], p[1], p[0] * p[1], p[0] * p[0], p[1] * p[1]]), yl, { lr: .3, l2: .02 });
                  m.step(400);
                }
                const feat = p => [p[0], p[1], p[0] * p[1], p[0] * p[0], p[1] * p[1]];
                const acc = m ? test.X.filter((p, i) => (m.predict(feat(p)) > .5 ? 1 : 0) === test.y[i]).length / test.X.length : .5;
                curve.push([labelled.length, acc]);
                // choose the next batch
                const unl = [];
                for (let i = 0; i < X.length; i++) if (labelled.indexOf(i) < 0) unl.push(i);
                let pick;
                if (active && m) {
                  const nRand = Math.round(st.batch * st.mix);
                  const scored = unl.map(i => ({ i: i, u: -Math.abs(m.predict(feat(X[i])) - .5) }));
                  scored.sort((a, b) => b.u - a.u);
                  pick = scored.slice(0, st.batch - nRand).map(s => s.i);
                  for (let k = 0; k < nRand; k++) pick.push(unl[R.int(unl.length)]);
                } else {
                  pick = [];
                  for (let k = 0; k < st.batch; k++) pick.push(unl[R.int(unl.length)]);
                }
                pick.forEach(i => { if (labelled.indexOf(i) < 0) labelled.push(i); });
              }
              return { curve: curve, labelled: labelled };
            }
            const rnd = runStrategy(false), act = runStrategy(true);

            const w1 = w * .52;
            const P = Viz.plot(ctx, w1, h, {
              xd: [0, Math.max(20, 5 + st.batch * st.rounds)], yd: [.45, 1],
              pad: { l: 46, r: 10, t: 16, b: 38 }
            }).frame({ xlabel: 'labels bought', ylabel: 'test accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.line(rnd.curve, { color: T.c1, width: 2.4 });
              P.line(act.curve, { color: T.c4, width: 2.4 });
              P.dots(rnd.curve, { r: 2.4, color: T.c1 });
              P.dots(act.curve, { r: 2.4, color: T.c4 });
            });

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, { xd: [-3, 3.2], yd: [-2.2, 2.6], pad: { l: 8, r: 8, t: 16, b: 30 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'where the active learner spent its budget' });
            P2.clip(() => {
              X.forEach((p, i) => P2.dots([[p[0], p[1]]], { r: 1.8, color: T.line, alpha: .8 }));
              act.labelled.forEach(i => P2.dots([[X[i][0], X[i][1]]], { r: 3.6, color: y[i] ? T.c2 : T.c1, stroke: true, strokeWidth: 1.2 }));
            });
            ctx.restore();

            const aLast = act.curve[act.curve.length - 1][1], rLast = rnd.curve[rnd.curve.length - 1][1];
            const eq = rnd.curve.find(c => c[1] >= aLast);
            out({
              rand: (rLast * 100).toFixed(1) + '%',
              act: (aLast * 100).toFixed(1) + '%',
              labels: act.curve[act.curve.length - 1][0],
              equiv: eq ? eq[0] + ' labels' : 'more than ' + rnd.curve[rnd.curve.length - 1][0]
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: 'random sampling' }, { c: 'var(--c4)', t: 'uncertainty sampling' }]);
        Viz.note(host, 'The right-hand panel shows where the active learner spent its money: almost every label sits on the boundary between the two moons, which is exactly where a label is informative. Now push <b>outlier contamination</b> to 10% and active learning collapses below random — it spends everything on garbage far from the data. Then add 30% random samples mixed in and watch it recover most of the loss. <b>That mixture is why production active-learning loops are never pure uncertainty sampling.</b>');
      }
    },
    quiz: [
      {
        q: 'Pure uncertainty sampling in batch mode typically fails because…',
        options: ['it is too slow', 'the top-k most uncertain points are near-duplicates of each other', 'it needs a calibrated model', 'it cannot handle multi-class'],
        answer: 1,
        why: 'Whatever makes one example ambiguous usually makes its near-neighbours ambiguous too, so a plain uncertainty score has no mechanism for noticing it has already asked essentially this question — you buy 500 labels for one ambiguous case, as the lab shows when contamination pulls every query toward the same corrupted region. Diversity terms (core-set, BADGE) exist specifically to fix this by scoring coverage of the embedding space, not just per-point uncertainty.'
      },
      {
        q: 'You have 800 labels and a source model from a very different domain. The right move is…',
        options: ['freeze everything and fit a linear head', 'fine-tune the last block and the head, with a low learning rate', 'train from scratch', 'fine-tune everything at the pretraining learning rate'],
        answer: 1,
        why: 'A different domain means the transferable, general-purpose features stop earlier in the network — the intuition box in §2.23.2 explains why — so more of the network genuinely needs updating than "freeze everything" would allow. But 800 labels is nowhere near enough data to support updating <i>all</i> of it without overfitting, which rules out the fourth option. Fine-tuning the last block plus the head, at a low rate, is the compromise the table in §2.23.2 recommends for exactly this small-data, different-domain cell.'
      },
      {
        q: 'LP-FT (linear probe then fine-tune) helps because…',
        options: ['it trains faster', 'a randomly initialised head sends large distorting gradients into the pretrained features', 'it uses less memory', 'it prevents overfitting on the head'],
        answer: 1,
        why: 'A freshly initialised head has no idea what it is doing, so the very first backward pass of ordinary fine-tuning sends large, essentially arbitrary gradients through the whole network trying to fix it — and those gradients pass straight through the pretrained features on their way, distorting representations that were already good before the head had learned anything sensible to predict. Linear-probing first gets the head roughly right using gradients that never touch the backbone, so the fine-tuning phase that follows starts from small, well-aimed corrections instead. The effect is measurably largest out of distribution, which is the population most exposed to a damaged representation.'
      }
    ],
    cards: [
      { q: 'The three active-learning pathologies', a: 'Outliers, batch redundancy, and sampling bias that invalidates the validation set. Mitigate with diversity terms and 10–20% random queries.' },
      { q: 'Freeze-or-fine-tune, in one line', a: 'More data and more domain distance → unfreeze more. Small + similar → linear probe.' },
      { q: 'Catastrophic forgetting', a: 'Fine-tuning hard on a narrow task degrades everything else; LoRA and replay data are the standard defences.' },
      { q: 'Prompt vs fine-tune crossover', a: 'Roughly a few hundred to a few thousand labels for a well-specified classification task.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.24 */
  ML.section({
    id: 'ranking', track: 'classical', num: '2.24', level: 2,
    title: 'Learning to rank',
    lede: 'Search, feeds, recommendations and the reranking stage of RAG (§5.1) are all the same problem: given a query and a set of candidates, produce an order. The loss is not accuracy and the metric is not AUC, and getting that distinction right is most of the job.',
    related: ['rag', 'metrics', 'gnn'],
    html: `
${H.tldr([
      'Ranking metrics are <b>position-weighted</b>: being right at rank 1 is worth far more than at rank 10. NDCG@k with a $1/\\log_2(i+1)$ discount is the standard.',
      'Three loss families: <b>pointwise</b> (predict the grade), <b>pairwise</b> (get every pair in the right order), <b>listwise</b> (optimise the metric directly, approximately). LambdaMART — pairwise gradients weighted by the NDCG change — is still the tabular workhorse.',
      'Click data is <b>position-biased</b>: users click the top result because it is at the top. Training on raw clicks teaches the model to reproduce the ranker that generated them.'
    ])}

<p>Hand a recruiter a stack of eight candidate résumés, all pre-screened as plausible fits, and ask them to work through it. They read the first one carefully. They read the second one carefully too, maybe. By the fifth, they are skimming, and there is a real chance the eighth never gets opened at all, however strong the candidate. A classifier trained to spot "plausible fit" and scored on ordinary accuracy would treat all eight identically — get the label right on each one independently and the accuracy number does not care what order you handed them over in. But the recruiter's attention very much does, and so does whoever is paying for the search to find a hire. <b>Learning to rank</b> exists because ordinary accuracy is the wrong objective for exactly this shape of problem, and getting that one substitution right — order instead of correctness — is most of what distinguishes a ranking system from a classifier with extra steps.</p>

<h2><span class="sn">2.24.1</span> The metrics</h2>
<p>If position matters, a metric has to weight early mistakes more than late ones. <b>NDCG</b> (normalised discounted cumulative gain) does this with two ideas stacked together: convert each document's relevance grade into a <i>gain</i>, and shrink that gain by a factor that gets smaller the further down the list it sits.</p>
$$\\mathrm{DCG@}k = \\sum_{i=1}^{k}\\frac{2^{rel_i}-1}{\\log_2(i+1)},\\qquad \\mathrm{NDCG@}k = \\frac{\\mathrm{DCG@}k}{\\mathrm{IDCG@}k}$$
<p>Read the DCG sum term by term: $rel_i$ is the graded relevance of whatever document sits at position $i$ — say 0 to 3, from irrelevant to perfect — and $2^{rel_i}-1$ turns that grade into a gain that grows steeply with relevance, so a perfect result at a given position is worth far more than a merely-good one, not just a bit more. The denominator $\\log_2(i+1)$ is the position discount: at $i=1$ it is $\\log_2 2 = 1$, no discount at all, and it grows slowly from there, so gains from deep positions are divided down toward nothing. Summing the discounted gains gives DCG, and dividing by IDCG — the DCG of the best possible ordering of the same documents — rescales the result to sit between 0 and 1, which is what makes NDCG comparable across queries that have different numbers of relevant documents.</p>

${H.worked('the same four documents, two orderings, two very different scores', `
<p>Take four documents with relevance grades $\\{3, 2, 1, 0\\}$ — one excellent, one good, one marginal, one useless. The best possible order is $[3,2,1,0]$, giving $\\mathrm{IDCG@4} = \\frac{2^3-1}{\\log_2 2} + \\frac{2^2-1}{\\log_2 3} + \\frac{2^1-1}{\\log_2 4} + \\frac{2^0-1}{\\log_2 5} = 7.000 + 1.893 + 0.500 + 0 = 9.393$.</p>
<p><b>Ordering A</b> — $[3, 2, 0, 1]$, a small mistake buried at the bottom of the list, swapping the two least-relevant documents: $\\mathrm{DCG@4} = 9.323$, giving $\\mathrm{NDCG@4} = 9.323/9.393 = 0.993$. Barely a dent.</p>
<p><b>Ordering B</b> — $[0, 2, 3, 1]$, the same four documents, but now the excellent one has been pushed down to position 3: $\\mathrm{DCG@4} = 5.823$, giving $\\mathrm{NDCG@4} = 5.823/9.393 = 0.620$. A single swap involving position 1, on the exact same four documents that scored 0.993 a moment ago, costs 37 points of NDCG.</p>
<p>Nothing about the <i>set</i> of documents changed between A and B, and nothing about the classifier's per-document correctness needs to have changed either — this is purely the cost of where a mistake happens. That is what "position-weighted" means in the tl;dr, made concrete.</p>`)}

${H.table(['Metric', 'What it measures', 'Use when'], [
      ['<b>NDCG@k</b>', 'graded relevance with a position discount, normalised to [0,1]', 'graded judgements exist; the default for search'],
      ['MRR', '$1/\\text{rank of the first relevant result}$', 'exactly one right answer — question answering, RAG'],
      ['MAP@k', 'mean of precision at each relevant hit', 'binary relevance, several right answers'],
      ['Recall@k', 'did the right thing make the top $k$ at all', '<b>the retrieval stage</b>, where a reranker follows (§5.1)'],
      ['Hit rate@k', 'binary: any relevant item in the top $k$', 'recommendations, simple dashboards'],
      ['Expected reciprocal rank', 'models the user stopping after a good result', 'when the cascade assumption matters']
    ])}
${H.key('Recall@k for the retriever, NDCG or MRR for the reranker. Confusing them is the most common ranking-evaluation error: a retriever that scores 0.8 recall@100 and 0.3 NDCG@10 is doing its job perfectly.')}

<p><b>What you are looking at.</b> Eight documents, each with a graded relevance score, listed in the current order; the bar chart underneath is the position discount curve $1/\\log_2(i+1)$ itself, so you can see the shrinking multiplier directly rather than trust the formula. The readout below computes DCG, IDCG, NDCG, MRR, AP and recall on the list exactly as it currently stands.</p>
<p><b>What to do with it.</b> Use the up/down buttons to reproduce the worked example: start from the ideal order, then swap the two lowest-relevance documents at the bottom and watch NDCG barely move; reset to ideal, then move the top document down two spots and watch NDCG drop hard.</p>
<p><b>The thing genuinely worth noticing.</b> The bar chart's height at rank 1 versus rank 8 <i>is</i> the reason those two swaps cost so differently — a swap's damage is the discount at its position, not some abstract notion of "how wrong" the reordering feels. A pointwise classifier that gets every single relevance grade correct but happens to output them in the wrong order scores perfectly on accuracy and can still score badly here, which is the whole justification for §2.24.2's insistence that the loss function should know about position too.</p>

${H.lab('ndcg', 'Reorder the list and watch the metrics move', 'Eight documents with graded relevance. Move them and every metric updates. The bar chart is the position discount — that is what makes a swap at ranks 1–2 cost more than the same swap at ranks 7–8.')}

<h2><span class="sn">2.24.2</span> The three loss families</h2>
${H.table(['Family', 'Loss on', 'Example', 'Trade'], [
      ['<b>Pointwise</b>', 'one document at a time — regress the grade or classify relevant/not', 'MSE, logistic', 'simplest; ignores that only the <i>order</i> matters, and wastes capacity on calibrating absolute grades'],
      ['<b>Pairwise</b>', 'pairs — penalise every inversion', 'RankNet: $\\log(1+e^{-\\sigma(s_i-s_j)})$', 'matches the task better; treats all inversions as equally bad, which the metric does not'],
      ['<b>Listwise</b>', 'the whole list', 'ListNet, LambdaRank/LambdaMART, softmax-CE over the list', 'closest to the metric; LambdaMART weights each pair’s gradient by $|\\Delta\\text{NDCG}|$ from swapping them']
    ])}
${H.intuition(`<p>LambdaRank's insight is beautifully pragmatic. NDCG is a step function of the scores — it changes only when two documents swap order — so it has zero gradient almost everywhere and is not directly optimisable by gradient descent, the way every other loss in this course is. Rather than smooth the metric into something differentiable, LambdaRank writes down the <i>gradient it wishes it had</i>. For a pair of documents $i,j$ with $i$ more relevant than $j$, RankNet already has a pairwise gradient, $\\lambda_{ij} = \\dfrac{-\\sigma}{1+e^{\\sigma(s_i-s_j)}}$, pushing $i$'s score above $j$'s. LambdaRank simply multiplies that gradient by $|\\Delta\\mathrm{NDCG}_{ij}|$, the change in NDCG if the two swapped places — the exact quantity the worked example above computed for two specific swaps. There is no loss function this is formally the derivative of, and it trains anyway. It was later shown to optimise a bound on the metric, which is a nice story about practice arriving first and theory catching up afterward.</p>`)}

<h2><span class="sn">2.24.3</span> Position bias, and how to correct it</h2>
<p>Users click the first result disproportionately, whatever it is — the recruiter reading résumé one carefully and barely glancing at résumé eight, again. If you train on raw clicks, you are not learning what is relevant; you are learning what was already on top, which is the previous ranker's opinion reflected back at you.</p>

${H.worked('the same relevance, two positions, two very different click rates', `
<p>Suppose two documents are genuinely equally relevant — each would satisfy the user 30% of the time it is actually looked at. One sits at position 1, where the examination probability (the chance a user looks at it at all) is essentially 1.0. The other sits at position 5, where a typical examination-decay curve puts the probability at roughly 0.041 — about 24 times rarer.</p>
<p>The <i>observed</i> click-through rate is relevance times examination: position 1 shows $1.0 \\times 0.30 = 0.30$, a healthy 30% CTR. Position 5 shows $0.041 \\times 0.30 \\approx 0.012$, a 1.2% CTR. Read naively, these look like wildly different documents — one is "30% relevant", the other looks "1.2% relevant" — when in truth they are identical. Divide each observed CTR by its own examination probability, $0.30/1.0 = 0.30$ and $0.012/0.041 \\approx 0.30$, and both recover the same true relevance exactly. That division is inverse propensity weighting, in miniature.</p>`)}

${H.steps([
      '<b>Inverse propensity weighting.</b> Estimate $p_i$, the probability a result at position $i$ is examined, and weight each click by $1/p_i$, exactly as the worked example just did. Unbiased, but high variance for deep positions — clip the weights.',
      '<b>Randomisation / interleaving.</b> Occasionally swap adjacent results, or interleave two rankers’ outputs, and measure preference directly. This buys unbiased data at a small cost in user experience, and interleaving needs an order of magnitude fewer sessions than an A/B test for the same power (§2.25 covers why: it removes between-user variance entirely, the same logic as a paired test).',
      '<b>Position as a feature.</b> Include the displayed position during training and set it to a constant at inference. Cheap, widely used, and only approximately correct — it assumes the bias is additive and separable.'
    ])}

<p><b>What you are looking at.</b> The green curve is the true, unobservable relevance at each of ten positions — deliberately <i>not</i> monotonically decreasing, because the ranker generating these clicks is imperfect and sometimes buries a good result. The blue curve is what your click logs actually show, and the dashed grey curve is the examination probability itself, scaled for comparison.</p>
<p><b>What to do with it.</b> With inverse propensity weighting off, read the correlation readout between observed CTR and true relevance at the default examination decay — it should already be well below 1. Now switch IPW on and watch the blue curve snap onto the green one, reproducing the recovery the worked example computed by hand.</p>
<p><b>The thing genuinely worth noticing.</b> Push the examination decay slider higher and the blue (observed) curve increasingly looks like nothing but a scaled copy of the grey (examination) curve — the correlation with true relevance keeps falling even though nothing about actual relevance changed. <b>Your click logs are measuring your old ranker's opinion of itself, not the world</b>, and the only reason IPW can fix it is that you happened to know the examination probabilities — which is exactly why randomisation and interleaving exist for the cases where you do not.</p>

${H.lab('posbias', 'What position bias does to your training data', 'The same underlying relevance, observed through a clicking user. Drag the examination decay and watch the click-through-rate curve detach from the relevance curve — and watch what inverse propensity weighting recovers.')}

${H.flag('Two-stage ranking is universal because it is the only affordable shape: a cheap retriever over millions of items (BM25, ANN over embeddings — §5.10), then an expensive reranker over the top 50–200 (cross-encoder, gradient-boosted trees). <b>The retriever’s recall is a hard ceiling on the whole system</b> — a reranker cannot promote a document the retriever never returned, which is why recall@100 is the number to watch during retrieval development.')}

${H.analogy(`<p>Two-stage ranking is how a hiring pipeline actually works, and it is worth naming the analogy explicitly because it explains why nobody builds it any other way. A recruiter does not carefully read every résumé in the applicant pool for every role — that pool might be a hundred thousand people. Instead, an inexpensive first pass (keyword search, a degree requirement, a location filter) cuts that pool down to a shortlist of fifty, cheaply and a little crudely. Only then does anyone spend real, expensive attention — an actual interview — carefully ordering those fifty. If the cheap first pass wrongly discards the best candidate in the whole pool, no amount of careful interviewing recovers them; they were never on the shortlist to begin with. That is recall@k versus NDCG@k in one sentence: the cheap stage’s job is to not lose the answer, the expensive stage’s job is to order what it was given.</p>`)}

${H.history(`<p>Learning to rank as its own subfield dates to the mid-2000s, when Microsoft Research’s RankNet (Burges et al., 2005) introduced the pairwise neural approach in §2.24.2’s table, trained on Bing’s search logs. LambdaRank followed within a couple of years, solving the "NDCG has no useful gradient" problem the intuition box above describes. The moment that cemented LambdaMART as the default tabular ranker was the 2010 Yahoo! Learning to Rank Challenge, an open competition with real search-engine data, which LambdaMART-based ensembles won decisively — the paper trail is a rare case of an industry benchmark directly settling which algorithm a whole field would standardise on for the next decade.</p>
<p>The more recent shift is the reranker itself, not the loss function: cross-encoder transformers (BERT-style models scoring a query and a single document jointly) now routinely replace or sit alongside LambdaMART in the second stage, trading LambdaMART’s speed for a deeper, more contextual notion of relevance — the two-stage <i>shape</i> from the analogy above has stayed fixed since RankNet; only what fills the expensive second box has changed.</p>`)}

${H.probe([
      ['Why NDCG rather than accuracy?', 'Ranking cares about order and position, not about getting each grade right — the worked example in §2.24.1 shows the same four documents scoring 0.993 or 0.620 purely depending on where one mistake sits. NDCG applies a position discount and normalises against the best possible ordering, so scores compare across queries with different numbers of relevant documents, which accuracy has no mechanism to do.'],
      ['What does LambdaMART actually differentiate?', 'Nothing, strictly — it specifies a gradient rather than deriving one from a loss function, exactly as §2.24.2 shows: RankNet’s pairwise gradient, scaled by $|\\Delta\\mathrm{NDCG}|$ for the pair in question. The tempting wrong answer is to assume there must be some loss being differentiated in the usual way, because that is how every other model in this course works — LambdaMART is the deliberate exception, and it provably optimises a bound on NDCG despite having no explicit objective.'],
      ['Your CTR model retrained on its own logs keeps degrading. Why?', 'A feedback loop through position bias: the model learns that top positions get clicks, promotes what it already promoted, and the resulting logs confirm the belief right back — exactly the mechanism in the two-position worked example, compounding round after round instead of correcting itself. Break it with randomisation, interleaving, or inverse propensity weighting; simply retraining more often makes the loop tighter, not weaker.'],
      ['A reranker improves NDCG@10 by 8 points but the end-to-end metric barely moves. Explain.', 'The retriever’s recall@k is the bottleneck, not the reranker. A reranker can only reorder the candidates it is handed — if the genuinely best document never made it into that candidate set in the first place, no amount of reordering recovers it. Measure recall at the reranker’s input size before concluding the reranker itself is the weak link.']
    ])}`,
    labs: {
      ndcg: function (host) {
        const el = ML.el;
        let rels = [3, 0, 2, 3, 1, 0, 1, 0];
        const st = Viz.controls(host, [
          { k: 'k', label: 'cut-off k', min: 1, max: 8, step: 1, value: 5, fmt: v => v }
        ], () => draw());
        const listHost = el('div', { style: 'margin:8px 0' });
        host.appendChild(listHost);
        const out = Viz.readout(host, [
          { k: 'dcg', label: 'DCG@k' }, { k: 'idcg', label: 'ideal DCG@k' },
          { k: 'ndcg', label: 'NDCG@k', cls: 'key' }, { k: 'mrr', label: 'MRR' },
          { k: 'ap', label: 'AP@k' }, { k: 'rec', label: 'recall@k' }
        ]);

        function draw() {
          listHost.innerHTML = '';
          rels.forEach((r, i) => {
            const row = el('div', {
              style: 'display:flex;align-items:center;gap:9px;padding:6px 10px;margin-bottom:4px;border-radius:8px;' +
                'border:1px solid var(--line);background:' + (i < st.k ? 'var(--panel)' : 'transparent') +
                ';opacity:' + (i < st.k ? 1 : .5)
            });
            row.appendChild(el('span', { style: 'font-family:var(--mono);font-size:11px;color:var(--faint);width:26px', text: '#' + (i + 1) }));
            row.appendChild(el('span', {
              style: 'font-family:var(--sans);font-size:13px;flex:1',
              html: 'document ' + String.fromCharCode(65 + i) + ' &nbsp;<b style="color:' +
                (r >= 3 ? 'var(--green)' : r >= 1 ? 'var(--amber)' : 'var(--faint)') + '">rel = ' + r + '</b>'
            }));
            row.appendChild(el('span', {
              style: 'font-family:var(--mono);font-size:11px;color:var(--blue)',
              text: 'gain ' + ((Math.pow(2, r) - 1) / Math.log2(i + 2)).toFixed(3)
            }));
            row.appendChild(el('button', { class: 'btn', style: 'padding:3px 8px;font-size:11px', text: '↑', onclick: () => { if (i > 0) { const t = rels[i]; rels[i] = rels[i - 1]; rels[i - 1] = t; draw(); S.redraw(); } } }));
            row.appendChild(el('button', { class: 'btn', style: 'padding:3px 8px;font-size:11px', text: '↓', onclick: () => { if (i < rels.length - 1) { const t = rels[i]; rels[i] = rels[i + 1]; rels[i + 1] = t; draw(); S.redraw(); } } }));
            listHost.appendChild(row);
          });
          const k = st.k;
          out({
            dcg: Num.dcg(rels, k).toFixed(4),
            idcg: Num.dcg(rels.slice().sort((a, b) => b - a), k).toFixed(4),
            ndcg: Num.ndcg(rels, k).toFixed(4),
            mrr: Num.mrr(rels).toFixed(4),
            ap: Num.apAtK(rels, k).toFixed(4),
            rec: Num.recallAtK(rels, k, rels.filter(r => r > 0).length).toFixed(3)
          });
        }
        Viz.buttons(host, [
          { label: 'ideal order', primary: true, on: () => { rels = rels.slice().sort((a, b) => b - a); draw(); S.redraw(); } },
          { label: 'worst order', on: () => { rels = rels.slice().sort((a, b) => a - b); draw(); S.redraw(); } },
          { label: 'shuffle', on: () => { rels = ML.shuffle(rels.slice()); draw(); S.redraw(); } }
        ]);
        const S = Viz.surface(host, {
          height: 190,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-.5, 7.5], yd: [0, 1.05], pad: { l: 46, r: 14, t: 16, b: 36 } })
              .frame({ xticks: [0, 1, 2, 3, 4, 5, 6, 7], xfmt: v => '#' + (v + 1), xlabel: 'rank position', ylabel: '1 / log₂(i+1)' });
            P.clip(() => {
              P.bars(rels.map((_, i) => 1 / Math.log2(i + 2)), {
                gap: .25,
                color: (v, i) => i < st.k ? T.c1 : T.line,
                labels: rels.map((_, i) => (1 / Math.log2(i + 2)).toFixed(2))
              });
            });
          }
        });
        draw();
        Viz.note(host, 'Move the most relevant document from rank 1 to rank 2 and NDCG drops noticeably; move it from rank 7 to rank 8 and it barely registers. That asymmetry <b>is</b> the discount curve above, and it is why a ranking loss must know about position — a pointwise classifier that gets every grade right but the order wrong scores well on accuracy and terribly here.');
      },

      posbias: function (host) {
        const st = Viz.controls(host, [
          { k: 'decay', label: 'examination decay with position', min: 0, max: 1.6, step: .05, value: .8, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'sessions logged', min: 200, max: 20000, step: 200, value: 4000, fmt: v => v.toLocaleString() },
          { k: 'ipw', label: 'apply inverse propensity weighting', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'corr', label: 'corr(observed CTR, true relevance)', cls: 'key' },
          { k: 'top', label: 'CTR at position 1' },
          { k: 'bot', label: 'CTR at position 10' },
          { k: 'bias', label: 'ratio explained by position' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(6);
            const K = 10;
            /* true relevance is deliberately NOT decreasing with position:
               the current ranker is imperfect, which is the whole point */
            const rel = [.30, .10, .45, .18, .08, .35, .12, .22, .06, .28];
            const exam = Array.from({ length: K }, (_, i) => Math.exp(-st.decay * i));
            const clicks = new Array(K).fill(0), shows = new Array(K).fill(0);
            for (let s = 0; s < st.n; s++) {
              for (let i = 0; i < K; i++) {
                shows[i]++;
                if (R() < exam[i] && R() < rel[i]) clicks[i]++;
              }
            }
            const ctr = clicks.map((c, i) => c / shows[i]);
            const est = st.ipw ? ctr.map((c, i) => Math.min(1, c / exam[i])) : ctr;
            const P = Viz.plot(ctx, w, h, { xd: [-.5, K - .5], yd: [0, .55], pad: { l: 50, r: 14, t: 16, b: 40 } })
              .frame({ xticks: Array.from({ length: K }, (_, i) => i), xfmt: v => '#' + (v + 1), xlabel: 'displayed position', ylabel: 'rate' });
            P.clip(() => {
              P.line(rel.map((v, i) => [i, v]), { color: T.c3, width: 2.6 });
              P.dots(rel.map((v, i) => [i, v]), { r: 3.4, color: T.c3 });
              P.line(est.map((v, i) => [i, v]), { color: T.c2, width: 2.4 });
              P.dots(est.map((v, i) => [i, v]), { r: 3.4, color: T.c2 });
              P.line(exam.map((v, i) => [i, v * .5]), { color: T.faint, width: 1.4, dash: [5, 4] });
            });
            out({
              corr: Num.corr(est, rel).toFixed(3),
              top: (ctr[0] * 100).toFixed(1) + '%',
              bot: (ctr[K - 1] * 100).toFixed(2) + '%',
              bias: (exam[0] / exam[K - 1]).toFixed(1) + '×'
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c3)', t: 'true relevance (unobservable)' },
          { c: 'var(--c2)', t: 'what your logs show' },
          { c: 'var(--faint)', t: 'examination probability (scaled)' }
        ]);
        Viz.note(host, 'With decay at 0.8, position 1 gets examined roughly 3,000× more often than position 10, and the observed CTR curve is essentially the examination curve wearing relevance as a decoration — the correlation with true relevance falls below 0.5. Switch on <b>inverse propensity weighting</b> and the estimate snaps back onto the true curve. The catch: you had to know the propensities, which is what randomisation or interleaving is for.');
      }
    },
    quiz: [
      {
        q: 'NDCG normalises DCG by…',
        options: ['the number of documents', 'the DCG of the ideal ordering', 'the query frequency', 'the maximum relevance grade'],
        answer: 1,
        why: 'DCG on its own is an unbounded number that grows with how many relevant documents a query happens to have, which makes it useless for comparing across queries — a query with ten relevant documents will always out-score one with two, regardless of ranking quality. Dividing by IDCG, the DCG of the best possible ordering of that <i>same</i> query’s documents, rescales every query onto the same 0–1 scale, exactly as the worked example in §2.24.1 computed for one specific four-document case.'
      },
      {
        q: 'LambdaMART’s gradient is RankNet’s pairwise gradient multiplied by…',
        options: ['the learning rate', '$|\\Delta\\mathrm{NDCG}|$ from swapping the pair', 'the document length', 'the inverse propensity'],
        answer: 1,
        why: 'LambdaMART specifies the gradient it wants rather than deriving it from a loss function — an unusual move worth remembering precisely because it is unusual. Pairs whose swap barely moves NDCG (two low-relevance documents near the bottom of the list, as in the first worked example) barely move the model; pairs whose swap moves NDCG a lot (a top-relevance document displaced from position 1) get a correspondingly large gradient. That weighting is what lets a pairwise-style update chase a listwise, position-aware metric.'
      },
      {
        q: 'Which metric should a first-stage retriever be tuned on?',
        options: ['NDCG@10', 'recall@k for the reranker’s candidate size', 'MRR', 'precision@1'],
        answer: 1,
        why: 'The retriever’s only job is to not lose the answer — it hands a candidate set to a reranker, which does the actual ordering, so the retriever’s own internal order barely matters. Its recall at the candidate-set size is a hard ceiling on everything downstream: a reranker cannot promote a document that was never retrieved in the first place, which is exactly the failure mode the last probe question below walks through.'
      },
      {
        q: 'Training a ranker on raw click logs primarily risks…',
        options: ['overfitting to rare queries', 'learning position bias and reproducing the previous ranker', 'label noise from bots', 'class imbalance'],
        answer: 1,
        why: 'Clicks confound relevance with examination probability — the worked example in §2.24.3 shows two equally relevant documents producing a 30% CTR at position 1 and a 1.2% CTR at position 5, purely from where they were shown, not from any difference in quality. Training naively on those numbers teaches the model that position 1 is what "relevant" looks like, which reproduces and entrenches whatever ranker generated the logs. IPW, randomisation or interleaving are the three standard ways to break that loop.'
      }
    ],
    cards: [
      { q: 'DCG formula', a: '$\\sum_i (2^{rel_i}-1)/\\log_2(i+1)$; NDCG divides by the ideal ordering’s DCG.' },
      { q: 'Three ranking loss families', a: 'Pointwise (grade), pairwise (inversions), listwise (metric-aware — LambdaMART).' },
      { q: 'Position bias correction', a: 'Inverse propensity weighting, randomisation/interleaving, or position-as-a-feature zeroed at inference.' },
      { q: 'Two-stage ranking', a: 'Cheap retriever over millions (recall@k), expensive reranker over the top 50–200 (NDCG/MRR). Retriever recall is a hard ceiling.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.25 */
  ML.section({
    id: 'experimentation', track: 'classical', num: '2.25', level: 2,
    title: 'Online experimentation: power, peeking, CUPED',
    lede: 'Offline metrics are a proxy. The experiment is the measurement — and it is a measurement with a variance, a duration and a startling number of ways to fool yourself. This is the section that decides whether your model launch is real.',
    prereq: ['intervals'],
    related: ['causal', 'sampling', 'mlops'],
    html: `
${H.tldr([
      'This section does not re-derive power, peeking or CUPED — §1.6 already built all three from scratch. It applies them to the specific, harder problem of shipping a <i>model</i>, and extends §1.6.7’s fixed-horizon-vs-bandit tradeoff into a full decision procedure.',
      'A model launch usually has a much stronger CUPED covariate available than a generic product change does: the model’s own predicted score. Using it — sometimes called CUPAC — routinely beats the variance reduction a raw historical metric gets you.',
      'Fixed-horizon test, sequential design, bandit, or interleaving: which one is right depends on whether you need a defensible number for a go/no-go call, or you are optimising traffic while you learn. Confusing the two is the most common experimentation mistake specific to ML teams.'
    ])}

<p>You have trained a new model — a classifier, a ranker, a recommender — and it beats the old one on every offline metric you can compute. The only measurement that actually matters now is an online one: does it move the number the business cares about, on real traffic, by an amount worth the switching cost? §1.6 built the entire statistical toolkit this question needs: the confidence-versus-credible distinction, the sample-size formula, why peeking inflates your error rate and how to stop it, the CUPED derivation, and even a full bandit framework with regret curves. None of that is repeated here. What follows is what changes when the thing under test is a model rather than a button colour — and it changes more than you might expect, because a model gives you levers a generic A/B test does not.</p>

<h2><span class="sn">2.25.1</span> Sizing a model launch</h2>
<p>The sample-size formula from §1.6.4, $n \\approx 2p(1-p)\\big(\\frac{z_{\\alpha/2}+z_\\beta}{\\delta}\\big)^2$, and the quadratic penalty for shrinking $\\delta$ it implies, apply unchanged to a model launch. What is different is the <i>size</i> of $\\delta$ you are usually chasing. A new ranking model that improves offline NDCG@10 by two points, a genuinely large offline win, might move downstream revenue by a few tenths of a percentage point — offline metrics and business metrics live on different scales, and the online $\\delta$ is almost always the smaller, harder-to-detect one. This is precisely why the next subsection matters more for a model launch than for an average product experiment: any variance reduction you can find is worth proportionally more when your true effect is already small.</p>

${H.table(['Quantity', 'Symbol', 'Meaning', 'Usual'], [
      ['Significance', '$\\alpha$', 'false-positive rate you accept', '0.05'],
      ['Power', '$1-\\beta$', 'chance of detecting a real effect of the assumed size', '0.80'],
      ['Baseline rate', '$p$', 'the current conversion rate', 'measured'],
      ['<b>MDE</b>', '$\\delta$', '<b>minimum detectable effect</b> — the smallest lift worth finding', 'a business decision'],
      ['Duration', '—', 'must cover at least one full weekly cycle', '1–4 weeks']
    ])}
${H.key('The MDE is not a statistical quantity — it is the smallest improvement that would change what you do. For a model launch, ask the offline-to-online conversion question first: given how offline gains have translated to online lift on past launches at your company, what online $\\delta$ does this model’s offline improvement plausibly imply? Size the test for that, not for the offline number.')}

<p><b>What you are looking at.</b> The same power-versus-sample-size trade §1.6 derived, now with your role as reader shifted from "reader" to "owner of a launch decision": the curve is the minimum detectable effect as a function of users per arm at three power levels, the dashed lines mark your chosen effect and the sample size it implies, and the dot is where they meet.</p>
<p><b>What to do with it.</b> Plug in your actual baseline conversion rate and the online lift you would consider this model launch a genuine win. Read off the required traffic and days to run — and if the number of days exceeds a business cycle you can tolerate, that is the signal to look for a bigger effect (a bigger rollout, more sensitive metric, or CUPED) rather than to peek your way to an early answer.</p>
<p><b>The thing genuinely worth noticing.</b> Try halving the relative lift you are trying to detect and watch the required sample size jump to exactly four times its previous value — the $1/\\delta^2$ relationship from §1.6.4, now costing you specifically because model improvements tend to arrive as small online deltas. This is the arithmetic reason "just detect a smaller lift" is rarely a free request.</p>

${H.lab('power', 'Sample size, MDE and duration — the trade, live', 'Move any two and the third follows. The curve is the power function; the shaded region is where a real effect of your assumed size would go undetected.')}

<h2><span class="sn">2.25.2</span> Peeking on a live model dashboard</h2>
<p>§1.6.4 derived exactly why stopping the moment a running p-value crosses 0.05 inflates your true false-positive rate to 20–35%, and named the fixes: fix the horizon, spend $\\alpha$ across pre-planned looks (group sequential), or use a bound that is valid at every moment simultaneously (always-valid / anytime-valid methods). Model launches are, if anything, more exposed to this mistake than an average product change, because ML teams build real-time model-quality dashboards as a matter of course — latency, error rate, prediction distribution, all updating by the minute — and the temptation to read a business metric off the same dashboard the moment it looks good is strong precisely because the infrastructure for watching continuously is already there.</p>

<p>There is a legitimate reason to watch continuously, though, and it is worth separating from the illegitimate one. A <b>canary release</b> — rolling a new model out to 1%, then 10%, then 50% of traffic — is not trying to answer "is this model better?" with a clean p-value; it is trying to answer "is this model catastrophically worse?" as fast as possible, so that a bad model gets caught at 1% of traffic instead of 100%. That is exactly the use case always-valid confidence sequences were built for: they let you monitor guardrail metrics continuously and stop the instant one breaches, with a genuine, pre-computed error-rate guarantee, rather than the informal "eyeball the dashboard and pull the cord" version that produces the 20–35% inflation. The distinction is not watch-versus-don't-watch; it is <i>using a method whose error rate under continuous monitoring is known</i> versus one whose error rate you are quietly assuming is 5% and is not.</p>

${H.table(['Fix', 'How it works', 'Cost'], [
      ['Fix the horizon', 'compute $n$, look once, at the end', 'no early stopping, ever — including for a disaster'],
      ['Bonferroni over looks', 'divide $\\alpha$ by the number of planned looks', 'crude and very conservative'],
      ['<b>Group sequential</b> (O’Brien–Fleming, Pocock)', 'spend $\\alpha$ across pre-planned interim analyses', 'the clinical-trials standard; needs the looks planned in advance'],
      ['<b>Always-valid / anytime-valid</b> (mSPRT, e-values, confidence sequences)', 'a bound valid at <i>every</i> moment simultaneously', '~10–25% more samples for an honest continuous dashboard — what canary releases and continuous-monitoring platforms actually need'],
      ['Bayesian with a decision rule', 'stop when $P(\\text{lift}>0)$ crosses a threshold', 'well-defined, but a threshold is not a guaranteed error rate']
    ])}
${H.pitfall('"We ran it Bayesian, so peeking is fine" is half right, as §1.6.4 already established, and it is worth restating for a model launch specifically because model teams reach for Bayesian dashboards often, expecting the framing itself to solve the problem. The posterior is valid at every moment — it is just conditioning on data. But a <i>stopping rule</i> layered on top of the posterior — "ship when $P(\\text{lift}>0)>0.95$" — still has a frequentist error rate, and continuous monitoring inflates that too, just usually by less than naive p-values. If you need a bounded false-positive rate on a canary, use an anytime-valid method deliberately, not a Bayesian dashboard by default.')}

<p><b>What you are looking at.</b> The same peeking simulation §1.6.4 built: thousands of simulated A/A tests, where both arms are provably identical, run under a fixed-horizon rule and a stop-the-moment-it's-significant rule, with the resulting false-positive rate of each plotted against the number of looks taken.</p>
<p><b>What to do with it.</b> Set the number of looks to match how often your team's dashboard actually gets checked during a real launch window — daily for two weeks is 14, hourly for two days is roughly 48 — and read off the true false-positive rate at that setting.</p>
<p><b>The thing genuinely worth noticing.</b> The curve you get here is identical in shape to §1.6.4's, because the mathematics has not changed — only the setting has. The point of repeating it in a model-launch section is that ML teams have the dashboards to make this mistake more easily than most, not that the mistake is somehow different for a model. If your canary infrastructure watches metrics continuously, it should be doing so with the always-valid row of the table above, not with a p-value computed as if it were only ever going to be read once.</p>

${H.lab('peek', 'Simulating the peeking problem, under a true null', 'Both variants are identical — there is no effect at all. The simulation runs thousands of A/A tests and counts how often each stopping rule declares a winner. The correct answer is 5%.')}

<h2><span class="sn">2.25.3</span> CUPED, and the covariate a model gives you for free</h2>
<p>§1.6.6 derived CUPED in full: subtract $\\theta(X-\\bar X)$ from the outcome $Y$, where $X$ is any pre-treatment covariate and $\\theta$ is chosen to minimise variance, and the result is unbiased with variance reduced by exactly $1-\\rho^2$, the squared correlation between the covariate and the outcome. The derivation does not care what $X$ is, only that it is measured strictly before randomisation. A generic A/B test typically uses last month's value of the same metric as $X$, which gives a useful but modest $\\rho$.</p>

<p>A model launch has a better covariate sitting right there in the training pipeline: <b>the model's own predicted score for each user, computed before the experiment starts</b>. If you are testing a model that predicts conversion probability, its pre-experiment prediction for each user is, by construction, whatever signal the model has managed to extract about that user's likely behaviour — which is usually more correlated with the actual outcome than a single historical data point is. Using a model's own prediction as the CUPED covariate is sometimes called <b>CUPAC</b> (control using predictions as covariates), and it is the natural extension of §1.6.6's result to a setting where you happen to have a predictive model lying around, which, in this section, you always do.</p>

${H.table(['Covariate $X$', 'Typical $\\rho$ with outcome', 'Variance reduction $1-\\rho^2$'], [
      ['Last month’s value of the same metric', '0.6', '36%'],
      ['A simple pre-experiment average (e.g. trailing 4-week mean)', '0.7', '51%'],
      ['<b>A well-calibrated model’s pre-experiment prediction (CUPAC)</b>', '0.85', '<b>72%</b>']
    ])}
${H.note('The numbers in that table are illustrative, not universal — the actual $\\rho$ depends entirely on how predictive your model genuinely is, and a poorly calibrated model is a worse covariate than a simple historical average. But the direction is structural: a model trained specifically to predict the outcome you are now measuring will, when it is any good at all, correlate with that outcome at least as well as an unconditional historical average does, because the model had access to the historical average as one of its inputs and more besides.')}

${H.pitfall('The one rule from §1.6.6 that CUPAC makes easier to violate by accident: the covariate must be measured strictly before randomisation. If your "pre-experiment prediction" pipeline actually scores users using features computed <i>during</i> the experiment window — a common accident when a feature store updates continuously — you have smuggled a post-treatment variable into the covariate and reintroduced exactly the bias CUPED exists to avoid. Audit the feature timestamps, not just the code path, before trusting a CUPAC-adjusted result.')}

${H.history(`<p>CUPED itself is a named, dated invention, which is unusual for a technique this fundamental: Alex Deng, Ya Xu, Ron Kohavi and Toby Walker published "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" in 2013, describing work done inside Microsoft's experimentation platform. Before that paper, teams reduced variance ad hoc — stratifying by a known heavy-user flag, say — without a general, provably-unbiased recipe for doing it with <i>any</i> pre-period covariate. The 2013 paper is also where the $1-\\rho^2$ result derived in §1.6.6 first appeared in this form, aimed squarely at an industry audience running thousands of experiments a year rather than at a statistics journal.</p>
<p>CUPAC — swapping a raw historical metric for a trained model's prediction as the covariate — is the natural next step once a team already has predictive models lying around for other purposes, and variants of it are now standard at Netflix, Booking.com and most large experimentation platforms, usually under a house-specific name. The underlying mathematics has not moved past the 2013 result at all; what moved is which covariate teams had sitting in their feature store to plug into it.</p>`)}

<p><b>What you are looking at.</b> The same simulated experiment §1.6.6 built: raw and CUPED-adjusted estimates of a treatment effect, with 95% confidence intervals, both centred on the same true effect.</p>
<p><b>What to do with it.</b> Set the pre/post correlation to 0.6 to see what a generic historical covariate buys you, then push it to 0.85 to see what a genuinely good model prediction buys you instead, and compare the two interval widths and the "equivalent extra users" readout directly.</p>
<p><b>The thing genuinely worth noticing.</b> Both bars sit on the truth regardless of $\\rho$ — CUPED's unbiasedness does not depend on how good the covariate is, only on it being pre-treatment. Only the <i>interval</i> narrows as $\\rho$ rises. That separation between "is the estimate correct" and "is the estimate precise" is worth being able to say out loud: a bad covariate never introduces bias, it just fails to buy you the free precision a good one would have.</p>

${H.lab('cuped', 'CUPED on simulated user metrics', 'Real numbers: the same experiment analysed with and without the adjustment. The variance reduction is exactly $1-\\rho^2$, and the confidence interval shrinks by $\\sqrt{1-\\rho^2}$.')}

<h2><span class="sn">2.25.4</span> Fixed test, sequential design, or bandit — choosing for a model launch</h2>
<p>§1.6.7 already built the fixed-horizon-versus-bandit tradeoff in general: an A/B test pays the full cost of running half your traffic through the losing arm for the entire duration, in exchange for a clean, defensible, unbiased estimate at the end; a bandit shifts traffic toward whatever currently looks best while it is still learning, saving reward at the cost of a messier estimate. For a model launch, that tradeoff sits inside a slightly larger decision, because a model launch is rarely just "old model versus new model" — it is often "which of several candidate models, or which of several hyperparameter settings from a Bayesian optimisation run (§2.21), should serve traffic."</p>

${H.table(['Situation', 'Right tool', 'Why'], [
      ['One new model, a go/no-go launch decision that must be defensible afterward', 'Fixed-horizon A/B test, sized per §2.25.1', 'You need the frequentist guarantee: a specific, pre-committed error rate that a stakeholder or auditor can check, not a policy that was still adapting while it collected data.'],
      ['Several candidate models or configs, no single launch decision, ongoing traffic', 'Multi-armed bandit (§1.6.7)', 'Minimising cumulative regret while you learn matters more than a clean point estimate for any one arm — you are not writing a launch report, you are running a live system.'],
      ['A canary rollout that must be stoppable early on a guardrail breach', 'Always-valid / group-sequential monitoring (§2.25.2)', 'You want the option to stop early <i>with a known error rate</i>, which a plain fixed-horizon test does not offer and a bandit does not either — a bandit optimises reward, not a bounded false-alarm rate.'],
      ['Two ranking models being compared, ordering is the entire question', 'Interleaving (§2.24.3)', 'Needs roughly an order of magnitude fewer sessions than an A/B test for the same power, because it removes between-user variance entirely — but only answers ordering questions, not absolute lift.']
    ])}
${H.intuition(`<p>Notice the family resemblance running through this whole course: a Bayesian optimisation loop (§2.21) chooses its next hyperparameter by balancing a good mean against high uncertainty; a multi-armed bandit (§1.6.7) chooses its next traffic allocation the same way; UCB's "optimism in the face of uncertainty" and expected improvement's exploit-plus-explore split are, structurally, the same idea wearing two different names. The reason a fixed-horizon A/B test looks so different from both is that it deliberately refuses to exploit early information at all, on purpose, because the thing it is optimising for is not reward — it is the width and honesty of a confidence interval you can defend after the fact. Picking the right tool from the table above is really picking which of those two goals — accumulated reward, or a defensible interval — actually describes your situation.</p>`)}

${H.analogy(`<p>A fixed-horizon A/B test is a criminal trial: the jury withholds judgment, deliberately, until every scheduled piece of evidence has been heard, precisely so that the eventual verdict can be defended against an appeal — "we still had a witness left to hear from" is exactly the objection a peeked, early-stopped experiment cannot answer. A bandit is a doctor doing triage in an emergency room: there is no jury to convince afterward, only patients in front of you right now, so the doctor commits to the treatment that currently looks best for each one while continuing to learn, and accepts that the record of who got what will look messy in hindsight. Both are rational. The trial and the triage are simply optimising for different things — a defensible verdict versus outcomes for the people currently in the room — and the mistake this section keeps warning about is running a trial with a triage doctor's instincts: peeking at the evidence early and declaring a verdict anyway.</p>`)}

<h2><span class="sn">2.25.5</span> The failure modes to name</h2>
${H.table(['Problem', 'Symptom', 'Fix'], [
      ['<b>Sample ratio mismatch</b>', 'arms are 50.4/49.6 instead of 50/50', '<b>stop and debug</b> — a chi-square p < 0.001 on the split means the assignment or logging is broken and every other number is suspect'],
      ['<b>Offline/online mismatch</b>', 'the model wins offline on every metric, and the online result is flat or negative', 'trust the online result — the offline metric is a proxy (§2.24’s recall-vs-NDCG distinction is one common cause), and a launch decision is made on the thing you actually optimised the business for'],
      ['Novelty / primacy effect', 'a big lift in week 1 that decays', 'run longer; analyse new vs returning users separately'],
      ['Network interference', 'treatment leaks to control (social, marketplaces)', 'cluster randomisation, switchback, or budget-split designs'],
      ['Multiple metrics', 'one of 20 guardrails is "significant"', 'pre-register one primary metric; Benjamini–Hochberg on the rest'],
      ['Simpson’s paradox', 'lift overall, loss in every segment', 'check the segment mix; a shifted traffic composition is usually the cause (§1.7)'],
      ['Underpowered launch', '"no significant difference, so ship it"', 'absence of evidence is not evidence of absence — report the interval, not the p-value']
    ])}
${H.flag('Interleaving deserves more attention than it gets for ranking changes: showing one user a merged list from both rankers and measuring which side’s results they click needs roughly an order of magnitude fewer sessions than an A/B test for the same statistical power, because it removes between-user variance entirely. It only works for ordering comparisons, which is exactly what §2.24 is about.')}

${H.probe([
      ['You look at the dashboard daily for two weeks and stop when p < 0.05. What is your real false-positive rate?', 'Around 20–35% depending on the number of looks, not 5% — the exact mechanism §1.6.4 derives. Use a group-sequential boundary or an always-valid confidence sequence, and if your team ships model launches often enough to build real-time dashboards, build the monitoring on one of those two, not on a p-value computed as if it would only be read once.'],
      ['Explain CUPAC to a product manager in one sentence.', 'We subtract off the part of each user’s outcome the model could already have predicted before the experiment started, which removes noise without touching the true effect of the launch — so the test finishes sooner, and it works especially well here because the model itself is an unusually good predictor of what we are about to measure.'],
      ['Your A/B split is 50.4% / 49.6% with 2 million users. Do you care?', 'Yes, enormously. At that scale the imbalance is wildly significant, which means assignment or logging is broken. Every metric in the readout is untrustworthy until it is explained — this is the same sample-ratio-mismatch check §1.6 assumes you already run, restated here because a model-serving pipeline adds its own place for the split to silently break, in the routing layer rather than the experiment framework.'],
      ['Halving the MDE requires how much more traffic?', 'Four times as much — $n \\propto 1/\\delta^2$, from §1.6.4 — which is why §2.25.1 spends a paragraph on the fact that model launches usually imply a small online $\\delta$: the quadratic penalty is not an abstract warning here, it is the reason your launch experiment needs more traffic than intuition suggests.'],
      ['You have three candidate models and want to find the best one while also serving good predictions the whole time. Fixed-horizon test or bandit?', 'Bandit, from the decision table in §2.25.4 — you are not trying to produce a single defensible launch number, you are trying to minimise the reward given up while you learn which model is best, which is precisely the regret-minimisation framing §1.6.7 built. Switch to a fixed-horizon test only when you need the result to survive being audited afterward.']
    ])}`,
    labs: {
      power: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'baseline conversion rate', min: .01, max: .5, step: .005, value: .12, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'lift', label: 'relative lift to detect', min: .01, max: .3, step: .005, value: .05, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'power', label: 'power', min: .5, max: .95, step: .05, value: .8, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'daily', label: 'daily traffic per arm', min: 200, max: 50000, step: 200, value: 5000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'n', label: 'users needed per arm', cls: 'key' },
          { k: 'days', label: 'days to run', cls: 'warn' },
          { k: 'abs', label: 'absolute effect' },
          { k: 'mde', label: 'MDE at 1 week' },
          { k: 'half', label: 'to halve the MDE' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const p = st.p, delta = p * st.lift;
            const n = Num.sampleSize(p, delta, .05, st.power);
            const days = Math.ceil(n / st.daily);
            const week = Num.mde(p, st.daily * 7, .05, st.power);
            const P = Viz.plot(ctx, w, h, {
              xd: [2, 6], yd: [0, .12],
              pad: { l: 54, r: 14, t: 16, b: 40 }
            }).frame({
              xticks: [2, 3, 4, 5, 6], xfmt: v => Math.pow(10, v) >= 1e6 ? (Math.pow(10, v) / 1e6) + 'M' : Math.pow(10, v).toLocaleString(),
              xlabel: 'users per arm', ylabel: 'detectable absolute lift', yfmt: v => (v * 100).toFixed(1) + '%'
            });
            P.clip(() => {
              [.8, .9, .95].forEach((pw, i) => {
                P.fn(lx => Num.mde(p, Math.pow(10, lx), .05, pw), { color: [T.c1, T.c4, T.c2][i], width: pw === st.power ? 2.8 : 1.3, alpha: pw === st.power ? 1 : .5, n: 140 });
              });
              P.hline(delta, { color: T.c3, dash: [4, 4], label: 'your effect' });
              P.vline(Math.log10(n), { color: T.c3, dash: [4, 4] });
              P.dots([[Math.log10(n), delta]], { r: 5.5, color: T.c3, stroke: true, strokeWidth: 2 });
            });
            out({
              n: n.toLocaleString(),
              days: days + (days > 28 ? ' ⚠ too long' : ''),
              abs: (delta * 100).toFixed(2) + ' pts',
              mde: (week * 100).toFixed(2) + ' pts',
              half: (4 * n).toLocaleString() + ' per arm'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: '80% power' }, { c: 'var(--c4)', t: '90%' }, { c: 'var(--c2)', t: '95%' }]);
        Viz.note(host, 'Try to detect a 1% relative lift on a 12% baseline: the answer is roughly two million users per arm. That is the honest reason most teams cannot measure small improvements — and the reason CUPED, which cuts the requirement by $1-\\rho^2$, is worth more than almost any modelling change.');
      },

      peek: function (host) {
        const st = Viz.controls(host, [
          { k: 'looks', label: 'times you check the dashboard', min: 1, max: 30, step: 1, value: 14, fmt: v => v },
          { k: 'alpha', label: 'nominal α', min: .01, max: .1, step: .01, value: .05, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nominal', label: 'α you believe', cls: 'key' },
          { k: 'actual', label: 'α you actually have', cls: 'bad' },
          { k: 'infl', label: 'inflation' },
          { k: 'fix', label: 'Bonferroni-corrected α' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const pts = [];
            for (let k = 1; k <= 30; k++) pts.push([k, Num.peekingFPR(k, st.alpha, 91)]);
            const actual = pts[st.looks - 1][1];
            const P = Viz.plot(ctx, w, h, { xd: [1, 30], yd: [0, Math.max(.4, actual * 1.2)], pad: { l: 52, r: 14, t: 16, b: 40 } })
              .frame({ xlabel: 'number of looks at the data', ylabel: 'false-positive rate', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.area(pts, { color: T.c2, alpha: .14 });
              P.line(pts, { color: T.c2, width: 2.8 });
              P.hline(st.alpha, { color: T.c3, dash: [5, 4], label: 'what you think it is' });
              P.dots([[st.looks, actual]], { r: 5.5, color: T.c2, stroke: true, strokeWidth: 2 });
            });
            out({
              nominal: (st.alpha * 100).toFixed(0) + '%',
              actual: (actual * 100).toFixed(1) + '%',
              infl: (actual / st.alpha).toFixed(1) + '×',
              fix: (st.alpha / st.looks).toFixed(4)
            });
          }
        });
        Viz.note(host, 'Every point on this curve is two thousand simulated A/A tests — <b>there is no effect anywhere in this simulation</b>. Check daily for two weeks and roughly one in four of your "wins" is noise. The curve flattens rather than reaching 100% because the test statistic is a random walk with drift zero: it becomes progressively harder to cross a boundary it has already had many chances at.');
      },

      cuped: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'pre/post correlation ρ', min: 0, max: .95, step: .05, value: .7, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'users per arm', min: 200, max: 20000, step: 200, value: 3000, fmt: v => v.toLocaleString() },
          { k: 'effect', label: 'true treatment effect', min: 0, max: .3, step: .01, value: .08, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'raw', label: 'raw estimate ± CI' },
          { k: 'adj', label: 'CUPED estimate ± CI', cls: 'good' },
          { k: 'vr', label: 'variance reduction', cls: 'key' },
          { k: 'theory', label: 'theory: 1 − ρ²' },
          { k: 'equiv', label: 'equivalent extra users' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(15);
            const n = st.n, rho = st.rho;
            const pre = [], post = [], arm = [];
            for (let i = 0; i < 2 * n; i++) {
              const x = R.normal(0, 1);
              const a = i < n ? 0 : 1;
              const y = rho * x + Math.sqrt(Math.max(0, 1 - rho * rho)) * R.normal(0, 1) + a * st.effect;
              pre.push(x); post.push(y); arm.push(a);
            }
            const c = Num.cuped(post, pre);
            const meanBy = (v, a) => Num.mean(v.filter((_, i) => arm[i] === a));
            const sdBy = (v, a) => Num.sd(v.filter((_, i) => arm[i] === a));
            const dRaw = meanBy(post, 1) - meanBy(post, 0);
            const dAdj = meanBy(c.adjusted, 1) - meanBy(c.adjusted, 0);
            const seRaw = Math.sqrt(sdBy(post, 0) ** 2 / n + sdBy(post, 1) ** 2 / n);
            const seAdj = Math.sqrt(sdBy(c.adjusted, 0) ** 2 / n + sdBy(c.adjusted, 1) ** 2 / n);

            const P = Viz.plot(ctx, w, h, { xd: [-.6, 1.6], yd: [-.12, .32], pad: { l: 54, r: 14, t: 18, b: 40 } })
              .frame({ xticks: [0, 1], xfmt: v => v === 0 ? 'raw' : 'CUPED', xlabel: 'analysis', ylabel: 'estimated lift' });
            P.clip(() => {
              P.hline(st.effect, { color: T.c3, dash: [5, 4], label: 'truth' });
              P.hline(0, { color: T.faint, dash: false, width: 1 });
              P.errbars([[0, dRaw - 1.96 * seRaw, dRaw + 1.96 * seRaw]], { color: T.c2, width: 2.4, cap: 12 });
              P.dots([[0, dRaw]], { r: 6, color: T.c2, stroke: true, strokeWidth: 2 });
              P.errbars([[1, dAdj - 1.96 * seAdj, dAdj + 1.96 * seAdj]], { color: T.c1, width: 2.4, cap: 12 });
              P.dots([[1, dAdj]], { r: 6, color: T.c1, stroke: true, strokeWidth: 2 });
            });
            const vr = 1 - (c.varAfter / c.varBefore);
            out({
              raw: dRaw.toFixed(4) + ' ± ' + (1.96 * seRaw).toFixed(4),
              adj: dAdj.toFixed(4) + ' ± ' + (1.96 * seAdj).toFixed(4),
              vr: (vr * 100).toFixed(1) + '%',
              theory: ((rho * rho) * 100).toFixed(1) + '%',
              equiv: vr < .99 ? Math.round(n / (1 - vr) - n).toLocaleString() : '—'
            });
          }
        });
        Viz.note(host, 'Both estimates are centred on the truth — CUPED is unbiased, because the covariate was measured before randomisation. The <b>interval is what changes</b>: at ρ = 0.7 the variance falls by 49%, which is worth as much as doubling your traffic and costs one line of SQL. This is the same idea as a control variate in §1.13, applied to the metric you already log.');
      }
    },
    quiz: [
      {
        q: 'Halving the minimum detectable effect requires roughly…',
        options: ['2× the sample', '4× the sample', '8× the sample', 'the same sample, run longer'],
        answer: 1,
        why: '$n\\propto 1/\\delta^2$, derived in full in §1.6.4. This quadratic is why small lifts are so expensive to measure, and it bites especially hard on model launches, where the online $\\delta$ implied by an offline win is often small to begin with (§2.25.1) — "just detect a smaller effect" quietly means quadrupling the traffic, not a minor adjustment.'
      },
      {
        q: 'Checking a fixed-horizon test daily for two weeks and stopping at p < 0.05 gives a true false-positive rate of about…',
        options: ['5%', '10%', '25%', '50%'],
        answer: 2,
        why: 'Roughly 20–35% depending on the number of looks, as §1.6.4 derives from the random-walk behaviour of the running test statistic. Group-sequential boundaries or always-valid inference fix it, and §2.25.2’s point is that ML teams have real-time dashboards ready-built for exactly this mistake — a canary rollout that wants to monitor continuously should be built on an always-valid method deliberately, not stumble into naive peeking because the infrastructure to check constantly already exists.'
      },
      {
        q: 'CUPED reduces variance by a factor of…',
        options: ['ρ', '1 − ρ²', '1/ρ', '√ρ'],
        answer: 1,
        why: 'Exactly the residual variance after regressing on the pre-period covariate, derived in §1.6.6 — the $R^2$ you remove is the variance you save. §2.25.3 extends this by pointing out that a model launch usually has an unusually strong covariate on hand: the model’s own pre-experiment prediction, which tends to correlate with the outcome more tightly than a plain historical average and therefore buys a larger reduction, precisely because a higher $\\rho$ pushes $1-\\rho^2$ closer to zero.'
      },
      {
        q: 'Your 50/50 experiment came out 50.4/49.6 across 2 million users. You should…',
        options: ['proceed, it is close enough', 'stop and debug assignment or logging before trusting any metric', 'reweight the arms', 'extend the experiment'],
        answer: 1,
        why: 'A sample ratio mismatch that significant means the randomisation or the logging is broken, which invalidates everything else in the readout — reweighting or extending the experiment both assume the data collected so far is trustworthy, which is exactly what a significant SRM tells you it is not. For a model-serving system specifically, the routing layer that decides which model scores a given request is a second place this can silently break, in addition to the experiment framework itself.'
      },
      {
        q: 'You have three candidate models to compare and want to minimise reward given up while you learn which is best, with no single go/no-go decision required afterward. The right tool is…',
        options: ['a fixed-horizon A/B test sized per §2.25.1', 'a multi-armed bandit (§1.6.7)', 'interleaving', 'a Bonferroni correction across the three'],
        answer: 1,
        why: 'The decision table in §2.25.4 turns on exactly this distinction: a fixed-horizon test buys a clean, auditable estimate at the cost of deliberately not exploiting what it learns along the way, which is the right trade when you need a defensible launch number. Here nothing needs to be defended after the fact — the goal is accumulated reward while learning, which is the regret-minimisation framing §1.6.7 built in full, with UCB and Thompson sampling as the concrete algorithms.'
      }
    ],
    cards: [
      { q: 'Sample size per arm', a: '$n \\approx 2p(1-p)\\left(\\frac{z_{\\alpha/2}+z_\\beta}{\\delta}\\right)^2$ (§1.6.4); halving δ costs 4× the traffic — and model launches usually imply a small δ.' },
      { q: 'Peeking', a: 'Repeated looks inflate α from 5% to 20–35% (§1.6.4). A canary that must watch continuously needs an always-valid method, not naive peeking.' },
      { q: 'CUPED and CUPAC', a: '§1.6.6’s $\\hat Y = Y-\\theta(X-\\bar X)$, unbiased whatever $X$ is. For a model launch, use the model’s own pre-experiment prediction as $X$ — usually a stronger covariate than a historical average.' },
      { q: 'Fixed test vs bandit vs interleaving', a: 'Need a defensible launch number → fixed-horizon test. Learning while serving, no single decision → bandit (§1.6.7). Two rankers, ordering only → interleaving (§2.24.3).' },
      { q: 'Sample ratio mismatch', a: 'A significant deviation from the intended split means the pipeline — possibly the model-routing layer itself — is broken. Debug before reading any metric.' }
    ]
  });
})();
