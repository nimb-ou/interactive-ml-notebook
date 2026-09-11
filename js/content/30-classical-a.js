/* ============================================================
   PART 2 — Core & classical ML (2.1 – 2.6)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.1 */
  ML.section({
    id: 'supervised-setup', track: 'classical', num: '2.1',
    title: 'The supervised setup; loss vs metric; split discipline',
    lede: 'The vocabulary the rest of Part 2 assumes, and the one habit that separates a real evaluation from a comforting one.',
    rests: 'Rests on §1.4 for why a holdout means anything at all.',
    html: `
<p>Say you have just finished a model that flags fraudulent card transactions. You score it against the transactions you trained it on and it is right 94 per cent of the time. Someone asks the only question that matters: <i>is it good?</i></p>

<p>You cannot actually answer that, not yet, and it is worth sitting with why. "Good" has to mean something like <i>right most of the time on transactions this model has never seen</i> — next week's transactions, next month's, the ones from a fraud ring that has not been invented yet. You do not have those. You have the rows on your laptop. The entire vocabulary this section builds exists to close the gap between the number you can compute and the question you are actually being asked.</p>

<h2><span class="sn">2.1.1</span> What you actually want, and what you can actually have</h2>

<p>Formally, a model is a function $f: \\mathcal{X} \\to \\mathcal{Y}$, read "f, a function from the input space to the output space" — it eats a row of features from $\\mathcal{X}$ and returns a prediction in $\\mathcal{Y}$. You have some way of scoring how wrong one prediction is, a <b>loss function</b> $\\ell(f(x), y)$, which compares the prediction $f(x)$ against the true label $y$ and returns a number: zero for a perfect prediction, larger for a worse one.</p>

<p>What you actually want is a model that scores well <i>on average, over every transaction that will ever occur</i> — not just the ones already sitting in your dataset. Write that population as a distribution $P$ over $(x, y)$ pairs, and the quantity you genuinely care about is the <b>expected loss</b>, or <b>risk</b>:</p>

$$R(f) = \\mathbb{E}_{(x,y)\\sim P}\\big[\\ell(f(x), y)\\big]$$

<p>Read this as: draw a transaction at random from the true, infinite population of transactions that could occur, score how wrong your model is on it, and average that over every transaction that population could ever produce. That is the honest target. It is also a number you can never compute, because you do not have $P$ — nobody has ever observed the full population of transactions that will occur between now and the end of time, and nobody ever will.</p>

<p>What you have instead is a finite sample of $n$ transactions drawn from that population. So you do the only thing available to you: you replace the expectation over the whole population with an average over the sample you actually hold, called the <b>empirical risk</b>,</p>

$$\\hat R(f) = \\frac{1}{n}\\sum_{i=1}^{n} \\ell(f(x_i), y_i),$$

<p>and you minimise <i>that</i> instead, a procedure called <b>empirical risk minimisation</b>. The whole of supervised learning is this substitution: swap a population average you cannot take for a sample average you can, and hope the two are close.</p>

<p>"Hope" is doing a lot of work in that sentence, and it should make you uneasy — a model that is excellent on its own training rows and useless on anything else has satisfied $\\hat R(f) \\approx 0$ while $R(f)$ stays large. The reason the substitution is not simply wishful thinking is the subject of §1.4: averages of independent draws concentrate tightly around the true expectation they estimate, at a rate you can write down, and that fact is the actual licence for treating a held-out sample average as a stand-in for the truth. This section takes that licence as given and builds the discipline around actually cashing it in — because concentration only protects an average computed on data nothing has touched yet, and the rest of §2.1 is entirely about how easy that protection is to spend by accident.</p>

<h2><span class="sn">2.1.2</span> Loss versus metric</h2>

<p>Once you have a loss, you might expect training to be the whole job: pick $\\ell$, minimise $\\hat R(f)$ using the machinery from §0.3 and §0.7, and read off how well you did. It is not, because the number you optimise and the number you report to the people paying for the model are very often not the same number, and confusing the two is one of the most common ways a technically competent model becomes a business disappointment.</p>

<p>Here is why they come apart. Suppose you tried to train directly on the metric everyone actually cares about — say, plain classification accuracy, the fraction of transactions correctly labelled. Accuracy is built from a hard decision: predict fraud if the model's score crosses some threshold, then count right versus wrong. Nudge a weight by a tiny amount and, for almost every transaction, the prediction does not cross the threshold and does not change at all — so accuracy, as a function of the weights, is flat almost everywhere and jumps in discrete steps at a few isolated points. Its gradient is zero nearly everywhere it is defined at all. Gradient descent, the entire method built in §0.3, has nothing to push against. You cannot climb a staircase by asking which way is downhill at a point on a flat step.</p>

<p>Logistic loss (§2.4) is the smooth stand-in: instead of a hard right/wrong, it scores <i>how confidently</i> wrong or right the model was, using a continuous, differentiable curve everywhere. Nudging a weight now always changes the loss by a computable, non-zero amount, so the gradient always has somewhere to point. That is the property a loss function is required to have, and a metric is not: the loss exists to be optimised, so it must be differentiable and it must decompose additively over examples so a gradient computed on one batch means something about the whole dataset. The metric exists to be read by a human deciding whether to ship, so it can be anything legible — AUC, KS, Gini, precision at a fixed approval rate, cost per approved loan — with no obligation to behave nicely under differentiation at all.</p>

${H.key('The loss is what you optimise; the metric is what you report. They are allowed to be different numbers measuring different things.')}

${H.analogy(`<p>Think of training a sprinter for a race. During practice you cannot measure race time every seven seconds — the race has not happened yet — so a coach trains against a proxy that <i>can</i> be measured continuously: heart-rate zones, stride cadence, split times on a track. Improving the proxy every session is the loss. Race day, a single number, is the metric, and it is the only one anyone outside the training programme actually cares about.</p>
<p>The two usually move together — better cadence tends to mean a faster race — but they can come apart. A sprinter who has optimised cadence in isolation might have terrible acceleration off the blocks, which the proxy never measured. That gap, between the thing you can train against every day and the thing you are actually judged on, is exactly the gap between a model's loss curve and its business metric.</p>`)}

<p>The gap matters most when the business's real question does not match the loss's shape at all. A model trained on logloss is answering "how well-calibrated are my probability estimates, on average, across the whole population?" A credit team asking for recall at a fixed 3 per cent approval rate is asking an entirely different question: "of the applicants I can actually afford to approve, how many of the good ones did I catch?" These are not the same question phrased two ways. They can have different answers for the same trained model, and the fix is never to retrain from scratch — it is to keep the well-behaved loss, which is doing its job, and separately choose where to place the decision threshold against the metric the business actually reads (§2.13).</p>

${H.table(['', 'Loss', 'Metric'], [
      ['Purpose', 'Guide the optimiser', 'Decide whether to ship'],
      ['Constraints', 'Differentiable, smooth, decomposes over examples', 'Anything the business can read'],
      ['Examples', 'logloss, MSE, hinge, Poisson deviance', 'AUC, KS, precision@k, £ per approval, lift'],
      ['When they diverge', 'You optimise the wrong thing efficiently', 'Fix: reweight, change the threshold (§2.13), or change the loss']
    ])}

<h2><span class="sn">2.1.3</span> The split, and the seal</h2>

<p>Empirical risk gave you a number computed on data you hold. Concentration (§1.4) told you that number is trustworthy — <i>provided</i> nothing about how you built the model depended on that same data. That proviso is the entire reason a single dataset gets carved into three pieces rather than used whole.</p>

<p><b>Train</b> is what the optimiser actually sees: the rows whose loss gets minimised, the ones §0.3's gradient descent walks downhill on. <b>Validation</b> is where you make every choice a human has to make that training cannot make for itself — which hyperparameters, which feature set, which model family, when to stop. <b>Test</b> is opened exactly once, at the very end, to produce the one number that goes in the deck, and then it is never looked at again for this model.</p>

<p>The reason for the third pile, and not just two, is that validation is not innocent. The moment you use a segment of data to <i>choose</i> anything — a learning rate, a threshold, whether random forest beat gradient boosting on this run — you have let information from that segment leak into your model indirectly, through your own decisions. Trying every hyperparameter and reporting whichever gave the best validation score is a search, and a search over noisy estimates finds noise that happens to look good exactly as reliably as it finds real improvement. Test exists to be the one segment that no search, however well-intentioned, has ever touched.</p>

<p>It is worth seeing exactly how much optimism a search can manufacture out of nothing, because the effect is larger and cheaper to trigger than intuition suggests. Take the simplest possible case: two models, both genuinely useless — coin flips with true AUC exactly 0.500 — scored on the same 500-row test set. Because the test set is finite, each model's <i>measured</i> AUC is not exactly 0.500; it wobbles around the truth with some standard error that shrinks as the test set grows, here about $\\text{se} \\approx 0.045$. If you then report whichever of the two happened to score higher, you are not reporting a random draw from that wobble any more — you are reporting its maximum, and a maximum is a biased estimate even when every ingredient going into it is perfectly unbiased.</p>

${H.deriv('why the best of even two noisy, unbiased scores is already above the truth', [
      ['$\\max(X_1, X_2) = \\dfrac{X_1+X_2}{2} + \\dfrac{|X_1-X_2|}{2}$', 'A plain algebraic identity for any two real numbers: the larger one equals their average plus half the gap between them. Nothing here has used probability yet.'],
      ['$\\mathbb{E}[\\max(X_1,X_2)] = \\mu + \\tfrac12\\,\\mathbb{E}|X_1-X_2|$', 'Take expectations of both sides. $X_1$ and $X_2$ each have mean $\\mu$ (here $\\mu=0.5$, the true AUC), so the average term contributes exactly $\\mu$; the gap term survives because $|X_1-X_2|$ is never negative, so its expectation cannot cancel to zero.'],
      ['$X_1 - X_2 \\sim \\mathcal{N}(0,\\, 2\\,\\text{se}^2)$', 'Both scores are independent draws with the same variance $\\text{se}^2$ around the truth, and the variance of a difference of independent quantities is the sum of their variances (§0.4).'],
      ['$\\mathbb{E}|X_1-X_2| = \\text{se}\\sqrt2 \\cdot \\sqrt{2/\\pi} = \\dfrac{2\\,\\text{se}}{\\sqrt\\pi}$', 'A standard fact about a zero-mean normal with standard deviation $\\sigma$: its expected absolute value is $\\sigma\\sqrt{2/\\pi}$. Plug in $\\sigma = \\text{se}\\sqrt2$ from the line above.'],
      ['$\\mathbb{E}[\\max(X_1,X_2)] = \\mu + \\dfrac{\\text{se}}{\\sqrt\\pi}$', 'Substitute back into line 2 and let the factor of 2 cancel. This is the whole result: picking the winner of two noisy, equally-worthless scores costs you $\\text{se}/\\sqrt\\pi$ of upward bias, for free, every single time.']
    ], 'Put numbers on it: with se ≈ 0.045, comparing just two coin-flip models already inflates the reported winner to 0.5 + 0.045/1.772 ≈ 0.525 on average — a model that is exactly as good as guessing reports itself as measurably above chance. The bias grows slowly (like the square root of the log of the candidate count) as you try more models, which is exactly the shape a simulation of this lab traces: at 30 candidates the average reported best sits near 0.59, and at 200 candidates near 0.62 — still every model a coin flip. This is the identical mechanism as multiple hypothesis testing (§1.6), wearing a different hat.')}

${H.history(`<p>Before held-out evaluation was standard practice, the obvious thing to do with a model was report how it performed on the data used to build it — and it repeatedly, reliably overstated how the model would do on anything new. The gap was not a rare embarrassment; it was the default outcome, because a flexible enough model can always be tuned to explain the particular noise sitting in front of it. Cross-validation and the train/validation/test split are the field's answer, and they carry the same logic as a pre-registered clinical trial or a sealed ballot box: the discipline is not there to catch dishonesty, it is there because a procedure that lets you peek and adjust will find something that looks like a result even when nothing real is there, and it will do so whether or not anyone involved intended to cheat.</p>`)}

<p><b>What you are looking at.</b> The chart plots one dot per candidate model, all pure noise with a true AUC of exactly 0.500 — the dashed green line. The x-axis is just which candidate you are looking at; the y-axis is the AUC each one happened to measure on the sealed 500-row test set. One dot, the reddest one, is whichever candidate scored highest — the one a real project would select and report.</p>

<p><b>What to do with it.</b> Leave the sliders at their defaults and read off the red dot's height against the green truth line: that gap is pure selection artefact, nothing else. Now drag <i>candidate models tried</i> from 30 up toward 200 and watch the red dot climb further from 0.500 even though not one underlying model has changed. Then drag <i>test-set size</i> up: the same search over the same number of candidates produces a smaller gap, because a bigger sealed set has a smaller standard error to exploit.</p>

<p><b>The thing genuinely worth noticing.</b> The amber dot shows that identical winning model, scored again on a completely fresh 500-row test set it has never touched. It falls straight back toward 0.500. Nothing about the model changed between the two measurements — only which data was allowed to grade it. That collapse is what "the test set is a bank vault, not a scratchpad" is actually protecting you from: the number you report has to come from a segment that never took part in the selection, or it is measuring the selection procedure rather than the model.</p>

${H.note('The test set is a bank vault, not a scratchpad. If you looked at it twice, say so out loud in the interview — candour here reads as seniority.')}

${H.lab('burn', 'Watch a test set burn', 'Every model here is pure noise — none has any real signal. Select the best one by test score and the reported number climbs anyway. That climb is the optimism you import every time you choose using the sealed segment.')}

<h2><span class="sn">2.1.4</span> Where the split rules change</h2>

<p>A plain random split — shuffle every row, cut it into three piles — assumes every row is an independent, interchangeable draw from the population you care about. That assumption is the quiet load-bearing wall behind everything in §2.1.3, and several ordinary data structures knock it straight down.</p>

<p><b>Time.</b> If you are predicting whether a customer churns next month, a random split will happily put a March row in training and a January row in test, which means the model is allowed to learn from the future to predict the past. In production you will never have that luxury — you will always be predicting forward from whatever has already happened — so evaluating any other way tells you nothing about the situation you are actually going to be in. The fix is an <b>out-of-time split</b>: train on everything before a cutoff date, test on everything after it, mirroring deployment exactly (§2.14 builds the rolling-origin version of this for when a single cutoff is not enough).</p>

<p><b>Entities with many rows.</b> Suppose your dataset has fifty transactions per customer rather than one. A random split scatters each customer's rows across both train and test, and a model can then partly succeed by recognising <i>the customer</i> — their typical spending pattern, their usual merchant — rather than learning the general pattern of fraud you actually wanted it to generalise. The reported test accuracy will look excellent and mean almost nothing the first time it meets a customer it has never seen. <b>Group k-fold</b>, which keeps every row belonging to one entity entirely on one side of the split, is the fix.</p>

<p><b>Rare positives.</b> If fraud is 1 in 500 transactions and you cut folds by ordinary random sampling, an unlucky fold can land with almost no fraud cases in it at all, and any metric computed on that fold is essentially noise. <b>Stratified k-fold</b> forces every fold to carry roughly the population's true positive rate, so no single split is degenerate by bad luck.</p>

<p><b>Nested geography or hierarchy.</b> Branches within a region, or patients within a hospital, share structure a random split does not respect — split at the row level and information about the region leaks between train and test through every other branch in it. Split at the coarsest unit you actually need the model to generalise across: if the deployment question is "will this work in a region we have not seen," hold out whole regions, not individual rows within them.</p>

${H.table(['Structure in the data', 'What a random split breaks', 'Correct split'], [
      ['Time', 'Trains on the future to predict the past', 'Out-of-time; rolling origin (§2.14)'],
      ['Entities with many rows', 'The model recognises the customer, not the pattern', 'Group k-fold on entity id'],
      ['Rare positives', 'A fold may contain almost no positives', 'Stratified k-fold'],
      ['Nested geography / hierarchy', 'Leaks neighbourhood-level information', 'Split on the coarsest unit you must generalise across']
    ])}

<p>Every row in that table is the same underlying failure wearing a different costume: the split stopped matching the question "how will this model perform on data it has genuinely never seen in any sense that matters." Before accepting a random split, it is worth asking explicitly what kind of never-seen-before you are trying to simulate — a customer, a date, a region — because that answer tells you exactly what has to be held out whole.</p>

${H.probe([
      ['Loss or metric — which do you optimise?', 'You optimise the loss and report the metric; naming the gap between them is the senior answer.'],
      ['Why is a random split wrong for time series?', 'It trains on the future. Use out-of-time validation and rolling-origin backtests.']
    ], 'Tuning a threshold on the test set and then reporting test performance at that threshold.')}`,
    labs: {
      burn: function (host) {
        const st = Viz.controls(host, [
          { k: 'k', label: 'candidate models tried', min: 1, max: 200, step: 1, value: 30, fmt: v => v },
          { k: 'n', label: 'test-set size', min: 100, max: 5000, step: 100, value: 500, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'best', label: 'best test AUC seen', cls: 'bad' },
          { k: 'truth', label: 'true AUC of that model', cls: 'key' },
          { k: 'opt', label: 'optimism imported' },
          { k: 'fresh', label: 'on a fresh test set' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(97);
            // every model is pure noise: true AUC 0.5. Observed AUC ~ N(0.5, se)
            const se = 0.5 / Math.sqrt(st.n / 4);
            const obs = [], fresh = [];
            for (let i = 0; i < st.k; i++) { obs.push(0.5 + R.normal(0, se)); fresh.push(0.5 + R.normal(0, se)); }
            let bi = 0; obs.forEach((v, i) => { if (v > obs[bi]) bi = i; });
            const P = Viz.plot(ctx, w, h, { xd: [0, Math.max(2, st.k)], yd: [0.5 - 4 * se, 0.5 + 4 * se] })
              .frame({ xlabel: 'candidate model', ylabel: 'measured AUC on the sealed test set', yfmt: v => v.toFixed(3) });
            P.clip(() => {
              obs.forEach((v, i) => P.dots([[i + .5, v]], { r: 3.4, color: i === bi ? T.red : T.blue, alpha: i === bi ? 1 : .6, stroke: i === bi }));
              P.hline(.5, { color: T.green, dash: [5, 4], label: 'the truth: every model is noise, AUC = 0.500' });
              P.dots([[bi + .5, fresh[bi]]], { r: 5, color: T.amber, stroke: true });
              P.text(bi + .5, fresh[bi], '  same model, fresh test set', { color: T.amber, font: '11px ui-sans-serif' });
            });
            out({
              best: obs[bi].toFixed(3), truth: '0.500',
              opt: '+' + ((obs[bi] - .5) * 1000).toFixed(0) + ' bps',
              fresh: fresh[bi].toFixed(3)
            });
          }
        });
        Viz.note(host, 'The best-of-k maximum drifts upward as k grows — with 200 candidates on a 500-row test set you will "find" an AUC near 0.55 in pure noise. This is the same mechanism as multiple testing (§1.6), and it is why the number you report must come from a segment no decision has touched.');
      }
    },
    quiz: [
      {
        q: 'You compare 40 model configurations on the test set and report the best. The reported number is…',
        options: ['unbiased', 'optimistically biased by the selection itself', 'pessimistic', 'unbiased if you used cross-validation for training'],
        answer: 1,
        why: 'Reporting the best of forty scores computed on the same sealed test set is reporting the maximum of forty noisy, individually unbiased estimates, and a maximum is a biased estimate of the truth even when every ingredient going into it is unbiased on its own — §2.1.3\'s derivation shows this bias grows with the number of candidates tried, not merely with luck. "Unbiased" is the tempting answer because each configuration\'s own score genuinely is unbiased for its own true performance; the distortion appears only once you start picking whichever one happens to look best, which is a property of the selection procedure, not of any single model. "Unbiased if you used cross-validation for training" is a second common confusion: cross-validation makes the training process itself more honest, but it says nothing about what happens afterwards if you then compare many such honest estimates and report the winner — the winner is still a maximum. The fix is architectural, not a fancier evaluation trick on the same data: select using validation, and open the test set exactly once, after every decision has already been made.'
      },
      {
        q: 'A model is trained on logloss but the business judges recall at a fixed 3% approval rate. The cleanest first response is…',
        options: ['Retrain with recall as the loss', 'Keep the loss, move the operating threshold, and report at the business operating point', 'Resample the data until recall improves', 'Report AUC instead'],
        answer: 1,
        why: 'Recall at a fixed approval rate is a statement about where you cut a ranked list, not about how the model was trained, so the right move is to leave the well-behaved, differentiable loss alone and choose the operating threshold deliberately against the business constraint, exactly the machinery §2.13 builds for turning a cost structure into a cut-off. Retraining with recall as the objective is the tempting first instinct, because it sounds like it targets what was asked for directly, but recall computed at a hard threshold is flat almost everywhere and has no usable gradient (§2.1.2) — you cannot train against a staircase the way you train against logloss. Resampling the data shifts the base rate the model sees and quietly wrecks its calibration (§2.12) without touching the actual mismatch, which was never about the loss function to begin with. Reporting AUC instead dodges the question rather than answering it, since the business asked for a number at one specific operating point, not a ranking summary averaged over all of them. The general lesson is that a mismatch between the loss you optimise and the metric you are judged on is very often a thresholding problem wearing a training-objective costume.'
      }
    ],
    cards: [
      { q: 'Loss vs metric', a: 'Loss = what you optimise (differentiable); metric = what you report (anything). The gap is where production disappointment lives.' },
      { q: 'The split rule', a: 'Train fits, validation chooses, test is opened once. Every decision made while looking at test imports optimism.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.2 */
  ML.section({
    id: 'bias-variance', track: 'classical', num: '2.2',
    title: 'Bias–variance, over/underfitting, double descent',
    lede: 'The frame for every model choice in this part — and the modern half of the picture that most candidates are a decade out of date on.',
    rests: 'Rests on §1.3.',
    html: `
<p>Fit a straight line to a scatter of noisy points and it misses the obvious curve in the data — every point is a little wrong in a way the model can never fix, however much data you feed it. Fit a degree-15 polynomial to the same handful of points instead, and it snakes through every single one exactly, error zero, and then does something wild the instant you ask it about a point it did not see. Two failures, both real, and they feel like opposite diseases: one model was not paying attention, the other was paying too much attention to the wrong thing. What this section does is show that they are literally two terms in the same equation, measured the same way, on the same axis.</p>

<h2><span class="sn">2.2.1</span> The decomposition</h2>

<p>Fix one input point $x$ and imagine repeating the entire experiment many times: draw a fresh training set, fit the model, record its prediction at that one point. Because the training set is random, the fitted model $\\hat f$ is a random object, and its prediction $\\hat f(x)$ is a random number that lands somewhere different each time. Two separate things can go wrong with the cloud of predictions this produces. It can be centred in the wrong place — every fit is systematically off in the same direction, however many times you repeat the experiment. Or it can be centred correctly but scattered wide — right on average, but wildly different from one training set to the next, so any single fit you actually deploy is a roll of the dice. The first is <b>bias</b>. The second is <b>variance</b>. Squared error at $x$, averaged over both the noise in $y$ and the randomness of which training set you happened to draw, decomposes exactly into these two pieces plus one more:</p>

$$\\mathbb{E}[(y-\\hat f(x))^2] = \\underbrace{(\\mathbb{E}[\\hat f(x)]-f(x))^2}_{\\text{bias}^2} + \\underbrace{\\mathrm{Var}(\\hat f(x))}_{\\text{variance}} + \\sigma^2$$

<p>Read the right-hand side left to right. <b>Bias</b> squared is the gap between where your fits are centred, $\\mathbb{E}[\\hat f(x)]$, and the true function $f(x)$ — a systematic error that averaging more fits together cannot repair, because it is a statement about where the average lands, not about scatter around it. <b>Variance</b> is how far a single fit typically strays from that average — the model's sensitivity to exactly which random training set it happened to be given. And $\\sigma^2$ is the variance of the label noise itself, the part of $y$ that no function of $x$, however well chosen, could ever have predicted.</p>

${H.deriv('the bias–variance decomposition, term by term', [
      ['$y = f(x) + \\varepsilon, \\quad \\mathbb{E}[\\varepsilon]=0,\\ \\mathrm{Var}(\\varepsilon)=\\sigma^2$', 'The model of how labels are generated: a true underlying function plus independent noise. $\\varepsilon$ is independent of the training set, so it is independent of $\\hat f$ too — the noise on a fresh test point has nothing to do with which training set built the model.'],
      ['$\\mathbb{E}[(y-\\hat f)^2] = \\mathbb{E}\\big[\\big((f-\\mathbb{E}\\hat f) + (\\mathbb{E}\\hat f-\\hat f) + \\varepsilon\\big)^2\\big]$', 'Substitute $y=f+\\varepsilon$, then add and subtract $\\mathbb{E}[\\hat f]$ inside the bracket — a number that changes nothing, since it adds zero. This regroups the error into three named pieces: a constant (the bias), a zero-mean random deviation (how far this particular fit is from the average fit), and the label noise.'],
      ['$= \\mathbb{E}[(f-\\mathbb{E}\\hat f)^2] + \\mathbb{E}[(\\mathbb{E}\\hat f-\\hat f)^2] + \\mathbb{E}[\\varepsilon^2] + \\text{cross terms}$', 'Expand the square of a sum of three terms exactly as you would $(p+q+r)^2$ in ordinary algebra: three squared terms plus three cross terms.'],
      ['every cross term has expectation zero', '$(f-\\mathbb{E}\\hat f)$ is a fixed constant, and $\\mathbb{E}[\\mathbb{E}\\hat f-\\hat f]=0$ by definition of $\\mathbb{E}\\hat f$, so the constant-times-deviation cross term vanishes. $\\varepsilon$ is independent of $\\hat f$ and of $f$, and has mean zero itself, so both terms involving $\\varepsilon$ vanish too.'],
      ['$\\mathbb{E}[(y-\\hat f)^2] = (f-\\mathbb{E}\\hat f)^2 + \\mathrm{Var}(\\hat f) + \\sigma^2$', 'What survives is exactly the three terms in the boxed formula: the squared bias (a constant, so its own expectation is itself), the variance of $\\hat f$ (the expected squared deviation of a random variable from its own mean, which is the definition of variance), and the irreducible noise.']
    ], 'σ² is a floor, not a target. It was baked into the data before you chose a model, so no amount of cleverness moves it — the honest goal of everything else in this section is trading bias against variance to get as close to that floor as the data allow.')}

${H.worked('worked number — when a deeper tree loses, and what fixes it', `
<p>Current model: bias² = 0.04, variance = 0.02, σ² = 0.01. Read off the boxed formula directly: expected error = 0.04 + 0.02 + 0.01 = <b>0.07</b>.</p>
<p>Deepen the tree so it can represent more structure. Bias² drops to 0.01, because a deeper tree really can bend to fit the true shape more closely. But variance rises to 0.06, because a deeper tree also has enough freedom to fit the particular noise in whichever training set it was handed — a different training set now produces a visibly different tree. Expected error = 0.01 + 0.06 + 0.01 = <b>0.08</b>. The fit on paper improved and the model that matters got worse.</p>
<p>This is the arithmetic behind every "why did my deeper model do worse in production" postmortem. Now bag a hundred such deep, high-variance trees — average their predictions instead of shipping one — and use the averaging formula for $B$ predictors with pairwise correlation $\\rho$ between them, $\\mathrm{Var}(\\text{average}) = \\rho\\sigma^2+(1-\\rho)\\sigma^2/B$ (derived from first principles in §1.3). With $\\sigma^2=0.06$, a realistic $\\rho=0.3$ between trees grown on overlapping bootstrap samples, and $B=100$: variance falls to $0.3\\times0.06 + 0.7\\times0.06/100 = 0.018+0.00042 \\approx \\mathbf{0.0184}$. Total expected error is now $0.01+0.0184+0.01 \\approx \\mathbf{0.038}$ — better than the shallow tree <i>and</i> the single deep one, because bagging kept the bias gain from going deep while giving back nearly all of the variance it cost. <mark>That is the entire idea of a random forest, in one line of arithmetic.</mark></p>`)}

${H.intuition(`<p>There is a reason "more flexible model" and "more variance" travel together so reliably, and it is worth stating plainly: a model with more free parameters has more ways to bend, and some of those ways will always happen to align with whatever noise sits in your particular training sample. A straight line has so little freedom that it cannot chase the noise even if it wanted to — which is exactly why its variance is low and its bias is stubbornly whatever it is. A degree-15 polynomial has fifteen ways to wiggle, more than enough to route itself through every noisy point exactly, and which fifteen wiggles it picks depends entirely on which noise it happened to see. Flexibility does not choose between fitting the signal and fitting the noise; it fits whatever is put in front of it, and the noise is different every time.</p>`)}

<p><b>What you are looking at.</b> Twenty independent training sets of the same size, drawn from the same underlying process, each fitted with a polynomial of the degree you choose. Every faint grey curve is one of those twenty fits. The thick blue curve is their average — a numerical stand-in for $\\mathbb{E}[\\hat f(x)]$, the quantity bias is measured against. The dashed green curve is the true function $f(x)$ those twenty datasets were generated from, which in real work you never get to see; the lab shows it to you so the two error sources become visible as two separate geometric facts rather than two abstract numbers.</p>

<p><b>What to do with it.</b> Set the polynomial degree to 0 or 1 and look at the gap between blue and green — that visible gap is bias, and it is large, because a near-flat curve simply cannot bend to follow the true shape. Now look at the grey curves: they sit almost on top of each other, because a model with almost no freedom fits almost the same way regardless of which twenty points it was given. Push the degree up past 8 or so and watch the picture invert: the blue average now tracks the green truth closely, so bias has nearly vanished, but the grey curves fan out wildly between the training points, each one snaking through its own sample's particular noise. That fan is variance, drawn rather than computed.</p>

<p><b>The thing genuinely worth noticing.</b> Leave the degree high, where the grey curves are chaos, and slide <i>ridge λ</i> up from zero. Watch the grey fan pull back in toward the blue average almost immediately, while the blue average drifts slightly away from the green truth. You are watching regularisation trade variance for bias in real time, on the same picture, with nothing hidden — which is the entire content of §2.3 rendered as a moving fan of curves instead of an equation.</p>

${H.lab('bv', 'Bias and variance, measured not asserted', 'Twenty independent training sets, one model family, fitted repeatedly. The grey curves are the individual fits; the blue is their average. Bias is the gap between the blue curve and the truth; variance is the spread of the grey ones. Move the degree and watch the two trade.')}

<h2><span class="sn">2.2.2</span> Double descent</h2>

<p>Everything so far predicts a single U-shaped curve: increase model capacity, bias falls, variance rises, and somewhere in the middle their sum is smallest. That is the classical picture, it is correct as far as it goes, and for decades it was assumed to be the whole story — keep adding capacity past the U's minimum and test error should simply keep climbing, because an ever-more-flexible model has an ever-larger menu of ways to memorise noise.</p>

<p>Modern practice broke that assumption in a very specific, reproducible place. Keep adding parameters until the model has <i>exactly</i> as many free parameters as there are training examples — the <b>interpolation threshold</b> — and the model can now fit the training data perfectly, error zero, by solving what is essentially a square system of equations with no slack left in it. Right at that threshold there is usually exactly one way to hit every training point exactly, and whichever one that is depends with extreme sensitivity on the specific noise in your specific sample: this is the same fragility as a design matrix on the edge of losing invertibility, the near-singular $X^\\mathsf{T}X$ from §0.7, where a tiny change to the data swings the fitted solution wildly. Test error at the interpolation threshold spikes for exactly that reason — not because the model is complex, but because the fitting problem has become numerically unstable.</p>

<p>Keep adding parameters past that point, though, and something genuinely surprising happens: test error <i>falls again</i>. Past the threshold there is no longer a single way to fit the training data exactly — there are infinitely many, an entire family of parameter settings that all achieve zero training error. Which one you end up with now depends on the optimiser, not just the data, and gradient descent and its relatives have a systematic preference among that family: among all the solutions that fit the training data equally well, they gravitate toward the one with the smallest norm, the least ornate. That preference is a form of regularisation nobody wrote into the loss function — <b>implicit regularisation</b> — and it does, silently, the same job that adding $\\lambda\\|w\\|^2$ to the loss does explicitly in §2.3. The more parameters you add beyond the threshold, the more room the optimiser has to find a smooth, small-norm interpolant instead of a jagged, forced one, and test error descends a second time.</p>

${H.history(`<p>The classical U-curve is not wrong within the regime it was derived for — it comes from decades of statistical learning theory built around models with far fewer parameters than training examples, where "more capacity always eventually overfits" was a completely reasonable summary of the evidence. What changed was the arrival of models, particularly neural networks, routinely trained with more parameters than data points, a regime the classical theory was never built to describe. Documented under the name double descent by machine learning theorists in the late 2010s, the phenomenon did not overturn bias–variance decomposition — every equation in §2.2.1 still holds exactly — it revealed that variance is not the monotonically increasing function of capacity the classical picture assumed. Past the interpolation threshold, more parameters can mean <i>less</i> effective variance, because the optimiser is now choosing among solutions rather than being forced into the one that happens to fit.</p>`)}

<p><b>What you are looking at.</b> A single family of models — random-feature regression, essentially a linear model on top of a bank of fixed random nonlinear features — fitted at increasing <i>width</i>, the number of those random features, from 1 up to 120. The dashed grey curve is training error; the solid blue curve is error measured on held-out data the fit never saw. The vertical red line marks the interpolation threshold, where the number of parameters equals the number of training samples $n$.</p>

<p><b>What to do with it.</b> With ridge λ at zero, watch training error (dashed grey) fall steadily toward zero as width grows — no surprise there, more parameters fit the training data better, exactly as classical intuition says. Now watch the blue test-error curve instead: it falls at first, just like the classical U-curve predicts, then spikes sharply right at the red interpolation-threshold line, then — and this is the part worth stopping on — falls again as width keeps growing past that line, often ending up lower than anywhere on the curve's left-hand side.</p>

<p><b>The thing genuinely worth noticing.</b> Raise ridge λ away from zero, even slightly, and watch the spike at the threshold soften dramatically while the rest of the curve barely moves. Explicit regularisation is filling in exactly the instability that implicit regularisation was covering for on its own — the same $\\lambda$ that added a constant to the diagonal of $X^\\mathsf{T}X$ in §0.7 is here damping the exact near-singular fitting problem that produces the spike. That single slider move is the cleanest demonstration on this site that the classical and modern pictures are not rival theories; they are the same mechanism, with and without the extra help.</p>

${H.lab('dd', 'The U-curve and its second half', 'Real fits: ridge-regularised random-feature regression at increasing width, evaluated on held-out data. The peak at parameters ≈ samples is real and reproducible; add a little ridge and watch it soften — which is the modern reading of what implicit regularisation does.')}

${H.probe([
      ['High variance — what do you do?', 'More data, stronger regularisation, a simpler hypothesis class, or bagging. In that order of preference if data is available.'],
      ['Does double descent kill bias–variance?', 'No — it extends it into the over-parameterised regime where implicit regularisation dominates.']
    ], 'Assuming more capacity always overfits. That belief is a decade out of date.')}`,
    labs: {
      bv: function (host) {
        const st = Viz.controls(host, [
          { k: 'deg', label: 'polynomial degree', min: 0, max: 12, step: 1, value: 3, fmt: v => v },
          { k: 'n', label: 'points per training set', min: 8, max: 60, step: 2, value: 16, fmt: v => v },
          { k: 'noise', label: 'noise σ', min: .05, max: 1, step: .05, value: .35, fmt: v => v.toFixed(2) },
          { k: 'lam', label: 'ridge λ', min: 0, max: 2, step: .02, value: 0, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'bias', label: 'bias²', cls: 'key' }, { k: 'var', label: 'variance' },
          { k: 'noise', label: 'σ² (irreducible)' }, { k: 'tot', label: 'total expected error' }
        ]);
        const truth = x => Math.sin(1.3 * x) * 1.4 + .25 * x;
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const REP = 20;
            const grid = []; for (let i = 0; i <= 60; i++) grid.push(-3 + 6 * i / 60);
            const curves = [];
            for (let r = 0; r < REP; r++) {
              const R = Num.rng(1000 + r * 37);
              const xs = [], ys = [];
              for (let i = 0; i < st.n; i++) { const x = -3 + 6 * R(); xs.push(x); ys.push(truth(x) + R.normal(0, st.noise)); }
              const beta = Num.ridgeFit(Num.polyDesign(xs, st.deg), ys, st.lam + 1e-8);
              curves.push(grid.map(x => Num.dot(beta, Num.polyDesign([x], st.deg)[0])));
            }
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3.5, 3.5] }).frame({ xlabel: 'x', ylabel: 'y' });
            P.clip(() => {
              curves.forEach(c => P.line(grid.map((x, i) => [x, c[i]]), { color: T.faint, width: 1, alpha: .45 }));
              const avg = grid.map((_, i) => Num.mean(curves.map(c => c[i])));
              P.line(grid.map((x, i) => [x, avg[i]]), { color: T.blue, width: 2.8 });
              P.fn(truth, { color: T.green, width: 2.2, dash: [6, 4] });
            });
            const avg = grid.map((_, i) => Num.mean(curves.map(c => c[i])));
            const bias2 = Num.mean(grid.map((x, i) => (avg[i] - truth(x)) ** 2));
            const varr = Num.mean(grid.map((_, i) => Num.variance(curves.map(c => c[i]))));
            out({
              bias: bias2.toFixed(3), var: varr.toFixed(3), noise: (st.noise * st.noise).toFixed(3),
              tot: (bias2 + varr + st.noise * st.noise).toFixed(3)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().green, t: 'truth f(x)' }, { c: Viz.theme().blue, t: 'average fit E[f̂]' }, { c: Viz.theme().faint, t: '20 individual fits' }]);
        Viz.note(host, 'Degree 0–1 → the blue average sits far from green (bias) and the grey curves are tight (low variance). Degree 10+ → blue tracks green but grey scatters wildly. Add ridge λ and watch variance collapse while bias creeps up: that is regularisation, measured.');
      },

      dd: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'training samples n', min: 15, max: 60, step: 5, value: 30, fmt: v => v },
          { k: 'lam', label: 'ridge λ', min: 0, max: 1, step: .002, value: 0, fmt: v => v.toFixed(3) },
          { k: 'noise', label: 'noise σ', min: .05, max: .8, step: .05, value: .3, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'peak', label: 'peak test error at', cls: 'key' }, { k: 'best', label: 'best width' }, { k: 'final', label: 'error at max width' }
        ]);
        const S = Viz.surface(host, {
          height: 310,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(5);
            const f = x => Math.sin(2.2 * x) + .4 * x;
            const xs = [], ys = [];
            for (let i = 0; i < st.n; i++) { const x = -2 + 4 * R(); xs.push(x); ys.push(f(x) + R.normal(0, st.noise)); }
            const xte = [], yte = [];
            for (let i = 0; i < 300; i++) { const x = -2 + 4 * R(); xte.push(x); yte.push(f(x)); }
            // random-feature model: features = cos(w x + b), width p
            const widths = [];
            for (let p = 1; p <= 120; p += 2) widths.push(p);
            const RW = Num.rng(31);
            const Wf = [], Bf = [];
            for (let j = 0; j < 130; j++) { Wf.push(RW.normal(0, 2.2)); Bf.push(RW() * 6.28); }
            const curveTr = [], curveTe = [];
            widths.forEach(p => {
              const Phi = xs.map(x => Array.from({ length: p }, (_, j) => Math.cos(Wf[j] * x + Bf[j])));
              const beta = Num.ridgeFit(Phi, ys, st.lam + 1e-9);
              const predTr = Phi.map(r => Num.dot(r, beta));
              const PhiTe = xte.map(x => Array.from({ length: p }, (_, j) => Math.cos(Wf[j] * x + Bf[j])));
              const predTe = PhiTe.map(r => Num.dot(r, beta));
              curveTr.push([p, Math.min(4, Num.mean(ys.map((v, i) => (v - predTr[i]) ** 2)))]);
              curveTe.push([p, Math.min(4, Num.mean(yte.map((v, i) => (v - predTe[i]) ** 2)))]);
            });
            const P = Viz.plot(ctx, w, h, { xd: [1, 120], yd: [0, Math.min(3, Math.max.apply(null, curveTe.map(c => c[1]))) * 1.05] })
              .frame({ xlabel: 'model width (number of random features)', ylabel: 'mean squared error' });
            P.clip(() => {
              P.line(curveTr, { color: T.faint, width: 1.6, dash: [5, 4] });
              P.line(curveTe, { color: T.blue, width: 2.6 });
              P.vline(st.n, { color: T.red, label: 'params ≈ n — the interpolation threshold' });
            });
            let peak = curveTe[0], best = curveTe[0];
            curveTe.forEach(c => { if (c[1] > peak[1]) peak = c; if (c[1] < best[1]) best = c; });
            out({ peak: 'width ' + peak[0], best: 'width ' + best[0], final: curveTe[curveTe.length - 1][1].toFixed(3) });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'test error' }, { c: Viz.theme().faint, t: 'train error' }, { c: Viz.theme().red, t: 'interpolation threshold' }]);
        Viz.note(host, 'Set λ = 0 and the spike at width ≈ n is dramatic; add a little ridge and it flattens. That is the whole modern story: at the interpolation threshold the minimum-norm solution is badly behaved, and either explicit regularisation or the optimiser’s implicit bias tames it.');
      }
    },
    quiz: [
      {
        q: 'bias² = 0.04, variance = 0.02, σ² = 0.01. You deepen the model: bias² → 0.01, variance → 0.06. What happened to expected error?',
        options: ['Fell from 0.07 to 0.05', 'Rose from 0.07 to 0.08', 'Unchanged', 'Cannot tell without more data'],
        answer: 1,
        why: 'Summing the new pieces gives $0.01+0.06+0.01=0.08$, strictly worse than the original $0.04+0.02+0.01=0.07$, even though the deeper model fits the training data more closely than before — a smaller bias² bought a larger variance, and the trade came out unfavourable overall. "Fell from 0.07 to 0.05" is the tempting answer if you assume variance must fall along with bias whenever a model is "improved," rather than actually adding the three stated terms; deepening a model routinely moves bias and variance in opposite directions, and the two must be summed, not netted by eye. "Unchanged" ignores that both bias² and variance genuinely moved by the stated amounts, and "cannot tell without more data" overlooks that the decomposition already supplies every number the question needs. The general lesson, developed in §2.2 and put to direct use in §2.16 for why ensembling exists at all, is that a model which fits training data more closely is not automatically a better model — only the sum of all three terms, never any one of them in isolation, tells you what happens on new data.'
      },
      {
        q: 'Test error peaks when parameters ≈ samples and then falls again with more parameters. This is…',
        options: ['a bug in the evaluation', 'double descent — the over-parameterised regime where implicit regularisation dominates', 'evidence that bias–variance is wrong', 'label leakage'],
        answer: 1,
        why: 'This is double descent, §2.2.2\'s extension of the classical bias–variance picture into the over-parameterised regime: past the interpolation threshold, where a model has enough parameters to fit the training data in more than one way, the optimiser\'s implicit preference for small-norm solutions acts as a form of regularisation nobody wrote into the loss, and test error falls a second time. "A bug in the evaluation" is the tempting reaction the first time you see this curve, because for decades the classical U-shape was the entire story and a second descent looks like something must have leaked between train and test — but the effect is real, reproducible, and concentrated specifically where parameters roughly equal the sample count. "Evidence that bias–variance is wrong" overstates the finding: every equation in the decomposition still holds exactly, because double descent is about variance behaving non-monotonically in parameter count, not about the decomposition breaking down. "Label leakage" would produce suspiciously good performance throughout training and validation alike, not a sharp peak concentrated right at the numerically unstable interpolation point, which is the specific signature of this phenomenon rather than of a data problem.'
      }
    ],
    cards: [
      { q: 'Bias–variance decomposition', a: '$\\mathbb{E}[(y-\\hat f)^2] = \\text{bias}^2 + \\text{variance} + \\sigma^2$; $\\sigma^2$ is the irreducible noise floor.' },
      { q: 'High variance — the fixes in order', a: 'More data → stronger regularisation → simpler class → bagging.' },
      { q: 'Double descent', a: 'Test error peaks at the interpolation threshold (params ≈ n) then descends again; variance is not monotone in parameter count.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.3 */
  ML.section({
    id: 'regularization', track: 'classical', num: '2.3',
    title: 'Regularisation: L1, L2, elastic net',
    lede: 'Rests on MAP (§1.5); enables the sparse, defensible scorecards of §2.11. Two arguments for why L1 is sparse — one geometric, one analytic, and the analytic one is the better answer.',
    html: `
<p>You just watched, in §2.2's lab, a single slider called $\\lambda$ pull a fan of wild, overfit curves back into a tight bundle, at the cost of nudging their average slightly away from the truth. That slider is not a special trick belonging to that one lab. It is the general-purpose knob for buying back variance at the price of a little bias, and this section is entirely about how it is built, why two apparently similar ways of building it behave completely differently, and which one to reach for.</p>

<h2><span class="sn">2.3.1</span> Why penalise a coefficient at all</h2>

<p>Ordinary least squares, from §0.7 and §2.4, asks for exactly one thing: the weight vector $w$ that makes $\\|Xw-y\\|^2$ as small as possible on the data you have. Nothing in that objective cares how large $w$ gets. If two features are nearly duplicates of each other — a length recorded in metres and the same length recorded in centimetres, say — least squares is perfectly happy to assign one of them an enormous positive weight and the other an enormous negative weight that very nearly cancels it, because the training loss cannot tell the difference between that and any other combination that fits equally well. §0.7 showed exactly this: the design matrix's $X^\\mathsf{T}X$ becomes singular or nearly so, and the fitted weights become numerically unstable — free to swing wildly in response to noise that has nothing to do with the underlying signal. Huge, noise-chasing coefficients are not a symptom of a good fit; they are the geometric signature of high variance, the same disease §2.2 spent an entire section diagnosing.</p>

<p>Regularisation is the fix stated as a modelling decision rather than a numerical patch: add a second term to the loss that grows whenever the weights grow, so the optimiser is no longer asked only "fit the data" but "fit the data <i>and</i> keep the weights small." Which penalty you add determines what "small" means, and that choice turns out to matter enormously.</p>

<h2><span class="sn">2.3.2</span> Two penalties, two very different notions of small</h2>

<p><b>L2, or ridge</b>, adds $\\lambda\\|w\\|_2^2 = \\lambda\\sum_j w_j^2$ to the loss — a penalty on the sum of squared weights. §0.7 already derived exactly what this does to the normal equations: it turns $X^\\mathsf{T}Xw=X^\\mathsf{T}y$ into $(X^\\mathsf{T}X+\\lambda I)w=X^\\mathsf{T}y$, adding $\\lambda$ to every eigenvalue of $X^\\mathsf{T}X$ and worked a concrete example — two perfectly correlated columns, eigenvalues 28 and 0, repaired to 28.1 and 0.1 by $\\lambda=0.1$ — showing the singular direction lifted off the floor rather than left to swing freely. That is the whole mechanism of ridge, already proven; nothing new needs deriving here. The behavioural upshot is that ridge shrinks every coefficient smoothly toward zero, never quite arriving, and when two features are correlated it splits the credit between them roughly evenly rather than picking a favourite. All features stay in the model, just damped.</p>

<p><b>L1, or lasso</b>, adds $\\lambda\\|w\\|_1 = \\lambda\\sum_j |w_j|$ instead — the sum of absolute values rather than squares. It sounds like a minor variation on the same theme, differing only in whether you square the weight or take its absolute value before summing. It is not a minor variation. Lasso does not merely shrink coefficients; past some point it drives them to <i>exactly</i> zero, deleting the corresponding feature from the model entirely. Two penalties, one character apart in notation, produce qualitatively different objects: ridge gives you every feature at reduced strength, lasso gives you a shorter list. Understanding why requires looking at the same fitting problem two different ways — first as geometry, then as calculus — and the two views turn out to agree completely, which is reassuring, because they are describing the same fact from different angles.</p>

${H.intuition(`<p>Both penalties are also, without changing a single number, exactly MAP estimation (§1.5) under a choice of prior on the weights. §1.5.6 works the full derivation and it is not repeated here — only the answer is needed. A zero-mean Gaussian prior $w_j\\sim\\mathcal N(0,\\tau^2)$ combined with Gaussian label noise of variance $\\sigma^2$ gives, once the data term has been normalised to the plain sum-of-squared-errors form every library actually fits, a ridge penalty with $\\lambda=\\sigma^2/\\tau^2$ — the noise variance <i>over</i> the prior variance, not the prior's own coefficient $1/(2\\tau^2)$ read off one line too early. A Laplace prior on the weights gives the lasso penalty by the identical route, $\\lambda=2\\sigma^2/b$. So "add an L2 penalty" and "believe, before seeing any data, that large weights are a priori unlikely, in a bell-curve sort of way, and that my measurements carry a known amount of noise" are the same modelling choice stated in two vocabularies — and the fact that a Laplace prior concentrates more of its probability mass right at zero than a Gaussian does is the same fact, in the language of §1.5, that produces sparsity here.</p>`)}

${H.pitfall('The shortcut "λ equals the inverse of the prior variance" is missing a factor. It skips the step where the data term gets normalised to coefficient 1, and the true answer is a ratio of two variances, $\\lambda=\\sigma^2/\\tau^2$ — noisier data or a more confident prior both push $\\lambda$ up; a vaguer prior pushes it toward 0, recovering plain MLE. §1.5.6 walks the arithmetic and shows the shortcut is wrong by exactly a factor of $2\\sigma^2$.')}

<h2><span class="sn">2.3.3</span> Why L1 is sparse — the geometric argument</h2>

<p>Restate the fitting problem in its constrained form: instead of adding a penalty to the loss, minimise the loss subject to a hard budget on how large the weights are allowed to be, $\\|w\\|_p \\le t$. This is not a different problem from the penalised one — for every $\\lambda$ there is a matching $t$ that gives the identical solution — it is simply the version that is easier to draw.</p>

<p>Picture the loss contours: nested ellipses (circles, if the features happen to be uncorrelated) centred on the unconstrained least-squares optimum, growing outward the way ripples spread from a stone dropped in water. Overlay the constraint region — the set of weight vectors satisfying the budget. For an L2 budget that region is a disc, smooth and round. For an L1 budget it is a diamond, a square rotated forty-five degrees, with sharp corners sitting exactly on the axes. The constrained solution is wherever the smallest loss-contour ripple first touches the constraint region as it expands outward from the centre.</p>

<p>Here is a case worked through with real numbers rather than asserted. Suppose the unconstrained optimum sits at $a=(2,\\, 0.3)$ — feature 1 clearly matters more than feature 2, but feature 2 is not exactly zero either — with circular loss contours, and a budget $t=1$.</p>

${H.deriv('why the diamond forces an exact zero here, and the circle never does', [
      ['minimise $(w_1-2)^2+(w_2-0.3)^2$ s.t. $|w_1|+|w_2|\\le 1$', 'The constrained least-squares problem, stated for this specific optimum and budget. Because $2+0.3=2.3>1$, the unconstrained optimum lies outside the budget, so the constraint is active and the solution sits exactly on its boundary.'],
      ['on the near edge, $w_1=s,\\ w_2=1-s,\\ s\\in[0,1]$', 'The optimum $(2,0.3)$ sits deep in the positive quadrant, so the relevant edge of the diamond is the one joining $(1,0)$ to $(0,1)$ — parametrise it with one free number $s$.'],
      ['$f(s)=(s-2)^2+(1-s-0.3)^2$', 'Substitute the parametrisation into the objective, reducing a two-dimensional constrained problem to an ordinary one-variable minimisation over $s\\in[0,1]$.'],
      ['$f\'(s)=4s-3.4=0 \\ \\Rightarrow\\ s^\\star=1.35$', 'Differentiate and solve as usual (§0.3). The unconstrained minimiser of $f$ along this line sits at $s=1.35$.'],
      ['$s^\\star=1.35 \\notin [0,1]$, and $f$ is convex, so $f$ decreases across the whole interval', 'A convex parabola with its own minimum to the right of an interval is strictly decreasing throughout that interval, so the constrained minimum over $s\\in[0,1]$ sits at the interval\'s right-hand endpoint.'],
      ['$s=1 \\ \\Rightarrow\\ (w_1,w_2)=(1,0)$', 'A vertex of the diamond — feature 2\'s coefficient is <i>exactly</i> zero, not merely small. Contrast the circle of the same radius: the closest point to $(2,0.3)$ on $w_1^2+w_2^2=1$ is the radial projection $t\\cdot a/\\|a\\|=(0.989,\\,0.148)$ — both coordinates strictly positive, because a circle has no corners to snap to.']
    ], 'Nothing about this outcome depended on choosing especially convenient numbers. Whenever the optimum is noticeably closer to one axis than the other relative to the budget — precisely the situation of "one feature matters more than another" — the nearest point on the diamond is a corner. Only when the optimum sits almost exactly on the 45° line, meaning the two features are about equally important, does the touch point sit in the interior of an edge rather than at a vertex. The circle has no corners anywhere, so it never produces this behaviour, full stop.')}

<p><b>What you are looking at.</b> Solid red contour lines are levels of the loss, centred on the red dot — the unconstrained optimum. The blue outline is the constraint boundary: a diamond for L1, a circle for L2, a rounded diamond for elastic net. The green dot is the actual constrained solution: the point where the smallest loss contour that still touches the constraint region makes contact.</p>

<p><b>What to do with it.</b> With the penalty set to L1, drag the red dot anywhere in the plane away from the two diagonal lines and watch the green dot land almost exactly on one of the diamond's four points, meaning one of the two coordinates reads as exactly zero in the readout below. Now switch the penalty to L2 with the red dot in the same place: the green dot slides to a point off both axes, matching the projection arithmetic worked through above.</p>

<p><b>The thing genuinely worth noticing.</b> Drag the red dot slowly toward the 45° diagonal, where the two features are equally weighted in the unconstrained fit. Watch the L1 solution slide off its vertex and creep along the diamond's edge before it finally reaches the corner nearest the diagonal itself — sparsity is not guaranteed for every possible loss, only overwhelmingly likely for one drawn at random, which is exactly the caveat the derivation above earns honestly rather than sweeps under the rug.</p>

${H.lab('geom', 'Diamond versus circle — drag the optimum', 'Move the unconstrained optimum (the red dot) and watch where the expanding contours first touch the constraint set. On L1 the touch point snaps to a vertex over most of the plane; on L2 it never does.')}

<h2><span class="sn">2.3.4</span> The analytic argument — the better answer in an interview</h2>

<p>The geometric picture is convincing but it is a picture, and "draw it and look" is a weaker answer than being able to say <i>why</i> in the language of calculus. Differentiate each penalty with respect to one coordinate $w$, holding everything else fixed. Ridge's contribution to the gradient is $\\frac{d}{dw}\\big(\\lambda w^2\\big) = 2\\lambda w$: a pull toward zero whose <i>strength is proportional to how far $w$ already is from zero</i>. As $w$ shrinks, the pull shrinks right along with it — a coefficient at $w=0.01$ feels almost no restoring force at all, so gradient descent can spend forever inching closer without the pull ever quite finishing the job. Zero is a place ridge's pull approaches but essentially never reaches exactly.</p>

<p>Lasso's penalty $\\lambda|w|$ is not differentiable exactly at $w=0$ — a sharp corner sits there, the same corner that produced the vertex in the geometric picture — but away from zero its derivative is $\\lambda\\,\\mathrm{sign}(w)$: plus $\\lambda$ for positive $w$, minus $\\lambda$ for negative $w$, a <i>constant</i> pull of fixed strength $\\lambda$ regardless of how close $w$ already is to zero. A constant pull that never weakens as $w\\to 0$ is exactly strong enough to stop $w$ dead once the loss's own gradient pulling it away from zero drops below $\\lambda$ in magnitude — at which point the two forces exactly cancel with $w$ sitting at zero, and it stays there.</p>

${H.key('Constant pull versus proportional pull: that is the whole difference between a penalty that eliminates a coefficient and one that merely tires it out.')}

<p>The mechanism becomes fully explicit in the coordinate-wise update lasso actually performs during fitting. Let $\\rho_j$ be what the least-squares fit would set coefficient $j$ to, ignoring the penalty for a moment. Lasso's update is the <b>soft-thresholding operator</b>,</p>

$$w_j \\leftarrow \\mathcal{S}_\\lambda(\\rho_j) = \\mathrm{sign}(\\rho_j)\\,\\max(0,\\, |\\rho_j| - \\lambda)$$

<p>Read this as: shrink the magnitude of $\\rho_j$ by $\\lambda$, but never let it cross through zero and come out the other side — if $\\lambda$ would overshoot, clamp at zero instead. Put numbers on it with $\\lambda=0.5$. A coefficient the unpenalised fit wanted at $\\rho=1.2$ comes out at $\\mathrm{sign}(1.2)\\times\\max(0, 1.2-0.5) = 0.7$: shrunk, but alive. A coefficient the unpenalised fit wanted at only $\\rho=0.3$ comes out at $\\max(0, 0.3-0.5)=\\max(0,-0.2)=0$: killed outright, because it was never strong enough to survive the constant pull. Ridge's proportional shrinkage on the same two numbers, $\\rho/(1+\\lambda)$ with $\\lambda=0.5$, gives $1.2/1.5=0.8$ and $0.3/1.5=0.2$ — both weakened, neither one reaching zero. Same $\\lambda$, same starting coefficients, fundamentally different arithmetic.</p>

<p><b>What you are looking at.</b> The dotted diagonal is the identity line, $w=\\rho$ — what would happen with no penalty at all. The solid blue curve is lasso's soft-thresholding update as a function of $\\rho$; the dashed red curve is ridge's proportional shrinkage on the same axis. The shaded band marks $|\\rho|<\\lambda$, the dead zone.</p>

<p><b>What to do with it.</b> Slide $\\lambda$ up from zero and watch the blue curve's flat segment through the origin widen — that flat segment <i>is</i> the set of unpenalised coefficients that lasso will zero out entirely at this strength. Watch the red curve at the same time: it rotates slightly toward the horizontal but never develops a flat segment of its own, at any $\\lambda$.</p>

<p><b>The thing genuinely worth noticing.</b> Outside the shaded band, the blue curve runs exactly parallel to the dotted identity line, offset downward by $\\lambda$ — a coefficient that survives lasso's threshold is not shrunk proportionally at all past that point, only by a fixed amount. Ridge does the opposite: every coefficient, large or small, is shrunk by the same <i>proportion</i>, never a flat amount. Large lasso survivors are barely touched; large ridge survivors lose a fixed percentage regardless of size. That is the full personality difference between the two penalties, visible in one picture.</p>

${H.lab('thresh', 'Soft threshold vs proportional shrinkage', 'The update rule each penalty implies, drawn. The flat segment in the middle of the L1 curve is the dead zone where coefficients are set to exactly zero — L2 has no such segment.')}

<h2><span class="sn">2.3.5</span> Elastic net, and practical guidance</h2>

<p>Lasso's sparsity has a cost the geometric picture does not immediately reveal: with several features that are strongly correlated with each other, lasso tends to pick one of them almost arbitrarily and zero out the rest, and which one it picks can flip between two runs on very slightly different data. That instability is exactly the situation ridge handles well — splitting credit across correlated columns rather than choosing a favourite — so <b>elastic net</b> simply combines the two penalties, $\\lambda_1\\|w\\|_1 + \\lambda_2\\|w\\|_2^2$, asking the L2 term to stabilise <i>which</i> members of a correlated group survive while the L1 term still decides <i>how many</i> features survive overall. It is the right default whenever you want both a short list and stability across correlated columns, which in practice is most of the time a short list is actually wanted.</p>

${H.table(['Situation', 'Choice', 'Why'], [
      ['Many correlated features, want stability', 'Ridge', 'Splits weight rather than choosing arbitrarily'],
      ['Need a short, defensible feature list', 'Lasso or elastic net', 'Exact zeros are a selection you can present'],
      ['Correlated groups AND sparsity', 'Elastic net', 'L1 selects, L2 stabilises which member is selected'],
      ['Trees / boosting', 'Neither, mostly', 'Use depth, min_child_weight, subsampling, and the λ/γ inside the split score (§2.8)'],
      ['Neural networks', 'Weight decay (AdamW), dropout, early stopping', 'Same idea, different implementation (§3.5)']
    ])}

<p>One habit sits above every row in that table. Suppose a true relationship is $\\text{score} = 0.00002\\times\\text{income} + 2.0\\times\\text{utilisation ratio}$, with income measured in pounds (typical value around 50{,}000) and utilisation as a 0-to-1 ratio (typical value around 0.5) — chosen so both terms contribute about equally to a typical prediction, $0.00002\\times50{,}000=1.0$ and $2.0\\times0.5=1.0$. An L2 penalty $\\lambda(w_1^2+w_2^2)$ sees $w_1^2 = (0.00002)^2 = 4\\times10^{-10}$, utterly negligible, against $w_2^2=(2.0)^2=4$ — so the penalty presses almost exclusively on the utilisation coefficient and leaves income's untouched, despite the two features mattering equally to the actual prediction. The regulariser has quietly become a function of which units you happened to record income in, rather than a function of how much each feature matters. <b>Standardising every feature to mean zero and unit variance before penalising</b> is what removes this artefact, by ensuring a one-unit change in every feature means the same thing: one standard deviation.</p>

${H.flag('Always standardise before penalising. A penalty on raw coefficients punishes features measured in small units and ignores features measured in large ones — the regulariser becomes a function of your unit choices.')}

${H.probe([
      ['Why does L1 select features and L2 not?', 'The diamond’s vertices sit on the axes, and L1’s constant subgradient pins coordinates at zero; L2’s pull vanishes at zero.'],
      ['When is elastic net the right default?', 'Correlated features plus a need for sparsity — lasso alone is unstable across correlated columns.']
    ], 'Claiming L2 gives sparsity, or that lasso is "just better". Lasso is unstable across correlated features; that is exactly what elastic net fixes.')}`,
    labs: {
      geom: function (host) {
        let opt = [1.6, .75];
        let drag = false;
        const st = Viz.controls(host, [
          { k: 'kind', label: 'penalty', type: 'buttons', value: 'l1', options: [{ v: 'l1', t: 'L1 — diamond' }, { v: 'l2', t: 'L2 — circle' }, { v: 'en', t: 'elastic net' }] },
          { k: 't', label: 'budget size', min: .2, max: 2, step: .02, value: .9, fmt: v => v.toFixed(2) },
          { k: 'rho', label: 'correlation of the loss contours', min: -.9, max: .9, step: .05, value: .5, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'w1', label: 'w₁ at optimum', cls: 'key' }, { k: 'w2', label: 'w₂ at optimum' }, { k: 'zero', label: 'exactly zero?' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-2.4, 2.4], yd: [-1.9, 1.9] }).frame({ xlabel: 'w₁', ylabel: 'w₂' });
            const A = [[1, st.rho], [st.rho, 1]];
            const loss = (a, b) => {
              const d = [a - opt[0], b - opt[1]];
              return d[0] * (A[0][0] * d[0] + A[0][1] * d[1]) + d[1] * (A[1][0] * d[0] + A[1][1] * d[1]);
            };
            const pen = (a, b) => st.kind === 'l1' ? Math.abs(a) + Math.abs(b)
              : st.kind === 'l2' ? Math.sqrt(a * a + b * b)
              : .5 * (Math.abs(a) + Math.abs(b)) + .5 * Math.sqrt(a * a + b * b);
            // constraint boundary
            const bd = [];
            for (let th = 0; th <= 6.3; th += .01) {
              const dx = Math.cos(th), dy = Math.sin(th);
              let lo = 0, hi = 6;
              for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; if (pen(dx * mid, dy * mid) > st.t) hi = mid; else lo = mid; }
              bd.push([dx * lo, dy * lo]);
            }
            // constrained optimum by scanning the boundary
            let best = bd[0], bl = 1e9;
            bd.forEach(p => { const v = loss(p[0], p[1]); if (v < bl) { bl = v; best = p; } });
            P.clip(() => {
              P.contours(loss, [bl, bl * 1.6, bl * 2.6, bl * 4, bl * 6, bl * 9], { color: T.red, alpha: .5 });
              P.contours(loss, [bl], { color: T.red, width: 2, alpha: .9 });
              P.line(bd.concat([bd[0]]), { color: T.blue, width: 2.4 });
              P.dots([opt], { r: 5, color: T.red, stroke: true });
              P.dots([best], { r: 6, color: T.green, stroke: true });
              P.line([[-2.4, 0], [2.4, 0]], { color: T.faint, width: 1, alpha: .6 });
              P.line([[0, -1.9], [0, 1.9]], { color: T.faint, width: 1, alpha: .6 });
            });
            P.text(opt[0], opt[1], '  unconstrained optimum', { color: T.red, font: '11px ui-sans-serif' });
            P.text(best[0], best[1], '  constrained solution', { color: T.green, font: '11px ui-sans-serif' });
            out({
              w1: best[0].toFixed(3), w2: best[1].toFixed(3),
              zero: (Math.abs(best[0]) < .02 || Math.abs(best[1]) < .02) ? 'yes — a vertex' : 'no'
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          if (e.type === 'down') drag = true;
          if (e.type === 'up') drag = false;
          if (drag && e.down) { opt = [P.ix(e.x), P.iy(e.y)]; S.redraw(); }
        });
        Viz.note(host, 'Drag the red dot around the plane on L1: the green solution snaps to an axis for most positions — that is feature selection happening geometrically. Switch to L2 and it essentially never does.');
      },

      thresh: function (host) {
        const st = Viz.controls(host, [
          { k: 'lam', label: 'λ', min: 0, max: 1.5, step: .02, value: .5, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3] })
              .frame({ xlabel: 'least-squares coefficient ρ (before penalty)', ylabel: 'coefficient after penalty' });
            P.clip(() => {
              P.fn(x => x, { color: T.faint, width: 1.2, dash: [4, 4] });
              P.fn(x => Math.sign(x) * Math.max(0, Math.abs(x) - st.lam), { color: T.blue, width: 2.8 });
              P.fn(x => x / (1 + st.lam), { color: T.red, width: 2.4, dash: [7, 4] });
              if (st.lam > 0) {
                ctx.globalAlpha = .12; ctx.fillStyle = T.blue;
                ctx.fillRect(P.x(-st.lam), P.pad.t, P.x(st.lam) - P.x(-st.lam), P.ph);
                ctx.globalAlpha = 1;
              }
            });
            ctx.fillStyle = T.blue; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('L1 soft threshold — flat dead zone → exact zeros', 60, 14);
            ctx.fillStyle = T.red; ctx.fillText('L2 proportional shrinkage — never reaches zero', 60, 30);
          }
        });
      }
    },
    quiz: [
      {
        q: 'Which statement about lasso is accurate?',
        options: ['It always outperforms ridge', 'Its constant subgradient can pin coefficients at exactly zero; it is unstable across correlated features', 'It is equivalent to ridge with a different λ', 'It requires standardised targets, not features'],
        answer: 1,
        why: 'Lasso\'s penalty has a constant subgradient at zero, $\\lambda\\,\\mathrm{sign}(w)$, which does not shrink away as a coefficient approaches zero, so it can pin a coefficient at exactly zero rather than merely nudging it toward zero — and among a group of correlated features it tends to pick one somewhat arbitrarily and zero out the rest, an instability elastic net exists specifically to repair. "It always outperforms ridge" is the tempting overreach: sparsity is an advantage only when the true model actually is sparse, and with many weakly-informative correlated features ridge\'s smooth shrinkage is frequently the stronger choice. "Equivalent to ridge with a different λ" misses that the two penalties have different geometry — L1\'s constraint region has corners on the axes, L2\'s is a smooth ball — and that geometry, not the size of λ, is what produces exact zeros in one but never the other. "Requires standardised targets, not features" inverts the actual requirement, which is exactly the failure mode the next question is built around: it is the features that must be standardised before either penalty is applied, and the target\'s scale plays no comparable role.'
      },
      {
        q: 'You forget to standardise features before applying an L2 penalty. What goes wrong?',
        options: ['Nothing', 'The penalty depends on the units of each feature, punishing small-unit features arbitrarily', 'The model becomes non-convex', 'Coefficients become exactly zero'],
        answer: 1,
        why: 'An L2 penalty punishes the raw squared size of a coefficient with no awareness of what units its feature happens to be measured in, so a feature recorded in millions gets a naturally tiny coefficient the penalty barely notices, while a feature recorded as a small ratio needs a much larger coefficient to carry the same real-world effect and is penalised far more heavily for it — the penalty ends up reflecting unit choices, not genuine importance. "Nothing" is tempting if you reason about regularisation only in the abstract, where the maths looks unit-agnostic on paper; the moment real features with mismatched units are involved, the asymmetry is immediate and large. "The model becomes non-convex" is simply false — adding an L2 penalty to a convex loss keeps the sum convex regardless of scaling, so this option tests whether "behaves badly" gets confused with "loses convexity," which are unrelated properties. "Coefficients become exactly zero" describes lasso\'s signature behaviour, not ridge\'s, and mixing up the two penalties\' effects is exactly the confusion the side-by-side comparison in §2.3 is built to prevent — the fix here, standardising every feature before fitting, should be a default step, not an afterthought.'
      }
    ],
    cards: [
      { q: 'Why is L1 sparse? (analytic)', a: 'Its subgradient is a constant $\\lambda\\,\\mathrm{sign}(w)$ that does not vanish at zero, so it pins coordinates there. L2’s $2\\lambda w$ vanishes.' },
      { q: 'Soft-thresholding operator', a: '$\\mathcal{S}_\\lambda(\\rho)=\\mathrm{sign}(\\rho)\\max(0,|\\rho|-\\lambda)$ — the lasso coordinate update.' },
      { q: 'When elastic net', a: 'Correlated features plus a need for sparsity; L1 alone picks one of a correlated group arbitrarily.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.4 */
  ML.section({
    id: 'linear-logistic', track: 'classical', num: '2.4',
    title: 'Linear regression, logistic regression, GLMs',
    lede: 'The backbone models: they feed calibration (§2.12), scorecards (§2.11), and the whole idea of a link function. Both derivations are two lines and both get asked.',
    html: `
<p>Two models anchor the whole of applied statistics before anyone mentions a neural network, and this section is both of them. One predicts a number. The other predicts a probability. They turn out to be the same idea wearing two different outfits, and seeing why is worth more than memorising either one separately.</p>

<h2><span class="sn">2.4.1</span> Linear regression: the result you already derived</h2>

<p>§0.7 built this from nothing and there is no reason to build it twice. Minimising $\\|y-Xw\\|^2$ led, in five lines you can go back and reread, to the normal equations $X^\\mathsf{T}Xw=X^\\mathsf{T}y$ and the closed form $w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; adding $\\lambda\\|w\\|^2$ to the objective added $\\lambda I$ inside the inverse, and §0.7 worked an exact numeric case — two collinear columns, eigenvalues 28 and 0, repaired to 28.1 and 0.1 — showing why that single term fixes a numerical problem and a statistical one simultaneously. §2.3 has since built the full geometric and analytic case for why that same $\\lambda$, applied as an L1 penalty instead, behaves completely differently. If squared error feels like an arbitrary choice of loss rather than a natural one, it is worth knowing it is not arbitrary at all: §1.5 shows that assuming Gaussian noise on the labels and maximising the likelihood <i>produces</i> squared error as the resulting negative log-likelihood, up to a constant. Least squares was never an assumption bolted onto probability; it is what probability demands once you commit to a Gaussian noise model.</p>

<p>What linear regression cannot do is predict a probability, and it is worth seeing concretely why not, because the failure is what motivates everything else in this section.</p>

<h2><span class="sn">2.4.2</span> Why you cannot just fit a line to 0 and 1</h2>

<p>Suppose you are predicting whether a loan applicant defaults, coded $y=1$ for default and $y=0$ for repaid, from their count of missed payments in the last year. Nothing stops you from running ordinary least squares on this data exactly as you would on any numeric target — the arithmetic does not know the label happens to only take two values. Suppose the fit comes out as $\\hat y = 0.05 + 0.12\\times(\\text{missed payments})$. For an applicant with zero missed payments this reads $\\hat y=0.05$, which passes as a small probability of default. For an applicant with ten missed payments it reads $\\hat y = 0.05+0.12\\times10 = 1.25$ — a "125 per cent probability of default," a number the model has no business producing and that nothing downstream can sensibly use.</p>

<p>This is not a fixable quirk of one bad fit. A straight line, unless it happens to be perfectly flat, runs from $-\\infty$ to $+\\infty$ as $x$ runs from $-\\infty$ to $+\\infty$ — that is what makes it a line — while a probability is only ever allowed to live in $[0,1]$. Asking a straight line to double as a probability is asking an unbounded object to behave like a bounded one, and it will eventually refuse, for large enough or small enough inputs, every single time. What is needed is a function that takes the linear predictor's entire unbounded range and squashes it into $(0,1)$ without ever quite touching either end.</p>

<h2><span class="sn">2.4.3</span> Building the sigmoid from the log-odds, rather than presenting it</h2>

<p>There is a natural quantity that already lives on the whole real line while still being built entirely out of a probability: the <b>log-odds</b>. The <b>odds</b> of an event with probability $p$ are $p/(1-p)$ — how many times more likely it is to happen than not. As $p$ ranges over $(0,1)$, the odds range over the whole of $(0,\\infty)$: odds near 0 as $p\\to0$, odds racing to infinity as $p\\to1$. Take a logarithm of the odds and the range stretches once more, from $(0,\\infty)$ to the whole real line $(-\\infty,\\infty)$ — because $\\ln$ sends numbers near zero to $-\\infty$ and large numbers to $+\\infty$.</p>

<p>That is exactly the range a linear predictor $w^\\mathsf{T}x$ already has. So instead of asking the linear predictor to <i>be</i> a probability — the mistake in §2.4.2 — ask it to be the log-odds, $\\ln\\!\\big(\\tfrac{p}{1-p}\\big) = w^\\mathsf{T}x$, and solve backwards for the probability that implies:</p>

${H.deriv('inverting the log-odds equation to recover a probability', [
      ['$\\ln\\!\\Big(\\dfrac{p}{1-p}\\Big) = z$', 'Definition: the linear predictor $z=w^\\mathsf{T}x$ is set equal to the log-odds of the event, by modelling choice rather than necessity.'],
      ['$\\dfrac{p}{1-p} = e^{z}$', 'Exponentiate both sides — the inverse operation to $\\ln$ (§0.3).'],
      ['$p = e^{z}(1-p) = e^{z} - e^{z}p$', 'Multiply through by $(1-p)$ to clear the fraction, then distribute.'],
      ['$p + e^{z}p = e^{z} \\ \\Rightarrow\\ p(1+e^{z}) = e^{z}$', 'Collect every term containing $p$ on the left.'],
      ['$p = \\dfrac{e^{z}}{1+e^{z}} = \\dfrac{1}{1+e^{-z}} =: \\sigma(z)$', 'Divide through by $1+e^z$, then multiply top and bottom by $e^{-z}$ to reach the more common form. This function has a name, the <b>sigmoid</b> or <b>logistic function</b>, and it was not chosen for having a nice S-shape — the S-shape is a consequence of insisting the log-odds be linear.']
    ], 'Try three values to see the shape: $\\sigma(0)=1/(1+1)=0.5$, the fifty-fifty point where log-odds are exactly zero. $\\sigma(1.2)\\approx0.769$: a linear predictor of $+1.2$ log-odds becomes a 76.9% probability. $\\sigma(-1.2)\\approx0.231$, its mirror image below 0.5 — $\\sigma$ is symmetric around $(0,\\,0.5)$ because flipping the sign of the log-odds exactly inverts which outcome is favoured.')}

<p>Logistic regression, in full, is: fit a linear predictor $w^\\mathsf{T}x$ exactly as in §2.4.1, then pass it through $\\sigma$ before reading it as a probability. Everything about the "linear" half of the name is still true — decision boundaries are still hyperplanes, since $\\sigma(w^\\mathsf{T}x)=0.5$ exactly when $w^\\mathsf{T}x=0$ — the only thing that changed is what the linear part is allowed to mean.</p>

<h2><span class="sn">2.4.4</span> The loss, and the gradient, both derived rather than dropped in</h2>

<p>§1.5 already established the general recipe: pick a noise model for how labels are generated, take the negative log-likelihood, and that negative log-likelihood <i>is</i> your loss. For a label that is either 0 or 1, the natural noise model is a Bernoulli distribution with success probability $p$, and §1.5's table already gives the resulting negative log-likelihood as $-\\sum_i[y_i\\log p_i+(1-y_i)\\log(1-p_i)]$ — the <b>cross-entropy</b> or <b>logloss</b>. This is not a heuristic penalty someone invented because it looked reasonable. It is the exact quantity maximum likelihood hands you once you commit to "each label is a coin flip with probability $p$," the same way squared error fell out of committing to Gaussian noise in §2.4.1.</p>

<p>What §1.5 does not do is differentiate it, because at that point in the course the sigmoid itself had not been built yet. Do that now. The chain rule needs the derivative of $\\sigma$ with respect to its own argument, which is short enough to derive once and reuse forever:</p>

${H.deriv('the sigmoid\'s own derivative, in three lines', [
      ['$\\sigma(z)=\\dfrac{1}{1+e^{-z}} = (1+e^{-z})^{-1}$', 'Rewrite as a power, to apply the chain rule for a negative exponent (§0.3).'],
      ['$\\sigma\'(z) = -(1+e^{-z})^{-2}\\times(-e^{-z}) = \\dfrac{e^{-z}}{(1+e^{-z})^2}$', 'Differentiate: bring the exponent down, subtract one from it, then multiply by the derivative of the inner function $1+e^{-z}$, which is $-e^{-z}$; the two minus signs cancel.'],
      ['$= \\dfrac{1}{1+e^{-z}}\\times\\dfrac{e^{-z}}{1+e^{-z}} = \\sigma(z)\\times\\Big(1-\\dfrac{1}{1+e^{-z}}\\Big)$', 'Split the fraction into two copies of $\\sigma(z)$-shaped pieces, then notice $\\dfrac{e^{-z}}{1+e^{-z}} = \\dfrac{(1+e^{-z})-1}{1+e^{-z}} = 1-\\sigma(z)$.'],
      ['$\\sigma\'(z) = \\sigma(z)\\big(1-\\sigma(z)\\big)$', 'A function whose derivative is expressible purely in terms of itself — no $z$ appears on the right at all. This is unusually convenient and it is the reason the final gradient below collapses so cleanly.']
    ])}

<p>Now assemble the full gradient of the loss for one training example, with $p=\\sigma(w^\\mathsf{T}x)$:</p>

${H.deriv('the logistic gradient, assembled from the two pieces above', [
      ['$\\mathcal{L} = -\\big[y\\log p + (1-y)\\log(1-p)\\big]$', 'The per-example cross-entropy from §1.5, restated with $p$ as shorthand for $\\sigma(w^\\mathsf{T}x)$.'],
      ['$\\dfrac{\\partial \\mathcal{L}}{\\partial p} = -\\dfrac{y}{p} + \\dfrac{1-y}{1-p}$', 'Differentiate with respect to $p$ directly, using $\\frac{d}{dp}\\log p = 1/p$ (§0.3) on each term.'],
      ['$\\dfrac{\\partial p}{\\partial w} = \\sigma\'(w^\\mathsf{T}x)\\,x = p(1-p)\\,x$', 'Chain rule through $z=w^\\mathsf{T}x$: the derivative of $p=\\sigma(z)$ with respect to $w$ is $\\sigma\'(z)$ times the derivative of $z$ with respect to $w$, which is simply $x$ (§0.7). Substitute the result derived just above.'],
      ['$\\dfrac{\\partial \\mathcal{L}}{\\partial w} = \\Big(-\\dfrac{y}{p}+\\dfrac{1-y}{1-p}\\Big)p(1-p)\\,x$', 'Multiply the two pieces together — the chain rule proper, $\\partial\\mathcal L/\\partial w = (\\partial\\mathcal L/\\partial p)(\\partial p/\\partial w)$.'],
      ['$= \\big[-y(1-p) + (1-y)p\\big]x = (p - y)\\,x$', 'Distribute $p(1-p)$ into the bracket: $-y(1-p)+(1-y)p = -y+yp+p-yp = p-y$. Every $p(1-p)$ factor cancels completely against the derivative of the log terms, which is exactly why the sigmoid and the cross-entropy were paired together rather than either being used alone.']
    ], 'Put a number on it: at $w^\\mathsf{T}x=1.2$, $p\\approx0.769$. Against a true label $y=1$ the gradient is $(0.769-1)x=-0.231x$ — small, because the model was already fairly confident and right. Against $y=0$ it is $(0.769-0)x=0.769x$ — large, because the model was confident and <i>wrong</i>, and the update says so in proportion to just how wrong.')}

${H.key('The update is prediction minus label, times the feature — structurally identical to linear regression\'s residual update, and the same form reappears inside softmax cross-entropy for more than two classes (§4.9).')}

<p>The objective this gradient descends is convex (§1.9), which is worth one sentence of relief: there is exactly one minimum, no ridges or false valleys to get trapped in, so any reasonable optimiser started anywhere eventually finds the same answer.</p>

<p><b>What you are looking at.</b> A scatter of two-class points on a plane with two features, $x_1$ and $x_2$. The coloured field is the model's current prediction $\\sigma(w^\\mathsf{T}x)$ at every point in the plane, and the solid line running through it is the decision boundary, $\\sigma(w^\\mathsf{T}x)=0.5$. The small inset panel tracks the logloss across every gradient step taken so far.</p>

<p><b>What to do with it.</b> Click directly on the plot to drop new points of whichever class is selected, then press <i>Train 200 steps</i> and watch the boundary rotate and slide until it separates the two colours as well as it can, while the inset loss curve falls. Switch the dataset to <i>circles</i>, where one class forms a ring around the other, and train again.</p>

<p><b>The thing genuinely worth noticing.</b> On the circles dataset the boundary never manages to separate the classes, no matter how long you train or how low the L2 slider goes, because a straight line simply cannot separate a ring from its centre — logistic regression is linear in whatever features you hand it, full stop. That specific, un-fixable failure is the entire motivation for the kernel trick two sections from now (§2.6) and for the hidden layers of §3.1: both exist purely to manufacture new features in which the same straight-line boundary becomes sufficient again.</p>

${H.lab('logistic', 'Logistic regression, trained live', 'Real gradient descent on data you can edit. Click to add points, watch the boundary move, the loss fall, and the weights converge. Turn on L2 and watch the boundary stiffen.')}

<h2><span class="sn">2.4.5</span> GLMs: the same recipe, a different noise model</h2>

<p>Nothing about the argument in §2.4.3 was specific to Bernoulli labels. The general pattern is: pick a distribution for $y$ from the exponential family, and pick a <b>link function</b> connecting the linear predictor to the mean of that distribution in whatever way keeps the mean inside its valid range — exactly the job the log-odds transform did above. Logistic regression is the case of a Bernoulli label with the logit link; Poisson regression, used for modelling counts that must stay non-negative, pairs a Poisson label with a log link, $\\ln(\\mathbb{E}[y])=w^\\mathsf{T}x$, for the identical reason a count cannot go negative any more than a probability can leave $[0,1]$; ordinary linear regression is the case of a Gaussian label with the identity link, because a Gaussian mean needs no transform to range over the whole real line to begin with. One recipe, three familiar models, differing only in which range needs protecting and how.</p>

<p>The link also fixes how you are allowed to read a coefficient. In logistic regression $w^\\mathsf{T}x$ <i>is</i> the log-odds, so increasing feature $j$ by one unit adds $w_j$ to the log-odds, which means it <i>multiplies</i> the odds by $e^{w_j}$ — the <b>odds ratio</b>. That single sentence is worth memorising verbatim, because it is exactly the sentence a scorecard is built on.</p>

${H.worked('worked number — from log-odds to scorecard points', `
<p>A credit score is an affine map of the log-odds, calibrated by two conventions: an <b>anchor</b> (a chosen score at chosen odds) and the <b>PDO</b> — points to double the odds.</p>
$$\\text{factor}=\\frac{\\text{PDO}}{\\ln 2},\\qquad \\text{offset}=\\text{anchor}-\\text{factor}\\cdot\\ln(\\text{anchor odds})$$
<p>With PDO = 20 and 600 points at 50:1 good:bad odds: factor = 20 ÷ 0.693 = <b>28.85</b>; offset = 600 − 28.85 × ln(50) = 600 − 28.85 × 3.912 = <b>487.1</b>.</p>
<p>An applicant whose model log-odds are 2.5 scores 487.1 + 28.85 × 2.5 = <b>559</b>. One with log-odds 4.2 scores 608. Sanity check that proves you understand it: 20 more points must double the odds — 559 → 579 moves log-odds from 2.5 to 3.19, and $e^{0.693} = 2$. ✓</p>
<p>Now read individual coefficients the way the odds-ratio sentence above says to. A WOE-binned "utilisation under 30 per cent" indicator with coefficient 0.916 means: having that property multiplies the odds of being a good account by $e^{0.916}\\approx 2.50$ — two and a half times more likely good-versus-bad than otherwise identical applicants without it. "Tenure five years or more" at coefficient 0.42 gives $e^{0.42}\\approx1.52$, roughly a 52% uplift in odds. "Recent arrears" at $-0.83$ gives $e^{-0.83}\\approx0.44$: odds of being good <i>fall</i> to 44% of what they were, because a negative log-odds contribution multiplies the odds by a number below 1.</p>
<p>Because the map is affine and WOE-binned features enter linearly (§2.11), each bin's contribution can be printed as whole points — which is the entire reason regulated lending still ships logistic scorecards when gradient boosting scores better: <mark>the model is auditable by arithmetic, not by a SHAP library.</mark></p>`)}

<p><b>What you are looking at.</b> The blue curve is the affine score-versus-log-odds mapping built from the current PDO, anchor score and anchor odds. The red dot marks one applicant at the chosen log-odds; the green segment shows what happens to their score if their log-odds increase by exactly $\\ln 2$. Beneath the chart, a committee-style breakdown adds up whole-point contributions from four WOE-binned features plus a base offset to reach a total score.</p>

<p><b>What to do with it.</b> Drag the PDO slider up and down and watch the blue line's slope — the <i>factor</i> — change with it: a larger PDO means more points have to be spent to double the odds, so the same log-odds range gets mapped to a gentler score range. Then move the anchor score and watch the whole line shift vertically without changing its slope, because the anchor sets where the line sits, not how steeply it climbs.</p>

<p><b>The thing genuinely worth noticing.</b> Move the applicant's log-odds slider and watch the green segment redraw at the new position — it always spans exactly the PDO value on the score axis, regardless of where on the curve it is drawn, because the mapping is affine and an affine map has the same slope everywhere. That constancy is the entire promise a scorecard is making to the committee reading it: 20 points means the same thing — one doubling of the odds — whether the applicant started at 480 or 680.</p>

${H.lab('scorecard', 'Scorecard points calculator', 'Move PDO and the anchor and watch the mapping change. The bottom strip is a real applicant’s points breakdown: each bin contributes whole points that a committee can add up by hand.')}

${H.probe([
      ['Logistic gradient?', '$(\\hat p - y)x$. Derive it in two lines using $\\sigma\' = \\sigma(1-\\sigma)$.'],
      ['Why not fit a line to probabilities?', 'Unbounded outputs and the wrong noise model; the logit link and Bernoulli likelihood fix both.'],
      ['What does ridge do to $X^\\mathsf{T}X$?', 'Adds λ to every eigenvalue, guaranteeing invertibility and shrinking low-variance directions hardest.']
    ])}`,
    labs: {
      logistic: function (host) {
        let data = Num.dataset('blobs', 60, .35, 12);
        let model = null, running = false, hist = [];
        const st = Viz.controls(host, [
          { k: 'lr', label: 'learning rate η', min: .05, max: 3, step: .05, value: .8, fmt: v => v.toFixed(2) },
          { k: 'l2', label: 'L2 strength', min: 0, max: 1, step: .01, value: 0, fmt: v => v.toFixed(2) },
          { k: 'cls', label: 'class to add on click', type: 'buttons', value: '1', options: [{ v: '0', t: 'class 0' }, { v: '1', t: 'class 1' }] }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'loss', label: 'logloss', cls: 'key' }, { k: 'acc', label: 'train accuracy' },
          { k: 'w', label: 'weights' }, { k: 'it', label: 'iterations' }
        ]);
        function reset() {
          model = Num.logistic(data.X, data.y, { lr: st.lr, l2: st.l2 });
          hist = []; S.redraw();
        }
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2, 3.2] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            if (model) {
              P.clip(() => Labs.boundary(P, (x, y) => model.predict([x, y]), { step: 4 }));
            }
            P.clip(() => Labs.points(P, data.X, data.y));
            if (hist.length > 2) {
              // inset loss curve
              const ix = P.pad.l + 8, iy = P.pad.t + 8, iw = Math.min(150, P.pw * .34), ih = 52;
              ctx.fillStyle = T.paper; ctx.globalAlpha = .85; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
              ctx.strokeStyle = T.line; ctx.strokeRect(ix, iy, iw, ih);
              const mx = Math.max.apply(null, hist);
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6; ctx.beginPath();
              hist.forEach((v, i) => {
                const X = ix + iw * i / (hist.length - 1), Y = iy + ih - ih * (v / (mx || 1));
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              });
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('logloss', ix + 4, iy + 3);
            }
            if (model) {
              const acc = Num.mean(data.X.map((x, i) => (model.predict(x) > .5 ? 1 : 0) === data.y[i] ? 1 : 0));
              out({
                loss: model.loss().toFixed(4), acc: (acc * 100).toFixed(1) + '%',
                w: '[' + model.w.map(v => v.toFixed(2)).join(', ') + ']', it: hist.length
              });
            }
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P || e.type !== 'down') return;
          data.X.push([P.ix(e.x), P.iy(e.y)]); data.y.push(+st.cls);
          reset();
        });
        Viz.buttons(host, [
          { label: 'Train 200 steps', primary: true, on: () => { for (let i = 0; i < 200; i++) { model.step(1); hist.push(model.loss()); } S.redraw(); } },
          { label: 'Animate', on: () => {
            running = !running;
            const t = setInterval(() => { if (!running) return clearInterval(t); model.step(3); hist.push(model.loss()); S.redraw(); }, 40);
            ML.onCleanup(() => clearInterval(t));
          } },
          { label: 'Reset weights', on: reset },
          { label: 'New data: circles', on: () => { data = Num.dataset('circles', 80, .25, 4); reset(); } },
          { label: 'New data: blobs', on: () => { data = Num.dataset('blobs', 60, .35, 12); reset(); } }
        ]);
        reset();
        Viz.note(host, 'On the circles dataset the boundary can never fit — logistic regression is linear in the features you give it. That failure is the motivation for kernels (§2.6) and for hidden layers (§3.1): both manufacture the features that make it linear again.');
      },

      scorecard: function (host) {
        const st = Viz.controls(host, [
          { k: 'pdo', label: 'PDO — points to double the odds', min: 10, max: 60, step: 1, value: 20, fmt: v => v },
          { k: 'anchor', label: 'anchor score', min: 400, max: 800, step: 5, value: 600, fmt: v => v },
          { k: 'odds', label: 'anchor odds (good:bad)', min: 5, max: 100, step: 1, value: 50, fmt: v => v + ':1' },
          { k: 'lo', label: 'applicant log-odds', min: -2, max: 8, step: .1, value: 2.5, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'factor', label: 'factor', cls: 'key' }, { k: 'offset', label: 'offset' },
          { k: 'score', label: 'applicant score' }, { k: 'check', label: '+PDO doubles odds?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const factor = st.pdo / Math.log(2);
            const offset = st.anchor - factor * Math.log(st.odds);
            const score = lo => offset + factor * lo;
            const P = Viz.plot(ctx, w, h, { xd: [-2, 8], yd: [Math.min(300, score(-2)) - 20, score(8) + 20] })
              .frame({ xlabel: 'model log-odds', ylabel: 'score', yfmt: v => v.toFixed(0) });
            P.clip(() => {
              P.fn(score, { color: T.blue, width: 2.6 });
              P.vline(st.lo, { color: T.text, dash: [4, 4] });
              P.dots([[st.lo, score(st.lo)]], { r: 6, color: T.red, stroke: true });
              P.line([[st.lo, score(st.lo)], [st.lo + Math.log(2), score(st.lo + Math.log(2))]], { color: T.green, width: 3 });
              P.text(st.lo + Math.log(2), score(st.lo + Math.log(2)), '  +' + st.pdo + ' pts = 2× the odds', { color: T.green, font: '11px ui-sans-serif' });
            });
            // points breakdown strip
            const bins = [['utilisation < 0.3', 0.916], ['tenure ≥ 5 yrs', 0.42], ['enquiries ≤ 2', 0.31], ['recent arrears', -0.83]];
            const bx = P.pad.l + 10, by = h - 96;
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('a committee-readable breakdown (WOE bins × factor):', bx, by - 12);
            let total = offset;
            bins.forEach((b, i) => {
              const pts = b[1] * factor;
              total += pts;
              ctx.fillStyle = T.text; ctx.fillText(b[0], bx, by + 12 + i * 17);
              ctx.fillStyle = pts >= 0 ? T.green : T.red; ctx.textAlign = 'right';
              ctx.fillText((pts >= 0 ? '+' : '') + pts.toFixed(0), bx + 250, by + 12 + i * 17);
              ctx.textAlign = 'left';
            });
            ctx.fillStyle = T.muted; ctx.fillText('base (offset)', bx, by + 12 + bins.length * 17);
            ctx.textAlign = 'right'; ctx.fillText(offset.toFixed(0), bx + 250, by + 12 + bins.length * 17);
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace';
            ctx.textAlign = 'left'; ctx.fillText('total', bx, by + 14 + (bins.length + 1) * 17);
            ctx.textAlign = 'right'; ctx.fillText(total.toFixed(0), bx + 250, by + 14 + (bins.length + 1) * 17);
            out({
              factor: factor.toFixed(2), offset: offset.toFixed(1), score: score(st.lo).toFixed(0),
              check: (score(st.lo + Math.log(2)) - score(st.lo)).toFixed(1) + ' pts ✓'
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'The logistic regression gradient for one example is…',
        options: ['$(y-\\hat p)x^2$', '$(\\hat p - y)x$', '$\\hat p(1-\\hat p)x$', '$-y\\log\\hat p$'],
        answer: 1,
        why: 'The logistic loss\'s gradient for one example works out to $(\\hat p-y)x$ — the model\'s error scaling the feature vector — because the $\\sigma\'=\\sigma(1-\\sigma)$ term produced by differentiating the sigmoid cancels exactly against a matching term from differentiating the log-loss itself, a cancellation worth deriving once (§2.4) rather than taking on faith. $(y-\\hat p)x^2$ is the tempting near-miss: it flips the sign and squares the feature rather than the error, most likely from confusing this gradient with the superficially similar squared-error gradient of linear regression. $\\hat p(1-\\hat p)x$ isolates only the sigmoid-derivative piece and forgets that it cancels against the loss\'s own derivative, reporting an intermediate step as though it were the final answer. $-y\\log\\hat p$ is not a gradient at all — it is one term of the loss function itself, before any differentiation has happened, a different confusion entirely: mistaking what you are minimising for the direction you move to minimise it.'
      },
      {
        q: 'Ridge adds λI inside the inverse. Besides regularising, what does that achieve?',
        options: ['It centres the features', 'It guarantees invertibility of $X^\\mathsf{T}X$ under collinearity or p > n', 'It makes the problem non-convex', 'It removes the intercept'],
        answer: 1,
        why: 'Adding $\\lambda I$ raises every eigenvalue of $X^\\mathsf{T}X$ by exactly $\\lambda$, so even a matrix that started out singular or nearly singular — from collinear features, or from more features than rows — becomes strictly invertible the moment $\\lambda>0$, meaning ridge solves a genuine numerical problem at the same time as it regularises. "It centres the features" is a plausible-sounding but unrelated claim: centring is a separate preprocessing step, and adding a constant to the diagonal of a matrix does nothing to any column\'s mean. "It makes the problem non-convex" gets the direction backwards — ridge\'s penalty is itself convex, so adding it to an already-convex loss keeps the sum convex, with a unique minimum now guaranteed by the newly-invertible matrix. "It removes the intercept" confuses ridge with the separate, common practice of excluding the intercept from the penalty, which is a design choice made alongside ridge, not a consequence of the $\\lambda I$ term. The two benefits — numerical stability and statistical regularisation — are genuinely the same term doing two jobs at once, which is exactly why the question asks for something "besides" regularising.'
      },
      {
        q: 'A scorecard uses PDO = 20 anchored at 600 = 50:1 odds. An applicant scores 620. Their odds are…',
        options: ['25:1', '50:1', '100:1', '200:1'],
        answer: 2,
        why: 'A PDO (points to double the odds) of 20 means, by construction, that every 20-point rise in score exactly doubles the odds, so moving from 600 to 620 is one full doubling: $50{:}1$ becomes $100{:}1$ by multiplying, not adding. $25{:}1$ is the tempting answer if you halve the odds for a 20-point rise instead of doubling them, which is the wrong direction — a higher score means lower risk and therefore better, higher odds of being good. $50{:}1$ mistakenly treats the 20-point move as having no effect at all, perhaps from misreading the anchor point as the answer rather than as the starting condition the question moves away from. $200{:}1$ overshoots by treating the move as two doublings instead of one, an easy slip if 620 is misread as 40 points above the anchor rather than 20. The general mechanism worth keeping is that a scorecard\'s point scale is logarithmic in odds by design, like a decibel or Richter scale, which is exactly why a fixed number of points always means a fixed multiplicative — never additive — change in odds, wherever on the scale you stand.'
      }
    ],
    cards: [
      { q: 'Normal equations', a: '$w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge: $(X^\\mathsf{T}X+\\lambda I)^{-1}X^\\mathsf{T}y$.' },
      { q: 'Logistic gradient', a: '$(\\hat p-y)x$ — prediction minus label times feature.' },
      { q: 'Scorecard conversion', a: 'factor = PDO/ln2, offset = anchor − factor·ln(anchor odds); score = offset + factor × log-odds.' },
      { q: 'What is $e^{w_j}$ in logistic regression?', a: 'The odds ratio for a one-unit change in feature $j$.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.5 */
  ML.section({
    id: 'knn-nb', track: 'classical', num: '2.5',
    title: 'k-NN and Naive Bayes',
    lede: 'Naive Bayes rests directly on §1.1; k-NN is the cleanest possible illustration of the §2.2 tradeoff — one hyperparameter moves you from pure variance to pure bias.',
    html: `
<p>Every other model in Part 2 works the same way: look at the training data once, boil it down into a compact set of parameters — a weight vector, a tree of splits — and then throw the training data away. From then on, prediction consults only the parameters. This section is about two models that refuse to do that. One keeps the entire training set forever and answers every question by looking the data up again. The other keeps almost nothing, one number per feature per class, and gets away with an assumption that is nearly always technically false. They could not be more different in spirit, and putting them side by side is instructive precisely because of that contrast.</p>

<h2><span class="sn">2.5.1</span> k-NN: prediction by looking up similar cases</h2>

<p>Imagine judging a job candidate not with a formula but by memory: think of the five most similar candidates you have hired before — similar background, similar interview performance — and predict this one will do roughly what they did. That is the entire idea of $k$-nearest neighbours, stated before any notation. Store every training example. To predict on a new point, measure its <b>distance</b> to every stored example — exactly the distance built from the norm in §0.2, $\\|a-b\\|=\\sqrt{\\sum_i(a_i-b_i)^2}$ — find the $k$ closest, and predict by majority vote among them for classification, or by their average for regression. There is no fitting step, no loss to minimise, no gradient to descend. The entire "model" is the training set itself, plus a rule for how to consult it.</p>

${H.analogy(`<p>A librarian with no formula for "what will this reader enjoy" but a very good memory for who liked what. Ask for a recommendation, and she thinks of the handful of past readers whose taste most resembled yours, and suggests whatever they enjoyed. If she only consults the single most similar reader ever recorded, one eccentric match can send you home with a strange book. If she consults the five hundred vaguely-similar readers on file, the recommendation becomes generic — safe, popular, and barely personalised to you at all. Somewhere between one and five hundred is a number of past readers narrow enough to reflect your actual taste and broad enough not to be thrown by a single outlier. That number is $k$.</p>`)}

<p>That trade is not a vague intuition; it is §2.2's bias–variance decomposition wearing a different name, and $k$-NN is the cleanest place on this site to see it, because a single knob moves you from one extreme to the other with nothing else changing. At $k=1$, every prediction is decided by exactly one training point — the closest one — so the decision boundary bends around every individual example, including every mislabelled or noisy one, producing a jagged, island-riddled boundary and <i>exactly</i> zero error on the training set, because every training point is trivially its own nearest neighbour. That is variance in its purest form: swap in a slightly different training set and the boundary, built entirely from single nearby points, changes shape everywhere. Push $k$ up toward the size of the whole dataset and the vote is taken over an ever-larger, ever more representative crowd; in the limit $k=n$, every prediction is just the overall majority class, identical everywhere on the plane regardless of which point you are asking about. That is bias in its purest form: a boundary so smooth it has stopped reacting to the input at all.</p>

<p><b>What you are looking at.</b> A scatter of labelled points in two dimensions and, layered behind them, the model's prediction at every point on the plane, coloured by class. The decision boundary is wherever that colouring flips.</p>

<p><b>What to do with it.</b> Set $k=1$ and look at the boundary: small islands of one colour wrap tightly around individual, sometimes clearly mislabelled-looking points, and the readout confirms 100% training accuracy sitting alongside a much lower leave-one-out accuracy — the textbook signature of a model that has memorised rather than learned. Now drag $k$ up toward 25 and watch the islands dissolve into one broad, smooth sweep, with training accuracy falling as leave-one-out accuracy typically improves.</p>

<p><b>The thing genuinely worth noticing.</b> Leave $k$ fixed at some middle value and instead drag the <i>scale feature 2</i> slider. Nothing about the underlying data changes — the points are the same points — but the boundary reshapes itself completely, because stretching one axis changes what "nearest" means without changing a single label. Distance takes your units at face value exactly as §0.2's pitfall box warned, and $k$-NN has no other information to fall back on: get the scaling wrong and the model is answering a different, unintended question, however well-tuned $k$ is.</p>

${H.lab('knn', 'k = 1 versus k = 25, on the same data', 'One hyperparameter, two completely different models. Watch the boundary go from jagged islands around individual points to a single smooth sweep — the left is variance, the right is bias, and §2.2 is that picture.')}

<h2><span class="sn">2.5.2</span> The curse of dimensionality</h2>

<p>There is a second, stranger way $k$-NN can fail, and it has nothing to do with $k$ at all. It shows up only as the number of features grows, and it is worth building up rather than asserting, because the conclusion is genuinely counter-intuitive: <i>in high enough dimensions, "nearest neighbour" stops meaning anything.</i></p>

<p>A squared distance between two random points, $\\|p-q\\|^2=\\sum_{i=1}^d (p_i-q_i)^2$, is a sum of $d$ roughly independent, similarly-distributed terms — one per feature. §1.4 already proved the relevant fact about sums of many independent quantities: they concentrate tightly around their expected value, with fluctuations that grow only like $\\sqrt{d}$ while the sum itself grows like $d$. So the <i>relative</i> sample-to-sample wobble in a squared distance, roughly $\\sqrt{d}/d = 1/\\sqrt{d}$, shrinks toward zero as $d$ grows. Every pair of random points ends up separated by almost exactly the same distance, whichever pair you pick — the nearest neighbour and the farthest neighbour in the entire dataset start to look equally far away, because the randomness that used to distinguish "close" from "far" has been averaged out by having so many features to sum over.</p>

<p>Simulated numbers make the effect concrete rather than theoretical. Draw 300 random points and one random query, all with independent Gaussian coordinates, and measure the ratio of the nearest point's distance to the farthest point's distance. At $d=2$ that ratio averages around <b>0.03</b> — the nearest neighbour is roughly thirty times closer than the farthest, distance is clearly informative. At $d=10$ it has already risen to about <b>0.29</b>. By $d=100$ it reaches roughly <b>0.70</b>, and by $d=1{,}000$ it sits near <b>0.90</b> — the nearest and farthest points in the entire dataset are barely distinguishable, and "find the nearest neighbour" has become "find one of the many points at essentially the typical distance."</p>

<p><b>What you are looking at.</b> The x-axis is the number of dimensions $d$, swept from 1 up to 800; the y-axis is the ratio of nearest to farthest distance, averaged over many random point sets at that dimension. The red dashed line at 1.0 marks the point where every distance has become indistinguishable.</p>

<p><b>What to do with it.</b> Read the curve left to right: it starts near zero, where distance is maximally informative, and rises steadily toward 1 as dimension grows, never turning back. Increase the <i>points</i> control and note the curve's shape barely moves — this is not a small-sample artefact that more data fixes, it is a geometric fact about how volume behaves in many dimensions.</p>

<p><b>The thing genuinely worth noticing.</b> The climb is not slow. By a few hundred dimensions — well short of the thousand-plus dimensions an ordinary sentence embedding uses — the ratio is already deep into the range where "nearest" and "farthest" are hard to tell apart from a few samples. This is exactly why production embeddings are engineered down to the hundreds rather than left at hundreds of thousands of raw dimensions, and why cosine similarity (§0.2), which compares direction rather than raw distance, tends to degrade more gracefully than Euclidean distance at high dimension.</p>

${H.lab('curse', 'The curse of dimensionality, measured', 'Distances between random points in d dimensions, computed here. As d grows the ratio of nearest to farthest approaches 1 — the geometric fact that kills distance-based methods in high dimensions.')}

${H.pitfall(`<p>Using k-NN, or an RBF kernel (§2.6), on unscaled features combines both failure modes at once: one column measured in large units silently dominates every distance calculation from the start, and if the feature count is also large, the concentration effect above is quietly compounding the damage on top. Scale first, always, and treat "how many features, really" as a design decision rather than an afterthought.</p>`)}

<h2><span class="sn">2.5.3</span> Naive Bayes: Bayes' rule, extended to many pieces of evidence at once</h2>

<p>§1.1 built Bayes' rule around a single piece of evidence: one positive fraud-detector alert, revised against a background rate, to land on the true probability of fraud. A real spam filter almost never has the luxury of just one clue — an email is dozens or hundreds of words, each one evidence, and the question is how to fold all of them into a single verdict using nothing but the same rule §1.1 already proved.</p>

<p>In principle Bayes' rule already tells you how: $P(c\\mid x)\\propto P(x\\mid c)P(c)$, exactly as in §1.1, where now $x=(x_1,\\ldots,x_T)$ is the entire vector of word-presence indicators for every word in the vocabulary. The obstacle is not the rule, it is $P(x\\mid c)$ itself — the full <i>joint</i> probability of every combination of words appearing together, given the class. With a vocabulary of just 10,000 words, the joint distribution over which subset appeared has $2^{10{,}000}-1$ free parameters to estimate — a number with over three thousand digits, comfortably beyond the roughly $10^{80}$ atoms believed to exist in the observable universe. No dataset that will ever be collected could estimate that table. Something has to give, and Naive Bayes is the specific thing that gives: assume the words are <b>conditionally independent given the class</b>, so the monstrous joint collapses into a product of one-word-at-a-time probabilities you can actually estimate — one number per word per class, ten thousand parameters instead of $2^{10{,}000}$.</p>

${H.deriv('from Bayes\' rule to the Naive Bayes classifier', [
      ['$P(c\\mid x) = \\dfrac{P(x\\mid c)\\,P(c)}{P(x)}$', 'Bayes\' rule exactly as derived in §1.1, applied here with $x$ standing for the entire feature vector rather than a single piece of evidence.'],
      ['$P(x\\mid c) = P(x_1,\\ldots,x_T\\mid c) \\overset{\\text{assume}}{=} \\prod_{t=1}^T P(x_t\\mid c)$', 'The "naive" step: assume each feature is conditionally independent of every other feature, once you already know the class. This is almost never exactly true — knowing a document contains "offer" makes "free" more likely too, class or no class — but it is what turns an intractable joint into a tractable product.'],
      ['$\\arg\\max_c P(c\\mid x) = \\arg\\max_c \\dfrac{P(c)\\prod_t P(x_t\\mid c)}{P(x)}$', 'Substitute the independence assumption back into Bayes\' rule.'],
      ['$= \\arg\\max_c P(c)\\prod_{t=1}^T P(x_t\\mid c)$', '$P(x)$ does not depend on $c$ at all — it is the same number under every candidate class — so it cannot change which class has the largest score, and it can be dropped entirely from a decision that only cares about the arg max.']
    ], 'The independence assumption is doing all the work, and it is worth being honest that it is usually false. What rescues the classifier is the very last step: an arg max only needs the correct class to score higher than the others, not for either score to be numerically accurate. Errors from the false independence assumption tend to distort both scores in similar, correlated ways, so the ranking between them survives even when neither individual number would.')}

${H.worked('worked Naive Bayes — three words, two classes', `
<p>Priors $P(\\text{spam})=0.3$, $P(\\text{ham})=0.7$. Word likelihoods "offer" 0.20 / 0.02, "meeting" 0.01 / 0.15, "free" 0.25 / 0.03 (spam / ham). Document = {offer, free}.</p>
${H.code(`spam score = 0.3 × 0.20 × 0.25 = 0.0150
ham  score = 0.7 × 0.02 × 0.03 = 0.00042
P(spam|doc) = 0.0150 / (0.0150 + 0.00042) = 0.973`)}
<p>The independence assumption is plainly false here too — "free" and "offer" co-occur constantly in real spam, they are not independent events even conditional on the message being spam — and the classification is still right, because the <i>ranking</i> survives the error even though 0.973 is overconfident about it. That is the whole story of Naive Bayes in one number: <mark>good classifier, bad probability</mark> — so if you need calibrated output, calibrate it (§2.12).</p>`)}

${H.history(`<p>Naive Bayes classifiers for spam predate almost every other machine-learning technique in production use today, and they became genuinely popular in the early 2000s precisely because the alternative — hand-written keyword rules, the exact approach §0.1 opens the entire course by watching collapse under its own contradictions — kept losing to spammers who simply learned which words the rules blocked. A model built from nothing but word-frequency counting, with no labelled linguistic structure at all, turned out to beat carefully hand-tuned rule lists, and it did so running comfortably on the hardware of the time. It is a useful check on intuition: a classifier with a conspicuously wrong core assumption, competing against careful human engineering, and winning anyway because the thing it actually needs to get right — the ranking, not the probability — was never the thing the wrong assumption damages most.</p>`)}

${H.table(['Variant', 'Uses', 'Best for'], [
      ['Multinomial', 'token counts', 'text, the default'],
      ['Bernoulli', 'presence / absence', 'very short documents, where a word appearing twice means little'],
      ['Gaussian', 'continuous features, normal within each class', 'a fast baseline; badly wrong on skewed financial features unless transformed']
    ])}

<p>One further piece of arithmetic hygiene matters more than it looks. A document of a thousand words, each contributing a likelihood around 0.01, multiplies out to roughly $0.01^{1000}=10^{-2000}$ — a number so far below the smallest positive value a 64-bit float can represent (around $5\\times10^{-324}$) that it silently rounds to exactly zero, long before you reach a thousand words. Work in log space instead — sum $\\log P$ rather than multiplying $P$ — and the identical calculation becomes a sum of a thousand ordinary-sized negative numbers with no underflow anywhere, because addition does not shrink the way repeated multiplication of small numbers does.</p>

<p><b>What you are looking at.</b> A worked factor table: the prior and each word's likelihood ratio for spam and for ham, multiplied down into a spam score and a ham score, then converted into a posterior probability bar at the bottom.</p>

<p><b>What to do with it.</b> Drag the "offer" and "free" likelihood-ratio sliders and watch the posterior bar respond immediately — each ratio is how many times more likely that word is in spam than in ham, so pushing a ratio up should, and does, push the posterior toward spam. Then toggle on the unseen word "quokka" with Laplace smoothing switched off.</p>

<p><b>The thing genuinely worth noticing.</b> With smoothing off, adding one single word the model has never seen before — probability exactly zero in both classes — collapses <i>both</i> the spam score and the ham score to zero simultaneously, because one factor of zero anywhere in a product kills the whole product, regardless of how strong the other nine hundred and ninety-nine words' evidence was. Laplace smoothing exists for exactly this failure: give every word a small non-zero floor probability in every class, and one unseen token becomes a mild vote of "no strong evidence either way" instead of an instant, total system failure.</p>

${H.lab('nb', 'Naive Bayes, editable', 'Change the priors and the word likelihoods, toggle Laplace smoothing, and watch both the decision and the confidence move. Add the unseen word and watch the un-smoothed model produce a zero.')}

${H.probe([
      ['Why does Naive Bayes work despite a false assumption?', 'Classification needs only the argmax of the class scores; dependence errors distort magnitudes far more than ranking.'],
      ['k = 1 vs k = 25?', 'Pure variance vs high bias — the same data, one hyperparameter.'],
      ['Why does k-NN fail in high dimensions?', 'Distance concentration: nearest and farthest neighbours become nearly equidistant.']
    ], 'Using k-NN or an RBF kernel on unscaled features — one column in large units then dominates every distance.')}`,
    labs: {
      knn: function (host) {
        const data = Num.dataset('blobs', 70, .55, 8);
        const st = Viz.controls(host, [
          { k: 'k', label: 'k (neighbours)', min: 1, max: 45, step: 2, value: 1, fmt: v => v },
          { k: 'scale', label: 'scale feature 2 by', min: .2, max: 5, step: .1, value: 1, fmt: v => '×' + v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'tr', label: 'training accuracy', cls: 'key' }, { k: 'loo', label: 'leave-one-out accuracy' }, { k: 'char', label: 'model character' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const X = data.X.map(p => [p[0], p[1] * st.scale]);
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2 * Math.max(1, st.scale), 3.2 * Math.max(1, st.scale)] })
              .frame({ xlabel: 'x₁', ylabel: 'x₂ (scaled)' });
            const predict = (a, b) => {
              const d = X.map((p, i) => [(p[0] - a) ** 2 + (p[1] - b) ** 2, data.y[i]]).sort((u, v) => u[0] - v[0]);
              let s = 0; for (let i = 0; i < Math.min(st.k, d.length); i++) s += d[i][1];
              return s / Math.min(st.k, d.length);
            };
            P.clip(() => Labs.boundary(P, predict, { step: 5 }));
            P.clip(() => Labs.points(P, X, data.y));
            const trAcc = Num.mean(X.map((p, i) => (predict(p[0], p[1]) > .5 ? 1 : 0) === data.y[i] ? 1 : 0));
            const loo = Num.mean(X.map((p, i) => {
              const d = X.map((q, j) => j === i ? [1e9, 0] : [(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, data.y[j]]).sort((u, v) => u[0] - v[0]);
              let s = 0; for (let t = 0; t < Math.min(st.k, d.length - 1); t++) s += d[t][1];
              return ((s / Math.min(st.k, d.length - 1)) > .5 ? 1 : 0) === data.y[i] ? 1 : 0;
            }));
            out({
              tr: (trAcc * 100).toFixed(1) + '%', loo: (loo * 100).toFixed(1) + '%',
              char: st.k <= 3 ? 'high variance' : st.k >= 21 ? 'high bias' : 'balanced'
            });
          }
        });
        Viz.note(host, 'At k = 1 training accuracy is 100% and leave-one-out is much lower — the textbook signature of memorisation. Now drag the scale slider: nothing about the data changed, but the boundary does, because distance is not unit-free.');
      },

      curse: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'points', min: 50, max: 1000, step: 50, value: 300, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'd2', label: 'near/far ratio @ d=2', cls: 'good' }, { k: 'd10', label: '@ d=10' },
          { k: 'd100', label: '@ d=100', cls: 'bad' }, { k: 'd1000', label: '@ d=1000', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(13);
            const dims = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400, 800];
            const ratios = dims.map(d => {
              const pts = Array.from({ length: st.n }, () => Array.from({ length: d }, () => R.normal(0, 1)));
              const q = Array.from({ length: d }, () => R.normal(0, 1));
              const ds = pts.map(p => { let s = 0; for (let i = 0; i < d; i++) s += (p[i] - q[i]) ** 2; return Math.sqrt(s); });
              return [d, Math.min.apply(null, ds) / Math.max.apply(null, ds)];
            });
            const P = Viz.plot(ctx, w, h, { xd: [0, 800], yd: [0, 1] })
              .frame({ xlabel: 'dimensions d', ylabel: 'nearest distance ÷ farthest distance' });
            P.clip(() => {
              P.line(ratios, { color: T.blue, width: 2.6 });
              P.dots(ratios, { r: 3.4, color: T.blue });
              P.hline(1, { color: T.red, dash: [4, 4], label: 'everything equidistant — "nearest" is meaningless' });
            });
            const get = d => { const r = ratios.find(x => x[0] >= d); return r ? r[1].toFixed(3) : '—'; };
            out({ d2: get(2), d10: get(10), d100: get(100), d1000: get(400) });
          }
        });
        Viz.note(host, 'This is why embeddings are 384–1536 dimensions rather than 100,000, why cosine similarity is preferred to Euclidean at high d, and why the answer to "just use k-NN" is often "on what distance?"');
      },

      nb: function (host) {
        const st = Viz.controls(host, [
          { k: 'prior', label: 'P(spam)', min: .02, max: .9, step: .01, value: .3, fmt: v => v.toFixed(2) },
          { k: 'offer', label: 'P(offer | spam) / P(offer | ham)', min: .5, max: 30, step: .5, value: 10, fmt: v => v.toFixed(1) + '×' },
          { k: 'free', label: 'P(free | spam) / P(free | ham)', min: .5, max: 30, step: .5, value: 8.3, fmt: v => v.toFixed(1) + '×' },
          { k: 'unseen', label: 'include an unseen word', type: 'toggle', value: false },
          { k: 'smooth', label: 'Laplace smoothing', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'spam', label: 'spam score' }, { k: 'ham', label: 'ham score' },
          { k: 'post', label: 'P(spam | doc)', cls: 'key' }, { k: 'call', label: 'decision' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const pHamOffer = .02, pHamFree = .03;
            const pSpamOffer = Math.min(.95, pHamOffer * st.offer), pSpamFree = Math.min(.95, pHamFree * st.free);
            let terms = [
              ['prior', st.prior, 1 - st.prior],
              ['"offer"', pSpamOffer, pHamOffer],
              ['"free"', pSpamFree, pHamFree]
            ];
            if (st.unseen) {
              const eps = st.smooth ? 1 / 1000 : 0;
              terms.push(['"quokka" (unseen)', eps, eps]);
            }
            let spam = 1, ham = 1;
            terms.forEach(t => { spam *= t[1]; ham *= t[2]; });
            const post = (spam + ham) > 0 ? spam / (spam + ham) : NaN;
            ctx.font = '12px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const x0 = 14, colS = Math.min(300, w * .48), colH = Math.min(420, w * .72);
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('factor', x0, 22); ctx.textAlign = 'right';
            ctx.fillText('spam', colS, 22); ctx.fillText('ham', colH, 22);
            terms.forEach((t, i) => {
              const y = 46 + i * 24;
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.fillText(t[0], x0, y);
              ctx.textAlign = 'right';
              ctx.fillStyle = t[1] === 0 ? T.red : T.text; ctx.fillText(t[1].toFixed(4), colS, y);
              ctx.fillStyle = t[2] === 0 ? T.red : T.text; ctx.fillText(t[2].toFixed(4), colH, y);
            });
            const yT = 46 + terms.length * 24 + 8;
            ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(x0, yT - 12); ctx.lineTo(colH, yT - 12); ctx.stroke();
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'left';
            ctx.fillText('product', x0, yT + 4);
            ctx.textAlign = 'right';
            ctx.fillStyle = spam === 0 ? T.red : T.blue; ctx.fillText(spam.toExponential(3), colS, yT + 4);
            ctx.fillStyle = ham === 0 ? T.red : T.green; ctx.fillText(ham.toExponential(3), colH, yT + 4);
            // posterior bar
            const by = yT + 34, bw = Math.max(120, Math.min(colH - x0, w - 28));
            if (isFinite(post)) {
              ctx.fillStyle = T.blue; ctx.fillRect(x0, by, bw * post, 22);
              ctx.fillStyle = T.green; ctx.fillRect(x0 + bw * post, by, bw * (1 - post), 22);
              ctx.fillStyle = '#fff'; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'left';
              if (post > .15) ctx.fillText(' spam ' + (post * 100).toFixed(1) + '%', x0 + 4, by + 11);
            } else {
              ctx.fillStyle = T.red; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left';
              ctx.fillText('Both products are zero — one unseen word destroyed the whole document. This is what Laplace smoothing exists for.', x0, by + 11);
            }
            out({
              spam: spam.toExponential(2), ham: ham.toExponential(2),
              post: isFinite(post) ? (post * 100).toFixed(1) + '%' : 'undefined',
              call: !isFinite(post) ? 'broken' : (post > .5 ? 'SPAM' : 'ham')
            });
          }
        });
        Viz.note(host, 'Turn on the unseen word with smoothing off and both scores collapse to zero — the classifier stops working entirely. That single failure mode is why smoothing is not optional.');
      }
    },
    quiz: [
      {
        q: 'k-NN with k = 1 has 100% training accuracy. What does that tell you?',
        options: ['The model is excellent', 'Nothing — every point is its own nearest neighbour; only held-out accuracy is informative', 'The data is separable', 'k should be lower'],
        answer: 1,
        why: 'A 1-nearest-neighbour classifier\'s nearest neighbour on its own training data is always itself, so 100% training accuracy is guaranteed by construction and says nothing about how the model behaves on a point it has not already memorised — only a holdout or leave-one-out estimate, which never lets a point be its own neighbour, is informative. "The model is excellent" is the tempting reading if you treat training accuracy the way you would for a model with limited capacity, where a high score is genuine evidence of fit — but $k=1$ has essentially unlimited capacity to memorise exactly the training set it was shown, a fundamentally different situation. "The data is separable" does not follow either: even hopelessly overlapping classes produce perfect $k=1$ training accuracy, because separability is a property of the true underlying classes while this number is a property of the evaluation procedure. "k should be lower" is not merely unhelpful but nonsensical, since $k=1$ is already the smallest value $k$ can take. The general principle, central to §2.1\'s whole split discipline, is that any accuracy computed on data a model has already seen is a fact about memorisation capacity, not generalisation, and $k=1$ is simply the starkest possible illustration of it.'
      },
      {
        q: 'Naive Bayes gives P(spam) = 0.973 on a document where its independence assumption is badly violated. The correct reading is…',
        options: ['The probability is trustworthy', 'The classification is likely right but the probability is overconfident; calibrate if you need probabilities', 'The model has failed', 'You must drop the correlated features'],
        answer: 1,
        why: 'Naive Bayes multiplies per-feature likelihoods as though the features were conditionally independent, and when they are not, correlated evidence gets counted more than once — inflating the magnitude of the final probability far more than it disturbs which class comes out on top, so the classification decision is usually still trustworthy even though the number attached to it is not. "The probability is trustworthy" is tempting precisely because a confident-looking number feels like it should mean what it says, but that conflates ranking with calibration, exactly the split §2.12 builds at length. "The model has failed" overreacts: a good classification paired with an overconfident probability does not mean the model is broken, only that honest magnitude was never something the independence assumption was built to deliver. "You must drop the correlated features" is an overcorrection likely to discard real predictive signal in order to fix a problem that has a cheaper, more targeted remedy — fitting a separate calibrator (Platt or isotonic, §2.12) on top of the existing scores. The lesson is the same one AUC-versus-calibration teaches throughout this course: a model can be an excellent judge of order and a poor reporter of magnitude at the very same time.'
      }
    ],
    cards: [
      { q: 'k-NN’s two failure modes', a: 'Unscaled features (distance dominated by large-unit columns) and high dimensions (distance concentration).' },
      { q: 'Naive Bayes in one line', a: '$\\arg\\max_c P(c)\\prod_t P(x_t|c)$ with conditional independence; good classifier, bad probability.' },
      { q: 'Why Laplace smoothing?', a: 'One unseen token would otherwise zero the entire product.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.6 */
  ML.section({
    id: 'svm', track: 'classical', num: '2.6',
    title: 'SVMs and kernels',
    lede: 'Uses convexity (§1.9) and PSD (§1.8); the kernel idea returns as the inner-product view of attention (§4.3).',
    html: `
<p>Two classes of points, cleanly separable by a straight line. Draw one. Now draw a second line that also separates them perfectly, and a third. There are infinitely many hyperplanes that get zero training errors on a separable dataset, and training error alone cannot choose between them — it reports zero for every single one. Yet they are obviously not equally good: some of these lines pass a hair's breadth from a training point, and the smallest jitter in a future example would land it on the wrong side. Some sit comfortably in the middle of the gap between the two clouds. Before writing down a single constrained optimisation problem, it is worth just staring at that picture and asking what "comfortably in the middle" should mean, numerically.</p>

<h2><span class="sn">2.6.1</span> The margin, measured before it is optimised</h2>

<p>A hyperplane in $d$ dimensions is the set of points satisfying $w^\\mathsf{T}x+b=0$. Take any two points $x_1,x_2$ that both lie on it: $w^\\mathsf{T}x_1+b=0=w^\\mathsf{T}x_2+b$, so subtracting gives $w^\\mathsf{T}(x_1-x_2)=0$ — the vector $w$ has a dot product of zero with every direction lying inside the hyperplane, which by §0.2's definition of orthogonality means $w$ is <b>perpendicular to the hyperplane itself</b>. That single fact is enough to work out the distance from any point to the plane, using the same projection idea §0.2 built for shadows.</p>

${H.deriv('the distance from a point to a hyperplane, from orthogonality alone', [
      ['$x_0 = x_p + r\\dfrac{w}{\\|w\\|}$, for some scalar $r$', 'Since $w$ is perpendicular to the plane, the displacement from the nearest point $x_p$ <i>on</i> the plane to $x_0$ must run entirely along $w$\'s direction — any component sideways would mean $x_p$ was not actually the nearest point. $r$ is the signed distance we want.'],
      ['$w^\\mathsf{T}x_p + b = 0$', '$x_p$ lies on the hyperplane by construction, so it satisfies the hyperplane\'s own equation.'],
      ['$w^\\mathsf{T}x_0 + b = w^\\mathsf{T}x_p + r\\dfrac{w^\\mathsf{T}w}{\\|w\\|} + b = r\\|w\\|$', 'Substitute the first line into $w^\\mathsf{T}x_0+b$ and use the second line to cancel $w^\\mathsf{T}x_p+b$ to zero; $w^\\mathsf{T}w=\\|w\\|^2$ (§0.2), so $\\|w\\|^2/\\|w\\|=\\|w\\|$.'],
      ['$r = \\dfrac{w^\\mathsf{T}x_0+b}{\\|w\\|}$, so distance $=\\dfrac{|w^\\mathsf{T}x_0+b|}{\\|w\\|}$', 'Divide through by $\\|w\\|$ and take the magnitude, since a distance cannot be negative.']
    ], 'Put a number on it: with $w=(3,4)$, $b=-5$, the plane is $3x_1+4x_2-5=0$ and $\\|w\\|=\\sqrt{9+16}=5$ exactly. The point $(4,4)$ is at distance $|3(4)+4(4)-5|/5 = 23/5 = 4.6$. Nothing here required calculus — it fell entirely out of $w$ being perpendicular to its own level set.')}

<p>For a correctly classified training point with label $y_i\\in\\{-1,+1\\}$, the <i>signed</i> version of that distance, $y_i(w^\\mathsf{T}x_i+b)/\\|w\\|$, is positive exactly when the classification is right. Define the <b>margin</b> of a classifier as the smallest such signed distance over the whole training set — how far the closest point of any kind sits from the boundary. A big margin means even the most marginal training example has breathing room; a margin that is barely positive means some point is sitting right on the line.</p>

<p>Here is the move that turns "maximise the margin" into a clean optimisation problem. The pair $(w,b)$ and the pair $(2w,2b)$ describe the exact same hyperplane — scaling both by any positive constant changes nothing about which points are on which side. That freedom can be spent: rescale $w$ and $b$ so that whichever training points are closest to the boundary satisfy $y_i(w^\\mathsf{T}x_i+b)=1$ exactly. With that convention fixed, the margin is no longer "the minimum of some expression over all points" — it is simply $1/\\|w\\|$, read straight off the formula above. A smaller $\\|w\\|$ is now, mechanically, a wider margin: $\\|w\\|=2$ gives a margin of $0.5$; halve $\\|w\\|$ to $1$ and the margin doubles to $1$. <b>Maximising the margin is exactly minimising $\\|w\\|$</b>, and minimising the mathematically nicer $\\tfrac12\\|w\\|^2$ gives the identical answer while staying easy to differentiate:</p>

$$\\min_{w,b} \\ \\tfrac12\\|w\\|^2 \\quad \\text{subject to} \\quad y_i(w^\\mathsf{T}x_i+b) \\ge 1 \\ \\text{for every } i$$

<p>Not one Lagrangian has appeared yet. This is purely geometry: pick the widest possible corridor that still has every training point on the correct side of it, stated as the smallest $\\|w\\|$ consistent with that requirement.</p>

<h2><span class="sn">2.6.2</span> Why the solution depends on so few points — the dual</h2>

<p>The problem above is a <i>constrained</i> minimisation with one inequality per training point, exactly the shape §1.9.5 built the general machinery for: form the Lagrangian, differentiate, and read off what the stationary condition implies. Carry that machinery over directly rather than rebuilding it.</p>

${H.deriv('the SVM dual, and where the support vectors come from', [
      ['$\\mathcal{L}(w,b,\\alpha) = \\tfrac12\\|w\\|^2 + \\sum_i \\alpha_i\\big[1-y_i(w^\\mathsf{T}x_i+b)\\big]$', 'The Lagrangian for this inequality-constrained problem, exactly the KKT construction of §1.9.5: one non-negative multiplier $\\alpha_i \\ge 0$ per training-point constraint.'],
      ['$\\nabla_w \\mathcal{L} = w - \\sum_i \\alpha_i y_i x_i = 0 \\ \\Rightarrow\\ w = \\sum_i \\alpha_i y_i x_i$', 'Set the derivative with respect to $w$ to zero, using $\\nabla_w\\|w\\|^2=2w$ from §0.7. The optimal weight vector is forced to be a weighted combination of the training points themselves — nothing else is available to build it from.'],
      ['$\\partial \\mathcal{L}/\\partial b = -\\sum_i \\alpha_i y_i = 0$', 'Set the derivative with respect to $b$ to zero as well; this becomes a constraint on the multipliers in the problem that remains.'],
      ['substitute both back into $\\mathcal{L}$ to eliminate $w,b$', 'Standard Lagrangian-dual manoeuvre (§1.9): having pinned down $w$ and the constraint on $b$, replace them in the Lagrangian to get an expression purely in terms of $\\alpha$, called the <b>dual</b>.'],
      ['$\\max_\\alpha \\sum_i \\alpha_i - \\tfrac12\\sum_{i,j}\\alpha_i\\alpha_j y_i y_j\\, x_i^\\mathsf{T}x_j \\ \\ \\text{s.t. } \\alpha_i\\ge0,\\ \\sum_i\\alpha_iy_i=0$', 'The data enters this problem only through pairwise inner products $x_i^\\mathsf{T}x_j$ — every single training point interacts with every other one exclusively through a dot product, never through its raw coordinates. Hold onto that observation; §2.6.4 spends the kernel trick entirely on it.']
    ], 'The single most useful consequence is not the formula itself but complementary slackness, already proved in general in §1.9.5: $\\alpha_i\\big[1-y_i(w^\\mathsf{T}x_i+b)\\big]=0$ for every $i$. A point strictly inside the correct side of the margin has $1-y_i(w^\\mathsf{T}x_i+b)<0$, so its multiplier is forced to $\\alpha_i=0$ — and since $w=\\sum_i\\alpha_iy_ix_i$, a zero multiplier means that point contributes nothing to $w$ whatsoever. Only points sitting exactly on the margin boundary can have $\\alpha_i>0$. Those are the <b>support vectors</b>, and the sparsity of an SVM is not a separate algorithmic trick bolted on afterwards — it is complementary slackness, proved once in §1.9 and inherited here for free.')}

<h2><span class="sn">2.6.3</span> Soft margin: when no hyperplane separates the data at all</h2>

<p>Everything above assumed a perfectly separating hyperplane exists. Real data rarely cooperates — two overlapping clouds with a handful of points wrong on either side is the ordinary case, and the constraint $y_i(w^\\mathsf{T}x_i+b)\\ge1$ for <i>every</i> point has no solution at all when even one pair of same-region, opposite-label points sits close together. The fix is to let each point violate its constraint by some amount $\\xi_i\\ge0$, called <b>slack</b>, and pay a price for doing so:</p>

$$\\min_{w,b,\\xi}\\ \\tfrac12\\|w\\|^2 + C\\sum_i\\xi_i \\quad \\text{s.t.}\\quad y_i(w^\\mathsf{T}x_i+b)\\ge1-\\xi_i,\\ \\ \\xi_i\\ge0$$

<p>It is worth seeing explicitly that this is not a new kind of object — it is hinge loss plus an L2 penalty in disguise, the exact combination §2.3 already built a full geometric and analytic case for.</p>

${H.deriv('the soft-margin SVM is hinge loss plus L2, in three lines', [
      ['for fixed $w,b$, the optimal $\\xi_i$ is as small as feasibility allows', 'The objective only ever wants $\\xi_i$ smaller, since it appears with a positive coefficient $C$, so at the optimum $\\xi_i$ sits at the smallest value the constraints $\\xi_i\\ge0$ and $\\xi_i\\ge1-y_i(w^\\mathsf{T}x_i+b)$ permit.'],
      ['$\\xi_i^\\star = \\max\\big(0,\\ 1-y_i(w^\\mathsf{T}x_i+b)\\big)$', 'Take the larger of the two lower bounds on $\\xi_i$ — exactly the definition of the <b>hinge loss</b> for this example.'],
      ['$\\min_{w,b}\\ \\tfrac12\\|w\\|^2 + C\\sum_i \\max\\big(0,\\,1-y_i(w^\\mathsf{T}x_i+b)\\big)$', 'Substitute $\\xi_i^\\star$ back into the objective, eliminating the slack variables entirely and leaving an ordinary unconstrained loss: hinge loss, summed over examples, plus a penalty on $\\|w\\|^2$ — precisely the loss-plus-penalty shape §2.3 built from scratch.']
    ], 'Divide the whole objective by $C$ and it becomes $\\frac{1}{2C}\\|w\\|^2 + \\sum_i\\text{hinge}_i$ — the standard §2.3 form with regularisation strength $\\lambda=1/C$. So $C$ is not a separate, mysterious SVM-specific dial; it is $1/\\lambda$ wearing a different name. Large $C$ means a small effective $\\lambda$, weak regularisation, a narrow margin that fits the training data hard; small $C$ means strong regularisation, a wide, forgiving margin that tolerates violations in exchange for a simpler boundary — the exact bias-for-variance trade §2.2 and §2.3 have already made twice.')}

<p><b>What you are looking at.</b> Training points on a plane, coloured by class, with a real soft-margin SVM solved live in the browser. The solid line through the middle is the decision boundary $w^\\mathsf{T}x+b=0$; the two faint parallel lines on either side are the margin boundaries, $w^\\mathsf{T}x+b=\\pm1$. Every point with an amber ring around it is a support vector — one of the points complementary slackness allowed to have $\\alpha_i>0$.</p>

<p><b>What to do with it.</b> Start on the linear kernel and blobs data, and watch which points get ringed: only the ones sitting on or just inside the margin, exactly as the derivation above predicts, never the ones sitting comfortably deep inside their own class's territory. Now drag $C$ down by several orders of magnitude and watch the margin visibly widen and more points get pulled into the ringed set as the boundary trades fit for width; drag $C$ up and the margin narrows back down toward the tightest line that still gets everything right.</p>

<p><b>The thing genuinely worth noticing.</b> Switch the dataset to <i>circles</i>, where one class rings the other, and stay on the linear kernel — the model is helpless, exactly as logistic regression was on the same shape in §2.4, because nothing about adding a margin changes the fact that a straight line cannot separate a ring from its centre. Now flip the kernel control to RBF with nothing else changed. The boundary curves to trace the ring perfectly, and the solver never once computed a lifted coordinate to do it — which is exactly what the next subsection explains.</p>

${H.lab('svm', 'Margin, slack, kernels — a real SMO solver', 'This trains an actual soft-margin SVM in your browser (simplified SMO). Ringed points are the support vectors; move C and γ and watch which points the solution depends on. On the circles data, switch to RBF and the boundary curves — without ever computing the lifted coordinates.')}

<h2><span class="sn">2.6.4</span> The kernel trick</h2>

<p>Recall the one fact flagged and held onto from the dual derivation: every training point interacts with every other one exclusively through the inner product $x_i^\\mathsf{T}x_j$, never through its raw coordinates on their own. That is a strange and useful accident of the algebra — it means the entire optimisation problem, and the entire decision rule built from its solution, could be handed a <i>different</i> function of two points in place of $x_i^\\mathsf{T}x_j$, and nothing in the derivation above would notice or care, provided that replacement function still behaves enough like an inner product for the argument to hold together.</p>

<p>Call that replacement a <b>kernel</b>, $K(x_i,x_j)$, and read it as: "the inner product these two points <i>would</i> have, if you first mapped them into some other, possibly much higher-dimensional feature space, and then took the ordinary dot product there." The trick is that you never actually construct that mapping or that higher-dimensional space — you only ever need the single number $K(x_i,x_j)$ that an inner product there would have produced, and every place the derivation used $x_i^\\mathsf{T}x_j$, substitute $K(x_i,x_j)$ instead. A straight line in the lifted space becomes a curved boundary back in the original one, purely because the lifting was curved.</p>

<p>The one requirement is that $K$ actually correspond to <i>some</i> inner product in <i>some</i> feature space — otherwise the whole justification for substituting it collapses, and nothing guarantees the resulting problem is even convex any more. §1.8 already proved the exact condition: a Gram matrix built from real inner products is always positive semi-definite, so a candidate kernel must be PSD too (<b>Mercer's condition</b>) or no such feature space exists for it to be secretly computing an inner product in.</p>

<p>The default choice, the RBF or Gaussian kernel $K(x,z)=\\exp(-\\gamma\\|x-z\\|^2)$, is worth computing once by hand: with $\\gamma=0.5$ and two points a squared distance of 2 apart, $K = \\exp(-0.5\\times2) = \\exp(-1) \\approx 0.368$ — points a little apart still register meaningful similarity; push them further apart and the exponential decay sends $K$ toward zero fast, so only genuinely nearby points influence each other much. That is what $\\gamma$ controls: a larger $\\gamma$ shrinks the effective neighbourhood of "nearby" faster, producing a more local, more wiggly boundary; a smaller $\\gamma$ treats a wider neighbourhood as similar, producing a smoother one.</p>

<p>The claim that the RBF kernel corresponds to an <i>infinite</i>-dimensional feature space is usually just asserted. It should not be, because it is one Taylor expansion away from being shown outright.</p>

${H.deriv('why the RBF kernel is an inner product in infinitely many dimensions', [
      ['$\\|x-z\\|^2 = \\|x\\|^2 - 2x^\\mathsf{T}z + \\|z\\|^2$', 'Expand the squared distance exactly as in §0.2, then split the sum inside the exponential into a product of three exponentials: $\\exp(-\\gamma\\|x-z\\|^2)=\\exp(-\\gamma\\|x\\|^2)\\exp(-\\gamma\\|z\\|^2)\\exp(2\\gamma x^\\mathsf{T}z)$.'],
      ['$\\exp(2\\gamma x^\\mathsf{T}z) = \\displaystyle\\sum_{k=0}^{\\infty} \\frac{(2\\gamma)^k}{k!}\\,(x^\\mathsf{T}z)^k$', 'The ordinary power series for $e^u$ with $u=2\\gamma x^\\mathsf{T}z$ — nothing kernel-specific yet, just the Taylor expansion of the exponential function you already know.'],
      ['each $(x^\\mathsf{T}z)^k$ is itself a valid polynomial kernel of degree $k$', 'A polynomial kernel is, by the same PSD argument as above, an inner product $\\phi_k(x)^\\mathsf{T}\\phi_k(z)$ where $\\phi_k$ lists every degree-$k$ monomial in the entries of its input. Degree $k$ alone already lives in a feature space of dimension growing like $\\binom{D+k-1}{k}$.'],
      ['$K(x,z) = \\exp(-\\gamma\\|x\\|^2)\\exp(-\\gamma\\|z\\|^2)\\displaystyle\\sum_{k=0}^{\\infty}\\frac{(2\\gamma)^k}{k!}\\,\\phi_k(x)^\\mathsf{T}\\phi_k(z)$', 'Substitute the previous line into the middle factor. This is now a weighted sum of inner products, one per degree $k=0,1,2,\\ldots$, which is itself a single inner product — stack every $\\phi_k$ end to end, scaled by $\\sqrt{(2\\gamma)^k/k!}$, into one feature vector with a block for every degree, and $K$ is the ordinary dot product of two such vectors.']
    ], 'Check it holds a number rather than a symbol: at $\\gamma=0.5$ and $x^\\mathsf{T}z=0.33$, the first 20 terms of the Taylor sum already reproduce $\\exp(2\\gamma x^\\mathsf{T}z)$ to twelve decimal places — the series converges fast. But convergence is not the same as termination: no finite prefix of that sum is exactly equal to the exponential, so the feature vector it represents genuinely has infinitely many non-zero blocks, one for every polynomial degree that exists. That vector could never be built, stored, or multiplied out by hand — only the single closed-form number $K(x,z)$ is ever actually computed, which is the entire kernel trick in one worked identity rather than a slogan.')}

${H.practice(`<p>The saving the kernel trick buys is not abstract even at ordinary, finite degrees. A degree-3 polynomial kernel on $D=100$ features has an explicit feature map with $\\binom{103}{3}=176{,}851$ coordinates — every triple product of the original 100 features, computed and stored for every single training point, just to reach degree 3. Evaluating $(1+x^\\mathsf{T}z)^3$ directly costs one dot product and two multiplications, $O(D)$ work, and returns the exact same number that an inner product between two 176,851-length vectors would have. That gap between $O(D)$ and $O(D^3)$ is already dramatic at a finite, nameable dimension; for the RBF kernel the equivalent explicit space is infinite, so there is no "expensive but possible" option to fall back on at all — the kernel function is the only way in.</p>`)}
${H.key('You never build the feature map. You only ever need the number the inner product in that space would have produced, and the kernel function computes that number directly.')}

<p>It is worth knowing, too, that this same infinite-dimensional space is exactly why, combined with a large enough $C$, an RBF kernel can separate literally any finite set of correctly-labelled points, memorisation included.</p>

<h2><span class="sn">2.6.5</span> Support vectors, and why SVMs lost tabular ML</h2>

<p>Everything about prediction after training reduces to the points complementary slackness spared: the fitted decision function is $\\sum_i \\alpha_i y_i K(x,x_i) + b$, summed only over support vectors, since every other point's $\\alpha_i$ is exactly zero and contributes nothing. That is why SVM prediction can be cheap even when training was expensive, and why a single mislabelled point sitting right on the boundary can distort the whole solution — it is exactly the kind of point most likely to earn a non-zero $\\alpha_i$ and become one of the few voices the model actually listens to.</p>

<p>What keeps SVMs from being the default choice on tabular data today is scale: the dual problem above involves an $n\\times n$ matrix of pairwise kernel values, so both memory and the solver's running time grow roughly with $n^2$. That is a minor inconvenience at ten thousand rows and an impossibility at ten million, which is exactly the regime gradient-boosted trees (§2.8) were built to handle natively — no kernel matrix, mixed categorical and numeric types without preprocessing, and training cost that scales far more gently with $n$. The margin idea did not disappear from practice, though; it resurfaces, in spirit, every time a modern method is judged by how far it pushes its decision away from the nearest awkward example rather than merely by whether it got the example right.</p>

${H.fig('KERNELS AS SIMILARITY', `
<p style="margin:0 0 8px">Every kernel is a similarity function, and choosing one is a modelling statement about what "close" means:</p>` +
      H.table(['Kernel', 'Form', 'Says'], [
        ['Linear', '$x^\\mathsf{T}z$', 'Similarity is alignment; the boundary is a hyperplane in the original features'],
        ['Polynomial', '$(1+x^\\mathsf{T}z)^d$', 'Interactions up to order $d$ matter'],
        ['RBF / Gaussian', '$\\exp(-\\gamma\\|x-z\\|^2)$', 'Only nearby points are similar; $1/\\sqrt\\gamma$ is the length scale'],
        ['Laplacian', '$\\exp(-\\gamma\\|x-z\\|_1)$', 'Same, with heavier tails — less smooth boundaries']
      ]), 'The RBF kernel corresponds to an infinite-dimensional feature map, which is why it can separate any finite dataset — and why unregularised (large C) RBF SVMs memorise happily.')}

${H.probe([
      ['What are support vectors?', 'The points on or inside the margin; they alone determine the boundary.'],
      ['What is C?', 'An inverse regularisation strength — the price of a margin violation, equivalent to hinge loss plus L2.'],
      ['Why must a kernel be PSD?', 'Otherwise it is not an inner product in any feature space and the dual is no longer convex.']
    ], 'Using an RBF kernel on unscaled features — the Euclidean distance is then dominated by whichever column happens to be measured in large units.')}`,
    labs: {
      svm: function (host) {
        let data = Num.dataset('blobs', 44, .5, 21);
        let model = null;
        const st = Viz.controls(host, [
          { k: 'kernel', label: 'kernel', type: 'buttons', value: 'linear', options: [{ v: 'linear', t: 'linear' }, { v: 'rbf', t: 'RBF' }, { v: 'poly', t: 'poly d=3' }] },
          { k: 'C', label: 'C (inverse regularisation)', min: -2, max: 2, step: .1, value: 0, fmt: v => Math.pow(10, v).toFixed(2) },
          { k: 'gamma', label: 'γ (RBF width)', min: -1.5, max: 1.2, step: .1, value: 0, fmt: v => Math.pow(10, v).toFixed(2) }
        ], fit);
        const out = Viz.readout(host, [
          { k: 'sv', label: 'support vectors', cls: 'key' }, { k: 'acc', label: 'training accuracy' },
          { k: 'C', label: 'C' }, { k: 'time', label: 'solve time' }
        ]);
        function fit() {
          const t0 = performance.now();
          model = Num.svm(data.X, data.y, {
            kernel: st.kernel, C: Math.pow(10, st.C), gamma: Math.pow(10, st.gamma), degree: 3, passes: 6
          });
          model._ms = (performance.now() - t0).toFixed(0);
          S.redraw();
        }
        const S = Viz.surface(host, {
          height: 350,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2, 3.2] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            if (model) {
              P.clip(() => {
                Labs.boundary(P, (x, y) => Num.sigmoid(model.decide([x, y]) * 2), { step: 5 });
                P.contours((x, y) => model.decide([x, y]), [-1, 1], { color: T.faint, width: 1.2, alpha: .9 });
                P.contours((x, y) => model.decide([x, y]), [0], { color: T.text, width: 2 });
              });
              P.clip(() => {
                Labs.points(P, data.X, data.y);
                model.sv.forEach(i => P.dots([data.X[i]], { r: 8, color: 'transparent', stroke: T.amber, strokeWidth: 2 }));
              });
              const acc = Num.mean(data.X.map((x, i) => ((model.decide(x) > 0 ? 1 : 0) === data.y[i]) ? 1 : 0));
              out({
                sv: model.sv.length + ' of ' + data.X.length, acc: (acc * 100).toFixed(1) + '%',
                C: Math.pow(10, st.C).toFixed(2), time: model._ms + ' ms'
              });
            }
          }
        });
        Viz.buttons(host, [
          { label: 'Blobs', on: () => { data = Num.dataset('blobs', 44, .5, 21); fit(); } },
          { label: 'Circles (needs RBF)', on: () => { data = Num.dataset('circles', 60, .18, 3); fit(); } },
          { label: 'Moons', on: () => { data = Num.dataset('moons', 56, .3, 9); fit(); } },
          { label: 'XOR', on: () => { data = Num.dataset('xor', 60, .3, 6); fit(); } }
        ]);
        Viz.legend(host, [{ c: Viz.theme().amber, t: 'support vectors (ringed)' }, { c: Viz.theme().text, t: 'decision boundary' }, { c: Viz.theme().faint, t: 'margin ±1' }]);
        fit();
        Viz.note(host, 'Load circles with a linear kernel: hopeless. Switch to RBF: solved. Nothing about the data changed — only the definition of similarity, and the algorithm never computed a single lifted coordinate.');
      }
    },
    quiz: [
      {
        q: 'Increasing C in a soft-margin SVM…',
        options: ['widens the margin and tolerates more violations', 'narrows the margin and penalises violations more — less regularisation', 'has no effect with an RBF kernel', 'is equivalent to increasing γ'],
        answer: 1,
        why: 'Increasing $C$ raises the price of letting any training point violate the margin, so the optimiser fits the training data harder and tolerates fewer violations, which narrows the margin and corresponds to weaker regularisation — the equivalent view, hinge loss plus an L2 penalty of size $1/C$, makes this explicit: a larger $C$ is a smaller effective penalty. "Widens the margin and tolerates more violations" gets the direction backwards, an easy sign error if you assume "more C, more of everything" without checking which quantity $C$ is actually the price of — it prices violating the margin, not having one. "Has no effect with an RBF kernel" is false: $C$ and a kernel\'s own parameters (an RBF\'s $\\gamma$, say) control two independent things — how much slack is tolerated versus how flexible the boundary\'s shape is allowed to be — and both stay active regardless of which kernel is chosen. "Equivalent to increasing γ" conflates those same two independent knobs; raising $\\gamma$ makes the boundary more locally wiggly, while raising $C$ makes the fit less tolerant of any violation, and mixing up the two is a common, costly confusion when tuning an SVM in practice.'
      },
      {
        q: 'Why is FlashAttention-style thinking irrelevant here but kernel PSD-ness essential?',
        options: ['PSD guarantees the kernel corresponds to an inner product, keeping the dual convex', 'PSD makes the kernel faster', 'PSD ensures features are scaled', 'It is a convention only'],
        answer: 0,
        why: 'Mercer\'s condition — that a valid kernel must be positive semi-definite — is exactly the requirement that some feature map exists whose inner product the kernel computes, which is the entire justification for the kernel trick: without a genuine underlying feature space, "replace the dot product with a kernel" is not well-defined, and the dual optimisation can lose convexity or become unbounded. "PSD makes the kernel faster" and "PSD ensures features are scaled" both describe properties a kernel might separately have, but neither is what PSD-ness actually guarantees — a kernel can be fast or slow, scaled or unscaled, independent of whether it is PSD at all, so these test whether performance gets confused with validity. "It is a convention only" is the most tempting option, because plenty of practical ML choices genuinely are conventions dressed up as rules — but PSD-ness is a hard mathematical requirement: without it, the implicit feature-space geometry that makes SVMs work is simply not there, and the dual can fail to have a well-defined optimum at all. The general test is whether you can tell "this makes the method faster or more convenient" apart from "this makes the method mathematically valid in the first place."'
      }
    ],
    cards: [
      { q: 'Soft-margin objective', a: '$\\min \\frac12\\|w\\|^2 + C\\sum\\xi_i$ — equivalently hinge loss + L2; $C$ is inverse regularisation.' },
      { q: 'The kernel trick', a: 'The dual depends on data only through inner products; replace them with a PSD kernel to work in an implicit feature space.' },
      { q: 'Why don’t SVMs own tabular ML?', a: 'Kernel matrices are $O(n^2)$; boosted trees scale and handle mixed types natively.' }
    ]
  });
})();
