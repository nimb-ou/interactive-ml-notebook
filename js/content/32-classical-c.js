/* ============================================================
   PART 2 — Core & classical ML (2.12 – 2.20)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.12 */
  ML.section({
    id: 'calibration', track: 'classical', num: '2.12',
    title: 'Class imbalance, calibration, and conformal prediction',
    lede: 'A model can rank flawlessly and still lie about probability. Builds on §2.1’s loss-vs-metric split; the practical stance is a genuinely contested topic, and §2.19 flags where it collides with fairness.',
    html: `
<p>Picture a lender who has just shipped a new probability-of-default model. Its AUC is 0.91 — excellent by any standard the team has ever shipped — and the head of risk is happy, right up until someone pulls every applicant the model scored at "around 12% chance of default" and counts how many actually defaulted. The true rate in that bucket is 24%. Twice what the model claimed. The model did not get worse at telling risky applicants from safe ones; on that question it is still the best model the team has ever built. It got the <i>number</i> wrong while getting the <i>order</i> right, and those turn out to be almost entirely separate achievements.</p>

<h2><span class="sn">2.12.1</span> Ranking is not the same claim as probability</h2>

<p>Start with what AUC actually measures, because the puzzle dissolves once you see it precisely. AUC is the probability that a randomly chosen positive example gets a higher score than a randomly chosen negative one (§2.13 builds this from the ROC curve). It is a statement about <b>order</b>: does the model put risky applicants above safe ones, on average, across every possible pair? It says nothing at all about whether the number attached to any one applicant means what it claims to mean. A model can get every single pairwise ordering right and still attach the label "12%" to a group of people who default at 24%, because nothing about ranking constrains the numbers to be honest — it only constrains their <i>order</i>.</p>

<p>Push that observation as far as it goes and you get a fact worth having in your pocket: if you take a model's scores and run them through <i>any</i> strictly increasing function — square them, take their square root, multiply every one by ten — the ranking metric does not move. Not approximately unchanged: identically, bit-for-bit unchanged.</p>

${H.deriv('why a monotone transform of the scores leaves AUC exactly where it was', [
      ['$\\mathrm{AUC} = P(\\text{score}(x^+) > \\text{score}(x^-))$', 'The definition: draw a random positive example $x^+$ and a random negative example $x^-$, and AUC is the chance the positive scores higher. Everything downstream follows from this one probability.'],
      ['Let $g$ be strictly increasing. Then $a > b \\iff g(a) > g(b)$ for any two numbers $a, b$.', 'This is what "strictly increasing" means: $g$ never flips the order of two numbers it is applied to. Squaring is strictly increasing on scores already confined to $[0,1]$; so is any sigmoid, any power, any positive linear rescaling.'],
      ['$P(\\text{score}(x^+) > \\text{score}(x^-)) = P(g(\\text{score}(x^+)) > g(\\text{score}(x^-)))$', 'Substitute the equivalence from line 2 directly into the probability from line 1 — the event being measured has not changed, only how it is spelled.'],
      ['$\\mathrm{AUC}(\\text{score}) = \\mathrm{AUC}(g \\circ \\text{score})$', 'Which is exactly the claim: transforming every score through the same increasing function leaves AUC untouched, because AUC only ever asks "which one is bigger," and a monotone map cannot change the answer to that question.']
    ], 'The identical argument applies to KS, to Gini (which is just $2\\cdot\\mathrm{AUC}-1$), to precision-at-a-fixed-approval-rate, and to every metric in §2.13 built purely from rank order. None of them can detect miscalibration, ever, by construction — which is exactly why a separate diagnostic is needed.')}

<p>So a model is free to distort its probabilities as much as it likes, in any way that preserves order, without a single ranking metric so much as flinching. Here is that freedom, made concrete with a number you can check by hand. Take five applicants whose <i>true</i, honestly-calibrated default probabilities happen to be 0.9, 0.7, 0.5, 0.3 and 0.1 — meaning if you gathered a large population of each type, that fraction of them would genuinely default. Now suppose a badly-tuned model reports the <i>square</i> of the true probability instead — a distortion that happens routinely when a score is pushed through too sharp a sigmoid, or when class weighting during training shifts the decision function without anyone correcting for it afterwards:</p>

${H.worked('a monotone distortion that wrecks calibration and leaves ranking untouched', `
${H.table(['true probability $p$', 'reported $p^2$', 'gap'], [
      ['0.90', '0.81', '−0.09'],
      ['0.70', '0.49', '−0.21'],
      ['0.50', '0.25', '−0.25'],
      ['0.30', '0.09', '−0.21'],
      ['0.10', '0.01', '−0.09']
    ])}
<p>Squaring is strictly increasing on $[0,1]$, so these five applicants are still ranked in exactly the right order — the riskiest applicant still gets the highest score, the safest the lowest, and AUC computed on any population built from this rule comes out identical to AUC computed on the honest probabilities. A direct simulation confirms it: 20,000 synthetic applicants, true probabilities and their squares both used as the score, give AUC $=0.83616$ <i>to five decimal places</i>, both ways — not similar, identical, because the algebra above guarantees it exactly.</p>
<p>But look at the gap column. Every single reported number understates the truth, by as much as 0.25 in the middle of the range. If you used $(a-y)^2$, the squared-error penalty a probability forecast earns per example, the exact identity $\\mathbb{E}[(a-y)^2] = (a-p)^2 + p(1-p)$ for a fixed prediction $a$ and $y\\sim\\text{Bernoulli}(p)$ tells you precisely how much this costs: averaged over the five applicants, the honest probabilities score a Brier of <b>0.1700</b> and the squared, distorted ones score <b>0.2034</b> — nineteen per cent worse, for a change that a ranking metric cannot see at all.</p>`)}

<p>That identity is worth sitting with for a second, because it is the whole reason the Brier score can catch what AUC cannot. $\\mathbb{E}[(a-y)^2] = (a-p)^2 + p(1-p)$ says the expected squared error of any fixed prediction $a$ against a truly Bernoulli$(p)$ outcome splits into two pieces: a term $(a-p)^2$ that is exactly zero when your prediction is honest and grows the moment it is not, plus a floor $p(1-p)$ that no model can ever beat — an applicant genuinely on a coin-flip footing will sometimes default and sometimes not, and no amount of cleverness changes that. AUC only ever sees the ordering of $a$ across examples; the Brier score sees $(a-p)^2$ directly, which is precisely the piece a monotone distortion inflates without moving.</p>

${H.key('A model can have a perfect ranking and useless probabilities at the same time. AUC, KS, Gini and precision-at-a-cutoff cannot detect this, because they are built only from order. You need a second, separate diagnostic — calibration — and it is not optional just because the ranking metric looks good.')}

${H.intuition(`<p>Think of the model as a judge scoring entrants in a competition. AUC asks only: did the judge rank the true best entrant above the true worst one? A judge who gives every entrant a mark between 9.0 and 9.9 can get every single ranking exactly right while making the marks themselves meaningless — nobody watching the scoreboard can tell a 9.4 from a genuinely strong performance versus a merely decent one, because the marks have been squashed into a narrow, uninformative band. Calibration is the separate question of whether "9.4" means the same thing across every judge, every entrant, every day of the competition — whether the number, taken at face value, tells the truth. A model can be an excellent judge of order and a terrible reporter of magnitude, at the exact same time, and lending, pricing and triage decisions all consume the magnitude, not the order.</p>`)}

<h2><span class="sn">2.12.2</span> Where miscalibration actually comes from in practice</h2>

<p>The squared-score example above is a toy, chosen because the arithmetic is checkable by hand. The most common real-world source of exactly this failure mode is the one the field has studied hardest: class rebalancing. Fraud sits at 1 in 500 transactions, default at 1 in 20 accounts, and the training folklore says imbalance confuses the optimiser, so oversample the minority class, or undersample the majority, or run SMOTE to synthesise plausible minority examples, until the training set looks 50/50.</p>

<p>Van den Goorbergh, van Smeden, Timmerman and Van Calster (<i>JAMIA</i> 29(9):1525–1534, 2022; doi:10.1093/jamia/ocac093) tested random undersampling, random oversampling and SMOTE on clinical prediction models and found all three <b>yielded poorly calibrated models</b> — the probability of belonging to the minority class was strongly overestimated — while <b>none reliably improved</b> the area under the ROC curve. Put in the language just built: resampling is, in the idealised case, close to a pure monotone transform of the honest probability. It moves the numbers without improving the ranking, which is exactly the failure mode the worked example demonstrated by hand.</p>

<p>It is worth seeing precisely <i>why</i> resampling does this, because the mechanism is one clean piece of algebra, not a vague warning. Suppose the features carry exactly the same information about an applicant regardless of how you sample the training set — only the proportion of positives and negatives changes, not what a positive or a negative looks like. Bayes' rule then says the log-odds a correctly-fitted model reports is $\\ln\\frac{P(x\\mid y{=}1)}{P(x\\mid y{=}0)}$ plus the log-odds of the class prior it was trained on. Change only the prior, by resampling, and the two models' log-odds differ by exactly a constant:</p>

${H.deriv('the exact log-odds shift that resampling introduces', [
      ['$\\mathrm{logit}\\,P(y{=}1\\mid x) = \\ln\\dfrac{P(x\\mid y{=}1)}{P(x\\mid y{=}0)} + \\ln\\dfrac{r}{1-r}$', 'Bayes\' rule for a binary outcome, rewritten in log-odds form: the class-conditional evidence about $x$, plus the prior log-odds of the base rate $r$ the model was trained against. This holds for the true population rate $r_0$ and equally for any resampled rate $r_1$, because resampling only changes how often you show the optimiser a positive versus a negative — it does not touch $P(x\\mid y)$.'],
      ['$\\mathrm{logit}_{r_1}(x) - \\mathrm{logit}_{r_0}(x) = \\ln\\dfrac{r_1}{1-r_1} - \\ln\\dfrac{r_0}{1-r_0} = \\Delta$', 'Subtract the two versions of line 1. The class-conditional evidence term is identical in both, so it cancels exactly, leaving a constant $\\Delta$ that depends only on the two base rates, not on $x$.'],
      ['$\\mathrm{logit}_{r_1}(x) = \\mathrm{logit}_{r_0}(x) + \\Delta$', 'Rearranged: the resampled model\'s log-odds equal the honest model\'s log-odds plus a fixed offset, for every applicant. Adding a constant to a logit is a strictly increasing transform of the resulting probability, so this is the monotone-distortion result from §2.12.1, arrived at from a completely different direction.']
    ], 'Plug in numbers from an actual credit book: true default rate $r_0=5\\%$, oversampled to $r_1=50\\%$. Then $\\Delta = \\ln(1) - \\ln(0.05/0.95) = 2.944$ — a shift you can verify by computing $\\sigma(\\mathrm{logit}(0.05)+2.944)$ and checking it lands exactly on $0.50$. Note $e^{2.944}\\approx 19$, precisely the oversampling factor needed to inflate a 1-in-20 minority class to parity — the same number appears twice because it is the same fact stated two ways.')}

<p>That derivation earns its keep twice over. First, it explains the JAMIA finding mechanically rather than by appeal to authority: oversampling to 50/50 on a 5%-base-rate book does not corrupt the model's judgement about who is risky, it adds a fixed +2.944 to every logit, which is precisely a monotone transform and precisely why AUC barely moves while every reported probability becomes fiction. Second, it tells you the fix in advance: a shift that is <i>exactly</i> additive in log-odds is exactly what a two-parameter Platt scaler is built to undo (§2.12.3) — which is why "resample, then recalibrate" is a coherent recipe rather than two unrelated pieces of advice bolted together. In practice the shift is rarely as clean as the idealised derivation, because the resampling method itself (SMOTE's synthetic points, undersampling's information loss) also nudges the class-conditional evidence term you were hoping to hold fixed — which is exactly why the empirical result is "poorly calibrated," not "calibrated up to a single known constant."</p>

<p><b>What you are looking at.</b> Score histograms for the positive and negative class sit behind a reliability curve: predicted probability along the bottom, the observed frequency of positives in that predicted-probability bin up the side, with the dashed diagonal marking perfect calibration. The line and dots trace the model's actual reliability curve for whichever treatment is selected.</p>

<p><b>What to do with it.</b> Set the treatment to "none" and note the AUC reported alongside. Switch to "oversample to 50/50" and watch two numbers at once: AUC, which moves in the third or fourth decimal place at most, and the mean predicted probability, which jumps from something close to the true base rate to something far higher — the empirical shadow of the $\\Delta$ derived above. Then try "class weights," which reweights the loss rather than duplicating or discarding rows, and compare where its curve sits.</p>

<p><b>The thing genuinely worth noticing.</b> The reliability curve for "oversample" or "undersample" leaves the diagonal almost immediately and stays off it, while the curve for "none" or "class weights" hugs it far more closely. That gap, with the AUC readout barely moving throughout, is the JAMIA result reproduced in your browser: resampling bought you nothing on the ranking metric and cost you the probabilities.</p>

${H.lab('resample', 'What resampling does to your probabilities', 'A logistic model trained here on imbalanced data, then retrained on a resampled version. The ranking barely moves; the reliability curve leaves the diagonal immediately. Both effects are computed, not asserted.')}

<h3>The practical stance, stated conditionally</h3>

<p>If you need the probabilities themselves — credit decisions, expected loss, pricing, anything feeding the threshold arithmetic of §2.13 — <b>leave the base rate intact</b>. Use class weights if the optimiser genuinely struggles with rare positives, or better, leave the loss alone and choose the decision threshold from the actual cost matrix, which is a separate lever built for exactly this job. If you resample anyway — sometimes a vendor pipeline insists on it, sometimes a team has inherited the practice — <mark>recalibrate afterwards</mark>, and say out loud why you are doing both. If you only need a ranking — who gets reviewed first, which claims get triaged sooner — the calibration harm may be genuinely irrelevant, because nothing downstream ever reads the probability as a probability. State the objective before prescribing the fix; that conditional answer, not a blanket rule, is the one a senior person gives.</p>

<h2><span class="sn">2.12.3</span> Two ways to put the numbers back, and how to choose</h2>

<p>Calibration is a post-processing step: keep the model exactly as trained — retraining is not required and usually not desirable, since the ranking was already good — and fit a small, separate function $c$ that maps the model's raw score to a corrected probability, $c(\\text{score}) \\approx P(y{=}1\\mid \\text{score})$. Because $c$ is fit on a held-out split the model never trained on, it can only be evaluated by how well it matches reality, never by how well it flatters the model.</p>

${H.table(['Method', 'What it fits', 'Use when'], [
      ['<b>Platt scaling</b>', 'A one-dimensional logistic regression on the scores — two parameters', 'Little calibration data, or a smooth sigmoidal distortion'],
      ['<b>Isotonic regression</b>', 'Any monotone mapping', 'Thousands of held-out rows and a non-sigmoidal distortion; can overfit into steps'],
      ['<b>Beta calibration</b>', 'A three-parameter family generalising Platt', 'When Platt is too rigid but isotonic overfits'],
      ['<b>Temperature scaling</b>', 'One scalar dividing the logits', 'Neural nets, multi-class; preserves the argmax exactly']
    ])}

<p><b>Platt scaling</b> takes the model's raw score, converts it to a logit, and fits an ordinary two-parameter logistic regression — $c(\\text{score}) = \\sigma(a \\cdot \\mathrm{logit}(\\text{score}) + b)$ — against the true labels on held-out data. Two numbers, $a$ and $b$, are all it can move. That is a strength when the calibration set is small: two parameters cannot overfit much, and if the true distortion genuinely is an additive logit shift — exactly the shape the resampling derivation above produces — Platt recovers it with a handful of hundred rows. It is a weakness when the true miscalibration is not smooth and sigmoidal: a model that is honest in the middle of its range but badly off at the extremes needs more shape than two numbers can supply.</p>

<p><b>Isotonic regression</b> drops the parametric assumption entirely and fits the best-fitting <i>monotone step function</i> through the held-out data — free to bend however the data demands, constrained only to never decrease. That flexibility is exactly what a non-sigmoidal distortion needs, and exactly what makes it hungry for data: with only a few hundred held-out rows, isotonic will happily fit noise into a staircase of spurious little steps, each one confidently wrong. As a rule of thumb, isotonic wants thousands of held-out rows before its extra flexibility earns its keep over Platt's two parameters.</p>

<p><b>What you are looking at.</b> The faint red curve is the raw, miscalibrated model's reliability curve — its distortion is controlled by the "model miscalibration" slider, which raises scores to that power exactly as in the worked example above. The green curve is the same model after whichever calibrator you select is fitted on a held-out calibration split of the size you choose, then evaluated on a fresh 1,500-row test set the calibrator never saw.</p>

<p><b>What to do with it.</b> With the calibrator set to "none," push the miscalibration slider away from 1 and watch the red curve peel off the diagonal — this is the worked example from §2.12.1, running live. Switch to "Platt" and watch the curve snap back close to the diagonal with only a couple hundred calibration rows. Now switch to "isotonic" with the same small calibration set and look at the readout: the curve visually returns to the diagonal too, but the expected calibration error often does not improve as cleanly, because isotonic is fitting a staircase to sparse data.</p>

<p><b>The thing genuinely worth noticing.</b> Drag "calibration rows" down to 50 with isotonic selected. The reliability curve stops looking like a curve at all and starts looking like a jagged, overconfident staircase — a direct picture of overfitting a monotone map to noise. Switch to Platt at the same 50 rows and the line stays smooth, because two parameters simply cannot chase individual data points the way an unconstrained step function can. Notice also the "AUC (unchanged by monotone maps)" readout: it barely moves across every setting, which is the §2.12.1 derivation confirming itself on live data — calibration is doing real work that the ranking metric was structurally blind to.</p>

${H.lab('calib', 'Reliability diagrams and two calibrators', 'A deliberately miscalibrated model, then Platt and isotonic fitted on a held-out split. Watch the Brier score decompose and the curve return to the diagonal — and watch isotonic overfit when you shrink the calibration set.')}

<p>Measure the result with three complementary tools, not one. A <b>reliability diagram</b> — predicted probability on one axis, observed frequency in that bin on the other — is the visual the labs above draw; read it by eye first. The <b>Brier score</b>, the mean squared error between predicted probability and outcome, is the single number, and it decomposes cleanly into a calibration term (how far predictions sit from the diagonal) and a refinement term (how much the predictions vary at all — a model that always predicts the base rate is perfectly calibrated and completely useless, scoring badly on refinement while scoring perfectly on calibration). <b>Expected calibration error</b> averages the gap between predicted and observed frequency across bins, weighted by how many examples land in each. Always fit the calibrator on a split the model never trained on — fitting it on the training fold reintroduces exactly the optimism §2.1's train/validation/test discipline exists to prevent, and the calibrator will report a diagonal that vanishes the moment fresh data arrives.</p>

${H.practice(`<p>In a shipped scorecard, the calibrator is usually the most-forgotten artefact in the whole pipeline. It gets fitted once, at launch, on whatever calibration split existed that quarter, and then the underlying model gets retrained six months later on fresher data without anyone refitting the calibrator to match. The result is a model that is calibrated to a population that no longer exists. Treat the calibrator as a versioned artefact with the same lifecycle as the model it sits on top of — retrain one, retrain both — and monitor its own drift the way §2.18 monitors everything else.</p>`)}

<h2><span class="sn">2.12.4</span> Conformal prediction — a guarantee for one prediction, not the average</h2>

<p>Calibration, however well done, answers a question about averages: across every applicant the model calls "12%," roughly 12% default. It says nothing about any one applicant. Two people can share the same predicted probability and be nothing alike in how confidently that number was arrived at — one sits in a dense, well-populated region of feature space the model has seen thousands of times before, the other is an unusual combination the model has barely encountered — and a single calibrated point estimate reports the same "12%" for both, silently discarding the difference.</p>

<p><b>Conformal prediction</b> answers a different, more useful question: not "what is my best guess," but "give me a range wide enough that I can guarantee it contains the truth, at whatever confidence level I choose" — and it does this with no assumption about the model being correct, no assumption about the data being Gaussian, nothing beyond one mild condition on how the data was collected. That combination — a real, provable guarantee, attached to any model you already have, at essentially no extra modelling cost — is why it has moved from a niche statistical curiosity to a default recommendation for anyone who has to put a number on uncertainty and mean it.</p>

<p>The simplest version, <b>split conformal</b>, works in three steps:</p>
${H.steps([
      'Hold out a calibration set the model never trained on.',
      'Compute a nonconformity score for each held-out point — for regression $|y-\\hat y|$; for classification $1-\\hat p_{\\text{true}}$.',
      'Take the $\\lceil (n+1)(1-\\alpha)\\rceil$-th smallest score as the quantile $q$.'
    ])}
<p>Then the prediction set $\\{y : \\text{score}(x,y) \\le q\\}$ contains the truth with probability at least $1-\\alpha$ — <i>marginally</i>, over exchangeable data, <b>with no assumption about the model being right</b>. Read $\\alpha$ (the Greek letter alpha) as your allowed miscoverage rate: $\\alpha=0.10$ targets 90% coverage.</p>

${H.worked('worked conformal interval', `
<p>Regression on loss-given-default, 1,000 calibration points, target 90% coverage ($\\alpha = 0.10$). Rank the absolute residuals; take the $\\lceil 1001 \\times 0.90 \\rceil = 901$st smallest. Suppose it is £2,340.</p>
<p>Every future prediction is then reported as $\\hat y \\pm £2{,}340$, and that interval covers the truth 90% of the time <i>whatever the model is</i> — gradient boosting, a neural net, a rule. Two honest caveats to volunteer: the guarantee is <b>marginal, not conditional</b>, so intervals are the same width for easy and hard cases unless you use a normalised or Mondrian variant; and it assumes <b>exchangeability</b> — that the calibration data and the future data are draws from the same underlying process, in no particular order — which time-series drift breaks; for those, adaptive conformal methods update the quantile online.</p>`)}

<p>Why does taking the $\\lceil(n+1)(1-\\alpha)\\rceil$-th smallest residual, rather than something simpler like the $(1-\\alpha)$ percentile of a Gaussian fit, produce an exact guarantee? Because of a symmetry argument that needs nothing about the model's correctness. If the calibration point and a genuine future point are exchangeable — interchangeable in distribution, so swapping which one you call "the new point" changes nothing about the joint distribution of scores — then the future point's nonconformity score is, before you see it, equally likely to land in any of the $n+1$ possible rank positions among the $n$ calibration scores plus itself. Setting the threshold at the $\\lceil(n+1)(1-\\alpha)\\rceil$-th smallest calibration score means the future score exceeds it in at most a $\\alpha$ fraction of those equally-likely rank positions — which is exactly the coverage guarantee, derived from counting, not from any assumption about what the model computed.</p>

<p><b>What you are looking at.</b> A curved function is being predicted with a slightly-wrong model, drawn as the solid blue line. The shaded band around it, of constant width $2q$, is the conformal prediction interval computed from the calibration procedure above. Fresh test points land inside the band (drawn faint) or outside it (drawn red), and the readout tallies what fraction actually landed inside — the empirical coverage — against the promised $1-\\alpha$.</p>

<p><b>What to do with it.</b> With drift off, move $\\alpha$ across its range and watch the empirical coverage track $1-\\alpha$ closely at every setting, even though the underlying model is visibly not the true function. That tracking is the guarantee working exactly as derived — it never required the model to be right, only the calibration set to be a fair sample of what is coming.</p>

<p><b>The thing genuinely worth noticing.</b> Switch drift on. The empirical coverage collapses well below the promised level, because the calibration set and the test set are no longer exchangeable — the test distribution has moved to somewhere the calibration quantile was never computed for. Nothing about the maths announces this failure in advance; the band still gets drawn, still looks reassuring, and simply stops being true. Now leave drift off and switch on heteroskedastic noise instead: the band stays a constant width everywhere, too wide where the curve is easy to predict and too narrow where it is hard, because plain split conformal has no mechanism for admitting "this region is harder" — that is precisely the gap that normalised and Mondrian conformal variants close, by scaling the nonconformity score by a local difficulty estimate before taking the quantile.</p>

${H.lab('conformal', 'Split conformal, end to end', 'Residuals ranked, the quantile taken, the interval applied to fresh data — and then the coverage measured on that fresh data. Move α and watch the empirical coverage track the promise. Turn on drift and watch the guarantee break exactly as the theory says it should.')}

<p>Why this matters for a real decision: a calibrated point probability tells you the average is right; a conformal set tells you <b>when the model does not know</b>. In credit that is the difference between an automatic decline and a referral to manual review, and "we route wide prediction sets to a human" is a substantially stronger answer in a validation meeting than "we picked a threshold and hoped."</p>

${H.probe([
      ['Does SMOTE improve models?', 'It can raise recall at a fixed threshold, but the JAMIA 2022 evidence is that it harms calibration and rarely improves AUC. The mechanism is a near-constant shift in log-odds (derived in §2.12.2) — a monotone transform that a ranking metric cannot see — and simply moving the threshold, or recalibrating, often reaches the same operating point without the collateral damage to the probabilities.'],
      ['Platt or isotonic?', 'Platt with little calibration data or a smooth distortion, because two parameters cannot chase noise into spurious steps the way an unconstrained monotone function can; isotonic with thousands of held-out rows and a distortion that is not sigmoidal in shape. When in doubt and data is scarce, default to Platt — the failure mode of over-fitting an isotonic staircase is worse than the failure mode of a slightly-too-rigid sigmoid.'],
      ['What exactly does conformal guarantee?', 'Marginal coverage at level 1−α over exchangeable data, with no assumption that the model is correct — the guarantee comes from a counting argument over ranks, not from statistical properties of the underlying model. It breaks specifically when exchangeability breaks, which is precisely what distribution drift does, so the guarantee is only as good as the assumption that tomorrow looks like the calibration set.']
    ], 'Reporting accuracy on a 1%-positive problem. Predicting "no" always scores 99%.')}`,
    labs: {
      resample: function (host) {
        const st = Viz.controls(host, [
          { k: 'base', label: 'true positive rate in the data', min: .01, max: .3, step: .005, value: .05, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'method', label: 'treatment', type: 'buttons', value: 'over', options: [{ v: 'none', t: 'none' }, { v: 'over', t: 'oversample to 50/50' }, { v: 'under', t: 'undersample' }, { v: 'weight', t: 'class weights' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'auc', label: 'AUC (ranking)', cls: 'key' }, { k: 'brier', label: 'Brier score' },
          { k: 'mean', label: 'mean predicted p' }, { k: 'true', label: 'true base rate' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(61), n = 1500;
            const X = [], y = [];
            for (let i = 0; i < n; i++) {
              const pos = R() < st.base;
              const x = [R.normal(pos ? 1.1 : 0, 1), R.normal(pos ? .7 : 0, 1)];
              X.push(x); y.push(pos ? 1 : 0);
            }
            // build the training set according to the treatment
            let Xtr = X.slice(), ytr = y.slice(), wts = null;
            const posIdx = y.map((v, i) => v ? i : -1).filter(i => i >= 0);
            const negIdx = y.map((v, i) => v ? -1 : i).filter(i => i >= 0);
            if (st.method === 'over') {
              Xtr = []; ytr = [];
              negIdx.forEach(i => { Xtr.push(X[i]); ytr.push(0); });
              for (let i = 0; i < negIdx.length; i++) { const j = posIdx[R.int(posIdx.length)]; Xtr.push(X[j]); ytr.push(1); }
            } else if (st.method === 'under') {
              Xtr = []; ytr = [];
              posIdx.forEach(i => { Xtr.push(X[i]); ytr.push(1); });
              for (let i = 0; i < posIdx.length; i++) { const j = negIdx[R.int(negIdx.length)]; Xtr.push(X[j]); ytr.push(0); }
            }
            const m = Num.logistic(Xtr, ytr, { lr: .6 });
            m.step(400);
            const p = X.map(x => m.predict(x));
            const roc = Num.rocCurve(p, y);
            const bins = Num.calibrationBins(p, y, 10).filter(b => b.n > 8);
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1] })
              .frame({ xlabel: 'predicted probability', ylabel: 'observed frequency' });
            P.clip(() => {
              P.line([[0, 0], [1, 1]], { color: T.faint, width: 1.4, dash: [5, 4] });
              P.line(bins.map(b => [b.pred, b.obs]), { color: st.method === 'none' || st.method === 'weight' ? T.green : T.red, width: 2.6 });
              P.dots(bins.map(b => [b.pred, b.obs]), { r: 4, color: st.method === 'none' || st.method === 'weight' ? T.green : T.red, stroke: true });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText(st.method === 'over' || st.method === 'under'
              ? 'above the diagonal = over-confident on the minority class'
              : 'on the diagonal = calibrated', w - 16, 10);
            out({
              auc: roc.auc.toFixed(4), brier: Num.brier(p, y).toFixed(4),
              mean: (Num.mean(p) * 100).toFixed(1) + '%', true: (Num.mean(y) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.note(host, 'Compare "none" with "oversample": AUC moves in the third or fourth decimal — resampling did not improve the ranking — while mean predicted probability jumps from the true base rate to something far higher. That is the JAMIA finding, reproduced here in your browser.');
      },

      calib: function (host) {
        const st = Viz.controls(host, [
          { k: 'distort', label: 'model miscalibration', min: .3, max: 3, step: .05, value: 2, fmt: v => v.toFixed(2) },
          { k: 'method', label: 'calibrator', type: 'buttons', value: 'none', options: [{ v: 'none', t: 'none' }, { v: 'platt', t: 'Platt' }, { v: 'iso', t: 'isotonic' }] },
          { k: 'ncal', label: 'calibration rows', min: 50, max: 3000, step: 50, value: 800, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ece', label: 'expected calibration error', cls: 'key' }, { k: 'brier', label: 'Brier score' },
          { k: 'auc', label: 'AUC (unchanged by monotone maps)' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(83);
            const make = n => {
              const raw = [], lab = [];
              for (let i = 0; i < n; i++) {
                const pTrue = R();
                const yy = R() < pTrue ? 1 : 0;
                raw.push(Math.pow(pTrue, st.distort));     // distorted score
                lab.push(yy);
              }
              return { raw, lab };
            };
            const cal = make(st.ncal), te = make(1500);
            let f = x => x;
            if (st.method === 'platt') {
              const Xc = cal.raw.map(v => [Math.log(Math.max(1e-6, v) / Math.max(1e-6, 1 - v))]);
              const lr = Num.logistic(Xc, cal.lab, { lr: .5 }); lr.step(500);
              f = x => lr.predict([Math.log(Math.max(1e-6, x) / Math.max(1e-6, 1 - x))]);
            } else if (st.method === 'iso') {
              const pairs = cal.raw.map((v, i) => [v, cal.lab[i]]).sort((a, b) => a[0] - b[0]);
              // PAVA
              let blocks = pairs.map(p => ({ sum: p[1], n: 1, x: p[0] }));
              let changed = true;
              while (changed) {
                changed = false;
                for (let i = 0; i < blocks.length - 1; i++) {
                  if (blocks[i].sum / blocks[i].n > blocks[i + 1].sum / blocks[i + 1].n) {
                    blocks[i] = { sum: blocks[i].sum + blocks[i + 1].sum, n: blocks[i].n + blocks[i + 1].n, x: blocks[i + 1].x };
                    blocks.splice(i + 1, 1); changed = true; break;
                  }
                }
              }
              f = x => {
                for (let i = 0; i < blocks.length; i++) if (x <= blocks[i].x) return blocks[i].sum / blocks[i].n;
                return blocks[blocks.length - 1].sum / blocks[blocks.length - 1].n;
              };
            }
            const p = te.raw.map(f);
            const bins = Num.calibrationBins(p, te.lab, 12).filter(b => b.n > 5);
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1] })
              .frame({ xlabel: 'predicted probability', ylabel: 'observed frequency' });
            P.clip(() => {
              P.line([[0, 0], [1, 1]], { color: T.faint, width: 1.4, dash: [5, 4] });
              const rawBins = Num.calibrationBins(te.raw, te.lab, 12).filter(b => b.n > 5);
              P.line(rawBins.map(b => [b.pred, b.obs]), { color: T.red, width: 1.6, alpha: .55 });
              P.line(bins.map(b => [b.pred, b.obs]), { color: T.green, width: 2.6 });
              P.dots(bins.map(b => [b.pred, b.obs]), { r: 4, color: T.green, stroke: true });
            });
            const ece = Num.sum(bins.map(b => (b.n / te.lab.length) * Math.abs(b.pred - b.obs)));
            out({
              ece: ece.toFixed(4), brier: Num.brier(p, te.lab).toFixed(4),
              auc: Num.rocCurve(p, te.lab).auc.toFixed(4)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().red, t: 'uncalibrated model' }, { c: Viz.theme().green, t: 'after the chosen calibrator' }]);
        Viz.note(host, 'AUC is invariant to any monotone transformation, so calibration never changes the ranking — it changes what the number <i>means</i>. Shrink the calibration set to 50 rows with isotonic selected and watch it produce a step function that fits noise; Platt, with two parameters, does not.');
      },

      conformal: function (host) {
        const st = Viz.controls(host, [
          { k: 'alpha', label: 'miscoverage α (target = 1−α)', min: .01, max: .3, step: .01, value: .1, fmt: v => (100 * (1 - v)).toFixed(0) + '% coverage' },
          { k: 'ncal', label: 'calibration points', min: 50, max: 2000, step: 50, value: 1000, fmt: v => v.toLocaleString() },
          { k: 'drift', label: 'break exchangeability (drift)', type: 'toggle', value: false },
          { k: 'hetero', label: 'heteroskedastic noise', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'q', label: 'quantile q', cls: 'key' }, { k: 'idx', label: 'rank taken' },
          { k: 'cov', label: 'empirical coverage on fresh data' }, { k: 'width', label: 'interval width' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(11);
            const f = x => 1.4 * Math.sin(x) + .5 * x;
            const noise = x => st.hetero ? .18 + .38 * Math.abs(x) : .55;
            // model = slightly wrong version of f
            const model = x => f(x) * .92 + .12;
            const cal = [];
            for (let i = 0; i < st.ncal; i++) { const x = -3 + 6 * R(); const yv = f(x) + R.normal(0, noise(x)); cal.push({ x: x, y: yv, s: Math.abs(yv - model(x)) }); }
            const scores = cal.map(c => c.s).sort((a, b) => a - b);
            const k = Math.ceil((st.ncal + 1) * (1 - st.alpha));
            const q = scores[Math.min(scores.length - 1, k - 1)];
            // fresh data (optionally drifted)
            const test = [];
            for (let i = 0; i < 260; i++) {
              const x = -3 + 6 * R();
              const shift = st.drift ? .55 : 0;
              const yv = f(x) + shift + R.normal(0, noise(x) * (st.drift ? 1.5 : 1));
              test.push({ x: x, y: yv });
            }
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-4, 4.5] }).frame({ xlabel: 'x', ylabel: 'y' });
            P.clip(() => {
              const band = [];
              for (let x = -3.2; x <= 3.21; x += .05) band.push([x, model(x) + q]);
              const band2 = [];
              for (let x = 3.2; x >= -3.21; x -= .05) band2.push([x, model(x) - q]);
              ctx.fillStyle = T.blue; ctx.globalAlpha = .13;
              ctx.beginPath();
              band.concat(band2).forEach((p, i) => { const X = P.x(p[0]), Y = P.y(p[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
              ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
              P.fn(model, { color: T.blue, width: 2.2 });
              test.forEach(t => {
                const covered = Math.abs(t.y - model(t.x)) <= q;
                P.dots([[t.x, t.y]], { r: 2.8, color: covered ? T.faint : T.red, alpha: covered ? .7 : 1 });
              });
            });
            const cov = Num.mean(test.map(t => Math.abs(t.y - model(t.x)) <= q ? 1 : 0));
            out({
              q: q.toFixed(3), idx: k + ' of ' + st.ncal,
              cov: (cov * 100).toFixed(1) + '%', width: (2 * q).toFixed(3)
            });
          }
        });
        Viz.note(host, 'With drift off, empirical coverage tracks 1−α closely — no assumption about the model was needed. Turn drift on and coverage collapses: exchangeability is the one thing conformal genuinely requires, and time-series data violates it routinely. Turn on heteroskedastic noise to see the other caveat: the band is a constant width, so it is too wide where the problem is easy and too narrow where it is hard, which is what normalised/Mondrian conformal fixes.');
      }
    },
    quiz: [
      {
        q: 'You apply SMOTE and your model’s mean predicted probability jumps from 5% to 43%. What happened?',
        options: ['The model got better at finding positives', 'The base rate the model believes in changed, so the probabilities are now wrong', 'The AUC must have improved', 'The features were rescaled'],
        answer: 1,
        why: 'Resampling toward 50/50 shifts every logit by a near-constant amount — Bayes\' rule says the class prior enters as an additive log-odds term, and only the prior changed. Adding a constant to every logit is a monotone transform, which is exactly why AUC barely moves while the reported probabilities stop meaning anything: ranking survives monotone transforms by construction, calibration does not. The tempting wrong answer, "the model got better," mistakes a shift in the number for a change in the underlying judgement — the model still separates risky from safe applicants exactly as well as before.'
      },
      {
        q: 'Split conformal with 1,000 calibration points and α = 0.10 takes which residual?',
        options: ['the 900th smallest', 'the ⌈1001 × 0.90⌉ = 901st smallest', 'the mean residual × 1.645', 'the 90th percentile of the training residuals'],
        answer: 1,
        why: 'The order statistic is $(n+1)$, not $n$, because the guarantee comes from an exchangeability argument that treats the future point as one of $n+1$ equally-likely draws — the calibration points plus the one you have not seen yet. Rounding up, rather than down, is what keeps the coverage promise at least $1-\\alpha$ rather than merely close to it. Using training residuals, the tempting shortcut when you already have them lying around, destroys the guarantee entirely: the model has seen those points, so its errors on them are optimistic by exactly the amount §2.1\'s train/test discipline exists to prevent.'
      },
      {
        q: 'Conformal prediction’s coverage guarantee requires…',
        options: ['a correctly specified model', 'Gaussian residuals', 'exchangeable data', 'a large training set'],
        answer: 2,
        why: 'The guarantee is a counting argument over ranks, not a statistical claim about the model or the residual distribution — it holds for a terrible model exactly as tightly as for a good one, which is the whole point of a distribution-free, model-agnostic method. Exchangeability — that calibration data and future data come from the same process, order irrelevant — is the one thing it cannot do without, and it is exactly what a drifting population violates, which is why the guarantee quietly stops holding under time-series drift without any error message telling you so.'
      }
    ],
    cards: [
      { q: 'Ranking vs calibration', a: 'AUC/KS/Gini survive any monotone transform of the scores, so a perfect ranking can sit on top of useless probabilities. Calibration is a separate, necessary check.' },
      { q: 'Resampling and calibration', a: 'SMOTE/over/under-sampling shift every logit by a near-constant amount (a monotone transform) and harm calibration (JAMIA 2022) without reliably improving AUC. Recalibrate, or move the threshold instead.' },
      { q: 'Platt vs isotonic', a: 'Platt: 2 parameters, little data, smooth distortion. Isotonic: any monotone map, needs thousands of rows, can overfit into steps.' },
      { q: 'Split conformal in three steps', a: 'Hold out a calibration set → score nonconformity → take the ⌈(n+1)(1−α)⌉-th smallest as q; the set {y: score ≤ q} covers at 1−α marginally, whatever the model.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.13 */
  ML.section({
    id: 'metrics', track: 'classical', num: '2.13',
    title: 'Evaluation metrics in depth',
    lede: 'The vocabulary that turns §2.1’s "the metric is what you report" into an actual number, with §1.6’s interval logic behind how much to trust it. Feeds §5.7’s cost-per-successful-task thinking.',
    html: `
<p>§2.1 drew a line between the loss you optimise and the metric you report, and promised the difference was not academic. This section is that promise redeemed. A model trained on logloss is, underneath, always answering the same narrow question — how well-calibrated is my probability, averaged over everyone — because that is the only question logloss knows how to ask. The business rarely asks that question. It asks "of the accounts we flag, how many are actually going to default," or "if we can only review one in twenty applications by hand, which twenty should it be," or "what does a missed fraud case actually cost us compared to a false alarm." None of those are logloss. Each one is a different metric, built to be read by a human making a decision rather than an optimiser taking a gradient step, and the whole discipline of this section is choosing the metric that actually answers the question being asked — and knowing that a single number, almost never, answers all of them at once.</p>

<h2><span class="sn">2.13.1</span> Everything starts from the confusion matrix</h2>

<p>A classifier at a fixed threshold produces exactly four counts, and every metric in this section is built from some ratio of them. <b>True positives</b> ($TP$) are the ones you flagged that really were positive; <b>false positives</b> ($FP$) are the ones you flagged that were not; <b>false negatives</b> ($FN$) are real positives you missed; <b>true negatives</b> ($TN$) are the negatives you correctly ignored. Two ratios matter more than the rest:</p>

$$\\text{precision} = \\frac{TP}{TP+FP}, \\qquad \\text{recall} = \\frac{TP}{TP+FN}$$

<p>Read precision as "of everyone I flagged, how many were genuinely positive" — it is a question about the quality of your alarm. Read recall as "of everyone who was genuinely positive, how many did I actually catch" — a question about the completeness of your alarm. Notice the two questions share a numerator, $TP$, but divide by entirely different populations: precision divides by what you predicted, recall divides by what was true. That is not a quirk of notation; it is why the two numbers can move in opposite directions from the same model. Lower the threshold and you flag more of everything — recall can only go up or stay flat, because you never un-flag a true positive you were already catching, while precision typically falls, because the newly-flagged examples are, on average, less confidently positive than the ones you were already catching.</p>

<p><b>F1</b> combines them into one number, but not by averaging:</p>

$$F_1 = \\frac{2 \\cdot \\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}$$

<p>This is the <b>harmonic mean</b>, and the choice is deliberate, not cosmetic. An ordinary arithmetic mean of 0.9 and 0.1 is 0.5 — it treats a model that is excellent on one axis and worthless on the other as "middling." The harmonic mean of the same two numbers is $2(0.9)(0.1)/(0.9+0.1) = 0.18$, close to the smaller number, because the harmonic mean is dominated by whichever input is worse. That is exactly the property you want from a single combined score: a fraud model that catches everything by flagging every transaction has recall 1.0 and precision near zero, and F1 correctly refuses to reward it, where a plain average would flatter it at 0.5.</p>

${H.worked('worked confusion matrix under imbalance', `
<p>10,000 cases, 100 of them positive (1%). The model flags 200 and catches 80 of the positives.</p>
${H.table(['', 'ACTUAL +', 'ACTUAL −'], [
      ['<b>PRED +</b>', '<b>TP = 80</b>', 'FP = 120'],
      ['<b>PRED −</b>', 'FN = 20', 'TN = 9,780']
    ])}
<p>Accuracy $=(80+9780)/10000 =$ <b>98.6%</b> — and completely uninformative, because predicting "negative" for everyone scores 99% on this population without catching a single positive. Precision $= 80/200 =$ <b>40%</b>. Recall $= 80/100 =$ <b>80%</b>. F1 $= 2(0.4)(0.8)/1.2 =$ <b>0.533</b>. Three of every five investigations opened on this model's say-so are wasted; that is the number the business cares about, and accuracy hides it entirely.</p>`)}

${H.key('Accuracy on an imbalanced problem is not a weak metric — it is close to meaningless. On a 1%-positive problem, predicting "no" for everyone scores 99% and catches nothing. Never quote accuracy without quoting the base rate next to it.')}

<p><b>What you are looking at.</b> The left panel shows two overlapping histograms of the model's raw score — blue for the negative class, red for the positive one — with a vertical line marking wherever the threshold slider currently sits. The right panel is the live confusion matrix built from that same threshold, colour-coded by cell, with a small ROC curve underneath marking exactly where this threshold's false-positive and true-positive rate land on it.</p>

<p><b>What to do with it.</b> Drag the threshold slider from one end to the other and read the five numbers in the readout as they move. Notice recall only ever increases as you lower the threshold — the model catches strictly more of the positives it was already catching, never fewer — while precision generally falls, because the additional flagged examples come from further out in the tail of the score distribution where the two classes overlap most. Watch the ROC-AUC readout throughout: it never moves, because nothing about the model or its ranking has changed, only where you sliced it.</p>

<p><b>The thing genuinely worth noticing.</b> Press the button that reproduces the 1%-positive worked example above, then hunt for the threshold that maximises accuracy. You will find it near the high end of the slider, where the model flags almost nothing — because with positives this rare, refusing to flag anyone is close to optimal for accuracy specifically, and yet it is the single worst possible threshold for actually catching fraud. That is the confusion matrix worked example, reproduced by your own hand on the slider rather than read off a table.</p>

${H.lab('confusion', 'The threshold is the model’s real dial', 'One trained model, one slider. Watch precision, recall, F1, the confusion matrix, and the position on both curves move together. Nothing about the model changes — only where you cut it.')}

<h2><span class="sn">2.13.2</span> ROC and PR are the same information, weighted differently</h2>

<p>Both curves are built by sweeping the threshold from one extreme to the other and plotting two of the four confusion-matrix ratios against each other at every setting. The <b>ROC curve</b> plots the true positive rate (recall) against the <b>false positive rate</b>, $FP/(FP+TN)$ — of everyone genuinely negative, what fraction did you wrongly flag. <b>ROC-AUC</b> is the area under that curve, and it has a clean probabilistic reading: it is exactly the probability that a randomly chosen positive example scores higher than a randomly chosen negative one. Because that statement never mentions how many positives or negatives there are — only how they rank against each other — ROC-AUC is <b>insensitive to the base rate</b>. That is a genuine virtue when you are comparing two models' underlying discriminative ability, and a genuine trap when you are asking whether either model is fit to deploy, because "insensitive to the base rate" also means "insensitive to exactly the thing that makes a rare-event problem hard."</p>

<p>Here is where that trap bites, using the worked confusion matrix again rather than a fresh abstraction. At the threshold that gave $TP=80$, $FP=120$: the false positive rate is $FP/(FP+TN) = 120/9{,}900 \\approx 1.2\\%$ — reassuringly tiny, because it is being divided by the 9,900 true negatives, an enormous denominator that any absolute number of false positives gets lost inside. Precision at the identical threshold, dividing the same 120 false positives by only 200 total flags, is 40% — three in five flagged cases are wrong. Same model, same threshold, same 120 mistaken flags; one framing calls it a rounding error, the other calls it a serious operational cost. The false positive rate is honest about proportions of the negative class; precision is honest about the burden on whoever has to act on every flag.</p>

${H.deriv('why ROC stays flattering while PR degrades as positives get rarer', [
      ['$\\text{FPR} = \\dfrac{FP}{FP+TN} = \\dfrac{FP}{N}$', 'The false positive rate divides by $N$, the total count of negatives — a number that grows every time the positive rate shrinks and the dataset size is held fixed, since almost everyone added is a negative.'],
      ['$\\text{precision} = \\dfrac{TP}{TP+FP}$', 'Precision divides by the total number flagged, which is a mix of real positives and mistakes, in a ratio that gets worse — more mistakes per real catch — exactly as positives get rarer, because there are fewer real positives available to flag correctly in the first place.'],
      ['Hold the classifier\'s FPR and TPR fixed, and shrink the true base rate $\\pi = P/(P+N)$ toward zero.', 'A fixed classifier quality (fixed TPR and FPR) is the realistic case: you are not changing the model, only asking how it behaves as the problem becomes rarer, which is exactly what happens moving from a 50%-positive toy dataset to a 1%-positive fraud book.'],
      ['$\\text{precision} = \\dfrac{TPR\\cdot\\pi}{TPR\\cdot\\pi + FPR\\cdot(1-\\pi)} \\to 0 \\text{ as } \\pi \\to 0$', 'Rewrite precision in terms of the base rate $\\pi$ using $TP=TPR\\cdot P$ and $FP=FPR\\cdot N$. As $\\pi\\to0$, the denominator is dominated by the $FPR\\cdot(1-\\pi)$ term — an ocean of negatives contributing a trickle of false positives that nonetheless swamps the vanishing pool of true positives — so precision collapses toward zero even while FPR itself, and therefore the whole ROC curve, has not moved at all.']
    ], 'This is the entire mechanism behind "0.95 AUC, useless model": ROC-AUC is built from a ratio, FPR, that is structurally forgiving of a huge negative class, while PR-AUC is built from precision, a ratio that is not. Both curves are drawn from the same underlying confusion matrices; they simply weight the same false positives completely differently.')}

<p><b>What you are looking at.</b> Two panels built from the identical scored population: the ROC curve on the left, precision-recall on the right, with the readout reporting ROC-AUC, PR-AUC (sometimes called average precision), Gini and KS for whatever positive rate and model quality you have set.</p>

<p><b>What to do with it.</b> Hold model quality fixed and drag the positive-rate slider down toward its minimum. Watch the ROC curve on the left barely change shape at all, while the PR curve on the right visibly sags and its area shrinks — the derivation above, happening in front of you rather than on paper.</p>

<p><b>The thing genuinely worth noticing.</b> At the lowest positive rate the lab allows, read both AUC numbers side by side: it is entirely possible to see a ROC-AUC above 0.90 sitting next to a PR-AUC well under 0.5, on the exact same model. Neither number is wrong. They are answering different questions, and under heavy imbalance only one of those questions — precision, and by extension PR-AUC — describes what a human reviewing the flagged cases will actually experience.</p>

${H.lab('roc', 'Same model, two stories', 'ROC and PR side by side on data whose imbalance you control. Push the positive rate to 1% and watch ROC stay beautiful while PR collapses — the single clearest demonstration of why a 0.95 AUC can be useless.')}

${H.key('Under heavy class imbalance, report both ROC-AUC and PR-AUC, and operate on PR. ROC-AUC compares model quality; PR-AUC tells you whether the model is deployable at all.')}

<h3>Credit-specific summaries</h3>
<p><b>Gini</b> $= 2\\cdot\\mathrm{AUC}-1$ is a simple rescaling — it maps AUC's range of $[0.5, 1]$ for a model better than chance onto $[0, 1]$, so AUC 0.75 is Gini 0.50 — and it carries zero new information beyond AUC. It survives purely as the unit a credit committee is used to reading, the same way a distance is not more true when quoted in kilometres rather than miles, but the room will ask for it in one unit specifically. <b>KS</b> (Kolmogorov–Smirnov) takes a different cut through the same scored population: sort every good and every bad customer by score, plot the cumulative share of goods and the cumulative share of bads as score decreases, and KS is the single largest vertical gap between those two cumulative curves — the score cut-off where the model best tells goods and bads apart, expressed as one number. It is, structurally, the same computation the ROC curve's diagonal-distance sometimes gets compared to, but read directly off cumulative distributions rather than off a plotted curve, which is why a risk team that has never heard of AUC will still recognise KS instantly. <b>Brier score</b>, built in §2.12, is the odd one out in this list: it is the only metric here that is sensitive to calibration rather than ranking, which is exactly why it belongs alongside Gini and KS rather than instead of them — the three together answer "can it separate," "can it separate," in two dialects, and "are the numbers honest," which neither dialect of separation can ever tell you.</p>

<h2><span class="sn">2.13.3</span> The threshold is not 0.5 — it is a business decision</h2>

<p>Every classifier you have met so far outputs a probability, and something downstream converts that probability into a yes/no action by comparing it to a threshold. The default instinct is to use 0.5 — approve if the model says less than 50% likely to default — and that instinct is worth interrogating, because 0.5 encodes an assumption that is almost never true: that being wrong in one direction costs exactly the same as being wrong in the other. It rarely does.</p>

<p>The right threshold falls straight out of comparing two expected costs. Approving an applicant who turns out to default costs $C_{FN}$ (a missed default — the model said "safe," reality said "no"); declining an applicant who would have repaid costs $C_{FP}$ (forgone margin on a good customer you turned away). For an applicant with model-estimated default probability $p$, approving is the better decision exactly when its expected cost is lower than declining's:</p>

${H.deriv('the cost-optimal threshold, from indifference between two actions', [
      ['expected cost of approving $= p \\cdot C_{FN}$', 'With probability $p$ the applicant defaults and you lose $C_{FN}$; with probability $1-p$ they repay and approving cost you nothing extra.'],
      ['expected cost of declining $= (1-p) \\cdot C_{FP}$', 'With probability $1-p$ the applicant was actually good and you have forgone $C_{FP}$ of margin by declining them; with probability $p$ they were a defaulter you correctly avoided, at no cost.'],
      ['Approve when $p \\cdot C_{FN} < (1-p)\\cdot C_{FP}$', 'Choose whichever action has the lower expected cost — the ordinary decision-theoretic rule, nothing specific to credit about this step.'],
      ['$p\\,(C_{FN}+C_{FP}) < C_{FP} \\;\\Longrightarrow\\; p < \\dfrac{C_{FP}}{C_{FP}+C_{FN}}$', 'Expand the bracket, collect every term with $p$ onto one side, and divide through by the positive quantity $C_{FN}+C_{FP}$. The inequality direction is preserved because that quantity is positive by definition — both costs are losses.'],
      ['$p^* = \\dfrac{C_{FP}}{C_{FP}+C_{FN}}$', 'The boundary of the inequality is the threshold itself: approve below $p^*$, decline above it. Notice $p^*$ depends only on the ratio of the two costs, never on the model — the model supplies $p$, the business supplies $p^*$, and the two are independent pieces of the decision.']
    ], 'The formula is symmetric in a way worth internalising: if declining a good customer is cheap relative to approving a bad one, $C_{FP}\\ll C_{FN}$, then $p^*$ is small and the model is made cautious; if it is the other way round, $p^*$ climbs toward 1 and the model approves almost everyone. A threshold of exactly 0.5 is the special case $C_{FP}=C_{FN}$, which is a claim about your business, not a property of the mathematics, and it is rarely the claim your business actually wants to make.')}

${H.worked('worked number — the threshold is a business decision', `
<p>Approving a customer who defaults costs £900 of loss. Declining a customer who would have been good costs £60 of forgone margin.</p>
$$p^* = \\frac{C_{FP}}{C_{FP}+C_{FN}} = \\frac{60}{60+900} = \\frac{60}{960} = 0.0625$$
<p>So the operating threshold is <b>6.25%</b>, a long way from 0.5. Nothing about the model changed; the business asymmetry chose the cut. Use 0.5 and you would approve everyone up to a 50% default probability, which given these costs would be catastrophically expensive — which is why <mark>"threshold 0.5" is almost never the right answer to a cost-sensitive problem</mark>, and why §2.12's insistence on calibrated probabilities matters here specifically: this formula computes a cut on the value of $p$ itself, so if $p$ is inflated or deflated by the kind of monotone distortion §2.12 describes, the threshold you compute is a cut at the wrong <i>true</i> probability even though the arithmetic above is flawless.</p>
<p>The senior addition: costs are rarely constant across applicants. Loss given default scales with exposure — a defaulted £50,000 mortgage and a defaulted £500 overdraft are not the same $C_{FN}$ — so the decision is really a per-application expected-value calculation, and the "threshold" becomes a surface across exposure and probability rather than a single number quoted in a policy document.</p>`)}

<p><b>What you are looking at.</b> The curve plots expected cost per application, in pounds, against every possible decision threshold from 0.01 to 0.99. The green dot marks the threshold that empirically minimises that curve on the scored population; the red dot marks the cost of using 0.5 regardless.</p>

<p><b>What to do with it.</b> Set the two cost sliders to the worked example's £900 and £60 and confirm the green dot sits at roughly 0.0625 — the empirical minimum tracking the formula derived above exactly, because the underlying scores in this lab are genuinely calibrated. Then push the costs toward equal and watch the green dot slide back toward 0.5, confirming that 0.5 is not wrong in general — it is exactly right in the one special case the formula predicts.</p>

<p><b>The thing genuinely worth noticing.</b> Read the "money left on the table" readout at the default £900/£60 split: it is the gap between the red and green dots' heights, multiplied out per application. Multiply that per-application figure by a realistic monthly application volume in your head, and the abstract "threshold 0.5 is often wrong" becomes a concrete number a finance director would recognise as real money — which is precisely the pitch that gets a cost-sensitive threshold shipped instead of quietly overridden back to 0.5 by whoever inherits the model next.</p>

${H.lab('cost', 'Cost-optimal threshold calculator', 'Set the two costs and the model’s scores; the optimal cut, the expected cost curve, and the money left on the table by using 0.5 are all computed.')}

<h2><span class="sn">2.13.4</span> Lift and gains — the table a business actually reads</h2>

<p>ROC and PR curves are the right tool for comparing models and choosing thresholds; they are close to unreadable in a room full of people who do not think in true-positive rates. <b>Lift and cumulative gains</b> are the same underlying information, translated into units a credit committee already speaks fluently: rank the whole population by score from riskiest to safest, cut it into ten equal-sized deciles, and ask what share of every actual bad account each decile contains.</p>

${H.worked('a concrete lift table, computed end to end', `
<p>10,000 accounts, 500 genuine bads (5% base rate), ranked by score and cut into deciles. Suppose the bad counts per decile, riskiest first, come out as follows — they must sum to 500, and they do:</p>
${H.table(['decile (1 = riskiest)', 'bads in decile', 'share of all bads', 'cumulative share'], [
      ['1', '210', '42.0%', '42.0%'],
      ['2', '120', '24.0%', '66.0%'],
      ['3', '70', '14.0%', '80.0%'],
      ['4', '40', '8.0%', '88.0%'],
      ['5', '25', '5.0%', '93.0%'],
      ['6–10', '35', '7.0%', '100.0%']
    ])}
<p><b>Lift</b> for the worst decile compares its share of bads to the 10% share a random decile would get by construction: $\\mathrm{lift} = 42.0\\% / 10\\% = 4.2\\times$ — the riskiest tenth of the book carries 4.2 times its fair share of the losses. <b>Cumulative gains</b> at 20% of the population (the top two deciles) is 66.0%: review the riskiest fifth of applicants and you catch two-thirds of the losses in the whole book. Both numbers are read directly off the cumulative-share column, and both are exactly the information an ROC or PR curve carries, in units of "accounts reviewed" and "losses caught" rather than "false positive rate" and "true positive rate."</p>`)}

<p>The relabelling is not a coincidence, and it is worth being able to say why out loud. A cumulative gains curve — cumulative share of bads caught on the y-axis, cumulative share of the population reviewed on the x-axis — is, point for point, the same curve as an ROC curve with true positive rate and a rescaled x-axis: reviewing the top $k\\%$ of the population by score is exactly the operating point where you are catching $TPR$ at whatever $FPR$ that cut-off happens to produce, and the ROC curve's own x-axis is a monotone reparameterisation of "population reviewed." Learn to move fluently between the two representations, because the ROC curve is what you compute and validate against, and the gains table is what actually gets presented — "review the riskiest fifth and catch two-thirds of the losses" and "AUC 0.78" are, quite literally, the same sentence spoken in two different rooms.</p>

<p><b>What you are looking at.</b> The bar chart on the left shows each decile's share of all bads, with the top two deciles picked out in a different colour and a dashed line at 10% marking what a random, scoreless decile would carry. The line on the right is the cumulative gains curve built from the same deciles, against the diagonal a random ranking would produce.</p>

<p><b>What to do with it.</b> Drag model quality down toward its minimum and watch the decile bars flatten toward the 10% dashed line — a model with no real signal cannot concentrate bads into its riskiest decile any better than a coin flip could, and the gains curve correspondingly collapses onto the diagonal.</p>

<p><b>The thing genuinely worth noticing.</b> Push model quality back up to its maximum and compare the shape of the gains curve here to the ROC curve in the previous lab, on the same underlying scores. They bow in exactly the same way, because they are the same curve — the exercise is not abstract once you have watched both labs respond identically to the same slider.</p>

${H.lab('lift', 'Deciles, lift and cumulative gains', 'The same scores as the ROC lab, in the business’s units. The table updates with the model quality slider — and the gains curve is exactly the ROC curve with the axes relabelled.')}

<h2><span class="sn">2.13.5</span> For regression targets, the same choice recurs in a different shape</h2>

<p>Once the target is a number rather than a class, the loss-versus-metric question does not go away; it changes what "punishing errors differently" means. <b>RMSE</b>, the square root of mean squared error, penalises large errors quadratically — an error twice as big counts four times as much — and is the right choice exactly when large misses are disproportionately expensive, which in lending they usually are: missing a £50,000 default is not "ten times as bad" as missing a £5,000 one, it can be existentially worse for the book. <b>MAE</b>, mean absolute error, treats every pound of error identically regardless of size, which makes it far more robust to the occasional huge outlier and a more honest summary of "typical" error, at the cost of not caring specially about the rare catastrophic miss.</p>

${H.worked('the same five errors, two metrics, two different verdicts', `
<p>Five predictions with absolute errors of £1, £1, £1, £1 and £10 — four small misses and one large one.</p>
<p>$\\mathrm{MAE} = (1+1+1+1+10)/5 = 14/5 = \\mathbf{£2.80}$. The outlier contributes its own value, £10, and nothing more.</p>
<p>$\\mathrm{RMSE} = \\sqrt{(1^2+1^2+1^2+1^2+10^2)/5} = \\sqrt{104/5} = \\sqrt{20.8} \\approx \\mathbf{£4.56}$. The outlier contributes its <i>square</i>, 100, which is 96% of the total inside the square root — one error out of five is responsible for almost the entire RMSE figure.</p>
<p>Same five predictions, and RMSE reports a number 63% higher than MAE, purely because one of the five errors was ten times the others. Neither number is wrong; they are answering "what is typical" and "how bad can it get, weighted by size" respectively, and quoting only one of them hides which question you answered.</p>`)}

<p><b>MAPE</b>, mean absolute percentage error, rescales every error by the size of the actual value, which makes it interpretable across wildly different magnitudes — "12% off" means the same thing for a £500 balance and a £5,000,000 exposure, where a raw pound figure would not. It has two sharp edges that are easy to forget. First, it explodes near zero: an actual value of £1 with a £5 error is a "400% error" ($|5-1|/1$), a nearly meaningless statement about anything close to a true zero. Second, it is <b>asymmetric</b>, and in the direction that surprises people: an <i>under</i>-prediction is capped at 100% error, because a non-negative forecast cannot go below zero — predict £0 against a true £5 and $|0-5|/5=100\\%$ is the worst under-prediction can ever score. An <i>over</i>-prediction has no such ceiling: predict £50 against the same true £5 and the error is $|50-5|/5 = 900\\%$, climbing without limit as the forecast grows. A forecasting process optimised, or even just evaluated carelessly, against MAPE therefore has a quiet incentive to shade predictions <i>down</i>, because under-prediction's downside is bounded at 100% while over-prediction's is not — which is exactly why MAPE quietly biases forecasts low, and that bias compounds silently across thousands of forecasts.</p>

<p><b>$R^2$</b> reports the fraction of variance in the target your model explains relative to a baseline — and that word, <i>relative</i>, is the entire content of the number. $R^2=0$ does not mean "the model explains nothing"; it means "the model does exactly as well as always predicting the mean," which for a wildly volatile target might already be a hard baseline to beat, and for a nearly-constant target might be a trivial one. Always name the baseline out loud when you quote $R^2$, because the number is meaningless without it.</p>

<p>And for skewed monetary targets specifically — loss given default, claim severity, anything bounded at zero with a long right tail — squared error implicitly assumes something Gaussian-shaped underneath it (§1.2), which a monetary loss almost never is. Evaluate in log space, or use a loss built for the actual shape of the data, such as a Tweedie or gamma deviance, rather than forcing an error metric onto a distribution it was never built to describe.</p>

${H.probe([
      ['AUC is 0.95 and the model is useless — how?', 'With 1% positives, the false-positive rate that drives ROC is divided by a huge negative-class denominator, so thousands of false positives barely register. Precision at the operating threshold, and PR-AUC generally, divide by the much smaller number actually flagged and can both be poor while ROC looks superb — the mechanism is derived in §2.13.2, not asserted.'],
      ['Gini vs AUC?', 'Gini = 2·AUC − 1: a linear rescaling that carries zero additional information. It survives purely because credit committees are trained to read it — quoting it costs nothing and buys fluency in the room.'],
      ['Which single metric would you take to a committee?', 'None alone. Separation (KS or Gini) answers "can the model rank," calibration (Brier, §2.12) answers "are the probabilities honest," and cost at the chosen threshold (§2.13.3) answers "what does deploying this actually cost." A single AUC number answers none of the three questions a committee is actually asking.']
    ], 'Reporting accuracy on a 1%-positive problem — predicting "no" always scores 99%.')}`,
    labs: {
      confusion: function (host) {
        const R = Num.rng(7);
        const N = 4000;
        let scores = [], labels = [];
        function gen(sep, base) {
          scores = []; labels = [];
          for (let i = 0; i < N; i++) {
            const pos = R() < base;
            labels.push(pos ? 1 : 0);
            scores.push(Num.sigmoid(R.normal(pos ? sep : -sep, 1.4)));
          }
        }
        const st = Viz.controls(host, [
          { k: 'thr', label: 'decision threshold', min: .01, max: .99, step: .01, value: .5, fmt: v => v.toFixed(2) },
          { k: 'sep', label: 'model quality (class separation)', min: .2, max: 3, step: .05, value: 1.3, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'positive rate', min: .01, max: .5, step: .01, value: .12, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => { gen(st.sep, st.base); S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'prec', label: 'precision', cls: 'key' }, { k: 'rec', label: 'recall' }, { k: 'f1', label: 'F1' },
          { k: 'acc', label: 'accuracy' }, { k: 'auc', label: 'ROC-AUC' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            if (!scores.length) gen(st.sep, st.base);
            const c = Num.confusion(scores, labels, st.thr);
            const roc = Num.rocCurve(scores, labels);
            // left: score distributions
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 40, r: Math.max(180, w * .42), t: 14, b: 34 } });
            const hp = Num.hist(scores.filter((_, i) => labels[i]), 28, 0, 1);
            const hn = Num.hist(scores.filter((_, i) => !labels[i]), 28, 0, 1);
            const mx = Math.max(Math.max.apply(null, hp.bins) / Math.max(1, hp.bins.reduce((a, b) => a + b, 0)),
                                Math.max.apply(null, hn.bins) / Math.max(1, hn.bins.reduce((a, b) => a + b, 0)));
            const P2 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, mx * 1.15], pad: { l: 40, r: Math.max(180, w * .42), t: 14, b: 34 } })
              .frame({ xlabel: 'model score', ylabel: 'share' });
            P2.clip(() => {
              const drawH = (hh, col) => {
                const tot = hh.bins.reduce((a, b) => a + b, 0) || 1;
                hh.centers.forEach((cc, i) => {
                  ctx.fillStyle = col; ctx.globalAlpha = .45;
                  const x0 = P2.x(cc - hh.w / 2), x1 = P2.x(cc + hh.w / 2);
                  ctx.fillRect(x0, P2.y(hh.bins[i] / tot), Math.max(1, x1 - x0 - 1), P2.y(0) - P2.y(hh.bins[i] / tot));
                  ctx.globalAlpha = 1;
                });
              };
              drawH(hn, T.blue); drawH(hp, T.red);
              P2.vline(st.thr, { color: T.text, width: 2, dash: false, label: 'threshold' });
            });
            // right: confusion matrix
            const rx = w - Math.max(170, w * .40) + 10, ry = 30, cw = Math.min(78, (w - rx - 20) / 2), ch = 44;
            const cells = [['TP', c.tp, T.green], ['FN', c.fn, T.amber], ['FP', c.fp, T.red], ['TN', c.tn, T.faint]];
            ctx.font = '10px ui-monospace, monospace';
            cells.forEach((cell, i) => {
              const x = rx + (i % 2) * (cw + 8), yy = ry + Math.floor(i / 2) * (ch + 8);
              // layout: TP FN / FP TN
              ctx.fillStyle = cell[2]; ctx.globalAlpha = .18; ctx.fillRect(x, yy, cw, ch); ctx.globalAlpha = 1;
              ctx.strokeStyle = T.line; ctx.strokeRect(x, yy, cw, ch);
              ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(cell[0], x + 6, yy + 5);
              ctx.fillStyle = T.text; ctx.font = 'bold 15px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
              ctx.fillText(String(cell[1]), x + cw - 6, yy + ch - 5);
              ctx.font = '10px ui-monospace, monospace';
            });
            ctx.fillStyle = T.faint; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('predicted + / actual + and −', rx, ry - 14);
            // mini ROC
            const mrx = rx, mry = ry + 2 * (ch + 8) + 16, ms = Math.min(110, w - rx - 20);
            ctx.strokeStyle = T.line; ctx.strokeRect(mrx, mry, ms, ms);
            ctx.strokeStyle = T.faint; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(mrx, mry + ms); ctx.lineTo(mrx + ms, mry); ctx.stroke(); ctx.setLineDash([]);
            ctx.strokeStyle = T.blue; ctx.lineWidth = 1.8; ctx.beginPath();
            roc.roc.forEach((p, i) => { const X = mrx + p[0] * ms, Y = mry + ms - p[1] * ms; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
            ctx.stroke();
            const fpr = c.fp / (c.fp + c.tn || 1), tpr = c.recall;
            ctx.fillStyle = T.red; ctx.beginPath(); ctx.arc(mrx + fpr * ms, mry + ms - tpr * ms, 4, 0, 6.3); ctx.fill();
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('ROC · your threshold in red', mrx, mry - 4);
            out({
              prec: (c.precision * 100).toFixed(1) + '%', rec: (c.recall * 100).toFixed(1) + '%',
              f1: c.f1.toFixed(3), acc: (c.accuracy * 100).toFixed(1) + '%', auc: roc.auc.toFixed(3)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Reproduce the worked example (1% positives)', primary: true, on: () => { st.$set('base', .01); st.$set('sep', 1.6); gen(st.sep, st.base); S.redraw(); } }
        ]);
        Viz.note(host, 'Set the positive rate to 1% and slide the threshold to where accuracy is maximised: the model flags almost nothing and scores 99%. Accuracy is not a metric on imbalanced problems, it is a decoy.');
      },

      roc: function (host) {
        const st = Viz.controls(host, [
          { k: 'sep', label: 'model quality', min: .2, max: 3.5, step: .05, value: 1.8, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'positive rate', min: .005, max: .5, step: .005, value: .01, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'auc', label: 'ROC-AUC', cls: 'good' }, { k: 'pr', label: 'PR-AUC', cls: 'bad' },
          { k: 'gini', label: 'Gini' }, { k: 'ks', label: 'KS' }, { k: 'baserate', label: 'PR baseline' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(19), N = 6000;
            const scores = [], labels = [];
            for (let i = 0; i < N; i++) {
              const pos = R() < st.base;
              labels.push(pos ? 1 : 0);
              scores.push(Num.sigmoid(R.normal(pos ? st.sep : -st.sep, 1.3)));
            }
            const r = Num.rocCurve(scores, labels);
            const half = (w - 60) / 2;
            const P1 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 40, r: w - 40 - half, t: 16, b: 36 } })
              .frame({ xlabel: 'false positive rate', ylabel: 'true positive rate' });
            P1.clip(() => {
              P1.line([[0, 0], [1, 1]], { color: T.faint, dash: [4, 4], width: 1.2 });
              P1.line(r.roc, { color: T.blue, width: 2.6 });
              P1.area(r.roc, { color: T.blue, alpha: .1 });
            });
            const P2 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 40 + half + 40, r: 14, t: 16, b: 36 } })
              .frame({ xlabel: 'recall', ylabel: 'precision' });
            P2.clip(() => {
              P2.line(r.pr, { color: T.red, width: 2.6 });
              P2.hline(st.base, { color: T.faint, dash: [4, 4], label: 'base rate' });
            });
            ctx.fillStyle = T.blue; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('ROC · looks excellent', 46, 2);
            ctx.fillStyle = T.red; ctx.fillText('PR · reveals the operating cost', 46 + half + 40, 2);
            out({
              auc: r.auc.toFixed(3), pr: r.ap.toFixed(3), gini: (2 * r.auc - 1).toFixed(3),
              ks: Num.ksStat(scores, labels).toFixed(3), baserate: st.base.toFixed(3)
            });
          }
        });
        Viz.note(host, 'At 1% positives an AUC of 0.95 routinely coexists with a PR-AUC near 0.4. The ROC curve’s x-axis is normalised by a huge negative class, so thousands of false positives barely move it — which is precisely the cost the business feels.');
      },

      cost: function (host) {
        const st = Viz.controls(host, [
          { k: 'cfn', label: 'cost of approving a defaulter (£)', min: 100, max: 3000, step: 50, value: 900, fmt: v => '£' + v },
          { k: 'cfp', label: 'cost of declining a good customer (£)', min: 10, max: 500, step: 10, value: 60, fmt: v => '£' + v },
          { k: 'sep', label: 'model quality', min: .3, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'default rate', min: .01, max: .3, step: .005, value: .06, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'opt', label: 'cost-optimal threshold', cls: 'key' }, { k: 'formula', label: 'C_FP/(C_FP+C_FN)' },
          { k: 'costOpt', label: 'cost at optimum' }, { k: 'cost50', label: 'cost at 0.5' }, { k: 'waste', label: 'money left on the table', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(37), N = 6000;
            const scores = [], labels = [];
            for (let i = 0; i < N; i++) {
              const pos = R() < st.base;
              labels.push(pos ? 1 : 0);
              scores.push(Num.sigmoid(R.normal(pos ? st.sep : -st.sep, 1.2)));
            }
            const costAt = t => {
              const c = Num.confusion(scores, labels, t);
              // "flagged" = declined. FN here = approved defaulter
              return (c.fn * st.cfn + c.fp * st.cfp) / N;
            };
            const pts = [];
            for (let t = .01; t <= .99; t += .01) pts.push([t, costAt(t)]);
            let best = pts[0]; pts.forEach(p => { if (p[1] < best[1]) best = p; });
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, Math.max.apply(null, pts.map(p => p[1])) * 1.1] })
              .frame({ xlabel: 'decision threshold', ylabel: 'expected cost per application (£)', yfmt: v => '£' + v.toFixed(0) });
            P.clip(() => {
              P.line(pts, { color: T.blue, width: 2.6 });
              P.vline(best[0], { color: T.green, width: 2, dash: false, label: 'cost-optimal' });
              P.vline(.5, { color: T.red, label: 'the default 0.5' });
              P.dots([best], { r: 5.5, color: T.green, stroke: true });
              P.dots([[.5, costAt(.5)]], { r: 5, color: T.red, stroke: true });
            });
            const theory = st.cfp / (st.cfp + st.cfn);
            out({
              opt: best[0].toFixed(3), formula: theory.toFixed(4),
              costOpt: '£' + best[1].toFixed(2), cost50: '£' + costAt(.5).toFixed(2),
              waste: '£' + (costAt(.5) - best[1]).toFixed(2) + ' / application'
            });
          }
        });
        Viz.note(host, 'The empirical optimum tracks $p^*=C_{FP}/(C_{FP}+C_{FN})$ whenever the probabilities are calibrated. With £900 versus £60 the cut is 6.25% — a long way from 0.5, and the per-application difference multiplied by your monthly volume is the business case for reading §2.12.');
      },

      lift: function (host) {
        const st = Viz.controls(host, [
          { k: 'sep', label: 'model quality', min: .2, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'bad rate', min: .01, max: .3, step: .005, value: .08, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'lift1', label: 'lift, worst decile', cls: 'key' }, { k: 'gain2', label: 'gains @ 20%' },
          { k: 'gain5', label: 'gains @ 50%' }, { k: 'ks', label: 'KS' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(53), N = 10000;
            const rows = [];
            for (let i = 0; i < N; i++) {
              const bad = R() < st.base;
              rows.push({ s: Num.sigmoid(R.normal(bad ? st.sep : -st.sep, 1.25)), bad: bad ? 1 : 0 });
            }
            rows.sort((a, b) => b.s - a.s);
            const totalBad = Num.sum(rows.map(r => r.bad));
            const deciles = [];
            for (let d = 0; d < 10; d++) {
              const seg = rows.slice(d * N / 10, (d + 1) * N / 10);
              const bads = Num.sum(seg.map(r => r.bad));
              deciles.push({ d: d + 1, bads: bads, share: bads / totalBad, rate: bads / seg.length });
            }
            let cum = 0; const gains = [[0, 0]];
            deciles.forEach((dd, i) => { cum += dd.share; gains.push([(i + 1) / 10, cum]); });
            const half = (w - 60) / 2;
            const P1 = Viz.plot(ctx, w, h, { xd: [.5, 10.5], yd: [0, Math.max.apply(null, deciles.map(d => d.share)) * 1.2], pad: { l: 42, r: w - 42 - half, t: 20, b: 38 } })
              .frame({ xlabel: 'decile (1 = riskiest)', ylabel: 'share of all bads', yfmt: v => (v * 100).toFixed(0) + '%' });
            P1.clip(() => {
              deciles.forEach(d => {
                const x0 = P1.x(d.d - .38), x1 = P1.x(d.d + .38);
                ctx.fillStyle = d.d <= 2 ? T.red : T.blue;
                ctx.fillRect(x0, P1.y(d.share), x1 - x0, P1.y(0) - P1.y(d.share));
              });
              P1.hline(.1, { color: T.faint, dash: [4, 4], label: 'a random 10%' });
            });
            const P2 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 42 + half + 40, r: 14, t: 20, b: 38 } })
              .frame({ xlabel: 'share of population reviewed', ylabel: 'share of bads caught', xfmt: v => (v * 100).toFixed(0) + '%', yfmt: v => (v * 100).toFixed(0) + '%' });
            P2.clip(() => {
              P2.line([[0, 0], [1, 1]], { color: T.faint, dash: [4, 4], width: 1.2 });
              P2.line(gains, { color: T.green, width: 2.6 });
              P2.dots(gains.slice(1), { r: 3.2, color: T.green });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('lift by decile', 48, 4);
            ctx.fillText('cumulative gains', 48 + half + 40, 4);
            out({
              lift1: (deciles[0].share / .1).toFixed(2) + '×',
              gain2: (gains[2][1] * 100).toFixed(0) + '%',
              gain5: (gains[5][1] * 100).toFixed(0) + '%',
              ks: Num.ksStat(rows.map(r => r.s), rows.map(r => r.bad)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'The gains curve is the ROC curve with the axes relabelled into business units. "Review the riskiest fifth and catch two thirds of the losses" and "AUC 0.78" are the same sentence spoken to two different audiences.');
      }
    },
    quiz: [
      {
        q: '10,000 cases, 1% positive. The model flags 200 and catches 80. Precision and recall are…',
        options: ['80% and 40%', '40% and 80%', '98.6% and 80%', '40% and 0.53'],
        answer: 1,
        why: 'Precision divides the 80 true catches by what was flagged, 200, giving 40%; recall divides the same 80 by what was actually positive, 100, giving 80%. The two ratios share a numerator and divide by different populations, which is exactly why they move in opposite directions as the threshold changes. The tempting wrong answer swaps them — a common slip, since both are proportions in [0,1] and nothing about the shape of the number tells you which population it was divided by. Accuracy, 98.6%, is a real number here but an uninformative one: predicting "no" for everyone scores 99%, higher still, while catching nothing.'
      },
      {
        q: 'Approving a defaulter costs £900; declining a good customer costs £60. The optimal threshold is…',
        options: ['0.5', '0.0625', '0.0667', '0.9375'],
        answer: 1,
        why: '$p^* = C_{FP}/(C_{FP}+C_{FN}) = 60/960 = 0.0625$, derived from setting the expected cost of approving equal to the expected cost of declining and solving for the break-even probability — not a rule of thumb, an indifference condition. It is only a valid cut on the true default probability if the model\'s $p$ is calibrated (§2.12); an inflated $p$, for instance from unrecalibrated resampling, moves every applicant across this threshold without their real risk having changed at all. 0.0667 is the tempting slip of computing $C_{FP}/C_{FN}$ without the sum in the denominator.'
      },
      {
        q: 'Under heavy imbalance, which pair should you report?',
        options: ['Accuracy and F1', 'ROC-AUC alone', 'ROC-AUC and PR-AUC, and operate on PR', 'Precision alone'],
        answer: 2,
        why: 'ROC-AUC is the right tool for comparing models\' underlying discriminative ability, because it is insensitive to the base rate; PR-AUC is the right tool for judging deployability, because precision responds directly to the base rate and reveals the operating cost ROC structurally cannot see (§2.13.2 derives exactly why). Reporting only one answers only one of the two questions a stakeholder is actually asking, and accuracy answers neither — it is dominated by the majority class regardless of how well the model handles the minority one.'
      }
    ],
    cards: [
      { q: 'Precision vs recall', a: 'Precision = TP/(TP+FP) — of what I flagged, how much was real. Recall = TP/(TP+FN) — of what was real, how much I caught. Same numerator, different denominators.' },
      { q: 'F1 is a harmonic mean, not an average', a: 'It is dominated by whichever of precision/recall is worse, so it cannot be flattered by excelling on one axis while failing the other.' },
      { q: 'Cost-optimal threshold', a: '$p^*=C_{FP}/(C_{FP}+C_{FN})$, from an indifference condition; £60 vs £900 gives 6.25%. Only meaningful if $p$ is calibrated.' },
      { q: 'Gini and KS', a: 'Gini = 2·AUC − 1 (a rescaling, no new information). KS = max gap between cumulative good and bad distributions.' },
      { q: 'When ROC lies', a: 'Under heavy imbalance: FPR is divided by the huge negative class and barely moves, while precision (and PR-AUC) divides by what was flagged and exposes the real cost.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.14 */
  ML.section({
    id: 'validation', track: 'classical', num: '2.14',
    title: 'Validation strategy, including time series and out-of-time',
    lede: 'Turns §2.1.4’s table of split failures into working machinery — how each scheme is actually built, and how much a wrong one costs, in AUC points you can compute. Rests on §1.4 for why an average estimates anything at all.',
    html: `
<p>A model reports AUC 0.89 in cross-validation. Six weeks after deployment, the monitoring dashboard reports 0.71 on live traffic, and nobody touched the model in between. This is not the drift story of §2.18 — not enough time has passed for the world to have changed. The validation number was never honest to begin with, and the model has been exactly as good as 0.71 the entire time; cross-validation was simply measuring the wrong thing and calling it performance.</p>

<p>§2.1.4 already named the culprits — time structure, entities with many rows, rare positives, nested hierarchy — and built the table of which split each one demands. This section does not repeat that argument. It builds the machinery: exactly how each scheme is constructed row by row, how badly the wrong one lies and by how much you can actually measure that lie, and one failure mode §2.1 did not cover at all — what happens when the thing you are validating is not the model, but the process that chose the model's hyperparameters.</p>

<h2><span class="sn">2.14.1</span> The schemes, built mechanically</h2>

<p><b>k-fold</b> partitions the $n$ rows into $k$ equal-sized blocks. Round $r$ trains on every block except block $r$ and validates on block $r$; do this for every $r$ from 1 to $k$ and every row has been validated on exactly once, never using a model that trained on it. Average the $k$ validation scores and you get an estimate with lower variance than a single holdout, for the same reason any average of several measurements beats one measurement (§1.4) — though the folds are not fully independent, since each pair of folds shares $k-2$ blocks of training data, so the variance reduction is real but not the full factor of $k$ a textbook independence assumption would promise.</p>

<p><b>Stratified k-fold</b> builds the same $k$ blocks with one added constraint: each block's share of positives matches the whole dataset's share, not whatever a random draw happens to produce. This matters more than it sounds like it should under imbalance. With a true positive rate of 1% and $k=10$, an unstratified fold has on average 10 positives in it if the dataset has 1,000 rows per fold — and by ordinary sampling variation, some folds will land with 4 and others with 16, and a fold's validation AUC computed from 4 positives is close to noise. Stratification removes that source of noise entirely, at essentially no cost.</p>

<p><b>Group k-fold</b> builds the $k$ blocks over <i>entities</i> rather than rows: assign each customer, account or device to exactly one block, then every row belonging to that entity follows it. A customer who appears 40 times in the dataset appears entirely in training or entirely in validation for any given fold, never split across the boundary.</p>

<p><b>Out-of-time (OOT)</b> abandons the fold structure altogether in favour of a single cut on the calendar: everything before date $T$ trains, everything after $T$ validates. <b>Rolling origin</b> repeats that cut at several dates — $T_1 < T_2 < \\cdots < T_m$ — training on everything before each $T_i$ and validating on a fixed window just after it, producing $m$ separate out-of-time estimates rather than one. A lender with two years of monthly data might fix a 12-month training window and slide the origin from month 13 through month 24, producing twelve estimates whose <i>spread</i>, not just their average, is the actual deliverable: a model that scores 0.81, 0.80, 0.82 and 0.79 across four origins is a materially different claim than one that scores 0.91, 0.85, 0.62 and 0.88, even though the two might average to a similar number. The second model is unstable in a way a single OOT split would never reveal.</p>

${H.analogy(`<p>Rolling-origin backtesting is a driving test administered the way real driving actually works: you never get examined on a road you already memorised the exam route for. Each origin date is a fresh exam — everything you have seen so far is fair preparation, everything after the exam date is unseen road — and taking the test four times at four different points in your training tells the examiner something a single pass-or-fail on one route cannot: whether you are a driver who generalises, or one who happened to memorise last Tuesday's particular junctions.</p>`)}

<p><b>What you are looking at.</b> Each coloured row is one round of the selected scheme: blue cells are rows used for training that round, red cells are the validation rows, amber marks an inner tuning fold under nested CV, and grey marks rows not yet reached in a time-ordered scheme. The strip of forty cells along each row is the dataset itself, laid out left to right in whatever order the scheme cares about — arbitrary row order for k-fold, calendar order for out-of-time and rolling origin.</p>

<p><b>What to do with it.</b> Switch to "group k-fold" and look for the small "g0", "g1" labels marking which customer group each column belongs to — notice every column sharing a group label stays the same colour within a round, never split between blue and red. Switch to "out-of-time" and watch the red block sit entirely to the right of every blue cell, with nothing interleaved. Switch to "nested CV" and watch the amber inner-tuning block rotate independently of the red outer-validation block within each round.</p>

<p><b>The thing genuinely worth noticing.</b> Switch to "stratified k-fold" and look at the white ticks marking positive-labelled rows: count them per fold and they land close to even, whereas switching to plain "k-fold" on the identical underlying data can leave some folds visibly tick-sparse and others tick-heavy, purely from where the random cut fell. The visual difference is the entire justification for stratification, with no formula required.</p>

${H.lab('cv', 'Every validation scheme, drawn', 'Switch between schemes and watch which rows are used for what. The group and time-series views make the failure modes obvious: a random fold splits a customer across train and validate, and a random fold on time trains on the future.')}

<h2><span class="sn">2.14.2</span> Choosing k: the estimate itself has a bias and a variance</h2>

<p>Everything above treated $k$ as a detail. It is not — $k$ trades one property of the <i>estimate</i> against another, and the trade is worth deriving rather than accepting "5 or 10, by convention" on faith.</p>

${H.deriv('why small k is pessimistically biased, and why the bias shrinks as k grows', [
      ['Each of the $k$ folds trains on $n(k-1)/k$ rows, not the full $n$ you will actually deploy with.', 'By construction: one block of $n/k$ rows is held out for validation, leaving $k-1$ of the $k$ blocks, or $n(k-1)/k$ rows, for training that round.'],
      ['More training data produces a model that is at least as good, in expectation — a <b>learning curve</b> that is flat or falling, essentially never rising, as sample size grows.', 'A learner given strictly more evidence about the same underlying pattern cannot become systematically worse at finding it; this is an empirical regularity about every model family in this course, not a theorem, but a famously reliable one.'],
      ['So each fold\'s validation score estimates the performance of a model trained on $n(k-1)/k$ rows — systematically worse than the model you actually ship, which trains on all $n$.', 'The fold\'s model is, by the previous line, at a genuine disadvantage relative to the deployed model, so its measured error is a pessimistic proxy for the error the deployed model will actually have.'],
      ['That pessimism shrinks as $n(k-1)/k \\to n$ — which happens exactly as $k\\to n$.', 'The fraction of data withheld from training each round, $1/k$, shrinks toward $1/n$ as $k$ grows, so the gap between "what each fold trained on" and "what you will deploy" narrows toward zero at the extreme case of leave-one-out CV, $k=n$.']
    ], 'Put real numbers on it at $n=1{,}000$: $k=2$ trains each fold on 500 rows, half the data — a large, real handicap relative to the 1,000-row deployed model. $k=5$ trains on 800 rows (80%); $k=10$ trains on 900 (90%); leave-one-out, $k=1{,}000$, trains on 999 rows (99.9%) — essentially the deployed model itself, evaluated on the one row it did not see. Bias shrinks monotonically as $k$ grows, with no exception.')}

<p>If bias were the only consideration, leave-one-out would always win. It is not the only consideration, because the $k$ validation scores k-fold produces are not independent measurements of the same quantity — they are built from training sets that overlap, and the overlap gets worse, not better, as $k$ grows.</p>

${H.deriv('why the folds become more correlated, not less, as k grows', [
      ['Fold $i$\'s training set and fold $j$\'s training set share every block except blocks $i$ and $j$ themselves: $n(k-2)/k$ rows in common.', 'Each fold trains on all blocks except its own held-out one, so two folds\' training sets differ only in which one block each of them excludes — everything else, $k-2$ of the $k$ blocks, is shared by both.'],
      ['As a fraction of each fold\'s own training set size, that overlap is $\\dfrac{n(k-2)/k}{n(k-1)/k} = \\dfrac{k-2}{k-1}$.', 'Divide the shared row count by the size of one fold\'s training set — both expressed with the common factor $n/k$, which cancels.'],
      ['That fraction increases with $k$: it is $3/4$ at $k=5$, $8/9\\approx0.889$ at $k=10$, $18/19\\approx0.947$ at $k=20$, and $\\to1$ as $k\\to n$.', 'A quick check of the formula at each value confirms the direction: more folds means each one holds out a smaller slice, so any two folds\' training sets differ by an ever-smaller fraction of themselves.']
    ], 'Two models trained on 94.7%-identical data (k=20) will make highly correlated errors — near-identical training sets tend to get near-identical rows wrong. Reusing the averaging-variance identity from §2.2\'s bagging derivation, $\\mathrm{Var}(\\text{mean of }k) = \\rho\\sigma^2 + (1-\\rho)\\sigma^2/k$: as $\\rho$ climbs toward 1 alongside $k$, the achievable variance reduction saturates well short of the naive $1/k$ a textbook independence assumption would promise, and can stop improving at all long before $k$ reaches $n$.')}

${H.flag('There is no clean closed-form answer for the exact variance of a k-fold estimate — the correlation ρ above depends on the model, the data and the metric, and estimating that variance honestly from a single run of cross-validation is a genuinely unresolved problem in the statistics literature, not a gap in this course\'s explanation. What survives is the shape of the trade-off: bias falls monotonically with k, variance-reduction efficiency does not rise monotonically with k, and $k=5$ or $k=10$ is where practice has settled empirically, not a value derived by minimising a formula in closed form.')}

<h2><span class="sn">2.14.3</span> Measuring the lie, not just naming it</h2>

<p>It is one thing to say "a random fold with entity leakage inflates your score." It is a stronger claim, and a far more useful one in a room that wants evidence, to say by how much. The mechanism is worth deriving in the idealised limit, because the limit makes the direction of the bias impossible to argue with.</p>

<p>Take four customers, each contributing three near-duplicate rows — the same customer's feature vector repeated with a sliver of noise $\\varepsilon$ added each time, and a label assigned once per customer, by a rule that has nothing to do with the features at all (a coin flip fixed at account opening, say). No genuinely learnable pattern connects $x$ to $y$ here; any honest model should score no better than chance. Train a nearest-neighbour classifier — predict the label of whichever training row is closest in feature space — under two splitting regimes.</p>

${H.deriv('the entity-leakage bound, in the noise-free limit', [
      ['Within-customer distance $= \\varepsilon$; between-customer distance $= d \\gg \\varepsilon$.', 'By construction: rows from the same customer are near-duplicates differing only by the noise $\\varepsilon$; rows from different customers sit a genuine distance $d$ apart in feature space.'],
      ['Random row split: for a validation row, its nearest training neighbour is, with probability $\\to 1$ as $\\varepsilon\\to0$, another row from the <i>same</i> customer.', 'Two of that customer\'s three rows are still in training after any row-level split that keeps at least one sibling row in-fold — overwhelmingly likely with only four customers and three rows each — and a same-customer row is closer by a factor of $d/\\varepsilon\\to\\infty$ than any other training row.'],
      ['So the classifier copies that customer\'s own label, correctly, essentially always: accuracy $\\to 1$, and since the labels are balanced across customers, $\\mathrm{AUC}\\to 1.0$.', 'A 1-nearest-neighbour rule that finds its own near-duplicate always predicts correctly — it is not learning the feature–label relationship, it is retrieving the answer it was trained on, dressed up as a different row.'],
      ['Group k-fold: a validation customer\'s rows are entirely absent from training, so the nearest training neighbour is some <i>other</i> customer, at distance $d$, whose label is independent of this customer\'s coin flip.', 'By construction, groups are assigned whole to one side of the split, so no near-duplicate of a held-out row can appear in training — the only neighbours available are genuinely different customers.'],
      ['The predicted label is uncorrelated with the true one: $\\mathrm{AUC} \\to 0.5$.', 'With no learnable signal and no leakage to exploit, a nearest-neighbour classifier does exactly as well as a coin flip — which is the honest answer, because a coin flip is, by construction, all there ever was to learn here.']
    ], 'The two limits, 1.0 and 0.5, are not a hypothetical worst case dreamt up for effect — they are the exact bookends of what entity leakage can do to a validation score, achieved here with a model that is not even trying to cheat. A gradient-boosted model with enough capacity to fit small idiosyncrasies of individual rows behaves the same way for the identical reason, just with a smaller and messier gap; the direction of the bias is never in doubt, only its size.')}

<p><b>What you are looking at.</b> Three bars, each the mean AUC of an actual logistic regression trained and validated five times under a different scheme, on the same synthetic population: random k-fold, group k-fold, and out-of-time, with a "rows per customer" and a "strength of the time trend" slider controlling how much entity structure and drift are baked into the data.</p>

<p><b>What to do with it.</b> Start with "rows per customer" near its minimum, where there is barely any entity structure to leak, and note the three bars sit close together. Now raise it toward its maximum and watch the random k-fold bar climb well above the other two — the derivation above, playing out on a real trained model rather than an idealised nearest-neighbour toy.</p>

<p><b>The thing genuinely worth noticing.</b> The readout's "size of the lie" converts the gap between random k-fold and out-of-time into basis points. Push both sliders to their maximum simultaneously and watch that number become large enough that, translated into a real credit book, it is the difference between a model a validator signs off on and one they send back — computed live from a model you can watch being trained, not a claim you have to take on faith.</p>

${H.lab('leak-cv', 'How much does the wrong scheme lie by?', 'The same data with an entity structure and a time trend, evaluated four ways. The gap between the random-k-fold number and the out-of-time number is the size of the lie — computed, not asserted.')}

<h2><span class="sn">2.14.4</span> Nested CV — validating the search, not just the model</h2>

<p>Every scheme so far assumes the model itself is fixed before validation starts. In practice it never is: you try several learning rates, several regularisation strengths, several tree depths, and you pick whichever configuration scored best in cross-validation. That selection step is invisible in the final number you report, and it is not innocent.</p>

<p>§2.1.3 derived exactly this mechanism in a different costume: report the best of even two pure-noise, equally-worthless models on a sealed test set, and the reported winner is biased upward by $\\mathrm{se}/\\sqrt\\pi$ purely from the act of choosing a maximum — before a single model has genuinely improved. Hyperparameter search is the identical derivation with "candidate model" relabelled "candidate configuration" and "test set" relabelled "validation folds": try sixty configurations, report the best cross-validation score, and that score is the biased maximum of sixty noisy estimates, inflated by selection in exactly the way §2.1.3 quantified, whether or not any of the sixty configurations was genuinely better than the others.</p>

<p><b>Nested cross-validation</b> is the fix, and it is built from the same idea that made the sealed test set trustworthy in §2.1.3: never let the same data both choose and grade. An outer loop splits the data into $k$ folds exactly as ordinary k-fold does, held back purely for grading. Inside each outer training fold, an entire inner cross-validation runs to select the best hyperparameters using only that fold's data — the inner loop's job is choosing, never grading. The selected configuration is then trained on the full outer-training fold and scored, once, on the outer-validation fold it has never influenced in any way. Average those $k$ outer scores and you have an estimate that has never let a fold both pick a winner and mark its own homework.</p>

${H.table(['', 'Who chooses the hyperparameters', 'Who grades the final score'], [
      ['Plain k-fold + report best CV score', 'The same folds being reported', 'The same folds — the fox marking the henhouse'],
      ['Nested CV', 'Inner loop, using only outer-training data', 'Outer loop, on data the inner loop never touched']
    ])}

${H.pitfall(`<p>Nested CV is computationally expensive — an outer $k$-fold wrapped around an inner $k$-fold multiplies the number of models trained by roughly $k^2$ times the size of the hyperparameter grid — and the temptation is always to skip it and report the plain CV number "just this once." The honest alternative when full nesting is too slow is a single untouched final holdout: search freely with ordinary cross-validation on everything else, then evaluate the winning configuration exactly once on data that took no part in the search at all. What is never acceptable is searching and reporting from the same folds and calling the number unbiased — that is precisely the mechanism §2.1.3 quantified, not a matter of opinion.</p>`)}

${H.probe([
      ['Why not k-fold on a time series?', 'It trains on the future to predict the past. Use out-of-time and rolling-origin backtests, which mirror the one thing deployment will actually ask of the model: predict forward from what has already happened.'],
      ['When do you need group k-fold?', 'Whenever one entity contributes multiple rows — otherwise a model can partly succeed by recognising the entity rather than learning the general pattern, and the noise-free bound in §2.14.3 shows the inflation this can cause is not a rounding error.'],
      ['Why nested CV?', 'The fold that chose your hyperparameters is no longer an unbiased judge of performance — it is the biased-maximum mechanism of §2.1.3, applied to a search over configurations instead of a search over models. The outer loop restores an unbiased estimate by never letting the same data choose and grade.']
    ])}`,
    labs: {
      cv: function (host) {
        const st = Viz.controls(host, [
          { k: 'scheme', label: 'scheme', type: 'select', value: 'kfold', options: [
            { v: 'holdout', t: 'single holdout' }, { v: 'kfold', t: 'k-fold' }, { v: 'strat', t: 'stratified k-fold' },
            { v: 'group', t: 'group k-fold' }, { v: 'nested', t: 'nested CV' }, { v: 'oot', t: 'out-of-time' }, { v: 'roll', t: 'rolling origin' }] },
          { k: 'k', label: 'folds', min: 2, max: 8, step: 1, value: 5, fmt: v => v }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const N = 40;
            const rowsFor = {
              holdout: 1, kfold: st.k, strat: st.k, group: st.k, nested: st.k, oot: 1, roll: 4
            }[st.scheme];
            const x0 = 60, wpx = w - 90, rh = Math.min(30, (h - 70) / rowsFor);
            const R = Num.rng(3);
            const groups = Array.from({ length: N }, (_, i) => Math.floor(i / 5) % 8);
            const labels = Array.from({ length: N }, () => R() < .2 ? 1 : 0);
            for (let r = 0; r < rowsFor; r++) {
              for (let i = 0; i < N; i++) {
                let kind = 'train';
                if (st.scheme === 'holdout') kind = i >= N * .8 ? 'valid' : 'train';
                else if (st.scheme === 'kfold' || st.scheme === 'strat') kind = (i % st.k === r) ? 'valid' : 'train';
                else if (st.scheme === 'group') kind = (groups[i] % st.k === r) ? 'valid' : 'train';
                else if (st.scheme === 'nested') kind = (i % st.k === r) ? 'valid' : ((i % st.k === (r + 1) % st.k) ? 'inner' : 'train');
                else if (st.scheme === 'oot') kind = i >= N * .75 ? 'valid' : 'train';
                else if (st.scheme === 'roll') {
                  const trainEnd = 12 + r * 6, validEnd = trainEnd + 6;
                  kind = i < trainEnd ? 'train' : (i < validEnd ? 'valid' : 'unseen');
                }
                const col = kind === 'train' ? T.blue : kind === 'valid' ? T.red : kind === 'inner' ? T.amber : T.line;
                ctx.fillStyle = col;
                ctx.globalAlpha = kind === 'unseen' ? .35 : 1;
                ctx.fillRect(x0 + i * (wpx / N), 40 + r * (rh + 4), wpx / N - 1.5, rh);
                ctx.globalAlpha = 1;
                if (st.scheme === 'strat' && labels[i]) {
                  ctx.fillStyle = '#fff'; ctx.globalAlpha = .85;
                  ctx.fillRect(x0 + i * (wpx / N) + 1, 40 + r * (rh + 4) + rh - 4, wpx / N - 3.5, 2.5);
                  ctx.globalAlpha = 1;
                }
                if (st.scheme === 'group' && r === 0) {
                  ctx.fillStyle = T.faint; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                  if (i % 5 === 2) ctx.fillText('g' + groups[i], x0 + (i + .5) * (wpx / N), 36);
                }
              }
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText(st.scheme === 'roll' ? 'run ' + (r + 1) : 'fold ' + (r + 1), x0 - 10, 40 + r * (rh + 4) + rh / 2);
            }
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const desc = {
              holdout: 'One split. Cheapest, highest-variance estimate.',
              kfold: 'Every row is validated exactly once; the estimate averages k splits.',
              strat: 'White ticks are positives — every fold keeps the class ratio.',
              group: 'All rows of a group land in the same fold; no entity spans the split.',
              nested: 'Amber = inner loop for hyperparameter search; red = outer loop for the reported estimate.',
              oot: 'Time flows left to right. Validation is strictly later than training.',
              roll: 'Several out-of-time estimates, each from a later origin. Grey = not yet seen at that point in time.'
            }[st.scheme];
            ctx.fillText(desc, x0, 44 + rowsFor * (rh + 4) + 6);
            const leg = [[T.blue, 'train'], [T.red, 'validate'], [T.amber, 'inner tuning'], [T.line, 'not yet seen']];
            leg.forEach((L, i) => {
              ctx.fillStyle = L[0]; ctx.fillRect(x0 + i * 110, h - 24, 12, 12);
              ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textBaseline = 'middle';
              ctx.fillText(L[1], x0 + i * 110 + 18, h - 18);
            });
          }
        });
      },

      'leak-cv': function (host) {
        const st = Viz.controls(host, [
          { k: 'entity', label: 'rows per customer', min: 1, max: 12, step: 1, value: 6, fmt: v => v },
          { k: 'drift', label: 'strength of the time trend', min: 0, max: 2, step: .05, value: .9, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rand', label: 'random k-fold AUC', cls: 'bad' }, { k: 'group', label: 'group k-fold AUC' },
          { k: 'oot', label: 'out-of-time AUC', cls: 'key' }, { k: 'lie', label: 'size of the lie' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(67);
            const nCust = 120, rows = [];
            for (let c = 0; c < nCust; c++) {
              const trait = R.normal(0, 1);                    // customer-level signal (leaks across folds)
              for (let j = 0; j < st.entity; j++) {
                const t = (c * st.entity + j) / (nCust * st.entity);
                const drift = st.drift * (t - .5);
                const z = 1.1 * trait + drift + R.normal(0, .9);
                rows.push({ x: [trait + R.normal(0, .35), t], y: R() < Num.sigmoid(z) ? 1 : 0, cust: c, t: t });
              }
            }
            function evalScheme(kind) {
              const K = 5, aucs = [];
              for (let k = 0; k < K; k++) {
                let tr, va;
                if (kind === 'rand') { tr = rows.filter((_, i) => i % K !== k); va = rows.filter((_, i) => i % K === k); }
                else if (kind === 'group') { tr = rows.filter(r => r.cust % K !== k); va = rows.filter(r => r.cust % K === k); }
                else { const cut = .55 + .09 * k; tr = rows.filter(r => r.t < cut); va = rows.filter(r => r.t >= cut && r.t < cut + .09); }
                if (!va.length || !tr.length) continue;
                const m = Num.logistic(tr.map(r => r.x), tr.map(r => r.y), { lr: .5 }); m.step(220);
                const p = va.map(r => m.predict(r.x));
                const lab = va.map(r => r.y);
                if (Num.sum(lab) === 0 || Num.sum(lab) === lab.length) continue;
                aucs.push(Num.rocCurve(p, lab).auc);
              }
              return aucs.length ? Num.mean(aucs) : NaN;
            }
            const a = evalScheme('rand'), b = evalScheme('group'), c = evalScheme('oot');
            const bars = [['random k-fold', a, T.red], ['group k-fold', b, T.amber], ['out-of-time', c, T.green]];
            const bx = 120, bw = w - bx - 90;
            bars.forEach((br, i) => {
              const y = 40 + i * 46;
              ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText(br[0], bx - 12, y + 12);
              ctx.fillStyle = br[2];
              const frac = Math.max(0, (br[1] - .5) / .45);
              ctx.fillRect(bx, y, bw * frac, 24);
              ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'left';
              ctx.fillText(isFinite(br[1]) ? br[1].toFixed(3) : '—', bx + bw * frac + 8, y + 12);
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('same data, same model, three validation schemes', bx, h - 30);
            out({
              rand: isFinite(a) ? a.toFixed(3) : '—', group: isFinite(b) ? b.toFixed(3) : '—',
              oot: isFinite(c) ? c.toFixed(3) : '—',
              lie: isFinite(a) && isFinite(c) ? '+' + ((a - c) * 1000).toFixed(0) + ' bps' : '—'
            });
          }
        });
        Viz.note(host, 'Raise "rows per customer" and the random-k-fold number inflates: the model has seen the same customer on both sides of the split. Raise the time trend and the out-of-time number falls behind both. The honest number is the one that matches how you will actually deploy.');
      }
    },
    quiz: [
      {
        q: 'Each customer contributes 8 rows and you use random 5-fold CV. What breaks?',
        options: ['Nothing', 'The same customer appears in train and validate, so you measure memorisation of the customer', 'The folds become imbalanced', 'Training becomes slower'],
        answer: 1,
        why: 'A random row-level split has no reason to keep one customer\'s rows together, so most customers end up with some rows in training and some in validation. The idealised bound in §2.14.3 shows exactly how far this can inflate the score — toward AUC 1.0 for a model that is doing nothing but recognising near-duplicates of rows it already trained on. Group k-fold, assigning whole entities to one side of the split, is the fix; nothing about the fold count or stratification touches this failure, which is why the other two options are tempting but wrong.'
      },
      {
        q: 'You tuned 60 hyperparameter configurations with 5-fold CV and report the best CV score. That number is…',
        options: ['unbiased', 'optimistic — the folds chose the hyperparameters, so nest the search or hold out a final set', 'pessimistic', 'valid if you used stratification'],
        answer: 1,
        why: 'This is §2.1.3\'s biased-maximum derivation wearing a different hat: reporting the best of many noisy cross-validation estimates is reporting a maximum, and a maximum of unbiased estimates is itself a biased estimate of the true best, inflated further the more configurations you tried. Stratification fixes fold composition, not selection bias, so it does not touch this problem at all. Nested CV, or a single untouched final holdout evaluated exactly once, is the only fix — the fold that chose the winner cannot also be trusted to grade it.'
      }
    ],
    cards: [
      { q: 'When is group k-fold mandatory?', a: 'Whenever one entity contributes multiple rows — otherwise validation measures entity recognition, not pattern learning.' },
      { q: 'Time-series validation', a: 'Out-of-time, plus rolling origin for several estimates whose spread matters as much as their average. Never shuffle.' },
      { q: 'Why nested CV', a: 'The fold that selected hyperparameters cannot also provide an unbiased performance estimate — the §2.1.3 biased-maximum result applied to a search over configurations.' },
      { q: 'Stratified vs group k-fold', a: 'Stratified balances the class ratio per fold (fixes imbalance). Group keeps one entity\'s rows together (fixes leakage). Different failure modes, different fixes.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.15 */
  ML.section({
    id: 'hyperparameters', track: 'classical', num: '2.15',
    title: 'Hyperparameter search and stacking',
    lede: 'Why random beats grid, what Bayesian optimisation actually buys, and why banks rarely ship the stack that wins competitions. Stacking’s payoff traces back to §2.7’s bagging-variance formula; every configuration it searches still needs §2.14’s honest validation to grade it.',
    html: `
<p>A gradient-boosted model has, conservatively, half a dozen hyperparameters worth tuning: learning rate, tree depth, the number of trees, a couple of regularisation strengths, the minimum rows per leaf. Grid search over even a modest 5 values per axis is $5^6 \\approx 15{,}600$ configurations — at a minute per fit, three weeks of continuous compute for one model. Nobody actually runs that grid. Everybody tunes far fewer combinations than the full grid contains and still finds something close to the best setting most of the time. The question worth asking before reaching for any particular search strategy is why that shortcut works at all — and the answer turns out to depend entirely on a fact about hyperparameters that has nothing to do with any specific algorithm.</p>

<h2><span class="sn">2.15.1</span> Why random search beats a grid, on the same budget</h2>

<p>The fact is this: in almost every real model, only a handful of hyperparameters actually move the validation score, and the rest barely matter across a wide range of settings. A learning rate that is off by 3× can sink a model; a minimum-rows-per-leaf that is off by 3× rarely changes much at all. Grid search has no way to know this in advance, so it spends its budget uniformly across every axis, important or not — which means most of that budget is wasted repeating values of the unimportant axes while barely varying the one axis that decides the outcome.</p>

<p>Make this precise with the exact numbers the lab below uses by default. Nine evaluations, two hyperparameters, arranged as a $3\\times3$ grid: three distinct values on each axis, each one repeated three times as the other axis varies. If only the first axis matters, the grid has tried exactly <b>three</b> distinct settings of the axis that decides everything, spending three evaluations confirming each one. Nine random draws from the same two-dimensional space, by contrast, land at nine independently-chosen coordinates — and because a continuous random draw essentially never repeats a value exactly, all <b>nine</b> are distinct settings of the important axis. Same budget, nine evaluations either way; the grid tested three settings of what matters, random tested nine.</p>

${H.deriv('why the distinct-values gap between grid and random grows with dimensionality', [
      ['A $d$-dimensional grid with $m$ points per axis uses $m^d$ evaluations and tests exactly $m$ distinct values of any one axis.', 'Every grid point is a full combination of one setting per axis; fixing attention on axis 1, its value repeats across every combination of the other $d-1$ axes, so only $m$ genuinely different settings of axis 1 are ever tried, however large $m^d$ is.'],
      ['$n$ random draws in the same $d$-dimensional space test, with probability 1, $n$ distinct values of any one axis.', 'A continuous random variable takes any particular value with probability zero, so $n$ independent draws are $n$ distinct real numbers on every axis simultaneously, at once — not just on the axis you care about, on all of them.'],
      ['Matching budgets, $n = m^d$: random tests $m^d$ distinct values of the important axis; grid tests $m$.', 'Substitute the shared budget. The ratio $m^d / m = m^{d-1}$ is the exact size of random search\'s advantage on the one axis that actually matters, and it grows without bound as $d$ grows, for a fixed per-axis resolution $m$.']
    ], 'Plug in the lab\'s own numbers as a sanity check: $d=2$, $m=3$, budget $n=9=3^2$. Random\'s advantage is $m^{d-1} = 3^1 = 3$ — exactly the 3-vs-9 distinct-value gap computed above. In six dimensions with the same per-axis resolution, the advantage becomes $3^5 = 243$-fold. This is the substance of Bergstra & Bengio\'s 2012 result: not that random search is a clever trick, but that a grid\'s structure actively fights the fact that most axes do not matter.')}

<p><b>What you are looking at.</b> Two scatter panels share the same axes: the horizontal one is the hyperparameter that actually determines quality, the vertical one is a hyperparameter that does nothing. The amber band marks where the important axis is well set. Left panel is grid's trial points, right panel is random's, both drawn from the same evaluation budget.</p>

<p><b>What to do with it.</b> Leave the budget at its default and count the grid dots' x-coordinates by eye: they fall on a handful of repeated vertical lines. The random panel's dots scatter across the whole width with no repeated x-coordinate. Read the "distinct good values" readout for both — it is counting exactly the quantity the derivation above computed.</p>

<p><b>The thing genuinely worth noticing.</b> Narrow "width of the good region" toward its minimum, simulating a hyperparameter that needs to be hit quite precisely to matter. Grid's discrete columns can now miss the band entirely between two adjacent grid lines, reporting zero good trials, while random — sampling continuously — almost always lands at least one point inside no matter how narrow the target, simply by scattering everywhere rather than committing to a fixed lattice in advance.</p>

${H.lab('search', 'Why random beats grid', 'The yellow band is the region where the one hyperparameter that matters is well set. Grid samples that band three times with the same value of the important axis; random samples it three times with three different values — and the readout counts how often each strategy finds a good configuration.')}

${H.intuition(`<p>The argument above rests on an assumption worth stating out loud, because it is what the result actually depends on: <b>low effective dimensionality</b>. You may be searching eight hyperparameters, but typically only one or two of them genuinely move the score, and you do not know in advance which. Grid search spends its budget as if every axis mattered equally — it insists on visiting every combination of values, which means the number of <i>distinct</i> values it tries along any single axis is only the grid's resolution, however many points you evaluate in total.</p>
<p>Random search makes no such commitment. Every sample is a fresh draw on <i>every</i> axis, so $n$ evaluations give you $n$ distinct values along the axis that turns out to matter, rather than the handful a grid allows. The saving is not that random search is cleverer — it is that grid search wastes its budget re-testing values of the axes that were never going to matter, and it wastes more of it the more irrelevant axes you add.</p>
<p>The corollary is the honest limitation: where <i>every</i> dimension genuinely matters and interacts, the advantage narrows considerably, because then the exhaustive coverage a grid buys is no longer wasted.</p>`)}

${H.history(`<p>This was argued properly by Bergstra and Bengio in 2012, and the paper is unusual in how quickly it changed practice. Grid search had been the default for years, partly because it feels systematic and is trivially parallel, and partly because a grid is easy to describe in a paper's methods section.</p>
<p>Their contribution was not a new algorithm — random sampling needs no invention — but the observation that the field's default was quantifiably worse on the same budget, for a reason nobody had made precise. It is a good reminder that a method can become standard because it looks rigorous rather than because anyone checked it, and that the check is sometimes a short argument rather than a large experiment.</p>`)}

<p><b>Bayesian optimisation</b> and its close relative TPE (tree-structured Parzen estimators) go a step further than either grid or random: after each evaluation, fit a cheap surrogate model of "how does the score depend on the hyperparameters so far," and choose the next point to evaluate where that surrogate predicts either a high score or high uncertainty — exploiting what looks promising while still exploring where you know the least. This pays for its own overhead specifically when each real evaluation is expensive — a large neural network trained for days, say — because the surrogate model's fitting cost is trivial by comparison and the informed choice of where to look next can save whole evaluations that grid or random would have wasted on clearly unpromising regions.</p>

<p><b>Successive halving</b> attacks the problem from an entirely different angle: instead of choosing evaluations more cleverly, spend less on each one until it has earned more. Start many configurations on a small budget — a handful of training epochs, a subsample of the data — rank them by their early performance, discard the worse half (or worse $1/\\eta$ fraction, for a cut factor $\\eta$), and give the survivors a larger budget. Repeat until one configuration remains. <b>Hyperband</b> is successive halving run at several different starting-budget-versus-number-of-configurations trade-offs simultaneously, because successive halving alone has one weak spot worth naming honestly: cutting on an early, noisy estimate can eliminate a genuinely good configuration that simply had a bad early run, and Hyperband hedges that risk by also trying brackets that give more configurations a longer initial look.</p>

<p><b>What you are looking at.</b> Each row is one rung of the tournament: blue cells are configurations still alive at that budget, green marks the configuration that is genuinely best (known here because this is a simulation), and the rung's assigned budget is labelled on the left. Configurations vanish from row to row as the worst fraction is cut.</p>

<p><b>What to do with it.</b> Leave noise near zero and watch the green cell survive every cut, reaching the final rung — with clean early signal, halving reliably finds the true best while spending only a fraction of what evaluating every configuration at full budget would have cost, exactly as the "compute used" versus "full-budget search would cost" readout quantifies.</p>

<p><b>The thing genuinely worth noticing.</b> Raise "noise in early estimates" and watch the green cell sometimes vanish at an early rung, cut on a bad early read despite being the true best configuration. This is not a bug in the method, it is the honest price of the entire strategy: successive halving buys enormous compute savings by trusting early, noisy estimates, and occasionally that trust is misplaced. Hyperband's answer is to run several brackets side by side rather than betting everything on one aggressive cutting schedule — a hedge against exactly this failure, not a way to eliminate it.</p>

${H.lab('halving', 'Successive halving, running', 'Configurations are trained on increasing budgets, with the worst half cut at each rung. Watch the total compute spent versus a full-budget random search that finds the same winner.')}

<h2><span class="sn">2.15.2</span> Stacking, and why banks rarely ship it</h2>

<p>Every search strategy above tunes one model. <b>Stacking</b> asks a different question: given several already-tuned models of genuinely different kinds — a gradient-boosted tree, a regularised logistic regression, perhaps a small neural network — can a learner do better by combining their opinions than any one of them does alone? The mechanism traces directly back to §2.7's bagging-variance formula, $\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$: averaging $B$ predictors each with variance $\\sigma^2$ and pairwise correlation $\\rho$ leaves a residual variance that never falls below $\\rho\\sigma^2$, however large $B$ grows, and only the <i>uncorrelated</i> share of the variance gets divided down. Bagging decorrelates by resampling the same algorithm on different data; stacking decorrelates by combining genuinely different algorithms, whose errors are correlated with each other far more weakly than two random forests grown from the same procedure ever are — a gradient-boosted tree and a linear model tend to get different examples wrong, precisely because they are built on different assumptions about the world.</p>

<p>Build a stack in two stages. First, train several diverse base models. Second, collect their predictions on the training set and use those predictions <i>as features</i> for a meta-learner — usually a simple, regularised model like logistic regression, because the meta-learner's job is just to learn how much to trust each base model, not to learn a new complex pattern from scratch. The entire trick, and the single detail that separates a stack that generalises from one that silently fails, is <i>how</i> those training-set predictions are generated.</p>

${H.pitfall(`<p>If you generate the meta-features by having each base model predict on the same rows it was trained on, the meta-learner is being handed each base model's <i>memorised</i> answers rather than its genuine ability to generalise. A base model that has overfit its training data will report near-perfect predictions on those exact rows, and the meta-learner — having no way to tell memorisation from real skill — learns to trust that overfit model more than it deserves. The fix is <b>out-of-fold prediction</b>: run ordinary k-fold cross-validation for each base model, and for every row use the prediction from the one fold where that row sat in validation, never in training. Concatenate those out-of-fold predictions across all $k$ folds and you have, for every training row, a prediction the model producing it never saw that row to make. That is the honest signal the meta-learner needs, and it is exactly the same discipline §2.1 and §2.14 build for evaluating a single model, applied one level up to evaluating the base models as <i>inputs</i> to a second model.</p>`)}

<p>One further subtlety trips people up in interviews specifically because it sounds like it should not be necessary: the base models used to <i>generate</i> the out-of-fold meta-features are the $k$ fold-specific copies, each trained on only $k-1$ folds — but the base models used to <i>serve</i> real predictions in production are retrained on the <i>entire</i> training set, all $k$ folds at once, because throwing away $1/k$ of your data at serving time for no reason would be wasteful. This means the meta-learner is trained on out-of-fold predictions from slightly-under-trained base models and then deployed against predictions from slightly-better-trained ones — a small distributional mismatch that is usually harmless in practice but worth being able to name, because "why not just serve with the fold models" is a fair question and "because you'd be discarding data for no benefit, and the mismatch this introduces is small compared to that cost" is the correct answer.</p>

<p><b>Blending</b> is the cheap, lower-ceiling cousin of stacking: rather than training a meta-learner on out-of-fold predictions, fix the base models' predictions on a single held-out set and learn a simple weighted average — sometimes not even learned, just tuned by hand — that minimises validation error. It avoids the fold bookkeeping entirely at the cost of using less data to learn the combination and losing the ability to model non-linear interactions between base models' predictions that a proper meta-learner could pick up.</p>

<p>Stacking reliably adds a point or two of AUC in practice, because it genuinely exploits disagreement between model families — which is exactly why it wins competitions, where the last point of AUC is worth fighting for and nothing else about the model matters once the leaderboard closes. In a regulated lending pipeline, the calculus is different. Every base model, plus the meta-learner sitting on top of them, is a separate artefact: separate conceptual-soundness justification, separate monitoring for drift (§2.18), separate SHAP story (§2.17) that now has to explain a decision two layers removed from any single interpretable model. The marginal AUC point from stacking rarely survives contact with the cost of validating, monitoring and explaining three or four models instead of one — know precisely how the mechanism works, because it is a fair interview question, and know when to argue against shipping it, because that judgement is the more senior half of the answer.</p>

${H.probe([
      ['Why does random search beat grid?', 'With few important dimensions, random draws sample the important axis at many distinct values instead of repeating a small set — the advantage grows as $m^{d-1}$ for a $d$-dimensional grid with $m$ points per axis, derived in §2.15.1, not merely observed empirically.'],
      ['What is the trick in stacking?', 'Out-of-fold predictions as meta-features, so the meta-learner sees each base model\'s genuine generalisation ability rather than its memorised training-set answers. Anything else teaches the meta-learner to trust whichever base model has overfit hardest.'],
      ['When would you not stack?', 'When each additional artefact must be independently validated, monitored and explained under model-risk governance (§2.18) — the typical one-or-two-point AUC gain rarely pays for the multiplied compliance surface, and a simpler model that a validator can reason about end to end is often the actual senior recommendation.']
    ])}`,
    labs: {
      search: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'evaluations budgeted', min: 4, max: 64, step: 1, value: 9, fmt: v => v },
          { k: 'band', label: 'width of the good region', min: .05, max: .5, step: .01, value: .18, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'gridHit', label: 'grid: distinct good values', cls: 'bad' },
          { k: 'randHit', label: 'random: distinct good values', cls: 'good' },
          { k: 'gridBest', label: 'grid best score' }, { k: 'randBest', label: 'random best score' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(29);
            const side = Math.max(2, Math.round(Math.sqrt(st.n)));
            const grid = [], rand = [];
            for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) grid.push([(i + .5) / side, (j + .5) / side]);
            for (let i = 0; i < st.n; i++) rand.push([R(), R()]);
            const centre = .62;
            const score = p => Math.exp(-Math.pow((p[0] - centre) / st.band, 2)) * (1 + .06 * p[1]);
            const half = (w - 60) / 2;
            const drawPanel = (pts, x0, title, col) => {
              const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: x0, r: w - x0 - half, t: 26, b: 34 } })
                .frame({ xlabel: 'the hyperparameter that matters', ylabel: 'the one that does not' });
              P.clip(() => {
                ctx.fillStyle = 'rgba(230,190,60,.22)';
                ctx.fillRect(P.x(centre - st.band), P.pad.t, P.x(centre + st.band) - P.x(centre - st.band), P.ph);
                P.dots(pts, { r: 4, color: col, stroke: true });
              });
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(title, x0 + 2, 8);
              return P;
            };
            drawPanel(grid, 44, 'grid · ' + grid.length + ' trials', Viz.theme().blue);
            drawPanel(rand, 44 + half + 40, 'random · ' + rand.length + ' trials', Viz.theme().red);
            const distinct = pts => new Set(pts.filter(p => Math.abs(p[0] - centre) < st.band).map(p => p[0].toFixed(3))).size;
            out({
              gridHit: distinct(grid), randHit: distinct(rand),
              gridBest: Math.max.apply(null, grid.map(score)).toFixed(3),
              randBest: Math.max.apply(null, rand.map(score)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'Grid’s trials share x-coordinates by construction: with 9 trials it explores 3 distinct values of the axis that matters. Random explores 9. In higher dimensions the gap widens exponentially — which is the whole of Bergstra & Bengio’s result.');
      },

      halving: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'configurations', min: 8, max: 64, step: 4, value: 32, fmt: v => v },
          { k: 'eta', label: 'cut factor η', min: 2, max: 4, step: 1, value: 3, fmt: v => v },
          { k: 'noise', label: 'noise in early estimates', min: 0, max: .3, step: .01, value: .12, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'compute', label: 'compute used (budget units)', cls: 'key' },
          { k: 'full', label: 'full-budget search would cost' },
          { k: 'saving', label: 'saving', cls: 'good' }, { k: 'found', label: 'found the true best?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(101);
            const configs = Array.from({ length: st.n }, (_, i) => ({ id: i, quality: R() }));
            const trueBest = configs.reduce((a, b) => a.quality > b.quality ? a : b);
            let alive = configs.slice(), budget = 1, spent = 0;
            const rungs = [];
            while (alive.length > 1) {
              alive.forEach(c => { c.obs = c.quality + R.normal(0, st.noise / Math.sqrt(budget)); spent += budget; });
              rungs.push({ budget: budget, alive: alive.slice() });
              alive.sort((a, b) => b.obs - a.obs);
              alive = alive.slice(0, Math.max(1, Math.floor(alive.length / st.eta)));
              budget *= st.eta;
            }
            rungs.push({ budget: budget, alive: alive.slice() });
            const maxB = budget;
            const x0 = 56, wpx = w - 90;
            rungs.forEach((r, i) => {
              const y = 34 + i * ((h - 80) / rungs.length);
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText('budget ' + r.budget, x0 - 8, y + 6);
              r.alive.forEach((c, j) => {
                const x = x0 + (j / st.n) * wpx;
                ctx.fillStyle = c.id === trueBest.id ? T.green : T.blue;
                ctx.globalAlpha = c.id === trueBest.id ? 1 : .55;
                ctx.fillRect(x, y, Math.max(2, wpx / st.n - 2), 12);
                ctx.globalAlpha = 1;
              });
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('each row = a rung; survivors move down and get more budget. Green = the truly best configuration.', x0, h - 34);
            out({
              compute: spent.toFixed(0), full: (st.n * maxB).toFixed(0),
              saving: (100 * (1 - spent / (st.n * maxB))).toFixed(0) + '%',
              found: alive[0] && alive[0].id === trueBest.id ? 'yes' : 'no — noise killed it early'
            });
          }
        });
        Viz.note(host, 'Raise the noise and successive halving starts discarding the true best configuration at an early rung on a bad estimate. That is the real trade: enormous compute savings against the risk of an early wrong call, which is exactly what Hyperband hedges by running several bracket sizes.');
      }
    },
    quiz: [
      {
        q: 'Random search beats grid search mainly because…',
        options: ['it is easier to implement', 'with few important dimensions it samples the important axis at many distinct values', 'it converges to the global optimum', 'it uses fewer evaluations'],
        answer: 1,
        why: 'A 3×3 grid tries three distinct values of each axis, however many total combinations it evaluates, because every grid point repeats each axis value across every setting of the other axes. Nine random draws try nine distinct values of whichever axis you look at, matching or beating the same budget. Neither method is guaranteed to find the global optimum, ruling out that option, and the advantage is about how the budget is spent, not the size of the budget itself, ruling out "fewer evaluations."'
      },
      {
        q: 'In stacking, the meta-learner must be trained on…',
        options: ['the base models’ in-fold predictions', 'out-of-fold predictions', 'the raw features only', 'the test set'],
        answer: 1,
        why: 'A base model\'s in-fold predictions reflect what it memorised about its own training rows, not what it can genuinely generalise to — feed those to the meta-learner and it learns to trust whichever base model overfit hardest, since overfitting looks like confidence from the meta-learner\'s vantage point. Out-of-fold predictions, generated the same way §2.14\'s cross-validation generates an honest score, are the only signal that reflects real generalisation ability. The raw features alone would defeat the purpose of stacking at all, and the test set is never touched until the whole pipeline — base models and meta-learner together — is finished.'
      }
    ],
    cards: [
      { q: 'Random vs grid', a: 'A d-dimensional grid with m points per axis tests only m distinct values of any one axis; n random draws test n. The advantage grows as $m^{d-1}$ with dimensionality.' },
      { q: 'Bayesian optimisation and TPE', a: 'Fit a cheap surrogate of the objective, sample where it predicts high score or high uncertainty. Pays off when each real evaluation is expensive.' },
      { q: 'Successive halving', a: 'Train many configs cheaply, cut the worst fraction each rung, give survivors more budget. Hyperband hedges against an early noisy cut killing the true best.' },
      { q: 'Stacking’s trick and its cost', a: 'Out-of-fold meta-features exploit disagreement between model families (§2.7’s bagging-variance logic, one level up); the cost is many more artefacts to validate, monitor and explain.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.17 */
  ML.section({
    id: 'interpretability', track: 'classical', num: '2.17',
    title: 'Interpretability, including SHAP',
    lede: 'Not optional in regulated lending: an adverse-action notice needs a reason code, and a reason code needs an attribution that sums to something exact.',
    html: `
<p>An applicant is declined for a loan and, under UK and US consumer credit law alike, is entitled to be told why. "The model scored you as high risk" is not an answer a regulator will accept, and it is not an answer the applicant can act on. What they are owed is something closer to "your credit utilisation is above 70%, and you have had five enquiries in the last six months" — specific, falsifiable, and traceable to an actual number in an actual file. Producing that sentence honestly, for one individual prediction rather than for the model in the abstract, is the entire problem this section solves, and it turns out to need real mathematics, not merely a plausible-sounding story.</p>

<h2><span class="sn">2.17.1</span> The methods, and what each one is actually entitled to claim</h2>

<p><b>Impurity-based feature importance</b> — the number a tree model reports for free, tallying how much each feature reduced impurity across every split that used it — looks like the obvious starting point and is quietly untrustworthy. The bias is mechanical: a continuous feature, or a categorical one with many levels, offers the tree far more candidate split points to try than a binary flag does, and with enough candidates to search over, some split will reduce impurity by chance alone even on a feature carrying no real signal — the same multiple-comparisons logic §2.1.3 built for model selection, now happening inside a single tree's split search. A feature with a thousand possible cut points is a thousand lottery tickets; a binary feature is one.</p>

<p><b>Permutation importance</b> sidesteps this by measuring something computed after training rather than during it: take a model that is already fitted, shuffle one column of the held-out data — breaking that column's relationship with the label while leaving every other column, and the marginal distribution of the shuffled one, untouched — and measure how much a chosen metric drops. A feature the model leaned on heavily produces a large drop when scrambled; a feature it ignored produces almost none. Because the measurement happens on held-out data after training, it inherits none of impurity importance's search-space bias.</p>

<p>It has its own honest weakness, and the mechanism is worth stating precisely rather than waved at. Suppose two features carry nearly identical information — a customer's stated income and their transaction-inferred spending capacity, say, correlated at 0.95. Shuffle income alone, and the model still has spending capacity sitting right there, carrying almost the same signal it would have gotten from income; the metric barely moves, because the model can substitute one for the other on the fly. Shuffle spending capacity alone and the same thing happens in reverse. Both features report low permutation importance individually, even though the <i>pair</i> of them, jointly, might be doing most of the work — permutation importance measures "how much does removing exactly this column hurt," and when a substitute is sitting right next to it, removing one column hurts very little, whatever the pair together contributes.</p>

<p><b>PDP</b> (partial dependence plots) shows, averaged across the whole dataset, how the prediction changes as one feature varies with all others held at their observed values and averaged over — a marginal, dataset-level effect. <b>ICE</b> (individual conditional expectation) plots the identical curve per individual instance rather than averaged, which matters whenever the effect of a feature genuinely differs across people: a PDP can show a flat average effect that hides two subpopulations moving in opposite directions, and only the individual ICE lines reveal that the average was concealing real heterogeneity.</p>

${H.worked('a model where the PDP says "no effect" and is completely wrong', `
<p>A toy model with one genuine interaction: $f(x_1,x_2) = x_1$ when $x_2=0$, and $f(x_1,x_2)=-x_1$ when $x_2=1$ — utilisation ($x_1$) drives the score up for one subgroup and down for the other, in exactly equal and opposite measure, depending on a second feature $x_2$ split evenly across the population.</p>
<p>The PDP for $x_1$ averages over the observed distribution of $x_2$: with $x_2$ split 50/50, $\\mathrm{PDP}(x_1) = 0.5\\,(x_1) + 0.5\\,(-x_1) = 0$ for every value of $x_1$ — a perfectly flat line, reporting that utilisation has <i>no effect on the prediction whatsoever</i>. That conclusion is exactly wrong: utilisation has the largest possible effect on every single individual in the dataset, and the two effects merely happen to cancel in the average. Plot ICE lines instead — one line per instance, holding that instance's own $x_2$ fixed while sweeping $x_1$ — and half the population traces a line with slope $+1$, the other half a line with slope $-1$, fanning out from the same flat PDP average that concealed both. The PDP was not a simplification of the truth here; it was a description of a population that does not exist, built by averaging away the one thing that actually mattered.</p>`)}

<p><b>LIME</b> takes a different approach entirely, local rather than marginal: around one specific prediction, sample many small perturbations of that one instance, weight each sample by its proximity to the original point, and fit a simple, interpretable model — usually linear — to those weighted, perturbed samples. The result approximates how the true model behaves in the immediate neighbourhood of this one applicant, at the cost of being a genuinely different, simpler model each time you ask about a different applicant, with no guarantee its coefficients relate to each other in any consistent way from one explanation to the next.</p>

<h2><span class="sn">2.17.2</span> SHAP — dividing credit the way a fair referee would</h2>

${H.analogy(`<p>Three colleagues ship a project together, and it is a clear success — how much of the credit does each one get? One honest way to answer: imagine every possible order the three of them could have joined the effort, and for each order, ask how much better the project got the moment each person joined, compared to just before. Average that "moment of joining" contribution for each person across every possible joining order, and you have a credit-splitting rule immune to a complaint like "well if I had joined first, I would have gotten more credit" — every possible position in the queue, first, second, last, has already been counted, in equal proportion, for everyone.</p>
<p>This exact idea — average marginal contribution across all orderings — is a genuine 1953 result in cooperative game theory, due to Lloyd Shapley, and Lundberg and Lee's 2017 paper showed that applying it to a model's features, treating "a feature joining the coalition" as "a feature being revealed to the model instead of hidden behind a background average," unifies LIME, DeepLIFT and several older attribution schemes as special or approximate cases of one underlying idea.</p>`)}

<p>Formally, treat the model's features as players in a game where the "payout" of a coalition $S$ of revealed features is $v(S)$, the model's expected output when only the features in $S$ are known and the rest are replaced by their values from a background distribution. The <b>Shapley value</b> $\\phi_i$ of feature $i$ is its average marginal contribution, $v(S\\cup\\{i\\}) - v(S)$, across every possible order in which features could be revealed.</p>

<p>Work this by hand on a toy model small enough to write out every ordering. Three binary risk flags — <code>x1</code> = utilisation above 70%, <code>x2</code> = a recent credit enquiry, <code>x3</code> = tenure under two years — feeding a score with one genuine interaction: being both highly utilised <i>and</i> recently enquiring is worse than the sum of the two flags alone, because it signals someone drawing down credit while actively shopping for more.</p>

${H.worked('the coalition values for one applicant with all three flags raised', `
${H.table(['coalition $S$ (flags known)', 'model output $v(S)$'], [
      ['∅ (nothing known — background average)', '0.10'],
      ['{utilisation}', '0.40'],
      ['{enquiry}', '0.30'],
      ['{tenure}', '0.20'],
      ['{utilisation, enquiry}', '0.75'],
      ['{utilisation, tenure}', '0.50'],
      ['{enquiry, tenure}', '0.40'],
      ['{utilisation, enquiry, tenure}', '0.85']
    ])}
<p>Notice {utilisation, enquiry} scores 0.75, not $0.40+0.30-0.10=0.60$ — the interaction adds an extra 0.15 when both are known together, exactly the "worse together" effect the scenario describes. This is precisely why a simple sum of individual effects cannot be the right attribution rule: it would miss the 0.15 that only exists when both flags are revealed together, and Shapley's averaging-over-orderings is what correctly splits that joint effect between the two features that jointly caused it.</p>`)}

${H.deriv('the exact Shapley values for utilisation, computed over all six orderings', [
      ['ordering U→E→T: marginal contribution of U is $v(\\{U\\})-v(\\emptyset)=0.40-0.10=0.30$', 'U joins first, so its marginal contribution is simply its value alone against the empty coalition — no interaction has fired yet because nothing else is present.'],
      ['ordering E→U→T: marginal contribution of U is $v(\\{E,U\\})-v(\\{E\\})=0.75-0.30=0.45$', 'U now joins second, after E — its marginal contribution includes the full interaction effect, since adding U to a coalition that already contains E is exactly where the 0.15 bonus is realised.'],
      ['ordering T→U→E: marginal contribution of U is $v(\\{T,U\\})-v(\\{T\\})=0.50-0.20=0.30$', 'U joins second again, but after T rather than E — no interaction term involves T, so U\'s marginal contribution here is the same as joining first.'],
      ['Across all six orderings (U→E→T, U→T→E, E→U→T, E→T→U, T→U→E, T→E→U), U\'s marginal contributions are 0.30, 0.30, 0.45, 0.45, 0.30, 0.45.', 'Compute the remaining three orderings the same way, tracking exactly which features have already joined before U does in each case — the full working is mechanical, six lines of subtraction. U scores 0.45 in any ordering where it joins last or second, after the interaction with E has already fired; it scores 0.30 whenever it joins before E regardless of T\'s position, since T carries no interaction term.'],
      ['$\\phi_{U} = \\dfrac{0.30+0.30+0.45+0.45+0.30+0.45}{6} = \\dfrac{2.25}{6} = 0.375$', 'Average U\'s marginal contribution across all six equally-weighted orderings — the definition of the Shapley value, applied directly with no shortcut.']
    ], 'The identical procedure gives $\\phi_E = 0.275$ and $\\phi_T = 0.100$, and the three sum to exactly $0.375+0.275+0.100=0.750$, which equals $v(\\{U,E,T\\})-v(\\emptyset) = 0.85-0.10 = 0.750$ to the last decimal. That equality is not a coincidence of this example — it is a theorem, and it is the property that makes SHAP usable in a legal document.')}

<p>That theorem is called <b>efficiency</b>, and stated in general it is the line every SHAP explanation is required to satisfy:</p>

$$\\sum_i \\phi_i = f(x) - \\mathbb{E}[f]$$

<p>Read it as: the attributions to every feature, added up, account for the <i>entire</i> gap between this prediction and the average prediction — nothing is left over, unexplained, and nothing is double-counted. Alongside efficiency, Shapley values are the <i>unique</i> attribution rule satisfying two further, easy-to-state fairness conditions: <b>missingness</b> (a feature that never affects $v(S)$ for any coalition gets exactly zero credit) and <b>consistency</b> (if a feature's marginal contribution only ever goes up when you compare one model to another, its attribution cannot decrease). Uniqueness is the actual selling point — SHAP is not one plausible way to split credit among several equally defensible options, it is the <i>only</i> way that satisfies those three conditions simultaneously, which is exactly the kind of claim that survives being cross-examined.</p>

<p>Computing this by brute force means summing over every ordering, and the number of orderings is $d!$ — 6 for three features, 24 for four, and already 3,628,800 for ten. <b>TreeSHAP</b> exploits the specific structure of a decision tree to compute the exact same quantity in time polynomial in the number of trees, leaves and depth, by tracking, for every path through the tree, how many orderings are consistent with that path having been reached — a combinatorial shortcut, not an approximation, which is precisely why it is the default for gradient-boosted models. <b>KernelSHAP</b> makes no assumption about the model at all: it samples a manageable number of coalitions, computes $v(S)$ for each by querying the model, and solves a specially-weighted linear regression whose coefficients converge to the Shapley values as the sample grows — general-purpose, but a sampling approximation, and considerably slower than TreeSHAP's exact polynomial-time route on the tree ensembles this course ships most often.</p>

<p><b>What you are looking at.</b> A waterfall chart starting from $\\mathbb{E}[f(x)]$, the base rate, on the left. Each bar is one feature's exact Shapley value $\\phi_i$ for the current applicant, ordered by magnitude, red for driving the score up and blue for pulling it down, stacking left to right until the final bar lands exactly on $f(x)$, this applicant's actual prediction.</p>

<p><b>What to do with it.</b> Drag any slider — utilisation, enquiries, tenure, income band — and watch every bar in the waterfall resize at once, not just the one you touched: because the underlying model has genuine interactions, changing one feature changes the marginal contribution every other feature is credited with, exactly as the hand-worked example above showed for utilisation and enquiry together.</p>

<p><b>The thing genuinely worth noticing.</b> Watch the "Σφᵢ (must equal the gap)" and "efficiency holds?" readouts at every setting you try. They match to machine precision, always — this is 24 orderings of four real features being summed in your browser, live, not a stylised illustration, and the efficiency property is not asserted anywhere in this lab, it falls out of the arithmetic every single time.</p>

${H.lab('shap', 'Exact Shapley values, computed over all orderings', 'Four features, sixteen coalitions, twenty-four orderings — small enough to compute the definition directly rather than approximate it. Change the inputs and watch the waterfall rebalance, always summing exactly to the prediction minus the base rate.')}

<h2><span class="sn">2.17.3</span> Reason codes, mechanically</h2>

<p>Turning Shapley values into the sentence a declined applicant is legally owed is a fixed pipeline, not a creative-writing exercise. Compute per-feature SHAP values for that one prediction; keep only the features pushing the score in the adverse direction, since a factor that helped the applicant is not a reason for the decline; rank the adverse features by the magnitude of their attribution; map each to a plain-language reason drawn from a fixed, pre-reviewed dictionary; and return the top three or four.</p>

<p>Two details matter, and both come up in interviews precisely because they sound like minor implementation choices and are actually the whole point. First, the mapping must run <b>from bins, not raw features or raw SHAP values</b> — "utilisation above 70%" is a reason a human being can read and act on; "utilisation SHAP +0.083" is a number that means nothing to anyone without a machine learning background, and handing it to an applicant satisfies the letter of a disclosure requirement while failing its entire purpose. Second, the dictionary of plain-language reasons is fixed in advance and reviewed by compliance before it ever reaches an applicant, so that the same underlying driver always produces the same wording, every time, for every applicant — generating the explanation text freely, with an LLM or otherwise, is exactly the wrong place to introduce any variability at all, because two applicants declined for identical reasons receiving differently-worded explanations is itself a fairness problem waiting to happen (§2.19).</p>

<h2><span class="sn">2.17.4</span> Global versus local, and the honest caveats</h2>

<p><b>Global importance</b> is the mean absolute SHAP value for a feature, averaged across a whole sample of predictions — it answers "which features matter most, on average, across the book." A single waterfall answers a narrower and, for an individual applicant, more relevant question: what explains <i>this</i> decision. The two can and do disagree, and the disagreement is informative rather than a bug: a feature can be globally minor — small average absolute SHAP, buried near the bottom of a global importance ranking — while being locally decisive for a specific handful of applicants whose circumstances happen to trigger it strongly. That asymmetry is also the cleanest way to state why permutation importance and SHAP sometimes rank features differently: permutation importance is inherently a global, metric-level measurement that lets correlated features substitute for each other and therefore under-credits both; SHAP allocates credit prediction by prediction, and a feature that matters enormously for a minority of applicants and not at all for the rest can show up as globally modest in one framework and locally dominant in the other, with neither number being wrong.</p>

<p><b>Counterfactual explanations</b> answer the question applicants actually ask, which is rarely "why," it is "what would I need to change." "Reduce utilisation below 55% and this decision would flip" is more directly useful than any attribution, precisely because it prescribes an action rather than describing a cause. It is also more dangerous to get wrong, because it is a promise about how the deployed model will behave in the future, not a description of how it behaved just now — it must be computed against the actual production model, respecting real feasibility constraints (an applicant cannot reduce their age, and should never be offered a counterfactual that pretends they can), and it stops being true the moment the model is retrained without the counterfactual engine being refreshed alongside it.</p>

${H.flag('SHAP explains the model, never the world. If two features are correlated, the credit split between them is an artefact of the sampling scheme, not a statement about which one the tree "really" used — take two perfect substitutes with $v(\\{1\\})=v(\\{2\\})=v(\\{1,2\\})=0.5$ and $v(\\emptyset)=0$: the Shapley values split evenly at $\\phi_1=\\phi_2=0.25$ each, by the same averaging-over-orderings mechanism worked above, regardless of whether the tree\'s splits happened to lean on feature 1 far more than feature 2 in practice. TreeSHAP\'s default handling of correlated features can therefore attribute importance to a feature the tree never really used, purely because a substitute was available. Interventional versus conditional expectations for the background distribution give different answers to the same question and the difference is not cosmetic — always state what your numbers are: attributions of <i>this model\'s output</i>, under a stated background distribution, never a causal claim about the applicant\'s actual risk.')}

${H.probe([
      ['TreeSHAP vs KernelSHAP?', 'TreeSHAP is exact — not approximate — and polynomial-time for tree ensembles, by exploiting the combinatorial structure of paths through the tree; KernelSHAP makes no model assumption and instead samples coalitions and solves a weighted regression, which converges to the true Shapley values but is a sampling approximation and considerably slower. Use TreeSHAP whenever the model is a tree ensemble, which in a lending pipeline is most of the time.'],
      ['Why do permutation importance and SHAP disagree?', 'Permutation importance is a global measurement that lets correlated features substitute for each other, under-crediting both; SHAP allocates credit locally, per prediction, so a feature that is globally modest but locally decisive for a subset of applicants can rank very differently between the two. Neither is wrong — they are answering "how much does removing this column hurt the metric, on average" versus "how much did this feature contribute to this one prediction," which are genuinely different questions.'],
      ['What makes SHAP usable for adverse-action notices?', 'Efficiency: the attributions sum exactly to the prediction minus the base value, verified to machine precision on any input, so the explanation is complete by construction — nothing about the decision is left unaccounted for, which is precisely the property a regulator or a court will test first.']
    ], 'Presenting SHAP values for correlated features as independent causal effects. They are attributions of this model\'s output, under a stated background distribution, not statements about the world or about which feature the model "truly" relied on.')}`,
    labs: {
      shap: function (host) {
        const feats = ['utilisation', 'recent enquiries', 'tenure (yrs)', 'income band'];
        const st = Viz.controls(host, [
          { k: 'f0', label: 'utilisation', min: 0, max: 1, step: .01, value: .94, fmt: v => v.toFixed(2) },
          { k: 'f1', label: 'recent enquiries', min: 0, max: 10, step: 1, value: 5, fmt: v => v },
          { k: 'f2', label: 'tenure (years)', min: 0, max: 20, step: 1, value: 9, fmt: v => v },
          { k: 'f3', label: 'income band', min: 1, max: 10, step: 1, value: 6, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'base', label: 'E[f(x)] base rate' }, { k: 'pred', label: 'f(x) prediction', cls: 'key' },
          { k: 'sum', label: 'Σφᵢ (must equal the gap)' }, { k: 'check', label: 'efficiency holds?' }
        ]);
        // a small non-linear model with interactions, plus a background distribution
        const R = Num.rng(71);
        const background = Array.from({ length: 60 }, () => [R(), R.int(11), R.int(21), 1 + R.int(10)]);
        function model(x) {
          const z = -2.1 + 2.9 * x[0] + 0.22 * x[1] - 0.07 * x[2] - 0.14 * x[3] + 0.9 * x[0] * (x[1] > 3 ? 1 : 0);
          return Num.sigmoid(z);
        }
        function shapExact(x) {
          const n = 4;
          const phi = new Array(n).fill(0);
          // v(S) = E_background[f(x_S, X_-S)]
          const cache = {};
          function v(S) {
            const key = S.join(',');
            if (cache[key] !== undefined) return cache[key];
            const val = Num.mean(background.map(b => {
              const z = [0, 1, 2, 3].map(i => S.indexOf(i) >= 0 ? x[i] : b[i]);
              return model(z);
            }));
            cache[key] = val; return val;
          }
          const perms = [];
          (function permute(arr, cur) {
            if (!arr.length) { perms.push(cur); return; }
            arr.forEach((a, i) => permute(arr.filter((_, j) => j !== i), cur.concat([a])));
          })([0, 1, 2, 3], []);
          perms.forEach(p => {
            const S = [];
            p.forEach(i => { const before = v(S.slice()); S.push(i); phi[i] += v(S.slice()) - before; });
          });
          return { phi: phi.map(p => p / perms.length), base: v([]), pred: v([0, 1, 2, 3]) };
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const x = [st.f0, st.f1, st.f2, st.f3];
            const { phi, base, pred } = shapExact(x);
            const order = [0, 1, 2, 3].sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
            const x0 = 150, bw = w - x0 - 90;
            const scale = bw / Math.max(.35, Math.abs(pred - base) * 2.4);
            let cum = base;
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const zeroX = x0 + bw * .18;
            ctx.fillStyle = T.muted; ctx.textAlign = 'right';
            ctx.fillText('E[f(x)] = ' + base.toFixed(3), x0 - 10, 26);
            ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(zeroX, 16); ctx.lineTo(zeroX, h - 40); ctx.stroke();
            order.forEach((i, k) => {
              const y = 52 + k * 34;
              const wpx = phi[i] * scale;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText(feats[i] + ' = ' + (i === 0 ? x[i].toFixed(2) : x[i]), x0 - 10, y);
              ctx.fillStyle = phi[i] >= 0 ? T.red : T.blue;
              const startX = zeroX + (cum - base) * scale;
              ctx.fillRect(Math.min(startX, startX + wpx), y - 10, Math.abs(wpx), 20);
              ctx.fillStyle = T.text; ctx.textAlign = phi[i] >= 0 ? 'left' : 'right';
              ctx.fillText((phi[i] >= 0 ? '+' : '') + phi[i].toFixed(3), startX + wpx + (phi[i] >= 0 ? 6 : -6), y);
              cum += phi[i];
            });
            const yf = 52 + 4 * 34 + 10;
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'right';
            ctx.fillText('f(x) = ' + pred.toFixed(3), x0 - 10, yf);
            ctx.fillStyle = T.text;
            ctx.fillRect(zeroX, yf - 9, Math.max(2, (pred - base) * scale), 18);
            out({
              base: base.toFixed(4), pred: pred.toFixed(4),
              sum: Num.sum(phi).toFixed(4),
              check: Math.abs(Num.sum(phi) - (pred - base)) < 1e-6 ? 'yes — exactly' : 'no'
            });
          }
        });
        Viz.note(host, 'These are exact Shapley values: all 24 orderings, evaluated against a 60-row background sample. Note the efficiency check — the attributions sum to the prediction minus the base value to machine precision, which is the property that makes them defensible in an adverse-action notice.');
      }
    },
    quiz: [
      {
        q: 'Impurity-based feature importance is biased toward…',
        options: ['binary features', 'high-cardinality and continuous features', 'the target-correlated features only', 'features with missing values'],
        answer: 1,
        why: 'A continuous or high-cardinality feature offers the tree far more candidate split points to search over, and with enough candidates some split reduces impurity by chance alone — the same multiple-comparisons mechanism §2.1.3 built for model selection, here operating inside a single tree\'s split search. A binary feature has one possible split and no equivalent lottery. Permutation importance, measured after training on held-out data, does not inherit this bias, which is why it is the safer default.'
      },
      {
        q: 'SHAP’s efficiency property states that…',
        options: 'the attributions are fast to compute|the attributions sum to f(x) − E[f]|correlated features get equal credit|the model must be a tree'.split('|'),
        answer: 1,
        why: 'The attributions to every feature add up to exactly the gap between this prediction and the base rate — verified to machine precision in the worked example and the lab alike. That completeness, not speed, is what makes an explanation defensible: nothing about the decision is left unaccounted for. Speed is TreeSHAP\'s property specifically, not SHAP\'s in general (KernelSHAP is slow), and "correlated features get equal credit" is only true for perfect substitutes, not correlated features generally.'
      },
      {
        q: 'A reason code should be phrased as…',
        options: ['"utilisation SHAP +0.083"', '"utilisation above 70%" — mapped from bins via a fixed reviewed dictionary', 'a free-text LLM explanation', 'the feature name alone'],
        answer: 1,
        why: 'A raw SHAP value satisfies the letter of a disclosure requirement while failing its purpose entirely — no applicant can act on "+0.083." Binning maps the number to something a human can read and respond to, and fixing the dictionary in advance guarantees the same driver always produces identical wording for every applicant, which matters for consistency (§2.19) as much as for clarity. Free text, however fluent, reintroduces exactly the variability a reviewed dictionary exists to remove.'
      }
    ],
    cards: [
      { q: 'SHAP efficiency', a: '$\\sum_i\\phi_i = f(x)-\\mathbb{E}[f]$ — attributions sum exactly to the deviation from the base rate. Uniquely guaranteed alongside missingness and consistency.' },
      { q: 'TreeSHAP vs KernelSHAP', a: 'Exact and polynomial-time for tree ensembles, via the combinatorics of tree paths, vs model-agnostic sampling approximation for any model.' },
      { q: 'The SHAP caveat', a: 'It explains the model under a stated background distribution, not the world; perfect substitute features split credit evenly regardless of which the tree actually leaned on.' },
      { q: 'Permutation importance’s blind spot', a: 'Two correlated features can each substitute for the other when shuffled, so both report low importance individually even though the pair matters a great deal.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.18 */
  ML.section({
    id: 'production', track: 'classical', num: '2.18',
    title: 'Production lifecycle, drift, survival, time series',
    lede: 'Where the previous seventeen sections either survive contact with reality or do not. §2.14’s validation discipline and §2.1’s split logic keep reappearing here, now under a clock that never stops.',
    html: `
<p>Every section so far has quietly assumed the world holds still long enough to fit a model to it, and that the moment you compute a feature is the same moment it will be computed again when the model actually runs. Neither assumption survives contact with a production system for more than a few months. Customers change, fraud rings adapt, macroeconomic conditions shift, and — the failure that catches out more teams than any of the other three combined — the code that computes a feature at 2am in a batch training job and the code that computes the "same" feature in a 200-millisecond API call at decision time turn out not to be the same code at all. This section is the discipline for staying honest once the model has left the notebook.</p>

<h2><span class="sn">2.18.1</span> The feature store, and the failure it exists to prevent</h2>

<p>Here is the failure in its purest form, small enough to compute by hand. A "5-day average transaction amount" feature is defined once, in a specification document, and implemented twice — once by the data science team building the training pipeline, once by the platform team building the serving path — and the two implementations disagree about whether "the last 5 days" includes today.</p>

${H.worked('the exact size of a one-day window bug', `
<p>Six days of transaction totals: day 1 through day 6 (day 6 is today, the day a decision is being made): £100, £120, £95, £130, £88, £150.</p>
<p><b>Training</b> computed this feature historically, on data where "today" had already fully passed, using the five days <i>strictly before</i> the decision day: $(100+120+95+130+88)/5 = 533/5 = \\mathbf{£106.60}$.</p>
<p><b>Serving</b> computes the same feature live, and its window definition off-by-one includes today itself: $(120+95+130+88+150)/5 = 583/5 = \\mathbf{£116.60}$.</p>
<p>A single boundary convention, decided differently by two people implementing "the same" feature from the same English sentence, produces a <b>£10.00 gap — a 9.4% relative difference</b> — on every single prediction the model will ever make in production, silently, with no error thrown anywhere, because both numbers are perfectly valid floating-point averages of perfectly valid transaction data.</p>`)}

<p>That is <b>training–serving skew</b>: the model is exactly as good as it was when validated, and the pipeline feeding it live data is quietly feeding it something different from what it was trained on. It is the most common production failure in machine learning and the least visible, because nothing crashes, nothing logs an error, and every individual number involved looks completely reasonable in isolation. The only symptom is a model that mysteriously underperforms its validation numbers from day one, for a reason nobody thinks to check because "the model" is the natural first suspect and is, in this story, entirely innocent.</p>

<p>A <b>feature store</b> is the infrastructure answer: one transformation definition, computed once, served two ways. The <b>offline store</b> is columnar, historical, and built for the very specific correctness property training requires — <b>point-in-time correctness</b>, meaning that when you ask "what was this feature's value as of decision moment $t$," the answer uses only data that genuinely existed at or before $t$, never a value computed with the benefit of hindsight. This is §2.11's leakage checklist, expressed as an infrastructure guarantee rather than a habit of the analyst writing the query. The <b>online store</b> is a low-latency key-value lookup, built for the very different property serving requires — answer in milliseconds, from whatever the feature's current value happens to be right now.</p>

${H.key('A feature store is one transformation definition fed to two different storage engines. If the offline and online paths are computed by separate code, you have rebuilt training–serving skew inside the very tool bought to eliminate it — the split between offline and online storage is the whole point, but only if a single definition feeds both.')}

<h2><span class="sn">2.18.2</span> Two kinds of drift, and only one you can see immediately</h2>

<p><b>Data drift</b> is a change in $P(x)$ — the population of applicants, transactions or accounts arriving at the model looks different from what it was trained on, whatever the true relationship between $x$ and $y$ happens to be. <b>Concept drift</b> is a change in $P(y\\mid x)$ — the relationship itself has shifted, so the same applicant who would have been low-risk last year is genuinely higher-risk this year, for reasons the model's inputs do not capture (a change in interest rates, a new fraud technique, a shift in regulation). The asymmetry between the two is what makes drift monitoring genuinely hard rather than a box-ticking exercise: data drift is visible the instant new inputs arrive, computed from features alone, while concept drift is invisible until outcomes are observed — and in consumer credit, an outcome can take twelve months to resolve. You can, in principle, be flying blind on concept drift for a year before the evidence needed to detect it even exists.</p>

<p>Monitor data drift with the <b>population stability index</b>, applied to each feature by comparing its distribution today against its distribution at training time, using the same binning for both:</p>

$$\\mathrm{PSI} = \\sum_i (a_i - e_i)\\ln\\frac{a_i}{e_i}$$

<p>Read this exactly the way you read IV in §2.11, because it is the identical computation with different inputs: $e_i$ is the expected (training-time) share of the population in bin $i$, $a_i$ is the actual (current) share in the same bin, and each bin contributes a term that is zero when the two shares match exactly and grows — always non-negatively, by the same Jeffreys-divergence structure §2.11 derived for IV — the further apart they are. Continuous features get the same treatment via KS, comparing cumulative distributions rather than binned ones.</p>

${H.worked('a PSI computation carried through, bin by bin', `
${H.table(['bin', 'expected share $e_i$', 'actual share $a_i$', 'contribution $(a_i-e_i)\\ln(a_i/e_i)$'], [
      ['1 (lowest)', '0.10', '0.05', '0.0347'],
      ['2', '0.20', '0.15', '0.0144'],
      ['3', '0.40', '0.30', '0.0288'],
      ['4', '0.20', '0.30', '0.0405'],
      ['5 (highest)', '0.10', '0.20', '0.0693']
    ])}
<p>$\\mathrm{PSI} = 0.0347+0.0144+0.0288+0.0405+0.0693 = \\mathbf{0.1877}$ — inside the conventional "moderate" band. Notice bin 5 alone contributes 0.0693, thirty-seven per cent of the total, from a population share that merely doubled, 10% to 20%: PSI is dominated by whichever bin moved the most in <i>relative</i> terms, not the bin that moved the most in absolute population share — bin 4 moved by the same 10 percentage points as bin 5 but contributes 42% less, because doubling from 10% to 20% is a bigger relative jump than going from 20% to 30%. This is exactly why the per-bin breakdown, not the single total, is the actual diagnostic: it tells you a new segment or a narrow pipeline change moved into bin 5 specifically, which "PSI = 0.19" alone never could.</p>`)}

<p>Conventional bands read PSI below 0.1 as stable, 0.1–0.25 as moderate drift worth investigating, above 0.25 as a significant shift that should trigger action.</p>

${H.flag('That is the same Jeffreys-divergence form as IV in §2.11, with different inputs (expected-vs-actual population shares here, good-vs-bad class shares there) — and the 0.1/0.25 thresholds are industry convention, not derived from any type-I error rate or hypothesis test. State them as convention and you sound like someone who has actually read the maths behind the number, rather than someone reciting a rule of thumb as though it were a proof.')}

<p><b>What you are looking at.</b> Two overlaid histograms across the same bins — blue for the expected, training-time distribution, red for the actual, current one — with amber labels marking whichever bins are contributing most heavily to the PSI total shown in the readout.</p>

<p><b>What to do with it.</b> Drag the mean-shift slider slowly away from zero and watch the PSI readout climb through the conventional bands one at a time, while the amber labels concentrate on whichever tail is moving. Then reset the shift to zero and instead widen the spread — PSI still rises, from a population that has not moved on average at all, just become more dispersed, which is a genuinely different kind of drift than a shift and worth being able to tell apart from the shape of the two histograms rather than the PSI number alone.</p>

<p><b>The thing genuinely worth noticing.</b> Produce a PSI just above 0.25 two different ways — once with a large mean shift and narrow spread, once with no mean shift and a large spread change — and compare the amber-highlighted bins in each case. The total can be identical while the diagnosis is completely different: one is "a new type of applicant has arrived," the other is "the same applicants have become more heterogeneous." A PSI figure without its per-bin breakdown collapses that distinction, which is exactly why a validator will always ask to see the bins, not just the summary statistic.</p>

${H.lab('psi', 'PSI, computed on two distributions you control', 'Drag the actual distribution away from the expected one and watch PSI cross the conventional bands. The per-bin contributions show which part of the population moved — which is the diagnostic, not the total.')}

<h2><span class="sn">2.18.3</span> Retraining and rollout, as a written policy rather than a judgement call</h2>

<p>Decide the retraining trigger before you need it, not while staring at a PSI dashboard wondering if today is the day. The usual triggers are a drift breach on a monitored feature, sustained metric decay past a pre-agreed band, or simply a calendar cadence — quarterly, say — regardless of whether drift has been detected, because concept drift can be invisible for months and a calendar trigger is the backstop for exactly that blind spot. Writing the trigger down in advance, before deployment, converts "should we retrain" from a judgement call made under pressure into a policy anyone can check the model against.</p>

<p>Roll a new model out in stages, each one buying evidence at increasing cost of being wrong. <b>Shadow mode</b> scores every live request and logs the result without acting on it at all — the cheapest possible way to see how a candidate model would have behaved on real, current traffic. <b>Canary</b> gives the new model a small, genuine share of live decisions, so any failure is contained to a small slice of the business rather than all of it. <b>Champion–challenger</b> runs both models properly, side by side, on a real and ongoing basis, as the final stage before the challenger becomes the new champion outright.</p>

<p>For a regulated lender, model risk governance in the SR 11-7 mould expects a specific shape of documentation before any of this happens, and a validator will ask for it section by section:</p>

${H.fig('WHAT A MODEL VALIDATOR ASKS FOR (SR 11-7 SHAPE)', H.table(['Section', 'What they want'], [
      ['<b>Conceptual soundness</b>', 'Why this model class, why these features, what is the economic rationale for each sign? Monotonic constraints (§2.8) answer half of this before it is asked.'],
      ['<b>Data lineage</b>', 'Where each feature came from, its availability at decision time, the treatment of missing values, and evidence there is no leakage (§2.11).'],
      ['<b>Outcome analysis</b>', 'Performance on a genuine out-of-time window (§2.14), by segment, with calibration (§2.12) and stability — not one aggregate AUC.'],
      ['<b>Benchmarking</b>', 'Against the incumbent and against a simple challenger; a logistic scorecard is the standard yardstick, and if boosting cannot beat it by a defensible margin, ship the scorecard.'],
      ['<b>Ongoing monitoring plan</b>', 'Which metrics, which thresholds, what happens when one breaches, and who owns it — written before deployment.'],
      ['<b>Limitations, stated by you</b>', 'Where the model should not be used. Validators trust a document that names its own weaknesses far more than one that does not.']
    ]))}

<h3>Champion–challenger, properly — and the trap that is easy to walk into blind</h3>

<p>The challenger must run on a <b>randomised</b> slice of live traffic, not merely "whatever the challenger happens to have scored" — randomisation is what turns the comparison into a causal read on the decision, in exactly the sense §1.7 built for any intervention, rather than an observational backtest confounded by whoever the challenger happened to see.</p>

<p>Two traps sit specifically in this stage, and both are easy to walk into without noticing. The first is a selection effect with a name: <b>reject inference</b>. Whatever policy is currently live only ever observes outcomes for the applicants it approved — the declined 30%, say, never take out the loan, so you never learn whether they would have defaulted. If you then compare a challenger's apparent performance using only the outcomes you have — the approved population — you are measuring both models on a systematically safer, narrower slice of the risk spectrum than either would actually face making live accept/decline decisions across the full applicant pool. A ranking metric computed inside a truncated, pre-screened range is a different, and generally worse, statement about discrimination than the same metric computed across the full range the model was actually built to judge — the genuinely hard cases the incumbent already screened out are simply invisible to the comparison. Reject inference is the credit-specific set of techniques — parcelling, augmentation, extrapolation from a small randomly-approved-anyway sample — for making an informed guess about the unobserved rejects' outcomes, and every one of them rests on an assumption you cannot fully verify, which is exactly why running a genuinely randomised acceptance slice, even a small one, is the only clean way to see outcomes across the whole score range rather than inferring them.</p>

<p>The second trap is timing: outcomes in consumer credit arrive months late, so the evaluation window and the success criterion for the champion–challenger test must be fixed <i>before</i> the test starts. Leave them open and you will be tempted into exactly the peeking problem §1.6 built the machinery to diagnose, now stretched out over a twelve-month lag that makes the temptation to check early, and stop the moment the numbers look good, far harder to resist than a same-day A/B test ever is.</p>

<p>There is a question worth asking before the test even begins, and it is rarely asked out loud: is the canary slice even big enough to tell the two models apart? §2.1.3 built the standard error of a measured AUC, $\\mathrm{se} \\approx 0.5/\\sqrt{n/4}$ for a test population of size $n$, to show how a search over many models manufactures optimism out of nothing. The identical formula answers a completely different, entirely practical question here: how many outcomes does a champion–challenger test need before its noise is smaller than the improvement you are hoping to detect?</p>

${H.worked('how large a champion–challenger test needs to be', `
<p>Suppose the challenger is hoped to beat the champion by a genuine 0.02 of AUC — a real, worthwhile improvement, not a rounding error. To resolve a gap of that size with any confidence, the standard error of the measurement needs to be comfortably smaller than the gap itself, say $\\mathrm{se}\\approx0.005$, a quarter of the effect being chased.</p>
<p>Solving $0.005 = 0.5/\\sqrt{n/4}$ for $n$ gives $\\sqrt{n/4} = 100$, so $n = 40{,}000$ resolved outcomes — not applications, <i>outcomes</i>, meaning fully-matured accounts with a known good/bad label. For comparison, $n=500$ outcomes — a perfectly respectable canary slice by application-volume standards — carries $\\mathrm{se}\\approx0.045$, nine times too noisy to distinguish a genuine 0.02 improvement from pure sampling variation.</p>
<p>This is the arithmetic behind a fact every credit risk team eventually learns the hard way: a champion–challenger test that looks statistically clean after a month of live traffic is very often still running on far too few <i>mature</i> outcomes to say anything about a modest AUC improvement, because outcomes accrue at the rate accounts season, not at the rate applications arrive. Pre-committing to a sample size, using exactly this formula, before the test starts is what stops "we don't have enough data yet" from being discovered only after the evaluation window has already closed.</p>`)}

<h2><span class="sn">2.18.4</span> Survival analysis, because default is a question of "when," not just "whether"</h2>

<p>The standard credit framing — "did this account default within 12 months, yes or no" — throws away two things at once, and the second is easy to miss. It throws away <i>when</i> the default happened, treating a default in month 2 identically to one in month 11. And it throws away every account whose 12-month window has not finished yet: an account opened four months ago has not had the chance to default in months 5 through 12, and scoring it "no" because it has not defaulted <i>yet</i> silently treats "hasn't happened yet" as "won't happen," which is a different claim entirely. Those incomplete accounts are <b>censored</b>, not negative, and the gap between the two is not a rounding error.</p>

${H.worked('exactly how much a naive rate understates the true 12-month default rate', `
<p>Ten accounts, with months observed and outcome. Five reached the full 12 months with no default (genuine goods): A, B, C, E, J. Three defaulted, at months 3, 6 and 9: D, F, H. Two were only booked recently and have accumulated 2 and 4 months of clean history respectively, with no default <i>so far</i>, but no way yet to know whether they will default in the months they have not lived through: G and I.</p>
<p>The <b>naive rate</b> counts every account without an observed default as good, including the two that have not had the chance to fail yet: $3/10 = \\mathbf{30.0\\%}$.</p>
<p>The <b>Kaplan–Meier estimate</b> instead only ever compares accounts to others still genuinely at risk at each moment, removing an account from the risk set the instant it is censored rather than assuming it survives to month 12:</p>
${H.table(['time', 'event or censor', 'at risk just before', 'survival factor', 'running $S(t)$'], [
      ['2', 'I censored', '10', '(no change)', '1.0000'],
      ['3', 'D event', '9', '$1-1/9=0.8889$', '0.8889'],
      ['4', 'G censored', '8', '(no change)', '0.8889'],
      ['6', 'F event', '7', '$1-1/7=0.8571$', '0.7619'],
      ['9', 'H event', '6', '$1-1/6=0.8333$', '0.6349'],
      ['12', 'A,B,C,E,J censored (study ends)', '5', '(no change)', '0.6349']
    ])}
<p>$S(12) = 0.6349$, so the Kaplan–Meier 12-month default rate is $1-0.6349 = \\mathbf{36.5\\%}$ — more than six percentage points above the naive figure, entirely because accounts G and I, whose fate is genuinely unresolved, were correctly excluded from the risk set at the moment they stopped being observable rather than silently counted as successes.</p>`)}

<p>Read the KM recursion in words: at each moment an event happens, multiply the running survival probability by "the fraction of those still genuinely at risk who survived this particular moment." A censored account leaves the risk set at the moment it is censored — it stops contributing to the denominator of every future step — but it is never credited as a survivor of moments it did not live through. That single design choice is the entire difference between 30.0% and 36.5% above, and it generalises to $S(t) = P(T>t)$, the non-parametric survival curve, which is the right first plot for comparing any two cohorts before reaching for a parametric model at all.</p>

<p><b>Cox proportional hazards</b> is the parametric next step: it models the hazard — the instantaneous chance of the event given survival so far — as a baseline hazard shared by everyone, scaled by a factor that depends on each account's covariates:</p>

$$h(t\\mid x) = h_0(t)\\exp(\\beta^\\mathsf{T}x)$$

<p>Read $h_0(t)$ as "whatever the population's baseline risk shape looks like over time, left completely unspecified," and $\\exp(\\beta^\\mathsf{T}x)$ as a single multiplier, the same at every point in time, that scales that baseline up or down depending on one account's features. Because the two pieces multiply, $e^{\\beta_j}$ has a clean, committee-friendly reading as a <b>hazard ratio</b>: it is the factor by which the instantaneous default rate changes for a one-unit increase in covariate $j$, holding everything else fixed, at <i>every</i> point in the loan's life simultaneously.</p>

${H.worked('reading a Cox coefficient as a hazard ratio', `
<p>Suppose a Cox model fits $\\beta_{\\text{recent enquiry}} = 0.336$ for the flag "an enquiry in the last three months." The hazard ratio is $e^{0.336} = \\mathbf{1.40}$: an account with a recent enquiry has a 40% higher instantaneous default hazard than an otherwise identical account without one, at every month of the loan's life. That reads almost exactly like an odds ratio from logistic regression (§2.4) — 40% higher odds — and for good reason: both are exponentiated linear coefficients, the difference being that Cox's multiplier applies to a hazard rate at every instant rather than to a single fixed-horizon odds.</p>`)}

<p>What survival analysis buys over the plain 12-month binary is threefold: it uses every partially-observed account properly rather than discarding or mis-scoring it; it gives you an explicit time axis, so the same underlying model can price a 6-month product and a 36-month product without being refit from scratch for each; and it maps naturally onto <b>lifetime expected credit loss</b> under IFRS 9, which is fundamentally a question about the whole shape of the survival curve, not a single point on it.</p>

<p>What it obliges you to check is the assumption baked into the name: <b>proportional hazards</b> means the ratio between any two accounts' hazards stays constant over time, and that is a real, falsifiable assumption rather than a formality. If a covariate's effect genuinely changes shape over the life of a loan — a recent-enquiry flag that predicts risk sharply in month 2 but says almost nothing by month 24, say — the plain Cox model is misspecified, and the standard diagnostic is to test the covariate's <b>Schoenfeld residuals</b> for a trend over time; a flat trend supports proportionality, a sloped one contradicts it. Know the modern alternative too: gradient boosting fitted against a survival objective — Cox's partial likelihood, or an accelerated-failure-time loss — typically beats a linear Cox model on raw discrimination while keeping the same honest treatment of censoring, at the usual cost of trading away the clean, single-coefficient hazard-ratio story a committee can read off a table.</p>

<p><b>What you are looking at.</b> Two Kaplan–Meier survival curves, one per cohort, each starting at $S(0)=1$ and stepping down at every genuine default event within the 12-month window; short tick marks along each curve mark where an account was censored rather than defaulted. The readout compares the naive default rate — treating every censored account as a survivor — against the Kaplan–Meier estimate on the same simulated data.</p>

<p><b>What to do with it.</b> Raise the censoring-rate slider and watch the gap between "naive" and "Kaplan–Meier" in the readout widen steadily — more accounts with genuinely unresolved fates means more that the naive method silently counts as goods, exactly the mechanism the worked table above computed by hand on ten accounts.</p>

<p><b>The thing genuinely worth noticing.</b> Push censoring high and compare how visually smooth and honest the KM curve remains against how badly the single naive percentage misleads at the same setting. In a 12-month PD model with heavy attrition — accounts closing early, customers churning to a competitor before their window completes — this is not a hypothetical distortion; it is a systematic optimism that no amount of feature engineering fixes, because the naive rate's problem was never about the features, it was about which accounts were allowed to count as evidence at all.</p>

${H.lab('km', 'Kaplan–Meier and censoring', 'Two cohorts, with censoring you control. Watch what happens to the naive "default rate" when censored accounts are scored as negatives — and watch the KM curve stay honest.')}

<h2><span class="sn">2.18.5</span> Time series and recommenders, briefly</h2>

<p>A model trained on a time series inherits an assumption every method in this course up to now has taken for granted without saying so: that the statistical relationship you are fitting today will still hold tomorrow. For ordinary tabular data that assumption is implicit in the very idea of a held-out test set (§2.1). For a time series it has a name, <b>stationarity</b>, and it can fail in an obvious way worth naming precisely: a series with a genuine upward trend has a mean that keeps moving, so "the average value" is not one number, it is a different number every month, and a model implicitly assuming a fixed mean will systematically under-forecast a growing series and over-forecast a shrinking one. <b>Differencing</b> — modelling the change from one period to the next rather than the level itself — often restores something closer to stationary behaviour, because a trend that grows steadily produces roughly constant differences even though the level itself never stops moving.</p>

<p>Three approaches see live use, and the choice between them is mostly about how many series you have and how much structure you can afford to hand-engineer. <b>Classical ARIMA and exponential smoothing (ETS)</b> suit a handful of well-behaved series where a statistician can look at each one individually — they are interpretable, cheap, and genuinely strong when a series is short and the classical assumptions roughly hold. <b>Gradient-boosted models over lag and calendar features</b> — yesterday's value, last week's value, day-of-week, is-a-holiday — turn forecasting into an ordinary tabular regression problem, and this is usually the strongest approach per unit of engineering effort for a business with dozens to hundreds of related series, because one model learns shared structure across all of them at once rather than fitting each in isolation. <b>Global deep models</b> earn their keep specifically when you have thousands of related series and enough shared structure between them that a single large model can learn patterns individual series are too short to reveal on their own — a single store's Tuesday sales history is noisy; ten thousand stores' Tuesday patterns, learned jointly, are not. Whichever approach you choose, backtest it with rolling origin (§2.14) — a time series is precisely the structure that method was built to validate honestly, and nothing about forecasting exempts it from that discipline.</p>

<p><b>Recommenders</b> sit in a related but distinct family: collaborative filtering and matrix factorisation learn from implicit feedback — clicks, purchases, dwell time — to place users and items in a shared space where proximity predicts affinity, with <b>cold start</b>, the problem of a brand-new user or item with no interaction history yet, handled by falling back on content features (a new item's category and price, a new user's stated preferences) until enough behavioural signal accumulates to let the collaborative signal take over.</p>

${H.probe([
      ['Data vs concept drift?', 'Inputs shifting, $P(x)$, versus the input–output relationship shifting, $P(y\\mid x)$. PSI and KS catch the first immediately, from features alone; the second only becomes visible once labels arrive, which in credit can be months — meaning you can be blind to concept drift for a long stretch before any evidence to detect it even exists.'],
      ['What is your retrain trigger?', 'PSI above 0.25 on a monitored feature, sustained metric decay outside a pre-agreed band, or a calendar cadence as a backstop for the concept drift you cannot yet see — written down before deployment, so it is policy rather than a judgement call made under pressure.'],
      ['Why survival rather than a 12-month binary?', 'It uses partially-observed accounts properly instead of mis-scoring them as negatives, models when rather than merely whether, and the worked example shows the naive rate can understate the true rate by several points purely from how censored accounts are counted — not a rounding error, a systematic bias.']
    ], 'Quoting PSI bands as if they were statistical tests. Call them conventions and you sound like someone who has read the maths.')}`,
    labs: {
      psi: function (host) {
        const st = Viz.controls(host, [
          { k: 'shift', label: 'mean shift of the actual distribution', min: -1.5, max: 1.5, step: .02, value: .3, fmt: v => v.toFixed(2) },
          { k: 'spread', label: 'spread change', min: .5, max: 2.5, step: .02, value: 1, fmt: v => '×' + v.toFixed(2) },
          { k: 'bins', label: 'bins', min: 4, max: 20, step: 1, value: 10, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'psi', label: 'PSI', cls: 'key' }, { k: 'band', label: 'conventional band' },
          { k: 'ks', label: 'KS statistic' }, { k: 'worst', label: 'largest single-bin contribution' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const nb = st.bins, lo = -4, hi = 4, wdt = (hi - lo) / nb;
            const e = [], a = [];
            for (let i = 0; i < nb; i++) {
              const x0 = lo + i * wdt, x1 = x0 + wdt;
              e.push(Num.normCdf(x1, 0, 1) - Num.normCdf(x0, 0, 1));
              a.push(Num.normCdf(x1, st.shift, st.spread) - Num.normCdf(x0, st.shift, st.spread));
            }
            const eS = Num.sum(e), aS = Num.sum(a);
            const eN = e.map(v => Math.max(1e-6, v / eS)), aN = a.map(v => Math.max(1e-6, v / aS));
            const contrib = eN.map((v, i) => (aN[i] - v) * Math.log(aN[i] / v));
            const psi = Num.sum(contrib);
            const P = Viz.plot(ctx, w, h, { xd: [lo, hi], yd: [0, Math.max(Math.max.apply(null, eN), Math.max.apply(null, aN)) * 1.35] })
              .frame({ xlabel: 'feature value (binned)', ylabel: 'population share' });
            P.clip(() => {
              eN.forEach((v, i) => {
                const x0 = lo + i * wdt;
                const px0 = P.x(x0 + wdt * .08), px1 = P.x(x0 + wdt * .48);
                ctx.fillStyle = T.blue; ctx.globalAlpha = .75;
                ctx.fillRect(px0, P.y(v), px1 - px0, P.y(0) - P.y(v));
                const qx0 = P.x(x0 + wdt * .52), qx1 = P.x(x0 + wdt * .92);
                ctx.fillStyle = T.red; ctx.fillRect(qx0, P.y(aN[i]), qx1 - qx0, P.y(0) - P.y(aN[i]));
                ctx.globalAlpha = 1;
                if (contrib[i] > psi * .18) {
                  ctx.fillStyle = T.amber; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                  ctx.fillText(contrib[i].toFixed(3), P.x(x0 + wdt / 2), P.y(Math.max(v, aN[i])) - 4);
                }
              });
            });
            const ks = Math.max.apply(null, eN.map((_, i) => {
              const ce = Num.sum(eN.slice(0, i + 1)), ca = Num.sum(aN.slice(0, i + 1));
              return Math.abs(ce - ca);
            }));
            out({
              psi: psi.toFixed(4),
              band: psi < .1 ? 'stable (<0.1)' : psi < .25 ? 'moderate (0.1–0.25)' : 'significant (>0.25)',
              ks: ks.toFixed(3), worst: Math.max.apply(null, contrib).toFixed(4)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'expected (training)' }, { c: Viz.theme().red, t: 'actual (production)' }, { c: Viz.theme().amber, t: 'per-bin PSI contribution' }]);
        Viz.note(host, 'Amber labels mark the bins carrying the PSI. A total above 0.25 with one bin contributing most of it is a different problem from a uniform spread shift — the first is usually a new segment or a pipeline change, the second a genuine population move.');
      },

      km: function (host) {
        const st = Viz.controls(host, [
          { k: 'hazard', label: 'hazard ratio of cohort B', min: .5, max: 3, step: .05, value: 1.6, fmt: v => v.toFixed(2) + '×' },
          { k: 'censor', label: 'censoring rate', min: 0, max: .8, step: .02, value: .45, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'n', label: 'accounts per cohort', min: 50, max: 800, step: 25, value: 300, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'naive', label: 'naive 12-month default rate (censored = good)', cls: 'bad' },
          { k: 'km', label: 'Kaplan–Meier estimate at 12m', cls: 'key' },
          { k: 'bias', label: 'optimism introduced' }, { k: 'hr', label: 'estimated hazard ratio' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(23);
            const cohorts = [0, 1].map(c => {
              const base = .055 * (c ? st.hazard : 1);
              const rows = [];
              for (let i = 0; i < st.n; i++) {
                const t = R.exp(base);
                const cen = R() < st.censor ? R() * 12 : 99;
                rows.push({ t: Math.min(t, cen), event: t <= cen && t <= 12 ? 1 : 0, censored: cen < t });
              }
              return rows;
            });
            const P = Viz.plot(ctx, w, h, { xd: [0, 12], yd: [.6, 1] })
              .frame({ xlabel: 'months since origination', ylabel: 'survival S(t) = P(no default)' });
            const curves = cohorts.map((rows, ci) => {
              const sorted = rows.slice().sort((a, b) => a.t - b.t);
              let atRisk = rows.length, S1 = 1;
              const pts = [[0, 1]];
              sorted.forEach(r => {
                if (r.t > 12) return;
                if (r.event) { S1 *= (1 - 1 / atRisk); pts.push([r.t, S1]); }
                atRisk--;
              });
              pts.push([12, S1]);
              P.clip(() => P.line(pts, { color: ci ? T.red : T.blue, width: 2.4 }));
              return { pts: pts, final: S1, rows: rows };
            });
            // censoring ticks
            P.clip(() => cohorts.forEach((rows, ci) => {
              rows.filter(r => r.censored && r.t <= 12).slice(0, 60).forEach(r => {
                const surv = curves[ci].pts.reduce((acc, p) => p[0] <= r.t ? p[1] : acc, 1);
                P.line([[r.t, surv - .006], [r.t, surv + .006]], { color: ci ? T.red : T.blue, width: 1, alpha: .55 });
              });
            }));
            const naive = Num.mean(cohorts[1].map(r => r.event));
            const kmRate = 1 - curves[1].final;
            out({
              naive: (naive * 100).toFixed(2) + '%', km: (kmRate * 100).toFixed(2) + '%',
              bias: '−' + ((kmRate - naive) * 100).toFixed(2) + ' pts',
              hr: (Math.log(curves[1].final) / Math.log(curves[0].final)).toFixed(2) + '×'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'cohort A' }, { c: Viz.theme().red, t: 'cohort B' }, { c: Viz.theme().faint, t: 'tick marks = censored accounts' }]);
        Viz.note(host, 'Raise censoring and the naive rate drifts further below the Kaplan–Meier estimate: accounts that simply have not been observed long enough are being counted as successes. In a 12-month PD model with heavy attrition, that is a systematic optimism you cannot fix by adding features.');
      }
    },
    quiz: [
      {
        q: 'PSI on a key feature jumps to 0.34 while your AUC is unchanged. What has happened?',
        options: ['Concept drift', 'Data drift — the inputs moved; the relationship may still hold', 'Label leakage', 'Nothing — PSI is unreliable'],
        answer: 1,
        why: 'PSI is built purely from $P(x)$ — it compares today\'s input distribution to training\'s, bin by bin, and never looks at labels at all. A 0.34 reading past the conventional 0.25 threshold says the population has genuinely shifted; it says nothing about whether the model\'s relationship to the outcome has, which is exactly why AUC — a statement about ranking under whatever relationship currently holds — can sit unchanged right alongside it. Concept drift is the option that would explain a moved AUC, not a moved PSI, and is invisible until labels arrive regardless.'
      },
      {
        q: 'Scoring censored accounts as non-defaults in a 12-month PD model…',
        options: ['is standard and unbiased', 'biases the estimated default rate downward — they are censored, not negative', 'only matters for small samples', 'is fixed by stratified sampling'],
        answer: 1,
        why: 'A censored account\'s fate is genuinely unresolved — it has not had the chance to default in the months it has not yet lived through — and counting it as a confirmed good silently converts "unknown" into "no," which can only ever push the estimated default rate down, never up. The worked ten-account example shows this is not a small-sample curiosity: a naive 30.0% versus a Kaplan–Meier 36.5% on the identical data. Stratified sampling addresses class-ratio balance across folds (§2.14), an entirely different problem, and does nothing to correct for censoring.'
      },
      {
        q: 'A challenger model is deployed on a randomised traffic slice. Why randomised rather than a backtest?',
        options: ['It is faster', 'You get a causal read on the decision rather than an estimate confounded by the incumbent’s selection', 'It requires less data', 'Regulators mandate it universally'],
        answer: 1,
        why: 'A backtest, or any comparison restricted to the population the incumbent already chose to approve, only ever sees outcomes for a systematically safer, pre-screened slice — the reject-inference problem, where the declined applicants\' true outcomes are simply never observed. Randomising the traffic slice severs the link between "which model is deciding" and "who gets observed," giving a genuinely causal comparison in the sense §1.7 built, rather than one confounded by whichever population each model happened to see.'
      }
    ],
    cards: [
      { q: 'PSI formula and bands', a: '$\\sum_i (a_i-e_i)\\ln(a_i/e_i)$; <0.1 stable, 0.1–0.25 moderate, >0.25 shift — conventions, not tests. Same Jeffreys form as IV (§2.11).' },
      { q: 'Data vs concept drift', a: '$P(x)$ vs $P(y|x)$. PSI sees the first immediately, from features alone; the second needs labels, which can take months.' },
      { q: 'Training–serving skew', a: 'Training and serving computing "the same" feature via different code paths. A one-day window boundary bug alone produced a 9.4% gap in the worked example.' },
      { q: 'Rollout order', a: 'Shadow → canary → champion–challenger on randomised traffic, with a pre-agreed evaluation window fixed before outcomes start arriving.' },
      { q: 'Naive rate vs Kaplan–Meier', a: 'Scoring censored accounts as good understates the true rate — 30.0% naive vs 36.5% KM on the same ten accounts in the worked example.' },
      { q: 'Cox model', a: '$h(t|x)=h_0(t)e^{\\beta^\\mathsf{T}x}$; $e^{\\beta_j}$ is a hazard ratio (e.g. $e^{0.336}=1.40$, a 40% higher hazard). Check proportional hazards on Schoenfeld residuals.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.19 */
  ML.section({
    id: 'fairness', track: 'classical', num: '2.19',
    title: 'Fairness, bias, and the regulatory frame',
    lede: 'Unavoidable in lending, frequently asked, and the section most candidates have never thought about properly. Rests on §2.13’s confusion matrix and §2.12’s calibration; genuinely contested ground, flagged as such throughout.',
    html: `
<p>The obvious first move, the moment someone raises the question of fairness, is to delete the protected attribute — race, sex, age, religion — from the training data and declare the problem solved. It is worth actually trying this in your head before reading on, because the reason it fails is the entire subject of this section, and it fails for a reason that is not obvious until you have watched it happen. A postcode is not race. It is also, in almost every country with a history of residential geography shaped by discrimination, close enough to race that a model can reconstruct most of what it needed from ethnicity using postcode alone. Drop sex, and shopping categories, job titles, or which hour of the day someone applies carry a shadow of it. The model never sees the word "race" or "sex" anywhere in its input, and it can still produce a decision that varies systematically by both — because the signal was never really deleted, it was merely renamed and scattered across a dozen ordinary-looking columns.</p>

<h2><span class="sn">2.19.1</span> The distinction the law actually makes</h2>

<p><b>Disparate treatment</b> is using a protected attribute directly in the decision. It is illegal in essentially every jurisdiction that regulates lending, and — because it is a bright line, "did the decision function read this column" — it is comparatively easy to avoid and to prove. <b>Disparate impact</b> is different in kind: a facially neutral rule, one that never looks at a protected attribute anywhere, that nonetheless produces materially different outcomes across groups. This is where every real argument in fairness actually happens, and the reason is exactly the failed thought experiment above: <mark>"fairness through unawareness" does not work</mark>, so removing the attribute does not remove disparate impact, it only removes your ability to <i>measure</i> it. A model built with no protected attribute anywhere in its pipeline can be more discriminatory than one that includes it explicitly for monitoring, precisely because the version that includes it is the only one where anyone can compute the gap and decide whether to act on it.</p>

<p>The disparate-impact doctrine itself has a specific origin worth knowing: US employment law recognised, in the early 1970s, that a facially neutral qualification requirement could function as a proxy for race even with no discriminatory intent behind it, and the same logic has since migrated into lending, insurance and now algorithmic decision-making generally. The doctrine's entire premise is that intent is not the test — outcome is, at least as a trigger for scrutiny, which is exactly why "the model never sees the protected attribute" has never been an adequate defence on its own.</p>

<h2><span class="sn">2.19.2</span> Five ways to define "fair," each measuring something §2.13 already named</h2>

<p>Every candidate fairness criterion below is built from quantities §2.13's confusion matrix already gave names to — approval rate, true positive rate, false positive rate, precision, and the reliability curve of §2.12 — applied separately within each protected group and then compared across groups. The differences between them are not technical minutiae; each one encodes a genuinely different idea of what "fair" means, and picking one is a value judgement dressed as a metric choice.</p>

${H.table(['Criterion', 'Equalises', 'The claim it makes'], [
      ['<b>Demographic parity</b>', 'approval rate', 'Outcomes should match regardless of merit. Strong, and often unlawful to engineer directly.'],
      ['<b>Equal opportunity</b>', 'TPR (recall)', 'Among people who would repay, all groups are approved at the same rate. Usually the most defensible.'],
      ['<b>Equalised odds</b>', 'TPR and FPR', 'Errors of both kinds fall equally on every group. Strictly stronger than equal opportunity.'],
      ['<b>Predictive parity</b>', 'precision', 'A flagged applicant means the same thing in every group.'],
      ['<b>Calibration within groups</b>', 'reliability curve', '"20% risk" means 20% in every group. The minimum bar for a probability used in pricing.']
    ])}

<p>Read demographic parity as the strongest and least defensible claim in the table: it says the approval rate must match across groups, full stop, regardless of whether the underlying true risk differs — which in a lending context means either ignoring real risk differences or deliberately engineering around them, and regulators in most jurisdictions treat engineering toward equal approval rates as its own form of disparate treatment. Equal opportunity is the narrower, usually more defensible claim: among people who genuinely would have repaid, every group is approved at the same rate — it says nothing about what happens to people who would not have repaid, which is exactly the asymmetry a lender can usually justify. Equalised odds adds the mirror condition on the other kind of error, and is strictly stronger — satisfying it implies satisfying equal opportunity, never the reverse. Predictive parity and calibration within groups both ask, in slightly different ways, whether the model's output means the same thing across groups: predictive parity for a hard decision, calibration for the probability behind it.</p>

<p>The question a room will actually ask is: can you not just satisfy all five at once? The honest answer is no, and not for a mundane engineering reason — it is a theorem.</p>

${H.deriv('why calibration and equalised odds cannot both hold when base rates differ', [
      ['$\\mathrm{PPV} = \\dfrac{TP}{TP+FP}$, and in a population of size $N$ with prevalence $p$: $TP=(1-\\mathrm{FNR})\\,p\\,N$, $FP=\\mathrm{FPR}\\,(1-p)\\,N$.', 'Ordinary confusion-matrix bookkeeping (§2.13): of the $pN$ true positives, a fraction $(1-\\mathrm{FNR})$ are caught; of the $(1-p)N$ true negatives, a fraction $\\mathrm{FPR}$ are wrongly flagged.'],
      ['$\\mathrm{PPV} = \\dfrac{(1-\\mathrm{FNR})\\,p}{(1-\\mathrm{FNR})\\,p + \\mathrm{FPR}\\,(1-p)}$', 'Substitute both counts from line 1 into the definition of PPV (precision) from line 1, and the population size $N$ cancels top and bottom.'],
      ['$\\mathrm{FPR} = \\dfrac{(1-\\mathrm{FNR})\\,p\\,(1-\\mathrm{PPV})}{\\mathrm{PPV}\\,(1-p)}$', 'Rearrange line 2 to isolate FPR — pure algebra: cross-multiply, collect every FPR term on one side, divide through. The result links FPR, FNR, PPV and the prevalence $p$ in one equation with no freedom left over.'],
      ['Hold PPV and FNR fixed at some common values across two groups A and B (predictive parity and equal opportunity\'s FNR side, both satisfied). Then $\\mathrm{FPR}_A = \\mathrm{FPR}_B$ forces $\\dfrac{p_A}{1-p_A} = \\dfrac{p_B}{1-p_B}$, i.e. $p_A = p_B$.', 'Plug the shared PPV and FNR into line 3 for both groups and set the two FPR expressions equal — every factor cancels except the prevalence-odds term, which can only match if the two groups\' true base rates are identical.']
    ], 'So if $p_A \\ne p_B$ — the two groups genuinely default at different true rates, for reasons the model did not invent — you cannot hold PPV equal (predictive parity, the discrete cousin of calibration) and FNR equal (the equal-opportunity half of equalised odds) without FPR necessarily coming out unequal too, which breaks equalised odds outright. This is the Kleinberg, Mullainathan & Raghavan (2016) and Chouldechova (2017) impossibility result, and the derivation shows exactly why: three quantities — PPV, FNR, FPR — are linked by prevalence through one equation, so fixing two of them across groups with different prevalence over-determines, and breaks, the third.')}

${H.worked('the impossibility, with numbers you can check against a confusion matrix', `
<p>Two groups, true default rates $p_A=10\\%$ and $p_B=20\\%$ — a real, pre-existing difference the model did not create. Demand equal predictive parity, $\\mathrm{PPV}=50\\%$ in both groups, and equal opportunity, $\\mathrm{FNR}=30\\%$ (so $\\mathrm{TPR}=70\\%$) in both groups.</p>
<p>Plugging both groups into the formula derived above:</p>
$$\\mathrm{FPR}_A = \\frac{(0.70)(0.10)(0.50)}{(0.50)(0.90)} = \\frac{0.035}{0.45} = \\mathbf{7.78\\%}, \\qquad \\mathrm{FPR}_B = \\frac{(0.70)(0.20)(0.50)}{(0.50)(0.80)} = \\frac{0.070}{0.40} = \\mathbf{17.50\\%}$$
<p>Check this directly against a confusion matrix built from a million applicants per group, and it matches exactly: group A yields $TP{=}70{,}000$, $FN{=}30{,}000$, $FP{=}70{,}000$ against $900{,}000$ true negatives, an FPR of $70{,}000/900{,}000=7.78\\%$; group B yields $TP{=}140{,}000$, $FN{=}60{,}000$, $FP{=}140{,}000$ against $800{,}000$ true negatives, an FPR of $140{,}000/800{,}000=17.50\\%$. Insisting on equal predictive parity and equal opportunity, with these real base rates, mechanically produces group B's good customers being wrongly declined more than twice as often as group A's — not from any flaw in the model, from arithmetic alone.</p>`)}

${H.intuition(`<p>Picture a tailor with one pattern trying to fit two people of different builds equally well on every single measurement at once — shoulder width, sleeve length, waist, all matched exactly between two bodies that are not the same shape. It cannot be done with one pattern; something has to give, and the tailor's actual choice is which measurement to prioritise and which to let differ. Fairness metrics are the same negotiation, played out in the language of confusion matrices instead of tailoring, and the impossibility result is the formal statement that no pattern — no single classifier — fits two populations with genuinely different base rates on every criterion simultaneously. The question was never "how do I satisfy them all," it was always "which one do I prioritise, and can I defend that choice out loud."</p>`)}

${H.flag('This is where the field genuinely disagrees, not merely on implementation but on the underlying value judgement, and it is worth stating both sides honestly rather than picking a winner. One camp argues equalised odds should take priority: calibration can be satisfied by a model that has simply learned and reproduced a historically biased base rate, so insisting on calibration risks laundering that history as "accuracy." The other camp argues calibration is close to non-negotiable the moment the number feeds pricing or an expected-loss calculation (§2.12): an uncalibrated "20% risk" that means 35% for one group is not a fairness improvement, it is a different kind of harm, arguably a more concrete and more immediately measurable one. Both positions are held by serious, technically fluent people, and the honest professional answer is not to assert one is simply correct — it is to name the trade-off explicitly, state which criterion the deployment context makes non-negotiable (calibration is close to mandatory wherever the number sets a price), and document that the choice was made deliberately rather than defaulted into.')}

<p><b>What you are looking at.</b> Five paired bars, blue for group A and red for group B, each showing one metric — approval rate, TPR, FPR, precision, and the observed default rate among applicants scored near the current threshold — computed from actual scored populations whose true base rates and decision thresholds you control.</p>

<p><b>What to do with it.</b> Press "equal thresholds" and watch the calibration bars (observed risk at the threshold) line up closely across groups, while the TPR and FPR bars sit visibly apart — a calibrated model, unequal odds. Press "try to equalise approval rates" instead and watch the reverse: the top bars converge while the calibration bars pull apart.</p>

<p><b>The thing genuinely worth noticing.</b> Press the third button, which equalises the two groups' true base rates rather than touching any threshold. Every gap in every row collapses toward zero at once, with no per-group tuning at all — the single free variable that actually resolves the impossibility is the one thing a classifier can never change: the real underlying difference in risk between the two populations. Move the base rates apart again and every gap reopens no matter how cleverly you set the two thresholds, which is the theorem, not an artefact of this particular simulation.</p>

${H.lab('fair', 'The impossibility, made concrete', 'Two groups with different base rates. Try to satisfy calibration within groups and equalised odds at the same time — the readout will tell you exactly how far you are from each, and you will not reach zero on both unless you equalise the base rates or make the classifier perfect.')}

<h3>Two ways the picture so far is still too simple</h3>

<p>Everything above compared exactly two groups, defined by one attribute. Real fairness monitoring checks several protected attributes — race, sex, age band — and the moment you ask about their <b>intersections</b> rather than each one separately, a second problem appears that has nothing to do with the impossibility theorem and everything to do with §1.6's statistics of small samples.</p>

${H.worked('why intersectional fairness checks run out of data fast', `
<p>A book of 100,000 applicants, split across 4 race categories, 2 sex categories and 3 age bands — 24 intersectional cells in total. Checked one attribute at a time, race alone averages roughly 25,000 applicants per group: using the same standard-error formula from §2.18.3, $\\mathrm{se}\\approx0.5/\\sqrt{25{,}000/4}\\approx0.0063$, tight enough to detect a real TPR or FPR gap of a few percentage points with confidence.</p>
<p>Now check the smallest race category, crossed with one sex and the smallest age band — a cell that might hold only 0.5% of the book, 500 applicants. The identical formula gives $\\mathrm{se}\\approx0.5/\\sqrt{500/4}\\approx0.0447$, more than seven times noisier. A genuine 3-point TPR gap in that cell is comfortably inside the noise and will not reliably show up on any dashboard, while an apparent 5-point gap that shows up by pure chance can look alarming and be nothing but sampling variation. Neither failure is hypothetical: both directions — missing a real disparity, and chasing a phantom one — are the predictable arithmetic consequence of checking a fine-grained intersection with a fixed total sample.</p>`)}

<p>Kearns, Neel, Roth and Wu named this <b>fairness gerrymandering</b> in 2018: a classifier can satisfy demographic parity or equalised odds on every single protected attribute checked marginally, one at a time, while still discriminating sharply against a specific intersection — young women in one particular region, say — that nobody's dashboard was ever built to isolate, precisely because that dashboard was built one attribute at a time. Checking intersections is the fix in principle and a genuine statistical struggle in practice, because the cells you most need to audit are, by construction, the smallest and the noisiest ones on the whole sheet.</p>

<p>The second simplification is the unit of analysis itself. Everything in this section has been <b>group fairness</b>: pick a partition of applicants by a protected attribute, and equalise some statistic across the parts. <b>Individual fairness</b>, developed by Dwork, Hardt, Pitassi, Reingold and Zemel in 2012, asks a differently-shaped question entirely: should two applicants who are genuinely similar on every attribute that legitimately matters receive similar decisions, regardless of which group either belongs to? It is a more intuitive notion of fairness to most people encountering the topic for the first time, and it is far harder to operationalise, because it requires a trusted similarity metric between applicants — a notion of "genuinely similar" that is not itself contaminated by the same historical bias the whole exercise is trying to correct for. Group fairness metrics remain the practical default precisely because they only require a protected-attribute label and a confusion matrix, both of which §2.19.2 has already given you the machinery to compute.</p>

${H.worked('the four-fifths rule, a genuinely separate question from the impossibility above', `
<p><b>Adverse-impact screen.</b> Group A approval rate 42%, group B 31%. The impact ratio is $31/42 = \\mathbf{0.738}$. Below the conventional 0.80 (four-fifths) threshold, so adverse impact is indicated and the rule now needs a business-necessity justification and a search for a less discriminatory alternative.</p>`)}

${H.flag('The four-fifths rule is a US enforcement convention originating in employment law, widely borrowed elsewhere including credit, and it deserves the same "convention, not law" caveat §2.18 attaches to PSI\'s bands — cite it as a screen regulators use to decide where to look harder, never as a statistical test with a derived significance level, and never as a legal standard that 0.80 compliance alone satisfies.')}

<h2><span class="sn">2.19.3</span> Mitigation, at three stages, with three different failure modes</h2>

<p><b>Pre-processing</b> intervenes before the model ever trains: reweight the training examples so each group's outcomes carry equal influence, or repair the features themselves so a protected attribute cannot be reconstructed from them as easily (reweighing, a disparate-impact remover). It is attractive because the model architecture and the serving pipeline stay completely untouched — but reweighting the training distribution is a close cousin of the resampling §2.12 already warned pushes probabilities away from the true base rate, so a fairness fix at this stage can directly fight the calibration a lending decision needs downstream, and the two teams responsible for each — fairness and model risk — need to be talking to each other, not shipping past one another.</p>

<p><b>In-processing</b> builds the fairness objective into training itself: add a penalty term for whichever disparity you are targeting directly into the loss function, or train an adversary alongside the main model whose job is to predict the protected attribute from the model's internal representation, and penalise the main model whenever the adversary succeeds (adversarial debiasing) — the idea being that a representation the adversary cannot decode the protected attribute from is one that cannot easily launder it into the final decision either. This is the most principled of the three stages, because the fairness constraint is optimised jointly with accuracy rather than bolted on before or after, and it is also the hardest to validate, because "did the adversary genuinely fail because the information is gone, or merely because it wasn't a strong enough adversary" is not a question with an easy empirical answer.</p>

<p><b>Post-processing</b> is mathematically the cleanest of the three: take an already-trained, already-validated model and adjust the decision threshold separately for each group to equalise whichever metric you have chosen. It is also legally the most dangerous, because a threshold that differs by group is, definitionally, a decision rule that treats the protected attribute as an input to the decision — which is disparate treatment's own definition, dressed up as a fairness fix. <mark>Never ship a per-group threshold without explicit legal sign-off</mark>, and expect that sign-off to be harder to get than the modelling work was.</p>

<p>Running alongside all three stages, and increasingly the actual compliance artefact rather than the model itself, is the <b>less discriminatory alternative</b> search: retrain with different feature sets, different constraints, different model families, and keep a documented record that you looked for a version of the model that achieves similar accuracy with a smaller disparity — even where that search does not change which model ships. A validator, and increasingly a regulator, wants evidence the search happened, not merely a final model that looks acceptable.</p>

<h2><span class="sn">2.19.4</span> The regulatory frame, mid-2026</h2>

<p>In the EU, the AI Act classes creditworthiness assessment of natural persons as a <b>high-risk</b> use under Annex III, which pulls in a specific bundle of obligations: a risk management system, data governance and bias examination, technical documentation, event logging, human oversight, and demonstrated accuracy, robustness and cybersecurity — plus registration in an EU database and a conformity assessment before deployment. The Act entered into force in August 2024; prohibitions on the small set of banned practices and AI-literacy duties applied from February 2025, and general-purpose-model obligations from August 2025.</p>

${H.flag('Flag when you cite any of this: the high-risk timetable specifically has been the subject of active amendment and simplification proposals since the Act entered into force, so the safest move in an interview or a real document is to state the direction of travel — obligations phasing in over 2025–2027 — and explicitly check the current date and current text before asserting a specific deadline as settled. A confidently wrong date is worse than an honestly hedged one.')}

<p>In the US the operative machinery is older and has not moved in the same way: <b>ECOA and Regulation B</b> require a specific, individualised adverse-action reason for any credit denial (§2.17's whole reason-code pipeline exists to satisfy this), and the <b>FCRA</b> governs how bureau data may be used and disclosed. In the UK, <b>Consumer Duty</b> adds a genuinely different kind of test on top of anti-discrimination law: an outcomes test, asking whether the product actually works for the customer it was sold to, which is a separate question from whether the underlying model discriminates between groups — a product can pass every fairness metric in §2.19.2 and still fail Consumer Duty if it is, say, structured in a way that predictably leads financially vulnerable customers into unaffordable debt.</p>

${H.probe([
      ['Just remove the protected attribute — problem solved?', 'No. Proxies — postcode for ethnicity, shopping category or application hour for sex — reconstruct most of what was removed, and dropping the attribute also destroys your own ability to measure the impact you have just made invisible. The usual practice is to retain it for testing and monitoring while excluding it from the decision function itself.'],
      ['Which fairness metric would you use?', 'Name one, justify it from the decision\'s actual costs, and say what you monitor alongside it — the impossibility result (§2.19.2) means you structurally cannot have every criterion at once when base rates genuinely differ, so "our model is fair" without naming which criterion is not an answer, it is the absence of one.'],
      ['What is the four-fifths rule?', 'A US enforcement convention, originating in employment law: an impact ratio below 0.80 indicates adverse impact and triggers a requirement to justify the rule or search for a less discriminatory alternative. It is a screen regulators use to decide where to look harder, not a statistical test and not a legal standard that compliance alone satisfies.']
    ], 'Proposing per-group thresholds as an obviously good fix. It is mathematically the cleanest and the one mitigation most likely to create legal exposure, because a threshold that differs by group treats the protected attribute as an input to the decision by definition.')}`,
    labs: {
      fair: function (host) {
        const st = Viz.controls(host, [
          { k: 'baseA', label: 'group A true default rate', min: .02, max: .4, step: .01, value: .10, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'baseB', label: 'group B true default rate', min: .02, max: .4, step: .01, value: .20, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'thrA', label: 'threshold for group A', min: .02, max: .6, step: .01, value: .15, fmt: v => v.toFixed(2) },
          { k: 'thrB', label: 'threshold for group B', min: .02, max: .6, step: .01, value: .15, fmt: v => v.toFixed(2) },
          { k: 'quality', label: 'model quality', min: .3, max: 3, step: .05, value: 1.3, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ratio', label: 'impact ratio (4/5 rule)', cls: 'key' },
          { k: 'tprGap', label: 'TPR gap (equal opportunity)' },
          { k: 'fprGap', label: 'FPR gap' },
          { k: 'calGap', label: 'calibration gap' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(89);
            function group(base, thr) {
              const n = 4000, scores = [], labels = [];
              for (let i = 0; i < n; i++) {
                const bad = R() < base;
                labels.push(bad ? 1 : 0);
                scores.push(Num.sigmoid(R.normal(bad ? st.quality : -st.quality, 1.3) + Math.log(base / (1 - base))));
              }
              const c = Num.confusion(scores, labels, thr);
              const approved = scores.filter(s => s < thr).length / n;
              // calibration: observed default rate among those predicted ~thr
              const nearIdx = scores.map((s, i) => Math.abs(s - thr) < .06 ? i : -1).filter(i => i >= 0);
              const obs = nearIdx.length ? Num.mean(nearIdx.map(i => labels[i])) : NaN;
              return { c: c, approved: approved, scores: scores, labels: labels, obsAtThr: obs, base: base };
            }
            const A = group(st.baseA, st.thrA), B = group(st.baseB, st.thrB);
            // bars
            const metrics = [
              ['approval rate', A.approved, B.approved],
              ['TPR (catch rate)', A.c.recall, B.c.recall],
              ['FPR', A.c.fp / (A.c.fp + A.c.tn || 1), B.c.fp / (B.c.fp + B.c.tn || 1)],
              ['precision', A.c.precision, B.c.precision],
              ['observed risk @ threshold', A.obsAtThr, B.obsAtThr]
            ];
            const x0 = 172, bw = w - x0 - 90;
            ctx.font = '11px ui-sans-serif'; ctx.textBaseline = 'middle';
            metrics.forEach((m, i) => {
              const y = 30 + i * 52;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText(m[0], x0 - 10, y + 14);
              [1, 2].forEach(k => {
                const v = isFinite(m[k]) ? m[k] : 0;
                ctx.fillStyle = k === 1 ? T.blue : T.red;
                ctx.fillRect(x0, y + (k - 1) * 15, bw * Math.max(0, Math.min(1, v)), 12);
                ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
                ctx.fillText(isFinite(m[k]) ? (m[k] * 100).toFixed(1) + '%' : '—', x0 + bw * Math.max(0, Math.min(1, v)) + 6, y + (k - 1) * 15 + 6);
                ctx.font = '11px ui-sans-serif';
              });
            });
            ctx.fillStyle = T.blue; ctx.fillRect(x0, h - 26, 11, 11);
            ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.fillText('group A', x0 + 16, h - 20);
            ctx.fillStyle = T.red; ctx.fillRect(x0 + 90, h - 26, 11, 11);
            ctx.fillStyle = T.muted; ctx.fillText('group B', x0 + 106, h - 20);
            const ratio = Math.min(A.approved, B.approved) / Math.max(A.approved, B.approved);
            out({
              ratio: ratio.toFixed(3) + (ratio < .8 ? ' ✗' : ' ✓'),
              tprGap: ((A.c.recall - B.c.recall) * 100).toFixed(1) + ' pts',
              fprGap: (((A.c.fp / (A.c.fp + A.c.tn || 1)) - (B.c.fp / (B.c.fp + B.c.tn || 1))) * 100).toFixed(1) + ' pts',
              calGap: isFinite(A.obsAtThr) && isFinite(B.obsAtThr) ? ((A.obsAtThr - B.obsAtThr) * 100).toFixed(1) + ' pts' : '—'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Equal thresholds (calibrated, unequal odds)', primary: true, on: () => { st.$set('thrA', .15); st.$set('thrB', .15); S.redraw(); } },
          { label: 'Try to equalise approval rates', on: () => { st.$set('thrA', .12); st.$set('thrB', .22); S.redraw(); } },
          { label: 'Equal base rates (the escape hatch)', on: () => { st.$set('baseA', .15); st.$set('baseB', .15); S.redraw(); } }
        ]);
        Viz.note(host, 'Press the third button: with equal base rates every criterion can be satisfied at once. Move them apart again and the gaps reappear no matter how you set the thresholds. That is the Kleinberg/Chouldechova impossibility, not a limitation of this simulation.');
      }
    },
    quiz: [
      {
        q: 'Group A approval rate 42%, group B 31%. The impact ratio is…',
        options: ['1.35 — no issue', '0.738 — below the four-fifths screen', '0.11 — below the screen', '0.80 exactly'],
        answer: 1,
        why: '31/42 = 0.738, below the conventional 0.80 threshold, so adverse impact is indicated and triggers a business-necessity justification and a search for a less discriminatory alternative. This is a screen for where to look harder, not a statistical test with a derived p-value — treating it as either "no issue" (1.35, the reciprocal, a common slip) or as a hard legal pass/fail at exactly 0.80 both overstate what the number is entitled to claim.'
      },
      {
        q: 'Base rates genuinely differ between groups. You can simultaneously achieve…',
        options: ['calibration within groups and equalised odds', 'neither', 'calibration within groups OR equalised odds, but not both', 'both, with enough data'],
        answer: 2,
        why: 'The algebra linking PPV, FNR, FPR and prevalence (derived in §2.19.2) shows fixing predictive parity and the FNR side of equalised odds across two groups forces their FPR apart whenever the base rates differ — not from insufficient data, from arithmetic. "With enough data" is the tempting wrong answer, because it treats this as an estimation problem rather than the structural impossibility it actually is: more data narrows your confidence interval around the same, unavoidably unequal, true FPR gap.'
      },
      {
        q: 'Which mitigation carries the greatest legal risk?',
        options: ['Reweighing the training data', 'Adversarial debiasing', 'Per-group decision thresholds', 'Dropping a proxy feature'],
        answer: 2,
        why: 'A threshold that differs by group is, by definition, a decision rule that uses the protected attribute as an input — which is disparate treatment\'s own definition dressed up as a fix, however good its intentions. Reweighing and adversarial debiasing both act before or during training and never make the protected attribute part of the live decision rule; dropping a proxy feature is generally safe but rarely sufficient on its own (§2.19.1).'
      }
    ],
    cards: [
      { q: 'Disparate treatment vs impact', a: 'Using a protected attribute in the decision vs a neutral rule producing materially unequal outcomes. Impact is where the real arguments happen.' },
      { q: 'The impossibility result', a: 'PPV, FNR and FPR are linked through prevalence by one equation (§2.19.2); unequal base rates make calibration/predictive-parity and equalised odds mutually exclusive unless the classifier is perfect (Kleinberg 2016; Chouldechova 2017).' },
      { q: 'Four-fifths rule', a: 'Impact ratio < 0.80 indicates adverse impact — a US enforcement convention used as a screen, not a legal standard or statistical test.' },
      { q: 'Why unawareness fails', a: 'Proxies (postcode, shopping category, application hour) reconstruct the attribute, and removing it destroys your own ability to measure and correct impact.' },
      { q: 'The genuinely contested question', a: 'Whether calibration or equalised odds should take priority when they conflict — serious people disagree; state the trade-off and your chosen priority explicitly, never assert "fair" unqualified.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.20 */
  ML.section({
    id: 'part2-recall', track: 'classical', num: '2.26',
    title: 'Rapid recall — Part 2 in eighteen lines',
    lede: 'The sheet to read the night before. Every line should unpack into a derivation or a worked number you can reproduce out loud — if it doesn’t, go back to the section named on its right.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', 'Normal equations $w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge adds $\\lambda I$.', '<a href="#/linear-logistic">2.4</a>'],
      ['2', 'Logistic gradient $(\\hat p - y)x$.', '<a href="#/linear-logistic">2.4</a>'],
      ['3', 'Error = bias² + variance + σ²; deeper is not always better.', '<a href="#/bias-variance">2.2</a>'],
      ['4', 'L1 is sparse because of vertices on the axes and a constant subgradient.', '<a href="#/regularization">2.3</a>'],
      ['5', 'Bagging variance $\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$ — decorrelate to beat the floor.', '<a href="#/trees">2.7</a>'],
      ['6', 'XGBoost $w^*=-G/(H+\\lambda)$; gain $=\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{G^2}{H+\\lambda}]-\\gamma$.', '<a href="#/boosting">2.8</a>'],
      ['7', 'The worked split gave gain 0.667 and weights ∓0.667.', '<a href="#/boosting">2.8</a>'],
      ['8', 'EM = tighten the ELBO (E), then maximise it (M).', '<a href="#/unsupervised">2.9</a>'],
      ['9', 'PCA = top eigenvectors of the covariance.', '<a href="#/pca">2.10</a>'],
      ['10', 'Gini = 2·AUC − 1; PR-AUC over ROC under imbalance.', '<a href="#/metrics">2.13</a>'],
      ['11', 'Accuracy 98.6% / precision 40% / recall 80% on the worked matrix.', '<a href="#/metrics">2.13</a>'],
      ['12', 'PSI <0.1 / 0.1–0.25 / >0.25; IV <0.02 / 0.1–0.3 / >0.3 — conventions ⚑.', '<a href="#/production">2.18</a>'],
      ['13', 'Threshold from costs: $p^*=C_{FP}/(C_{FP}+C_{FN})$ — 6.25%, not 0.5.', '<a href="#/metrics">2.13</a>'],
      ['14', 'Conformal: rank residuals, take the $(n{+}1)(1-\\alpha)$ quantile — coverage without assumptions.', '<a href="#/calibration">2.12</a>']
    ])}
${H.table(['#', 'Also memorise', 'Section'], [
      ['15', 'SMOTE harms calibration (JAMIA 2022); recalibrate or do not resample.', '<a href="#/calibration">2.12</a>'],
      ['16', 'Time series: out-of-time and rolling origin, never shuffled k-fold.', '<a href="#/validation">2.14</a>'],
      ['17', 'Fairness: four-fifths screen; calibration and equalised odds are incompatible when base rates differ.', '<a href="#/fairness">2.19</a>'],
      ['18', 'SHAP efficiency: $\\sum\\phi_i = f(x)-\\mathbb{E}[f]$; TreeSHAP exact for trees.', '<a href="#/interpretability">2.17</a>']
    ])}

${H.lab('drill2', 'Part 2 drill', 'Eighteen prompts, shuffled. Answer aloud before flipping.')}`,
    labs: {
      drill2: function (host) {
        const cards = [
          ['Normal equations, and what ridge adds.', '$w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge adds $\\lambda I$ inside the inverse — regularises and guarantees invertibility.'],
          ['Logistic gradient.', '$(\\hat p-y)x$ — prediction minus label times feature.'],
          ['Bias–variance decomposition.', 'bias² + variance + irreducible σ².'],
          ['Why is L1 sparse?', 'Diamond vertices sit on the axes; the constant subgradient $\\lambda\\,\\mathrm{sign}(w)$ pins coefficients at zero.'],
          ['Bagging variance formula.', '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$.'],
          ['XGBoost optimal leaf weight.', '$w^*=-G/(H+\\lambda)$.'],
          ['XGBoost gain formula.', '$\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}]-\\gamma$.'],
          ['EM in five words.', '"Tighten the bound, then maximise."'],
          ['PCA, in one line.', 'Top eigenvectors of the covariance; $\\Sigma w=\\lambda w$ from the Lagrangian.'],
          ['Gini in terms of AUC.', 'Gini = 2·AUC − 1.'],
          ['Cost-optimal threshold.', '$p^*=C_{FP}/(C_{FP}+C_{FN})$; £60 vs £900 gives 6.25%.'],
          ['The worked confusion matrix numbers.', 'Accuracy 98.6%, precision 40%, recall 80%, F1 0.53 at 1% positives.'],
          ['PSI bands and the caveat.', '<0.1 / 0.1–0.25 / >0.25 — conventions, and the same Jeffreys form as IV.'],
          ['WOE and IV.', '$\\ln(\\%good/\\%bad)$ per bin; $\\mathrm{IV}=\\sum(\\%good-\\%bad)\\mathrm{WOE}$.'],
          ['Split conformal in three steps.', 'Calibration set → nonconformity scores → $\\lceil(n{+}1)(1-\\alpha)\\rceil$ quantile.'],
          ['SMOTE’s documented effect.', 'Harms calibration, rarely improves AUC (JAMIA 2022). Recalibrate or move the threshold.'],
          ['When is group k-fold mandatory?', 'Whenever one entity contributes multiple rows.'],
          ['The fairness impossibility.', 'Calibration within groups and equalised odds cannot both hold when base rates differ.'],
          ['SHAP efficiency property.', '$\\sum_i\\phi_i=f(x)-\\mathbb{E}[f]$.'],
          ['Monotonic constraints — the three benefits.', 'Domain agreement, a validator story, and free regularisation.']
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
        q: 'Which single Part 2 idea is used by the most other sections?',
        options: ['t-SNE', 'The bias–variance decomposition', 'DBSCAN', 'MAPE'],
        answer: 1,
        why: 'Bias² + variance + irreducible noise is the lens the rest of Part 2 keeps reaching for whenever it needs to justify a design choice: why added flexibility helps only up to a point (§2.2), why regularisation deliberately trades bias for variance (§2.3), why bagging and boosting attack variance and bias from opposite directions (§2.7–2.8), and why the validation protocol exists at all (§2.14) — it is shared vocabulary underneath all of them, not one topic sitting alongside them. "t-SNE" and "DBSCAN" are tempting mainly because they are memorable, visually distinctive techniques from earlier in Part 2, but each is a self-contained tool for one job — visualisation, density clustering — that the rest of Part 2 never needs to invoke to make its own arguments. "MAPE" is a single evaluation metric, useful in its own narrow context (§2.13) but not a framework other sections build on top of. The general principle is that some ideas in a course are terminal facts and some are load-bearing infrastructure other sections depend on, and bias-variance is the clearest example of the second kind, which is why §2.2 develops it before anything that needs it.'
      }
    ]
  });
})();
