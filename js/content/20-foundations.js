/* ============================================================
   PART 1 — Mathematical & statistical foundations (1.1 – 1.5)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 1.1 */
  ML.section({
    id: 'bayes', track: 'foundations', num: '1.1',
    title: 'Probability, conditioning, Bayes',
    lede: 'Every model that outputs a probability is doing one thing: taking a background rate and revising it in the light of evidence. This section builds that revision from ordinary counting, shows why a "99% accurate" test can be wrong 98% of the time, and ends with the one-line form you can compute in your head. It is the update rule underneath Naive Bayes, the probabilistic reading of logistic regression, calibration, and MAP — and getting the background rate wrong here makes every downstream number wrong by exactly the same factor.',
    rests: 'Establishes: §2.5 Naive Bayes · §2.4 logistic regression · §2.12 calibration · §1.5 MAP.',
    html: `
<p>You have shipped a fraud detector. It is a good one. Measured on held-out transactions it catches 95 per cent of the fraudulent ones, and it wrongly flags only 3 per cent of the legitimate ones. Those two numbers are on the model card, and both are true.</p>

<p>This morning it fired on a transaction. The operations team, who have a limited number of phone calls in them, ask you a perfectly reasonable question: <i>how likely is it that this particular transaction is actually fraud?</i></p>

<p>The tempting answer is 95 per cent, and it is not merely a little wrong. To see why, stop reasoning and start counting. Suppose fraud runs at 1 in 500 transactions, which is on the high side for card-present retail. Take 100,000 transactions and sort them into piles by hand.</p>

<ul>
<li>200 of them are fraudulent. The detector catches 95 per cent of those, so <b>190 fraudulent transactions get flagged</b> and 10 slip through.</li>
<li>99,800 of them are legitimate. The detector wrongly flags 3 per cent of those, so <b>2,994 legitimate transactions get flagged</b>.</li>
</ul>

<p>Now look only at the flagged pile, because that is the pile operations actually sees. It contains $190 + 2{,}994 = 3{,}184$ transactions, of which 190 are genuinely fraud. That is $190 / 3{,}184 = 5.97$ per cent. When the alarm sounds, the transaction is fraudulent about <b>one time in seventeen</b>, not nineteen times in twenty.</p>

<p>Nothing was wrong with the model. What went wrong was a swap of two questions that sound alike and are not alike at all. The model card reports "if the transaction is fraud, how often does the alarm sound?". Operations asked "if the alarm sounds, how often is the transaction fraud?". Both are ratios of the same two piles, but they divide by different denominators, and when the two underlying groups are wildly different in size, the two answers can differ by a factor of sixteen.</p>

<p>Ordinary arithmetic got you the answer above, but it got it by enumerating 100,000 imaginary transactions. That does not scale to three pieces of evidence, or to a hypothesis with five possible values, and it gives you no way to see <i>what</i> in the setup drove the answer. What is needed is a small piece of notation that does the counting for you and makes the moving parts visible. That notation is conditional probability, and the rule for reversing the two questions is Bayes' theorem.</p>

<h2><span class="sn">1.1.1</span> Conditioning is renormalising, not a new kind of number</h2>

<p>Write $P(A)$ for the probability of an event $A$, read aloud as "P of A". Write $P(A \\mid B)$ for the probability of $A$ <i>given</i> that $B$ has happened; the vertical bar is read as "given", and everything to its right is the evidence you are standing on. So $P(\\text{fraud} \\mid \\text{flagged})$ is read "the probability of fraud, given flagged", and it is the number operations wanted.</p>

<p>A conditional probability is not a new species of quantity. It is the same probability measure <b>renormalised to the world in which the evidence is true</b>. The recipe has exactly two steps: delete every outcome incompatible with $B$, then rescale what is left so that it adds to one again. Written down, that is</p>

$$P(A \\mid B) = \\frac{P(A \\cap B)}{P(B)}$$

<p>where $A \\cap B$, read "A and B", is the event that both happen. Say the formula in words: of all the ways $B$ could turn out, what fraction of them also have $A$ in them? The denominator $P(B)$ is doing the rescaling; it is the size of the shrunken world. Check it against the count above. Out of 100,000 transactions, the "flagged and fraudulent" pile holds 190, so $P(\\text{fraud} \\cap \\text{flagged}) = 190/100{,}000 = 0.0019$. The "flagged" pile holds 3,184, so $P(\\text{flagged}) = 0.03184$. Dividing gives $0.0019 / 0.03184 = 0.0597$ — the same 5.97 per cent, obtained by the same division, now with names on the parts.</p>

${H.analogy(`<p>Think of a probability distribution as one kilogram of fine sand spread unevenly over a table, with each region of the table standing for a different way the world could turn out. Total sand: one kilogram, always. Where the sand is piled deep, that outcome is likely.</p>
<p>Learning that $B$ is true does two things to the table. First it sweeps away, entirely, all the sand lying outside $B$ — those outcomes are now known not to have happened. Second, and this is the step people forget, it re-weighs what is left and calls <i>that</i> the new kilogram. The sand inside $B$ has not moved and its relative piles have not changed shape; only the scale on the weighing machine changed.</p>
<p>Two consequences drop out immediately. Conditioning can never rearrange the ordering of possibilities inside $B$, because it multiplies every surviving pile by the same constant. And conditioning on an event that had almost no sand on it to begin with will magnify small differences enormously, because you are dividing by a very small number. Both facts will matter within the next two pages.</p>`)}

<p>Two events are <b>independent</b> when learning one tells you nothing about the other, which written out means $P(A \\mid B) = P(A)$, or equivalently $P(A \\cap B) = P(A)P(B)$. Independence is a strong claim about the world and almost never true by accident; §1.2 and §2.5 both turn on how much damage assuming it wrongly does.</p>

<h2><span class="sn">1.1.2</span> Bayes' theorem: the formula that reverses the bar</h2>

<p>The whole difficulty in the fraud story was that you knew $P(\\text{flagged} \\mid \\text{fraud})$ and wanted $P(\\text{fraud} \\mid \\text{flagged})$. Getting from one to the other takes one line of algebra. The event "A and B" does not care which order you name it in, so $P(A \\cap B)$ can be written two ways using the definition above: as $P(A \\mid B)P(B)$, and as $P(B \\mid A)P(A)$. Set those equal and divide by $P(B)$:</p>

$$P(A\\mid B) = \\frac{P(B\\mid A)\\,P(A)}{P(B)}, \\qquad P(B) = \\sum_i P(B\\mid A_i)P(A_i)$$

<p>The left-hand side is called the <b>posterior</b> — what you believe after seeing the evidence. On the right, $P(B \\mid A)$ is the <b>likelihood</b>, which is how well the hypothesis $A$ explains the evidence you saw, and $P(A)$ is the <b>prior</b>, which is what you believed before any evidence arrived. The denominator $P(B)$ is the <b>evidence</b> or marginal likelihood, and the second formula says how to compute it: run through every mutually exclusive way $B$ could have come about, ask how likely $B$ is under each, weight by how likely each was in the first place, and add. That is the <b>law of total probability</b>. The symbol $\\sum$ is the Greek capital sigma and means "add up what follows", with $i$ running over the list of hypotheses.</p>

<p>Read the whole thing left to right in four words: <b>posterior is proportional to likelihood times prior</b>. The denominator is a normalising constant and carries no information about which hypothesis is right; its only job is to make the posterior probabilities add to one. That is worth internalising early, because in the models of Part 4 the denominator is frequently impossible to compute and the entire discipline of approximate inference exists to route around it.</p>

${H.worked('worked number — the low base rate', `
<p>The classic version of the same trap, in medicine. A disease has prevalence $P(D) = 0.001$, so one person in a thousand has it. The test has sensitivity $P(+\\mid D) = 0.99$, meaning it catches 99 per cent of true cases, and specificity $0.95$, meaning it correctly clears 95 per cent of healthy people — so its false-positive rate is $P(+\\mid \\neg D) = 0.05$, where $\\neg D$ is read "not D".</p>
$$P(D\\mid +) = \\frac{0.99 \\times 0.001}{0.99\\times0.001 + 0.05\\times0.999} = \\frac{0.00099}{0.05094} = 0.0194$$
<p>The numerator is "sick <i>and</i> positive". The denominator adds the second way of testing positive — "healthy <i>and</i> positive" — because those are the only two routes to a plus sign.</p>
<p>A positive result from a test people would happily describe as "99 per cent accurate" is correct about <mark>2 per cent of the time</mark>. Count it out again if the algebra feels slippery. In 100,000 people: 100 are sick and 99 of them are caught; 99,900 are healthy and 4,995 of them test positive anyway. The false positives outnumber the true ones by roughly fifty to one, and no amount of sensitivity fixes that. Even a <i>perfect</i> test that never misses a case would flag 100 true positives against the same 4,995 false ones, moving the answer from 1.94 per cent to 1.96 per cent. Only a better prior or a better specificity moves this number.</p>
<p>Push specificity from 95 per cent to 99.5 per cent and the false positives fall from 4,995 to 499, so the posterior becomes $99/598 = 16.5$ per cent — an eight-fold improvement from one change. Screen a targeted population with 10 per cent prevalence instead, and you get 9,900 true positives against 4,500 false ones, or 68.8 per cent. <b>Who you test matters more than how good the test is.</b></p>`)}

${H.intuition(`<p>The result above feels wrong the first several times you meet it, and the reason is a specific mental habit: people treat the prior as a piece of soft context that a hard measurement is entitled to overrule. It is not. The prior is the population you are drawing from, and evidence can only ever <i>multiply</i> it.</p>
<p>Here is the reframing that makes it stick. Evidence does not deliver you to a conclusion; it moves you a fixed distance from wherever you started. A test with a likelihood ratio of 20 always multiplies your odds by 20. If you start at odds of 1 in 999 you finish at roughly 1 in 50, and if you start at odds of 1 in 5 you finish at 4 to 1 on. Same test, same evidence, wildly different conclusions, because the evidence was never in the business of telling you where to end up.</p>
<p>This is also why "the model is 95 per cent accurate" is not a sentence you can act on. Accuracy is a statement about the likelihood side only. Until someone tells you the base rate, it is arithmetically impossible to say what a positive prediction is worth.</p>`)}

<p><b>What you are looking at.</b> The square grid is 10,000 people, one person per cell, laid out in a hundred rows of a hundred. The cells are coloured by which of the four buckets each person falls into, filled in a fixed order from the top-left corner so that the buckets appear as solid blocks rather than scattered dots: first the true positives in near-black, then the false positives in red, then the missed cases in amber, and finally the correctly-cleared majority in a very faint grey. The legend beside the grid gives the head-count in each bucket. Below the legend, a single horizontal bar re-draws only the flagged group — black for the part that is genuinely sick, red for the part that is not — and that bar's black fraction <i>is</i> the posterior. Underneath it the odds calculation is written out live. The three sliders are prevalence, sensitivity and specificity; the numeric readout adds the likelihood ratio and the raw counts per 10,000.</p>

<p><b>What to do with it.</b> Start by dragging prevalence, alone, from its default of 0.10 per cent up towards 35 per cent, and watch two things at once: the black block at the top-left grows steadily, and the red block that follows it shrinks. Stop when the horizontal bar below is about half black — you have just found the prevalence at which a positive result becomes an even-money bet. Then press the four preset buttons in turn. <i>Population screen</i> and <i>Targeted</i> change only the prior; <i>Better specificity</i> and <i>Perfect sensitivity</i> change only the test.</p>

<p><b>The thing genuinely worth noticing.</b> Press <i>Perfect sensitivity</i>, which sets the detector to a standard no real instrument achieves: it never misses a single case. The posterior barely twitches. Now press <i>Better specificity</i>, which merely halves the false-positive rate ten times over, from 5 per cent to 0.5 per cent, and the posterior leaps. The asymmetry is not a quirk of these numbers, it is structural: sensitivity acts on the small sick population and specificity acts on the enormous healthy one, so a given percentage improvement in specificity removes vastly more people from the flagged pile. When someone asks you to improve a screening system, this lab tells you which dial to reach for before you have written any code.</p>

${H.lab('bayes', 'The base rate, drawn as people', 'Each cell is one person out of 10,000, filled in bucket order from the top-left. Move prevalence and watch the red block swamp the black one — the posterior is the black fraction of the flagged bar below. Compare pushing specificity 95% → 99.5% against pushing sensitivity 99% → 100%.')}

<h2><span class="sn">1.1.3</span> Depth signal: work in odds</h2>

<p>The formula above has an awkward denominator that forces you to enumerate every hypothesis before you can produce a single number. There is a version with no denominator at all, and it is the version to use at a whiteboard, in your head, and in any code that only needs to rank hypotheses rather than report calibrated probabilities.</p>

<p>The trick is to stop tracking probabilities and start tracking <b>odds</b>. The odds of an event are the ratio of its probability to the probability of its complement: a probability of $0.2$ is odds of $0.2/0.8 = 1/4$, spoken as "one to four" or "four to one against". Odds and probabilities carry identical information — $p = o/(1+o)$ converts back — but odds behave far better under Bayes, because the awkward constant cancels.</p>

${H.deriv('posterior odds = likelihood ratio × prior odds', [
      ['$P(D\\mid+) = \\dfrac{P(+\\mid D)\\,P(D)}{P(+)}$', 'Bayes\' theorem applied to the hypothesis "diseased", with the evidence being a positive test. Nothing new; this is the formula from the previous subsection with names substituted in.'],
      ['$P(\\neg D\\mid+) = \\dfrac{P(+\\mid \\neg D)\\,P(\\neg D)}{P(+)}$', 'The very same rule applied to the complementary hypothesis, "not diseased". The denominator is the identical number $P(+)$ in both lines, because $P(+)$ is the overall chance of a positive result and never mentions which hypothesis is true.'],
      ['$\\dfrac{P(D\\mid+)}{P(\\neg D\\mid+)} = \\dfrac{P(+\\mid D)\\,P(D)\\,/\\,P(+)}{P(+\\mid \\neg D)\\,P(\\neg D)\\,/\\,P(+)}$', 'Divide line 1 by line 2. This is legitimate whenever $P(\\neg D \\mid +) > 0$, which holds unless the test is infallible.'],
      ['$= \\dfrac{P(+\\mid D)}{P(+\\mid \\neg D)} \\times \\dfrac{P(D)}{P(\\neg D)}$', 'The $P(+)$ appears once on the top and once on the bottom, so it cancels exactly. This is the whole point: the odds form never needs the law of total probability, and therefore never needs you to enumerate the hypotheses.'],
      ['$\\log \\text{posterior odds} = \\log \\mathrm{LR} + \\log \\text{prior odds}$', 'Take logarithms of both sides. The logarithm of a product is the sum of the logarithms, and $\\log$ is strictly increasing, so it preserves every ordering and loses no information.']
    ], 'The first factor on line 4 is called the <b>likelihood ratio</b>, written LR: how much more often this evidence appears under the hypothesis than against it. It is a pure property of the test. The second factor is the prior odds, a pure property of the population. Multiplying them is the entire content of Bayes\' theorem, and taking logs turns that multiplication into addition — which is precisely <i>why</i> a logistic regression is a sum of evidence terms and why a credit scorecard can add up points (§2.4).')}

$$\\underbrace{\\frac{P(D\\mid+)}{P(\\neg D\\mid+)}}_{\\text{posterior odds}} = \\underbrace{\\frac{P(+\\mid D)}{P(+\\mid \\neg D)}}_{\\text{likelihood ratio}} \\times \\underbrace{\\frac{P(D)}{P(\\neg D)}}_{\\text{prior odds}}$$

<p>Run the medical numbers through it. The likelihood ratio is $\\mathrm{LR} = 0.99/0.05 = 19.8$: a positive result is 19.8 times more common among the sick than among the healthy. The prior odds are $1{:}999$. Multiply, and the posterior odds are $19.8{:}999$, which is about $1{:}50$ — the same 2 per cent, computed without a calculator and without ever writing down $P(+)$.</p>

<p>The fraud detector goes the same way. Its likelihood ratio is $0.95/0.03 = 31.7$, and its prior odds are $1{:}499$. Posterior odds $31.7{:}499 = 1{:}15.7$, giving $1/16.7 = 6.0$ per cent, matching the count at the top of the section. The odds form also answers the design question directly: to make a flagged transaction more likely fraudulent than not, you need posterior odds of at least $1{:}1$, hence $\\mathrm{LR} \\ge 499$. Holding sensitivity at 95 per cent, that requires a false-positive rate of $0.95/499 = 0.0019$, so specificity would have to climb from 97 per cent to 99.81 per cent. In plain terms: you would have to cut false alarms by a factor of about sixteen. That is the real specification for the next model, and it fell out of one division.</p>

${H.note('Taking logs turns the multiplication into addition: log-posterior-odds = log-likelihood-ratio + log-prior-odds. That is <i>why</i> a logistic regression is a sum of evidence terms, and why its intercept is exactly the log prior odds.')}

<h2><span class="sn">1.1.4</span> Stacking evidence, and the assumption that makes it cheap</h2>

<p>One test rarely settles anything. The odds form tells you what a second, independent test does: it multiplies again. Two independent positive results on the medical test give posterior odds of $19.8^2 \\times (1/999) = 392/999 = 0.392$, which is a probability of $0.392/1.392 = 28.2$ per cent. A third positive takes it to $19.8^3/999 = 7.77$, or 88.6 per cent. Weak evidence compounds fast, which is the mathematical content of the phrase "a preponderance of evidence".</p>

<p>That calculation smuggled in a serious assumption, and it is worth naming it because an entire classifier is built on it. Multiplying likelihood ratios requires the two results to be <b>conditionally independent given the hypothesis</b>: among sick people, knowing the first test came back positive must tell you nothing about the second. Running the <i>same</i> test twice violates this badly, because whatever made the first result positive — an interfering substance, a mislabelled sample, an unusual physiology — is still present for the second. Two different tests probing different mechanisms come much closer to satisfying it.</p>

<p>Naive Bayes (§2.5) is exactly this multiplication applied to every feature at once, with conditional independence assumed across the board. The assumption is nearly always false, the resulting probabilities are nearly always badly calibrated, and the resulting <i>ranking</i> is often perfectly good — which is why the model refuses to die.</p>

${H.pitfall(`<p>The specific error is called the <b>prosecutor's fallacy</b>, and it is the fraud story wearing a wig. An expert testifies that the chance of a random person matching the forensic evidence is one in a million. Counsel then invites the jury to conclude that there is a one-in-a-million chance the defendant is innocent. Those are different quantities: the expert stated $P(\\text{match} \\mid \\text{innocent})$, and counsel reported it as $P(\\text{innocent} \\mid \\text{match})$.</p>
<p>Put the base rate in. If the search covered a city of ten million adults, then about ten innocent people are expected to match by chance alone. Add the one guilty person and the match, on its own and with no other evidence, points at roughly one suspect in eleven — around 9 per cent, not 99.9999 per cent. The evidence is genuinely powerful; it is a likelihood ratio of a million. It is simply being applied to prior odds of one in ten million.</p>
<p>The identical mistake in an interview sounds like this: "the model has 95 per cent precision on the positive class, so if it fires we are 95 per cent sure". Precision is already a posterior and that sentence is fine. But "the model has 95 per cent recall, so if it fires we are 95 per cent sure" is the fallacy verbatim, and it is the single most common probability error candidates make.</p>`)}

${H.practice(`<p>Here is where this bites in production rather than on paper. Fraud and default datasets are wildly imbalanced, so it is standard to rebalance the training set — keep every positive, discard most of the negatives — before fitting. Suppose the true positive rate is 1 per cent and you downsample negatives to reach a 50/50 training mix. You have just changed the prior odds from $1{:}99$ to $1{:}1$, and the model has dutifully learnt the wrong one.</p>
<p>Everything the model reports is now inflated, and by a precisely known amount. Because the log-odds are additive, the fix is a single constant subtracted from every logit the model emits: $\\log(1/99) - \\log(1/1) = -\\log 99 = -4.595$. Check it. A rebalanced model that outputs 0.5 has emitted a logit of 0; corrected, the logit is $-4.595$ and the probability is $1/(1 + e^{4.595}) = 1/100 = 0.01$, which is the true base rate, exactly as it should be for an example the model finds perfectly ambiguous.</p>
<p>Two consequences follow. Rebalancing leaves the <i>ranking</i> untouched, so AUC and any threshold-free metric are unaffected — which is why practitioners get away with it for so long. And it destroys calibration completely, so anything that consumes the probability as a number, such as an expected-loss decision rule or a risk-adjusted price, is wrong until you apply the correction. §2.12 is the full treatment.</p>`)}

${H.history(`<p>The rule is named for Thomas Bayes, a Presbyterian minister whose essay on the problem was found among his papers and published in 1763, two years after his death, by his friend Richard Price. Bayes had been chewing on an unglamorous-sounding question: given that you have watched an event happen a certain number of times out of a certain number of trials, what can you say about its underlying rate? That is the inverse of the problem everyone had been solving, which was to compute the chance of outcomes given a known rate — hence the old name for the subject, <i>inverse probability</i>.</p>
<p>Pierre-Simon Laplace arrived at the same idea independently in the 1770s and did far more with it, applying it to demography, astronomy and jurisprudence over the following forty years. It then fell substantially out of fashion. Through the first half of the twentieth century the dominant school treated the prior as unscientific — an invitation to smuggle opinion into a calculation — and built an alternative statistics around long-run frequencies instead. The argument was philosophical, occasionally bad-tempered, and is not settled to everyone's satisfaction even now.</p>
<p>What settled it in practice was machinery. Once computers could sample from posteriors that had no closed form, the objection that Bayesian calculations were intractable stopped biting, and the remaining objection — that priors are subjective — turned out to be much less troubling in machine learning than elsewhere, because in ML the prior is usually a regulariser chosen by cross-validation rather than a belief defended in a seminar (§1.5).</p>
<p>The failure mode this section is about has its own literature. A 1978 study in the <i>New England Journal of Medicine</i> by Casscells, Schoenberger and Graboys posed a version of the low-prevalence question to physicians and medical students at a teaching hospital: prevalence 1 in 1,000, false-positive rate 5 per cent. As reported, only about a fifth of the sixty respondents gave an answer near the correct 2 per cent, and the most popular answer was 95 per cent. Kahneman and Tversky were mapping the same phenomenon in the same decade and gave it the name it still has: <b>base-rate neglect</b>.</p>`)}

${H.more('why the denominator is the part that ruins everyone\'s week', `<p>In the examples above, $P(B)$ was a sum over two hypotheses and took a second to compute. That is a property of toy problems. In a real model the hypothesis is a parameter vector $\\theta$ living in a continuous space of possibly billions of dimensions, and the denominator becomes an integral:</p>
$$p(\\theta \\mid \\text{data}) = \\frac{p(\\text{data}\\mid\\theta)\\,p(\\theta)}{\\int p(\\text{data}\\mid\\theta')\\,p(\\theta')\\,d\\theta'}$$
<p>That integral is, in general, hopeless. It has no closed form, and numerical quadrature dies at about a dozen dimensions. Essentially every technique with "Bayesian" in its name is a strategy for avoiding it.</p>
<p>There are three main escapes and you will meet all of them. <b>Conjugacy</b> picks a prior whose algebraic form survives multiplication by the likelihood, so the posterior comes out in the same family and the integral is done once, by hand, forever — this is why Beta priors on rates keep appearing (§1.2, §1.6). <b>Sampling</b> methods such as MCMC construct a random walk whose long-run visiting frequency <i>is</i> the posterior, so you never evaluate the denominator, you only ever compare ratios in which it cancels — the same cancellation as the odds form above, exploited at industrial scale. <b>Variational inference</b> gives up on the exact posterior and instead finds the closest member of a tractable family, turning integration into optimisation; the KL divergence terms in §4.12 are this idea in its modern clothes.</p>
<p>Meanwhile MAP estimation (§1.5), which is what almost every regularised model in this course is secretly doing, dodges the problem by asking only for the <i>location of the peak</i> of the posterior. A peak does not care about the normalising constant, so the integral never appears. That is the whole reason MAP is cheap and full Bayes is not.</p>`)}

<h2><span class="sn">1.1.5</span> Where this shows up later</h2>
${H.table(['Downstream', 'The Bayes content'], [
      ['Naive Bayes (§2.5)', 'Posterior ∝ prior × ∏ likelihoods, with conditional independence assumed across features — the evidence-stacking of §1.1.4 applied to every column at once'],
      ['Logistic regression (§2.4)', 'The model outputs log-posterior-odds directly; the sigmoid inverts the odds transform, and the intercept is the log prior odds'],
      ['Calibration (§2.12)', 'A probability is only meaningful if the base rate it implies matches reality — resampling changes the prior and breaks this, correctable by the constant shift derived above'],
      ['MAP (§1.5)', 'The prior over parameters is the regulariser; L2 is a Gaussian prior, L1 a Laplace one'],
      ['Thompson sampling (§1.6)', 'Maintain a Beta posterior per arm, sample from it, act — exploration falls out of the posterior width for free'],
      ['Conformal prediction (§2.12)', 'The escape hatch when you refuse to model the likelihood at all and want coverage guarantees regardless']
    ])}

${H.probe([
      ['The test is 99% accurate — why is the posterior only 2%?', 'Because false positives drawn from the huge healthy majority swamp the handful of true positives. Per 100,000: 99 true positives against 4,995 false ones.'],
      ['What raises it?', 'A higher prior — targeted rather than population screening — or better specificity. Sensitivity barely moves it, because it acts only on the small sick population; even a perfect test takes 1.94% to 1.96%.'],
      ['Give the odds form.', 'Posterior odds = likelihood ratio × prior odds; here 19.8 × 1/999 ≈ 1/50. In logs it is addition, which is why logistic regression sums evidence.'],
      ['You downsampled negatives to balance the classes. What breaks?', 'Calibration, by a known constant: every logit is inflated by log(train prior odds / true prior odds). Ranking metrics such as AUC are untouched, which is why the bug survives review.'],
      ['Why can you multiply likelihood ratios from two tests?', 'Only under conditional independence given the hypothesis. Repeating the same test violates it; two tests probing different mechanisms approximately satisfy it. This is exactly the Naive Bayes assumption.']
    ], 'Quoting sensitivity as if it were $P(D\\mid+)$. It is the single most common probability error in interviews, and the tell is a candidate who answers the posterior question without ever asking for the base rate.')}`,
    labs: {
      bayes: function (host) {
        const st = Viz.controls(host, [
          { k: 'prev', label: 'prevalence P(D)', min: .0005, max: .35, step: .0005, value: .001, fmt: v => (v * 100).toFixed(2) + '%' },
          { k: 'sens', label: 'sensitivity P(+|D)', min: .5, max: 1, step: .005, value: .99, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'spec', label: 'specificity P(−|¬D)', min: .5, max: 1, step: .005, value: .95, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'post', label: 'P(D | +) posterior', cls: 'key' },
          { k: 'lr', label: 'likelihood ratio' },
          { k: 'tp', label: 'true positives / 10k' },
          { k: 'fp', label: 'false positives / 10k' },
          { k: 'ppv', label: 'flagged / 10k' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const N = 10000, side = 100;
            const sick = st.prev * N, tp = sick * st.sens, fn = sick - tp;
            const well = N - sick, fp = well * (1 - st.spec), tn = well - fp;
            const post = tp / (tp + fp || 1);
            const gw = Math.min(h - 34, w * .5), cs = gw / side;
            const ox = 8, oy = 14;
            const nTP = Math.round(tp), nFP = Math.round(fp), nFN = Math.round(fn);
            for (let i = 0; i < N; i++) {
              const r = Math.floor(i / side), c = i % side;
              let col;
              if (i < nTP) col = T.dark ? '#f5f7ff' : '#12141c';
              else if (i < nTP + nFP) col = T.red;
              else if (i < nTP + nFP + nFN) col = T.amber;
              else col = T.dark ? 'rgba(255,255,255,.07)' : 'rgba(0,0,0,.06)';
              ctx.fillStyle = col;
              ctx.fillRect(ox + c * cs, oy + r * cs, Math.max(.8, cs - .35), Math.max(.8, cs - .35));
            }
            ctx.strokeStyle = T.line; ctx.strokeRect(ox, oy, gw, gw);
            const lx = ox + gw + 22;
            const rows = [
              [T.dark ? '#f5f7ff' : '#12141c', Math.round(tp) + ' true positives (sick, flagged)'],
              [T.red, Math.round(fp) + ' false positives (healthy, flagged)'],
              [T.amber, Math.round(fn) + ' missed cases (sick, cleared)'],
              [T.dark ? 'rgba(255,255,255,.12)' : 'rgba(0,0,0,.1)', Math.round(tn) + ' correct negatives']
            ];
            ctx.font = '12px ui-sans-serif, system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            rows.forEach((r, i) => {
              ctx.fillStyle = r[0]; ctx.fillRect(lx, oy + 6 + i * 23, 12, 12);
              ctx.strokeStyle = T.line; ctx.strokeRect(lx, oy + 6 + i * 23, 12, 12);
              ctx.fillStyle = T.text; ctx.fillText(r[1], lx + 20, oy + 12 + i * 23);
            });
            const by = oy + 118, bw = Math.max(80, w - lx - 20);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'bottom';
            ctx.fillText('among the flagged:', lx, by - 6);
            ctx.fillStyle = T.dark ? '#f5f7ff' : '#12141c'; ctx.fillRect(lx, by, bw * post, 24);
            ctx.fillStyle = T.red; ctx.fillRect(lx + bw * post, by, bw * (1 - post), 24);
            ctx.fillStyle = T.text; ctx.textBaseline = 'top'; ctx.font = 'bold 13px ui-monospace, monospace';
            ctx.fillText((post * 100).toFixed(1) + '% actually sick', lx, by + 30);
            ctx.font = '11px ui-sans-serif'; ctx.fillStyle = T.muted;
            ctx.fillText('posterior odds = LR × prior odds', lx, by + 52);
            ctx.fillText('= ' + (st.sens / (1 - st.spec)).toFixed(1) + ' × ' + (st.prev / (1 - st.prev)).toFixed(4) +
              ' = ' + (st.sens / (1 - st.spec) * st.prev / (1 - st.prev)).toFixed(3) + ' : 1', lx, by + 70);
            out({
              post: (post * 100).toFixed(2) + '%',
              lr: (st.sens / (1 - st.spec || 1e-9)).toFixed(1),
              tp: Math.round(tp), fp: Math.round(fp), ppv: Math.round(tp + fp)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Population screen (0.1%)', on: () => { st.$set('prev', .001); S.redraw(); } },
          { label: 'Targeted (10%)', on: () => { st.$set('prev', .10); S.redraw(); } },
          { label: 'Better specificity (99.5%)', on: () => { st.$set('spec', .995); S.redraw(); } },
          { label: 'Perfect sensitivity', on: () => { st.$set('sens', 1); S.redraw(); } }
        ]);
      }
    },
    quiz: [
      {
        q: 'Prevalence 1%, sensitivity 90%, specificity 90%. Roughly what is $P(D\\mid+)$?',
        options: ['about 90%', 'about 50%', 'about 8%', 'about 1%'],
        answer: 2,
        why: 'Count 10,000 people. 100 are sick and 90 of those are caught. 9,900 are healthy and 10% of them — 990 people — are flagged anyway. So 90 of the 1,080 flagged people are genuinely sick, which is 8.3%. The tempting answer is 90%, because both the sensitivity and the specificity happen to be 90% and it feels as though the answer must be too. It is not, because those two percentages are applied to populations that differ by a factor of ninety-nine. Any time a base rate is small, the false positives are drawn from a much larger pool and simply outnumber the true ones. §1.1.2 works the same shape of calculation with a 0.1% base rate.'
      },
      {
        q: 'Which change raises the posterior most in a low-prevalence screen?',
        options: ['Raising sensitivity from 95% to 99%', 'Raising specificity from 95% to 99%', 'Doubling the sample size', 'Lowering the decision threshold'],
        answer: 1,
        why: 'False positives number (1 − specificity) × the huge healthy population, so cutting the false-positive rate from 5% to 1% removes four fifths of them. At a 0.1% base rate that takes the posterior from about 1.9% to about 8.7%, whereas the sensitivity change moves it by less than a tenth of a percentage point. Sensitivity is the tempting answer because it is the number most often quoted as "accuracy", and because catching more true cases sounds like it must be the bigger win. The general principle: an improvement is worth what it is worth multiplied by the size of the population it acts on. Sensitivity acts on the tiny sick group; specificity acts on everyone else. Doubling the sample size changes nothing at all, since every quantity here is a rate.'
      },
      {
        q: 'Posterior odds equal…',
        options: ['prior odds ÷ likelihood ratio', 'likelihood ratio × prior odds', 'likelihood ratio + prior odds', 'sensitivity × prevalence'],
        answer: 1,
        why: 'Write Bayes twice, once for the hypothesis and once for its complement, then divide one by the other. The evidence term $P(+)$ is the same number in both lines, so it cancels and you are left with a product of two ratios. The addition option is tempting because the rule really does become an addition — but only after taking logarithms, which is a different statement. That log form is exactly why a scorecard can add up points and why a logistic regression is a sum of evidence terms (§2.4). The last option confuses the joint probability of one cell with a ratio.'
      },
      {
        q: 'You trained on a set rebalanced to 50/50 from a true positive rate of 1%. The model reports 0.5 for a case. What is the calibrated probability?',
        options: ['0.5, rebalancing does not affect probabilities', 'about 0.01', 'about 0.25', 'not knowable without a held-out validation set'],
        answer: 1,
        why: 'Rebalancing replaced the true prior odds of 1:99 with 1:1, so every log-odds the model emits is inflated by log 99 ≈ 4.595. Subtracting that from a logit of 0 gives −4.595, and the sigmoid of that is 1/100 = 0.01. The first option is tempting because rebalancing genuinely leaves the ranking untouched, so AUC and every threshold-free metric look fine — which is exactly why this bug survives code review and is caught only when someone multiplies a probability by a pound value. §2.12 develops the correction and its alternatives.'
      }
    ],
    cards: [
      { q: 'Bayes, in the form to say at a whiteboard', a: 'Posterior odds = likelihood ratio × prior odds. In logs: log-posterior-odds = log LR + log prior odds.' },
      { q: 'A "99% accurate" test on a 0.1% base rate gives what posterior?', a: '≈ 2%. False positives from the healthy majority outnumber true positives ~50:1.' },
      { q: 'What is $P(B)$ in the denominator of Bayes?', a: 'The joint marginalised over every way B could occur: $\\sum_i P(B\\mid A_i)P(A_i)$ — the law of total probability.' },
      { q: 'Sensitivity or specificity — which moves a low-prevalence posterior?', a: 'Specificity, because it acts on the large healthy population. Perfect sensitivity barely moves it.' },
      { q: 'Effect of downsampling negatives on the model output', a: 'Ranking unchanged; every logit inflated by log(train odds / true odds), so calibration breaks by a known constant.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.2 */
  ML.section({
    id: 'distributions', track: 'foundations', num: '1.2',
    title: 'The distributions that recur',
    lede: 'Six distributions, and one idea that makes them worth learning: you never choose a loss function, you choose a story about how your target was generated, and the loss is then forced. Match the support and shape of a distribution to your target, and its negative log-likelihood is the thing you minimise.',
    rests: 'Establishes: the noise models behind every loss in §1.5 and the link functions in §2.4.',
    html: `
<p>A colleague hands you a table of support tickets and asks for tomorrow's volume. One row per day, one column holding a whole number: 14, 9, 22, 0, 31. It is a regression problem, so you reach for the standard move — fit a model, minimise mean squared error, ship it.</p>

<p>Three things go wrong, and none of them is a bug.</p>

<p>The first is that the model predicts $-3.4$ tickets for a quiet Sunday. Nothing in squared error knows that the target cannot go below zero, so the model happily wanders into territory that does not exist. The second is that squared error treats an error of 3 tickets as equally serious whether the day's true volume was 2 or 200 — but being out by 3 on a base of 2 is a catastrophe and being out by 3 on a base of 200 is noise. The third only shows up in the residual plot: the spread of the errors grows with the level, so the fitted intervals are far too wide on quiet days and far too narrow on busy ones.</p>

<p>You could patch each of these. Clip the predictions at zero, model the logarithm instead, fit a separate variance model. Every patch is a guess, none of them talks to the others, and you now have three arbitrary decisions where you had one. The situation is exactly the one that arose in §1.1 before Bayes' theorem: ordinary tools get an answer, but they get it by improvisation, and you cannot see what is driving the result.</p>

<p>The move that resolves all three at once is to stop asking "which loss?" and start asking a different question: <b>what kind of quantity is this, and what does randomness look like for it?</b> Ticket counts are non-negative whole numbers arising from many independent people each having a small chance of writing in. There is a distribution that describes exactly that situation, its variance grows with its mean automatically, and it cannot produce a negative number. Choose it, and the loss, the link function and the error structure all arrive together, already consistent with one another.</p>

${H.key('Pick the distribution whose support and shape match your target, and its negative log-likelihood is your loss.')}

<p>That sentence collapses §1.2, §1.5 and §2.4 into a single move, and it means you never memorise a loss function again. §1.5 proves the bridge in both directions; this section builds the vocabulary you need to walk across it.</p>

<h2><span class="sn">1.2.1</span> What a distribution is, and why support comes first</h2>

<p>§0.4 introduced a random variable as a numerical outcome of a random process, and a distribution as the object that says how the probability is spread over those outcomes. Three properties are worth pulling out and naming, because choosing a distribution is really choosing them.</p>

<p>The <b>support</b> is the set of values the variable can take: the outcomes that get any probability at all. This is the crudest property and by far the most useful, because it is the only one you can check without any modelling judgement. A count has support $\\{0, 1, 2, \\ldots\\}$. A binary label has support $\\{0,1\\}$. A duration has support $(0, \\infty)$. A probability has support $[0,1]$. If your chosen distribution can produce values your target cannot, the model will eventually produce them too, and it will be your problem.</p>

<p>The <b>shape</b> is how the mass is arranged inside that support: symmetric or skewed, one hump or several, thin-tailed or heavy-tailed. And the <b>parameters</b> are the small number of knobs — usually one or two — that slide the shape around. The parameters are what your model predicts; the shape and support are what you assume.</p>

<p>One distinction from §0.4 is worth restating in one line because it trips people up here specifically. For a discrete variable the object is a <b>probability mass function</b>, written $p(k)$, and its values are honest probabilities that add to one. For a continuous variable the object is a <b>probability density function</b>, written $f(x)$, and its values are <i>not</i> probabilities — only its integral over an interval is. A density is allowed to exceed 1: a uniform distribution on $[0, 0.5]$ has density exactly 2 everywhere on its support, because $2 \\times 0.5 = 1$. Nothing has gone wrong. This matters in §1.5, where the "likelihood" being maximised for a continuous target is a product of densities, and can therefore be larger than one without embarrassment.</p>

<h2><span class="sn">1.2.2</span> The six, one at a time</h2>

<h3>Bernoulli and Categorical — the distributions of a label</h3>

<p>A <b>Bernoulli</b> variable is one coin flip. Support $\\{0,1\\}$, one parameter $p$ which is the probability of the 1. Its mass function is usually written as a single expression rather than two cases:</p>

$$p(y) = p^{\\,y}(1-p)^{1-y}$$

<p>which is a small trick worth pausing on. Substitute $y = 1$ and the exponents become 1 and 0, leaving $p$. Substitute $y = 0$ and they become 0 and 1, leaving $1-p$. One formula, both cases, and — crucially — a formula you can take the logarithm of and differentiate, which is exactly what §1.5 does to produce cross-entropy.</p>

<p>Its mean is $p$ and its variance is $p(1-p)$. Notice what that variance does: it is zero at $p=0$ and at $p=1$, and largest at $p = 0.5$, where it equals $0.25$. A near-certain event is a near-deterministic one and carries almost no variance; maximum uncertainty sits at even odds. That figure $0.25$ is not decoration — it is the worst-case variance that §1.4 plugs into every test-set sample-size calculation.</p>

<p>A <b>Categorical</b> variable is the same idea with $K$ faces instead of two: one roll of a $K$-sided die, with probabilities $p_1, \\ldots, p_K$ that add to one. It is what a softmax layer outputs and what a next-token loss scores against.</p>

<h3>Binomial — counting the successes</h3>

<p>Add up $n$ independent Bernoulli variables, all with the same $p$, and you get a <b>Binomial</b>: the number of successes in $n$ trials. Support $\\{0, 1, \\ldots, n\\}$. Because expectation is linear and, under independence, variances add (§1.3), its mean and variance follow immediately without any new work: $\\mathbb{E}[X] = np$ and $\\mathrm{Var}(X) = np(1-p)$.</p>

<p>The Binomial is where the number of correct predictions on a test set lives. A model with true accuracy 0.9 evaluated on 1,000 rows produces a Binomial(1000, 0.9) count of correct answers, with mean 900 and variance $1000 \\times 0.9 \\times 0.1 = 90$, so a standard deviation of $\\sqrt{90} = 9.49$ correct answers, or 0.95 accuracy points. That is the entire reason §1.4 exists.</p>

<h3>Gaussian — the shape that many small nudges produce</h3>

<p>The <b>Gaussian</b> or normal distribution has support all of $\\mathbb{R}$, two parameters — the mean $\\mu$ (Greek mu) and the variance $\\sigma^2$ (sigma squared) — and the density</p>

$$f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\,e^{-(x-\\mu)^2 / 2\\sigma^2}$$

<p>Read it from the inside out. The bracket $(x-\\mu)$ is how far you are from the centre. Squaring it makes the shape symmetric and makes distant points count much more. Dividing by $2\\sigma^2$ converts that distance into units of the distribution's own width, so a point two standard deviations out is equally surprising whether $\\sigma$ is a millimetre or a mile. The minus sign and the exponential turn "surprising" into "unlikely", falling away faster than any polynomial. The fraction in front is a constant chosen purely so the whole thing integrates to one; it depends on $\\sigma$ but not on $x$, which is why it becomes an additive constant in §1.5 and quietly disappears.</p>

<p>The standard landmarks are worth having by heart: about 68 per cent of the mass lies within one standard deviation of the mean, about 95 per cent within two, and about 99.7 per cent within three. The 1.96 that shows up in every confidence interval is the exact two-sided 95 per cent point.</p>

<p>The Gaussian recurs for two independent reasons, and it is worth keeping them apart. The first is the central limit theorem (§1.4): anything that is a sum or average of many small independent contributions ends up Gaussian regardless of what the contributions looked like. The second is that among all distributions on $\\mathbb{R}$ with a given mean and variance, the Gaussian is the one with the most entropy — the one that assumes the least beyond those two numbers. So when you genuinely know nothing except a centre and a spread, it is the honest choice rather than a lazy one.</p>

<h3>Poisson — counting rare events in a fixed window</h3>

<p>The <b>Poisson</b> distribution has support $\\{0,1,2,\\ldots\\}$, one parameter $\\lambda$ (Greek lambda), and mass function</p>

$$p(k) = \\frac{e^{-\\lambda}\\lambda^k}{k!}$$

<p>where $k!$ is $k$ factorial, the product $1 \\times 2 \\times \\cdots \\times k$. Its defining feature is startling in its economy: <b>mean and variance are both $\\lambda$</b>. One number sets the level and the spread simultaneously, which is what makes it the natural model for counts and also what makes it fail in the specific way described later in this section.</p>

<p>Where does it come from? Take a Binomial with a huge number of trials and a tiny probability each — a million customers, each with a one-in-a-hundred-thousand chance of writing in today — and hold the product $np$ fixed at $\\lambda$. In the limit, the Binomial becomes exactly Poisson.</p>

${H.deriv('Poisson is the Binomial limit: $n\\to\\infty$, $p = \\lambda/n$', [
      ['$P(K=k) = \\binom{n}{k} p^k (1-p)^{n-k}$', 'The Binomial mass function: choose which $k$ of the $n$ trials succeed, then multiply the probability of those successes by the probability of the remaining failures.'],
      ['$= \\dfrac{n(n-1)\\cdots(n-k+1)}{k!}\\left(\\dfrac{\\lambda}{n}\\right)^{k}\\left(1-\\dfrac{\\lambda}{n}\\right)^{n-k}$', 'Substitute $p = \\lambda/n$, and expand the binomial coefficient as a falling product of $k$ terms over $k!$. Nothing has been approximated yet; this is still exact for every $n$.'],
      ['$= \\dfrac{\\lambda^k}{k!}\\cdot\\underbrace{\\dfrac{n(n-1)\\cdots(n-k+1)}{n^k}}_{A}\\cdot\\underbrace{\\left(1-\\dfrac{\\lambda}{n}\\right)^{n}}_{B}\\cdot\\underbrace{\\left(1-\\dfrac{\\lambda}{n}\\right)^{-k}}_{C}$', 'Pull the $\\lambda^k$ out of the middle factor, split the last power into an exponent of $n$ and an exponent of $-k$, and group. This is pure bookkeeping — every factor is still exactly what it was.'],
      ['$A \\to 1$', 'Factor $A$ is a product of exactly $k$ terms, $\\frac{n}{n}\\cdot\\frac{n-1}{n}\\cdots\\frac{n-k+1}{n}$, and $k$ is held fixed while $n$ grows. Each term tends to 1, and a fixed-length product of terms tending to 1 tends to 1.'],
      ['$B \\to e^{-\\lambda}$', 'This is the defining limit of the exponential function: $(1 + x/n)^n \\to e^{x}$ as $n \\to \\infty$, applied with $x = -\\lambda$.'],
      ['$C \\to 1$', 'The base tends to 1 and the exponent $-k$ is fixed, so the whole factor tends to $1^{-k} = 1$.'],
      ['$P(K=k) \\to \\dfrac{e^{-\\lambda}\\lambda^k}{k!}$', 'Multiply the four surviving pieces. The limit of a product of convergent factors is the product of their limits, so the Binomial mass function converges pointwise to the Poisson one.']
    ], 'This is why rare events counted over many opportunities are Poisson, and why the parameter is a <i>rate</i> rather than a probability. It also tells you when the approximation is good in practice: you need $n$ large and $p$ small, and the usual rule of thumb is $n \\ge 20$ with $p \\le 0.05$. Check it numerically with $n = 1{,}000$ and $p = 0.002$, so $\\lambda = 2$. The Binomial probability of exactly three events is 0.18063; the Poisson value is 0.18045. They agree to three decimal places, on a calculation that is far easier to do in the Poisson form.')}

${H.analogy(`<p>Stand a paving slab out in light, steady rain and count the drops that land on it in ten seconds. The rain does not coordinate. Each drop falls where it falls, unaware of the others, and the slab is small compared with the sky. Repeat the ten-second count many times and you will find the counts follow a Poisson distribution, with $\\lambda$ set by the rainfall rate and the size of the slab.</p>
<p>Two features of this picture are the assumptions in disguise. Drops must be independent, so this fails for hail that arrives in gusts. And the rate must be constant over the window, so it fails if the shower is easing off. When real count data is over-dispersed — more variable than Poisson allows — it is almost always because one of those two conditions is broken: events arrive in clumps, or the underlying rate itself varies from window to window.</p>`)}

<h3>Exponential and Gamma — how long until it happens</h3>

<p>Poisson counts events in a window. The <b>Exponential</b> distribution measures the gap between them. Support $(0,\\infty)$, one rate parameter $\\lambda$, density $f(x) = \\lambda e^{-\\lambda x}$, mean $1/\\lambda$ and variance $1/\\lambda^2$. If tickets arrive at a Poisson rate of six per hour, the waiting time to the next one is Exponential with $\\lambda = 6$ per hour, mean ten minutes.</p>

<p>Its defining property is <b>memorylessness</b>: $P(X > s + t \\mid X > s) = P(X > t)$. Having already waited twenty minutes tells you nothing about how much longer you will wait. This is a strong and often wrong assumption — machine parts wear out, customers who have not churned for three years are less likely to churn than new ones — and knowing that it is baked in is exactly why survival analysis (§2.18) reaches for the Weibull instead, which allows the hazard to rise or fall with age.</p>

<p>The <b>Gamma</b> distribution generalises it: the waiting time until the $k$-th event, rather than the first. Two parameters, a shape $k$ and a scale $\\theta$, mean $k\\theta$, variance $k\\theta^2$. Setting $k = 1$ recovers the Exponential exactly. Gamma is also the standard prior on a rate or a precision, and it is one half of the negative binomial construction below.</p>

<h3>Beta — a distribution over a probability</h3>

<p>The <b>Beta</b> distribution lives on $[0,1]$, which makes it the natural home for an unknown probability. Two shape parameters $\\alpha$ and $\\beta$ (alpha and beta), density proportional to $x^{\\alpha-1}(1-x)^{\\beta-1}$, and mean $\\alpha/(\\alpha+\\beta)$.</p>

<p>The reason it keeps appearing is <b>conjugacy</b>. Put a Beta$(\\alpha, \\beta)$ prior on a coin's bias, observe $s$ successes and $f$ failures, and the posterior is Beta$(\\alpha + s, \\beta + f)$ — the same family, with the counts simply added on. Look back at §1.1: the posterior is proportional to the likelihood times the prior, and here the likelihood $p^s(1-p)^f$ has exactly the same algebraic shape as the prior, so the product folds back into the family and the terrifying integral in the denominator is never needed.</p>

<p>This gives $\\alpha$ and $\\beta$ a beautifully concrete reading as <b>pseudo-counts</b>: a Beta$(2,2)$ prior is precisely the statement "pretend I have already seen one success and one failure". Observe 7 successes in 10 trials on top of that prior and the posterior is Beta$(9,5)$, with mean $9/14 = 0.643$ rather than the raw $0.7$. The prior has pulled the estimate towards a half, and it will pull less and less as real data accumulates. That is Laplace smoothing, that is what stops a Naive Bayes classifier being destroyed by a single unseen word (§2.5), and it is the mechanism behind Thompson sampling in §1.6.</p>

<h3>The summary table</h3>

${H.table(['Distribution', 'PMF / PDF', 'Mean', 'Variance', 'The ML use'], [
      ['<b>Bernoulli</b>', '$p^y(1-p)^{1-y}$', '$p$', '$p(1-p)$', 'binary labels → cross-entropy'],
      ['<b>Categorical</b>', '$\\prod_k p_k^{y_k}$', '$p_k$', '$p_k(1-p_k)$', 'softmax targets, next-token loss'],
      ['<b>Gaussian</b>', '$\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-(x-\\mu)^2/2\\sigma^2}$', '$\\mu$', '$\\sigma^2$', 'CLT limit → squared error'],
      ['<b>Poisson</b>', '$e^{-\\lambda}\\lambda^k/k!$', '$\\lambda$', '$\\lambda$', 'counts → Poisson deviance, log link'],
      ['<b>Exponential / Gamma</b>', '$\\lambda e^{-\\lambda x}$', '$1/\\lambda$', '$1/\\lambda^2$', 'waiting times, survival, time-to-default'],
      ['<b>Beta / Dirichlet</b>', '$x^{\\alpha-1}(1-x)^{\\beta-1}/B(\\alpha,\\beta)$', '$\\alpha/(\\alpha+\\beta)$', 'small when $\\alpha+\\beta$ large', 'conjugate priors on rates; smoothing']
    ])}

<p>Two relationships tie the table together and are worth having ready. <b>Binomial is a sum of Bernoullis</b>, so it inherits its mean and variance from linearity, and it tends to Gaussian by the CLT (§1.4). <b>Poisson is the Binomial limit</b> in the other direction, when $n \\to \\infty$ and $np \\to \\lambda$ — which is why rare-event counts are Poisson, and why Poisson regression rather than a linear model is the default for claim or default counts.</p>

<p><b>What you are looking at.</b> The blue curve or the blue stems are the exact distribution, computed from its formula: stems with dots for the discrete families, a filled curve for the continuous ones. The pale blue histogram behind them is not theory — it is an actual sample, drawn in the browser, so you are comparing a distribution with draws from it. The amber vertical line marks the true mean. On the three discrete distributions, and only on those, a dashed red curve is overlaid: it is the Gaussian with the same mean and variance, which is the approximation the central limit theorem promises. The readout gives the parameters, the theoretical mean and variance, and the sample mean and variance from the draws.</p>

<p><b>What to do with it.</b> The two parameter sliders are deliberately generic, because each distribution reads them differently, and the readout's "parameters" field tells you what they currently mean. Bernoulli takes $p$ as the first slider over ten; Binomial takes $n$ as three times the first slider and $p$ as the second over ten; Poisson and Exponential take $\\lambda$ straight from the first; Gaussian takes $\\mu$ as the first slider minus five and $\\sigma$ as half the second, so the defaults give a standard normal; Beta takes $\\alpha$ and $\\beta$ directly; Gamma takes shape from the first and scale from half the second. Start on Gaussian, then move the samples slider from 800 towards 4,000 and watch the sample mean in the readout creep towards the theoretical one. The random number generator is re-seeded identically on every redraw, so you are extending one fixed sequence rather than reshuffling — the histogram settles rather than dancing.</p>

<p><b>The thing genuinely worth noticing.</b> Switch to Binomial and leave the second slider near the middle, so $p$ is around 0.5. The dashed red Gaussian sits almost exactly on the stems even though $n$ is only fifteen. Now drag the second slider down towards 0.5, making $p \\approx 0.05$: the distribution collapses against the wall at zero, goes strongly skewed, and the red Gaussian visibly fails — it puts a substantial slice of its mass on negative counts, which are impossible. Concretely, at $n=15$ and $p=0.05$ the mean is 0.75 and the standard deviation is 0.84, so the matched Gaussian assigns about 19 per cent of its probability to values below zero. That single picture is the answer to "when is the normal approximation safe?" and it is why the folklore rule asks for $np$ and $n(1-p)$ both comfortably above ten. It is also the reason §1.4 prefers Hoeffding's inequality, which needs no approximation at all, over a CLT interval.</p>

${H.lab('dist', 'The distribution gallery, live', 'Change the parameters and watch the shape, the moments, and the sampled histogram. The dashed red curve on discrete plots is the Gaussian the CLT promises — notice how fast Binomial gets there at p ≈ 0.5 and how badly it fails once the distribution is pressed against zero.')}

<h2><span class="sn">1.2.3</span> Choosing by support and shape</h2>

<p>Here is the selection procedure in table form. Read the first column as a question about your target column, and the last column tells you what §1.5 will hand you once you take the negative logarithm of the corresponding likelihood.</p>

${H.table(['If your target is…', 'Support', 'Distribution', 'Resulting loss'], [
      ['yes / no', '{0,1}', 'Bernoulli', 'binary cross-entropy'],
      ['one of K classes', '{1..K}', 'Categorical', 'softmax cross-entropy'],
      ['a real number, symmetric error', 'ℝ', 'Gaussian', 'squared error'],
      ['a real number, occasional wild errors', 'ℝ', 'Laplace or Student-t', 'absolute error / robust loss'],
      ['a count', '{0,1,2,…}', 'Poisson (negative binomial if over-dispersed)', 'Poisson deviance'],
      ['a duration / time-to-event', '(0,∞)', 'Exponential / Weibull / Gamma', 'survival likelihood (§2.18)'],
      ['money with a mass at zero', '[0,∞)', 'Tweedie', 'Tweedie deviance'],
      ['a probability or rate', '[0,1]', 'Beta', 'beta likelihood / logit-normal']
    ])}

<p>Notice how much of the modelling decision the support column has already made for you before any judgement is required. Half of the errors in the ticket-forecasting story at the top of this section were caused by picking a row whose support did not match the data.</p>

${H.flag('Over-dispersion is the standard trap with counts, and it follows directly from the one-parameter economy that makes Poisson attractive: the model <i>forces</i> variance to equal the mean, so it has no way to represent data that is more variable than that. Real claim counts, ticket volumes and click counts almost always are, because the underlying rate itself varies from day to day and from customer to customer. The honest fix is the negative binomial — a Poisson whose rate is itself Gamma-distributed — which adds one dispersion parameter and lets variance exceed the mean. Suppose your tickets have sample mean 20 and sample variance 90. Poisson insists the variance is 20, so it will understate your prediction intervals by more than a factor of two. The negative binomial parameterises variance as $\\mu + \\mu^2/r$, so $90 = 20 + 400/r$ gives $r = 5.7$, and your intervals are honest again.')}

${H.practice(`<p>Four things go wrong with this material on real data, in roughly descending order of how often they cost someone a weekend.</p>
<p><b>The target is a ratio.</b> Conversion rate, click-through rate, loss ratio. It is tempting to model the ratio directly with a Gaussian, but a ratio computed from three trials and a ratio computed from three million trials are wildly different in reliability, and a Gaussian on the ratio treats them identically. Model the numerator with a Binomial or Poisson and carry the denominator as an exposure or offset term instead. This is the single most common structural mistake in applied count modelling.</p>
<p><b>Money has a spike at zero.</b> Insurance claim amounts, customer spend, refund values: most rows are exactly zero and the rest are continuous and right-skewed. No distribution in the standard list handles both at once, which is precisely what the Tweedie family exists for — it is a Poisson number of Gamma-distributed amounts, so it has an atom at zero and a continuous tail. Modelling it as a Gaussian gives you predictions that are neither zero nor plausible.</p>
<p><b>Heavy tails are not outliers.</b> When a residual plot shows a handful of enormous errors, the reflex is to delete those rows. Sometimes correct. Often they are simply what the distribution looks like, and deleting them means you have fitted a model to a fiction and will be surprised again next quarter. A Student-t likelihood keeps them and downweights them, which is the honest version of the same instinct (§1.5).</p>
<p><b>Check the assumption, do not assume it.</b> The Poisson equidispersion check takes one line: compare the sample variance of the residuals against their mean. If the ratio is meaningfully above 1, you have over-dispersion and your intervals are lying. Do this before you tune anything.</p>`)}

${H.history(`<p>Poisson published the distribution in 1837 inside a book about the probability of judicial verdicts, and it attracted almost no attention for sixty years. It became famous through one of the most charming datasets in statistics. In 1898 Ladislaus Bortkiewicz published <i>Das Gesetz der kleinen Zahlen</i> — the law of small numbers — in which he tabulated deaths by horse kick in the Prussian army: ten corps observed over twenty years, giving 200 corps-years and 122 deaths in total, an average of 0.61 per corps-year.</p>
<p>The observed counts were 109 corps-years with no deaths, 65 with one, 22 with two, 3 with three and 1 with four. The Poisson distribution with $\\lambda = 0.61$ predicts 108.7, 66.3, 20.2, 4.1 and 0.6. The agreement is almost uncomfortable. What made the example land was that horse kicks are about as far from a designed experiment as data gets — and yet the same formula that governs radioactive decay and telephone call arrivals reproduced it, because the underlying structure is identical: many independent opportunities, each with a tiny probability.</p>
<p>The Gaussian's history contains a lovely inversion of the argument this section makes. Gauss, in 1809, did not assume Gaussian errors and deduce least squares. He assumed the answer — that the arithmetic mean of repeated measurements ought to be the most probable value of the quantity — and asked which error law would make that true. The Gaussian is what came back. So for a century the association between squared error and the normal distribution ran in the opposite direction from the one §1.5 will present, and the modern telling only became the standard one after Fisher put likelihood on a general footing in the 1920s.</p>`)}

${H.more('the exponential family, which is why any of this is systematic', `<p>Every distribution in the table above except the heavy-tailed ones can be written in a single common form:</p>
$$p(y \\mid \\eta) = h(y)\\,\\exp\\!\\big(\\eta\\, T(y) - A(\\eta)\\big)$$
<p>Here $\\eta$ (eta) is called the natural parameter, $T(y)$ is a sufficient statistic — usually just $y$ itself — and $A(\\eta)$ is a normalising term called the log-partition function. The family is not a curiosity; it is the reason generalised linear models exist as one framework rather than a dozen recipes.</p>
<p>Three facts fall out of that form and each one explains something you would otherwise have to memorise. First, differentiating $A$ gives the mean: $A'(\\eta) = \\mathbb{E}[T(y)]$, and differentiating twice gives the variance. That is why the Poisson has variance equal to its mean and the Bernoulli has variance $p(1-p)$ — both are second derivatives of the same object, not separate facts.</p>
<p>Second, when you connect a linear predictor $z = w \\cdot x$ to the distribution through the <b>canonical link</b>, meaning you set $\\eta = z$ directly, the gradient of the negative log-likelihood with respect to $z$ collapses to $\\hat{y} - y$ for every member of the family. Prediction minus label. That is why logistic regression, linear regression and Poisson regression all have the same update rule despite modelling completely different things, and §1.5 derives the logistic case explicitly.</p>
<p>Third, every exponential family member has a conjugate prior, which is where the Beta–Bernoulli and Gamma–Poisson pairings come from. They are not lucky coincidences somebody noticed; they are a structural consequence of the form above.</p>`)}

${H.probe([
      ['Why is cross-entropy the loss for classification?', 'It is the negative log-likelihood of a Bernoulli or Categorical label — not a heuristic. §1.5 derives it in four lines.'],
      ['When would you use Poisson regression rather than linear regression?', 'Non-negative integer counts. The log link keeps predictions positive and the variance scales with the mean automatically, so quiet days and busy days both get sensible intervals.'],
      ['Your count data has variance well above its mean. What now?', 'Negative binomial, which is a Poisson with a Gamma-distributed rate and one extra dispersion parameter. Fitting Poisson anyway leaves the point predictions roughly right and the uncertainty badly understated.'],
      ['Why is Beta a natural prior on a rate?', 'Its support is exactly [0,1] and it is conjugate to Bernoulli and Binomial, so the posterior is Beta again with the counts simply added — which makes Thompson sampling one line of code (§1.6).'],
      ['A density value of 2.5 — is something wrong?', 'No. Densities are not probabilities; only their integrals are. A uniform on [0, 0.5] has density 2 everywhere. Likelihoods for continuous targets can legitimately exceed one.']
    ])}`,
    labs: {
      dist: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'distribution', type: 'select', value: 'gauss', options: [
            { v: 'bern', t: 'Bernoulli' }, { v: 'binom', t: 'Binomial' }, { v: 'gauss', t: 'Gaussian' },
            { v: 'pois', t: 'Poisson' }, { v: 'expo', t: 'Exponential' }, { v: 'beta', t: 'Beta' }, { v: 'gamma', t: 'Gamma' }] },
          { k: 'a', label: 'parameter 1', min: .05, max: 10, step: .05, value: 5, fmt: v => v.toFixed(2) },
          { k: 'b', label: 'parameter 2', min: .05, max: 10, step: .05, value: 2, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'samples drawn', min: 0, max: 4000, step: 50, value: 800, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'p', label: 'parameters', cls: 'key' }, { k: 'm', label: 'mean' }, { k: 'v', label: 'variance' },
          { k: 'sm', label: 'sample mean' }, { k: 'sv', label: 'sample var' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(31);
            const d = st.d;
            let pts = [], samples = [], xd, discrete = true, mean = 0, varr = 0, plabel = '';
            if (d === 'bern') {
              const p = Math.min(.99, st.a / 10);
              xd = [-0.6, 1.6]; pts = [[0, 1 - p], [1, p]]; mean = p; varr = p * (1 - p);
              plabel = 'p = ' + p.toFixed(2);
              for (let i = 0; i < st.n; i++) samples.push(R() < p ? 1 : 0);
            } else if (d === 'binom') {
              const n = Math.max(1, Math.round(st.a * 3)), p = Math.min(.99, st.b / 10);
              xd = [-.5, n + .5]; mean = n * p; varr = n * p * (1 - p);
              plabel = 'n = ' + n + ', p = ' + p.toFixed(2);
              for (let k = 0; k <= n; k++) pts.push([k, Num.binomPmf(k, n, p)]);
              for (let i = 0; i < st.n; i++) { let s = 0; for (let j = 0; j < n; j++) s += R() < p ? 1 : 0; samples.push(s); }
            } else if (d === 'pois') {
              const lam = st.a; xd = [-.5, Math.max(10, lam * 3)]; mean = lam; varr = lam;
              plabel = 'λ = ' + lam.toFixed(2);
              for (let k = 0; k <= xd[1]; k++) pts.push([k, Num.poisPmf(k, lam)]);
              for (let i = 0; i < st.n; i++) samples.push(R.poisson(lam));
            } else if (d === 'gauss') {
              discrete = false;
              const mu = st.a - 5, sdv = Math.max(.15, st.b / 2);
              xd = [mu - 4 * sdv, mu + 4 * sdv]; mean = mu; varr = sdv * sdv;
              plabel = 'μ = ' + mu.toFixed(2) + ', σ = ' + sdv.toFixed(2);
              for (let i = 0; i <= 220; i++) { const x = xd[0] + (xd[1] - xd[0]) * i / 220; pts.push([x, Num.normPdf(x, mu, sdv)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.normal(mu, sdv));
            } else if (d === 'expo') {
              discrete = false;
              const lam = Math.max(.1, st.a); xd = [0, 6 / lam]; mean = 1 / lam; varr = 1 / (lam * lam);
              plabel = 'λ = ' + lam.toFixed(2);
              for (let i = 0; i <= 220; i++) { const x = xd[0] + (xd[1] - xd[0]) * i / 220; pts.push([x, lam * Math.exp(-lam * x)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.exp(lam));
            } else if (d === 'beta') {
              discrete = false;
              const a = Math.max(.1, st.a), b = Math.max(.1, st.b);
              xd = [0, 1]; mean = a / (a + b); varr = a * b / ((a + b) * (a + b) * (a + b + 1));
              plabel = 'α = ' + a.toFixed(2) + ', β = ' + b.toFixed(2);
              for (let i = 1; i < 220; i++) { const x = i / 220; pts.push([x, Num.betaPdf(x, a, b)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.beta(a, b));
            } else {
              discrete = false;
              const k = Math.max(.15, st.a), th = Math.max(.1, st.b / 2);
              xd = [0, Math.max(4, (k * th) * 3)]; mean = k * th; varr = k * th * th;
              plabel = 'k = ' + k.toFixed(2) + ', θ = ' + th.toFixed(2);
              for (let i = 1; i <= 220; i++) { const x = xd[0] + (xd[1] - xd[0]) * i / 220; pts.push([x, Num.gammaPdf(x, k, th)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.gamma(k) * th);
            }
            const maxp = Math.max.apply(null, pts.map(p => p[1])) || 1;
            const P = Viz.plot(ctx, w, h, { xd: xd, yd: [0, maxp * 1.25] })
              .frame({ xlabel: discrete ? 'k' : 'x', ylabel: discrete ? 'probability' : 'density' });
            if (samples.length) {
              const nb = discrete ? Math.round(xd[1] - xd[0]) : 34;
              const hh = Num.hist(samples, Math.max(2, nb), xd[0], xd[1]);
              const scale = discrete ? 1 / samples.length : 1 / (samples.length * hh.w);
              P.clip(() => {
                hh.bins.forEach((c, i) => {
                  const x0 = hh.lo + i * hh.w, den = c * scale;
                  ctx.fillStyle = T.blue; ctx.globalAlpha = .18;
                  ctx.fillRect(P.x(x0), P.y(den), Math.max(1, P.x(x0 + hh.w) - P.x(x0) - 1), P.y(0) - P.y(den));
                  ctx.globalAlpha = 1;
                });
              });
            }
            if (discrete) {
              P.clip(() => pts.forEach(p => {
                P.line([[p[0], 0], [p[0], p[1]]], { color: T.blue, width: 4 });
                P.dots([[p[0], p[1]]], { r: 3.4, color: T.blue });
              }));
              P.clip(() => P.fn(x => Num.normPdf(x, mean, Math.sqrt(varr)), { color: T.red, width: 1.4, dash: [5, 4] }));
            } else {
              P.clip(() => { P.area(pts, { color: T.blue, alpha: .16 }); P.line(pts, { color: T.blue, width: 2.4 }); });
            }
            P.vline(mean, { color: T.amber, label: 'mean' });
            out({
              p: plabel, m: mean.toFixed(3), v: varr.toFixed(3),
              sm: samples.length ? Num.mean(samples).toFixed(3) : '—',
              sv: samples.length ? Num.variance(samples).toFixed(3) : '—'
            });
          }
        });
        Viz.note(host, 'The blue histogram is actual draws, generated here. Sample moments converge to the theoretical ones at the $1/\\sqrt n$ rate §1.4 quantifies.');
      }
    },
    quiz: [
      {
        q: 'Claim counts per policy-year have variance clearly larger than their mean. The right first model is…',
        options: ['linear regression on the raw count', 'Poisson regression', 'negative binomial regression', 'logistic regression'],
        answer: 2,
        why: 'Poisson has one parameter doing two jobs: it sets both the mean and the variance, so it cannot represent data that is more variable than its own level. Over-dispersion is exactly what the negative binomial exists for — it is a Poisson whose rate is itself Gamma-distributed, adding one dispersion parameter so that variance can exceed the mean. Poisson is the tempting answer because it is the correct <i>family</i> for counts and the point predictions it produces are often perfectly reasonable; what breaks is the uncertainty, and uncertainty is invisible until someone relies on it. Linear regression fails on support, since it can predict negative counts.'
      },
      {
        q: 'Why does the Beta distribution keep appearing as a prior on a probability?',
        options: ['It is maximum-entropy on ℝ', 'Its support is [0,1] and it is conjugate to Bernoulli/Binomial', 'It has infinite variance', 'It is the CLT limit'],
        answer: 1,
        why: 'Support matches the quantity — a probability cannot be 1.4 — and conjugacy means the posterior is Beta again with the observed counts simply added to the parameters. That turns Bayesian updating into two additions, which is why Thompson sampling fits on one line (§1.6). The maximum-entropy option is tempting because that phrase does justify the Gaussian, but it justifies the Gaussian on the real line given a mean and variance, which is a different claim about a different support. Beta has perfectly finite variance, and it is not a limit of anything.'
      },
      {
        q: 'At n = 15 and p = 0.05, why does the Gaussian approximation to the Binomial fail?',
        options: ['because n is not a power of two', 'because the distribution is pressed against zero and strongly skewed, so a symmetric curve puts mass on impossible values', 'because the Binomial has no variance in this regime', 'because the CLT requires p > 0.5'],
        answer: 1,
        why: 'The mean is 0.75 and the standard deviation is 0.84, so a matched Gaussian assigns roughly 19 per cent of its probability to negative counts, which cannot happen. The CLT is a statement about a limit, and how quickly you reach that limit depends on how skewed the summands are; a very lopsided coin needs far more flips than a fair one. This is the reason for the folklore rule that np and n(1−p) should both comfortably exceed ten, and the reason §1.4 prefers bounds that hold at every finite n over approximations that hold eventually.'
      }
    ],
    cards: [
      { q: 'The distribution-selection rule', a: 'Match support and shape to your target; the negative log-likelihood of that distribution is your loss.' },
      { q: 'Poisson as a limit', a: 'Binomial with $n\\to\\infty$, $np\\to\\lambda$ — rare events in many trials. Mean and variance are both $\\lambda$.' },
      { q: 'Mean and variance of a Binomial', a: '$np$ and $np(1-p)$. Bernoulli variance peaks at 0.25 when $p=0.5$.' },
      { q: 'Over-dispersion, and its fix', a: 'Count variance exceeds the mean, which Poisson forbids. Use negative binomial: variance $\\mu + \\mu^2/r$.' },
      { q: 'Why Beta and Bernoulli pair up', a: 'Conjugacy: posterior is Beta$(\\alpha+s, \\beta+f)$, so $\\alpha,\\beta$ read as pseudo-counts.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.3 */
  ML.section({
    id: 'expectation', track: 'foundations', num: '1.3',
    title: 'Expectation, variance, covariance',
    lede: 'One asymmetry, and it is the whole section: linearity of expectation always holds, while variances add only when the things being added are uncorrelated. That single cross term is the reason averaging models helps, the reason it stops helping, and the entire theory of bagging in one line of algebra.',
    rests: 'Feeds: §2.2 bias–variance · §2.10 PCA · every ensemble argument in §2.7.',
    html: `
<p>Fit a decision tree to your training data. Now draw a bootstrap resample of the same size — same number of rows, sampled with replacement, so some rows appear twice and about a third do not appear at all — and fit a second tree to that. Predict the same customer with both. The first tree says 0.31 and the second says 0.47.</p>

<p>Neither tree is wrong. Neither is buggy. A tree chooses its splits greedily, so a small change in which rows it sees can flip an early split, and a flipped early split rewrites everything below it. The model is <b>unstable</b>: its output is partly a function of the data and partly a function of the accident of which rows you happened to collect.</p>

<p>The obvious response is to average. Fit five hundred trees on five hundred resamples and take the mean prediction; the accidents should cancel. That instinct is correct, and it is worth being precise about how correct, because the obvious quantitative version of it is badly wrong.</p>

<p>The obvious version says: if one tree has variance $\\sigma^2$, then the average of $B$ of them has variance $\\sigma^2/B$, so five hundred trees give a five-hundred-fold reduction. Anyone who has actually built a random forest knows this does not happen. Going from 50 trees to 500 trees typically buys you almost nothing, and the curve flattens long before that. Something in the naive argument is false.</p>

<p>What is false is the assumption hiding inside "variances divide by $B$". That step needs the trees to be uncorrelated, and trees fitted to resamples of the same dataset are not remotely uncorrelated — they share most of their rows, they see the same dominant feature, and they make many of the same mistakes for the same reasons. Repairing the argument requires knowing exactly how variance behaves under addition, which is what this section is about. The repaired formula turns out to explain not just why the curve flattens but where the floor is and what you would have to do to lower it, and the answer to that last question is the entire design of the random forest.</p>

<h2><span class="sn">1.3.1</span> The three definitions, and how to read them</h2>

<p>§0.4 introduced expectation as a long-run average and variance as a spread. Here they need to be algebraic objects you can manipulate, so it is worth writing them down carefully.</p>

$$\\mathbb{E}[X]=\\sum_x x\\,p(x), \\quad \\mathrm{Var}(X)=\\mathbb{E}[X^2]-\\mathbb{E}[X]^2, \\quad \\mathrm{Cov}(X,Y)=\\mathbb{E}[XY]-\\mathbb{E}[X]\\mathbb{E}[Y]$$

<p>The symbol $\\mathbb{E}$ is a stylised capital E, read "the expectation of" or "the expected value of". The first formula says: go through every value the variable can take, multiply it by how likely that value is, and add. It is a weighted average with probabilities as the weights. A fair six-sided die has $\\mathbb{E}[X] = (1+2+3+4+5+6)/6 = 3.5$, which is not a face the die has — expectation is a centre of mass, not a prediction.</p>

<p><b>Variance</b> is the average squared distance from that centre. The definition you probably met first is $\\mathrm{Var}(X) = \\mathbb{E}[(X - \\mathbb{E}[X])^2]$, and the form written above is the same thing after expanding the square, which is usually easier to compute. Take the die again: $\\mathbb{E}[X^2] = (1+4+9+16+25+36)/6 = 91/6 = 15.167$, and $\\mathbb{E}[X]^2 = 3.5^2 = 12.25$, so the variance is $2.917$. The square root of that, $1.708$, is the <b>standard deviation</b>, and it is the one to quote to humans because it is in the same units as the data.</p>

<p><b>Covariance</b> is the same construction applied to two variables at once: $\\mathrm{Cov}(X,Y) = \\mathbb{E}[(X-\\mathbb{E}[X])(Y-\\mathbb{E}[Y])]$. Read the product inside. When $X$ is above its mean at the same times $Y$ is above its mean, both brackets are positive and the product is positive. When one is above while the other is below, the product is negative. Covariance averages those products, so it is positive when the two move together, negative when they move oppositely, and zero when there is no consistent pattern. Setting $Y = X$ recovers the variance exactly, which tells you that variance is not a separate concept: <b>variance is a variable's covariance with itself.</b></p>

<p>Collect the pairwise covariances of $d$ features into a $d \\times d$ grid and you have the <b>covariance matrix</b> $\\Sigma$ (capital sigma), with variances down the diagonal and covariances off it. It is symmetric, because $\\mathrm{Cov}(X,Y) = \\mathrm{Cov}(Y,X)$, and it is positive semi-definite (§1.8), meaning $v^\\mathsf{T}\\Sigma v \\ge 0$ for every direction $v$. That second fact is not decoration either: $v^\\mathsf{T}\\Sigma v$ is literally the variance of your data projected onto the direction $v$, and a variance cannot be negative. It is also exactly what makes PCA an eigenproblem and kernel methods convex.</p>

<h2><span class="sn">1.3.2</span> The asymmetry to memorise</h2>

${H.key('Linearity of expectation always holds; variances add only under independence.')}

<p>The first half of that sentence is stronger than it looks. $\\mathbb{E}[aX + bY] = a\\,\\mathbb{E}[X] + b\\,\\mathbb{E}[Y]$ holds with <b>no assumptions whatsoever</b> about how $X$ and $Y$ relate. They may be dependent, identical, adversarially constructed — it does not matter. This is why so many slick probability arguments run on expectations: you can decompose a complicated quantity into pieces, take expectations term by term, and never once justify an independence claim.</p>

<p>The second half is where the care is required.</p>

${H.deriv('$\\mathrm{Var}(X+Y) = \\mathrm{Var}(X) + \\mathrm{Var}(Y) + 2\\,\\mathrm{Cov}(X,Y)$', [
      ['$\\mathrm{Var}(X+Y) = \\mathbb{E}\\big[\\big((X+Y) - \\mathbb{E}[X+Y]\\big)^2\\big]$', 'The definition of variance applied to the single random variable $X+Y$: the average squared distance of that variable from its own mean.'],
      ['$= \\mathbb{E}\\big[\\big((X-\\mu_X) + (Y-\\mu_Y)\\big)^2\\big]$', 'By linearity of expectation, $\\mathbb{E}[X+Y] = \\mu_X + \\mu_Y$. Subtracting that from $X+Y$ and regrouping gives the sum of the two individual deviations. Note that linearity was used here without any independence assumption, exactly as advertised.'],
      ['$= \\mathbb{E}\\big[(X-\\mu_X)^2 + 2(X-\\mu_X)(Y-\\mu_Y) + (Y-\\mu_Y)^2\\big]$', 'Expand the square of a sum, $(p+q)^2 = p^2 + 2pq + q^2$. This is ordinary algebra performed on the random quantity inside the expectation, before any averaging happens, so it needs no probabilistic justification at all.'],
      ['$= \\mathbb{E}[(X-\\mu_X)^2] + 2\\,\\mathbb{E}[(X-\\mu_X)(Y-\\mu_Y)] + \\mathbb{E}[(Y-\\mu_Y)^2]$', 'Linearity of expectation again, this time to split the expectation of a three-term sum and to pull the constant 2 outside. Still no independence used, which is the point: this identity is unconditionally true.'],
      ['$= \\mathrm{Var}(X) + 2\\,\\mathrm{Cov}(X,Y) + \\mathrm{Var}(Y)$', 'Each of the three terms is now, verbatim, one of the definitions from the previous subsection.']
    ], 'So variances never simply add. What is true is that variances add <i>plus a cross term</i>, and that cross term vanishes precisely when the covariance is zero. Independence implies zero covariance, so independence is sufficient — but it is stronger than necessary, and the honest statement of the condition is "uncorrelated". The failure case is easy to feel: set $Y = X$, so the two are perfectly dependent. Then $\\mathrm{Var}(X+X) = \\mathrm{Var}(2X) = 4\\,\\mathrm{Var}(X)$, not $2\\,\\mathrm{Var}(X)$. The naive rule would have been wrong by a factor of two.')}

<h2><span class="sn">1.3.3</span> Bagging: the cross term, applied</h2>

<p>Now return to the five hundred trees. Model the situation as honestly as possible with as few assumptions as possible. Let $f_1, \\ldots, f_B$ be the predictions of $B$ models at one fixed input. Assume they are identically distributed, each with variance $\\sigma^2$, and that any two of them have the same pairwise correlation $\\rho$ (Greek rho), so that $\\mathrm{Cov}(f_b, f_c) = \\rho\\sigma^2$ whenever $b \\ne c$. Nothing here assumes the models are independent; $\\rho$ is exactly the knob measuring how far from independent they are.</p>

${H.deriv('the variance of an average of $B$ correlated models', [
      ['$\\mathrm{Var}\\!\\left(\\frac{1}{B}\\sum_b f_b\\right) = \\frac{1}{B^2}\\,\\mathrm{Var}\\!\\left(\\sum_b f_b\\right)$', 'Scaling a random variable by a constant $a$ multiplies its variance by $a^2$, because variance is built from a squared deviation. Here $a = 1/B$.'],
      ['$\\mathrm{Var}\\!\\left(\\sum_b f_b\\right) = \\sum_b \\mathrm{Var}(f_b) + \\sum_{b \\ne c}\\mathrm{Cov}(f_b, f_c)$', 'The two-variable identity from the previous derivation, extended to $B$ terms. Expanding the square of a $B$-term sum produces every squared term once and every cross product twice, and the second sum here runs over ordered pairs, which absorbs that factor of two.'],
      ['$= B\\sigma^2 + B(B-1)\\rho\\sigma^2$', 'Substitute the assumptions. There are exactly $B$ diagonal terms, each contributing $\\sigma^2$. There are $B(B-1)$ ordered pairs with $b \\ne c$, each contributing $\\rho\\sigma^2$.'],
      ['$\\mathrm{Var}(\\bar f) = \\frac{B\\sigma^2 + B(B-1)\\rho\\sigma^2}{B^2} = \\frac{\\sigma^2}{B} + \\frac{B-1}{B}\\rho\\sigma^2$', 'Divide by $B^2$ as line 1 requires, then cancel one factor of $B$ from each term.'],
      ['$= \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$', 'Write $\\frac{B-1}{B}$ as $1 - \\frac{1}{B}$ and collect the two terms that carry a $1/B$: $\\frac{\\sigma^2}{B} - \\frac{\\rho\\sigma^2}{B} = \\frac{(1-\\rho)\\sigma^2}{B}$. What is left over is the constant $\\rho\\sigma^2$.']
    ], 'Set $\\rho = 0$ and you recover the naive answer $\\sigma^2/B$, so the naive argument was not wrong, it was a special case of a formula whose general form has an extra term. That extra term is the whole story.')}

$$\\mathrm{Var}\\!\\left(\\frac{1}{B}\\sum_b f_b\\right) = \\underbrace{\\rho\\sigma^2}_{\\text{floor}} + \\underbrace{\\frac{1-\\rho}{B}\\sigma^2}_{\\text{killed by more models}}$$

<p>Read the two terms. The right-hand term contains $B$ and goes to zero as you add models: this is the part averaging buys you, and it is genuinely free. The left-hand term contains no $B$ at all. <b>Adding models cannot touch it.</b> It is a floor set entirely by how correlated your models are with one another, and the only way through the floor is to make them less alike.</p>

<p>Put numbers on it, because the numbers are what make the design decision obvious. Take $\\sigma^2 = 1$ and $\\rho = 0.6$, which is a realistic correlation for bagged trees on tabular data.</p>

<ul>
<li>One tree: variance $1.000$.</li>
<li>Ten trees: $0.6 + 0.4/10 = 0.640$. A 36 per cent reduction, for the price of nine extra fits.</li>
<li>One hundred trees: $0.6 + 0.4/100 = 0.604$. Going from ten trees to a hundred — ninety extra fits, ten times the training cost — bought a further 5.6 per cent.</li>
<li>One thousand trees: $0.6004$. Effectively nothing.</li>
</ul>

<p>Now change the other dial. Keep $B = 100$ and halve the correlation from $0.6$ to $0.3$: the variance falls from $0.604$ to $0.3 + 0.7/100 = 0.307$, a <b>49 per cent</b> reduction. Halving $\\rho$ was nine times more valuable than multiplying $B$ by ten, and it costs nothing at training time.</p>

<p>That comparison is the random forest, stated as arithmetic. Bagging alone attacks only the second term. Breiman's addition — at every split, consider only a random subset of the features — deliberately handicaps each individual tree, raising $\\sigma^2$ slightly, in exchange for decorrelating the trees and lowering $\\rho$ a great deal. It is a bad trade for any single tree and an excellent trade for the ensemble (§2.7). Once you can write the formula above, random forests need no further defence, and so do the extremely-randomised variants that push the same idea harder.</p>

${H.analogy(`<p>You want to know the temperature of a room and you have twenty cheap thermometers. Each one is imprecise, so you take all twenty readings and average them. The random jitter in each thermometer is independent of the others, so it averages away beautifully: twenty readings give you roughly a $\\sqrt{20}$-fold improvement in the jitter.</p>
<p>Now suppose all twenty thermometers are sitting on the same windowsill in direct sunlight. Every one of them reads three degrees high. Averaging twenty readings gives you a number that is precisely, confidently, three degrees wrong — and taking two hundred readings instead would give you the same wrong number with even more confidence.</p>
<p>The sunlight is $\\rho\\sigma^2$. It is the error the instruments share, and no amount of averaging removes shared error. The fix is not more thermometers; it is thermometers in different places. That is exactly what feature subsampling does to trees, and it is why "just add more models" is advice with a ceiling built into it.</p>`)}

<p><b>What you are looking at.</b> The horizontal axis is the number of models $B$, running from 1 to 100. The vertical axis is the variance of their average. The blue curve is the formula derived above, plotted as a continuous function of $B$, with dots marking the specific values at $B = 1, 5, 10, 25, 50$ and $100$ so you can read them off. The red horizontal line is the floor $\\rho\\sigma^2$ — the value the blue curve is falling towards and will never reach. The gap between the curve and the line at any $B$ is exactly the second term, which is what more models still have left to buy you. The readout gives the floor, the variance at $B = 10$ and $B = 100$, and the total percentage reduction relative to a single model.</p>

<p><b>What to do with it.</b> Set the correlation slider to zero first. The red floor drops to the axis and the blue curve heads for it like a textbook $1/B$ decay, which is the naive story everyone starts with. Now raise $\\rho$ slowly. Watch the floor rise to meet the curve, and watch the curve go flat earlier and earlier: at $\\rho = 0.8$ the curve is essentially horizontal by $B = 20$. The second slider scales everything vertically without changing the shape, because $\\sigma^2$ multiplies both terms.</p>

<p><b>The thing genuinely worth noticing.</b> Fix $\\rho = 0.6$ and read the "reduction at B = 100" figure, which rounds to 40 per cent. Now drop $\\rho$ to $0.3$ and read it again: 69 per cent. You did not add a single model. This is the pedagogical payload of the whole section — the second term is the one everyone thinks about and the first term is the one that decides the outcome. If you take one habit from this lab, let it be that when an ensemble stops improving, the diagnostic question is not "how many models do I have?" but "how alike are they?".</p>

${H.lab('bag', 'The bagging variance formula, as a picture', 'Move ρ and watch the floor rise. The gap between curve and floor is what more trees can still buy; random forests lower the floor itself by lowering ρ, which is why feature subsampling beats simply fitting more trees.')}

<h2><span class="sn">1.3.4</span> Covariance is a shape</h2>

<p>Covariance has an inconvenient property: its units are the product of the two variables' units. The covariance between income in pounds and age in years is measured in pound-years, and switching income to thousands of pounds divides it by a thousand without anything about the data having changed. So covariance is impossible to interpret on its own, and the standard repair is to divide out both scales:</p>

$$\\rho_{XY} = \\frac{\\mathrm{Cov}(X,Y)}{\\sqrt{\\mathrm{Var}(X)}\\,\\sqrt{\\mathrm{Var}(Y)}}$$

<p>This is the <b>Pearson correlation</b>, and it always lies between $-1$ and $1$. If that structure looks familiar, it should. Centre each variable by subtracting its mean, treat the $n$ observations as a vector of length $n$, and the formula above is a sum of pairwise products divided by two lengths — which is exactly the cosine similarity of §0.2. <mark>Correlation is the cosine of the angle between two mean-centred data columns.</mark> A correlation of 1 means the two centred columns point in identical directions in observation space; a correlation of 0 means they are orthogonal. Every geometric intuition you built about cosine similarity transfers here unchanged, including the important one: correlation says nothing about magnitudes, only about alignment.</p>

<p><b>What you are looking at.</b> Each blue dot is one observation with two features, plotted against each other. The red outline is the one-sigma ellipse of the sample covariance matrix and the fainter outline outside it is the two-sigma ellipse. The two arrows both start at the cloud's centre: the green one is the first eigenvector of the sample covariance matrix and the amber one is the second, each drawn with length $\\sqrt{\\lambda_i}$ so that the longer arrow is the direction of greater spread. The readout gives the sample covariance, the sample correlation, both eigenvalues, and the share of total variance carried by the first principal component.</p>

<p><b>What to do with it.</b> Begin with both standard deviations at 1 and the correlation at 0. The cloud is round, the ellipse is a circle, and the two eigenvalues are nearly equal — with no preferred direction, the eigenvectors are essentially arbitrary and will jitter as you nudge the point count. Now push the correlation towards 0.8. The cloud shears into a cigar, the green arrow swings to lie along it, and the "PC1 explains" readout climbs to about 90 per cent. That number is not a coincidence: with unit variances the eigenvalues are exactly $1+\\rho$ and $1-\\rho$, so PC1's share is $(1+\\rho)/2$, which at $\\rho = 0.8$ is 0.9. Then set the correlation back to zero and make the two standard deviations very different instead — the ellipse becomes an axis-aligned cigar, and PC1 has found a direction that reflects nothing but a choice of units.</p>

<p><b>The thing genuinely worth noticing.</b> The random draws are re-seeded identically on every redraw, so as you move the correlation slider each individual dot keeps its identity and the whole cloud deforms continuously rather than being resampled. Watch one dot near the edge while you sweep $\\rho$ from $-0.9$ to $0.9$. You are watching a single fixed set of independent draws being <i>sheared</i> — which is precisely what a covariance matrix is: a linear map applied to round noise (§0.2). That also explains the last observation in the previous paragraph. PCA finds the axes of the ellipse, and the ellipse depends on your units, so PCA on unstandardised features tells you which column was measured in the smallest units and very little else (§2.10).</p>

${H.lab('cov', 'Covariance as an ellipse', 'Two features, one cloud, the one- and two-sigma ellipses. The tilt is the covariance; the arrows are eigenvectors of the sample covariance scaled by $\\sqrt{\\lambda_i}$ — which is all PCA (§2.10) is.')}

${H.pitfall(`<p>The one-sigma ellipse does not contain 68 per cent of the points, and expecting it to is the most common misreading of a plot like the one above. That figure is a one-dimensional fact. In two dimensions the squared Mahalanobis distance follows a chi-squared distribution with two degrees of freedom, and the ellipse at $k$ standard deviations contains $1 - e^{-k^2/2}$ of the mass. So the one-sigma ellipse holds about <b>39 per cent</b> of the points, and the two-sigma ellipse about 86 per cent — not 95.</p>
<p>The effect strengthens with dimension, and it is the same phenomenon as the volume-migrates-to-the-shell result in §0.2. In high dimensions almost none of the probability mass sits near the centre of a Gaussian, because there is so much more room further out. This is why "it is within one standard deviation on every feature" is a much weaker statement about a 50-feature record than it sounds, and why anomaly detectors that threshold per-feature z-scores flag either everything or nothing.</p>`)}

<h2><span class="sn">1.3.5</span> Two facts that save you later</h2>

<p><b>Correlation is not dependence.</b> Zero correlation means no consistent <i>linear</i> relationship, and nothing more. Let $X$ be symmetric about zero and set $Y = X^2$. Then $\\mathrm{Cov}(X, Y) = \\mathbb{E}[X^3] - \\mathbb{E}[X]\\mathbb{E}[X^2] = 0 - 0 = 0$, because the odd moments of a symmetric distribution vanish. The correlation is exactly zero while $Y$ is a deterministic function of $X$ — total dependence, perfectly hidden. Mutual information (§1.10) catches what Pearson misses, and this counterexample is the one to have ready in an interview. The converse direction is safe only in one special case: for jointly Gaussian variables, zero correlation really does imply independence.</p>

<p><b>Variance of a mean.</b> For $n$ independent draws with common variance $\\sigma^2$, the cross terms all vanish and the derivation above collapses to $\\mathrm{Var}(\\bar X_n) = \\sigma^2/n$. Take the square root and the standard error of a sample mean is $\\sigma/\\sqrt{n}$. This is the single most consequential formula in applied statistics, it is the $\\rho = 0$ special case of the bagging formula, and it is why §1.4's bounds all carry a $\\sqrt{n}$ and why precision must be bought in quadratic instalments.</p>

${H.more('the law of total variance, and where bias–variance comes from', `<p>There is a decomposition that sits underneath §2.2 and is worth meeting here, where the machinery is fresh:</p>
$$\\mathrm{Var}(Y) = \\underbrace{\\mathbb{E}\\big[\\mathrm{Var}(Y \\mid X)\\big]}_{\\text{unexplained}} + \\underbrace{\\mathrm{Var}\\big(\\mathbb{E}[Y \\mid X]\\big)}_{\\text{explained}}$$
<p>In words: the total variability of your target splits cleanly into the average variability that remains once you know $X$, plus the variability of the conditional mean as $X$ moves around. The first term is the noise no model can remove — even a perfect model that knew the true conditional distribution would face it — and the second is the signal a model can in principle capture.</p>
<p>Two things follow. The first term is a hard floor on achievable error, so a model that is not beating it is not necessarily bad and a model claiming to beat it is almost certainly leaking. And the split is where the bias–variance decomposition comes from: §2.2 performs the same manoeuvre on the prediction error rather than on $Y$ itself, splitting it into irreducible noise, squared bias, and estimation variance — with the last of those being exactly the quantity the bagging formula above controls.</p>`)}

${H.practice(`<p><b>Sample correlations are far noisier than they look.</b> A correlation of 0.3 computed on 30 observations feels like evidence. It is not: the 95 per cent confidence interval, computed with Fisher's transformation, runs from about $-0.07$ to $0.60$. It includes zero. Two features can look meaningfully related in a small sample purely by accident, and feature-selection procedures that rank by correlation on small data are mostly ranking noise. Always ask how many rows a reported correlation was computed from.</p>
<p><b>Cross-validation folds are correlated, so the naive standard error is too small.</b> The tempting move after a 5-fold cross-validation is to take the five fold scores, compute their standard deviation, divide by $\\sqrt 5$, and quote that as the standard error of the estimate. That calculation assumes the five numbers are independent, and they are not: any two folds share four fifths of their training data, so their errors are positively correlated. The cross term is positive, the true variance is larger than the naive one, and the interval you report is too narrow. This is not a small technicality — it is known that no unbiased estimator of the variance of $K$-fold cross-validation exists in general. Treat fold-to-fold spread as a rough diagnostic of stability, not as a confidence interval, and prefer a proper resampling procedure or a genuinely held-out set when you need to defend a number (§2.14).</p>
<p><b>Covariance matrices are hard to estimate when $d$ is large.</b> A $d \\times d$ covariance matrix has $d(d+1)/2$ distinct entries, so 100 features means 5,050 parameters. Estimating that from a few hundred rows produces a matrix that is singular or nearly so, and every downstream method that inverts it — Mahalanobis distance, linear discriminant analysis, Gaussian mixture models — becomes numerically unstable in exactly the way §0.2 warned about. Shrinkage estimators, which pull the sample covariance towards a diagonal target, are the standard fix and are one line in most libraries.</p>`)}

${H.history(`<p>The idea that you could estimate the variability of a statistic by resampling your own data, rather than by deriving a formula, is startlingly recent. Bradley Efron introduced the bootstrap in 1979, and its reception was mixed precisely because it looked like something for nothing: how can shuffling the data you already have tell you about data you did not collect? The answer is that the empirical distribution is a legitimate stand-in for the true one, so sampling from it mimics sampling from the world.</p>
<p>Bagging — bootstrap aggregating — came from Leo Breiman in 1996, and it is the bootstrap turned from a measurement device into a modelling device: instead of using the resamples to estimate variance, use them to reduce it. Tin Kam Ho had independently published the random-subspace idea in 1995, fitting trees on random subsets of features. Breiman combined the two in the 2001 random forests paper, and the combination is exactly the two-term formula in §1.3.3: bootstrap resampling attacks $1/B$, feature subsampling attacks $\\rho$.</p>
<p>It is worth noticing what the sequence says about the field. The formula was not derived first and implemented second. The method worked, visibly and repeatedly, and the variance decomposition is the explanation that came afterwards and told everyone which knob mattered. Boosting (§2.8) has a similar history and a completely different mechanism — it reduces bias rather than variance, which is why its failure modes are the opposite ones.</p>`)}

${H.probe([
      ['Write the bagging variance formula.', '$\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ — averaging removes only the second term, so decorrelation is the remaining lever once $B$ is large.'],
      ['Why do random forests subsample features at each split?', 'To lower $\\rho$. It makes each individual tree slightly worse and the ensemble considerably better, because the floor $\\rho\\sigma^2$ is what dominates once $B$ is in the hundreds.'],
      ['Does zero correlation imply independence?', 'No — only for jointly Gaussian variables. $Y = X^2$ with symmetric $X$ has exactly zero correlation and complete dependence.'],
      ['Your 5-fold CV scores have standard deviation 0.02. Is the standard error 0.02/√5?', 'No. The folds share training data and are positively correlated, so the cross term is positive and the true variance exceeds the naive one. Fold spread is a stability diagnostic, not a confidence interval.'],
      ['What is $v^\\mathsf{T}\\Sigma v$?', 'The variance of the data projected onto direction $v$. It cannot be negative, which is precisely the statement that $\\Sigma$ is positive semi-definite — and maximising it over unit $v$ is PCA.']
    ], 'Asserting that averaging $B$ models divides variance by $B$. That is the $\\rho = 0$ special case, and it is the wrong case for anything trained on overlapping data.')}`,
    labs: {
      bag: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'pairwise correlation ρ', min: 0, max: .95, step: .01, value: .4, fmt: v => v.toFixed(2) },
          { k: 'sig', label: 'single-model variance σ²', min: .2, max: 3, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'floor', label: 'floor ρσ²', cls: 'key' }, { k: 'b10', label: 'variance @ B=10' },
          { k: 'b100', label: 'variance @ B=100' }, { k: 'red', label: 'reduction @ B=100' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const f = B => st.rho * st.sig + (1 - st.rho) / B * st.sig;
            const P = Viz.plot(ctx, w, h, { xd: [1, 100], yd: [0, st.sig * 1.05] })
              .frame({ xlabel: 'number of models B', ylabel: 'variance of the average' });
            P.clip(() => {
              P.fn(f, { color: T.blue, width: 2.6, n: 300 });
              P.hline(st.rho * st.sig, { color: T.red, label: 'floor ρσ² — only decorrelation goes below' });
              [1, 5, 10, 25, 50, 100].forEach(B => P.dots([[B, f(B)]], { r: 4, color: T.blue, stroke: true }));
            });
            out({
              floor: (st.rho * st.sig).toFixed(3), b10: f(10).toFixed(3), b100: f(100).toFixed(3),
              red: (100 * (1 - f(100) / st.sig)).toFixed(0) + '%'
            });
          }
        });
      },
      cov: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'correlation between features', min: -.95, max: .95, step: .01, value: 0, fmt: v => v.toFixed(2) },
          { k: 'sx', label: 'σ of feature 1', min: .3, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'sy', label: 'σ of feature 2', min: .3, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'points', min: 40, max: 1200, step: 20, value: 400, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'cov', label: 'sample covariance', cls: 'key' }, { k: 'cor', label: 'correlation' },
          { k: 'l1', label: 'λ₁' }, { k: 'l2', label: 'λ₂' }, { k: 'ev', label: 'PC1 explains' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(17);
            const pts = [];
            for (let i = 0; i < st.n; i++) {
              const z1 = R.normal(0, 1), z2 = R.normal(0, 1);
              pts.push([st.sx * z1, st.sy * (st.rho * z1 + Math.sqrt(1 - st.rho * st.rho) * z2)]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2, 3.2] }).frame({ xlabel: 'feature 1', ylabel: 'feature 2' });
            P.clip(() => P.dots(pts, { r: 2.6, color: T.blue, alpha: .5 }));
            const pc = Num.pca(pts);
            [1, 2].forEach(k => {
              const e = [];
              for (let t = 0; t <= 6.3; t += .05) {
                const a = Math.sqrt(Math.max(0, pc.values[0])) * k * Math.cos(t);
                const b = Math.sqrt(Math.max(0, pc.values[1])) * k * Math.sin(t);
                e.push([pc.mean[0] + a * pc.vectors[0][0] + b * pc.vectors[1][0],
                        pc.mean[1] + a * pc.vectors[0][1] + b * pc.vectors[1][1]]);
              }
              P.clip(() => P.line(e, { color: k === 1 ? T.red : T.faint, width: k === 1 ? 2 : 1, alpha: k === 1 ? 1 : .7 }));
            });
            pc.vectors.forEach((v, i) => {
              const s = Math.sqrt(Math.max(0, pc.values[i]));
              P.arrow(pc.mean[0], pc.mean[1], pc.mean[0] + v[0] * s, pc.mean[1] + v[1] * s, { color: i ? T.amber : T.green, width: 2.4 });
            });
            const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
            out({
              cov: Num.cov(xs, ys).toFixed(3), cor: Num.corr(xs, ys).toFixed(3),
              l1: pc.values[0].toFixed(3), l2: pc.values[1].toFixed(3),
              ev: (pc.explained[0] * 100).toFixed(0) + '%'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().red, t: '1σ ellipse' }, { c: Viz.theme().green, t: 'eigenvector 1 (PC1)' }, { c: Viz.theme().amber, t: 'eigenvector 2' }]);
      }
    },
    quiz: [
      {
        q: 'You double a bagged ensemble from 50 to 100 trees and variance barely moves. Why?',
        options: ['the trees are too shallow', 'ρ is high, so you are already at the floor ρσ²', 'the learning rate is too high', 'you need more features'],
        answer: 1,
        why: 'The variance of the average is ρσ² + (1−ρ)σ²/B. The second term is the only one that contains B, and by B = 50 it has already shrunk to almost nothing; the first term is a floor that no number of trees can go below. Shallow trees is the tempting distractor because depth genuinely does affect a single tree\'s variance σ², but changing σ² scales both terms together and does not change the shape of the curve. The productive move is to lower ρ — random feature subsets at each split — which is exactly what distinguishes a random forest from plain bagging (§2.7).'
      },
      {
        q: '$\\mathrm{Var}(X+Y)=\\mathrm{Var}(X)+\\mathrm{Var}(Y)$ when…',
        options: ['always', 'X and Y are uncorrelated', 'X and Y are identically distributed', 'X and Y are positive'],
        answer: 1,
        why: 'The full identity carries a cross term, Var(X) + Var(Y) + 2Cov(X,Y), and it holds unconditionally. Additivity therefore requires exactly one thing: that the covariance is zero. "Always" is tempting because expectation really is unconditionally linear, and it is easy to assume variance inherits that — set Y = X and the naive rule gives 2Var(X) where the truth is 4Var(X). Note that "uncorrelated" is weaker than "independent": independence implies zero covariance but is not required for it, so the honest condition is the one stated here.'
      },
      {
        q: 'The one-sigma ellipse of a two-dimensional Gaussian contains roughly what fraction of the points?',
        options: ['68%', '39%', '95%', '50%'],
        answer: 1,
        why: 'The 68 per cent figure is one-dimensional. In two dimensions the squared Mahalanobis distance is chi-squared with two degrees of freedom, so the k-sigma ellipse contains 1 − exp(−k²/2), which is 39 per cent at k = 1 and 86 per cent at k = 2. The intuition that fails here is that "within one sigma" should mean the same thing in every dimension; in fact the proportion of mass near the centre falls as dimension rises, which is the same volume-migrates-outward effect described in §0.2 and the reason per-feature z-score thresholds behave so badly on wide data.'
      }
    ],
    cards: [
      { q: 'Bagging variance formula', a: '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$ — more models remove only the second term; only decorrelation lowers the first.' },
      { q: 'Linearity vs additivity', a: 'Expectation is linear unconditionally; variance adds only when covariance is zero.' },
      { q: 'Ellipse axes of a 2-D cloud', a: 'Eigenvectors of the covariance matrix with lengths $\\sqrt{\\lambda_i}$ — i.e. PCA.' },
      { q: 'Correlation, geometrically', a: 'The cosine of the angle between two mean-centred data columns (§0.2).' },
      { q: 'Zero correlation counterexample', a: '$Y=X^2$ with symmetric $X$: correlation exactly 0, dependence total.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.4 */
  ML.section({
    id: 'concentration', track: 'foundations', num: '1.4',
    title: 'LLN, CLT, concentration — why generalisation is possible at all',
    lede: 'Nothing forces a number measured on data you have to say anything about data you do not have. This section is the argument that it does anyway: averages of independent things cluster tightly around their true value, at a rate you can compute, and that rate is the theoretical licence for the entire train/test protocol. A limit is not a guarantee about your 4,000-row test set; concentration is.',
    rests: 'Licenses: §2.1 the split · §2.14 trusting a validation number · §2.13 every reported metric.',
    html: `
<p>You held out 500 rows, scored the model on them, and got 91.2 per cent accuracy. You wrote 91.2 per cent in the slide deck. Two weeks after launch, production accuracy is sitting at 88 per cent, and someone asks whether the model got worse.</p>

<p>Before you can answer that you have to answer something more basic, and it is the question this entire section exists to make answerable. <b>Was 91.2 per cent ever a fact?</b></p>

<p>Consider what that number actually is. Those 500 rows were a sample. Had you shuffled the data with a different random seed you would have held out 500 different rows and got a different number. Nothing about the model would have changed. So the reported accuracy is not a property of the model alone — it is a property of the model <i>and</i> of an accident, and the size of the accident determines whether 91.2 and 88 are in conflict or in perfect agreement.</p>

<p>Here is the uncomfortable part. There is no purely logical argument that a measurement on 500 rows tells you anything at all about the next 500. The rows you did not collect are not obliged to resemble the rows you did. Every guarantee in machine learning has to come from an assumption about how the data was produced, and the assumption is always the same one: the rows are independent draws from a fixed distribution, and the test rows are drawn from the same distribution as the production ones. Grant that, and something remarkable becomes provable. Refuse it, and nothing in this section applies — which is why distribution shift (§2.16) is not a nuisance but a direct attack on the licence.</p>

<p>Grant it, then. What follows is the answer to "how big can the accident be?", and it turns out to be answerable with a precision that surprises people.</p>

<h2><span class="sn">1.4.1</span> Why averaging works, and why the rate is what it is</h2>

<p>Start with the mechanism rather than the theorem. Each test row contributes a 1 if the model got it right and a 0 if it did not. The reported accuracy is the average of those $n$ numbers. Write the true accuracy as $\\mu$ (mu) and the measured one as $\\bar X_n$, read "x bar sub n", the standard notation for a sample mean of $n$ items.</p>

<p>Each row's contribution differs from $\\mu$ by some amount, and those deviations are what you are trying to average away. There are two naive stories about what happens when you add $n$ of them up, and both are wrong in instructive ways.</p>

<p>The first naive story is that the errors accumulate. If each row is off by roughly the same amount and they all push the same way, then $n$ rows would be off by $n$ times as much, the sum would grow like $n$, the average would not shrink at all, and measurement would be impossible. This is what happens when the errors are <i>correlated</i> — which is precisely the situation in §1.3 with a high $\\rho$, and precisely the situation you are in when your 500 test rows are 500 events from 50 users.</p>

<p>The second naive story is that the errors cancel exactly. If they did, one row would be enough and nobody would need a test set.</p>

<p>The truth sits between the two, and §1.3 already gave you the exact position. For independent draws, variances add. So the <i>variance</i> of the sum of $n$ deviations is $n\\sigma^2$, and the standard deviation of the sum — the typical size of the total error — is $\\sigma\\sqrt{n}$. Divide by $n$ to turn the sum into an average, and the typical error of the average is</p>

$$\\mathrm{sd}(\\bar X_n) = \\frac{\\sigma\\sqrt{n}}{n} = \\frac{\\sigma}{\\sqrt{n}}$$

<p>There is the $1/\\sqrt{n}$, and notice that it did not require a new idea. It is $\\mathrm{Var}(\\bar X_n) = \\sigma^2/n$ from §1.3 with a square root taken. The reason precision improves like the square root of the sample size rather than linearly is simply that <b>errors combine like variances rather than like lengths</b>, because independent deviations pull in unrelated directions and partially cancel — but only partially.</p>

${H.analogy(`<p>A man leaves a lamp-post and takes one step every second, choosing left or right by a coin flip. After $n$ seconds, how far from the lamp-post is he?</p>
<p>He has walked $n$ steps, so the furthest he could be is $n$. He could also be exactly back at the lamp-post. The answer that actually occurs is neither: his typical distance is about $\\sqrt{n}$ steps. After 100 steps he is around 10 steps from the lamp-post. After 10,000 steps — nearly three hours of walking — he is typically about 100 steps away, which is one per cent of the distance he has covered.</p>
<p>This is the same arithmetic as the paragraph above, and it is the most useful mental image for the whole section. The walk does not stand still, and it does not run away in a straight line; it spreads out like the square root of time. Your accuracy estimate is a random walk of exactly this kind, and $\\sigma/\\sqrt{n}$ is how far it has typically strayed from the truth.</p>
<p>The image also tells you why more data has diminishing returns without your having to memorise anything. To halve the walker's typical distance you must quarter the number of steps. To halve the error on a measurement, you need four times the data.</p>`)}

<p>One more question deserves an answer before moving on, because it is the one that separates people who have understood this from people who have memorised it. <b>Could a cleverer estimator do better than $1/\\sqrt{n}$?</b> No. For estimating the mean of a bounded variable from $n$ independent samples, no procedure can beat this rate in the worst case; the limit is imposed by how much information $n$ samples contain, not by the crudeness of taking an average. The quadratic price for precision is a fact about the data, not about your method. That is worth knowing because it kills a whole category of hopeful thinking: no amount of statistical sophistication rescues a 200-row test set.</p>

<h2><span class="sn">1.4.2</span> Two limits, and why neither is what you need</h2>

<p>Classical probability offers two famous theorems about $\\bar X_n$, and it is important to see precisely why they do not close the argument.</p>

<p>The <b>law of large numbers</b> says that as $n$ grows without bound, $\\bar X_n$ converges to $\\mu$. This is reassuring and operationally useless. It says the accident eventually disappears; it says nothing about how big the accident is at $n = 500$. A statement about infinity does not constrain any finite number.</p>

<p>The <b>central limit theorem</b> is far more informative. It says the rescaled error becomes Gaussian:</p>

$$\\sqrt{n}\\,(\\bar X_n - \\mu) \\;\\to\\; \\mathcal{N}(0, \\sigma^2)$$

<p>Read it as: the error shrinks like $1/\\sqrt n$, and once you multiply it back up by $\\sqrt n$ to undo that shrinkage, what is left settles into a bell curve with variance $\\sigma^2$ — regardless of what the individual draws looked like. This is why the normal distribution is everywhere, and it is what every confidence interval you have ever seen is built on. But it too is a limiting statement. It tells you the shape the error distribution approaches, not how close it is at your $n$, and how fast the approach happens depends on the shape of what you are averaging — badly skewed variables converge much more slowly than symmetric ones, as §1.2's Binomial lab showed directly.</p>

<p>What you actually want is a statement of this form: <i>for this specific $n$, with no limits and no approximations, the probability that my measurement is off by more than this much is at most this.</i> Statements of that form are called <b>concentration inequalities</b>, and they are what licenses a number computed on a finite test set.</p>

<h2><span class="sn">1.4.3</span> Hoeffding's inequality</h2>

<p>The workhorse is due to Wassily Hoeffding. For independent random variables each bounded in the interval $[a,b]$, with sample mean $\\bar X_n$ and true mean $\\mu$,</p>

$$P\\big(|\\bar X_n - \\mu| \\ge t\\big) \\le 2\\exp\\!\\left(\\frac{-2nt^2}{(b-a)^2}\\right)$$

<p>Take it apart symbol by symbol. On the left, $t$ is the tolerance you care about — how far wrong you are willing to be — and the whole left-hand side is the probability of being wrong by at least that much, in either direction. The 2 on the right is there because "either direction" means two tails. Inside the exponential, $n$ appears in the numerator, so the bound falls <b>exponentially</b> as you add rows. The $t^2$ means that doubling your demanded precision costs four times as much, exactly as the random-walk picture predicted. And $(b-a)$ is the range of the quantity being averaged; for 0/1 loss it is 1, which is why this inequality is such a natural fit for accuracy, error rate, precision, recall, or any other metric built from counting.</p>

${H.key('Averaging kills independent noise, and it does so at a rate you can quote — exponentially in the number of rows, with a tolerance that shrinks like $1/\\sqrt{n}$.')}

<p>The practical value of the inequality is that it inverts. You rarely want to know the failure probability at a given $n$; you want to know the $n$ that delivers a failure probability you can live with.</p>

${H.deriv('how many test rows do I need?', [
      ['$P(|\\bar X_n - \\mu| \\ge t) \\le 2\\exp\\!\\left(\\dfrac{-2nt^2}{(b-a)^2}\\right)$', 'Hoeffding\'s inequality as stated above. It requires only independence and boundedness — no assumption about the shape of the distribution, and no limit.'],
      ['$\\le 2\\exp(-2nt^2)$', 'Specialise to 0/1 loss, where every observation is either 0 or 1 so the range $b-a$ equals 1 and the denominator disappears. This covers accuracy, error rate, and any metric that is an average of indicator variables.'],
      ['$2\\exp(-2nt^2) \\le \\delta$', 'Now impose the requirement. $\\delta$ (delta) is the failure probability you are willing to accept; $\\delta = 0.05$ is the usual "95 per cent confidence". If the bound is below $\\delta$, then so is the thing it bounds.'],
      ['$\\exp(-2nt^2) \\le \\delta/2$', 'Divide both sides by 2. Dividing an inequality by a positive constant preserves its direction.'],
      ['$-2nt^2 \\le \\ln(\\delta/2)$', 'Take natural logarithms of both sides. $\\ln$ is strictly increasing, so it preserves the direction of an inequality between positive quantities.'],
      ['$2nt^2 \\ge \\ln(2/\\delta)$', 'Multiply through by $-1$, which reverses the inequality, and use $-\\ln(\\delta/2) = \\ln(2/\\delta)$. Since $\\delta < 2$, the right-hand side is positive, as it must be.'],
      ['$n \\ge \\dfrac{\\ln(2/\\delta)}{2t^2}$', 'Divide by the positive quantity $2t^2$. This is the sample-size formula: rows needed, as a function of the tolerance you want and the confidence you demand.']
    ], 'Read the final line for its shape before its numbers. The confidence $\\delta$ enters through a <i>logarithm</i>, so demanding 99 per cent confidence instead of 95 per cent costs only a factor of $\\ln(200)/\\ln(40) = 1.44$. The tolerance $t$ enters as an inverse <i>square</i>, so halving it costs a factor of four. Confidence is cheap; precision is expensive. That asymmetry is the single most useful thing to carry out of this section, and it is why nobody ever argues about whether to use 95 or 99 per cent while everybody argues about how much data they need.')}

${H.worked('worked number — how many test rows?', `
<p>0/1 loss, so $b-a=1$. Want 95 per cent confidence that the estimate is within $t = 0.02$ — that is, two accuracy points either side:</p>
$$2e^{-2nt^2} = 0.05 \\;\\Rightarrow\\; n = \\frac{\\ln 40}{2(0.0004)} = \\frac{3.6889}{0.0008} \\approx 4{,}612 \\text{ rows}$$
<p>Check it forwards: $2\\exp(-2 \\times 4{,}612 \\times 0.0004) = 2\\exp(-3.69) = 0.050$. Correct.</p>
<p>Now want half the width, $t = 0.01$: $n = 3.6889/0.0002 \\approx 18{,}400$ rows — four times as many for one extra point of precision. Error falls like $1/\\sqrt n$, so <b>precision is bought in quadratic instalments</b>.</p>
<p>Run it in the other direction on the 500-row holdout from the top of this section. With $n = 500$ and $\\delta = 0.05$, the guaranteed tolerance is $t = \\sqrt{\\ln(40)/(2 \\times 500)} = \\sqrt{0.00369} = 0.0607$, or <b>±6.1 accuracy points</b>. So 91.2 per cent was, honestly reported, "somewhere between 85 and 97 per cent". Production coming in at 88 per cent is not evidence of anything at all. Nobody needed to investigate, and the slide deck should never have carried three significant figures.</p>
<p>A 50-row holdout is worse than most people imagine: $t = \\sqrt{3.6889/100} = 0.192$, or <b>±19 points</b>. The normal-approximation interval is kinder at ±14 points, but a metric with a fourteen-point interval is not a metric, it is a rumour.</p>`)}

<p><b>What you are looking at.</b> The horizontal axis is the number of test rows $n$, from 50 to 20,000. The vertical axis is the probability that your measured value is off by more than the tolerance $t$ — so lower is better, and the curves all fall to the right. Three bounds are drawn: Chebyshev in amber, Bernstein in green and Hoeffding in blue, each one an upper bound on that failure probability under different assumptions. The red horizontal line sits at 5 per cent, so wherever a curve crosses below it you have 95 per cent confidence. The dashed vertical line marks whatever $n$ the first slider is set to, and the blue dot is the Hoeffding bound at that exact point. The readout gives all three bounds at the current $n$, the number of rows needed for 95 per cent confidence at the current tolerance, and — for comparison — the half-width of the ordinary CLT confidence interval.</p>

<p><b>What to do with it.</b> Leave the tolerance at its default ±2.0 points and drag $n$ until the blue dot drops under the red line. It happens at about 4,600, matching the worked number above. Now change the tolerance to ±1.0 points and repeat: the crossing moves out to roughly 18,400. Watch the "n for 95%" readout as you move the tolerance slider continuously — it is the sample-size formula computed live, and dragging $t$ from 0.02 to 0.01 makes it jump by a factor of four rather than two.</p>

<p><b>The thing genuinely worth noticing.</b> The third slider is the variance $\\sigma^2$, and at its default value of 0.25 — the worst case for a 0/1 variable, since Bernoulli variance peaks at $p = 0.5$ — the green Bernstein curve sits <i>above</i> the blue Hoeffding one. Bernstein, the more sophisticated bound, is losing. Now drag $\\sigma^2$ down to 0.09, which is the variance you actually have if your model is 90 per cent accurate. The green curve plunges: at $n = 1{,}000$ it falls from about 92 per cent to about 25 per cent, and at $n = 4{,}612$ it is about 0.014 per cent — which the readout rounds to 0.0 — where Hoeffding is still sitting at 5. The lesson is not that Bernstein is better. It is that <b>a bound is only as tight as the assumption it is allowed to use</b>. Hoeffding is forced to imagine the hardest possible variable consistent with being bounded in $[0,1]$; Bernstein is told the variance and can exploit the fact that a 90-per-cent-accurate model produces mostly ones. Turn that into a sample size and the difference is stark: Bernstein needs about 1,783 rows for ±2 points at 95 per cent confidence, against Hoeffding's 4,612. Knowing your accuracy is high is worth two thirds of your test set.</p>

${H.lab('conc', 'The concentration ladder, and the sample-size calculator', 'Each rung assumes more and pays in tightness. Move n and t; the calculator answers the question you are actually asked — how many rows do I need? Then drop σ² from 0.25 to 0.09 and watch Bernstein overtake Hoeffding.')}

<h2><span class="sn">1.4.4</span> The ladder</h2>

<p>Hoeffding is one rung on a ladder of inequalities. Each rung assumes strictly more about the random variable and is repaid in a tighter bound. Knowing the ladder is useful because it tells you what to reach for when the standard tool is too loose.</p>

${H.table(['Inequality', 'Assumes', 'Bound'], [
      ['<b>Markov</b>', 'non-negativity only', '$P(X\\ge a)\\le \\mathbb{E}[X]/a$'],
      ['<b>Chebyshev</b>', '+ finite variance', '$P(|X-\\mu|\\ge t)\\le \\sigma^2/t^2$'],
      ['<b>Hoeffding</b>', '+ bounded range', '$2\\exp(-2nt^2/(b-a)^2)$'],
      ['<b>Bernstein</b>', '+ variance and bound together', '$2\\exp\\!\\left(\\frac{-nt^2}{2\\sigma^2+2Mt/3}\\right)$']
    ])}

<p><b>Markov</b> is the base of everything and needs almost nothing: if a quantity cannot be negative, it cannot often be much larger than its average, because a large value has to be paid for by the average. It is almost always too weak to use directly, and almost always the first step in proving something stronger.</p>

<p><b>Chebyshev</b> adds a finite variance and gets a bound that decays like $1/t^2$ — polynomially. Applied to a sample mean, whose variance is $\\sigma^2/n$, it gives $P(|\\bar X_n - \\mu| \\ge t) \\le \\sigma^2/(nt^2)$. That is genuinely useful and it is remarkably weak. At $n = 1{,}000$ with $t = 0.02$ and the worst-case $\\sigma^2 = 0.25$, it bounds the failure probability at $0.25/(1000 \\times 0.0004) = 62.5$ per cent, which tells you essentially nothing. It gets there having assumed essentially nothing.</p>

<p><b>Hoeffding</b> adds boundedness and jumps from polynomial decay to <i>exponential</i> decay. That jump is the whole reason machine learning has a theory, and it is worth understanding where it comes from.</p>

${H.intuition(`<p>Why does knowing the variable is bounded buy you an exponential rather than a polynomial?</p>
<p>Chebyshev only knows one thing about the distribution: its second moment. A distribution with a given variance is allowed to be extremely nasty — it can put a tiny sliver of probability enormously far out, and Chebyshev has to remain true even for that distribution. So the bound has to be pessimistic in a way that ruins it.</p>
<p>Boundedness removes that possibility entirely. If every draw lies in $[0,1]$, then for the average of a thousand draws to be off by 0.02, roughly twenty draws' worth of deviation has to line up in the same direction, out of a thousand independent chances. Independent events do not conspire cheaply: the probability of many independent things simultaneously pulling the same way is a <i>product</i> of small numbers, and a product of many small numbers is exponentially small. Hoeffding is the formalisation of exactly that sentence.</p>
<p>The technical device that makes it rigorous is worth naming because it explains the shape of every bound in the table. Rather than bounding $P(\\bar X_n - \\mu \\ge t)$ directly, you bound $P(e^{s(\\bar X_n - \\mu)} \\ge e^{st})$ for a positive $s$ you get to choose, apply Markov to that non-negative quantity, use independence to turn the expectation of a product into a product of expectations, and finally pick the $s$ that makes the result smallest. The exponential in the answer is there because you put an exponential in by hand. This is the <b>Chernoff method</b>, and every rung above Chebyshev is an instance of it.</p>`)}

<p><b>Bernstein</b> refines Hoeffding by taking both the variance and the bound into account, which is why it wins so decisively in the lab when the variance is small. Look at its denominator: $2\\sigma^2 + 2Mt/3$. When $\\sigma^2$ is large the first term dominates and Bernstein behaves like a variance-driven bound; when the variance is tiny the second term takes over and the decay becomes even faster. Hoeffding is effectively Bernstein forced to assume the worst possible variance, which for a bounded variable is $(b-a)^2/4$.</p>

${H.note('Hoeffding is the rung where the bound becomes useful for a test set, because it needs only boundedness — which 0/1 loss gives you for free, with no estimation and no assumption you could get wrong.')}

<h2><span class="sn">1.4.5</span> The part everybody skips: you did not measure one model</h2>

<p>Everything above bounds the error of <i>one</i> number measured once. That is not what anyone does. You trained forty configurations, scored all forty on the same validation set, and picked the best. The winner's score is not a measurement of that model; it is the maximum of forty measurements, and the maximum of a set of noisy numbers is biased upwards.</p>

<p>Quantify it. Suppose you have 1,000 validation rows and twenty candidate models that are genuinely equally good, all at 90 per cent true accuracy. Each model's measured score has a standard error of $\\sqrt{0.9 \\times 0.1 / 1000} = 0.0095$, or about one accuracy point. The expected maximum of twenty independent draws from a standard normal is about 1.87 standard deviations above the mean, so <b>the winner will appear about 1.8 accuracy points better than it truly is</b>, purely from selection. Nothing was learnt. You have measured the luckiest model, not the best one. (Real candidates are correlated with each other, which shrinks the effect somewhat but does not remove it.)</p>

<p>The repair is the <b>union bound</b>, which is trivial to state and surprisingly powerful: the probability that at least one of $m$ events happens is at most the sum of their individual probabilities. Apply it across your $m$ models, demand that <i>all</i> of them are accurate to within $t$ simultaneously, and the sample-size formula picks up one extra term:</p>

$$n \\ge \\frac{\\ln(2m/\\delta)}{2t^2}$$

<p>Now look at what that costs. For ±2 points at 95 per cent confidence, one model needs 4,612 rows. One hundred models need $\\ln(4000)/0.0008 = 10{,}368$ rows — a little over twice as many. Ten thousand models need $\\ln(400{,}000)/0.0008 = 16{,}124$ rows. Multiplying the number of candidates by a hundred cost you a factor of 1.55 in data.</p>

<p>That logarithm is one of the most important facts in learning theory, and it deserves a moment. It says that searching a large space of hypotheses is <i>affordable</i>: the price of comparing many models grows only logarithmically in how many you compare. If the price grew linearly, machine learning would not work, because a linear model with $d$ real-valued weights is effectively an enormous hypothesis space. Making that argument precise for infinite hypothesis classes — where you cannot simply count $m$ — is what VC dimension and Rademacher complexity are for, and this union-bound calculation is the finite, honest version of the same idea. It is also, immediately and practically, why a leaderboard with two thousand submissions against one public test set is measuring luck.</p>

${H.pitfall(`<p>The specific mistake is <b>reusing the test set</b>, and it does not feel like a mistake while you are making it, because each individual look is innocent. You evaluate, you adjust a hyperparameter, you evaluate again. Nobody trained on the test set. Yet after fifty rounds of this, the test set has become a training set with a very slow gradient signal passing through your own judgement, and its number no longer means what it says.</p>
<p>The tell is a test score that keeps improving in increments smaller than the standard error. If your test set gives ±1 point and your last six changes bought 0.3 points each, you have not improved the model six times; you have found six ways to fit the noise in that particular sample.</p>
<p>Three defences, in ascending order of discipline. Use a validation set for all decisions and touch the test set once, at the end, to produce the number you report. Report an interval rather than a point, which makes the noise floor visible to you and to everyone reading. And if you must iterate many times against held-out data, set the tolerance you need using the union-bound formula above with $m$ equal to the number of looks you honestly expect to take.</p>`)}

${H.practice(`<p><b>Your rows are usually not independent, and this is the assumption that fails first.</b> Every bound above requires independent draws. A test set of 5,000 clickstream events drawn from 500 users is not 5,000 independent observations, because events from the same user share that user's habits, device and intent. The standard adjustment is the design effect: with clusters of size $m$ and within-cluster correlation $\\rho$, the effective sample size is $n / (1 + (m-1)\\rho)$. With 10 events per user and $\\rho = 0.5$, that is $5{,}000 / 5.5 = 909$ effective rows. Hoeffding on 909 rows guarantees ±4.5 points, not the ±1.9 you would have claimed from 5,000. If your data has natural groups — users, sessions, patients, shops — split by group and count your sample size in groups.</p>
<p><b>Report intervals, not points.</b> The single highest-return habit in this whole section is to write "88.4% ± 1.2" instead of "88.4%". It costs one line of code, it makes the reuse pitfall above self-diagnosing, and it prevents the entire class of meeting in which two numbers that differ by less than their noise are debated for an hour. For a proportion, the normal-approximation interval $\\hat p \\pm 1.96\\sqrt{\\hat p(1-\\hat p)/n}$ is adequate whenever $n\\hat p$ and $n(1-\\hat p)$ both exceed about ten; below that, use a Wilson or bootstrap interval instead, for exactly the reason §1.2's lab demonstrated.</p>
<p><b>For anything that is not a simple average, bootstrap it.</b> AUC, F1, calibration error and ranking metrics are not means of independent terms, so Hoeffding does not apply to them directly. Resample your test set with replacement two thousand times, recompute the metric each time, and read the 2.5th and 97.5th percentiles. This takes about five lines and works for any metric at all.</p>
<p><b>Size the test set on purpose.</b> Before splitting, decide what difference would change a decision. If a one-point difference matters commercially, you need roughly 18,000 test rows and there is no way around it; if only a five-point difference matters, 740 rows will do. Deciding this in advance converts an argument about splits into an arithmetic problem.</p>`)}

<h2><span class="sn">1.4.6</span> See the limits happen</h2>

<p><b>What you are looking at.</b> This lab runs an experiment thousands of times over. Each experiment draws $n$ values from the chosen distribution and computes their mean; the blue histogram is the distribution of those means across all the repeats. So one bar does not represent one data point — it represents how often a whole experiment produced a mean in that range. The red curve is the Gaussian the central limit theorem predicts, centred on the true mean with standard deviation $\\sigma/\\sqrt n$, and the amber vertical line marks the true mean itself. The readout gives the true mean, the average of the sample means, the observed spread of the sample means, the $\\sigma/\\sqrt n$ that theory predicts, and the skew of the distribution of means.</p>

<p><b>What to do with it.</b> Start on the Bernoulli option, which is the accuracy case, with $n = 1$. The histogram is two spikes, because a single draw is either 0 or 1 and nothing else is possible — no bell curve anywhere in sight. Now walk $n$ up: 2, 5, 10, 30. The spikes multiply, merge, and a recognisable bell appears surprisingly early. Watch the "observed SE" and "σ/√n predicted" rows converge as you go, and watch the histogram narrow by exactly the amount that formula demands — at $n = 100$ it should be about a third as wide as at $n = 10$, because $\\sqrt{10} = 3.16$.</p>

<p><b>The thing genuinely worth noticing.</b> Switch to the heavy-tailed Pareto option and push $n$ all the way to 200. The histogram is still visibly skewed, with a long right tail, and the red Gaussian still does not fit. This is the honest answer to the folklore that "$n \\ge 30$ is enough for the CLT". Thirty is not a theorem; how fast you converge depends on the shape of what you are averaging. The specific Pareto used here has a tail index of 2.5, which means its variance is finite — so the CLT does apply and the limit does exist — but its third moment is infinite, and the standard theorem quantifying the speed of convergence (Berry–Esseen) depends on exactly that third moment. With it infinite, the approach to normality is slower than $1/\\sqrt n$ and you can watch that happen. Two readouts will not agree on this option: the predicted-SE row uses a value hard-coded in the lab rather than the distribution's true standard deviation of 1.49, so read the observed SE and ignore the prediction here. On the other three distributions they match closely, which is the point.</p>

${H.lab('clt', 'LLN and CLT, simulated', 'Each bar is how often a whole experiment produced a mean in that range. Pick a distribution — including a badly skewed one — and watch the distribution of sample means become Gaussian. The heavy-tailed option is still not Gaussian at n = 200, which shows why "n ≥ 30" is folklore rather than a theorem.')}

${H.history(`<p>Jacob Bernoulli proved the first law of large numbers and it took him about twenty years. The result appeared in <i>Ars Conjectandi</i> in 1713, eight years after his death. What he wanted was not the limiting statement everyone quotes but the practical one this section is about: how many trials do you need before your observed frequency is close enough to the true rate that a reasonable person would act on it?</p>
<p>His worked example asked for a proportion to be pinned down within one fiftieth, with what he called moral certainty — odds of 1000 to 1. His answer was 25,550 trials. He appears to have found this discouraging, and it is easy to see why: no experiment available to him produced 25,550 of anything. Run the same specification through Hoeffding's 1963 inequality and you get about 9,500. Two and a half centuries of sharpening the same argument cut the requirement by nearly two thirds, and the modern bound is still conservative.</p>
<p>The intervening steps: de Moivre found the normal approximation to the binomial in 1733, giving the first version of the CLT; Laplace generalised it; Bienaymé and Chebyshev produced the variance-based inequality in the 1850s and 1860s; Bernstein sharpened it in the 1920s; and Hoeffding published the bounded-variable version in 1963, in a paper that has since been cited by essentially everyone in statistical learning theory.</p>
<p>The last step is the one that made this a machine learning subject rather than a statistics one. In 1971 Vapnik and Chervonenkis asked what happens when you do not measure one fixed quantity but choose the best of a whole family of them — exactly the model-selection problem of §1.4.5 — and showed that the union-bound argument extends to infinite families through a capacity measure now called VC dimension. That is the moment generalisation stopped being a hope and became a theorem with a hypothesis you could check.</p>`)}

${H.more('the Chernoff method in four lines, for the curious', `<p>Every bound below Chebyshev on the ladder comes from one manoeuvre. Let $S = \\sum_i (X_i - \\mu)$ and pick any $s > 0$.</p>
$$P(S \\ge nt) = P\\big(e^{sS} \\ge e^{snt}\\big) \\le \\frac{\\mathbb{E}[e^{sS}]}{e^{snt}} = \\frac{\\prod_i \\mathbb{E}[e^{s(X_i-\\mu)}]}{e^{snt}}$$
<p>The first equality holds because $x \\mapsto e^{sx}$ is strictly increasing, so the two events are literally the same event. The inequality is Markov's, applied to the non-negative variable $e^{sS}$. The final step splits the expectation of a product into a product of expectations, which is where independence is spent — and it is the only place it is used.</p>
<p>The quantity $\\mathbb{E}[e^{s(X-\\mu)}]$ is the <b>moment generating function</b>, and bounding it is the whole game. Hoeffding's lemma says that for a variable bounded in $[a,b]$ it is at most $\\exp(s^2(b-a)^2/8)$. Substitute, and you have $\\exp(ns^2(b-a)^2/8 - snt)$. That is a quadratic in $s$ sitting inside an exponential, so minimise it over $s$ — the optimum is $s = 4t/(b-a)^2$ — and out drops $\\exp(-2nt^2/(b-a)^2)$, which is Hoeffding's inequality. Double it for the two-sided version.</p>
<p>A variable whose moment generating function is bounded by a Gaussian's in this way is called <b>sub-Gaussian</b>, and that is the property doing the real work. Bounded implies sub-Gaussian, which is why boundedness is enough. It also tells you exactly when these bounds stop applying: heavy-tailed losses are not sub-Gaussian, so a regression metric on unbounded errors gets no exponential guarantee, and you should bootstrap instead.</p>`)}

${H.probe([
      ['Why is a bigger test set more trustworthy?', 'Estimation error falls like $1/\\sqrt n$ because independent deviations combine as variances, and Hoeffding turns that into an explicit exponential tail bound that holds at every finite $n$.'],
      ['How many rows do you need?', 'Invert Hoeffding: $n = \\ln(2/\\delta)/(2t^2)$. Roughly 4,600 for ±2 points at 95 per cent, 18,000 for ±1.'],
      ['CLT or concentration for a finite sample?', 'Concentration. The CLT is asymptotic and its speed depends on the shape of the summands; Hoeffding holds at every $n$ with no approximation.'],
      ['You compared 200 configurations on one validation set. What changes?', 'The union bound: you need $\\ln(2m/\\delta)/(2t^2)$ rows for all of them to be simultaneously accurate. It grows only logarithmically in $m$ — 200 models cost about 2.4× the rows of one — which is why searching large spaces is affordable at all.'],
      ['Your test rows are 10 events each from 500 users. Now what?', 'They are not independent, so the effective sample size is roughly $n/(1+(m-1)\\rho)$. At $\\rho=0.5$ that is 909, not 5,000, and the honest interval is about two and a third times wider. Split by user and count in users.']
    ], 'Reporting a metric from a 50-row holdout as if it were a fact. Its normal-approximation interval is about ±14 points and its Hoeffding guarantee is ±19, so the number carries roughly one bit of information.')}`,
    labs: {
      conc: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'test rows n', min: 50, max: 20000, step: 50, value: 1000, fmt: v => v.toLocaleString() },
          { k: 't', label: 'tolerance t', min: .005, max: .2, step: .005, value: .02, fmt: v => '±' + (v * 100).toFixed(1) + ' pts' },
          { k: 'sig', label: 'σ² (Chebyshev / Bernstein)', min: .01, max: .25, step: .005, value: .25, fmt: v => v.toFixed(3) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'hoef', label: 'Hoeffding bound', cls: 'key' }, { k: 'cheb', label: 'Chebyshev bound' },
          { k: 'bern', label: 'Bernstein bound' }, { k: 'need', label: 'n for 95% @ this t' }, { k: 'ci', label: 'CLT 95% half-width' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const hoef = n => Math.min(1, 2 * Math.exp(-2 * n * st.t * st.t));
            const cheb = n => Math.min(1, st.sig / (n * st.t * st.t));
            const bern = n => Math.min(1, 2 * Math.exp(-n * st.t * st.t / (2 * st.sig + 2 * st.t / 3)));
            const P = Viz.plot(ctx, w, h, { xd: [50, 20000], yd: [0, 1] })
              .frame({ xlabel: 'test rows n', ylabel: 'P(estimate off by more than t)' });
            P.clip(() => {
              P.fn(cheb, { color: T.amber, width: 2, n: 300 });
              P.fn(bern, { color: T.green, width: 2, n: 300 });
              P.fn(hoef, { color: T.blue, width: 2.6, n: 300 });
              P.hline(.05, { color: T.red, label: '5% risk' });
              P.vline(st.n, { color: T.text, dash: [3, 3] });
              P.dots([[st.n, hoef(st.n)]], { r: 5, color: T.blue, stroke: true });
            });
            out({
              hoef: (hoef(st.n) * 100).toFixed(1) + '%', cheb: (cheb(st.n) * 100).toFixed(1) + '%',
              bern: (bern(st.n) * 100).toFixed(1) + '%',
              need: Math.ceil(Math.log(40) / (2 * st.t * st.t)).toLocaleString(),
              ci: '±' + (1.96 * Math.sqrt(.25 / st.n) * 100).toFixed(2) + ' pts'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'Hoeffding' }, { c: Viz.theme().green, t: 'Bernstein' }, { c: Viz.theme().amber, t: 'Chebyshev' }, { c: Viz.theme().red, t: '5% risk line' }]);
        Viz.note(host, 'Hoeffding is conservative next to the CLT interval — that is the price of a guarantee holding at every finite n rather than in the limit.');
      },

      clt: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'underlying distribution', type: 'select', value: 'bern', options: [
            { v: 'bern', t: 'Bernoulli(0.3) — accuracy' }, { v: 'expo', t: 'Exponential — skewed' },
            { v: 'unif', t: 'Uniform' }, { v: 'pareto', t: 'Heavy-tailed (Pareto)' }] },
          { k: 'n', label: 'sample size n per experiment', min: 1, max: 200, step: 1, value: 30, fmt: v => v },
          { k: 'reps', label: 'experiments repeated', min: 100, max: 8000, step: 100, value: 2000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mu', label: 'true mean' }, { k: 'sm', label: 'mean of means', cls: 'key' },
          { k: 'se', label: 'observed SE' }, { k: 'pred', label: 'σ/√n predicted' }, { k: 'skew', label: 'skew of means' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(23);
            let draw, mu, sig;
            if (st.d === 'bern') { draw = () => R() < .3 ? 1 : 0; mu = .3; sig = Math.sqrt(.21); }
            else if (st.d === 'expo') { draw = () => R.exp(1); mu = 1; sig = 1; }
            else if (st.d === 'unif') { draw = () => R(); mu = .5; sig = Math.sqrt(1 / 12); }
            else { draw = () => Math.pow(1 - R(), -1 / 2.5); mu = 2.5 / 1.5; sig = 2.4; }
            const means = [];
            for (let r = 0; r < st.reps; r++) { let s = 0; for (let i = 0; i < st.n; i++) s += draw(); means.push(s / st.n); }
            const lo = Math.max(Math.min.apply(null, means), mu - 5 * sig / Math.sqrt(st.n));
            const hi = Math.min(Math.max.apply(null, means), mu + 5 * sig / Math.sqrt(st.n));
            const hh = Num.hist(means, 40, lo, hi);
            const dens = hh.bins.map(c => c / (means.length * hh.w));
            const P = Viz.plot(ctx, w, h, { xd: [lo, hi], yd: [0, Math.max.apply(null, dens) * 1.2 || 1] })
              .frame({ xlabel: 'sample mean of n draws', ylabel: 'density' });
            P.clip(() => {
              hh.centers.forEach((c, i) => {
                ctx.fillStyle = T.blue; ctx.globalAlpha = .3;
                const x0 = P.x(c - hh.w / 2), x1 = P.x(c + hh.w / 2);
                ctx.fillRect(x0, P.y(dens[i]), Math.max(1, x1 - x0 - 1), P.y(0) - P.y(dens[i]));
                ctx.globalAlpha = 1;
              });
              P.fn(x => Num.normPdf(x, mu, sig / Math.sqrt(st.n)), { color: T.red, width: 2.2 });
              P.vline(mu, { color: T.amber, label: 'true mean' });
            });
            const m2 = Num.mean(means), s2 = Num.sd(means);
            out({
              mu: mu.toFixed(3), sm: m2.toFixed(4), se: s2.toFixed(4),
              pred: (sig / Math.sqrt(st.n)).toFixed(4),
              skew: Num.mean(means.map(v => Math.pow((v - m2) / (s2 || 1), 3))).toFixed(2)
            });
          }
        });
        Viz.note(host, 'Red curve = the Gaussian the CLT predicts. On the heavy-tailed option the match stays poor even at n = 200 — the CLT needs finite variance.');
      }
    },
    quiz: [
      {
        q: 'You want validation accuracy to ±1 point at 95% confidence with 0/1 loss. Roughly how many rows?',
        options: ['about 500', 'about 4,600', 'about 18,000', 'about 100,000'],
        answer: 2,
        why: 'Invert Hoeffding: n = ln(2/δ)/(2t²), which at δ = 0.05 and t = 0.01 gives ln(40)/0.0002 ≈ 18,400. The 4,600 option is tempting because it is the right answer to the neighbouring question — ±2 points rather than ±1 — and halving the tolerance feels as though it should roughly double the requirement. It quadruples it, because t enters as an inverse square. That asymmetry is the practical core of the section: confidence is cheap, since δ enters through a logarithm, while precision is expensive.'
      },
      {
        q: 'Which statement about the CLT is correct?',
        options: ['It guarantees the error on your finite test set', 'It is asymptotic; a finite-sample guarantee needs a concentration inequality', 'It applies to any distribution including infinite-variance ones', 'It requires the data itself to be Gaussian'],
        answer: 1,
        why: 'The CLT describes a limit and says nothing rigorous about any particular n, and how fast you approach that limit depends on the shape of what you are averaging — the heavy-tailed option in the lab is still visibly skewed at n = 200. Hoeffding and Bernstein hold at every n with no approximation, which is what licenses a test-set number. The last option is the most common misconception and gets the theorem backwards: the CLT is remarkable precisely because it does not care what the individual draws look like, only that they are independent with finite variance.'
      },
      {
        q: 'You evaluated 100 model configurations on one 5,000-row validation set. What should you conclude about the winner?',
        options: ['its score is an unbiased estimate of its true quality', 'its score is biased upward, and the union bound says you needed about 10,400 rows for a ±2-point guarantee across all 100', 'nothing at all can be said', 'you need 100 times more rows'],
        answer: 1,
        why: 'The maximum of 100 noisy scores is systematically above the truth, because the winning configuration is partly the luckiest one. The union bound repairs the guarantee by requiring n ≥ ln(2m/δ)/(2t²), which at m = 100 is about 10,368 rows against 4,612 for a single model. The "100 times more rows" answer is the intuitive one and is far too pessimistic — m enters through a logarithm, so a hundredfold increase in candidates costs only about 2.25× the data. That logarithm is why searching large hypothesis spaces is possible at all, and it is the finite ancestor of VC theory.'
      }
    ],
    cards: [
      { q: 'Hoeffding’s inequality', a: '$P(|\\bar X_n-\\mu|\\ge t)\\le 2\\exp(-2nt^2/(b-a)^2)$ — exponential in n, needs only independence and boundedness.' },
      { q: 'Rows needed for ±2 points at 95%', a: '≈ 4,600; for ±1 point ≈ 18,000 — error falls like $1/\\sqrt n$, so precision costs quadratically.' },
      { q: 'Why does averaging work?', a: 'Independent deviations combine as variances, so the sum grows like $\\sqrt n$ and the average shrinks like $1/\\sqrt n$.' },
      { q: 'Comparing m models on one holdout', a: 'Union bound: $n \\ge \\ln(2m/\\delta)/(2t^2)$. Cost grows only logarithmically in m.' },
      { q: 'Clustered test rows', a: 'Effective n is $n/(1+(m-1)\\rho)$. Ten rows per user at $\\rho=0.5$ divides your sample size by 5.5.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.5 */
  ML.section({
    id: 'mle-map', track: 'foundations', num: '1.5',
    title: 'MLE, MAP, and the bridge to loss and regularisation',
    lede: 'The single most reused idea on this site: it turns every model in Parts 2 and 4 into an optimisation problem you can write down from scratch.',
    rests: 'Consumed by: every loss in Part 2 · the softmax objective in §4.9 · the KL penalties in §4.12.',
    html: `
<h2><span class="sn">1.5.1</span> Maximum likelihood</h2>
<p>Maximum likelihood maximises $\\prod_i p(y_i \\mid x_i; \\theta)$. Products of small numbers underflow and do not differentiate pleasantly, so take a logarithm and flip the sign: <mark>the negative log-likelihood <i>is</i> your loss.</mark> Everything else is bookkeeping about which noise model you assumed.</p>

${H.table(['Assume', 'NLL becomes', 'Known as'], [
      ['Gaussian noise', '$\\frac{1}{2\\sigma^2}\\sum_i (y_i-\\hat y_i)^2 + c$', 'squared error / MSE'],
      ['Bernoulli label', '$-\\sum_i [y_i\\log p_i + (1-y_i)\\log(1-p_i)]$', 'binary cross-entropy / logloss'],
      ['Categorical label', '$-\\sum_i \\log p_{i,y_i}$', 'softmax cross-entropy (§4.9)'],
      ['Poisson counts', '$\\sum_i (\\hat\\lambda_i - y_i\\log\\hat\\lambda_i)$', 'Poisson deviance'],
      ['Laplace noise', '$\\sum_i |y_i-\\hat y_i|$', 'MAE — why MAE is robust to outliers']
    ])}
<p>Note the Gaussian case: $\\sigma^2$ enters only as a multiplicative scale, which is why nobody estimates it before fitting.</p>

<h2><span class="sn">1.5.2</span> MAP adds a prior</h2>
$$\\arg\\max_\\theta\\; \\log p(\\text{data}\\mid\\theta) + \\log p(\\theta)$$
<p>A zero-mean Gaussian prior $\\mathcal{N}(0,\\tau^2)$ contributes $-\\frac{1}{2\\tau^2}\\|\\theta\\|_2^2$ — an <b>L2 penalty with $\\lambda = 1/(2\\tau^2)$</b>. A Laplace prior contributes $-\\frac{1}{b}\\|\\theta\\|_1$ — <b>L1</b>. So:</p>
${H.key('Regularisation is a prior, and the regularisation strength is the prior’s inverse variance.')}
<p>Let $\\tau^2 \\to \\infty$ (a flat prior) and MAP degenerates to MLE — the cleanest one-line answer to "when are MAP and MLE the same?"</p>

${H.lab('assum', 'Assumption → loss → penalty', 'Pick a noise model and a prior; the objective assembles itself. Compare squared error against absolute error at a large residual — that ratio is exactly why one is robust and the other is not.')}

${H.lab('shrink', 'A prior is a spring: watch coefficients shrink', 'Real ridge and lasso fits on correlated data, recomputed as you move λ. Ridge shrinks smoothly and never quite arrives at zero; lasso pins coefficients to exactly zero one at a time.')}

${H.probe([
      ['Where does cross-entropy come from?', 'It is the negative log-likelihood of a Bernoulli (or Categorical) label — not a heuristic.'],
      ['Why does L2 shrink weights?', 'It is a zero-mean Gaussian prior; the log-prior is a quadratic pull toward zero.'],
      ['When is MAP the same as MLE?', 'As the prior variance → ∞, i.e. a flat prior.']
    ])}`,
    labs: {
      assum: function (host) {
        const objBox = ML.el('div', { class: 'box', style: 'margin:14px 0 0' });
        const st = Viz.controls(host, [
          { k: 'noise', label: 'noise model (likelihood)', type: 'select', value: 'gauss', options: [
            { v: 'gauss', t: 'Gaussian → squared error' }, { v: 'lap', t: 'Laplace → absolute error' },
            { v: 'huber', t: 'Huber (robust hybrid)' }, { v: 'pois', t: 'Poisson → deviance' }] },
          { k: 'prior', label: 'prior on weights', type: 'buttons', value: 'none', options: [{ v: 'none', t: 'flat → MLE' }, { v: 'gauss', t: 'Gaussian → L2' }, { v: 'lap', t: 'Laplace → L1' }] },
          { k: 'tau', label: 'prior sd τ (smaller = stronger)', min: .1, max: 4, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => { S.redraw(); obj(); });
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const L = {
              gauss: r => .5 * r * r,
              lap: r => Math.abs(r),
              huber: r => Math.abs(r) < 1 ? .5 * r * r : Math.abs(r) - .5,
              pois: r => Math.exp(r) - r - 1
            };
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [0, 5] })
              .frame({ xlabel: 'residual  (ŷ − y)', ylabel: 'loss contributed' });
            P.clip(() => {
              P.fn(L.gauss, { color: T.faint, width: 1.4, dash: [4, 4] });
              P.fn(L[st.noise], { color: T.blue, width: 2.8 });
              if (st.prior !== 'none') {
                const lam = st.prior === 'gauss' ? 1 / (2 * st.tau * st.tau) : 1 / st.tau;
                P.fn(t => st.prior === 'gauss' ? lam * t * t : lam * Math.abs(t), { color: T.red, width: 2, dash: [6, 3] });
              }
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText('dashed grey = squared error, for comparison', w - 16, 8);
            ctx.fillStyle = T.red; ctx.fillText(st.prior === 'none' ? '' : 'dashed red = the penalty term as a function of a weight', w - 16, 24);
          }
        });
        host.appendChild(objBox);
        function obj() {
          const nll = {
            gauss: '\\tfrac{1}{2\\sigma^2}\\sum_i (y_i-\\hat y_i)^2',
            lap: '\\tfrac{1}{b}\\sum_i |y_i-\\hat y_i|',
            huber: '\\sum_i \\rho_\\delta(y_i-\\hat y_i)',
            pois: '\\sum_i (\\hat\\lambda_i - y_i\\log\\hat\\lambda_i)'
          }[st.noise];
          const pen = st.prior === 'none' ? '' : (st.prior === 'gauss'
            ? ' \\;+\\; \\lambda\\|\\theta\\|_2^2,\\quad \\lambda=\\tfrac{1}{2\\tau^2}=' + (1 / (2 * st.tau * st.tau)).toFixed(3)
            : ' \\;+\\; \\lambda\\|\\theta\\|_1,\\quad \\lambda=\\tfrac{1}{b}=' + (1 / st.tau).toFixed(3));
          objBox.innerHTML = '<p class="boxtitle">the objective you are minimising</p>$$' + nll + pen + '$$' +
            '<p class="small" style="margin:0">' + (st.prior === 'none'
              ? 'Flat prior — this is plain maximum likelihood.'
              : (st.prior === 'gauss'
                ? 'Gaussian prior with τ = ' + st.tau.toFixed(2) + ' → ridge. As τ → ∞, λ → 0 and MAP becomes MLE.'
                : 'Laplace prior with scale ' + st.tau.toFixed(2) + ' → lasso; the constant subgradient is what pins coefficients at exactly zero (§2.3).')) + '</p>';
          ML.typeset(objBox);
        }
        obj();
      },

      shrink: function (host) {
        const R = Num.rng(19);
        const n = 90, p = 6;
        const Xr = [], y = [];
        const trueW = [1.8, -1.2, 0.9, 0.0, 0.0, 0.0];
        for (let i = 0; i < n; i++) {
          const base = R.normal(0, 1);
          const x = [base, base * .85 + R.normal(0, .5), R.normal(0, 1), R.normal(0, 1), base * .6 + R.normal(0, .8), R.normal(0, 1)];
          Xr.push(x); y.push(Num.dot(x, trueW) + R.normal(0, 1.1));
        }
        const st = Viz.controls(host, [
          { k: 'kind', label: 'penalty', type: 'buttons', value: 'l2', options: [{ v: 'l2', t: 'ridge (L2)' }, { v: 'l1', t: 'lasso (L1)' }] },
          { k: 'logl', label: 'λ', min: -4, max: 3, step: .05, value: -2, fmt: v => Math.pow(10, v).toFixed(3) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nz', label: 'non-zero coefs', cls: 'key' }, { k: 'norm', label: '‖w‖' }, { k: 'mse', label: 'train MSE' }
        ]);
        function fit(lam, l1) {
          if (!l1) return Num.ridgeFit(Xr, y, lam);
          let w = new Array(p).fill(0);
          for (let it = 0; it < 160; it++) {
            for (let j = 0; j < p; j++) {
              let rho = 0, zz = 0;
              for (let i = 0; i < n; i++) {
                let pred = 0; for (let k = 0; k < p; k++) if (k !== j) pred += Xr[i][k] * w[k];
                rho += Xr[i][j] * (y[i] - pred); zz += Xr[i][j] * Xr[i][j];
              }
              w[j] = Math.sign(rho) * Math.max(0, Math.abs(rho) - lam * n / 2) / (zz || 1);
            }
          }
          return w;
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w2, h, T) {
            const lams = [];
            for (let e = -4; e <= 3.01; e += .12) lams.push(Math.pow(10, e));
            const paths = lams.map(l => fit(l, st.kind === 'l1'));
            const P = Viz.plot(ctx, w2, h, { xd: [-4, 3], yd: [-2.2, 2.4] })
              .frame({ xlabel: 'log₁₀ λ', ylabel: 'coefficient value', xfmt: v => v.toFixed(0) });
            const cols = Labs.palette(p);
            P.clip(() => {
              for (let j = 0; j < p; j++) P.line(lams.map((l, i) => [Math.log10(l), paths[i][j]]), { color: cols[j], width: 2 });
              P.hline(0, { color: T.faint, dash: [2, 3] });
              P.vline(st.logl, { color: T.text, dash: [4, 4] });
            });
            const wNow = fit(Math.pow(10, st.logl), st.kind === 'l1');
            wNow.forEach((v, j) => P.dots([[st.logl, v]], { r: 4.6, color: cols[j], stroke: true }));
            const pred = Xr.map(x => Num.dot(x, wNow));
            out({
              nz: wNow.filter(v => Math.abs(v) > 1e-4).length + ' / ' + p,
              norm: Math.sqrt(Num.dot(wNow, wNow)).toFixed(2),
              mse: Num.mean(y.map((v, i) => (v - pred[i]) ** 2)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'True coefficients are [1.8, −1.2, 0.9, 0, 0, 0], with features 1, 2 and 5 correlated. Ridge splits weight between the correlated columns and keeps everything; lasso selects — and <i>which</i> of a correlated group it picks is unstable, which is exactly what elastic net fixes.');
      }
    },
    quiz: [
      {
        q: 'A colleague says "we chose MSE because it is standard". The more precise justification is…',
        options: ['it is fastest to compute', 'it is the negative log-likelihood under Gaussian noise', 'it is robust to outliers', 'it is the only differentiable loss'],
        answer: 1,
        why: 'Squared error is the Gaussian NLL up to a constant and the 1/2σ² scale. Heavy-tailed noise implies Laplace (MAE) or Huber instead.'
      },
      {
        q: 'Ridge with penalty λ‖w‖² corresponds to which prior?',
        options: ['Uniform on [−1,1]', 'Zero-mean Gaussian with variance 1/(2λ)', 'Laplace with scale 1/λ', 'Beta(1,1)'],
        answer: 1,
        why: 'The log of a zero-mean Gaussian prior is −‖w‖²/(2τ²); matching gives λ = 1/(2τ²). A strong penalty is a confident prior that weights are near zero.'
      },
      {
        q: 'Residuals contain several genuine extreme outliers. Taking that seriously implies which loss?',
        options: ['Squared error', 'Absolute error (Laplace) or Huber', 'Cross-entropy', 'Poisson deviance'],
        answer: 1,
        why: 'A heavy-tailed noise model yields a loss growing linearly rather than quadratically, so one point cannot dominate the fit.'
      }
    ],
    cards: [
      { q: 'The MLE→loss bridge', a: 'The NLL of your assumed noise model IS your loss: Gaussian→MSE, Bernoulli→cross-entropy, Poisson→deviance, Laplace→MAE.' },
      { q: 'The MAP→regulariser bridge', a: 'Gaussian prior → L2 with λ=1/(2τ²); Laplace prior → L1. Strength is the prior’s inverse variance.' },
      { q: 'When does MAP equal MLE?', a: 'When the prior variance → ∞ (flat prior).' }
    ]
  });

})();
