/* ============================================================
   PART 1 — Foundations (1.6 – 1.11)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 1.6 */
  ML.section({
    id: 'intervals', track: 'foundations', num: '1.6',
    title: 'Confidence vs credible intervals; testing; A/B; peeking',
    lede: 'Feeds evaluation in §2.13 and every experiment you will be asked to design on the spot. The peeking simulation below is the fastest way to stop believing a running p-value.',
    rests: 'Rests on §1.4 (why a sample says anything) and §1.2 (the Beta prior that makes Thompson sampling one line).',
    html: `
<p>Your experiment finished this morning. Arm A, the current checkout page, converted 4.00 per cent of the 40,000 users who saw it. Arm B, the new one, converted 4.40 per cent of its 40,000. That is 1,600 conversions against 1,760, a relative improvement of 10 per cent, and the person who designed arm B is standing at your desk asking a perfectly reasonable question: <i>how sure are we?</i></p>

<p>You already know how to produce a number. The estimated difference is 0.40 percentage points. Its standard error, using the machinery of §1.4, is about 0.14 points, so the difference is nearly three standard errors from zero, and a statistics package will hand back a 95 per cent interval of roughly $[0.13,\\; 0.67]$ points and a p-value of about 0.005. You could read those out loud and everyone would nod.</p>

<p>Now try to say what they <i>mean</i>. Almost every sentence that comes naturally is false. "There is a 95 per cent chance the true lift is between 0.13 and 0.67 points" is false. "There is a 0.5 per cent chance the new page is no better" is false. "We are 99.5 per cent confident B wins" is false. Each of them takes a statement about the behaviour of a procedure and quietly converts it into a statement about the world, and the conversion is not licensed by anything you computed.</p>

<p>This section is about that gap. It is worth taking slowly, because the gap is where experiments get mis-called — not by people who cannot do the arithmetic, but by people who did the arithmetic correctly and then said the wrong English sentence about it.</p>

<h2><span class="sn">1.6.1</span> What a confidence interval actually promises</h2>

<p>Start with what is random and what is not. There is a true conversion rate for arm B. Call it $\\theta$, the Greek letter theta, and read $\\theta$ as "the parameter" — the fixed, unknown number that the world is using. You do not know it and you never will. What you have instead is a sample: 40,000 users who happened to arrive, each of whom happened to convert or not. Run the experiment again next week with 40,000 different users and you will get a different count, a different estimate, and a different interval.</p>

<p>So in the frequentist picture, <b>$\\theta$ is fixed and the interval is random</b>. That single sentence dissolves most of the confusion. Asking "what is the probability that $\\theta$ lies in $[0.13, 0.67]$?" is asking for the probability of a statement about two fixed numbers and a third fixed number. It is either in or it is not. The probability is 1 or 0, and you do not know which.</p>

<p>What the 95 per cent refers to is the <b>procedure</b>. A confidence interval is a recipe: take the data, compute the estimate, compute the standard error, and go out $\\pm 1.96$ standard errors. That recipe has a property you can state without knowing $\\theta$ at all:</p>

${H.key('If you ran this recipe on fresh data over and over, 95 per cent of the intervals it produced would contain the true value. The guarantee attaches to the recipe, not to any one interval it produced.')}

${H.analogy(`<p>Picture a game of horseshoes. The peg is hammered into the ground and never moves — that is $\\theta$. You throw a hoop with your eyes closed — that is your experiment. The hoop is what varies from throw to throw.</p>
<p>"This hoop is a 95 per cent hoop" means: over many throws, 95 per cent of them ring the peg. It is a statement about your throwing, established before you threw. Once a particular hoop is lying on the ground it either rings the peg or it does not, and no amount of staring at that hoop turns the 95 per cent into a property of it.</p>
<p>The Bayesian question — "given where this hoop landed, how likely is the peg to be inside it?" — is a genuinely different question. It is answerable, but only if you were willing to say something in advance about where pegs tend to be. That extra willingness is called a prior, and it is the subject of the next subsection.</p>`)}

${H.worked('the interval for arm B, computed by hand', `<p>Arm B converted $1{,}760$ of $40{,}000$ users, so $\\hat p = 1760/40000 = 0.044$. The hat on $\\hat p$ means "estimated from data", to keep it visually distinct from the unknown truth $p$.</p>
<p>For a proportion, the standard error is $\\sqrt{\\hat p(1-\\hat p)/n}$. Here that is $\\sqrt{0.044 \\times 0.956 / 40000} = \\sqrt{1.0516 \\times 10^{-6}} = 0.001026$, or about 0.103 percentage points.</p>
<p>A 95 per cent interval reaches out $1.96$ standard errors either side: $0.044 \\pm 1.96 \\times 0.001026 = 0.044 \\pm 0.00201$, giving $[0.0420,\\; 0.0460]$, or 4.20 per cent to 4.60 per cent.</p>
<p>Where does 1.96 come from? It is the point on a standard normal curve with 2.5 per cent of the area beyond it in each tail, so 95 per cent sits between $-1.96$ and $+1.96$. Change the confidence level and only that number changes: 90 per cent uses 1.645, 99 per cent uses 2.576. Higher confidence buys you a wider interval and nothing else — the data did not improve.</p>`)}

<p>Two things follow from the arithmetic that are worth reading off the formula directly. The width shrinks like $1/\\sqrt{n}$, so quadrupling your sample halves your interval — the same $\\sqrt{n}$ that governs everything in §1.4. And the width does not depend on $\\theta$ being anywhere in particular; the recipe is blind to the truth, which is exactly why its guarantee can be stated in advance.</p>

<p><b>What you are looking at.</b> Each horizontal line in the plot below is one complete simulated experiment. The simulator draws a fresh sample from a distribution whose true mean is 0, marked by the amber vertical line, computes that sample's interval, and draws it as a line segment with a small dot at the sample mean. Blue segments cover the truth; red segments miss it entirely. The readout gives the observed coverage across all the experiments shown, the raw miss count, and the mean half-width of the intervals.</p>

<p><b>What to do with it.</b> Press <i>Run again</i> a dozen times with the default settings and watch the observed coverage bounce around 95 per cent — sometimes 91.7, sometimes 98.3, because 60 experiments is itself a small sample. Then drag the confidence level down to 50 per cent and press it again: the intervals become dramatically shorter and roughly half of them turn red. Nothing is broken. A 50 per cent interval is doing exactly what it promised, which is to miss half the time.</p>

<p><b>The thing genuinely worth noticing.</b> No individual red interval is defective. Its data were drawn from the same distribution by the same code as every blue one; it was simply unlucky. That is the whole content of the frequentist guarantee, and it is why "there is a 95 per cent probability the truth is in <i>this</i> interval" is not a sentence the procedure entitles you to say. If you want that sentence, you have to buy it, and the next subsection names the price.</p>

${H.lab('coverage', 'Coverage, simulated — what "95%" actually promises', 'Each horizontal line is one experiment’s interval. Red ones miss the true value. Run enough of them and the miss rate converges to 5% — that, and nothing else, is what the number means.')}

${H.pitfall(`<p>Drag the sample size down to 5 and run the simulation repeatedly. The observed coverage does not sit at 95 per cent; it sits near 88 per cent. Over 20,000 simulated experiments the figure is 88.1 per cent, and at $n = 10$ it is 91.8 per cent, and at $n = 30$ it is 93.9 per cent. The interval is quietly too narrow, and it under-covers exactly when you have least data and most need the warning.</p>
<p>The cause is that the recipe uses $1.96$, a quantile of the normal distribution, while dividing by an <i>estimated</i> standard deviation. With few observations that estimate is itself noisy, and it is too small about as often as it is too large — but a too-small estimate produces a too-narrow interval, and the misses pile up faster than the extra-wide intervals rescue them. The fix is Student's $t$ distribution, whose quantiles are wider precisely to pay for the estimated denominator: at $n = 5$ the right multiplier is 2.776, not 1.96.</p>
<p>The general lesson outlives the specific fix. A confidence interval's promise is conditional on its assumptions holding, and "the standard error is known" is an assumption. When you see coverage that misses its nominal level, the procedure has not failed at random — some assumption in it is false, and finding out which one is the entire diagnostic exercise.</p>`)}

<h2><span class="sn">1.6.2</span> Credible intervals, and what the prior buys you</h2>

<p>Suppose you genuinely want the forbidden sentence. You want to say "given the data I have seen, there is a 95 per cent probability that the conversion rate is between these two numbers." To say that, $\\theta$ has to be the kind of thing that <i>has</i> a probability distribution — which means treating your uncertainty about $\\theta$ as a distribution, updating it with Bayes' rule (§1.1), and reading an interval off the result.</p>

<p>That is a <b>credible interval</b>: an interval containing 95 per cent of the posterior probability mass. It is a direct probability statement about the parameter, it is the thing everyone wanted all along, and it is available only because you supplied a starting distribution — the <b>prior</b> — before seeing the data.</p>

${H.worked('the same data, both ways, on a small sample', `<p>A landing page converted 8 of its first 100 visitors. Take a flat prior, $\\mathrm{Beta}(1,1)$, which is the uniform distribution on $[0,1]$ and says "before seeing anything, every conversion rate is equally plausible". The conjugate update of §1.2 gives the posterior $\\mathrm{Beta}(1+8,\\; 1+92) = \\mathrm{Beta}(9, 93)$.</p>
<p><b>Credible interval.</b> The 2.5th and 97.5th percentiles of $\\mathrm{Beta}(9,93)$ are 4.16 per cent and 15.01 per cent. So: given this prior and this data, there is a 95 per cent probability that the true rate lies in $[4.16\\%,\\; 15.01\\%]$. You may also ask questions the frequentist interval cannot answer at all — for instance, the posterior probability that the rate exceeds 5 per cent is 93.4 per cent, a number you can put straight into a decision.</p>
<p><b>Confidence interval.</b> The textbook normal-approximation interval is $0.08 \\pm 1.96\\sqrt{0.08 \\times 0.92/100} = 0.08 \\pm 0.0532$, giving $[2.68\\%,\\; 13.32\\%]$.</p>
<p>The two intervals are different — the credible one is shifted right and asymmetric, because the Beta posterior knows the rate cannot be negative and the normal approximation does not. With a few hundred more observations they would agree to two decimal places. <mark>Agreement of the numbers never implies agreement of the statements.</mark></p>`)}

<p>Now watch what the prior actually does. Suppose that instead of professing ignorance you had a year of history saying pages like this convert at about 4 per cent, and you encode that as $\\mathrm{Beta}(20, 480)$ — a distribution centred on 4 per cent and carrying about as much weight as 500 previous observations. The same 8-in-100 result now updates it to $\\mathrm{Beta}(28, 572)$, with posterior mean 4.67 per cent and a 95 per cent credible interval of $[3.13\\%,\\; 6.49\\%]$.</p>

<p>Same data, wildly different conclusion: 8.8 per cent under the flat prior, 4.7 per cent under the informative one. That is not a bug and it is not a scandal. It is the honest accounting of the fact that 100 observations is thin evidence, and what you believed beforehand should still be doing most of the work. But it does mean the credible interval's directness comes with an obligation: <b>you must be able to defend the prior</b>, because someone else's prior gives a different interval from the same data.</p>

${H.intuition(`<p>The cleanest way to hold the two objects apart is to ask what is being quantified over.</p>
<p>A <b>confidence</b> interval quantifies over <i>repetitions</i>: across many hypothetical datasets, this recipe covers the truth 95 per cent of the time. The randomness lives in the data. No prior is required, and the guarantee holds whoever you are.</p>
<p>A <b>credible</b> interval quantifies over <i>parameter values</i>: given this one dataset and this prior, 95 per cent of the posterior mass sits here. The randomness lives in your uncertainty about $\\theta$. A prior is required, and the answer is yours rather than everyone's.</p>
<p>Neither is more correct. They answer different questions, and the reason the distinction matters in practice is that stakeholders always want the Bayesian sentence and analysts usually hand them the frequentist number. If you are going to say the Bayesian sentence, do the Bayesian computation.</p>`)}

${H.history(`<p>The awkwardness is historical, not mathematical. Bayes' rule dates from the 1760s and was the ordinary way of reasoning about uncertainty for well over a century — Laplace used it to estimate the mass of Saturn and reported his answer as a direct probability.</p>
<p>The frequentist apparatus was built in the 1920s and 1930s, largely by Fisher, Neyman and Pearson, out of a specific dissatisfaction: for scientific work, an answer that depends on the investigator's prior opinion looked like an answer that could be manipulated. Neyman's confidence interval was engineered to be exactly the object that gives a guarantee without a prior — and the price of that engineering is that its guarantee is about the procedure rather than about the parameter. The awkward English is not sloppiness. It is the shape of the thing that was deliberately built.</p>
<p>Modern practice mixes freely. A/B testing platforms increasingly report posteriors and "probability B beats A" because that is what decisions need, while regulated settings still demand pre-registered frequentist tests because their guarantee does not depend on anyone's beliefs.</p>`)}

<h2><span class="sn">1.6.3</span> Testing, and what a p-value is not</h2>

<p>A test asks a narrower question than an interval: not "how large is the effect?" but "is the data embarrassing for the hypothesis that there is no effect at all?"</p>

<p>Set up the pieces in the order they are actually used. The <b>null hypothesis</b> $H_0$ is the boring world: the two arms have identical conversion rates and every difference you see is sampling noise. You fix a <b>type-I error rate</b> $\\alpha$ — the proportion of boring worlds in which you are willing to be fooled into declaring an effect — and you fix it <i>before</i> looking at the data, because after looking there is no such thing as a rate. Then you compute a <b>test statistic</b>, a single number summarising how far the data has strayed from what $H_0$ predicts, and finally the <b>p-value</b>:</p>

$$p = P(\\text{a test statistic at least this extreme} \\mid H_0 \\text{ is true})$$

<p>Read the vertical bar as "given". So the p-value is a conditional probability whose condition is the null hypothesis. It says: <i>if nothing were happening, data this extreme or more would arise this often.</i> Small $p$ means the data would be surprising in the boring world. That is all it means.</p>

${H.deriv('the two-proportion z statistic, and why the standard error is pooled', [
      ['$\\hat p_A = c_A/n, \\quad \\hat p_B = c_B/n$', 'Start with the two observed conversion rates: counts divided by users. These are the only quantities the data supplies.'],
      ['$\\hat\\delta = \\hat p_B - \\hat p_A$', 'The estimated difference — the quantity the experiment exists to measure. Its true counterpart $\\delta$ is zero under the null.'],
      ['$\\mathrm{Var}(\\hat\\delta) = \\mathrm{Var}(\\hat p_B) + \\mathrm{Var}(\\hat p_A)$', 'Variances of independent quantities add, and a difference contributes the same variance as a sum because $\\mathrm{Var}(-X) = \\mathrm{Var}(X)$. Independence holds because the two arms are separate users — §1.3 is where this rule comes from and where it fails.'],
      ['$= \\frac{p_B(1-p_B)}{n} + \\frac{p_A(1-p_A)}{n}$', 'Each arm is a mean of $n$ Bernoulli draws, and a Bernoulli variable with success probability $p$ has variance $p(1-p)$, so its mean over $n$ draws has variance $p(1-p)/n$.'],
      ['$\\approx \\frac{2\\bar p(1-\\bar p)}{n}, \\quad \\bar p = \\frac{c_A+c_B}{2n}$', 'Under the null the two rates are <i>the same number</i>, so estimating them separately throws away information. Pool both arms into one estimate $\\bar p$ and the two terms collapse into one, doubled. This is why the formula uses a pooled rate: the null is what you are computing the probability under, so you are entitled to assume it here.'],
      ['$z = \\dfrac{\\hat\\delta}{\\sqrt{2\\bar p(1-\\bar p)/n}}$', 'Divide the estimate by its standard error to get a unitless score: how many standard errors from zero the observed difference sits. The central limit theorem (§1.4) says $z$ is approximately standard normal under the null when $n$ is large.'],
      ['$p\\text{-value} = 2\\,\\big(1 - \\Phi(|z|)\\big)$', 'Convert the score into a tail probability. $\\Phi$ is the standard normal cumulative distribution function, so $1 - \\Phi(|z|)$ is the mass beyond $|z|$ in one tail; the factor of 2 counts the other tail, because a difference in either direction would have been interesting. Dropping the 2 gives a one-sided test, which is a different pre-registered decision, not a free halving of your p-value.']
    ], 'This is precisely the computation the peeking lab performs at every look. Being able to reconstruct it from Bernoulli variances is a common whiteboard exercise, and the pooled standard error is the step candidates most often cannot justify.')}

<p>Now the three sentences that must never be said, each of which is a different way of flipping the conditional bar around. The p-value is <b>not</b> the probability that the null is true. It is not the probability you are wrong. And $1 - p$ is not the probability the effect is real. Getting from $P(\\text{data} \\mid H_0)$ to $P(H_0 \\mid \\text{data})$ requires Bayes' rule and therefore a prior on $H_0$ — precisely the base-rate step of §1.1, and skipping it here produces exactly the same error as reading a 99 per cent accurate medical test as a 99 per cent chance of disease.</p>

<p>One more piece of vocabulary, because it is the other half of every test. <b>Power</b> is $1 - \\beta$, the probability of detecting an effect that genuinely exists. Convention sets it at 80 per cent, which means a real effect of the size you designed for still gets missed one run in five. A test with 30 per cent power that returns "not significant" has told you almost nothing, and the word for it is not "negative result" but "underpowered".</p>

<h2><span class="sn">1.6.4</span> Peeking, the most expensive habit in experimentation</h2>

<p>Here is how it actually happens. The test is scheduled for sixteen days. On day three someone opens the dashboard, because the dashboard exists and it updates hourly. The p-value reads 0.31. On day six it reads 0.12. On day nine it dips to 0.04 and there is a small celebration in the channel; the feature is shipped that afternoon, the test is stopped, and the remaining seven days are never run.</p>

<p>Nothing in that story involves dishonesty. Every individual p-value was computed correctly. And yet the false-positive rate of the procedure that was actually followed is nowhere near 5 per cent.</p>

<p>To see why, watch what the running p-value does over the life of a null test. Early on, when only a few hundred users have arrived, the estimate is wildly noisy and the p-value can land anywhere. As users accumulate the estimate settles down and the p-value drifts toward large values, because there is nothing to find. But the path from one to the other is a <b>random walk</b>, and it wanders. Every time you look, you give that wandering path another opportunity to be below 0.05 at the moment you happen to be looking.</p>

${H.analogy(`<p>You have a coin that you claim is fair, and I get to test it. Under the rule "flip it 100 times and call it biased if heads exceed 58", I will wrongly accuse you about 5 per cent of the time. That is a controlled error rate.</p>
<p>Now change the rule to: flip it as long as you like, and accuse the moment the running proportion of heads gets far enough from a half. That rule accuses you eventually, with probability one, for a perfectly fair coin. A fair random walk crosses any fixed boundary sooner or later; the mathematical statement is the law of the iterated logarithm, and the practical statement is that a stopping rule which only ever looks for evidence in one direction will always eventually find it.</p>
<p>Peeking at a running experiment is the second rule wearing business clothes. The dashboard is the unlimited flipping, and "ship it when significant" is the accusation.</p>`)}

<p>The size of the effect is not subtle. Simulating the null A/A test in the lab below — both arms converting at 10 per cent, 2,000 users per arm, several thousand repetitions — gives these realised false-positive rates:</p>

${H.table(['Looks taken', 'Declared a winner (true rate: 5%)'], [
      ['1 (fixed horizon, no peeking)', '4.8%'],
      ['5', '13.9%'],
      ['10', '19.3%'],
      ['20', '24.3%'],
      ['50', '33.5%'],
      ['100', '36.5%']
    ])}

<p>Twenty looks — a fortnight's test glanced at twice a day — turns a 5 per cent error rate into roughly 25 per cent. One declared win in four is then noise. And note the shape of the growth: it is steep at first and then flattens, because consecutive looks are highly correlated. The look on day nine sees almost the same data as the look on day eight, so it is not really an independent extra chance. That correlation is the only reason the number is 25 per cent rather than $1 - 0.95^{20} = 64$ per cent.</p>

<p><b>What you are looking at.</b> The vertical axis is the running p-value and the horizontal axis is users observed per arm, so each faint line is the entire history of one simulated experiment's p-value as its data accumulated. The amber horizontal line marks $\\alpha = 0.05$. Lines drawn in red are runs that dipped below that line at some look and would therefore have been stopped and declared a winner; the last run is highlighted in blue, with a red dot marking the moment it would have been stopped. The three readouts are the fixed-horizon false-win rate, which reads only the final look, the peeking false-win rate, which counts any run that ever crossed, and the number of simulations accumulated so far.</p>

<p><b>What to do with it.</b> Leave the true lift at "none", so both arms are genuinely identical and every declared win is false by construction. Press <i>Run 1 experiment</i> five or six times and just watch the blue path wander — you are looking at pure noise, and some of those paths dive under the amber line and come back up. Then press <i>Run 200</i> two or three times and read the two rate columns. The fixed-horizon column should settle near 5 per cent; the peeking column will be four or five times higher. Now drag the number of peeks from 20 down to 1 and back up to 100; the simulations reset each time a control moves, so run 200 again at each setting and watch only the second column climb.</p>

<p><b>The thing genuinely worth noticing.</b> The two columns are computed from <i>the same simulated experiments</i>. No extra data, no different arms, no change to the statistics — the only difference is the rule for when you are allowed to stop. That is the whole lesson: a false-positive rate is a property of a decision procedure, not of a dataset, and looking early changes the procedure even when it does not change a single number in the data. Raise the true lift above zero and the columns stop meaning "false wins" at all: with a real effect present they become power, so the labels then read wrong and the peeking column being higher is a genuine advantage, which is precisely why sequential methods exist rather than a blanket ban on early stopping.</p>

${H.lab('peek', 'Peeking, simulated on a null A/A test', 'Both arms are identical — every "win" here is false. Run the simulation and compare the fixed-horizon error rate against the stop-when-significant rate. The counter is the number that ends arguments.')}

<p>The fix is not discipline. Telling analysts not to look at a dashboard that exists is a plan that fails on contact with a launch deadline. The fix is method: <mark>pre-register the sample size, or use a procedure that is valid under continuous monitoring.</mark> There are three that you should be able to name.</p>

${H.table(['Approach', 'How it stays valid', 'What it costs'], [
      ['<b>Alpha spending</b> (O’Brien–Fleming, Pocock)', 'Budget the total $\\alpha$ across a pre-declared schedule of looks, spending very little early and the remainder at the end', 'You must fix the number and timing of looks in advance; early stopping requires a much smaller p-value than 0.05'],
      ['<b>Always-valid p-values / confidence sequences</b>', 'Build an interval that is simultaneously valid at <i>every</i> sample size, so no stopping rule can invalidate it', 'The intervals are wider than fixed-horizon ones at every point — you pay for the freedom to stop whenever you like'],
      ['<b>Bayesian posterior with a decision rule</b>', 'The posterior itself does not depend on why you stopped, so the belief statement stays coherent under any stopping rule', 'The <i>decision rule</i> layered on top still has a frequentist error rate, and it is not automatically 5%']
    ])}

${H.flag(`It is often claimed that Bayesian A/B testing "solves" peeking outright. The precise statement is narrower and worth getting right. The posterior distribution given the data does not depend on the stopping rule, so the belief you hold is unaffected by when you chose to stop. But the moment you convert that belief into a shipping decision — "ship when $P(B > A) > 0.95$" — you have created a procedure with a type-I error rate, and monitoring it continuously inflates that rate too, though generally less than with naive p-values and by an amount that depends heavily on the prior. If you need a bounded false-positive rate, a prior is not a substitute for a sequential design.`)}

${H.worked('worked number — how long must this test run?', `
<p>For two proportions at $\\alpha=0.05$ two-sided and 80% power, sample size per arm is well approximated by</p>
$$n \\approx \\frac{16\\,p(1-p)}{\\delta^2}$$
<p>where $p$ is the baseline conversion rate and $\\delta$ is the <i>absolute</i> difference you want to be able to detect. The 16 is a rounding of $2(z_{0.975}+z_{0.80})^2 = 2(1.96+0.84)^2 \\approx 15.7$: the first quantile controls how often you cry wolf and the second controls how often you miss a real wolf, and both enter as squares because they are widths measured in standard errors and variance is what adds.</p>
<p>Take a baseline conversion of 4% and a target of detecting a 10% relative lift. Relative lifts have to be converted to absolute ones before they enter the formula: $\\delta = 0.04 \\times 0.10 = 0.004$, four thousandths, not 10.</p>
$$n \\approx \\frac{16 \\times 0.04 \\times 0.96}{0.004^2} = \\frac{0.6144}{0.000016} \\approx 38{,}400 \\text{ per arm}$$
<p>Nearly 77,000 users across both arms. At 5,000 eligible users a day that is <b>16 days</b> — and you commit to those 16 days before you start, because that commitment is the thing that makes the 5% real. (Using the exact constant 15.7 rather than 16 gives 37,675, which rounds to the same fortnight; the approximation is not the weak link in this estimate.)</p>
<p>The lesson worth carrying: $n$ scales with $1/\\delta^2$, so <mark>halving the effect you want to detect quadruples the test</mark>. Most "inconclusive" experiments were never powered to conclude anything.</p>`)}

<p><b>What you are looking at.</b> The curve is that formula plotted against the relative lift you want to detect, with sample size per arm on the vertical axis. The dashed vertical line marks your current target lift and the dot sits where it meets the curve. When the CUPED correlation slider is above zero a second, lower curve appears in green showing the same requirement after variance reduction. The readout converts the curve into the numbers a planning meeting needs: users per arm, total users, days to run at your traffic, and days saved.</p>

<p><b>What to do with it.</b> Set the baseline to 4 per cent and the relative lift to 10 per cent and confirm you get the 38,400 computed above. Now halve the lift to 5 per cent and watch the required $n$ go to exactly four times as much, 153,600 per arm — you can check the arithmetic against the readout rather than trusting the shape of the curve. Then leave the lift alone and drag the baseline conversion instead, noting that rarer events are harder: at a 1 per cent baseline the same relative lift needs far more traffic, because $\\delta$ shrank along with $p$.</p>

<p><b>The thing genuinely worth noticing.</b> The curve is not steep, it is <i>vertical</i> at the left-hand end. This is the single most useful fact for the conversation where a product manager asks whether you could "just detect a 2 per cent lift instead of 10 per cent". That is a twenty-five-fold increase in traffic, which at 5,000 users a day is not a longer experiment but a different year. Being able to say that in the meeting, with the arithmetic, is worth more than any modelling you will do that week.</p>

${H.lab('power', 'Sample size and power calculator', 'The same formula, live. Watch the quadratic blow-up as you shrink the effect you want to detect, and watch CUPED move the line for free.')}

<h2><span class="sn">1.6.5</span> Multiple testing is peeking’s sibling</h2>

<p>Peeking multiplies your chances across time. Testing many metrics multiplies them across space, and the arithmetic is the same arithmetic. If twenty metrics are all null and all independent, the probability that at least one crosses $\\alpha = 0.05$ is $1 - 0.95^{20} = 0.64$. A dashboard with twenty guardrail metrics will therefore show you a "significant" movement about two times in three, every time, forever.</p>

<p>Two corrections are worth knowing, and they control genuinely different things.</p>

<p><b>Bonferroni</b> divides $\\alpha$ by the number of tests: with 20 metrics, each must clear $0.0025$ instead of $0.05$. It controls the <b>family-wise error rate</b> — the probability of even one false positive anywhere in the family — and it is conservative because it assumes the worst about how the tests are correlated. For a handful of pre-registered metrics that conservatism costs little and buys a claim you can defend.</p>

<p><b>Benjamini–Hochberg</b> controls the <b>false discovery rate</b> instead: not "no false positives at all" but "at most 10 per cent of the things I flag are false". Sort the p-values, find the largest $k$ with $p_{(k)} \\le k\\alpha/m$, and reject everything up to it. When you are screening thousands of candidate features and expect a real subset among them, this is the right instrument, because Bonferroni at that scale rejects essentially nothing.</p>

${H.practice(`<p>The organisational fix matters more than the statistical one. Declare <b>one primary metric</b> in the experiment document, before the experiment starts, along with the effect size you intend to detect and the date you will stop. Everything else is either a guardrail — a metric that can veto a launch by getting worse, tested one-sided and generously — or exploratory, which means it may generate a hypothesis for the next experiment and may not be used to justify this one.</p>
<p>The failure mode this prevents has a shape you will recognise once you have seen it. A test comes back flat on the primary metric. Someone slices by browser, by country, by new-versus-returning, and reports that the effect is strongly positive for returning users on Android in Germany. There are hundreds of such slices, so several of them are significant by construction. The correct response is not to argue about the subgroup; it is to run a new experiment powered for that subgroup, which converts a fishing expedition into a hypothesis.</p>`)}

<h2><span class="sn">1.6.6</span> CUPED — buying power back for free</h2>

<p>Sample size is set by variance, and by this point in the section you have seen how brutally $n \\propto 1/\\delta^2$ punishes you. So anything that reduces the variance of the outcome reduces the test length proportionally, and there is one trick that does it with no cost in bias and almost no engineering.</p>

<p>The idea rests on an observation about your users: much of the variation in how much someone spends this month is explained by how much they spent last month, and last month happened <i>before</i> the experiment started, so it cannot possibly have been affected by the treatment. That pre-period quantity is therefore free information about which users were always going to be heavy or light, and subtracting the part of the outcome it predicts removes noise without removing signal.</p>

<p>The adjusted outcome is</p>

$$Y_{\\text{adj}} = Y - \\theta\\,(X - \\mathbb{E}[X])$$

<p>where $Y$ is the experiment outcome, $X$ is the pre-period covariate, $\\mathbb{E}[X]$ is read "the expectation of X" and means its average across all users in the experiment, and $\\theta$ is a coefficient you get to choose. Subtracting the centred covariate leaves the mean untouched — that is what centring guarantees — so the estimator stays unbiased for any $\\theta$ whatsoever. The only question is which $\\theta$ shrinks the variance most.</p>

${H.deriv('the optimal $\\theta$, and the variance it delivers', [
      ['$\\mathrm{Var}(Y_{\\text{adj}}) = \\mathrm{Var}(Y) - 2\\theta\\,\\mathrm{Cov}(Y,X) + \\theta^2\\mathrm{Var}(X)$', 'Expand the variance of a difference. The cross term appears because $Y$ and $X$ are correlated; if they were independent the middle term would vanish and the adjustment could only ever hurt. Centring $X$ does not affect any variance or covariance, so it drops out of this line entirely.'],
      ['$\\dfrac{d}{d\\theta} = -2\\,\\mathrm{Cov}(Y,X) + 2\\theta\\,\\mathrm{Var}(X) = 0$', 'The expression is a quadratic in $\\theta$ with a positive leading coefficient, so it is a parabola opening upwards and its unique minimum is where the derivative vanishes (§0.3).'],
      ['$\\theta^\\star = \\dfrac{\\mathrm{Cov}(Y,X)}{\\mathrm{Var}(X)}$', 'Solve. This is exactly the slope of the least-squares regression of $Y$ on $X$ (§2.4), which is the useful way to remember it: CUPED regresses out the pre-period covariate.'],
      ['$\\mathrm{Var}(Y_{\\text{adj}}) = \\mathrm{Var}(Y) - \\dfrac{\\mathrm{Cov}(Y,X)^2}{\\mathrm{Var}(X)}$', 'Substitute $\\theta^\\star$ back in: the second and third terms combine into a single subtraction. The subtracted quantity is a square divided by a variance, hence non-negative — the adjustment can never increase the variance at the optimal $\\theta$.'],
      ['$= \\mathrm{Var}(Y)\\,(1-\\rho^2)$', 'Use the definition of correlation, $\\rho = \\mathrm{Cov}(Y,X)/\\sqrt{\\mathrm{Var}(Y)\\mathrm{Var}(X)}$, and the subtracted term becomes $\\rho^2\\mathrm{Var}(Y)$. Read $\\rho$ as the Greek letter rho.']
    ], 'The final line is the whole result: variance falls by the factor $(1-\\rho^2)$, and since required sample size is proportional to variance, so does the length of your experiment. It is also exactly the quantity called $R^2$ in a simple regression, which is not a coincidence — you are removing the part of $Y$ that $X$ explains.')}

<p>Put numbers on it. At $\\rho = 0.6$, variance falls by $1 - 0.36 = 0.64$, a 36 per cent reduction, and the sixteen-day test above becomes $16 \\times 0.64 = 10.24$, so ten or eleven days. At $\\rho = 0.7$ the reduction is 49 per cent and the test roughly halves. Pre-period behaviour usually does predict post-period behaviour, so correlations in this range are common rather than optimistic, which is why this is close to free power.</p>

<p>Two conditions have to hold, and both are easy to check. The covariate must be measured strictly <i>before</i> randomisation — using anything measured during the experiment risks conditioning on something the treatment affected, which is the mediator trap of §1.7. And it must be defined identically in both arms, which it will be, since it predates the split.</p>

<h2><span class="sn">1.6.7</span> Bandits, when a fixed horizon is the wrong instrument</h2>

<p>Everything so far has assumed you want a clean estimate at the end. That assumption has a price, and it is worth stating plainly: an A/B test deliberately sends half its traffic to the losing arm for the entire run, including the last day, by which point it is fairly sure which arm is losing. If the arms are a good headline and a bad one, the cost is a fortnight of lost clicks that you knowingly chose to lose.</p>

<p>A <b>multi-armed bandit</b> makes the opposite trade. It shifts traffic toward whatever is currently winning while it is still learning, so it loses less reward — at the cost of a biased, less precise estimate of each arm's true value, because the arms it stopped showing have small and non-random samples.</p>

${H.table(['Algorithm', 'Rule', 'Character'], [
      ['ε-greedy', 'explore at fixed rate ε, else exploit', 'trivial; never stops wasting ε'],
      ['UCB', 'play $\\arg\\max_a\\ \\hat\\mu_a + \\sqrt{2\\ln t / n_a}$', 'uncertainty itself is a reason to try something'],
      ['Thompson sampling', 'draw a parameter from each arm’s posterior, play the argmax', 'one line with a Beta prior; usually the best of the three in practice']
    ])}

<p>Look at the UCB rule for a moment, because it encodes an idea worth stealing. The score of an arm is its estimated mean $\\hat\\mu_a$ plus a bonus $\\sqrt{2\\ln t/n_a}$ that grows when the arm has been played rarely ($n_a$ small) and shrinks as the total round count $t$ grows only logarithmically. So an arm gets played either because it looks good or because you are unsure about it, and the second reason decays exactly as fast as the evidence accumulates. <b>Optimism in the face of uncertainty</b> is the name of the principle, and it is the whole algorithm.</p>

<p>The formal accounting object is <b>regret</b>: the total reward you lost by not having played the best arm every round from the start. A fixed even split accumulates regret linearly, because it pays the full gap on every impression it sends to a loser, forever. Good bandit algorithms achieve regret growing like $\\log T$, which means the per-round cost of learning tends to zero.</p>

<p><b>What you are looking at.</b> Each curve is cumulative regret against rounds for one policy on the same simulated set of arms: green for Thompson sampling, blue for UCB, amber for ε-greedy, red for a fixed even A/B split. Lower is better, and the slope at any point is how much reward that policy is currently giving away per round. The controls set the number of arms, the gap in conversion rate between the best arm and the rest, the number of rounds, and ε.</p>

<p><b>What to do with it.</b> Start at the defaults and read the shapes rather than the numbers: the red line is straight, and the other three bend. Watch UCB in particular — it is often the worst line early, because it insists on trying everything before it will commit, and then it flattens. Then drag the gap down to its smallest setting and look again.</p>

<p><b>The thing genuinely worth noticing.</b> With a tiny gap between the arms, every algorithm looks mediocre and the curves bunch together. That is not a failure of the algorithms; it is the regime where there is very little reward to win, because the arms are nearly equally good, and simultaneously a great deal of data needed to tell them apart. This is exactly the case where an A/B test's clean unbiased estimate is worth more than the reward a bandit could have saved you — you are not trying to earn clicks, you are trying to learn a number you will defend in a decision meeting. That is the honest answer to "should we use a bandit or an A/B test?", and it is a better answer than a preference.</p>

${H.lab('bandit', 'Three bandits racing', 'Regret curves, computed live. Watch UCB’s early exploration cost and Thompson’s quiet dominance; then set the arms nearly equal and see every algorithm struggle — that is the case where an A/B test’s unbiased estimate is worth more than the reward you gave up.')}

<p><b>Contextual bandits</b> condition the choice on features of the current user, which is the honest description of most personalisation systems. Note what that implies for your data. The logs are now produced by a policy that chose actions based on context, so the observed outcomes are confounded by that policy; evaluating a <i>new</i> policy on those logs by naive averaging will be wrong. The correction is inverse-propensity weighting, which is the subject of the next section, and the fact that a bandit hands you a causal-inference problem is a good reason to read it.</p>

${H.more('why the winner’s effect size is inflated, and by how much', `<p>Suppose you stop a test the first moment it crosses significance. The effect size you report at that moment is not an unbiased estimate of the truth — it is systematically too large, and the mechanism is simple. You stopped <i>because</i> the estimate was extreme. Among all the paths a null-or-small effect could take, you selected one that happened to be running high at that instant.</p>
<p>This is the same phenomenon as regression to the mean (§1.7), and it has a practical signature: features shipped on the strength of an optimally-stopped test routinely fail to reproduce their measured lift in the following quarter's numbers, and the shortfall is blamed on novelty effects or seasonality when the honest explanation is the stopping rule. Sequential designs address this too — O'Brien–Fleming boundaries are deliberately severe early precisely so that an early stop requires evidence strong enough that the inflation is small — and some frameworks report a bias-corrected estimate at the stopping time.</p>
<p>The interview version of this point is short: <i>an effect size measured at an optimal stopping time is inflated by construction, and reporting it as unbiased is the error.</i></p>`)}

${H.probe([
      ['Why can’t I stop as soon as it is significant?', 'Repeated correlated looks multiply the chance of a spurious crossing; you need a sequential correction that budgets α across looks.'],
      ['Confidence vs credible, one line each?', 'Coverage of the procedure vs probability of the parameter given a prior.'],
      ['How would you halve the runtime of an underpowered test?', 'CUPED on a pre-period covariate, or accept detecting a larger effect — n scales with 1/δ².'],
      ['A stakeholder asks "what is the probability B is better?" — what do you say?', 'That the frequentist output does not answer it, and then either compute the posterior probability under a stated prior, or give the confidence interval with the correct interpretation. Do not relabel 1 − p as that probability.']
    ], 'Reporting the observed effect size at an optimal stopping time as unbiased. It is inflated by construction.')}`,
    labs: {
      coverage: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'sample size per experiment', min: 5, max: 200, step: 5, value: 30, fmt: v => v },
          { k: 'conf', label: 'confidence level', min: .5, max: .99, step: .01, value: .95, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'reps', label: 'experiments shown', min: 20, max: 200, step: 10, value: 60, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'cov', label: 'observed coverage', cls: 'key' }, { k: 'miss', label: 'intervals missing' }, { k: 'w', label: 'mean half-width' }
        ]);
        let seed = 5;
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(seed);
            const mu = 0, sig = 1, z = Num.normPpf(1 - (1 - st.conf) / 2);
            const rows = [];
            for (let r = 0; r < st.reps; r++) {
              const xs = []; for (let i = 0; i < st.n; i++) xs.push(R.normal(mu, sig));
              const m = Num.mean(xs), se = Num.sd(xs, true) / Math.sqrt(st.n);
              rows.push({ m: m, lo: m - z * se, hi: m + z * se, hw: z * se });
            }
            const span = Math.max(1.2, Math.max.apply(null, rows.map(r => Math.abs(r.m) + r.hw)) * 1.05);
            const P = Viz.plot(ctx, w, h, { xd: [-span, span], yd: [0, st.reps], pad: { l: 30, r: 14, t: 12, b: 32 } })
              .frame({ yticks: [], xlabel: 'estimate of the mean (true value = 0)' });
            let miss = 0;
            rows.forEach((r, i) => {
              const hit = r.lo <= mu && r.hi >= mu;
              if (!hit) miss++;
              P.line([[r.lo, i + .5], [r.hi, i + .5]], { color: hit ? T.blue : T.red, width: 2, alpha: hit ? .55 : 1 });
              P.dots([[r.m, i + .5]], { r: 1.9, color: hit ? T.blue : T.red });
            });
            P.vline(0, { color: T.amber, dash: false, width: 1.6, label: 'truth' });
            out({
              cov: (100 * (1 - miss / rows.length)).toFixed(1) + '%',
              miss: miss + ' of ' + rows.length,
              w: Num.mean(rows.map(r => r.hw)).toFixed(3)
            });
          }
        });
        Viz.buttons(host, [{ label: 'Run again', primary: true, on: () => { seed = Math.floor(Math.random() * 1e6); S.redraw(); } }]);
        Viz.note(host, 'Nothing about a single red interval is "wrong" — the guarantee was always about the long-run frequency of the procedure, which is exactly why the phrase "95% probability the true value is in <i>this</i> interval" is a Bayesian statement requiring a prior.');
      },

      peek: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'users per arm (final)', min: 200, max: 8000, step: 200, value: 2000, fmt: v => v.toLocaleString() },
          { k: 'looks', label: 'number of peeks', min: 1, max: 100, step: 1, value: 20, fmt: v => v },
          { k: 'p', label: 'true conversion (both arms)', min: .02, max: .5, step: .01, value: .1, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'lift', label: 'true lift in B', min: 0, max: .3, step: .005, value: 0, fmt: v => v === 0 ? 'none (A/A)' : '+' + (v * 100).toFixed(1) + '%' }
        ], () => { runs = []; S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'fixed', label: 'fixed-horizon false wins', cls: 'key' },
          { k: 'peeked', label: 'peeking false wins', cls: 'bad' },
          { k: 'runs', label: 'simulations' }
        ]);
        let runs = [], trace = null;
        function simulate(nSim) {
          const R = Num.rng(Math.floor(Math.random() * 1e6));
          for (let s = 0; s < nSim; s++) {
            let ca = 0, cb = 0, stopped = false, stopAt = 0;
            const pathP = [];
            const step = Math.max(1, Math.floor(st.n / st.looks));
            for (let i = 1; i <= st.n; i++) {
              ca += R() < st.p ? 1 : 0;
              cb += R() < st.p * (1 + st.lift) ? 1 : 0;
              if (i % step === 0 || i === st.n) {
                const pa = ca / i, pb = cb / i, pp = (ca + cb) / (2 * i);
                const se = Math.sqrt(2 * pp * (1 - pp) / i) || 1e-9;
                const zz = (pb - pa) / se;
                const pv = 2 * (1 - Num.normCdf(Math.abs(zz)));
                pathP.push([i, Math.max(1e-4, pv)]);
                if (!stopped && pv < .05) { stopped = true; stopAt = i; }
              }
            }
            runs.push({ fixed: pathP[pathP.length - 1][1] < .05, peeked: stopped, stopAt: stopAt, path: pathP });
            if (s === nSim - 1) trace = runs[runs.length - 1];
          }
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [0, st.n], yd: [0, .6] })
              .frame({ xlabel: 'users observed per arm', ylabel: 'running p-value' });
            P.clip(() => {
              runs.slice(-40).forEach(r => {
                P.line(r.path, { color: r.peeked ? T.red : T.faint, width: 1, alpha: r.peeked ? .5 : .3 });
              });
              if (trace) {
                P.line(trace.path, { color: T.blue, width: 2.2 });
                if (trace.peeked) {
                  const pt = trace.path.find(p => p[0] >= trace.stopAt);
                  if (pt) { P.dots([pt], { r: 6, color: T.red, stroke: true }); P.text(pt[0], pt[1], '  stopped here — "significant"', { color: T.red, font: '11px ui-sans-serif' }); }
                }
              }
              P.hline(.05, { color: T.amber, label: 'α = 0.05' });
            });
            const f = runs.filter(r => r.fixed).length, pk = runs.filter(r => r.peeked).length;
            out({
              fixed: runs.length ? (100 * f / runs.length).toFixed(1) + '%' : '—',
              peeked: runs.length ? (100 * pk / runs.length).toFixed(1) + '%' : '—',
              runs: runs.length
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Run 1 experiment', primary: true, on: () => { simulate(1); S.redraw(); } },
          { label: 'Run 200', on: () => { simulate(200); S.redraw(); } },
          { label: 'Reset', on: () => { runs = []; trace = null; S.redraw(); } }
        ]);
        Viz.note(host, 'With <b>lift = none</b>, every declared win is false by construction. The fixed-horizon column should sit near 5%; the peeking column climbs with the number of looks — that gap is the entire argument for sequential methods.');
      },

      power: function (host) {
        const st = Viz.controls(host, [
          { k: 'base', label: 'baseline conversion p', min: .005, max: .5, step: .005, value: .04, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'rel', label: 'relative lift to detect', min: .01, max: .5, step: .005, value: .10, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'traffic', label: 'eligible users per day', min: 200, max: 40000, step: 200, value: 5000, fmt: v => v.toLocaleString() },
          { k: 'rho', label: 'CUPED covariate correlation ρ', min: 0, max: .9, step: .01, value: 0, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'n', label: 'n per arm', cls: 'key' }, { k: 'tot', label: 'total users' },
          { k: 'days', label: 'days to run' }, { k: 'saved', label: 'days saved by CUPED', cls: 'good' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const p = st.base;
            const nFor = (rel, rho) => 16 * p * (1 - p) * (1 - rho * rho) / Math.pow(p * rel, 2);
            const P = Viz.plot(ctx, w, h, { xd: [.01, .5], yd: [0, Math.min(4e5, nFor(.02, 0) * 1.05)] })
              .frame({ xlabel: 'relative lift you want to detect', ylabel: 'n per arm', yfmt: v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0), xfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.fn(r => nFor(r, 0), { color: T.blue, width: 2.6, n: 260 });
              if (st.rho > 0) P.fn(r => nFor(r, st.rho), { color: T.green, width: 2.4, n: 260 });
              P.vline(st.rel, { color: T.text, dash: [4, 4] });
              P.dots([[st.rel, nFor(st.rel, st.rho)]], { r: 5.5, color: st.rho > 0 ? T.green : T.blue, stroke: true });
            });
            const n = nFor(st.rel, st.rho), n0 = nFor(st.rel, 0);
            out({
              n: Math.ceil(n).toLocaleString(), tot: Math.ceil(2 * n).toLocaleString(),
              days: Math.ceil(2 * n / st.traffic), saved: st.rho > 0 ? Math.max(0, Math.ceil(2 * n0 / st.traffic) - Math.ceil(2 * n / st.traffic)) : 0
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'plain difference-in-proportions' }, { c: Viz.theme().green, t: 'with CUPED variance reduction (1−ρ²)' }]);
      },

      bandit: function (host) {
        const st = Viz.controls(host, [
          { k: 'k', label: 'arms', min: 2, max: 8, step: 1, value: 4, fmt: v => v },
          { k: 'gap', label: 'gap between best and rest', min: .005, max: .2, step: .005, value: .05, fmt: v => (v * 100).toFixed(1) + ' pts' },
          { k: 'T', label: 'rounds', min: 500, max: 20000, step: 500, value: 5000, fmt: v => v.toLocaleString() },
          { k: 'eps', label: 'ε for ε-greedy', min: .01, max: .5, step: .01, value: .1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'eg', label: 'ε-greedy regret' }, { k: 'ucb', label: 'UCB regret' },
          { k: 'ts', label: 'Thompson regret', cls: 'good' }, { k: 'ab', label: 'A/B test regret', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(77);
            const k = st.k, best = .3 + st.gap;
            const mu = Array.from({ length: k }, (_, i) => i === 0 ? best : .3 - R() * .02);
            function run(policy) {
              const n = new Array(k).fill(0), s = new Array(k).fill(0);
              let regret = 0; const path = [];
              for (let t = 1; t <= st.T; t++) {
                let a;
                if (policy === 'eg') a = R() < st.eps ? R.int(k) : argmax(n.map((c, i) => c ? s[i] / c : 1e9));
                else if (policy === 'ucb') a = argmax(n.map((c, i) => c ? s[i] / c + Math.sqrt(2 * Math.log(t) / c) : 1e9));
                else if (policy === 'ts') a = argmax(n.map((c, i) => R.beta(1 + s[i], 1 + c - s[i])));
                else a = t % k;  // A/B: equal split throughout
                const r = R() < mu[a] ? 1 : 0;
                n[a]++; s[a] += r;
                regret += best - mu[a];
                if (t % Math.ceil(st.T / 180) === 0) path.push([t, regret]);
              }
              return { path: path, regret: regret };
            }
            function argmax(a) { let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; }
            const res = { eg: run('eg'), ucb: run('ucb'), ts: run('ts'), ab: run('ab') };
            const maxR = Math.max(res.ab.regret, res.eg.regret) * 1.05;
            const P = Viz.plot(ctx, w, h, { xd: [0, st.T], yd: [0, maxR] })
              .frame({ xlabel: 'rounds', ylabel: 'cumulative regret (reward lost)' });
            P.clip(() => {
              P.line(res.ab.path, { color: T.red, width: 2 });
              P.line(res.eg.path, { color: T.amber, width: 2 });
              P.line(res.ucb.path, { color: T.blue, width: 2 });
              P.line(res.ts.path, { color: T.green, width: 2.6 });
            });
            out({
              eg: res.eg.regret.toFixed(1), ucb: res.ucb.regret.toFixed(1),
              ts: res.ts.regret.toFixed(1), ab: res.ab.regret.toFixed(1)
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().green, t: 'Thompson sampling' }, { c: Viz.theme().blue, t: 'UCB' },
          { c: Viz.theme().amber, t: 'ε-greedy' }, { c: Viz.theme().red, t: 'fixed A/B split (regret grows linearly)' }
        ]);
        Viz.note(host, 'The A/B line is straight because an even split keeps paying the full gap on every losing impression for the whole run. That straight line is the <i>price of an unbiased estimate</i> — sometimes worth paying, which is the actual answer to "bandit or A/B?"');
      }
    },
    quiz: [
      {
        q: 'You monitor an A/B test continuously and stop the moment p < 0.05. Your realised false-positive rate is closest to…',
        options: ['5%', '10%', '20–30% with a handful of looks, and worse with continuous monitoring', 'exactly 50%'],
        answer: 2,
        why: 'Every look is another opportunity for the random walk of the test statistic to be below the boundary at the moment you happen to be watching, and simulating the lab’s null A/A test gives about 24% with twenty looks. The tempting answer is 5%, because each individual p-value was computed correctly — and it was; what changed is the decision procedure wrapped around it, and a false-positive rate is a property of the procedure rather than of any single calculation. Note that the inflation is not $1-0.95^{20}=64\\%$ either, because consecutive looks see almost the same data and are therefore heavily correlated. With unlimited looks on a genuinely null effect the crossing probability tends to 1. Alpha-spending, always-valid p-values or a pre-registered horizon restore validity; §1.6.4 works through all three.'
      },
      {
        q: 'Baseline 4%, you want to detect a 5% relative lift instead of 10%. The required sample size…',
        options: ['halves', 'stays the same', 'doubles', 'quadruples'],
        answer: 3,
        why: 'Sample size goes as $1/\\delta^2$, so halving the effect you want to detect multiplies the requirement by four: 38,400 per arm becomes 153,600. "Doubles" is the tempting answer because halving something feels like it should have a proportional cost, and the quadratic comes from variance rather than from the effect — the standard error shrinks only like $1/\\sqrt{n}$, so you need four times the data to halve it. The practical consequence is the one worth carrying into a planning meeting: a request to detect a fifth of the original effect is a twenty-five-fold increase in traffic, which is usually a different year rather than a longer experiment. §1.6.4 has the arithmetic.'
      },
      {
        q: 'CUPED with a pre-period covariate correlated ρ = 0.7 with the outcome reduces variance by…',
        options: ['70%', '49%', '30%', '7%'],
        answer: 1,
        why: 'The variance of the CUPED-adjusted outcome is $(1-\\rho^2)$ times the original, and $1-0.49=0.51$, so about half the variance remains and the test needs roughly half the samples for the same power. The tempting answer is 70%, reading the correlation straight off as though it were the reduction — but the derivation in §1.6.6 subtracts $\\mathrm{Cov}(Y,X)^2/\\mathrm{Var}(X)$, and squares appear because variances are what add. This is also exactly the $R^2$ of a simple regression of the outcome on the covariate, which is a useful way to estimate the gain before you implement anything: fit the regression on last quarter\'s data and read off $R^2$.'
      }
    ],
    cards: [
      { q: 'Confidence vs credible', a: 'Coverage of the procedure over repeated experiments vs probability of the parameter given data and a prior.' },
      { q: 'A/B sample size formula', a: '$n \\approx 16p(1-p)/\\delta^2$ per arm at α=0.05, 80% power. Halving δ quadruples n.' },
      { q: 'CUPED', a: '$Y_{adj}=Y-\\theta(X-\\mathbb{E}X)$ with a pre-period covariate; variance falls by $(1-\\rho^2)$, unbiased.' },
      { q: 'Bandit regret', a: 'Cumulative reward lost vs always playing the best arm; good algorithms grow like $\\log T$, an even A/B split grows like $T$.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.7 */
  ML.section({
    id: 'causal', track: 'foundations', num: '1.7',
    title: 'The causal inference a data scientist actually uses',
    lede: 'Every causal question is the same question: what would have happened to these same units under the other treatment? The counterfactual is missing by construction, so all technique is about reconstructing it credibly.',
    rests: 'Directly relevant to credit-policy decisions and to attributing a drift event to a cause (§2.18 lifecycle).',
    html: `
<p>A bank sends a monthly "financial health check" email to its card customers. A year later someone in analytics pulls the numbers. Customers who opened the email defaulted at 3.2 per cent. Customers who never opened it defaulted at 9.2 per cent. The email appears to cut defaults by six percentage points, which on this portfolio would be worth several million a year, and there is now a slide deck proposing to make the email impossible to unsubscribe from.</p>

<p>You can probably feel that something is wrong, and the useful exercise is to say precisely what. It is not the sample size — there are ten thousand customers and the difference is enormous. It is not a coding error. It is that <b>the people who open a financial health check email are not the same kind of people as those who do not</b>, and they were already going to default less, email or no email.</p>

<p>Here is the same data with that spelled out. Suppose 8,000 of the customers are financially comfortable and default at 2 per cent, while 2,000 are struggling and default at 20 per cent. Comfortable customers open the email 70 per cent of the time; struggling customers, who have rather more urgent post to deal with, open it 20 per cent of the time. And suppose — this is the crucial part — <b>the email does absolutely nothing</b>.</p>

${H.code(`openers      = 0.70 x 8000 comfortable  +  0.20 x 2000 struggling  = 5600 + 400 = 6000
  defaults   = 5600 x 0.02  +  400 x 0.20  =  112 + 80  = 192   ->  3.2%

non-openers  = 0.30 x 8000 comfortable  +  0.80 x 2000 struggling  = 2400 + 1600 = 4000
  defaults   = 2400 x 0.02  +  1600 x 0.20  =  48 + 320  = 368   ->  9.2%

true effect of the email on anyone at all: zero`)}

<p>Every number in the six-point gap came from the mix, not from the email. Check any single row and the email changes nothing: comfortable customers default at 2 per cent whether they opened it or not, struggling customers at 20 per cent either way. The gap exists because opening the email is a <i>symptom</i> of being the sort of customer who was never going to default.</p>

<p>This is the entire subject in one example. Something — financial comfort — causes both the treatment and the outcome, and it therefore contaminates any comparison you make between treated and untreated groups. The rest of this section is about naming that structure precisely, recognising the cases where it is not present, and knowing which of the standard repairs applies when you cannot simply randomise.</p>

<h2><span class="sn">1.7.1</span> The counterfactual, and why it is missing</h2>

<p>Start with what you would need in order to be certain. For one particular customer, you would want to know two things: what happens if they receive the email, and what happens if they do not. Write those <b>potential outcomes</b> as $Y(1)$ and $Y(0)$ — read $Y(1)$ as "the outcome under treatment" and $Y(0)$ as "the outcome under control". The causal effect for that customer is the difference $Y(1) - Y(0)$.</p>

<p>And now the problem that defines the field: you can only ever see one of them. The customer either got the email or did not. The other number is not merely unmeasured, it is not <i>in the world</i> to be measured. This is called the <b>fundamental problem of causal inference</b>, and it means every causal estimate you will ever produce is, underneath, a reconstruction of something that did not happen.</p>

<p>Since individual effects are unrecoverable, the standard target is an average. The <b>average treatment effect</b> is</p>

$$\\mathrm{ATE} = \\mathbb{E}[Y(1) - Y(0)]$$

<p>where $\\mathbb{E}$ is the expectation symbol of §1.3 and means "average over the whole population". Read the expression aloud as: the average outcome in a world where everyone is treated, minus the average outcome in a world where nobody is. Both worlds are hypothetical, but their difference is estimable, because averages do not require seeing both outcomes for the same person — they only require that the treated group you observe is a fair stand-in for what the control group would have done, and vice versa.</p>

<p>Two relatives of the ATE are worth naming, because interviewers ask and because they answer different business questions. The <b>ATT</b>, average treatment effect on the treated, is $\\mathbb{E}[Y(1)-Y(0) \\mid T=1]$ — the effect among those who actually got the treatment, which is what you want when asking "was this campaign worth running?" The <b>CATE</b>, conditional average treatment effect, is $\\mathbb{E}[Y(1)-Y(0)\\mid X=x]$ — the effect for a particular kind of person, which is what personalisation and uplift modelling actually need.</p>

${H.intuition(`<p>The naive comparison, difference in observed means, is $\\mathbb{E}[Y \\mid T{=}1] - \\mathbb{E}[Y \\mid T{=}0]$. Write it out in potential-outcome terms and it splits into two pieces:</p>
<p>$\\underbrace{\\mathbb{E}[Y(1)\\mid T{=}1] - \\mathbb{E}[Y(0)\\mid T{=}1]}_{\\text{the ATT — what you want}} \\;+\\; \\underbrace{\\mathbb{E}[Y(0)\\mid T{=}1] - \\mathbb{E}[Y(0)\\mid T{=}0]}_{\\text{selection bias}}$</p>
<p>The second bracket compares the untreated outcomes of the treated group with the untreated outcomes of the control group. It asks: would these two groups have differed even with no treatment at all? In the email example it compares a group that would have defaulted at about 2 per cent against one that would have defaulted at about 9 per cent, and it accounts for the entire observed six-point gap.</p>
<p>Every technique in this section is a way of making that second bracket zero, or of arguing that it is. Randomisation makes it zero by construction. Everything else makes it zero <i>by assumption</i>, and the assumption is the thing you will be asked to defend.</p>`)}

<h2><span class="sn">1.7.2</span> Drawing the structure: nodes, arrows and paths</h2>

<p>To reason about which comparisons are safe, you need a way to write down what you believe causes what. The tool is a <b>directed acyclic graph</b>, or DAG: variables are nodes, an arrow $A \\to B$ means "$A$ has some direct causal influence on $B$", and "acyclic" means you cannot follow arrows in a loop and arrive back where you started, because nothing causes itself.</p>

<p>The email story is three nodes. Financial comfort $X$ points at email-opening $T$, because comfortable people open it more. Financial comfort also points at default $Y$, because comfortable people default less. And $T$ may or may not point at $Y$ — that arrow is the thing you are trying to measure.</p>

<p>Now the key move. Information flows along a path in the graph regardless of which way the arrows point. The path $T \\leftarrow X \\to Y$ carries association from $T$ to $Y$ even though no arrow runs from one to the other, because both are being wiggled by the same $X$. A path like this, which starts with an arrow pointing <i>into</i> your treatment, is called a <b>backdoor path</b>, and an open backdoor path is exactly what confounding is.</p>

<p>There are only three ways three nodes can be wired, and knowing all three is most of practical causal reasoning.</p>

${H.table(['Structure', 'Picture', 'Open or closed by default?', 'What conditioning on X does'], [
      ['<b>Chain</b> (mediator)', '$T \\to X \\to Y$', 'Open — $T$ affects $Y$ through $X$', '<b>Closes</b> it. You remove the effect that travels through $X$ and are left with the direct effect only.'],
      ['<b>Fork</b> (confounder)', '$T \\leftarrow X \\to Y$', 'Open — a backdoor path, this is confounding', '<b>Closes</b> it. This is the case where "controlling for X" is exactly right.'],
      ['<b>Collider</b>', '$T \\to X \\leftarrow Y$', 'Closed — a common effect blocks the path', '<b>Opens</b> it. Conditioning creates an association between $T$ and $Y$ that did not exist.']
    ])}

<p>The first two rows match intuition. The third does not, and it is the single most valuable idea in this section, because it is the reason the folk rule "control for everything you have" is wrong rather than merely wasteful.</p>

<h3>Why conditioning on a common effect creates association from nothing</h3>

<p>Take a lender that approves an application if <i>either</i> the applicant's income is high <i>or</i> their credit score is high. Suppose that in the population income and credit score are completely independent — knowing one tells you nothing about the other — and to keep the arithmetic trivial, suppose each is "high" or "low" with probability one half, in all four combinations equally often.</p>

${H.code(`population (equally likely)          approved?  (high income OR high score)
  high income, high score              yes
  high income, low  score              yes
  low  income, high score              yes
  low  income, low  score              NO

among the approved, three cells remain, each 1/3:
  P(high score | high income, approved) = 1/2      <- one of the two high-income cells
  P(high score | low  income, approved) = 1        <- the only surviving low-income cell`)}

<p>Inside the approved file, low income now <i>predicts</i> a high credit score with certainty, and an analyst studying that file would report a strong negative association between income and creditworthiness. There is no such association in the world. It was manufactured by looking only at approved applicants — that is, by conditioning on approval, which is a common effect of both variables.</p>

<p>The mechanism, once you see it, is obvious and impossible to unsee: if you know the application got in, and you know one of the two reasons it might have got in is absent, then the other reason must be present. This is often called <b>Berkson's paradox</b> or explaining-away, and it appears everywhere a dataset is a selected subset rather than a population.</p>

${H.analogy(`<p>Two switches control one light through an OR gate: the light is on when either switch is up. The switches are wired to nothing else and are flipped independently, so they carry no information about each other.</p>
<p>Now stand in a room where you can see the light is on. Someone tells you the left switch is down. You instantly know the right switch is up. The light being on did not change the wiring — the switches are still independent — but <i>your knowledge of the light</i> makes them dependent for you.</p>
<p>Statistical conditioning is exactly "restricting attention to cases where the light is on". This is why the harm is not hypothetical: filtering a dataset, adding a covariate to a regression, and choosing a study population are all the same operation as far as the graph is concerned.</p>`)}

<h3>And why conditioning on a mediator answers a different question</h3>

<p>Suppose the email genuinely works, and it works because opening it prompts customers to log into the app and look at their balance, which prompts them to make a payment. The graph is $T \\to X \\to Y$ with $X$ = app logins. Now put app logins into your regression as a control.</p>

<p>The coefficient on the email is no longer the effect of the email. It is the effect of the email <i>holding app usage fixed</i> — that is, the effect through any channel other than the one the email actually works by. If the entire mechanism runs through logins, that coefficient is approximately zero, and you would conclude the email does nothing while looking directly at data showing it works.</p>

<p>This is not bias in the sense of a wrong answer to the question. It is a correct answer to a question nobody asked. The distinction has names — <b>total effect</b> is what you get without conditioning, <b>direct effect</b> is what you get with — and the discipline is to decide which one your decision requires <i>before</i> choosing your controls. If you are deciding whether to send the email, you want the total effect. If you are deciding whether to redesign the app, the direct-versus-indirect split is precisely the question.</p>

${H.key('Whether to control for a variable is decided by its causal role, not by whether it improves fit. Control for confounders; never control for colliders; control for mediators only when you deliberately want the direct effect.')}

<p><b>What you are looking at.</b> The left half is the DAG for whichever structure you have selected, with $T$, $Y$ and the third variable $X$ drawn as circles and arrows showing the assumed causal directions; the $T \\to Y$ arrow is drawn darker because it is the effect being estimated. When you turn on conditioning, the $X$ node fills in blue to show it has been included in the model. The right half is a pair of bars: the true effect that the simulator built into the data, in green, and the estimate the analysis returns, coloured red when it has drifted more than 0.12 away from the truth. Underneath the bars is a sentence describing what just happened, and the readout gives the true effect, the estimate and their difference. Behind all this the lab is generating 1,200 rows from the selected structure and either taking a raw difference in means or running a regression that includes $X$.</p>

<p><b>What to do with it.</b> Work through the four structures in order, toggling conditioning on and off in each. On <b>confounder</b> the unadjusted estimate reads about 2.55 against a true effect of 1.00, and switching conditioning on brings it to about 1.01 — adjustment repairs it. On <b>collider</b> the unadjusted estimate is essentially exact at 1.00, and conditioning drags it to about $-0.06$: you have destroyed a true effect of one by adding a variable. On <b>mediator</b> the estimate drops from about 1.67 to about 0.21 when you condition. On <b>randomised</b>, toggling conditioning changes the estimate by less than a thousandth. Then take the strength slider on the confounder and run it from 0 to 2: at zero the bias vanishes, because a confounder with no influence is not a confounder.</p>

<p><b>The thing genuinely worth noticing.</b> The confounder panel and the collider panel are the same operation producing opposite results. Adding a control variable is not a cautious, conservative act that can only reduce bias — in the collider panel it takes a perfectly good estimate and ruins it, and nothing in the data would tell you that had happened. Two datasets can be numerically indistinguishable and require opposite analyses, and the only thing that distinguishes them is the causal story, which lives outside the data. That is why the first artefact of a causal analysis is a diagram, not a regression.</p>

${H.lab('dag', 'Four small DAGs, and the bias each one creates', 'Simulated data with a known true effect. Toggle what you condition on and watch the estimate move away from the truth — including the case that surprises people: conditioning on a collider <i>creates</i> bias where there was none.')}

${H.practice(`<p>Two honest notes about reading that lab. First, the "bias" readout on the <b>mediator</b> panel is a slight misnomer: when you condition, the estimator is not making an error, it is correctly estimating the <i>direct</i> effect while the green bar still shows the total effect. The number is a difference between two estimands, not a mistake.</p>
<p>Second, on the <b>randomised</b> panel the estimate barely moves when you condition — the two values agree to four decimal places — and that can read as "adjustment does nothing here". What adjustment buys under randomisation is not a shift in the point estimate but a reduction in its standard error, which this lab does not display. That variance reduction is real and it is exactly CUPED from §1.6 wearing different clothes. Adjusting for pre-treatment covariates in a randomised experiment is good practice; adjusting for post-treatment ones is the mediator trap.</p>`)}

<h2><span class="sn">1.7.3</span> Simpson’s paradox — confounding with a memorable name</h2>

${H.worked('two underwriting policies, and an aggregate that lies', `
<p>Two underwriting policies, default outcomes by applicant segment:</p>
${H.code(`Prime      A: 81/900 default = 9.0%    B:   8/100 = 8.0%   -> B better
Subprime   A: 20/100 default = 20%      B: 171/900 = 19%    -> B better
----------------------------------------------------------------------
Overall    A: 101/1000 = 10.1%          B: 179/1000 = 17.9%  -> A better (!)`)}
<p>Check the arithmetic: policy A saw 900 prime applicants at a 9 per cent default rate, giving 81 defaults, plus 100 subprime at 20 per cent, giving 20, for 101 in total. Policy B saw the mirror image — 100 prime and 900 subprime — and its 8 and 171 defaults sum to 179.</p>
<p>B wins in <i>every</i> segment and loses overall, by a wide margin, because B was applied mostly to subprime applicants and subprime applicants default more whatever policy you use. The segment causes both the policy assignment and the outcome, which is the fork structure exactly, so the aggregate comparison is not a causal one. <mark>The reversal is not a paradox, it is confounding with a memorable name.</mark></p>
<p>And note the trap in the obvious lesson. "Always disaggregate" is not the fix; disaggregating by a collider or a mediator makes things worse, as the previous subsection showed. The fix is a diagram that tells you which variables to condition on.</p>`)}

<p><b>What you are looking at.</b> Six bars, in three groups separated by dashed lines. The left group is the prime segment's default rate under each policy, the middle group is the subprime segment's, and the right group is the aggregate over both. Blue is policy A and red is policy B throughout. The within-segment rates are fixed by the simulator at 9 per cent versus 8 per cent for prime and 20 per cent versus 19 per cent for subprime, so B is better in both segments no matter what you do. Only the two aggregate bars respond to the controls, which set what fraction of each policy's applicants were subprime.</p>

<p><b>What to do with it.</b> Press <i>Reproduce the notebook's numbers</i> to set A at 10 per cent subprime and B at 90 per cent, and confirm the aggregate bars read 10.1 per cent and 17.9 per cent, matching the table above. Then drag policy B's subprime share slowly down from 90 per cent while watching the two aggregate bars. Somewhere near the point where the two mixes meet, the aggregate ordering flips and the readout's verdict changes from "A is better" to "B is better", while the four segment bars have not moved by a pixel.</p>

<p><b>The thing genuinely worth noticing.</b> Press <i>Equal mixes</i>. With both policies applied to the same blend of applicants, the aggregate agrees with the segments and the paradox is simply gone. Nothing about the policies changed; what changed is that the confounder is now balanced across the comparison. <b>That is what randomisation buys you</b> — not better data, but a guarantee that every confounder, including the ones you never thought to measure, is balanced in expectation. Seeing the paradox switch off when you balance one variable by hand is the cheapest available intuition for why randomised assignment is worth so much.</p>

${H.lab('simpson', 'Simpson’s paradox, with the mixing under your control', 'Move the share of each policy that went to subprime applicants. Watch the overall comparison flip while the within-segment comparison never moves.')}

<h2><span class="sn">1.7.4</span> When you cannot randomise: four strategies and their prices</h2>

<p>Randomisation severs the arrow into treatment, which closes every backdoor path at once, including paths through variables you never measured or imagined. That is why it is the gold standard and why "run an experiment" is the correct first answer to almost every causal question.</p>

<p>But often you cannot. The treatment already happened. Randomising would be unethical, or illegal, or would require withholding a product from customers who have been promised it. Each of the four standard alternatives buys back a piece of what randomisation gave you, and each charges an assumption in exchange. <b>The assumption is the interesting part</b> — the estimation is usually three lines of library code, and the interview is entirely about what you had to assume.</p>

${H.table(['Strategy', 'Idea', 'The assumption an interviewer will make you defend'], [
      ['<b>Propensity scores</b>', '$e(x)=P(T{=}1\\mid x)$; match or inverse-probability-weight so treated and control covariate distributions coincide', 'No unmeasured confounding (conditional ignorability) <i>plus</i> overlap. Check both, and always report covariate balance after weighting.'],
      ['<b>Difference-in-differences</b>', 'Subtract the control group’s pre-period trend from the treated group’s change', '<b>Parallel trends</b> — absent treatment both groups would have moved together. Testable in spirit by inspecting several pre-periods.'],
      ['<b>Instrumental variables</b>', 'Find $Z$ that shifts treatment but affects $Y$ only through $T$', 'Relevance + exclusion. Weak instruments give wildly unstable estimates, so report first-stage strength.'],
      ['<b>Regression discontinuity</b>', 'Compare units just above and just below a cut-off', 'No manipulation of the running variable at the threshold']
    ])}

<h3>Propensity scores, and the two things that go wrong</h3>

<p>The <b>propensity score</b> $e(x) = P(T=1 \\mid x)$ is the probability that a unit with covariates $x$ received treatment — in the email story, the probability that a customer with this profile opens the email. You estimate it with an ordinary classifier, then either match each treated unit to a control unit with a similar score, or weight every unit by the inverse of the probability it received the treatment it actually received. The effect of the weighting is to build a pseudo-population in which treatment assignment no longer depends on $x$, which is what randomisation would have given you.</p>

<p>The remarkable fact making this work is that you do not need to match on every covariate separately; matching on the single scalar $e(x)$ is enough to balance all the covariates that went into it. That is a genuine reduction of a hard high-dimensional problem to an easy one-dimensional one.</p>

<p>What goes wrong is always one of two things. <b>Ignorability</b> is the assumption that the covariates you have include every common cause; if financial stress drives both opening and default and you never measured it, weighting on the variables you did measure repairs nothing, and no diagnostic run on your data can tell you. <b>Overlap</b> is the assumption that both treatments actually occur at every covariate profile: if no customer over 70 ever opens the email, then $e(x) \\approx 0$ there, the inverse weight explodes, and a handful of atypical units end up carrying the entire estimate. Trim or report the region of common support; a small number of enormous weights is the classic signature of an estimate about to be very unstable.</p>

<p>And whatever you do, report <b>covariate balance after weighting</b> — the standardised difference in each covariate's mean between treated and control, before and after. An estimate without a balance table is an assertion.</p>

<h3>Difference-in-differences, and the assumption you can almost see</h3>

<p>A region gets a new collections policy in March; the other regions do not. You cannot compare the regions directly, because the treated region was chosen for a reason. But you can compare <i>changes</i>. Take the treated region's change from February to April, subtract the control regions' change over the same window, and any influence that is constant within a region — however unmeasured — cancels in the first subtraction, while anything that moved everyone, such as a national interest-rate change, cancels in the second.</p>

<p>What does not cancel is a difference in <i>trend</i>. If the treated region was already deteriorating faster than the others before March, the method will attribute that continued deterioration to the policy. This is <b>parallel trends</b>, and while it is fundamentally an assumption about a counterfactual and therefore untestable, it leaves fingerprints: plot several pre-treatment periods for both groups. If the lines have been moving in parallel for six months, the assumption is at least plausible. If they have been diverging, it is not, and no amount of estimation will fix it.</p>

<h3>Instrumental variables, for when confounding is unmeasured</h3>

<p>Sometimes there is no hope of measuring the confounder. An <b>instrument</b> $Z$ is a variable that shoves the treatment around for reasons that have nothing to do with the outcome, and the trick is to use only the variation in treatment that $Z$ explains — variation that is, by construction, unrelated to the confounder.</p>

<p>The canonical shape is an <b>encouragement design</b>. You cannot randomise who opens the email, but you can randomise who receives a reminder to read it. The reminder shifts opening, and — this is the assumption — it affects defaults only by shifting opening, never directly. Two conditions carry the whole method: <b>relevance</b>, meaning $Z$ genuinely moves $T$, and <b>exclusion</b>, meaning $Z$ has no path to $Y$ except through $T$. Relevance is testable and you must report it, conventionally as the first-stage F statistic, because a <i>weak</i> instrument does not merely give a noisy answer, it gives an answer that is biased toward the naive confounded one and whose confidence intervals are not trustworthy. Exclusion is untestable and is argued, not demonstrated.</p>

<p>One subtlety with a real consequence: an instrument does not generally recover the ATE. It recovers the <b>local average treatment effect</b>, the effect among the units whose behaviour the instrument actually changed — the "compliers", the people who opened the email because they got the reminder. If those people differ from the population, and they usually do, the number you have is theirs and not everyone's, and saying so is a mark of competence rather than a weakness in the analysis.</p>

<h3>Regression discontinuity, the closest thing to a free experiment</h3>

<p>Credit policies are full of hard cut-offs: approve at a score of 660, decline at 659. Nobody believes that one point of score makes a person meaningfully different, so applicants just above and just below the line are effectively a randomised pair, and comparing their outcomes estimates the effect of approval at the threshold. Lending is unusually rich in these thresholds, which makes regression discontinuity unusually useful in credit work.</p>

<p>The assumption is that units cannot precisely manipulate their position relative to the cut-off. If brokers know the threshold and can nudge a file from 659 to 661, then those who ended up just above are exactly those with a broker willing to nudge, and the two sides of the line differ systematically. The standard diagnostic is a density plot of the running variable: a bump immediately above the threshold is the signature of manipulation. As with an instrument, the estimate is local — it tells you about the effect at 660, not at 720.</p>

${H.more('the backdoor criterion, stated properly', `<p>The three-node rules generalise into a single criterion. A set of variables $S$ suffices to identify the causal effect of $T$ on $Y$ if two conditions hold: no variable in $S$ is a descendant of $T$, and $S$ blocks every backdoor path from $T$ to $Y$ — every path that begins with an arrow pointing into $T$.</p>
<p>"Blocks" means, along that path, either $S$ contains a non-collider node on it, which closes it, or the path contains a collider that is not in $S$ and none of whose descendants are in $S$, in which case the path was already closed. Both halves of the first condition earn their keep: excluding descendants of $T$ is what forbids conditioning on mediators, and the collider clause is what forbids over-controlling.</p>
<p>Given a DAG, this is a purely mechanical check that software can perform, and the whole difficulty of applied causal work therefore relocates to writing down the DAG in the first place. That step is domain knowledge, not statistics. The most useful thing this framework offers is not the algorithm; it is that it forces the assumptions into a picture other people can disagree with.</p>`)}

<h2><span class="sn">1.7.5</span> Three biases that break credit models specifically</h2>

<ul>
<li><b>Selection bias.</b> You only observe repayment for applicants you approved, so your training data is the population your <i>old</i> policy chose. This is why reject inference exists, and why a model trained on approved-only data systematically misjudges the region near the cut-off — that region is precisely where the old policy was most selective and where your labels are therefore least representative. The strongest fix is not a clever imputation but a small randomised acceptance band: approve a few per cent of applicants just below the line, at deliberate cost, and buy unbiased labels exactly where you need them.</li>
<li><b>Survivorship.</b> The same problem in time. Accounts that closed or were written off drop out of the panel, so a snapshot of currently-open accounts describes the survivors and not the cohort. Any performance metric computed on "active customers" inherits this, and the distortion always runs in the flattering direction.</li>
<li><b>Regression to the mean.</b> Pick the twenty worst-performing branches, intervene, and measure again: they will improve, and the intervention will get the credit. But an extreme measurement is extreme partly because of genuine performance and partly because of noise, and the noise part does not repeat. A control group of equally bad branches that you left alone is the only defence, and noticing that you need one is the whole skill. Both §1.6's caution about optimally-stopped effect sizes and this bias are the same statistical phenomenon: selecting on an extreme value and then being surprised when it moderates.</li>
</ul>

${H.history(`<p>The modern framework arrived late and in two pieces. Fisher formalised randomisation in the 1920s and 1930s, and it is not an exaggeration to say that the randomised experiment is one of the most consequential inventions of the century — but Fisher himself argued, into the 1950s, that the association between smoking and lung cancer might be confounded by a genetic factor causing both, and he was not obviously being unreasonable given the tools then available. The absence of a language for reasoning about non-experimental causation was a real gap, not merely a rhetorical one.</p>
<p>The language arrived in two traditions. Neyman's potential-outcome notation was extended by Rubin in the 1970s into what is now the standard framework for estimating effects, and it is the vocabulary in which propensity scores, matching and most econometrics are written. Judea Pearl's graphical models, developed through the 1980s and 1990s, gave the complementary object: a way to encode assumptions as a picture and read off mechanically which adjustments are valid. The two are formally compatible and most practitioners use both — potential outcomes to define the estimand, graphs to decide what to condition on.</p>`)}

${H.probe([
      ['IV or DiD?', 'IV when confounders are unmeasured but a valid instrument exists; DiD when you have a clean pre/post and a control group with credible parallel trends.'],
      ['What does propensity weighting assume?', 'Conditional ignorability and overlap — and you must show covariate balance after weighting, not just report the estimate.'],
      ['Your regression coefficient on the treatment changed a lot when you added a control. Is that good?', 'Unanswerable without knowing the control’s causal role. If it is a confounder, the change is the bias being removed; if it is a mediator, you have switched estimand to the direct effect; if it is a collider, you have just introduced bias. Draw the graph before interpreting the movement.'],
      ['What effect does an instrument actually estimate?', 'The local average treatment effect — the effect among compliers, the units whose treatment the instrument moved. Say so rather than calling it the ATE.']
    ], 'Controlling for a collider (a common <i>effect</i>) or a mediator. Both induce bias rather than removing it — "control for everything" is not a strategy.')}`,
    labs: {
      dag: function (host) {
        const st = Viz.controls(host, [
          { k: 'g', label: 'graph', type: 'buttons', value: 'conf', options: [
            { v: 'conf', t: 'confounder' }, { v: 'coll', t: 'collider' }, { v: 'med', t: 'mediator' }, { v: 'rand', t: 'randomised' }] },
          { k: 'adj', label: 'condition on the third variable X', type: 'toggle', value: false },
          { k: 'str', label: 'strength of X’s effect', min: 0, max: 2, step: .05, value: 1.2, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'truth', label: 'true effect of T on Y', cls: 'key' },
          { k: 'naive', label: 'estimate' }, { k: 'bias', label: 'bias', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(41), n = 1200, TRUE = 1.0;
            const rows = [];
            for (let i = 0; i < n; i++) {
              let X, Tt, Y;
              if (st.g === 'conf') { X = R.normal(0, 1); Tt = (st.str * X + R.normal(0, 1)) > 0 ? 1 : 0; Y = TRUE * Tt + st.str * X + R.normal(0, 1); }
              else if (st.g === 'coll') { Tt = R() < .5 ? 1 : 0; Y = TRUE * Tt + R.normal(0, 1); X = st.str * Tt + st.str * Y + R.normal(0, 1); }
              else if (st.g === 'med') { Tt = R() < .5 ? 1 : 0; X = st.str * Tt + R.normal(0, 1); Y = 0.4 * Tt + st.str * X + R.normal(0, 1); }
              else { Tt = R() < .5 ? 1 : 0; X = R.normal(0, 1); Y = TRUE * Tt + st.str * X + R.normal(0, 1); }
              rows.push({ X: X, T: Tt, Y: Y });
            }
            const trueEff = st.g === 'med' ? 0.4 + st.str * st.str : TRUE;
            let est;
            if (!st.adj) {
              const t1 = rows.filter(r => r.T === 1), t0 = rows.filter(r => r.T === 0);
              est = Num.mean(t1.map(r => r.Y)) - Num.mean(t0.map(r => r.Y));
            } else {
              const D = rows.map(r => [1, r.T, r.X]);
              const beta = Num.ridgeFit(D, rows.map(r => r.Y), 1e-8);
              est = beta[1];
            }
            // draw DAG
            const cx = w * .27, cy = h * .5;
            const nodes = {
              T: { x: cx - 70, y: cy + 30, l: 'T' }, Y: { x: cx + 70, y: cy + 30, l: 'Y' }, X: { x: cx, y: cy - 55, l: 'X' }
            };
            const edges = {
              conf: [['X', 'T'], ['X', 'Y'], ['T', 'Y']],
              coll: [['T', 'X'], ['Y', 'X'], ['T', 'Y']],
              med: [['T', 'X'], ['X', 'Y'], ['T', 'Y']],
              rand: [['X', 'Y'], ['T', 'Y']]
            }[st.g];
            edges.forEach(([a, b]) => {
              const A = nodes[a], B = nodes[b];
              const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy);
              const ux = dx / L, uy = dy / L;
              ctx.strokeStyle = (a === 'T' && b === 'Y') ? T.text : T.faint;
              ctx.lineWidth = (a === 'T' && b === 'Y') ? 2.2 : 1.5;
              ctx.beginPath(); ctx.moveTo(A.x + ux * 20, A.y + uy * 20); ctx.lineTo(B.x - ux * 22, B.y - uy * 22); ctx.stroke();
              const ang = Math.atan2(dy, dx);
              ctx.fillStyle = ctx.strokeStyle;
              ctx.beginPath();
              ctx.moveTo(B.x - ux * 20, B.y - uy * 20);
              ctx.lineTo(B.x - ux * 20 - 8 * Math.cos(ang - .4), B.y - uy * 20 - 8 * Math.sin(ang - .4));
              ctx.lineTo(B.x - ux * 20 - 8 * Math.cos(ang + .4), B.y - uy * 20 - 8 * Math.sin(ang + .4));
              ctx.closePath(); ctx.fill();
            });
            Object.keys(nodes).forEach(k => {
              const N = nodes[k];
              ctx.beginPath(); ctx.arc(N.x, N.y, 19, 0, 6.3);
              ctx.fillStyle = (k === 'X' && st.adj) ? T.blue : T.panel;
              ctx.fill(); ctx.strokeStyle = (k === 'X' && st.adj) ? T.blue : T.line; ctx.lineWidth = 2; ctx.stroke();
              ctx.fillStyle = (k === 'X' && st.adj) ? '#fff' : T.text;
              ctx.font = 'bold 14px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(N.l, N.x, N.y);
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(st.adj ? 'conditioning on X (blue)' : 'not conditioning on X', cx, h - 18);
            // bias bar
            const bx = w * .58, bw = w * .36;
            ctx.textAlign = 'left'; ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = T.muted;
            ctx.fillText('true effect', bx, 40); ctx.fillText('estimate', bx, 84);
            const scale = bw / Math.max(2.4, Math.abs(est) * 1.2, trueEff * 1.2);
            ctx.fillStyle = T.green; ctx.fillRect(bx, 48, Math.max(0, trueEff) * scale, 18);
            ctx.fillStyle = Math.abs(est - trueEff) > .12 ? T.red : T.green;
            ctx.fillRect(bx, 92, Math.max(0, est) * scale, 18);
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace';
            ctx.fillText(trueEff.toFixed(2), bx + Math.max(0, trueEff) * scale + 6, 58);
            ctx.fillText(est.toFixed(2), bx + Math.max(0, est) * scale + 6, 102);
            const msg = {
              conf: st.adj ? 'Correct: adjusting for the confounder removes the backdoor path.' : 'Biased: X opens a backdoor path T ← X → Y.',
              coll: st.adj ? 'Now biased: conditioning on a collider OPENS a path that was closed.' : 'Correct: the collider path is closed unless you condition on it.',
              med: st.adj ? 'This is the direct effect only — the mediated part has been removed.' : 'This is the total effect (direct + mediated).',
              rand: 'Randomisation cuts X → T, so the naive difference is already unbiased; adjusting only reduces variance.'
            }[st.g];
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left';
            wrapText(ctx, msg, bx, 132, bw, 16);
            out({ truth: trueEff.toFixed(3), naive: est.toFixed(3), bias: (est - trueEff).toFixed(3) });
          }
        });
        function wrapText(ctx, text, x, y, maxW, lh) {
          const words = text.split(' '); let line = '', yy = y;
          words.forEach(word => {
            const test = line + word + ' ';
            if (ctx.measureText(test).width > maxW && line) { ctx.fillText(line, x, yy); line = word + ' '; yy += lh; }
            else line = test;
          });
          ctx.fillText(line, x, yy);
        }
      },

      simpson: function (host) {
        const st = Viz.controls(host, [
          { k: 'mixA', label: 'share of policy A applied to subprime', min: 0, max: 1, step: .01, value: .1, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'mixB', label: 'share of policy B applied to subprime', min: 0, max: 1, step: .01, value: .9, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'n', label: 'applicants per policy', min: 200, max: 5000, step: 100, value: 1000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'a', label: 'policy A overall' }, { k: 'b', label: 'policy B overall' },
          { k: 'winner', label: 'aggregate says', cls: 'bad' }, { k: 'seg', label: 'every segment says', cls: 'good' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            // within-segment rates: B is strictly better in both
            const rate = { prime: { A: .09, B: .08 }, sub: { A: .20, B: .19 } };
            const nA_sub = st.n * st.mixA, nA_pr = st.n - nA_sub;
            const nB_sub = st.n * st.mixB, nB_pr = st.n - nB_sub;
            const defA = nA_pr * rate.prime.A + nA_sub * rate.sub.A;
            const defB = nB_pr * rate.prime.B + nB_sub * rate.sub.B;
            const oA = defA / st.n, oB = defB / st.n;
            const P = Viz.plot(ctx, w, h, { xd: [0, 4], yd: [0, .26], pad: { l: 46, r: 14, t: 16, b: 46 } })
              .frame({ xticks: [], ylabel: 'default rate', yfmt: v => (v * 100).toFixed(0) + '%' });
            const bars = [
              { x: .55, v: rate.prime.A, c: T.blue, l: 'A · prime' }, { x: 1.0, v: rate.prime.B, c: T.red, l: 'B · prime' },
              { x: 1.7, v: rate.sub.A, c: T.blue, l: 'A · sub' }, { x: 2.15, v: rate.sub.B, c: T.red, l: 'B · sub' },
              { x: 2.9, v: oA, c: T.blue, l: 'A · all' }, { x: 3.35, v: oB, c: T.red, l: 'B · all' }
            ];
            bars.forEach(b => {
              const x0 = P.x(b.x - .18), x1 = P.x(b.x + .18);
              ctx.fillStyle = b.c; ctx.fillRect(x0, P.y(b.v), x1 - x0, P.y(0) - P.y(b.v));
              ctx.fillStyle = T.text; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
              ctx.fillText((b.v * 100).toFixed(1) + '%', (x0 + x1) / 2, P.y(b.v) - 4);
              ctx.fillStyle = T.muted; ctx.textBaseline = 'top'; ctx.font = '10px ui-sans-serif';
              ctx.fillText(b.l, (x0 + x1) / 2, P.y(0) + 6);
            });
            ctx.strokeStyle = T.line; ctx.setLineDash([4, 4]);
            [1.35, 2.55].forEach(x => { ctx.beginPath(); ctx.moveTo(P.x(x), P.pad.t); ctx.lineTo(P.x(x), P.y(0)); ctx.stroke(); });
            ctx.setLineDash([]);
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('PRIME SEGMENT', P.x(.78), P.pad.t + 2);
            ctx.fillText('SUBPRIME SEGMENT', P.x(1.93), P.pad.t + 2);
            ctx.fillText('AGGREGATE', P.x(3.12), P.pad.t + 2);
            out({
              a: (oA * 100).toFixed(1) + '%', b: (oB * 100).toFixed(1) + '%',
              winner: oA < oB ? 'A is better' : 'B is better',
              seg: 'B is better'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Reproduce the notebook’s numbers', primary: true, on: () => { st.$set('mixA', .1); st.$set('mixB', .9); st.$set('n', 1000); S.redraw(); } },
          { label: 'Equal mixes (no confounding)', on: () => { st.$set('mixA', .5); st.$set('mixB', .5); S.redraw(); } }
        ]);
        Viz.note(host, 'Set both mixes equal and the paradox vanishes: with the confounder balanced, the aggregate agrees with the segments. That is exactly what randomisation buys you.');
      }
    },
    quiz: [
      {
        q: 'You have a strong pre-period, a treated group, a control group, and unmeasured confounders that are stable over time. The natural estimator is…',
        options: ['propensity score matching', 'difference-in-differences', 'instrumental variables', 'a naive difference in means'],
        answer: 1,
        why: 'Difference-in-differences subtracts each group from its own past, so anything constant within a group — however unmeasured — cancels, and then subtracts the control group\'s change, so anything that moved everyone cancels too. That is exactly the situation described: confounders that are unmeasured but stable. Propensity matching is the tempting alternative because it is the technique people learn first, but it requires you to have <i>measured</i> the confounders, which the question rules out. What does not cancel is a difference in trend, so the assumption you must defend is parallel trends, and the way you support it is by plotting several pre-treatment periods for both groups (§1.7.4).'
      },
      {
        q: 'Conditioning on a collider…',
        options: ['removes confounding bias', 'has no effect', 'creates a spurious association between its causes', 'is always required'],
        answer: 2,
        why: 'Conditioning on a common effect opens a path between its causes that was closed before, creating an association where none existed — the approval example in §1.7.2 turns two genuinely independent variables into a perfect negative association simply by looking at approved applications only. "Removes confounding bias" is tempting because adding controls feels conservative, as though more information could only help; the collider panel of the DAG lab shows it taking an exactly correct estimate of 1.00 and dragging it to about $-0.06$. Two datasets can be numerically identical and require opposite analyses, and the only thing distinguishing them is the causal story, which is why the first artefact of a causal analysis is a diagram rather than a regression.'
      },
      {
        q: 'A credit model is trained only on approved applicants. What is the problem called and what fixes it?',
        options: ['Overfitting; use regularisation', 'Selection bias; reject inference or a randomised holdout above the cut-off', 'Concept drift; retrain', 'Class imbalance; resample'],
        answer: 1,
        why: 'The labels exist only for applicants the old policy approved, so the training population is a selected subset rather than the population the model will score — and the selection was hardest exactly near the cut-off, which is where the new model most needs to be accurate. "Overfitting; use regularisation" is tempting because the symptom looks similar, a model that performs worse in production than in validation, but regularisation cannot manufacture information about applicants whose outcomes were never observed. Reject inference attempts to impute those outcomes; the stronger fix is a small randomised acceptance band just below the line, which buys genuinely unbiased labels at a deliberate and budgeted cost. §1.7.5 also connects this to survivorship and regression to the mean, which are the same selection problem in different clothes.'
      }
    ],
    cards: [
      { q: 'ATE and why it is hard', a: '$\\mathbb{E}[Y(1)-Y(0)]$; you never observe both potential outcomes for one unit, so the counterfactual must be reconstructed.' },
      { q: 'Assumptions: propensity / DiD / IV', a: 'Ignorability + overlap; parallel trends; relevance + exclusion.' },
      { q: 'Simpson’s paradox', a: 'A within-segment ordering reverses in the aggregate because segment membership confounds assignment — confounding with a memorable name.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.8 */
  ML.section({
    id: 'linear-algebra', track: 'foundations', num: '1.8',
    title: 'Linear algebra: eigen, SVD, PSD',
    lede: 'Foundation for PCA, kernel validity, attention as a bilinear form, second-order optimisation, and the low-rank claim that LoRA rests on.',
    rests: 'Consumed by §2.10 PCA · §2.6 kernels · §4.3 attention · §4.13 LoRA.',
    html: `
<p>From §0.2 you already have the picture that matters: a matrix is a machine that moves arrows. Feed it a vector, it hands back a different vector, and because the map is linear it does this by stretching, squashing, shearing, rotating or flipping the whole plane at once without tearing it or moving the origin.</p>

<p>Take one concrete machine and play with it. Let</p>

$$A = \\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \\end{bmatrix}$$

<p>and push a few arrows through it by hand. The arrow $(1,0)$ comes out as $(2,1)$ — longer, and turned upward. The arrow $(0,1)$ comes out as $(1,2)$ — longer, and turned to the right. Almost everything you feed this matrix comes back pointing somewhere else, which is exactly what "the map moves arrows about" means.</p>

<p>Now try $(1,1)$. The first component of the output is $2 \\times 1 + 1 \\times 1 = 3$ and the second is $1 \\times 1 + 2 \\times 1 = 3$, so $(1,1)$ comes out as $(3,3)$. That is the <i>same arrow</i>, three times as long. It did not turn at all. Try $(1,-1)$: the output is $(2 - 1,\\; 1 - 2) = (1,-1)$, unchanged entirely.</p>

<p>Two special directions, out of the infinitely many available, that this matrix refuses to rotate. Finding them, understanding why they exist, and learning what they tell you about the matrix is the content of this section — and it is the machinery underneath PCA, kernel methods, second-order optimisation and low-rank adaptation, all of which are later sections that will make far more sense if this one lands.</p>

<h2><span class="sn">1.8.1</span> Eigenvectors: the directions a matrix leaves alone</h2>

<p>Write the observation as an equation. A non-zero vector $v$ is an <b>eigenvector</b> of a square matrix $A$ if</p>

$$Av = \\lambda v$$

<p>for some number $\\lambda$, the Greek letter lambda, called the corresponding <b>eigenvalue</b>. Read the equation aloud as: <i>applying the matrix to this particular vector does the same thing as multiplying it by a single number.</i> The whole grid of $d^2$ entries, in this one direction, collapses to a single scale factor. "Eigen" is German for "own" or "characteristic", so these are the matrix's own directions.</p>

<p>For our $A$ the two eigenpairs are $v_1 = (1,1)$ with $\\lambda_1 = 3$, and $v_2 = (1,-1)$ with $\\lambda_2 = 1$. Note that eigenvectors have no natural length: if $Av = \\lambda v$ then $A(5v) = 5Av = 5\\lambda v = \\lambda(5v)$, so any scalar multiple is equally an eigenvector with the same eigenvalue. What is well defined is the <b>direction</b> — the line through the origin — which is why software always returns eigenvectors normalised to length one and why the sign it picks is arbitrary.</p>

<p>The eigenvalue carries the meaning. A value greater than 1 stretches along that direction, between 0 and 1 shrinks it, exactly 1 leaves it fixed, and a negative eigenvalue flips the arrow to point the other way along the same line. An eigenvalue of exactly zero says something stronger: $Av = 0$, so an entire direction is crushed to a point, the matrix is singular, and — as §0.2 argued — no inverse can bring that direction back.</p>

${H.analogy(`<p>Stand a pane of stretchy rubber on a table and pull it diagonally. Draw an arrow on the sheet at some random angle, then pull: the arrow gets longer and also swings toward the direction of the pull. Nearly every arrow you draw behaves that way.</p>
<p>But there are two arrows that do not swing — one lying exactly along the pull, and one lying exactly across it. Those two get longer or shorter without turning, and every other arrow's behaviour is a blend of what those two are doing. If someone tells you the direction of those two special arrows and how much each one stretched, they have told you everything about the pull.</p>
<p>That is precisely what an eigendecomposition is: the claim that a complicated-looking transformation is nothing more than independent stretches along a few special axes, described in the wrong coordinate system.</p>`)}

<h3>Why anyone cares: what happens when you apply a matrix many times</h3>

<p>Eigenvectors would be a curiosity if matrices were only ever applied once. They are load-bearing because matrices get applied repeatedly — in Markov chains, in the propagation of an error through a deep network's layers, in an optimiser taking step after step on the same quadratic landscape — and repeated application is where they become the only sensible language.</p>

<p>The reason is one line of algebra. If $Av = \\lambda v$, then $A^2v = A(\\lambda v) = \\lambda(Av) = \\lambda^2 v$, and so on:</p>

$$A^k v = \\lambda^k v$$

<p>Applying the matrix $k$ times to an eigenvector just raises its eigenvalue to the $k$-th power. And since any vector can be written as a combination of the eigenvectors, the fate of <i>every</i> vector under repeated application is governed by the eigenvalues alone.</p>

${H.worked('watching a generic arrow surrender to the top eigenvector', `<p>Start with $x = (1,0)$, which is nothing special. Write it in terms of the eigenvectors: $(1,0) = \\tfrac12(1,1) + \\tfrac12(1,-1)$, which you can check by adding the components.</p>
<p>Now apply $A$ ten times. Each eigen-component evolves independently, scaled by its own eigenvalue raised to the tenth power:</p>
<p>$A^{10}x = \\tfrac12 \\cdot 3^{10}\\,(1,1) + \\tfrac12 \\cdot 1^{10}\\,(1,-1) = \\tfrac12 \\cdot 59049\\,(1,1) + \\tfrac12(1,-1)$</p>
<p>$= (29524.5 + 0.5,\\; 29524.5 - 0.5) = (29525,\\; 29524)$</p>
<p>The ratio of the two components is $29525/29524 = 1.000034$. The arrow started at a 0° angle and has been dragged to within two thousandths of a degree of the $\\lambda = 3$ eigenvector. The second component did not disappear — it is still exactly $\\tfrac12(1,-1)$, unchanged — it has simply been swamped by a component that grew 59,049-fold.</p>
<p>This is <b>power iteration</b>, the oldest algorithm for finding the top eigenvector: multiply by the matrix repeatedly, renormalise, and wait. It is also, unmodified, the original PageRank. And it is why the ratio $|\\lambda_2/\\lambda_1|$ is the thing that governs convergence: here that ratio is $1/3$, so each application shrinks the contamination threefold. When the top two eigenvalues are close, the same algorithm crawls.</p>`)}

<p>The same arithmetic explains a phenomenon you will meet again in §3.4. If every eigenvalue of $A$ has magnitude below 1, then $A^k \\to 0$ and any signal passed through the map repeatedly dies. If any eigenvalue exceeds 1 in magnitude, $A^k$ blows up. Vanishing and exploding gradients in a recurrent network are that statement about the recurrent weight matrix, and gradient clipping is the crude but effective response to the second case.</p>

<h3>Diagonalisation: the same matrix in better coordinates</h3>

<p>Collect the eigenvectors as the columns of a matrix $Q$ and the eigenvalues along the diagonal of a matrix $\\Lambda$ (capital lambda, zeros everywhere off the diagonal). Then the eigenvector equations for all directions at once read $AQ = Q\\Lambda$, and if $Q$ is invertible,</p>

$$A = Q\\Lambda Q^{-1}$$

<p>Read this right-to-left, in the order the operations actually apply to a vector, and it stops being an identity and becomes a story. $Q^{-1}$ rewrites your vector in the eigenvector coordinate system — "how much of each special direction are you made of?" $\\Lambda$ then scales each of those coordinates by its own eigenvalue, which is trivial because a diagonal matrix does nothing but multiply each coordinate separately. $Q$ translates the answer back into ordinary coordinates. <mark>A matrix is a diagonal matrix that has been written down in the wrong basis.</mark></p>

${H.deriv('why $A^k = Q\\Lambda^k Q^{-1}$, and why that makes matrix powers easy', [
      ['$A^2 = (Q\\Lambda Q^{-1})(Q\\Lambda Q^{-1})$', 'Substitute the decomposition twice. Nothing has been assumed beyond $A$ being diagonalisable, which is what having a full set of independent eigenvectors means.'],
      ['$= Q\\Lambda (Q^{-1}Q) \\Lambda Q^{-1}$', 'Matrix multiplication is associative, so the brackets may be moved freely. It is not commutative, so the <i>order</i> may not be changed — that distinction is the one people get wrong.'],
      ['$= Q\\Lambda I \\Lambda Q^{-1} = Q\\Lambda^2 Q^{-1}$', '$Q^{-1}Q$ is the identity by definition of an inverse, and the identity can be dropped. The two $\\Lambda$s are now adjacent and combine.'],
      ['$A^k = Q\\Lambda^k Q^{-1}$', 'Repeat the argument $k$ times; every interior $Q^{-1}Q$ cancels in the same way, leaving one $Q$ on the outside and one $Q^{-1}$ on the other.'],
      ['$\\Lambda^k = \\mathrm{diag}(\\lambda_1^k, \\ldots, \\lambda_d^k)$', 'A diagonal matrix raised to a power just raises each diagonal entry to that power, because the entries never interact. This is the payoff: the hard object, a matrix power, reduces to $d$ ordinary numbers raised to a power.']
    ], 'The same substitution defines functions of matrices generally: $\\exp(A) = Q\\exp(\\Lambda)Q^{-1}$, $A^{-1} = Q\\Lambda^{-1}Q^{-1}$, $A^{1/2} = Q\\Lambda^{1/2}Q^{-1}$. The last one requires every eigenvalue to be non-negative — which is exactly the positive semi-definiteness condition of §1.8.4, and is why a covariance matrix has a square root and can be used to generate correlated Gaussian samples.')}

<h3>The symmetric case, which is the case you will actually meet</h3>

<p>In general $Q$ is just some invertible matrix, eigenvalues can be complex, and a matrix might not be diagonalisable at all. Nearly every matrix that matters in machine learning avoids all of this, because it is <b>symmetric</b>: $A = A^\\mathsf{T}$, the grid unchanged when flipped across its diagonal. Covariance matrices are symmetric. Gram and kernel matrices are symmetric. Hessians of twice-differentiable functions are symmetric. That is most of the objects in this course.</p>

<p>The <b>spectral theorem</b> says that for a real symmetric matrix, three good things happen at once: every eigenvalue is real, eigenvectors belonging to different eigenvalues are automatically orthogonal, and $Q$ can be chosen orthogonal, meaning $Q^{-1} = Q^\\mathsf{T}$. The decomposition becomes</p>

$$A = Q\\Lambda Q^\\mathsf{T}$$

<p>with no inverse to compute anywhere. Geometrically, a symmetric matrix is a pure stretch along a set of mutually perpendicular axes — no shear and no rotation, once you look along the right axes. Our example matrix $\\begin{bmatrix} 2 & 1 \\\\ 1 & 2\\end{bmatrix}$ is symmetric, and sure enough its eigenvectors $(1,1)$ and $(1,-1)$ meet at a right angle: their dot product is $1 \\times 1 + 1 \\times (-1) = 0$, orthogonal exactly as §0.2 defined it.</p>

${H.more('the characteristic polynomial, which is how you find them by hand', `<p>The geometry came first deliberately, because the algebra is easy to memorise and easy to learn nothing from. Here it is anyway.</p>
<p>Rearrange $Av = \\lambda v$ into $(A - \\lambda I)v = 0$. This says the matrix $A - \\lambda I$ maps a non-zero vector to zero, so it must be singular, so its determinant vanishes:</p>
<p>$\\det(A - \\lambda I) = 0$</p>
<p>Expanding that determinant gives a polynomial in $\\lambda$ of degree $d$ — the <b>characteristic polynomial</b> — whose roots are the eigenvalues. For our matrix:</p>
<p>$\\det\\begin{bmatrix} 2-\\lambda & 1 \\\\ 1 & 2-\\lambda\\end{bmatrix} = (2-\\lambda)^2 - 1 = \\lambda^2 - 4\\lambda + 3 = (\\lambda-3)(\\lambda-1)$</p>
<p>so $\\lambda = 3$ and $\\lambda = 1$, as found by hand at the top of the section. Substituting each root back into $(A-\\lambda I)v = 0$ and solving gives the corresponding eigenvector.</p>
<p>Two facts fall out of the polynomial that are worth carrying as checks on any numerical answer: the eigenvalues sum to the <b>trace</b> (the sum of the diagonal, here $2+2 = 4 = 3+1$) and multiply to the <b>determinant</b> (here $4 - 1 = 3 = 3 \\times 1$). Note also that nobody computes eigenvalues this way for $d$ larger than about 3 — root-finding on a high-degree polynomial is numerically appalling — so real software uses iterative methods instead (§1.15).</p>`)}

<h2><span class="sn">1.8.2</span> The SVD, and why it is the more useful object</h2>

<p>Eigendecomposition has two limitations, and both are fatal in practice. It requires a <b>square</b> matrix, so it says nothing at all about your $[n \\times d]$ data matrix with a million rows and 300 columns. And even for square matrices it can fail: the shear $\\begin{bmatrix} 1 & 1 \\\\ 0 & 1\\end{bmatrix}$ has only one eigen-direction, not two, so there is no basis of eigenvectors to change into.</p>

<p>The <b>singular value decomposition</b> repairs both. For <i>any</i> real matrix $A$ of shape $[m \\times n]$, rectangular and rank-deficient included, there exist</p>

$$A = U\\Sigma V^\\mathsf{T}$$

<p>where $U$ is $[m \\times m]$ orthogonal, $V$ is $[n \\times n]$ orthogonal, and $\\Sigma$ (capital sigma — here a matrix, not a summation) is $[m \\times n]$ with non-negative numbers $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge 0$ down its diagonal and zeros elsewhere. Those are the <b>singular values</b>.</p>

<p>Read it right-to-left again, as three things done to a vector in sequence. $V^\\mathsf{T}$ is orthogonal, so it is a pure rotation (possibly with a reflection) — it turns space without distorting it. $\\Sigma$ stretches each coordinate axis by its own singular value. $U$ rotates again. So:</p>

${H.key('Every linear map, without exception, is a rotation, then an axis-aligned stretch, then another rotation. There is no fourth kind of thing a matrix can do.')}

<p>That is a genuinely surprising claim the first time you meet it — the shear above certainly does not look like a rotate-stretch-rotate — and the lab below exists to make it visible one stage at a time.</p>

<p>The connection to eigenvectors is exact rather than analogous. Compute $A^\\mathsf{T}A$, which is square, symmetric and $[n \\times n]$ whatever shape $A$ was:</p>

$$A^\\mathsf{T}A = (U\\Sigma V^\\mathsf{T})^\\mathsf{T}(U\\Sigma V^\\mathsf{T}) = V\\Sigma^\\mathsf{T}U^\\mathsf{T}U\\Sigma V^\\mathsf{T} = V(\\Sigma^\\mathsf{T}\\Sigma)V^\\mathsf{T}$$

<p>using $(XY)^\\mathsf{T} = Y^\\mathsf{T}X^\\mathsf{T}$ to reverse the product, and $U^\\mathsf{T}U = I$ because $U$ is orthogonal. The right-hand side is a symmetric eigendecomposition. So <b>the right singular vectors of $A$ are the eigenvectors of $A^\\mathsf{T}A$, and the singular values are the square roots of its eigenvalues.</b> The SVD is the eigendecomposition of a square, symmetric object built out of a matrix that may have been neither.</p>

<p><b>What you are looking at.</b> The dashed pale circle is the unit circle — every vector of length one — before anything is done to it. The solid red outline is that circle after the stage you have selected. The blue and green arrows are the two columns of the matrix $A$ itself, drawn only at the input and final stages. At the final stage two amber arrows also appear: these are the left singular vectors scaled by their singular values, so they lie along the ellipse's major and minor axes and their lengths are $\\sigma_1$ and $\\sigma_2$. The readout gives both singular values, the condition number $\\kappa = \\sigma_1/\\sigma_2$, an effective rank, and the determinant.</p>

<p><b>What to do with it.</b> Step through the four stages in order with the default matrix. Stage 2 applies $V^\\mathsf{T}$: the shape is still a circle, because rotating a circle does nothing visible — that is the point, and it is worth sitting with, because it tells you the first rotation is choosing which directions will be stretched, not doing any stretching. Stage 3 applies $\\Sigma$ and the circle becomes an axis-aligned ellipse with semi-axes 2.069 and 0.677. Stage 4 applies $U$ and the ellipse tilts into its final orientation. Then drag either column arrow — you are editing the matrix by hand — and watch all four stages update. The arrows are drawn only at stages 1 and 4, so drag there.</p>

<p><b>The thing genuinely worth noticing.</b> Press <i>Make it nearly rank-1</i>. That preset is $\\begin{bmatrix} 1.5 & 0.75 \\\\ 1.2 & 0.6\\end{bmatrix}$, whose second column is exactly half the first, so it is not nearly rank one — it is exactly rank one. The determinant reads 0.000, $\\sigma_2$ collapses to zero, the condition number readout runs off to an absurd number, and the red ellipse flattens into a line segment: the entire plane has been squashed onto one direction. Then press <i>Ill-conditioned</i>, which is $\\begin{bmatrix} 2 & 1.9 \\\\ 1.9 & 1.85\\end{bmatrix}$. Its determinant is $2 \\times 1.85 - 1.9^2 = 0.09$, comfortably non-zero, so it is invertible and a textbook would call it fine — yet $\\sigma_1 = 3.83$ against $\\sigma_2 = 0.024$, a condition number of about 163. The ellipse is a sliver. This is the shape of numerical trouble: not a matrix that is singular, but one that is <i>nearly</i> singular, where the inverse exists and amplifies error by a factor of 163 while doing so.</p>

${H.lab('svd', 'SVD is rotate · scale · rotate', 'Drag the matrix columns and watch the unit circle become an ellipse in three separate stages. σ₁/σ₂ is the condition number; when σ₂ ≈ 0 the matrix is effectively rank one — the same picture that explains why low-rank adapters work.')}

<h3>Truncation, Eckart–Young, and the theorem underneath a lot of machine learning</h3>

<p>Write the SVD a second way, as a sum instead of a product. If $u_k$ and $v_k$ are the $k$-th columns of $U$ and $V$, then</p>

$$A = \\sum_{k=1}^{r} \\sigma_k\\, u_k v_k^\\mathsf{T}$$

<p>Each term $u_k v_k^\\mathsf{T}$ is an <b>outer product</b>: a column vector times a row vector, producing a full-sized matrix of rank one. So the SVD decomposes any matrix into a sum of rank-one layers, ordered by importance, with $\\sigma_k$ saying how much each layer contributes. This is the more useful mental image, because it invites the obvious question: what if you keep only the first few layers?</p>

<p>Keeping the largest $r$ terms gives a rank-$r$ matrix $A_r$, and the <b>Eckart–Young theorem</b> says something remarkably strong about it: among <i>all</i> matrices of rank $r$, that truncation minimises the Frobenius-norm error $\\|A - A_r\\|_F$. Not "is a reasonable heuristic" — is provably optimal. The error it leaves behind is $\\sqrt{\\sum_{k>r}\\sigma_k^2}$, the discarded singular values in quadrature.</p>

<p>Almost every low-rank idea in machine learning is a use of this theorem. PCA (§2.10) is the SVD of a centred data matrix, with the singular values reporting how much variance each component carries. Latent semantic analysis is the same operation on a term–document matrix. And <b>LoRA</b> (§4.13) is the bet that the <i>update</i> a fine-tune makes to a weight matrix is well approximated by a rank-$r$ matrix, so you can store $BA$ with $r \\ll d$ and lose almost nothing. Eckart–Young is what licenses the phrase "lose almost nothing" — it is what tells you the discarded part is exactly the small singular values.</p>

<p><b>What you are looking at.</b> On the left, a 48×48 grid of values drawn as a greyscale image. In the middle, the same grid reconstructed from only the top $r$ singular values. On the right, the singular value spectrum drawn as horizontal bars, longest at the top, with the bars you are keeping in blue and the discarded ones in grey. The readout gives the fraction of squared "energy" retained, the relative reconstruction error, the count of numbers you had to store versus the 2,304 in the original, and the resulting compression ratio.</p>

<p><b>What to do with it.</b> Drag $r$ down to 1 and back up slowly. At $r = 1$ you keep 58.4 per cent of the energy and the reconstruction is a smeared ghost of the original. At $r = 2$ you are at 99.1 per cent, and at $r = 3$ the energy readout rounds to 100.0 per cent. Look at the error readout at $r = 3$ though: it says 1.5 per cent, not 0. Those two are not inconsistent — error is the square root of the missing energy, and $\\sqrt{1 - 0.9998} = 0.0146$. Square roots magnify small deficits, which is why "99.9 per cent of variance explained" is a much weaker statement than it sounds.</p>

<p><b>The thing genuinely worth noticing.</b> At $r = 4$ the error hits exactly zero and stays there however far you push the slider, and the spectrum bars beyond the fourth are empty. That is not a property of images; it is a property of <i>this</i> image. Each of the four terms the simulator adds together is a function of the row index multiplied by a function of the column index, and every such separable term contributes exactly rank one — so the picture is rank 4 by construction. A photograph would show a spectrum that decays steeply but never terminates, so truncation there is a genuine approximation with a genuine error, and the interesting question becomes where to stop. Read this lab as a clean demonstration of the mechanism rather than as evidence about how real data behaves.</p>

${H.lab('lowrank', 'Rank-r approximation, on a real matrix', 'A 48×48 image reconstructed from its top r singular values, computed live. Watch the error and the storage cost fall out of the same curve — and notice how few components carry the structure.')}

<h2><span class="sn">1.8.3</span> Positive semi-definiteness, or: which matrices are bowls</h2>

<p>One more property, and it is the one that decides whether an optimisation problem is easy. Take a symmetric matrix $A$ and build the scalar</p>

$$q(x) = x^\\mathsf{T}Ax$$

<p>This is a <b>quadratic form</b>. Despite the intimidating notation it is just a number: by §0.2's rule for matrix-vector products, $Ax$ is a vector, and dotting it with $x$ gives a scalar. For $A = \\begin{bmatrix} 2 & 0 \\\\ 0 & 3\\end{bmatrix}$ and $x = (x_1,x_2)$ it works out to $2x_1^2 + 3x_2^2$ — a bowl, curving upward in every direction from the origin. For $A = \\begin{bmatrix} 1 & 0 \\\\ 0 & -1\\end{bmatrix}$ it is $x_1^2 - x_2^2$, a saddle: uphill along one axis, downhill along the other.</p>

<p>The difference between those two pictures is entirely a matter of signs of eigenvalues, and it has a name. $A$ is <b>positive semi-definite</b>, written $A \\succeq 0$, when $x^\\mathsf{T}Ax \\ge 0$ for every $x$ — equivalently, when every eigenvalue is $\\ge 0$. It is <b>positive definite</b>, $A \\succ 0$, when the inequality is strict for every non-zero $x$, equivalently when every eigenvalue is strictly positive.</p>

<p>The equivalence is worth seeing rather than accepting, and for a symmetric matrix it is two lines. Write $x$ in the eigenvector basis as $x = \\sum_k c_k v_k$. Then $Ax = \\sum_k c_k \\lambda_k v_k$, and because the eigenvectors are orthonormal, all the cross terms in the dot product vanish and</p>

$$x^\\mathsf{T}Ax = \\sum_k \\lambda_k c_k^2$$

<p>Every $c_k^2$ is non-negative, so the whole expression is non-negative for all $x$ exactly when every $\\lambda_k$ is. The quadratic form is a bowl precisely when the matrix stretches, and never reflects, along each of its own axes.</p>

${H.intuition(`<p>Positive semi-definite means "no direction in which this thing curves downward". That single sentence is why the property keeps appearing in three otherwise unrelated places.</p>
<p><b>Covariance.</b> A variance cannot be negative, and $x^\\mathsf{T}\\Sigma x$ <i>is</i> a variance — specifically the variance of the scalar projection $x \\cdot z$ of your random vector $z$ onto the direction $x$. So covariance matrices are PSD for the simplest possible reason, and it also tells you what a zero eigenvalue means: a direction along which your data has no spread at all, which is a perfectly collinear combination of features.</p>
<p><b>Kernels.</b> A kernel is supposed to be an inner product in some feature space you never construct. Inner products of real vectors give PSD Gram matrices, so if your kernel matrix has a negative eigenvalue, no such feature space exists and the "kernel trick" is meaningless. That is Mercer's condition (§2.6), and the practical consequence is that the SVM dual stops being a convex problem and its solver stops being trustworthy.</p>
<p><b>Hessians.</b> A function is convex exactly when its Hessian is PSD everywhere (§1.9). Bowls have no bad local minima, so the problem is "solved" in a sense that neural networks are not.</p>
<p>Three fields, three vocabularies, one condition: nothing curves the wrong way.</p>`)}

${H.table(['Object', 'Why it is PSD', 'What that buys'], [
      ['Covariance $\\Sigma$', 'It is $\\mathbb{E}[(x-\\mu)(x-\\mu)^\\mathsf{T}]$, a sum of outer products', 'PCA is an eigenproblem with real, non-negative eigenvalues'],
      ['Gram / kernel matrix $K$', 'Mercer’s condition', 'The SVM dual is a convex QP with a unique optimum'],
      ['Hessian of a convex function', 'Definition of convexity', 'No bad local minima; Newton’s method is well-posed']
    ])}

${H.practice(`<p>PSD matrices arrive from theory and then fail numerically. A covariance matrix estimated from data is PSD in exact arithmetic, but in floating point its smallest eigenvalues come out as tiny negative numbers such as $-3 \\times 10^{-17}$, and the Cholesky factorisation that a Gaussian sampler or a Gaussian process needs will refuse to run. The standard repair is to add $\\varepsilon I$ for a small $\\varepsilon$ — called jitter, or a nugget, or ridge, depending on which field is speaking, and it is the identical operation to the $\\lambda I$ that rescues linear regression in §2.3.</p>
<p>It also fails structurally. Estimate a covariance matrix over 500 assets from 250 days of returns and it cannot possibly have rank above 250, so at least 250 eigenvalues are exactly zero, and the matrix is singular no matter how clean your data is. Whenever the number of features approaches the number of observations, shrinkage is not optional. And if you ever build a kernel matrix by hand — an edit-distance similarity, say, or a hand-tuned correlation matrix — check its smallest eigenvalue before trusting anything downstream, because plausible-looking similarity functions are very often not PSD.</p>`)}

${H.probe([
      ['What is the SVD in one sentence?', 'Rotate, scale along axes, rotate — and truncating the singular values gives the best low-rank approximation.'],
      ['Why must a kernel be PSD?', 'It must correspond to an inner product in some feature space; otherwise the dual problem is not convex and the "kernel trick" is meaningless.'],
      ['How does this connect to LoRA?', 'If the fine-tuning update is well approximated by a rank-r matrix, storing $BA$ with $r \\ll d$ loses almost nothing — Eckart–Young is the licence.'],
      ['Eigendecomposition or SVD — when does the distinction bite?', 'Eigen needs a square matrix and can fail to exist; SVD always exists and handles rectangular data matrices. For a symmetric PSD matrix the two coincide, which is why PCA can be described either way.']
    ])}`,
    labs: {
      svd: function (host) {
        let A = [[1.6, .9], [.4, 1.1]];
        let drag = null;
        const st = Viz.controls(host, [
          { k: 'stage', label: 'stage', type: 'buttons', value: 'all', options: [
            { v: 'in', t: '1 · input' }, { v: 'v', t: '2 · VᵀX rotate' }, { v: 'sv', t: '3 · ΣVᵀX scale' }, { v: 'all', t: '4 · UΣVᵀX' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 's1', label: 'σ₁', cls: 'key' }, { k: 's2', label: 'σ₂' }, { k: 'cond', label: 'condition κ' },
          { k: 'rank', label: 'effective rank' }, { k: 'det', label: 'det' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const M = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]];
            const { U, s, V } = Num.svd(M);
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-2.4, 2.4] }).frame({});
            const circle = [];
            for (let t = 0; t <= 6.3; t += .04) circle.push([Math.cos(t), Math.sin(t)]);
            const applyVt = p => [V[0][0] * p[0] + V[0][1] * p[1], V[1][0] * p[0] + V[1][1] * p[1]];
            const applyS = p => [s[0] * p[0], s[1] * p[1]];
            const applyU = p => [U[0][0] * p[0] + U[1][0] * p[1], U[0][1] * p[0] + U[1][1] * p[1]];
            let shape = circle, label = 'unit circle';
            if (st.stage === 'v') { shape = circle.map(applyVt); label = 'after VᵀX — a rotation, still a circle'; }
            else if (st.stage === 'sv') { shape = circle.map(p => applyS(applyVt(p))); label = 'after ΣVᵀX — stretched along the axes by σ₁, σ₂'; }
            else if (st.stage === 'all') { shape = circle.map(p => applyU(applyS(applyVt(p)))); label = 'after UΣVᵀX = AX — the final ellipse'; }
            P.clip(() => {
              P.line(circle, { color: T.faint, width: 1.2, dash: [4, 3] });
              P.line(shape, { color: T.red, width: 2.6 });
              if (st.stage === 'all' || st.stage === 'in') {
                P.arrow(0, 0, A[0][0], A[1][0], { color: T.blue, width: 2.4 });
                P.arrow(0, 0, A[0][1], A[1][1], { color: T.green, width: 2.4 });
              }
              if (st.stage === 'all') {
                const u1 = [U[0][0], U[0][1]], u2 = [U[1][0], U[1][1]];
                P.arrow(0, 0, u1[0] * s[0], u1[1] * s[0], { color: T.amber, width: 2 });
                P.arrow(0, 0, u2[0] * s[1], u2[1] * s[1], { color: T.amber, width: 1.4 });
              }
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText(label, w / 2, 8);
            out({
              s1: s[0].toFixed(3), s2: s[1].toFixed(3),
              cond: (s[0] / (s[1] || 1e-9)).toFixed(1),
              rank: s[1] / (s[0] || 1) < .05 ? '≈ 1' : '2',
              det: (A[0][0] * A[1][1] - A[0][1] * A[1][0]).toFixed(3)
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          const x = P.ix(e.x), y = P.iy(e.y);
          if (e.type === 'down') {
            const d1 = (A[0][0] - x) ** 2 + (A[1][0] - y) ** 2, d2 = (A[0][1] - x) ** 2 + (A[1][1] - y) ** 2;
            drag = Math.min(d1, d2) < .3 ? (d1 < d2 ? 0 : 1) : null;
          } else if (e.type === 'move' && e.down && drag !== null) { A[0][drag] = x; A[1][drag] = y; S.redraw(); }
          else if (e.type === 'up') drag = null;
        });
        Viz.buttons(host, [
          { label: 'Make it nearly rank-1', on: () => { A = [[1.5, .75], [1.2, .6]]; S.redraw(); } },
          { label: 'Make it orthogonal', on: () => { A = [[.9, -.44], [.44, .9]]; S.redraw(); } },
          { label: 'Ill-conditioned', on: () => { A = [[2, 1.9], [1.9, 1.85]]; S.redraw(); } }
        ]);
      },

      lowrank: function (host) {
        const N = 48;
        // a structured synthetic "image": low-rank content plus detail
        const M = [];
        for (let i = 0; i < N; i++) {
          const row = new Float64Array(N);
          for (let j = 0; j < N; j++) {
            const x = i / N, y = j / N;
            row[j] = Math.sin(6 * x) * Math.cos(5 * y) + .6 * Math.sin(11 * y) + .5 * (x > .6 && y < .4 ? 1 : 0)
              + .35 * Math.exp(-((x - .3) ** 2 + (y - .7) ** 2) * 30);
          }
          M.push(row);
        }
        const { U, s, V } = Num.svd(M.map(r => Array.from(r)));
        const total = s.reduce((a, b) => a + b * b, 0);
        const st = Viz.controls(host, [
          { k: 'r', label: 'rank r kept', min: 1, max: 24, step: 1, value: 4, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'keep', label: 'energy kept', cls: 'key' }, { k: 'err', label: 'relative error' },
          { k: 'nums', label: 'numbers stored' }, { k: 'comp', label: 'compression' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const r = st.r;
            const rec = Array.from({ length: N }, () => new Float64Array(N));
            for (let k = 0; k < r; k++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) rec[i][j] += s[k] * U[k][i] * V[k][j];
            const cell = Math.min((h - 60) / N, (w * .42) / N);
            const draw = (mat, ox, title) => {
              for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
                const v = Math.max(-1.6, Math.min(1.6, mat[i][j]));
                const t = (v + 1.6) / 3.2;
                const g = Math.round(255 * t);
                ctx.fillStyle = 'rgb(' + Math.round(40 + g * .75) + ',' + Math.round(50 + g * .7) + ',' + Math.round(90 + g * .6) + ')';
                ctx.fillRect(ox + j * cell, 30 + i * cell, cell + .6, cell + .6);
              }
              ctx.strokeStyle = T.line; ctx.strokeRect(ox, 30, N * cell, N * cell);
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
              ctx.fillText(title, ox, 24);
            };
            draw(M, 10, 'original · rank ' + N);
            draw(rec, 20 + N * cell, 'reconstruction · rank ' + r);
            // spectrum
            const sx = 30 + 2 * N * cell, sw = Math.max(60, w - sx - 14);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
            ctx.fillText('singular values', sx, 24);
            const smax = s[0];
            s.slice(0, 24).forEach((v, i) => {
              ctx.fillStyle = i < r ? T.blue : T.line;
              const bh = 6;
              ctx.fillRect(sx, 32 + i * (bh + 2), sw * v / smax, bh);
            });
            let kept = 0; for (let k = 0; k < r; k++) kept += s[k] * s[k];
            const stored = r * (2 * N + 1);
            out({
              keep: (100 * kept / total).toFixed(1) + '%',
              err: (100 * Math.sqrt(Math.max(0, 1 - kept / total))).toFixed(1) + '%',
              nums: stored.toLocaleString() + ' vs ' + (N * N).toLocaleString(),
              comp: (N * N / stored).toFixed(1) + '×'
            });
          }
        });
        Viz.note(host, 'This is Eckart–Young in action: the rank-r truncation is provably the best rank-r approximation in Frobenius norm. LoRA (§4.13) is the bet that a fine-tuning update to a weight matrix has the same shape — a few directions carrying almost everything.');
      }
    },
    quiz: [
      {
        q: 'Which is true of the SVD?',
        options: ['It only exists for symmetric matrices', 'It exists for any matrix and its truncation is the optimal low-rank approximation', 'It requires the matrix to be invertible', 'It produces complex eigenvalues in general'],
        answer: 1,
        why: 'Every real matrix has a real SVD — rectangular, singular and non-diagonalisable ones included — and Eckart–Young says the rank-$r$ truncation minimises Frobenius error among <i>all</i> rank-$r$ matrices, not merely among truncations. The first option is tempting because it confuses the SVD with the eigendecomposition, which really does need a square matrix and can fail even then; §1.8.2 shows the shear $[[1,1],[0,1]]$ as a square matrix with only one eigen-direction but a perfectly ordinary SVD. Getting the optimality claim right matters because it is what licenses PCA, latent semantic analysis and LoRA in one stroke: the discarded part is provably the small singular values and nothing else.'
      },
      {
        q: 'A kernel matrix that is not positive semi-definite means…',
        options: ['the data is not normalised', 'there is no feature space in which the kernel is an inner product, so the problem is no longer convex', 'the kernel is too smooth', 'the bandwidth γ is too small'],
        answer: 1,
        why: 'Mercer\'s condition is exactly the statement that a PSD kernel matrix corresponds to an inner product in some, possibly infinite-dimensional, feature space. If an eigenvalue is negative there is no such space, so the "kernel trick" has nothing to be a trick for, and the SVM dual stops being a convex quadratic programme with a unique optimum. The bandwidth answer is tempting because $\\gamma$ is the knob people actually turn and mis-tuning it does cause visible trouble — but a badly tuned RBF kernel is still PSD, so the symptom is overfitting rather than an invalid problem. This bites in practice when you hand-build a similarity matrix from something like an edit distance: plausible-looking similarity functions frequently are not PSD, so check the smallest eigenvalue before trusting anything downstream.'
      },
      {
        q: 'A 2×2 matrix has $\\sigma_1 = 3.83$, $\\sigma_2 = 0.024$ and determinant 0.09. Which statement is correct?',
        options: ['It is singular, so no inverse exists', 'It is invertible, but solving a system with it amplifies error by a factor of about 160', 'Its eigenvalues must be complex', 'It is orthogonal, since the determinant is non-zero'],
        answer: 1,
        why: 'A non-zero determinant means the matrix is invertible, so nothing has been flattened and the first option is wrong. But invertibility is a yes/no fact and the condition number $\\kappa=\\sigma_1/\\sigma_2 \\approx 160$ is the quantitative one, and it says the map stretches 160 times more along one direction than another — so the inverse magnifies whatever error is in your right-hand side by up to that factor. "Singular" is tempting because $\\sigma_2$ looks like zero next to $\\sigma_1$, and that instinct is nearly right in the way that matters: numerical trouble arrives well before a matrix is exactly singular, which is why "nearly collinear features" is already a problem in §2.3 and why ridge\'s $\\lambda I$ helps. You can reproduce this exact matrix with the <i>Ill-conditioned</i> preset in the SVD lab.'
      }
    ],
    cards: [
      { q: 'SVD in one sentence', a: 'Rotate · scale · rotate; $A=U\\Sigma V^\\mathsf{T}$, and truncation gives the best low-rank fit (Eckart–Young).' },
      { q: 'PSD definition and consequence', a: '$x^\\mathsf{T}Ax\\ge0$ ⟺ all eigenvalues ≥ 0. Covariance and kernel matrices are PSD, making their optimisations convex.' },
      { q: 'Condition number', a: '$\\kappa=\\sigma_{\\max}/\\sigma_{\\min}$ — how much the map stretches unevenly; it governs both numerical stability and gradient-descent speed.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.9 */
  ML.section({
    id: 'calculus-ml', track: 'foundations', num: '1.9',
    title: 'Calculus for ML: convexity, conditioning, Newton, Lagrange',
    lede: 'Every trainable thing on this site is minimised with the machinery in this section — and most optimisation pain is one number, the condition number.',
    rests: 'Consumed by §2.8 (XGBoost is a Newton step), §3.5 optimisers, §4.11 training at scale.',
    html: `
<p>Every trainable thing on this site is fitted the same way. You have a loss, a number saying how badly the model is doing, and you have parameters, the knobs you are allowed to turn. The derivative tells you which way each knob should turn. §0.3 established that much, and §0.7 established how to do it when the knobs are a vector rather than a single number.</p>

<p>What neither of them answered is the question that actually decides whether training works: <b>how far do you turn the knob?</b></p>

<p>Consider a loss surface shaped like this one, and take it seriously as a picture rather than as a formula:</p>

$$f(x, y) = \\tfrac{1}{2}\\left(x^2 + 18y^2\\right)$$

<p>It is a bowl, so there is a unique minimum at the origin and no local minima to get stuck in. But it is a badly proportioned bowl. Along $y$ it is eighteen times as steep as along $x$ — a long narrow valley, or a ravine. Now start at $(-2.2,\\; 1.2)$ and try to walk downhill by repeatedly taking a step against the gradient.</p>

<p>A step size small enough to be safe in the steep $y$ direction is far too small to make progress along the shallow $x$ direction, so you inch along the valley floor while zig-zagging across it. A step size large enough to make good progress in $x$ overshoots in $y$, bounces to the other wall, overshoots again, and diverges. There is no learning rate that is right for both directions at once, because the problem itself has two very different scales in it.</p>

<p>That single tension explains an enormous amount of optimisation practice — why you standardise features, why momentum helps, what Adam is actually doing, and why second-order methods converge in a handful of steps but nobody uses them at scale. The number that quantifies the tension is called the condition number, and this section is about it.</p>

<h2><span class="sn">1.9.1</span> Curvature: the second derivative, promoted to a matrix</h2>

<p>The gradient $\\nabla f$ collects the first partial derivatives, points in the direction of steepest ascent, and vanishes at any stationary point. From §0.7 you also have two identities that recur constantly:</p>

$$\\nabla_x(a^\\mathsf{T}x) = a, \\qquad \\nabla_x(x^\\mathsf{T}Ax) = (A + A^\\mathsf{T})x \\;\\;(= 2Ax \\text{ for symmetric } A)$$

<p>The gradient alone tells you a direction and says nothing about distance. To get distance you need to know how quickly the slope itself is changing, which is the second derivative — and in many variables the second derivatives form a matrix. The <b>Hessian</b> $H$ has entries</p>

$$H_{ij} = \\frac{\\partial^2 f}{\\partial \\theta_i\\, \\partial \\theta_j}$$

<p>read as "the rate at which the slope along $\\theta_i$ changes as you move along $\\theta_j$". For twice-continuously-differentiable functions the order of differentiation does not matter, so $H$ is symmetric — which means everything §1.8 said about symmetric matrices applies to it, and that is the reason this section can borrow eigenvalues so freely.</p>

<p>For our ravine, $f = \\tfrac12(x^2 + 18y^2)$ has gradient $(x,\\; 18y)$ and Hessian $\\begin{bmatrix} 1 & 0 \\\\ 0 & 18\\end{bmatrix}$, constant everywhere. Its eigenvalues are 1 and 18, its eigenvectors are the two coordinate axes, and those two numbers are literally the curvatures along the valley and across it.</p>

<p>The object that ties gradient and Hessian together is the <b>second-order Taylor expansion</b>. Around a current point $\\theta$, for a small step $\\Delta$:</p>

$$f(\\theta + \\Delta) \\approx f(\\theta) + \\nabla f(\\theta)^\\mathsf{T}\\Delta + \\tfrac12 \\Delta^\\mathsf{T}H\\Delta$$

<p>In words: the new loss is the old loss, plus a linear term that says how the slope pays off over the step, plus a quadratic term that says how much the slope changes during the step. The first two terms alone are what gradient descent implicitly uses, and the third is the correction that Newton's method exploits. Notice that the quadratic term is exactly the quadratic form of §1.8.3, so its sign behaviour is decided by the Hessian's eigenvalues.</p>

<h2><span class="sn">1.9.2</span> Convexity, and what it actually rules out</h2>

<p>A function is <b>convex</b> if the straight line joining any two points on its graph never dips below the graph itself. Picture a rope stretched between two points on the surface: for a convex function the rope always lies on or above the surface.</p>

<p>The calculus version is the one you use. A twice-differentiable $f$ is convex if and only if $H \\succeq 0$ everywhere — its Hessian is positive semi-definite at every point, meaning no direction curves downward anywhere. That is precisely the condition §1.8.3 built.</p>

<p>Convexity buys you one thing, and it is worth a great deal: <mark>every local minimum of a convex function is a global minimum.</mark> There are no valleys to get trapped in, so an algorithm that goes reliably downhill will reach the answer, and "the optimum" is a well-defined object you can talk about without asking which run of the optimiser you meant. Linear regression, ridge, logistic regression, and SVMs are all convex, which is why they are considered solved problems, why two implementations agree to six decimal places, and why nobody reports the random seed.</p>

<p>Neural networks are not convex, and no amount of cleverness makes them so — swapping two hidden units gives a different parameter vector with identical loss, so the landscape has vast numbers of equivalent minima by symmetry alone. What saves deep learning is empirical rather than theoretical: in high dimensions, the bad stationary points are overwhelmingly saddles rather than local minima, because a point is a local minimum only if the Hessian curves upward in <i>every one</i> of millions of directions, and that is an extremely demanding coincidence. Saddles have escape directions, and stochastic gradient noise finds them.</p>

<h2><span class="sn">1.9.3</span> Newton's method: use the curvature you already computed</h2>

<p>Gradient descent is $\\theta \\leftarrow \\theta - \\eta\\nabla f$, where $\\eta$ (the Greek letter eta) is the learning rate. The step direction is the negative gradient and the step length is set by a number you chose by hand. That hand-chosen number is the weak point, and the Taylor expansion suggests a way to avoid choosing it.</p>

${H.deriv('Newton’s step, from the quadratic model', [
      ['$m(\\Delta) = f(\\theta) + g^\\mathsf{T}\\Delta + \\tfrac12\\Delta^\\mathsf{T}H\\Delta$', 'Write the second-order Taylor model of the loss, abbreviating $g = \\nabla f(\\theta)$. This is a genuine quadratic function of the step $\\Delta$ — the only unknown — and unlike $f$ itself, a quadratic can be minimised exactly.'],
      ['$\\nabla_\\Delta m = g + H\\Delta$', 'Differentiate with respect to $\\Delta$ using the two identities above: the linear term contributes $g$, and the quadratic term contributes $H\\Delta$ rather than $2H\\Delta$ because of the $\\tfrac12$ out front. $H$ is symmetric, so $(H+H^\\mathsf{T})/2 = H$.'],
      ['$g + H\\Delta = 0$', 'Set the gradient of the model to zero, which locates its stationary point. If $H \\succ 0$ the model is a genuine bowl, so this stationary point is its unique minimum rather than a saddle or a maximum.'],
      ['$\\Delta = -H^{-1}g$', 'Solve for the step. In practice you never form $H^{-1}$; you solve the linear system $H\\Delta = -g$, which is cheaper and numerically better behaved (§1.15).'],
      ['$\\theta \\leftarrow \\theta - H^{-1}\\nabla f$', 'The Newton update. Note there is no learning rate: the curvature supplied the step length, which is the entire point. On a quadratic the model is exact, so this lands on the minimum in a single step from anywhere.']
    ], 'Read the update geometrically: $H^{-1}$ rescales each eigen-direction of the Hessian by the inverse of its curvature, so steep directions get short steps and shallow directions get long ones — automatically, and by exactly the right amounts. That is why Newton is immune to conditioning, and why every practical optimiser is in some sense trying to approximate it cheaply.')}

<p>Two consequences. First, near a minimum Newton converges <i>quadratically</i>: the number of correct digits roughly doubles per iteration, so it arrives in a handful of steps where gradient descent needs thousands. Second, forming and factorising $H$ costs $O(p^3)$ in the parameter count $p$, and merely storing it costs $O(p^2)$ — for a model with a hundred million parameters the Hessian would need more memory than exists. That is the whole reason first-order methods dominate at scale, and why <b>L-BFGS</b>, which builds a low-rank approximation to $H^{-1}$ from a window of recent gradients, is the compromise used on medium-sized problems.</p>

<p>One special case is worth carrying to interviews: <mark>XGBoost's split score is a Newton step in function space</mark> (§2.8). It Taylor-expands the loss to second order around the current predictions, keeps the per-example gradients $g_i$ and curvatures $h_i$, and the optimal leaf value falls out as $-G/(H+\\lambda)$ — a gradient divided by a curvature, which is $-H^{-1}g$ for a one-dimensional parameter. Recognising that formula as Newton rather than as an arbitrary boosting rule is the difference between reciting and understanding.</p>

<h2><span class="sn">1.9.4</span> Conditioning explains most optimisation pain</h2>

<p>Now make the ravine precise. For a quadratic with Hessian eigenvalues $\\lambda_{\\min} \\le \\cdots \\le \\lambda_{\\max}$, define the <b>condition number</b></p>

$$\\kappa = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}}$$

<p>read as "kappa". It is the ratio of the sharpest curvature to the shallowest — the aspect ratio of the bowl. It is the same $\\kappa$ that §1.8 defined as $\\sigma_1/\\sigma_2$, and the two agree whenever the matrix is symmetric PSD.</p>

${H.deriv('why $\\kappa$ sets both the largest safe step and the speed', [
      ['$f(x) = \\tfrac12\\sum_i \\lambda_i x_i^2$', 'Any quadratic can be written this way by rotating into the Hessian’s eigenvector basis, using the spectral theorem of §1.8. The coordinates now move independently, which is the only reason this analysis is tractable.'],
      ['$\\nabla f = (\\lambda_1 x_1, \\ldots, \\lambda_d x_d)$', 'Differentiate coordinate by coordinate. Each coordinate’s gradient depends only on itself, so gradient descent runs $d$ separate one-dimensional problems in parallel.'],
      ['$x_i \\leftarrow x_i - \\eta\\lambda_i x_i = (1-\\eta\\lambda_i)\\,x_i$', 'Substitute into the gradient-descent update. Each coordinate is multiplied by the fixed factor $(1-\\eta\\lambda_i)$ on every step — a geometric sequence, one per direction.'],
      ['$|1-\\eta\\lambda_i| < 1 \\iff 0 < \\eta < 2/\\lambda_i$', 'A geometric sequence converges to zero exactly when its ratio has magnitude below 1. Above $2/\\lambda_i$ the factor is more negative than $-1$, so the coordinate flips sign and grows — that is what divergence looks like, and it is why divergence oscillates rather than drifting away.'],
      ['$\\eta < 2/\\lambda_{\\max}$', 'The step must be safe in <i>every</i> direction simultaneously, so the sharpest curvature sets the ceiling for all of them. This is the mathematical statement of "the steep direction dictates the step size".'],
      ['$|1-\\eta\\lambda_{\\min}| \\to \\dfrac{\\kappa-1}{\\kappa+1}$', 'Now ask how fast the <i>slowest</i> coordinate converges when $\\eta$ is chosen as well as possible. Optimising $\\eta$ to balance the fastest and slowest directions gives $\\eta = 2/(\\lambda_{\\max}+\\lambda_{\\min})$, at which both extreme factors equal $(\\kappa-1)/(\\kappa+1)$ in magnitude.']
    ], 'Put numbers on the last line. At $\\kappa = 10$ the per-step contraction is $9/11 = 0.818$, and reducing the error by a factor of $10^4$ takes about 46 steps. At $\\kappa = 100$ it is $99/101 = 0.980$, and the same reduction takes about 461 steps. Ten times the condition number, ten times the steps — the relationship is linear in $\\kappa$, and $\\kappa$ in real problems is routinely in the thousands.')}

<p>Every practical fix in the optimiser toolbox is a response to this one quantity.</p>

${H.table(['Fix', 'Mechanism'], [
      ['Feature scaling / normalisation', 'Shrinks κ directly by making the curvature more isotropic'],
      ['Momentum', 'Accumulates the consistent shallow direction while cancelling the alternating steep one'],
      ['Adam / RMSProp', 'Per-coordinate scaling approximates dividing by curvature'],
      ['Newton / L-BFGS', 'Removes the problem outright by rescaling with $H^{-1}$ — at a cost nobody pays at scale'],
      ['Batch norm / layer norm', 'Keeps the effective curvature stable across depth (§3.6)']
    ])}

${H.intuition(`<p>Feature scaling deserves a sentence of its own, because it is the cheapest item on that list and the one most often skipped. For linear regression the Hessian is $X^\\mathsf{T}X$, so the curvature along each parameter is set by the <i>scale</i> of the corresponding feature. Measure income in pounds and age in years and one column has a variance millions of times the other's, so $\\kappa$ is millions and gradient descent crawls.</p>
<p>Standardising every column to mean zero and unit variance makes the diagonal of $X^\\mathsf{T}X$ uniform and collapses $\\kappa$ to whatever the correlations between features demand — often a factor of thousands, for one line of preprocessing. This is the same act that §0.2 justified for distances and §0.5 insists on fitting inside the cross-validation fold, and it is now paying for itself a third time.</p>`)}

<p><b>What you are looking at.</b> The faint closed curves are contours of the loss $f = \\tfrac12(x^2 + \\kappa y^2)$ — lines of equal height, so a circular set of contours is a well-conditioned bowl and a set of thin ellipses is a ravine. The horizontal axis is the shallow direction, the vertical is the steep one. The green dot is the fixed starting point, the red dot at the origin is the minimum, and the blue path with its dots is the actual trajectory the selected optimiser takes, computed step by step rather than sketched. The readout gives the number of steps taken to bring the loss below $10^{-4}$, the largest theoretically stable learning rate $2/\\lambda_{\\max}$, the final loss, and the predicted gradient-descent contraction factor.</p>

<p><b>What to do with it.</b> Leave everything at its defaults — $\\kappa = 18$, $\\eta = 0.08$, plain gradient descent — and note that it converges in 61 steps with a visible zig-zag. The readout says the maximum stable step is 0.111. Now raise $\\eta$ to 0.110 and the path survives but crawls, taking more than the 220 steps the simulation runs; raise it one notch further to 0.115 and the readout switches to "diverged". You have just located a theoretical boundary experimentally, and it sits exactly where $2/18 = 0.1111$ said it would. Then switch to Newton and watch the step count drop to 1, from any starting point and any $\\kappa$.</p>

<p><b>The thing genuinely worth noticing.</b> Put the optimiser back to GD, leave $\\eta$ at 0.08, and slide $\\kappa$ from 18 down to 5. The step count does not improve — it stays at 61. That looks like it contradicts everything above, and it is the most instructive thing in the lab. With $\\eta$ held fixed, the shallow direction contracts by $(1 - \\eta \\times 1) = 0.92$ per step regardless of $\\kappa$, so nothing changes; what $\\kappa$ actually controls is <i>how large $\\eta$ is allowed to be</i>. Push $\\kappa$ up to 40 with $\\eta$ still at 0.08 and the run diverges outright, because $0.08 > 2/40$. The honest experiment is to raise $\\kappa$ to 60 and simultaneously drop $\\eta$ to 0.030, just inside the new stability limit of 0.033: now gradient descent needs 166 steps where it needed 61. That is the cost of conditioning, and you can only see it once you stop holding fixed the thing $\\kappa$ is supposed to constrain.</p>

<p>A second surprise in the same lab, worth being honest about: at the default $\\kappa = 18$ and $\\eta = 0.08$, <b>momentum is slower than plain gradient descent</b>, 93 steps against 61. Momentum's advantage is not universal — it helps when the step size is forced to be small relative to the shallow direction's needs, which is the high-$\\kappa$ regime. Run the $\\kappa = 60$, $\\eta = 0.030$ setting and the ordering reverses as it should: gradient descent 166 steps, momentum 100, Adam 96. If a technique appears to help in every configuration you have tried, you probably have not tried the configuration where it does not.</p>

${H.lab('cond', 'One step size, two landscapes', 'Set the condition number and watch gradient descent zig-zag. Then switch to momentum, Adam and Newton on the identical surface — the traces are computed, not drawn.')}

<h2><span class="sn">1.9.5</span> Constrained problems, and the multiplier that is a price</h2>

<p>Not every optimisation is free. PCA asks for the direction of maximum variance, but only among directions of unit length — without that restriction you would simply make the vector longer and claim infinite variance. An SVM minimises a norm subject to every training point being correctly classified with margin. A budget-constrained bidding policy maximises value subject to spend.</p>

<p>The general problem is: minimise $f(x)$ subject to $g(x) = 0$. The trick is to build a single function whose unconstrained stationary points are the constrained solutions — the <b>Lagrangian</b></p>

$$\\mathcal{L}(x, \\lambda) = f(x) - \\lambda\\,g(x)$$

<p>and set all its partial derivatives to zero. Differentiating with respect to $\\lambda$ recovers the constraint $g(x)=0$ exactly, so nothing has been lost. Differentiating with respect to $x$ gives $\\nabla f = \\lambda \\nabla g$: at the solution, <b>the two gradients are parallel</b>.</p>

<p>That condition is worth understanding rather than memorising. $\\nabla g$ points perpendicular to the constraint surface, so any direction you can move <i>along</i> the surface is perpendicular to $\\nabla g$. If $\\nabla f$ had any component along the surface, you could slide that way and improve $f$ while still satisfying the constraint — so you would not be at the optimum. The only way to be stuck is for $\\nabla f$ to point entirely perpendicular to the surface, which is to say parallel to $\\nabla g$. <mark>You are at the constrained optimum exactly when you cannot improve the objective without violating the constraint.</mark></p>

<p>And $\\lambda$ itself is not bookkeeping. It equals $\\partial f^\\star/\\partial c$, the rate at which the optimal value would improve if the constraint were relaxed by one unit — which is why economists call it a <b>shadow price</b>. If the constraint is a budget, $\\lambda$ is what one more pound of budget is worth to you. If nobody would pay for a relaxation, $\\lambda$ is zero and the constraint was not binding.</p>

${H.deriv('PCA, derived in five lines — and why the answer is an eigenvector', [
      ['maximise $w^\\mathsf{T}\\Sigma w$ s.t. $w^\\mathsf{T}w = 1$', 'The variance of the data projected onto direction $w$ is $w^\\mathsf{T}\\Sigma w$ (§1.8.3 showed why this quadratic form is a variance). The constraint pins the length so that "more variance" cannot be achieved by simply scaling $w$ up.'],
      ['$\\mathcal{L} = w^\\mathsf{T}\\Sigma w - \\lambda(w^\\mathsf{T}w - 1)$', 'Form the Lagrangian. The sign convention on $\\lambda$ is arbitrary and affects nothing but the sign of $\\lambda$ itself.'],
      ['$\\nabla_w\\mathcal{L} = 2\\Sigma w - 2\\lambda w = 0$', 'Differentiate using the identity $\\nabla_x(x^\\mathsf{T}Ax) = 2Ax$ for symmetric $A$ from §1.9.1, applied to both terms — the constraint term is the same form with $A = I$.'],
      ['$\\Sigma w = \\lambda w$', 'Divide by 2. This is the eigenvector equation, arrived at from an optimisation problem that never mentioned eigenvectors. <i>Every</i> stationary direction of the projected variance is an eigenvector of the covariance matrix.'],
      ['$w^\\mathsf{T}\\Sigma w = w^\\mathsf{T}(\\lambda w) = \\lambda$', 'Substitute back to evaluate the objective at a stationary point, using $w^\\mathsf{T}w = 1$. The variance captured by an eigen-direction <i>is</i> its eigenvalue, so to maximise variance you take the largest one.']
    ], 'This is the whole of §2.10 in five lines, and it is the cleanest illustration of why §1.8 comes before it. "Find the direction of maximum spread" and "find the top eigenvector of the covariance matrix" are not two related facts. They are the same sentence, one stated as an optimisation and one as an algebraic condition.')}

<p>Inequality constraints, $g(x) \\le 0$, generalise this to the <b>KKT conditions</b>, and one of them earns its keep immediately: <i>complementary slackness</i>, which says $\\lambda_i\\, g_i(x) = 0$ for every constraint. Either the constraint is tight, $g_i = 0$, or its multiplier is zero. Since the multiplier is the price of the constraint, a constraint that is not binding is worth nothing, exactly as you would expect.</p>

<p>That is <i>why</i> only support vectors matter in an SVM (§2.6). Training points comfortably on the correct side of the margin have non-binding constraints, so their multipliers are zero, so they contribute nothing to the solution and could be deleted from the dataset without changing the fitted boundary by a hair. The sparsity of the SVM is not an algorithmic trick; it is complementary slackness showing up in the data.</p>

${H.practice(`<p>Three things go wrong with this material in real work, and none of them are conceptual.</p>
<p><b>You cannot see $\\kappa$.</b> Nobody computes the Hessian of a deep network to check its conditioning. The observable symptoms are what you learn to read: a loss that decreases smoothly then plateaus far above zero, a loss that oscillates at a fixed amplitude, or a run that diverges at a learning rate only slightly above one that worked. All three are conditioning, and the first responses are always the same — normalise the inputs, then check that the learning rate is not sitting just under the stability edge.</p>
<p><b>Convexity claims are about the objective, not the model.</b> Logistic regression is convex in its weights, and stays convex when you add L2. It stops being convex the moment you put a neural network inside it. A candidate who says "logistic regression is convex" without being able to say <i>in what</i> has memorised the sentence.</p>
<p><b>Newton is not a plug-in replacement.</b> Away from a minimum the Hessian of a non-convex loss has negative eigenvalues, and $-H^{-1}g$ then points confidently <i>uphill</i> along those directions. Practical second-order methods must modify the Hessian — trust regions, damping, or adding $\\tau I$ until it is positive definite — and this is why "just use Newton" is not the free win it looks like on a quadratic.</p>`)}

${H.probe([
      ['Why is Newton faster in steps but rarely used at scale?', 'It rescales by curvature so κ stops mattering, but forming and inverting $H$ is $O(p^3)$ in the parameter count.'],
      ['What sets the maximum stable learning rate?', '$2/\\lambda_{\\max}$ for a quadratic — the sharpest curvature direction.'],
      ['Where do you meet Lagrange multipliers in ML?', 'PCA, the SVM dual, and any constrained fairness or budget objective.'],
      ['Your loss oscillates without diverging. What is happening and what do you change?', 'The learning rate is just under $2/\\lambda_{\\max}$, so the sharpest direction alternates sign each step instead of settling. Lower the learning rate, or normalise the inputs to shrink $\\lambda_{\\max}$ — those act on the same quantity from opposite sides.']
    ])}`,
    labs: {
      cond: function (host) {
        const st = Viz.controls(host, [
          { k: 'kappa', label: 'condition number κ', min: 1, max: 60, step: 1, value: 18, fmt: v => v.toFixed(0) },
          { k: 'lr', label: 'learning rate η', min: .005, max: .35, step: .005, value: .08, fmt: v => v.toFixed(3) },
          { k: 'opt', label: 'optimiser', type: 'buttons', value: 'gd', options: [
            { v: 'gd', t: 'GD' }, { v: 'mom', t: 'momentum' }, { v: 'adam', t: 'Adam' }, { v: 'newton', t: 'Newton' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'steps', label: 'steps to converge', cls: 'key' }, { k: 'stable', label: 'max stable η' },
          { k: 'final', label: 'final loss' }, { k: 'rate', label: 'GD rate (κ−1)/(κ+1)' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const a = 1, b = st.kappa;              // f = ½(a x² + b y²)
            const f = (x, y) => .5 * (a * x * x + b * y * y);
            const P = Viz.plot(ctx, w, h, { xd: [-2.6, 2.6], yd: [-1.5, 1.5] })
              .frame({ xlabel: 'shallow direction', ylabel: 'steep direction' });
            P.contours(f, [.05, .2, .5, 1, 2, 4, 8, 16], { color: T.faint, alpha: .55 });
            let x = -2.2, y = 1.2, vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0;
            const path = [[x, y]];
            let steps = 0;
            for (let t = 1; t <= 220; t++) {
              const gx = a * x, gy = b * y;
              if (st.opt === 'gd') { x -= st.lr * gx; y -= st.lr * gy; }
              else if (st.opt === 'mom') { vx = .9 * vx - st.lr * gx; vy = .9 * vy - st.lr * gy; x += vx; y += vy; }
              else if (st.opt === 'adam') {
                mx = .9 * mx + .1 * gx; my = .9 * my + .1 * gy;
                sx = .999 * sx + .001 * gx * gx; sy = .999 * sy + .001 * gy * gy;
                const mhx = mx / (1 - Math.pow(.9, t)), mhy = my / (1 - Math.pow(.9, t));
                const shx = sx / (1 - Math.pow(.999, t)), shy = sy / (1 - Math.pow(.999, t));
                x -= (st.lr * 4) * mhx / (Math.sqrt(shx) + 1e-8);
                y -= (st.lr * 4) * mhy / (Math.sqrt(shy) + 1e-8);
              } else { x -= gx / a; y -= gy / b; }
              if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 8 || Math.abs(y) > 8) { path.push([Math.sign(x) * 8, Math.sign(y) * 8]); steps = -1; break; }
              path.push([x, y]);
              if (steps === 0 && f(x, y) < 1e-4) steps = t;
            }
            P.clip(() => {
              P.line(path, { color: T.blue, width: 1.8 });
              P.dots(path.filter((_, i) => i % 4 === 0), { r: 2.4, color: T.blue, alpha: .8 });
              P.dots([[0, 0]], { r: 5, color: T.red, stroke: true });
              P.dots([path[0]], { r: 5, color: T.green, stroke: true });
            });
            out({
              steps: steps === -1 ? 'diverged' : (steps || '> 220'),
              stable: (2 / st.kappa).toFixed(3),
              final: f(path[path.length - 1][0], path[path.length - 1][1]).toExponential(1),
              rate: ((st.kappa - 1) / (st.kappa + 1)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'Raise κ with GD selected and watch the zig-zag: the step size is capped by the steep direction while all the progress is needed in the shallow one. Momentum cancels the alternating component; Adam rescales per coordinate; Newton divides by the curvature exactly and lands in one step.');
      }
    },
    quiz: [
      {
        q: 'Gradient descent on a quadratic with κ = 100 is slow. Which explanation is precise?',
        options: ['The loss is non-convex', 'η is capped by 2/λmax while progress is needed along λmin, so the rate degrades like (κ−1)/(κ+1)', 'The gradient is zero', 'The Hessian is singular'],
        answer: 1,
        why: 'With $\\kappa=100$ the step size is capped at $2/\\lambda_{\\max}$ by stability in the sharpest direction, while all the remaining progress is needed along the shallowest, and the per-step contraction works out to $99/101=0.980$ — about 461 steps to gain four digits where a well-conditioned problem needs 46. "The loss is non-convex" is the tempting answer because slowness is popularly blamed on landscape pathology, but a quadratic with $\\kappa=100$ is perfectly convex with a unique global minimum; it is simply badly proportioned. This is why feature scaling, momentum, Adam and Newton all belong on the same list in §1.9.4: they are four different attacks on the same number.'
      },
      {
        q: 'XGBoost’s split gain uses second derivatives because…',
        options: ['it is faster', 'it is a Newton step in function space: curvature gives better leaf values and a loss-agnostic gain formula', 'first derivatives are unavailable', 'it prevents overfitting by itself'],
        answer: 1,
        why: 'Taylor-expanding the loss to second order around the current predictions and minimising the resulting quadratic is precisely a Newton step, and §2.8 derives the leaf weight $-G/(H+\\lambda)$ from it — a gradient divided by a curvature, which is $-H^{-1}g$ for a one-dimensional parameter. "It is faster" is tempting because second-order methods are associated with speed, but computing per-example second derivatives is extra work per split, not less; the payoff is better leaf values and a gain formula that works for any twice-differentiable loss without rederivation. Recognising this formula as Newton rather than as an arbitrary boosting rule is what §1.9.3 is for.'
      },
      {
        q: 'Maximising $w^\\mathsf{T}\\Sigma w$ subject to $\\|w\\| = 1$ leads to which stationarity condition?',
        options: ['$\\Sigma w = \\lambda w$, so $w$ is an eigenvector and the maximum value is the largest eigenvalue', '$w = \\Sigma^{-1}\\mathbf{1}$, the inverse covariance applied to a vector of ones', '$w$ must equal the mean of the data', 'The problem is unbounded, so no solution exists'],
        answer: 0,
        why: 'Forming the Lagrangian $w^\\mathsf{T}\\Sigma w - \\lambda(w^\\mathsf{T}w - 1)$ and differentiating gives $2\\Sigma w - 2\\lambda w = 0$, which is the eigenvector equation — and substituting back shows the objective equals $\\lambda$ at any stationary point, so the maximum is the largest eigenvalue. The last option is tempting because the objective genuinely <i>is</i> unbounded without the constraint: you could scale $w$ up forever. That is precisely why the constraint is there, and it is a good illustration of the multiplier as a price, since $\\lambda$ measures how much the objective would improve if the unit-norm restriction were loosened. This five-line derivation is the whole of PCA (§2.10), which is why it is worth being able to produce it rather than recognise it.'
      }
    ],
    cards: [
      { q: 'Convexity test', a: 'Twice-differentiable $f$ is convex iff its Hessian is PSD everywhere; then no bad local minima.' },
      { q: 'Condition number’s two consequences', a: 'Max stable step $\\eta<2/\\lambda_{\\max}$ and convergence rate $(\\kappa-1)/(\\kappa+1)$ per step.' },
      { q: 'Lagrange multiplier, in words', a: 'The price of the constraint — at the optimum the gradients of objective and constraint are parallel.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.10 */
  ML.section({
    id: 'information', track: 'foundations', num: '1.10',
    title: 'Information theory: entropy, cross-entropy, KL, perplexity',
    lede: 'Feeds cross-entropy (§1.5, §2.4), tree splits (§2.7), mutual-information feature screening (§2.11), and the KL penalties in RLHF and DPO (§4.12).',
    html: `
<p>Here is a game with a number in it. I have picked tomorrow's weather in a city that only ever does four things, and you have to work out which by asking yes/no questions. How few questions do you need?</p>

<p>If the four outcomes were equally likely, the answer is clearly two. Ask "is it one of the first two?", then ask which of the remaining pair — two questions, always, and there is no cleverer scheme. Four options, two questions, because $2^2 = 4$.</p>

<p>But weather is not uniform. Suppose the long-run frequencies in this city are</p>

${H.code(`sunny   1/2
cloudy  1/4
rain    1/8
snow    1/8`)}

<p>Now two questions is wasteful, because half the time the answer is "sunny" and you are spending a second question on a case you could have finished in one. Ask instead: <i>is it sunny?</i> Half the time you are done in one question. If not, ask <i>is it cloudy?</i> — a quarter of the time you are done in two. Otherwise ask <i>is it rain?</i>, which settles the last two cases in three questions each. The average number of questions is</p>

$$\\tfrac12 \\times 1 + \\tfrac14 \\times 2 + \\tfrac18 \\times 3 + \\tfrac18 \\times 3 = 0.5 + 0.5 + 0.375 + 0.375 = 1.75$$

<p>One and three-quarter questions on average, against two for the naive scheme. And no scheme does better — 1.75 is a floor imposed by the probabilities themselves, not by your cleverness at designing questions.</p>

<p>Notice what each outcome cost. Sunny took 1 question and has probability $\\tfrac12$. Cloudy took 2 and has probability $\\tfrac14$. Rain and snow took 3 each and have probability $\\tfrac18$. In every case the number of questions is $\\log_2(1/p)$: $\\log_2 2 = 1$, $\\log_2 4 = 2$, $\\log_2 8 = 3$. Rarer outcomes cost more to specify, and they cost exactly the logarithm of their rarity.</p>

<p>That observation is the whole of information theory, and everything else in this section is bookkeeping on top of it.</p>

<h2><span class="sn">1.10.1</span> Entropy: the floor on how well you can do</h2>

<p>Define the <b>surprise</b> of an outcome with probability $p$ as $\\log(1/p) = -\\log p$. It is large for rare events, zero for certain ones, and — this is why the logarithm rather than some other decreasing function — it is <b>additive</b> for independent events. Two independent outcomes of probability $\\tfrac12$ each have joint probability $\\tfrac14$, and their surprises add: $1 + 1 = 2$ bits. Information from separate sources should accumulate, and the logarithm is what makes it do so.</p>

<p><b>Entropy</b> is the average surprise:</p>

$$H(p) = -\\sum_i p_i \\log p_i$$

<p>Read $\\sum$ as "add up over every outcome $i$" (§0.2 introduced the symbol) and read $H(p)$ as "the entropy of the distribution $p$". Each term weights an outcome's surprise by how often it happens. With $\\log_2$ the unit is <b>bits</b>, and the answer is the average number of yes/no questions in the optimal questioning scheme; with natural $\\log$ the unit is <b>nats</b>, which is what machine-learning code uses because $\\ln$ differentiates more cleanly. One nat is $1/\\ln 2 \\approx 1.443$ bits, and the choice changes nothing except a constant factor.</p>

<p>Check the definition against the weather: $-(\\tfrac12\\log_2\\tfrac12 + \\tfrac14\\log_2\\tfrac14 + \\tfrac18\\log_2\\tfrac18 + \\tfrac18\\log_2\\tfrac18) = \\tfrac12(1) + \\tfrac14(2) + \\tfrac18(3) + \\tfrac18(3) = 1.75$ bits. The formula reproduces the questioning game exactly, which is the point of having introduced the game first.</p>

${H.analogy(`<p>Entropy is the length of the shortest honest description, averaged over what you are describing.</p>
<p>If you had to phone a colleague every morning with the weather and pay by the second, you would invent abbreviations: one grunt for sunny, because you say it constantly, and a longer phrase for snow, because you almost never do. Every efficient code that has ever existed does this — Morse code gives E a single dot and Q four symbols, because English uses E about sixty times as often.</p>
<p>Entropy is the statement that this trade has a hard limit. Given the frequencies, there is a shortest possible average message length, you cannot beat it, and you can get arbitrarily close to it. A distribution that is nearly certain is nearly free to describe; a uniform one is maximally expensive. Compression and prediction are the same problem seen from two ends, which is why a language model is, quite literally, a compressor.</p>`)}

<p>Two properties follow immediately and are worth being able to state. Entropy is maximised by the uniform distribution — with $k$ equally likely outcomes it is exactly $\\log_2 k$ bits, and nothing can exceed that. And it is minimised, at zero, by a distribution that puts all its mass on one outcome, because there is nothing left to communicate. Entropy measures uncertainty, and this is why the same quantity shows up as a splitting criterion in decision trees (§2.7): a split is good when the child nodes have lower entropy than the parent, meaning you know more about the label after the split than before.</p>

<h2><span class="sn">1.10.2</span> Cross-entropy: what it costs to use the wrong code</h2>

<p>Now suppose you did not know the weather frequencies. You assumed all four outcomes were equally likely and built the two-question scheme accordingly, and then reality delivered the actual distribution. Your average cost is 2 bits per day, against the 1.75 you could have achieved. You are paying 0.25 bits a day for being wrong about the world.</p>

<p>Those three numbers have names. Write $p$ for the true distribution and $q$ for the distribution you assumed. Then:</p>

$$H(p) = -\\sum_i p_i \\log p_i \\qquad\\text{(the floor: your code matches reality)}$$
$$H(p,q) = -\\sum_i p_i \\log q_i \\qquad\\text{(cross-entropy: a code built for } q\\text{, paying for draws from } p)$$
$$D_{KL}(p\\,\\|\\,q) = \\sum_i p_i \\log\\frac{p_i}{q_i} \\qquad\\text{(the excess you pay for being wrong)}$$

<p>Read $H(p,q)$ as "the cross-entropy of $p$ relative to $q$" and $D_{KL}(p\\|q)$ as "the KL divergence from $q$ to $p$", also called relative entropy. Look at the structure of each: all three are averages over the true distribution $p$ — the $p_i$ out front never changes — and what varies is what is inside the logarithm. Entropy scores you against your own probabilities, cross-entropy scores you against someone else's, and KL is the gap.</p>

${H.deriv('the identity to have cold: $H(p,q) = H(p) + D_{KL}(p\\|q)$', [
      ['$D_{KL}(p\\|q) = \\sum_i p_i \\log\\dfrac{p_i}{q_i}$', 'Start from the definition of the divergence.'],
      ['$= \\sum_i p_i\\big(\\log p_i - \\log q_i\\big)$', 'The logarithm of a ratio is the difference of logarithms — the identity that makes this whole subject work, and the reason logs are used rather than raw probabilities.'],
      ['$= \\sum_i p_i \\log p_i \\;-\\; \\sum_i p_i \\log q_i$', 'Split the sum into two sums. This is legitimate because both pieces converge; the standard conventions handle the $p_i = 0$ terms, where $0\\log 0$ is taken to be 0.'],
      ['$= -H(p) + H(p,q)$', 'Recognise both pieces. The first sum is the negative of entropy and the second is cross-entropy, straight from their definitions.'],
      ['$H(p,q) = H(p) + D_{KL}(p\\|q)$', 'Rearrange. Total cost equals the unavoidable floor plus the penalty for being wrong.']
    ], 'Since $D_{KL} \\ge 0$ always, with equality only when $q = p$ exactly, cross-entropy can never fall below entropy. Check it on the weather: $H(p,q) = 2$ bits with a uniform $q$, $H(p) = 1.75$, so $D_{KL} = 0.25$ bits — the excess computed by hand at the top of this subsection.')}

<p>Now the payoff, and the reason this section exists inside a machine-learning course.</p>

${H.key('Minimising cross-entropy is minimising KL divergence to the true distribution. $H(p)$ does not contain your parameters, so it is a constant you cannot affect — driving the loss down is driving your model toward the truth, in this precise sense.')}

<h3>Why the loss function is called cross-entropy, which is not a coincidence</h3>

<p>Take one training example whose true label is class 3 out of five. Its "true distribution" is one-hot: $p = (0,0,1,0,0)$, all the mass on the right answer. Your model outputs $q = (q_1,\\ldots,q_5)$. Compute the cross-entropy:</p>

$$H(p,q) = -\\sum_i p_i \\log q_i = -\\log q_3$$

<p>because every term with $p_i = 0$ vanishes and the one surviving term has $p_i = 1$. That is exactly the familiar classification loss — the negative log of the probability the model assigned to the correct class. Average it over the dataset and you have the training objective used by essentially every classifier on this site.</p>

<p>Notice also that for a one-hot $p$, $H(p) = 0$: there is no uncertainty in a label you have been told. So for classification, cross-entropy <i>equals</i> the KL divergence, with no constant to subtract. And §1.5 arrived at the identical formula from a completely different direction — as the negative log-likelihood of a Categorical distribution. Three names, one function: <b>NLL of a categorical, cross-entropy, and KL to the empirical distribution are the same object.</b> If an interviewer asks why cross-entropy is the loss for classification, either derivation is a correct answer and having both is better.</p>

<h2><span class="sn">1.10.3</span> KL is not symmetric, and the direction changes the answer</h2>

<p>$D_{KL}(p\\|q)$ and $D_{KL}(q\\|p)$ are different numbers. That is not a technicality; it is the source of a behaviour you will meet in variational inference, in distillation, and in the KL penalties that keep RLHF and DPO from wandering (§4.12).</p>

${H.worked('the asymmetry, on two-outcome distributions you can check by hand', `<p>Let $p = (0.5,\\, 0.5)$, a fair coin, and $q = (0.9,\\, 0.1)$, a heavily biased one. Working in bits:</p>
<p>$D_{KL}(p\\|q) = 0.5\\log_2\\frac{0.5}{0.9} + 0.5\\log_2\\frac{0.5}{0.1} = 0.5(-0.848) + 0.5(2.322) = 0.737$ bits.</p>
<p>$D_{KL}(q\\|p) = 0.9\\log_2\\frac{0.9}{0.5} + 0.1\\log_2\\frac{0.1}{0.5} = 0.9(0.848) + 0.1(-2.322) = 0.531$ bits.</p>
<p>Same two distributions, two different answers, and neither is a distance in the geometric sense — KL also fails the triangle inequality, so it is a divergence rather than a metric.</p>
<p>The asymmetry has a readable cause. Each divergence is an average weighted by its <i>first</i> argument, so it only penalises you where that first distribution puts mass. $D_{KL}(p\\|q)$ takes the fair coin's view and is horrified by $q$'s tiny 0.1 on an outcome that happens half the time. $D_{KL}(q\\|p)$ takes the biased coin's view, and $p$ never assigns a small probability where $q$ assigns a large one, so it is less alarmed.</p>`)}

<p>Generalise that last paragraph and you have the standard vocabulary:</p>

<ul>
<li><b>Forward KL</b>, $D_{KL}(p\\|q)$, averages over the truth. Wherever $p$ has mass and $q$ does not, $\\log(p/q)$ blows up, so the penalty for missing part of the truth is severe. A model fitted this way <b>spreads out to cover everything</b>, at the cost of putting mass in places nothing ever happens. Mode-covering. This is the direction maximum-likelihood training uses, which is why generative models trained by likelihood tend to be blurry rather than incomplete.</li>
<li><b>Reverse KL</b>, $D_{KL}(q\\|p)$, averages over the model. Now the damage is done by putting mass where $p$ is near zero, so the model <b>retreats into a region it is confident about</b> and simply ignores parts of the truth. Mode-seeking. This is the direction variational inference minimises, which is exactly why a variational posterior is famously over-confident and under-dispersed.</li>
</ul>

<p><b>What you are looking at.</b> The blue filled curve is the target $p$, a mixture of two Gaussians whose separation you control. The dashed red curve is $q$, a single Gaussian whose mean and standard deviation are the two sliders — a family that cannot possibly represent $p$ exactly, which is what makes the question "what is the best compromise?" meaningful. The readout reports forward KL, reverse KL, cross-entropy and the entropy of $p$, all computed by numerical integration over the grid, in nats. The two buttons run a grid search over the mean and standard deviation to find the $q$ that minimises each divergence, then move the sliders there.</p>

<p><b>What to do with it.</b> <b>First push the separation slider up to about 5</b>, then press the two fit buttons in turn. Minimising forward KL parks $q$ at mean 0 with a standard deviation of about 2.65 — a single wide bump straddling both modes, fitting neither well but covering everything. Minimising reverse KL abandons the middle and collapses onto one mode, at a mean near $\\pm 2.5$ with a standard deviation of about 0.85. That is mode-covering and mode-seeking, side by side, on identical data.</p>

<p><b>The thing genuinely worth noticing.</b> Now drop the separation back to its default of 3.4 and press the reverse-KL button again. It does <i>not</i> collapse: the best single Gaussian under reverse KL is still centred at 0, with a standard deviation of 1.75 against the forward fit's 1.85. The two fits are nearly identical, and the famous mode-seeking behaviour has simply not appeared. Mode collapse is not automatic — it wins only when the modes are far enough apart that the valley between them is genuinely low-density, and here the crossover sits between separations of 4 and 4.5. There is a clean number underneath this: a $q$ that fits one mode of a 50/50 mixture perfectly pays a reverse KL of $\\ln 2 \\approx 0.693$ nats, the cost of ignoring half the mass, and that ceiling does not grow with separation — whereas the straddling solution's reverse KL keeps climbing as the modes separate, reaching 2.25 nats at a separation of 6. Collapse wins when the climbing curve crosses the flat one. Any claim about KL direction that does not mention how separated the modes are is only half a claim.</p>

${H.lab('kl', 'Forward vs reverse KL, and why the direction matters', 'A bimodal target p and a single Gaussian q you control. Push the separation slider up before pressing the fit buttons: forward KL spreads to cover both modes, and reverse KL collapses onto one — but only once the modes are far enough apart.')}

<h2><span class="sn">1.10.4</span> Mutual information: dependence of any shape</h2>

<p>Correlation measures whether two variables move up and down together, and it is blind to any relationship that is not roughly linear. Mutual information asks a broader question: <b>how much does knowing one variable reduce your uncertainty about the other?</b></p>

$$I(X;Y) = H(X) - H(X\\mid Y) = D_{KL}\\big(p(x,y)\\,\\|\\,p(x)p(y)\\big)$$

<p>The first form reads directly: the uncertainty you had about $X$, minus the uncertainty remaining once $Y$ is known. The second says the same thing differently — it is the KL divergence between how the two variables actually behave together and how they would behave if they were independent. So $I = 0$ exactly under independence, and any dependence at all makes it positive. It is also symmetric, $I(X;Y) = I(Y;X)$, which the first form does not make obvious and the second does.</p>

${H.worked('zero correlation, one full bit of mutual information', `<p>Let $X$ be uniform on $\\{-2, -1, 1, 2\\}$ and let $Y = X^2$, so $Y$ is 4 or 1, each with probability one half. $Y$ is a deterministic function of $X$ — you could hardly ask for a stronger dependence.</p>
<p>The correlation is exactly zero. $\\mathbb{E}[X] = 0$ by symmetry, and $\\mathbb{E}[XY] = \\mathbb{E}[X^3] = \\frac{(-8) + (-1) + 1 + 8}{4} = 0$, so the covariance vanishes. A correlation-based feature screen would discard $X$ as useless for predicting $Y$.</p>
<p>Now the information calculation. $H(X) = \\log_2 4 = 2$ bits, since $X$ has four equally likely values. Given $Y = 4$, $X$ is either $-2$ or $2$ with equal probability, so $H(X \\mid Y{=}4) = 1$ bit, and the same for $Y = 1$. Hence $H(X\\mid Y) = 1$ bit and</p>
<p>$I(X;Y) = 2 - 1 = 1$ bit.</p>
<p>Knowing $Y$ halves the number of possibilities for $X$, which is exactly one bit's worth of information, and correlation reported none of it. This is why mutual information is the right screen for non-monotonic feature relationships (§2.11) — and also why it is expensive: estimating it from continuous data requires binning or a density estimate, both of which are unreliable in high dimensions and biased upward on small samples.</p>`)}

<h2><span class="sn">1.10.5</span> Perplexity is entropy wearing a different hat</h2>

$$\\mathrm{PPL} = \\exp(H) = \\exp\\!\\left(-\\frac{1}{N}\\sum_i \\log p(x_i)\\right)$$

<p>Take the average negative log-likelihood per token — which is cross-entropy, in nats, against the observed data — and exponentiate it. That is all perplexity is, and the exponential exists purely to put the number on a scale people can reason about.</p>

<p>That scale is the <b>effective branching factor</b>. If a model were uniformly uncertain among $k$ options at every step, its entropy would be $\\log k$ and its perplexity exactly $k$. So a perplexity of 20 means the model is exactly as uncertain as someone choosing uniformly among 20 possibilities each time, whatever its actual distribution looks like. Halving perplexity is a large improvement; shaving 2 per cent off it is a rounding error dressed up as progress.</p>

<p>Two consequences worth carrying. Perplexity is comparable only between models sharing a <b>tokenizer</b> (§4.2), because the per-token denominator changes when the segmentation changes — a model that splits words into more pieces gets an easier prediction task per piece and a flatteringly low number. And perplexity measures compression, not usefulness: a model can have lower perplexity and be worse at your task, which is why §4.16 insists on task evaluations rather than a leaderboard of loss values.</p>

<p><b>What you are looking at.</b> Five candidate next tokens, with two bars each: blue for the true distribution $p$, which is fixed at $(0.5, 0.2, 0.15, 0.1, 0.05)$, and red for the model's prediction $q$. The two controls reshape $q$ only. "Sharpness" raises the model's logits to a power, so above 1 it makes $q$ more confident than $p$ and below 1 flattens it toward uniform; "error" moves probability mass from the top token to the last one, simulating a model that backs the wrong horse. The readouts are entropy, cross-entropy, KL and perplexity, in bits.</p>

<p><b>What to do with it.</b> Start at the defaults: sharpness 1, error 0. Here $q$ equals $p$ exactly, and the readout shows $H(p) = 1.923$ bits, cross-entropy also 1.923, KL zero, and perplexity $2^{1.923} = 3.79$. Now drag the error slider to 20 per cent: cross-entropy rises to 2.005 bits, KL to 0.082, and perplexity to 4.01. The model got worse and every quantity noticed.</p>

<p><b>The thing genuinely worth noticing.</b> Put the error back to zero and raise sharpness to 2 instead. The model is now <i>more confident about exactly the right ranking</i> — it still puts the most mass on the true top token, more than before — and cross-entropy rises anyway, to 2.225 bits, with perplexity at 4.67. Push sharpness to 5 and cross-entropy reaches 4.635 bits and perplexity 24.8, an appalling score for a model whose ordering is perfect. Cross-entropy is a <b>proper scoring rule</b>: it is minimised only by reporting your true beliefs, and it punishes confidence that is not earned far more harshly than it punishes admitted uncertainty. That property is why cross-entropy trains calibrated probabilities (§2.12), and it is also why accuracy and cross-entropy can rank two models in opposite orders without either metric being broken.</p>

${H.lab('ppl', 'Entropy, cross-entropy, KL and perplexity on a real distribution', 'Move the predicted distribution away from the truth and watch every quantity respond. The perplexity readout is the same number a language-model training log prints.')}

<h2><span class="sn">1.10.6</span> The log-sum-exp trick</h2>

<p>Everything above assumes you can actually compute these quantities in floating point, and there is one place where the naive implementation fails catastrophically. It appears inside every softmax, interviewers like it, and it takes one line to fix.</p>

<p>The quantity you need is $\\log\\sum_i e^{z_i}$, the normaliser of a softmax. In IEEE double precision the largest representable number is about $1.8 \\times 10^{308}$, so $e^{z}$ overflows to infinity once $z$ exceeds about 709.78. Logits routinely get large during training, and once one term is infinite the sum is infinite, the softmax computes $\\infty/\\infty$, and the result is <code>NaN</code> — which then propagates through every subsequent gradient and kills the run.</p>

<p>The fix is to pull out the largest exponent first:</p>

$$\\log\\sum_i e^{z_i} = m + \\log\\sum_i e^{z_i - m}, \\qquad m = \\max_i z_i$$

${H.deriv('why the shift is exact rather than an approximation', [
      ['$\\sum_i e^{z_i} = \\sum_i e^{m}\\,e^{z_i - m}$', 'Multiply and divide each term by $e^m$, using $e^{a} = e^{m}e^{a-m}$. Nothing has been approximated; this is the exponential law applied term by term.'],
      ['$= e^{m}\\sum_i e^{z_i - m}$', 'The factor $e^m$ does not depend on $i$, so it comes outside the sum.'],
      ['$\\log\\sum_i e^{z_i} = m + \\log\\sum_i e^{z_i-m}$', 'Take logarithms, and the product becomes a sum. The identity is exact for any $m$ whatsoever — choosing the maximum is what makes it numerically safe rather than what makes it true.'],
      ['$z_i - m \\le 0 \\;\\Rightarrow\\; e^{z_i - m} \\in (0, 1]$', 'With $m$ the maximum, every shifted exponent is at most zero, so every term is between 0 and 1 and none can overflow. The largest term is exactly 1, so the sum is at least 1 and its logarithm is well defined — underflow of the small terms is harmless because they were negligible anyway.']
    ], 'The same identity is what makes FlashAttention’s <b>online softmax</b> (§4.7) possible. Partial sums computed on different tiles of the attention matrix carry different running maxima, and the identity lets you merge them by rescaling with the new maximum — which is why the full $s \\times s$ score matrix never has to exist in memory.')}

<p><b>What you are looking at.</b> Five lines of a live calculation on four logits whose relative pattern is fixed at $(1, 0.6, 0.2, -0.4)$ and whose overall magnitude is your single control. The lines show the logits themselves, the naive sum of exponentials, the stabilised log-sum-exp, and the softmax probability of the first token computed both ways. Values that have failed are drawn in red, working ones in green.</p>

<p><b>What to do with it.</b> Start at magnitude 20 and both methods agree: the naive sum is about $4.85 \\times 10^8$ and both routes give $p_1 = 0.9997$. Push the slider up gradually. At 700 the naive sum is about $1.0 \\times 10^{304}$ — enormous, and still finite. At 710 it becomes <code>Infinity</code>, the naive softmax reads <code>NaN</code>, and the stabilised version continues to report $p_1 = 1.000000$ without noticing anything happened.</p>

<p><b>The thing genuinely worth noticing.</b> The failure has no warning zone. At 709 everything is correct to full precision; at 710 the output is not slightly wrong but meaningless, and in a real training loop the <code>NaN</code> spreads to every parameter within one backward pass and the loss curve simply stops. This is the shape of most numerical failures (§1.15): they are cliffs, not slopes. It is also the concrete reason the frameworks want <i>logits</i> rather than probabilities — <code>CrossEntropyLoss</code> and <code>BCEWithLogitsLoss</code> take raw scores precisely so they can apply this shift internally, and hand-rolling <code>log(softmax(x))</code> throws that protection away.</p>

${H.lab('lse', 'Overflow, and the one-line fix', 'Push the logits up and watch the naive computation return Infinity while the stabilised version keeps returning the right answer. The shift cancels exactly — this is not an approximation.')}

${H.history(`<p>All of this comes from one paper: Claude Shannon's <i>A Mathematical Theory of Communication</i>, published in 1948 while he was at Bell Labs, which invented the field essentially complete. The motivating problem was not learning at all — it was telephony and telegraphy, where you are paid to push messages down an expensive wire and want to know how few symbols you can get away with.</p>
<p>Shannon's move was to argue that the meaning of a message is irrelevant to how many symbols it needs, and that only the statistics of the source matter. That is what makes the theory transplant so cleanly into machine learning: a model that predicts the next token well is, by exactly Shannon's argument, a model that could compress the text well, and the cross-entropy loss printed in your training log is the number of nats per token a compressor built from your model would spend. The name "bit", incidentally, appears in print for the first time in that paper, credited by Shannon to his colleague John Tukey.</p>`)}

${H.practice(`<p>Two habits that follow directly from this material.</p>
<p><b>Label smoothing</b> replaces the one-hot target with something like $(0.9$ on the true class, the remaining $0.1$ spread over the rest$)$. In the language of this section, you have stopped claiming the true distribution is a point mass. Since cross-entropy with a one-hot target is minimised only by putting <i>all</i> the probability on the true class — which requires an infinite logit — the un-smoothed objective is quietly pushing your model toward overconfidence forever. Smoothing gives it a finite target to reach and usually improves calibration, at a small cost in raw accuracy.</p>
<p><b>Report the loss in nats per token and know what a good value is.</b> A cross-entropy of 0.693 nats on a binary problem is exactly $\\ln 2$, which is what you get from predicting 50/50 every time — that is your baseline, and a model above it is worse than a coin. For a $k$-class problem the equivalent floor is $\\ln k$. Being able to say "our loss is 1.9 nats and uniform guessing would be 2.3" turns an uninterpretable number into a statement about whether the model has learned anything.</p>`)}

${H.probe([
      ['Cross-entropy vs KL?', '$H(p,q)=H(p)+D_{KL}(p\\|q)$; minimising one minimises the other since $H(p)$ is fixed.'],
      ['Why is KL asymmetric and does it matter?', 'Forward KL is mode-covering (it punishes $q\\approx0$ where $p>0$); reverse KL is mode-seeking. It decides how a variational or distilled model fails.'],
      ['What is perplexity?', 'exp of mean NLL per token — an effective branching factor, comparable only within a fixed tokenizer.'],
      ['Why is cross-entropy the loss for classification?', 'Two equivalent answers: it is the negative log-likelihood of a Categorical label (§1.5), and it is the KL divergence from the model to the one-hot empirical distribution, whose entropy is zero.']
    ])}`,
    labs: {
      kl: function (host) {
        const st = Viz.controls(host, [
          { k: 'mu', label: 'q mean μ', min: -4, max: 4, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'sd', label: 'q sd σ', min: .2, max: 3.5, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'sep', label: 'separation of p’s modes', min: 1, max: 6, step: .1, value: 3.4, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'fwd', label: 'forward KL(p‖q)', cls: 'key' }, { k: 'rev', label: 'reverse KL(q‖p)' },
          { k: 'ce', label: 'cross-entropy H(p,q)' }, { k: 'hp', label: 'entropy H(p)' }
        ]);
        const grid = [];
        for (let x = -9; x <= 9; x += .02) grid.push(x);
        function pdfP(x, sep) { return .5 * Num.normPdf(x, -sep / 2, .8) + .5 * Num.normPdf(x, sep / 2, .8); }
        function kls(mu, sd, sep) {
          let fwd = 0, rev = 0, ce = 0, hp = 0;
          grid.forEach(x => {
            const p = pdfP(x, sep), q = Math.max(1e-12, Num.normPdf(x, mu, sd));
            const dx = .02;
            if (p > 1e-12) { fwd += p * Math.log(p / q) * dx; ce += -p * Math.log(q) * dx; hp += -p * Math.log(p) * dx; }
            rev += q * Math.log(q / Math.max(1e-12, p)) * dx;
          });
          return { fwd, rev, ce, hp };
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-7, 7], yd: [0, .42] }).frame({ xlabel: 'x', ylabel: 'density' });
            P.clip(() => {
              const pPts = grid.map(x => [x, pdfP(x, st.sep)]);
              const qPts = grid.map(x => [x, Num.normPdf(x, st.mu, st.sd)]);
              P.area(pPts, { color: T.blue, alpha: .14 });
              P.line(pPts, { color: T.blue, width: 2.4 });
              P.line(qPts, { color: T.red, width: 2.4, dash: [6, 4] });
            });
            const r = kls(st.mu, st.sd, st.sep);
            out({ fwd: r.fwd.toFixed(3), rev: r.rev.toFixed(3), ce: r.ce.toFixed(3), hp: r.hp.toFixed(3) });
          }
        });
        Viz.buttons(host, [
          { label: 'Minimise forward KL(p‖q)', primary: true, on: () => { opt('fwd'); } },
          { label: 'Minimise reverse KL(q‖p)', on: () => { opt('rev'); } }
        ]);
        function opt(which) {
          let bestMu = 0, bestSd = 1, best = 1e9;
          for (let mu = -3; mu <= 3.01; mu += .1) for (let sd = .25; sd <= 3.2; sd += .1) {
            const v = kls(mu, sd, st.sep)[which];
            if (v < best) { best = v; bestMu = mu; bestSd = sd; }
          }
          st.$set('mu', +bestMu.toFixed(2)); st.$set('sd', +bestSd.toFixed(2)); S.redraw();
        }
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'p — the true bimodal distribution' }, { c: Viz.theme().red, t: 'q — your single Gaussian' }]);
        Viz.note(host, 'Forward KL punishes q ≈ 0 wherever p > 0, so it spreads to cover both modes (blurry but safe). Reverse KL punishes q > 0 where p ≈ 0, so it collapses onto one mode (sharp but incomplete). This is the same trade-off that appears in variational inference and in distillation.');
      },

      ppl: function (host) {
        const words = ['the', 'model', 'predicts', 'a', 'token'];
        const st = Viz.controls(host, [
          { k: 'temp', label: 'model sharpness (1/T)', min: .2, max: 5, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'shift', label: 'model’s error (mass moved to the wrong token)', min: 0, max: .8, step: .01, value: 0, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'h', label: 'H(p) bits' }, { k: 'ce', label: 'H(p,q) bits', cls: 'key' },
          { k: 'kl', label: 'KL(p‖q) bits' }, { k: 'ppl', label: 'perplexity' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const pTrue = [.5, .2, .15, .1, .05];
            let logits = pTrue.map(v => Math.log(v));
            let q = Num.softmax(logits.map(v => v * st.temp));
            // move mass from the top token to the last one
            const moved = q[0] * st.shift;
            q = q.slice(); q[0] -= moved; q[4] += moved;
            const P = Viz.plot(ctx, w, h, { xd: [-.5, 4.5], yd: [0, .8], pad: { l: 44, r: 14, t: 14, b: 42 } })
              .frame({ xticks: [], ylabel: 'probability' });
            pTrue.forEach((v, i) => {
              const x0 = P.x(i - .34), x1 = P.x(i - .02);
              ctx.fillStyle = T.blue; ctx.fillRect(x0, P.y(v), x1 - x0, P.y(0) - P.y(v));
              const x2 = P.x(i + .02), x3 = P.x(i + .34);
              ctx.fillStyle = T.red; ctx.fillRect(x2, P.y(q[i]), x3 - x2, P.y(0) - P.y(q[i]));
              ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText('"' + words[i] + '"', P.x(i), P.y(0) + 6);
            });
            const H = Num.entropy(pTrue);
            const CE = -Num.sum(pTrue.map((v, i) => v * Math.log2(Math.max(1e-12, q[i]))));
            out({
              h: H.toFixed(3), ce: CE.toFixed(3), kl: (CE - H).toFixed(3),
              ppl: Math.pow(2, CE).toFixed(2)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'p — true next-token distribution' }, { c: Viz.theme().red, t: 'q — the model’s prediction' }]);
        Viz.note(host, 'Cross-entropy never drops below the entropy of the truth — that floor is the irreducible uncertainty of language, and it is why a perplexity of 1 is not a target.');
      },

      lse: function (host) {
        const st = Viz.controls(host, [
          { k: 'scale', label: 'logit magnitude', min: 1, max: 1200, step: 1, value: 20, fmt: v => v.toFixed(0) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'naive', label: 'naive Σeᶻ', cls: 'bad' }, { k: 'lse', label: 'stabilised log Σeᶻ', cls: 'good' },
          { k: 'p0', label: 'softmax p₁ (naive)' }, { k: 'p0s', label: 'softmax p₁ (stable)' }
        ]);
        const S = Viz.surface(host, {
          height: 200,
          draw: function (ctx, w, h, T) {
            const z = [1, .6, .2, -.4].map(v => v * st.scale);
            const naiveSum = z.reduce((a, v) => a + Math.exp(v), 0);
            const m = Math.max.apply(null, z);
            const stable = m + Math.log(z.reduce((a, v) => a + Math.exp(v - m), 0));
            const pNaive = Math.exp(z[0]) / naiveSum;
            const pStable = Math.exp(z[0] - stable);
            ctx.font = '13px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
            const lines = [
              ['logits z', '[' + z.map(v => v.toFixed(0)).join(', ') + ']', T.text],
              ['naive   Σ exp(z)', isFinite(naiveSum) ? naiveSum.toExponential(3) : 'Infinity  ← overflow', isFinite(naiveSum) ? T.text : T.red],
              ['stable  m + log Σ exp(z−m)', stable.toFixed(4), T.green],
              ['softmax p₁ naive', isFinite(pNaive) ? pNaive.toFixed(6) : 'NaN  ← the training run dies here', isFinite(pNaive) ? T.text : T.red],
              ['softmax p₁ stable', pStable.toFixed(6), T.green]
            ];
            lines.forEach((L, i) => {
              ctx.fillStyle = T.muted; ctx.fillText(L[0], 12, 18 + i * 30);
              ctx.fillStyle = L[2]; ctx.font = 'bold 13px ui-monospace, monospace';
              ctx.fillText(L[1], Math.min(w - 200, 250), 18 + i * 30);
              ctx.font = '13px ui-monospace, monospace';
            });
            out({
              naive: isFinite(naiveSum) ? naiveSum.toExponential(1) : '∞',
              lse: stable.toFixed(2),
              p0: isFinite(pNaive) ? pNaive.toFixed(4) : 'NaN',
              p0s: pStable.toFixed(4)
            });
          }
        });
        Viz.note(host, 'Push the magnitude past ~710 and float64 gives up. Every framework implements softmax and cross-entropy this way — which is also why you pass <i>logits</i>, not probabilities, to <code>BCEWithLogitsLoss</code> and <code>CrossEntropyLoss</code>.');
      }
    },
    quiz: [
      {
        q: 'Minimising cross-entropy is equivalent to…',
        options: ['maximising entropy of the model', 'minimising KL divergence to the true distribution, up to a constant', 'minimising the L2 norm of the logits', 'maximising mutual information'],
        answer: 1,
        why: 'The identity $H(p,q)=H(p)+D_{KL}(p\\|q)$ splits your total coding cost into an unavoidable floor and a penalty for being wrong, and only the second term contains your parameters — so every step that lowers cross-entropy lowers the divergence to the truth by exactly the same amount. "Maximising the entropy of the model" is tempting because entropy and cross-entropy share a name and both appear in the identity, but they move in opposite directions here: sharpening a correct model raises cross-entropy, as the perplexity lab shows when sharpness 2 pushes it from 1.923 to 2.225 bits. The same identity explains why cross-entropy can never fall below $H(p)$, which is why a perplexity of 1 is not a target on genuinely uncertain data.'
      },
      {
        q: 'A model reports perplexity 12 on its own tokenizer and a competitor reports 9 on a different tokenizer. What can you conclude?',
        options: ['The competitor is better', 'Nothing directly — perplexity is only comparable within a fixed tokenizer', 'The competitor has more parameters', 'Both are equally good'],
        answer: 1,
        why: 'Perplexity is the exponential of the mean negative log-likelihood <i>per token</i>, so changing what counts as a token changes the denominator and therefore the number. A tokenizer that splits words into more pieces makes each individual prediction easier and reports a flatteringly low perplexity for identical modelling ability. "The competitor is better" is tempting because 9 is plainly less than 12 and both are labelled with the same word, which is exactly why the comparison gets made in public. To compare properly, evaluate both models on a shared tokenization or on bits per character, and then note that perplexity measures compression rather than usefulness anyway — §4.16 is about the task evaluations that actually decide whether a model is better for you.'
      },
      {
        q: 'You fit a single Gaussian to a bimodal target by minimising reverse KL(q‖p). The result will typically…',
        options: ['cover both modes with high variance', 'collapse onto one mode', 'be identical to forward KL', 'fail to converge'],
        answer: 1,
        why: 'Reverse KL averages over $q$ rather than over $p$, so it is punished for putting mass where the target is near zero and is not punished for missing regions where the target has mass; the cheapest escape is to retreat into one mode and ignore the other. That is mode-seeking, and it is why variational posteriors are famously over-confident. The important caveat, visible in the KL lab, is that this is not automatic: at the lab\'s default separation of 3.4 the reverse-KL fit still straddles both modes, and collapse only wins once the separation passes about 4.5, when the flat $\\ln 2$ cost of ignoring half the mass finally beats the ever-growing cost of covering an empty valley. Forward KL, the direction maximum likelihood uses, does the opposite and spreads to cover everything.'
      }
    ],
    cards: [
      { q: 'The cross-entropy identity', a: '$H(p,q)=H(p)+D_{KL}(p\\|q)$ — so minimising cross-entropy minimises KL up to a constant.' },
      { q: 'Forward vs reverse KL', a: 'Forward $D(p\\|q)$ is mode-covering; reverse $D(q\\|p)$ is mode-seeking.' },
      { q: 'Log-sum-exp trick', a: '$\\log\\sum e^{z_i}=m+\\log\\sum e^{z_i-m}$ with $m=\\max z_i$; behind stable softmax and FlashAttention’s online softmax.' },
      { q: 'Perplexity', a: '$\\exp$ of mean NLL per token — an effective branching factor, comparable only within one tokenizer.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.11 */
  ML.section({
    id: 'part1-recall', track: 'foundations', num: '1.16',
    title: 'Rapid recall — Part 1 in twelve lines',
    lede: 'The night-before sheet. If you can produce each line from memory with its derivation sketch, Part 1 is done.',
    html: `
<p>This page is deliberately unlike every other page in Part 1. Everywhere else the job was to build an idea slowly enough that it became yours; here the job is compression. Twelve lines, each of which should unpack into a paragraph you can produce out loud, without notes, from the line alone.</p>

<p>Use it as a test rather than as a summary. Cover the right-hand column, read a line, and say the explanation aloud before you look — the discomfort of failing to retrieve something is what actually moves it into memory, whereas rereading a passage you already recognise produces a warm feeling of competence and almost no learning. If a line will not unpack, that is the section to reopen, and it is the only reliable signal you have about where the gaps are.</p>

${H.table(['#', 'The line', 'Section'], [
      ['1', 'Posterior odds = LR × prior odds.', '<a href="#/bayes">1.1</a>'],
      ['2', 'Low base rate ⇒ a 99%-accurate positive is ~2% right.', '<a href="#/bayes">1.1</a>'],
      ['3', 'Estimation error falls like $1/\\sqrt n$; Hoeffding quantifies the tail.', '<a href="#/concentration">1.4</a>'],
      ['4', 'NLL is the loss; the noise model picks it.', '<a href="#/mle-map">1.5</a>'],
      ['5', 'Gaussian prior → L2; Laplace prior → L1.', '<a href="#/mle-map">1.5</a>'],
      ['6', 'Variances add only under independence — hence bagging.', '<a href="#/expectation">1.3</a>'],
      ['7', 'CI = coverage of the procedure; credible = probability of θ.', '<a href="#/intervals">1.6</a>'],
      ['8', 'Peeking inflates α; use alpha-spending or always-valid p-values.', '<a href="#/intervals">1.6</a>'],
      ['9', 'ATE targets; propensity needs ignorability + overlap; DiD needs parallel trends; IV needs relevance + exclusion.', '<a href="#/causal">1.7</a>'],
      ['10', 'SVD = rotate · scale · rotate; truncation is the best low-rank fit.', '<a href="#/linear-algebra">1.8</a>'],
      ['11', 'Convex ⟺ $H \\succeq 0$; Newton = curvature-rescaled step.', '<a href="#/calculus-ml">1.9</a>'],
      ['12', 'Cross-entropy = KL + constant.', '<a href="#/information">1.10</a>']
    ])}

<h2>The dependency graph, in words</h2>

<p>The twelve lines are not equally load-bearing, and knowing which ones carry weight is the difference between revising and merely rereading.</p>

<p><b>§1.5 and §1.8 are the two that everything else stands on.</b> MLE and MAP supply every loss function and every penalty you will meet in Parts 2 to 4 — squared error, cross-entropy, L1, L2 and the reason each is the right choice — so a gap there reappears in every later section wearing a new name. Eigenvectors and the SVD supply PCA, the validity of kernels, attention read as a bilinear form, the conditioning story behind every optimiser, and the low-rank claim that LoRA rests on.</p>

<p><b>§1.4 licenses the entire train/test protocol.</b> Without concentration there is no argument that a score computed on a held-out sample says anything about a score on future data, and every claim made in Part 2 about model selection quietly depends on it.</p>

<p><b>§1.10 becomes unavoidable the moment you touch a language model.</b> Cross-entropy, perplexity, KL penalties and the log-sum-exp trick are the working vocabulary of Part 4, not background material.</p>

<p><b>§1.6 and §1.7 are the two that get tested in conversation rather than on paper.</b> Nobody will ask you to derive a confidence interval on a whiteboard, but you will be asked what a p-value means, whether you can stop a test early, and why controlling for a variable made a coefficient move. Those answers are judged on precision of language, which is why both sections spend so long on exactly what each object does and does not claim.</p>

<p>If revision time is short, revise §1.5, §1.8, §1.4 and §1.10 properly and treat the rest as reference. If interview time is close, invert it: rehearse §1.6 and §1.7 out loud, because they are the ones that come up in the conversation part of the day.</p>

<h2>How to use the drill</h2>

<p>The cards below are the twelve lines plus the specific formulas and numbers that turn each one into an answer rather than a slogan. Two rules make the difference between a useful ten minutes and a wasted one. <b>Answer out loud before flipping</b>, because recognising an answer and producing it are different skills and only one of them is what an interview tests. And <b>be strict with the "I knew it" button</b> — the running tally is only worth anything if it measures production rather than familiarity.</p>

${H.lab('drill1', 'Twelve-line drill', 'Shuffled prompts from Part 1. Say the answer out loud before flipping — recall is trained by retrieval failure, not by rereading.')}

<h2>Numbers from Part 1 worth carrying</h2>

<p>Quantitative claims are the cheapest way to sound as though you have actually run experiments, and the expensive way to sound as though you have not is to gesture at magnitudes you cannot defend. These are the numbers from Part 1 that survive being questioned, each with the arithmetic that produces it.</p>

${H.table(['Quantity', 'Value'], [
      ['Rows for ±2 points at 95% (0/1 loss)', '≈ 4,600 · ±1 point: ≈ 18,000'],
      ['A/B sample size', '$n \\approx 16p(1-p)/\\delta^2$ per arm'],
      ['Worked A/B example', '4% baseline, +10% relative → 38,400 per arm → 16 days at 5k/day'],
      ['Halving the detectable effect', 'Quadruples $n$: 5% relative lift on the same baseline needs 153,600 per arm'],
      ['Peeking, 20 looks on a null test', 'Realised false-positive rate ≈ 24%, against a nominal 5%'],
      ['CUPED gain', 'variance × $(1-\\rho^2)$; ρ = 0.6 → 36% reduction'],
      ['20 metrics at α = 0.05', '64% chance of a spurious win ($1-0.95^{20}$)'],
      ['Bayes worked example', 'prevalence 0.001, sens 0.99, spec 0.95 → posterior 1.94%'],
      ['Confounding, worked', 'A 6-point gap in default rates from a treatment with exactly zero effect'],
      ['Condition number cost', 'κ = 100 contracts by 0.98 per step — about 461 steps for four digits'],
      ['Entropy of a 4-outcome source at ½,¼,⅛,⅛', '1.75 bits; a uniform code costs 2, so KL = 0.25 bits']
    ])}

<p>One last piece of advice about all of it. Every number above is checkable, and the point of carrying them is not that anyone will ask for them directly — it is that a claim with arithmetic attached ends an argument, and a claim without it starts one.</p>`,
    labs: {
      drill1: function (host) {
        const cards = [
          ['State Bayes in odds form.', 'Posterior odds = likelihood ratio × prior odds. In logs, addition.'],
          ['Why is a 99%-accurate test only 2% right at 0.1% prevalence?', 'False positives from the healthy 99.9% swamp the true positives ~50:1.'],
          ['Hoeffding’s bound.', '$P(|\\bar X-\\mu|\\ge t)\\le 2e^{-2nt^2/(b-a)^2}$.'],
          ['Rows needed for ±2 points at 95%?', '≈ 4,600. For ±1 point, ≈ 18,000.'],
          ['Where does cross-entropy come from?', 'The negative log-likelihood of a Bernoulli/Categorical label.'],
          ['L2 corresponds to which prior, with what λ?', 'Zero-mean Gaussian; $\\lambda = 1/(2\\tau^2)$.'],
          ['Bagging variance formula.', '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$.'],
          ['Confidence vs credible interval.', 'Coverage of the procedure vs probability of the parameter given a prior.'],
          ['Why does peeking inflate α?', 'Repeated correlated looks multiply the chance of crossing the boundary.'],
          ['A/B sample size formula.', '$n\\approx 16p(1-p)/\\delta^2$ per arm; $n \\propto 1/\\delta^2$.'],
          ['DiD’s key assumption.', 'Parallel trends absent treatment.'],
          ['What does conditioning on a collider do?', 'Creates bias — it opens a path between the collider’s causes.'],
          ['SVD in one sentence.', 'Rotate · scale · rotate; truncation is the best low-rank approximation (Eckart–Young).'],
          ['Definition of PSD and why it matters.', '$x^\\mathsf{T}Ax\\ge0$; covariance and kernels are PSD, making the optimisation convex.'],
          ['Convexity test and Newton’s step.', '$H\\succeq0$ everywhere; $\\theta\\leftarrow\\theta-H^{-1}\\nabla f$.'],
          ['Condition number’s two consequences.', 'Max stable $\\eta<2/\\lambda_{\\max}$; rate $(\\kappa-1)/(\\kappa+1)$.'],
          ['Cross-entropy in terms of KL.', '$H(p,q)=H(p)+D_{KL}(p\\|q)$.'],
          ['Log-sum-exp trick.', 'Subtract the max: $\\log\\sum e^{z_i}=m+\\log\\sum e^{z_i-m}$.'],
          ['Perplexity, and its one caveat.', 'exp of mean NLL per token; comparable only within one tokenizer.'],
          ['CUPED’s variance reduction.', 'Factor $(1-\\rho^2)$, unbiased, using a pre-period covariate.']
        ];
        let order = cards.map((_, i) => i).sort(() => Math.random() - .5);
        let i = 0, showA = false, right = 0, seen = 0;
        const face = ML.el('div', { class: 'card-face', style: 'cursor:pointer;border:1px solid var(--line);border-radius:12px;background:var(--panel)' });
        host.appendChild(face);
        const pos = ML.el('span');
        function draw() {
          const c = cards[order[i]];
          face.innerHTML = showA ? '<div class="a">' + c[1] + '</div>' : '<div><b>' + c[0] + '</b></div>';
          pos.textContent = (i + 1) + ' / ' + cards.length + (seen ? '  ·  ' + right + '/' + seen + ' recalled' : '');
          ML.typeset(face);
        }
        face.addEventListener('click', () => { showA = !showA; draw(); });
        const nav = ML.el('div', { class: 'cardnav' });
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Flip', onclick: () => { showA = !showA; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'I knew it', onclick: () => { if (showA) { right++; seen++; } i = (i + 1) % cards.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Missed it', onclick: () => { if (showA) seen++; i = (i + 1) % cards.length; showA = false; draw(); } }));
        nav.appendChild(pos);
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Shuffle', onclick: () => { order = order.sort(() => Math.random() - .5); i = 0; showA = false; right = seen = 0; draw(); } }));
        host.appendChild(nav);
        draw();
      }
    },
    quiz: [
      {
        q: 'Which pair of Part 1 sections is load-bearing for the largest number of later sections?',
        options: ['1.6 and 1.7', '1.5 and 1.8', '1.2 and 1.3', '1.9 and 1.10'],
        answer: 1,
        why: 'MLE and MAP supply every loss function and every penalty in Parts 2 to 4, and eigen/SVD supplies PCA, kernel validity, attention read as a bilinear form, the conditioning story behind every optimiser and the low-rank claim underneath LoRA. The tempting answer is 1.6 and 1.7, because intervals and causal inference feel like the most professionally serious material — and they are heavily tested in conversation, but very few later sections actually consume them. The distinction worth internalising is between what gets asked about and what gets used: revise 1.5 and 1.8 to understand later material, and rehearse 1.6 and 1.7 to answer questions about it.'
      }
    ]
  });
})();
