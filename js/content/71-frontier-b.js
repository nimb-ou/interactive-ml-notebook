/* ============================================================
   PART 6 — Beyond the notebook, continued: bandits (6.6),
   detection and segmentation (6.7), privacy-preserving ML (6.8).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 6.6 */
  ML.section({
    id: 'bandits', track: 'frontier', num: '6.6', level: 2,
    title: 'Bandits and online decisions',
    lede: 'The simplest reinforcement learning problem, and the one you are most likely to actually deploy: repeatedly choose an action, observe a reward, and learn while you earn. It is also the honest answer to "can we stop wasting half our traffic on the losing variant?".',
    prereq: ['bayesian-inference'],
    related: ['rl', 'experimentation', 'gp-bayesopt'],
    html: `
${H.tldr([
      'Regret — the reward you forwent by not always playing the best arm — is the objective. Good algorithms achieve $O(\\log T)$ regret; $\\epsilon$-greedy with a fixed $\\epsilon$ achieves $O(\\epsilon T)$, which is linear and therefore bad.',
      '<b>UCB</b> is deterministic optimism: play the arm with the highest plausible mean. <b>Thompson sampling</b> is randomised optimism: draw once from each posterior and play the winner. Thompson usually wins empirically and is trivially parallel.',
      'A bandit is <i>not</i> a replacement for an A/B test. It optimises cumulative reward; an experiment estimates an effect. If you need an unbiased effect size, run the experiment.'
    ])}

<h2><span class="sn">6.6.1</span> The problem, and the objective</h2>
<p>$K$ arms, each with an unknown reward distribution of mean $\\mu_k$. At each round choose $a_t$ and observe $r_t$. Define</p>
$$R_T = \\underbrace{T\\max_k\\mu_k}_{\\text{best possible}} - \\underbrace{\\sum_{t=1}^{T}\\mu_{a_t}}_{\\text{what you got}}$$
${H.key('The whole subject is one trade: every round spent learning about an arm is a round not spent exploiting the best one. Regret measures exactly the cost of that learning, and the lower bound says you cannot beat $\\Omega(\\log T)$.')}
${H.analogy(`<p>You move to a new town with five restaurants and want to maximise how many good meals you eat before you move away again. Always returning to the first decent place you find (greedy) risks missing a better one nearby forever. Visiting all five equally often forever (fixed-ε exploration) wastes meals on the bad ones long after you already know they are bad. What you actually want is to try the untested places early, more often when you are more uncertain about them, and quickly settle into a routine of mostly eating at whichever currently looks best — trying a new place slightly more when your confidence in the current favourite is low, slightly less as the evidence piles up. That instinct, made mathematically precise, is exactly UCB and Thompson sampling; the "regret" you are minimising is the gap between the meals you actually ate and the meals you would have eaten had you known the best restaurant from day one.</p>`)}
${H.table(['Policy', 'Rule', 'Regret', 'Verdict'], [
      ['Greedy', 'always play the current best', '<b>linear</b> — can lock onto a bad arm forever', 'never'],
      ['ε-greedy (fixed ε)', 'explore uniformly with probability ε', '$O(\\epsilon T)$ — linear', 'a baseline, not a method'],
      ['ε-greedy (decaying $\\epsilon_t \\propto 1/t$)', 'explore less over time', '$O(\\log T)$', 'fine, needs tuning'],
      ['<b>UCB1</b>', '$\\hat\\mu_k + \\sqrt{2\\ln t / n_k}$', '$O(\\log T)$, with a matching lower bound', 'deterministic, no tuning, strong guarantees'],
      ['<b>Thompson sampling</b>', 'draw $\\theta_k\\sim$ posterior, play $\\arg\\max$', '$O(\\log T)$, optimal constants', '<b>usually best in practice</b>, and parallelises'],
      ['EXP3', 'exponential weights', '$O(\\sqrt{TK\\log K})$', 'the adversarial setting — no distributional assumption']
    ])}

${H.lab('bandit', 'Race the three policies', 'Real simulation with real posteriors. The left panel is cumulative regret — lower is better — and the right shows how each policy allocated its pulls. Watch greedy lock onto the wrong arm and never find out.')}

${H.deriv('where UCB’s $\\sqrt{2\\ln t / n_k}$ comes from', [
      ['$P(|\\hat\\mu_k - \\mu_k| \\ge \\epsilon) \\le 2e^{-2n_k\\epsilon^2}$', 'Hoeffding’s inequality for a mean of $n_k$ bounded samples (§1.4). This is the only probabilistic ingredient.'],
      ['set $2e^{-2n_k\\epsilon^2} = t^{-4}$', 'Choose a failure probability that shrinks fast enough that the total probability of <i>ever</i> being wrong stays finite when summed over all rounds.'],
      ['$\\epsilon = \\sqrt{2\\ln t / n_k}$', 'Solve. This is the confidence radius, and it is the entire bonus term.'],
      ['play $\\arg\\max_k\\ \\hat\\mu_k + \\epsilon_k$', '<b>Optimism in the face of uncertainty</b>: act as if each arm is as good as it plausibly could be. An arm is then played either because it is good or because it is under-explored — and either way you learn something.']
    ], 'The bonus shrinks as $1/\\sqrt{n_k}$ for an arm you keep pulling and grows as $\\sqrt{\\ln t}$ for one you neglect, so every arm is revisited eventually — but only logarithmically often if it is bad.')}

<h2><span class="sn">6.6.2</span> Thompson sampling, and why it is so good</h2>
${H.steps([
      'Maintain a posterior over each arm’s mean. For binary rewards with a Beta$(\\alpha,\\beta)$ prior this is the conjugate update from §1.14: $\\alpha_k \\mathrel{+}= \\text{successes}$, $\\beta_k \\mathrel{+}= \\text{failures}$.',
      'Each round, draw <b>one</b> sample $\\theta_k$ from each posterior.',
      'Play $\\arg\\max_k\\theta_k$, observe the reward, update that arm’s posterior.'
    ])}
${H.intuition(`<p>Thompson sampling plays each arm with exactly the probability that it is the best one. That is a remarkable property to get from three lines of code: it explores in proportion to <i>the probability that exploring is worth it</i>, which is precisely the right amount, and it needs no confidence bound, no tuning constant, and no clock.</p>
<p>It also parallelises trivially. If ten workers each draw their own posterior sample, they naturally diversify — no coordination required. UCB, being deterministic, gives all ten workers the same arm unless you add machinery.</p>`)}

<h2><span class="sn">6.6.3</span> Contextual bandits, and the honest caveats</h2>
<p>When each round comes with a context $x_t$ (this user, this page), you want $\\arg\\max_a \\hat f(x_t,a)$ with an exploration bonus. LinUCB assumes $f$ is linear in the features and maintains a confidence ellipsoid per arm; neural bandits replace the linear model and keep the bonus heuristically.</p>
${H.table(['Failure mode', 'What happens', 'Mitigation'], [
      ['<b>Non-stationarity</b>', 'the best arm changes and the bandit has already committed', 'discount old observations, sliding window, or restart periodically'],
      ['Delayed rewards', 'conversions arrive hours later, so the bandit updates on stale information', 'batch updates; model the delay explicitly'],
      ['<b>Feedback loops</b>', 'the bandit stops showing an arm, so it never learns it improved', 'floor the exploration probability; keep a small random slice'],
      ['Biased logs', 'the logged data reflects the policy, not the world', 'log propensities — off-policy evaluation is impossible without them'],
      ['No unbiased effect size', 'you cannot report "variant B lifted conversion by 3.2%"', '<b>run an A/B test if that is the deliverable</b>']
    ])}
${H.flag('The frequent claim that "bandits are strictly better than A/B tests" is wrong in a specific way. A bandit minimises cumulative regret; an experiment estimates a treatment effect with a known error rate. If the deliverable is a decision about which of two permanent options to keep, and you need a defensible number for it, run the experiment. If the deliverable is "make money across 5,000 headlines that expire in a week", run the bandit. Different objectives, different tools.')}
${H.worked('when the bandit is obviously right', `
<p>You have 200 candidate email subject lines and one send window. An A/B test across 200 arms needs an enormous sample per arm to be powered, and by the time it concludes the campaign is over.</p>
<p>Thompson sampling reaches near-optimal allocation within a few thousand impressions and spends the remaining traffic on the winners. <b>Expected regret over 100,000 sends is a few hundred suboptimal impressions rather than 99,500.</b> Nobody needs a confidence interval on subject line #147 — the entire deliverable is the cumulative click count.</p>`)}
${H.history(`<p>Thompson's original 1933 paper proposed the randomised-sampling idea for clinical trial allocation, decades before "regret" was formalised as an objective — it was rediscovered and named only once the bandit framing caught up with it. Robbins (1952) posed the sequential-allocation problem in essentially its modern form. Lai and Robbins (1985) proved the $\\Omega(\\log T)$ lower bound that makes "logarithmic regret" the target rather than an arbitrary nice-sounding rate. Auer, Cesa-Bianchi and Fischer's UCB1 (2002) gave the first clean, deterministic algorithm matching that bound with a short proof. Thompson sampling then sat relatively unused for most of the 20th century — Chapelle and Li's 2011 empirical study at Yahoo is usually credited with reviving it, by showing it beat UCB in practice on real advertising traffic despite UCB's tidier regret proof, which is precisely why the table above lists it as "usually best in practice" rather than "provably best".</p>`)}

${H.probe([
      ['Why is fixed-ε greedy bad asymptotically?', 'It keeps exploring at a constant rate forever, so regret grows linearly: $O(\\epsilon T)$. Decay ε, or use a policy whose exploration self-terminates.'],
      ['Explain the UCB bonus.', 'A Hoeffding confidence radius $\\sqrt{2\\ln t/n_k}$: play the arm with the highest plausible mean. It shrinks with pulls and grows with elapsed time, so neglected arms are revisited.'],
      ['Why is Thompson sampling preferred in practice?', 'It plays each arm with the probability that it is optimal — the right exploration rate, with no tuning — and it parallelises naturally because the randomness diversifies the workers.'],
      ['Your bandit converged and then the winner degraded. What went wrong?', 'Non-stationarity plus a feedback loop: the bandit stopped sampling the alternatives, so it could not notice. Use discounting or a sliding window, and floor the exploration probability.'],
      ['When would you refuse to use a bandit?', 'When you need an unbiased, defensible effect estimate — a launch decision, a regulatory filing, a paper. Bandits optimise reward, not estimation.']
    ])}`,
    labs: {
      bandit: function (host) {
        const st = Viz.controls(host, [
          { k: 'K', label: 'arms', min: 2, max: 8, step: 1, value: 5, fmt: v => v },
          { k: 'gap', label: 'gap between best and rest', min: .01, max: .3, step: .01, value: .06, fmt: v => (v * 100).toFixed(0) + ' pts' },
          { k: 'T', label: 'rounds', min: 200, max: 20000, step: 200, value: 4000, fmt: v => v.toLocaleString() },
          { k: 'eps', label: 'ε for ε-greedy', min: 0, max: .5, step: .01, value: .1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'greedy', label: 'greedy regret', cls: 'bad' },
          { k: 'eps', label: 'ε-greedy regret' },
          { k: 'ucb', label: 'UCB regret' },
          { k: 'ts', label: 'Thompson regret', cls: 'good' },
          { k: 'best', label: 'best arm rate', cls: 'key' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(5);
            const rates = Array.from({ length: st.K }, (_, i) => i === 0 ? .35 + st.gap : .35 - R() * .06);
            const shuffled = Num.rng(2).shuffle(rates);
            const bestIdx = shuffled.indexOf(Math.max.apply(null, shuffled));
            const runs = {
              greedy: Num.bandit(shuffled, 'greedy', { seed: 11 }),
              eps: Num.bandit(shuffled, 'eps', { seed: 12, eps: st.eps }),
              ucb: Num.bandit(shuffled, 'ucb', { seed: 13 }),
              ts: Num.bandit(shuffled, 'thompson', { seed: 14 })
            };
            Object.keys(runs).forEach(k => runs[k].step(st.T));

            const w1 = w * .58;
            const maxR = Math.max.apply(null, Object.keys(runs).map(k => runs[k].regret)) * 1.1 || 1;
            const P = Viz.plot(ctx, w1, h, { xd: [0, st.T], yd: [0, maxR], pad: { l: 50, r: 10, t: 16, b: 38 } })
              .frame({ xlabel: 'rounds', ylabel: 'cumulative regret' });
            const cols = { greedy: T.c2, eps: T.c4, ucb: T.c1, ts: T.c3 };
            P.clip(() => {
              Object.keys(runs).forEach(k => {
                const hh = runs[k].history;
                const stepN = Math.max(1, Math.floor(hh.length / 200));
                const pts = [];
                for (let i = 0; i < hh.length; i += stepN) pts.push([i, hh[i].regret]);
                P.line(pts, { color: cols[k], width: 2.2 });
              });
            });

            ctx.save(); ctx.translate(w1, 0);
            const ww = w - w1;
            const P2 = Viz.plot(ctx, ww, h, { xd: [0, 1], yd: [0, 1], pad: { l: 34, r: 12, t: 26, b: 38 } });
            const M = ['greedy', 'eps', 'ucb', 'ts'].map(k => runs[k].pulls.map(v => v / st.T));
            P2.heat(M, {
              rows: ['greedy', 'ε-gr', 'UCB', 'TS'],
              cols: shuffled.map((r, i) => (i === bestIdx ? '★' : '') + (r * 100).toFixed(0)),
              lo: 0, hi: 1, cell: v => (v * 100).toFixed(0) + '%'
            });
            ctx.fillStyle = T.muted; ctx.font = '10.5px ui-sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('share of pulls per arm (★ = best)', ww / 2, h - 12);
            ctx.restore();

            out({
              greedy: runs.greedy.regret.toFixed(1),
              eps: runs.eps.regret.toFixed(1),
              ucb: runs.ucb.regret.toFixed(1),
              ts: runs.ts.regret.toFixed(1),
              best: (Math.max.apply(null, shuffled) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c2)', t: 'greedy' }, { c: 'var(--c4)', t: 'ε-greedy' },
          { c: 'var(--c1)', t: 'UCB1' }, { c: 'var(--c3)', t: 'Thompson' }
        ]);
        Viz.note(host, 'The shapes are the theory made visible. Greedy and ε-greedy are <b>straight lines</b> — linear regret — while UCB and Thompson bend over into logarithmic curves. Look at the pull-share panel: Thompson concentrates 85–95% of its pulls on the best arm while still touching the others; greedy sometimes puts 100% on the <i>wrong</i> arm and never recovers, because it has no mechanism that could ever revisit the decision. Now shrink the gap to 0.01 and every method struggles — <b>regret scales as $\\sum_k \\log T/\\Delta_k$, so near-identical arms are expensive to separate and cheap to get wrong.</b>');
      }
    },
    quiz: [
      {
        q: 'ε-greedy with a fixed ε has regret…',
        options: ['$O(\\log T)$', 'linear in $T$', '$O(\\sqrt T)$', 'zero'],
        answer: 1,
        why: 'A fixed ε keeps exploring the same amount forever, so it pays roughly $\\epsilon\\Delta$ of regret every single round with no mechanism to reduce that rate as evidence accumulates — sum a constant cost over $T$ rounds and you get $O(\\epsilon T)$, linear regret, exactly as §6.6.1\'s table states. "$O(\\log T)$" is the tempting wrong answer precisely because it is the right answer for the closely-related policy one row below in the same table, decaying ε-greedy with $\\epsilon_t \\propto 1/t$ — mixing up the fixed and decaying variants is an easy slip since both share the name "ε-greedy". "$O(\\sqrt T)$" belongs to a different setting entirely, the adversarial bandit that EXP3 is built for, where no distributional assumptions can be made about the arms at all. "Zero" would mean the algorithm never regrets anything, which is impossible for any method that keeps spending rounds on arms already known to be worse. The general principle, which §6.6.1\'s worked derivation of the UCB bonus makes precise for a smarter policy, is that exploration has to taper off as evidence accumulates for regret to grow slower than linearly — a constant exploration rate can never produce that taper.'
      },
      {
        q: 'The UCB bonus $\\sqrt{2\\ln t/n_k}$ comes from…',
        options: ['the central limit theorem', 'Hoeffding’s inequality with a failure probability that shrinks in $t$', 'a Bayesian posterior', 'cross-validation'],
        answer: 1,
        why: 'The §6.6.1 derivation gets there in three lines from one probabilistic fact: Hoeffding\'s inequality (§1.4) bounds $P(|\\hat\\mu_k-\\mu_k|\\ge\\epsilon)\\le 2e^{-2n_k\\epsilon^2}$, and setting this failure probability equal to $t^{-4}$ — small enough that the total chance of ever being wrong across all rounds stays finite — and solving for $\\epsilon$ gives exactly the bonus $\\sqrt{2\\ln t/n_k}$ used in UCB1\'s rule. "The central limit theorem" is a tempting substitute because it is the other major tool this course uses for reasoning about sample means (§1.4), but the CLT only describes a limiting distribution as $n\\to\\infty$ and gives no finite-sample guarantee at any particular $t$ and $n_k$ — exactly the kind of number a bandit algorithm needs at every single round, which is why Hoeffding, not the CLT, is the tool that actually works here. "A Bayesian posterior" describes Thompson sampling\'s mechanism instead, covered two sections later in §6.6.2, and confusing the two is an easy slip since both are answers to "how much should I explore this arm" — but UCB\'s bonus is a frequentist confidence radius with no prior distribution anywhere in its derivation. "Cross-validation" has no role in an online decision problem at all, since there is no held-out set to validate against when each round\'s data arrives only after the choice is made. The general principle is optimism in the face of uncertainty: converting a concentration bound into an upper confidence bound lets an algorithm act on the best plausible value of an unknown quantity rather than only its current point estimate.'
      },
      {
        q: 'Thompson sampling plays each arm with probability…',
        options: ['1/K', 'proportional to its mean', 'equal to the posterior probability that it is the best arm', 'proportional to its variance'],
        answer: 2,
        why: 'Thompson sampling draws exactly one sample $\\theta_k$ from each arm\'s posterior and plays the argmax, and §6.6.2 states the remarkable property that follows directly from that procedure: an arm gets played with precisely the probability that it actually is the best one, which is exactly the right amount of exploration with no confidence bound, no tuning constant, and no clock needed anywhere in the rule. "Proportional to its mean" is the tempting confusion because it sounds like a sensible exploration rule and mean reward is the quantity everyone actually cares about — but playing proportional to the mean would keep exploring a known-bad arm forever just because its mean is nonzero, never converging the way logarithmic-regret policies must. "1/K" describes uniform random exploration, the fixed-ε policy this section\'s first quiz question already showed has linear regret — the opposite of what a good bandit policy should do as evidence accumulates. "Proportional to its variance" would explore the most uncertain arm hardest regardless of whether it looks remotely competitive, which is closer to pure exploration than to the exploration-exploitation balance Thompson sampling actually strikes. The general principle, which is also why §6.6.2 notes Thompson sampling parallelises for free, is that matching your action probabilities to your posterior belief about optimality is a stronger, more automatic form of "optimism" than any hand-tuned confidence radius.'
      },
      {
        q: 'You need to report "variant B lifted conversion by 3.2% ± 0.8%". You should…',
        options: ['use a bandit', 'run a randomised A/B test', 'use UCB and read off the means', 'use Thompson sampling with a long burn-in'],
        answer: 1,
        why: 'A bandit allocates traffic adaptively towards whichever arm currently looks best, which means the arms are not sampled with equal, known probabilities — and any effect size computed from that data inherits the bias of the very policy that generated it, exactly the failure mode §6.6.3\'s "biased logs" row names. A randomised A/B test, by contrast, assigns units to each arm with fixed, known probabilities, which is precisely what licenses an unbiased estimate with a defensible error bar like the ± 0.8% asked for here. "Use UCB and read off the means" is the tempting shortcut because UCB does maintain a running mean per arm that looks exactly like the number you want, but that mean is computed from a sample that was deliberately skewed towards good-looking arms, not drawn independently of the outcome — reading it off does not make it unbiased. "Use Thompson sampling with a long burn-in" makes the same mistake in a different guise: no amount of additional running time removes the fact that a bandit\'s sampling is a function of the rewards it has already observed, which is the structural reason it cannot deliver an unbiased effect size regardless of how long it runs. The general principle, stated directly as the flagged distinction in §6.6.3, is that a bandit minimises cumulative regret while an experiment estimates a treatment effect — different objectives, and the deliverable determines which tool is the right one, not which tool happens to be running already.'
      }
    ],
    cards: [
      { q: 'Regret', a: '$R_T = T\\max_k\\mu_k - \\sum_t\\mu_{a_t}$. Good policies are $O(\\log T)$; fixed-ε greedy is linear.' },
      { q: 'UCB1', a: 'Play $\\arg\\max_k \\hat\\mu_k + \\sqrt{2\\ln t/n_k}$ — optimism in the face of uncertainty, from Hoeffding.' },
      { q: 'Thompson sampling', a: 'Draw once from each posterior, play the argmax. Plays each arm with the probability it is best; parallelises free.' },
      { q: 'Bandit vs A/B test', a: 'Bandit minimises cumulative regret; the experiment estimates an effect. Different objectives — pick by deliverable.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.7 */
  ML.section({
    id: 'vision-tasks', track: 'frontier', num: '6.7', level: 2,
    title: 'Detection, segmentation, and the metrics that go with them',
    lede: 'Classification asks "what is in this image". Detection asks "what, and where, and how many" — and that change turns the evaluation into a matching problem with its own vocabulary of IoU, NMS and mAP that comes up in every computer-vision interview.',
    prereq: ['cnn'],
    related: ['cnn', 'multimodal', 'metrics'],
    html: `
${H.tldr([
      '<b>IoU</b> — intersection over union — is the currency. A detection counts as correct if its IoU with a ground-truth box exceeds a threshold, usually 0.5, and each ground truth can be matched at most once.',
      '<b>NMS</b> removes duplicate detections of the same object: sort by confidence, keep the top box, suppress everything overlapping it above a threshold, repeat.',
      '<b>mAP</b> is the mean over classes of average precision. COCO’s headline number averages AP over ten IoU thresholds from 0.50 to 0.95, which is why COCO mAP is so much lower than PASCAL VOC mAP for the same model.'
    ])}

<h2><span class="sn">6.7.1</span> The task family</h2>
${H.table(['Task', 'Output', 'Metric', 'Representative models'], [
      ['Classification', 'one label per image', 'top-1 / top-5 accuracy', 'ResNet, ViT'],
      ['<b>Object detection</b>', 'boxes + labels + scores', '<b>mAP</b>', 'Faster R-CNN, YOLO, DETR, RT-DETR'],
      ['Semantic segmentation', 'a class per pixel', 'mean IoU', 'U-Net, DeepLab, SegFormer'],
      ['Instance segmentation', 'a mask per object', 'mask AP', 'Mask R-CNN, SOLOv2'],
      ['Panoptic segmentation', 'both — things and stuff', 'panoptic quality (PQ)', 'Mask2Former'],
      ['Promptable segmentation', 'mask from a point/box prompt', 'IoU with the intended object', 'SAM / SAM 2'],
      ['Keypoints', 'landmark coordinates', 'OKS-based AP', 'HRNet, ViTPose'],
      ['Open-vocabulary detection', 'boxes for arbitrary text queries', 'mAP on unseen classes', 'GroundingDINO, OWL-ViT']
    ])}
${H.intuition(`<p>Two structural choices distinguish the detector families. <b>Two-stage</b> (Faster R-CNN) proposes regions, then classifies each — accurate, slower. <b>One-stage</b> (YOLO, RetinaNet) predicts boxes directly on a dense grid — faster, and needed focal loss to cope with the extreme foreground/background imbalance that a dense grid creates. <b>Set-based</b> (DETR) treats detection as predicting a set and matches predictions to ground truth with the Hungarian algorithm — which removes the need for anchors and NMS entirely, at the cost of famously slow convergence.</p>`)}
${H.analogy(`<p>IoU is the same idea as a lost-property office matching a found umbrella against a filed description. "Black umbrella, curved handle, about a metre long" either overlaps a filed description almost completely, partially, or not at all — and the office needs a threshold for how much overlap counts as "close enough" to be the same item before returning it. A detector's predicted box and a labelled ground-truth box are being matched the same way: IoU is the overlap score, and 0.5 (or COCO's sliding scale up to 0.95) is the threshold the office has agreed on. Two detections both claiming the same real umbrella — the duplicate-box problem NMS exists to solve — is exactly the lost-property office accidentally logging the same umbrella twice and having to notice and merge the entries.</p>`)}

${H.lab('detect', 'IoU, NMS and average precision, on boxes you control', 'Move the predicted boxes and the thresholds. Every number — IoU per pair, which boxes survive NMS, the precision/recall curve, the AP — is computed from the boxes shown, with the same greedy matching COCO uses.')}

<h2><span class="sn">6.7.2</span> The three things that get asked</h2>
${H.deriv('why AP is computed with interpolated precision', [
      ['sort detections by confidence, descending', 'The PR curve is traced by lowering the score threshold one detection at a time.'],
      ['match greedily to unmatched ground truth at IoU ≥ τ', 'Each ground-truth box can be claimed once. A second detection of the same object is a <b>false positive</b>, which is precisely what NMS exists to prevent.'],
      ['record $(\\text{recall}, \\text{precision})$ after each detection', 'Recall is monotone non-decreasing; precision zig-zags, because one bad detection drops it and the next good one lifts it.'],
      ['$\\text{AP} = \\frac{1}{101}\\sum_{r\\in\\{0,0.01,\\dots,1\\}}\\max_{\\tilde r \\ge r}p(\\tilde r)$', 'COCO’s 101-point interpolation: at each recall level take the <i>best precision achievable at that recall or higher</i>. The max removes the zig-zag, which would otherwise make AP depend on the noise in the ordering rather than on the quality of the detector.']
    ], 'mAP averages this over classes; COCO’s headline mAP additionally averages over IoU thresholds 0.50:0.05:0.95, which rewards precise localisation and is why the same model scores ~55 on COCO and ~80 on VOC.')}
${H.table(['Question', 'The answer that lands'], [
      ['Why do we need NMS?', 'Dense predictors fire many times on one object. Without suppression every duplicate is a false positive, so precision — and therefore AP — collapses.'],
      ['What is soft-NMS?', 'Instead of deleting overlapping boxes, <i>decay</i> their scores by a function of IoU. Recovers genuinely overlapping objects that hard NMS deletes; worth 1–2 AP on crowded scenes.'],
      ['Why does DETR not need NMS?', 'It predicts a fixed-size set and trains with bipartite Hungarian matching, so duplicate predictions are penalised during training rather than removed afterwards.'],
      ['Why focal loss?', 'A one-stage detector evaluates ~100k anchors, almost all background. Cross-entropy is dominated by easy negatives; focal loss down-weights them by $(1-p_t)^\\gamma$ so the rare hard examples drive the gradient.'],
      ['Dice vs cross-entropy for segmentation?', 'Cross-entropy is per-pixel and ignores the imbalance of a small foreground; Dice optimises overlap directly and handles it. Most production recipes sum both.']
    ], 'plain')}

<h2><span class="sn">6.7.3</span> Segmentation, briefly</h2>
$$\\text{IoU} = \\frac{|A\\cap B|}{|A\\cup B|}, \\qquad \\text{Dice} = \\frac{2|A\\cap B|}{|A|+|B|} = \\frac{2\\,\\text{IoU}}{1+\\text{IoU}}$$
${H.note('Dice and IoU are monotone transformations of each other, so they always rank models the same way — but Dice is always the larger number, which is why a paper reporting Dice and one reporting IoU are not directly comparable. Dice is more forgiving of small objects, which is why medical imaging prefers it.')}
${H.flag('Segment Anything (SAM) changed the practical picture: for many tasks you no longer train a segmentation model, you prompt one with a point or a box from a detector, and fine-tune only if the domain is genuinely unusual (medical, satellite, microscopy). That is the same "cheapest thing that could work" ladder as §5.13, applied to vision.')}
${H.history(`<p>Girshick et al.'s R-CNN (2014) first showed a CNN classifying region proposals could beat hand-engineered detectors, at the cost of running the network separately on every proposal. Fast R-CNN (2015) shared computation across proposals within one image; Faster R-CNN (2015) replaced the external proposal algorithm with a learned region-proposal network, folding the whole pipeline into one trainable model. YOLO (Redmon et al., 2016) and SSD (Liu et al., 2016) then showed a single dense pass could skip proposals entirely, trading some accuracy for real-time speed. Lin et al.'s focal loss and RetinaNet (2017) closed most of that accuracy gap by directly fixing the foreground/background imbalance a dense grid creates. Carion et al.'s DETR (2020) removed anchors and NMS altogether by framing detection as set prediction. Kirillov et al.'s Segment Anything (2023) then did to segmentation what large language models did to NLP: one model, prompted rather than retrained per task, covering the majority of practical segmentation needs out of the box.</p>`)}

${H.probe([
      ['Define IoU and state the usual threshold.', 'Intersection area over union area; 0.5 is the classic threshold, and COCO averages AP over 0.50 to 0.95 in steps of 0.05.'],
      ['Walk me through NMS.', 'Sort by confidence; take the highest-scoring box; remove every remaining box whose IoU with it exceeds the threshold; repeat on what is left. Per class, usually.'],
      ['Why is COCO mAP so much lower than VOC mAP?', 'COCO averages over ten IoU thresholds up to 0.95, so it rewards precise localisation; VOC uses 0.5 only.'],
      ['Your detector has high recall and low precision. What do you change first?', 'The NMS IoU threshold and the score threshold — you are probably keeping duplicates. Then look at whether the confidence is calibrated (§2.12).']
    ])}`,
    labs: {
      detect: function (host) {
        const gts = [[60, 40, 190, 175], [230, 70, 340, 200], [120, 210, 250, 320]];
        let dets = [
          { box: [70, 50, 195, 180], score: .95 },
          { box: [55, 35, 185, 170], score: .80 },   // duplicate of the first
          { box: [235, 80, 350, 205], score: .88 },
          { box: [300, 230, 400, 320], score: .62 },  // false positive
          { box: [130, 215, 245, 315], score: .55 },
          { box: [125, 205, 255, 325], score: .40 }   // duplicate of the third
        ];
        const st = Viz.controls(host, [
          { k: 'nms', label: 'NMS IoU threshold', min: .1, max: .95, step: .05, value: .5, fmt: v => v.toFixed(2) },
          { k: 'iou', label: 'match IoU threshold', min: .1, max: .95, step: .05, value: .5, fmt: v => v.toFixed(2) },
          { k: 'conf', label: 'score threshold', min: 0, max: .95, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'useNms', label: 'apply NMS', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kept', label: 'detections kept', cls: 'key' },
          { k: 'tp', label: 'true positives', cls: 'good' },
          { k: 'fp', label: 'false positives', cls: 'bad' },
          { k: 'prec', label: 'precision' },
          { k: 'rec', label: 'recall' },
          { k: 'ap', label: 'AP @ this IoU' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const filtered = dets.filter(d => d.score >= st.conf);
            const keepIdx = st.useNms
              ? Num.nms(filtered.map(d => d.box), filtered.map(d => d.score), st.nms).keep
              : filtered.map((_, i) => i);
            const kept = keepIdx.map(i => filtered[i]).sort((a, b) => b.score - a.score);
            const ap = Num.averagePrecision(kept, gts, st.iou);

            const w1 = w * .58;
            const sc = Math.min(w1 / 430, h / 350);
            ctx.save();
            ctx.translate(8, 8); ctx.scale(sc, sc);
            ctx.fillStyle = T.panel; ctx.fillRect(0, 0, 420, 340);
            ctx.strokeStyle = T.line; ctx.strokeRect(0, 0, 420, 340);
            // ground truth
            gts.forEach((g, i) => {
              ctx.strokeStyle = T.c3; ctx.lineWidth = 2.5; ctx.setLineDash([6, 4]);
              ctx.strokeRect(g[0], g[1], g[2] - g[0], g[3] - g[1]);
              ctx.setLineDash([]);
              ctx.fillStyle = T.c3; ctx.font = '11px ui-monospace, monospace';
              ctx.fillText('gt ' + (i + 1), g[0] + 3, g[1] - 4);
            });
            // detections
            const matched = new Array(gts.length).fill(false);
            kept.forEach((d, i) => {
              let bi = -1, bv = st.iou;
              gts.forEach((g, gi) => { if (matched[gi]) return; const v = Num.iou(d.box, g); if (v >= bv) { bv = v; bi = gi; } });
              const isTP = bi >= 0;
              if (isTP) matched[bi] = true;
              ctx.strokeStyle = isTP ? T.c1 : T.c2; ctx.lineWidth = 2;
              ctx.strokeRect(d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]);
              ctx.fillStyle = isTP ? T.c1 : T.c2;
              ctx.font = '10px ui-monospace, monospace';
              ctx.fillText(d.score.toFixed(2) + (isTP ? ' ✓ IoU ' + bv.toFixed(2) : ' ✗ FP'), d.box[0] + 3, d.box[3] + 12);
            });
            // suppressed
            if (st.useNms) {
              filtered.forEach((d, i) => {
                if (keepIdx.indexOf(i) >= 0) return;
                ctx.strokeStyle = T.faint; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
                ctx.strokeRect(d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]);
                ctx.setLineDash([]);
              });
            }
            ctx.restore();

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, { xd: [0, 1], yd: [0, 1.05], pad: { l: 44, r: 12, t: 16, b: 40 } })
              .frame({ xlabel: 'recall', ylabel: 'precision' });
            P2.clip(() => {
              if (ap.curve.length) {
                P2.line([[0, ap.curve[0].p]].concat(ap.curve.map(c => [c.r, c.p])), { color: T.c1, width: 2.6 });
                P2.dots(ap.curve.map(c => [c.r, c.p]), { r: 3.4, color: T.c1 });
                P2.area(ap.curve.map(c => [c.r, c.p]), { color: T.c1, alpha: .12 });
              }
            });
            ctx.restore();

            const tp = ap.curve.length ? Math.round(ap.curve[ap.curve.length - 1].r * gts.length) : 0;
            out({
              kept: kept.length + ' of ' + dets.length,
              tp: tp, fp: kept.length - tp,
              prec: kept.length ? (tp / kept.length * 100).toFixed(0) + '%' : '—',
              rec: (tp / gts.length * 100).toFixed(0) + '%',
              ap: ap.ap.toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c3)', t: 'ground truth' }, { c: 'var(--c1)', t: 'true positive' },
          { c: 'var(--c2)', t: 'false positive' }, { c: 'var(--faint)', t: 'suppressed by NMS' }
        ]);
        Viz.note(host, 'Switch <b>apply NMS</b> off: the duplicate boxes reappear, each becomes a false positive because its ground truth is already claimed, precision halves and AP collapses. That is the whole argument for NMS in one toggle. Now raise the <b>match IoU threshold</b> to 0.9: detections that looked perfect stop counting, AP drops sharply — which is exactly why COCO’s averaging up to 0.95 produces numbers so much lower than VOC’s single 0.5.');
      }
    },
    quiz: [
      {
        q: 'Non-maximum suppression works by…',
        options: ['removing low-confidence boxes', 'keeping the highest-scoring box and suppressing others that overlap it above an IoU threshold', 'averaging overlapping boxes', 'reranking with a second model'],
        answer: 1,
        why: 'NMS works by repeated greedy selection: sort every detection by confidence, keep the highest-scoring box, remove every remaining box whose IoU with it clears the threshold, then repeat on whatever is left — usually run separately within each class, since two overlapping boxes for different object classes are not duplicates. "Removing low-confidence boxes" describes what the separate score threshold does, a different control in the same lab, and it is an easy pair to conflate because both are filters applied before a detection counts — but a low score alone does not make a box a duplicate, and a high-confidence box can still be suppressed if it overlaps an even higher-confidence one. "Averaging overlapping boxes" is the failure NMS is built to avoid rather than what it does: averaging two boxes, one on target and one badly placed, produces a box that fits neither object well, whereas NMS keeps the single most confident box intact and simply discards its duplicates. "Reranking with a second model" describes an entirely different, heavier technique (a learned re-ranker) and adds a whole extra model where NMS needs none. The general principle, made concrete in the detect lab by toggling NMS off and watching precision collapse, is that a dense detector fires on the same object multiple times, and every uncollapsed duplicate becomes a false positive the moment its one matching ground-truth box has already been claimed.'
      },
      {
        q: 'COCO mAP is lower than VOC mAP for the same model because…',
        options: ['COCO has more classes', 'COCO averages AP over IoU thresholds from 0.50 to 0.95, rewarding precise localisation', 'COCO images are larger', 'COCO uses a different precision definition'],
        answer: 1,
        why: 'COCO\'s headline mAP averages AP over ten IoU thresholds from 0.50 to 0.95 in steps of 0.05, whereas VOC\'s mAP only ever checks 0.5 — a box whose overlap with the ground truth clears 0.5 comfortably (still a true positive on VOC) can easily fall well short of 0.9 or 0.95, so COCO is effectively asking a much harder question of the same detections and produces a lower number for identical predictions, roughly ~55 versus ~80 for the same model as §6.7.2 states. "COCO has more classes" and "COCO images are larger" both describe genuine differences between the two datasets, and it is tempting to reach for "the datasets are just different" as the explanation — but neither property changes what counts as a correct detection, which is entirely a function of the IoU threshold, not the class count or image size. "COCO uses a different precision definition" is close but imprecise: the precision calculation itself, 101-point interpolated AP, is identical in both benchmarks (§6.7.2\'s derivation); what differs is which IoU thresholds that identical calculation gets averaged over. The general principle is that a metric\'s headline number is only comparable across two reports if the matching criterion behind it is the same, which is exactly why the lab lets you raise the match-IoU slider yourself and watch AP fall as detections that "looked perfect" at 0.5 stop counting.'
      },
      {
        q: 'DETR eliminates NMS because…',
        options: ['it predicts only one box', 'it uses bipartite Hungarian matching during training, so duplicates are penalised in the loss', 'it has a higher confidence threshold', 'it uses transformers'],
        answer: 1,
        why: 'DETR frames detection as predicting a fixed-size set and matches predictions to ground truth with the Hungarian algorithm during training, so a duplicate prediction is directly penalised by the loss the moment it competes with another prediction for the same ground-truth box — duplicate suppression happens inside training, not as a separate post-processing pass. "It predicts only one box" is simply factually wrong and easy to check against §6.7.1\'s task table: DETR predicts a fixed-size set of boxes per image, typically far more than the actual number of objects, with most trained to predict "no object". "It has a higher confidence threshold" describes a tunable post-hoc knob that any detector could adjust, and it would not by itself stop a detector from firing on the same object twice at high confidence — the fix DETR uses is architectural, built into how the loss is computed, not a threshold set after the fact. "It uses transformers" names the correct architecture family but not the mechanism that actually removes NMS; a transformer-based detector that still predicted boxes independently, without one-to-one matching, would need NMS exactly as much as a convolutional one. The general principle, stated directly in §6.7.1, is that anchors and NMS are needed specifically because dense, independent per-location predictions have no built-in way to avoid duplicating each other — reframe detection as set prediction with a matching loss, and the duplicate problem is solved before inference ever runs.'
      },
      {
        q: 'Focal loss exists because…',
        options: ['boxes are hard to regress', 'a dense one-stage detector is overwhelmed by easy background anchors', 'IoU is non-differentiable', 'segmentation masks are imbalanced'],
        answer: 1,
        why: 'A one-stage detector like RetinaNet evaluates roughly 100,000 anchors per image, and the overwhelming majority are easy background with no object present at all, so ordinary cross-entropy — which weights every example equally — ends up dominated by the sheer volume of easy negatives rather than driven by the rare, informative, hard examples that actually determine detection quality. "Boxes are hard to regress" describes a genuinely separate part of a detector\'s loss, the box-coordinate regression term, which focal loss does not touch at all; conflating the classification and regression branches is an easy slip since both are part of the same overall loss function. "IoU is non-differentiable" is true in its plain form but irrelevant here, since focal loss is a classification loss, not the IoU-based term used for box regression, and the two problems have entirely separate fixes in a detector\'s design. "Segmentation masks are imbalanced" describes a real and structurally similar problem — small foreground against large background — but it belongs to segmentation\'s Dice-versus-cross-entropy tradeoff in §6.7.3, not to a detector\'s anchor imbalance, even though the underlying imbalance intuition rhymes across both. The general principle, worth carrying to any densely-sampled prediction problem, is that a loss which treats every example identically will always be dominated by whichever category is most numerous, and focal loss\'s $(1-p_t)^\\gamma$ term fixes this by shrinking the contribution of confidently-correct easy examples rather than changing what the network is asked to predict.'
      }
    ],
    cards: [
      { q: 'IoU and Dice', a: '$\\text{IoU}=|A\\cap B|/|A\\cup B|$; $\\text{Dice}=2\\text{IoU}/(1+\\text{IoU})$ — monotone in each other, Dice always larger.' },
      { q: 'NMS', a: 'Sort by score, keep the top box, suppress overlaps above the IoU threshold, repeat. Soft-NMS decays scores instead of deleting.' },
      { q: 'mAP', a: 'Mean over classes of AP, where AP is 101-point interpolated precision. COCO also averages IoU 0.50:0.05:0.95.' },
      { q: 'Detector families', a: 'Two-stage (accurate), one-stage + focal loss (fast), set-based DETR (no anchors, no NMS, slow to converge).' }
    ]
  });

  /* ------------------------------------------------------------------ 6.8 */
  ML.section({
    id: 'privacy', track: 'frontier', num: '6.8', level: 3,
    title: 'Privacy-preserving machine learning',
    lede: 'Models memorise. That is not a bug you can patch — it is a measurable property of training on data, and it has produced verbatim training-text extraction from language models and identity recovery from released statistics. Differential privacy is the only framework that gives a guarantee rather than a hope.',
    related: ['safety', 'fairness', 'pretraining'],
    html: `
${H.tldr([
      'Anonymisation by removing identifiers <b>does not work</b>. 87% of Americans are uniquely identified by ZIP + birthdate + sex; the Netflix Prize dataset was de-anonymised against public IMDb reviews.',
      '<b>Differential privacy</b>: the output distribution must be nearly the same whether or not any single individual is in the dataset. Formally $P(M(D)\\in S)\\le e^{\\varepsilon}P(M(D\')\\in S)+\\delta$ for datasets differing in one record.',
      'DP-SGD achieves this by clipping each per-example gradient and adding calibrated Gaussian noise. It costs accuracy, it costs compute, and it is the only defence with a proof.'
    ])}

<h2><span class="sn">6.8.1</span> Why anonymisation fails</h2>
${H.table(['Incident', 'What was released', 'What happened'], [
      ['Massachusetts GIC, 1997', '"anonymised" hospital records', 'the Governor’s records re-identified by joining ZIP + birthdate + sex against a voter roll'],
      ['AOL search logs, 2006', 'user IDs replaced with numbers', 'individual users identified from the content of their own queries'],
      ['Netflix Prize, 2007', 'anonymised ratings', 'de-anonymised by matching rating patterns to public IMDb reviews'],
      ['Membership inference, 2017 onward', 'a trained model’s API', 'whether a specific record was in the training set, inferred from confidence patterns'],
      ['Training-data extraction, 2021 onward', 'a language model’s outputs', 'verbatim recovery of memorised text, including personal data']
    ])}
${H.key('The lesson these share: privacy is not a property of a <i>dataset</i>, it is a property of the <b>release mechanism</b>. Any auxiliary information the attacker might obtain later is part of the threat model, and you cannot enumerate it in advance. This is exactly what differential privacy is designed around.')}

<h2><span class="sn">6.8.2</span> Differential privacy</h2>
$$P\\big(M(D)\\in S\\big) \\;\\le\\; e^{\\varepsilon}\\,P\\big(M(D')\\in S\\big) + \\delta \\quad\\text{for all }S\\text{, and all }D,D'\\text{ differing in one record}$$
${H.intuition(`<p>Read it as a promise to an individual: <i>whatever is published, it would have been almost exactly as likely to be published had you not participated at all.</i> So nothing that happens to you afterwards can be much more likely because you took part. That is a guarantee about <b>you</b>, not about the dataset, and it holds regardless of what the attacker already knows or later learns.</p>`)}
${H.analogy(`<p>Imagine confessing something to a room of a thousand people who each independently flip a coin: if heads, they say your secret out loud; if tails, they say a made-up secret instead. Anyone listening from outside hears a confession, but cannot tell whether it came from you or was fabricated by a coin flip — your presence in the room barely changed what they heard, because ninety-nine per cent of the room would have said roughly the same things whether or not you were there. Differential privacy formalises exactly this: the "coin flips" are the calibrated noise added by the mechanism $M$, and $\\varepsilon$ measures how much more revealing the room's chatter becomes because you specifically walked in. A smaller ε means more coin flips relative to true confessions — more noise, more protection, and a blurrier signal for anyone trying to extract the truth about any one person.</p>`)}
${H.table(['Property', 'Statement', 'Why it matters'], [
      ['Composition', '$k$ queries at $\\varepsilon$ each cost at most $k\\varepsilon$ (better under advanced composition)', 'the privacy budget is <b>spent</b>, and it does not come back'],
      ['Post-processing', 'anything computed from a DP output is still DP', 'you can publish, model and re-plot freely'],
      ['Group privacy', 'a group of $k$ people gets $k\\varepsilon$', 'families and households are weaker-protected than individuals'],
      ['Robust to auxiliary data', 'the guarantee does not assume what the attacker knows', '<b>the property anonymisation lacks</b>']
    ])}
${H.table(['Mechanism', 'Noise', 'Gives'], [
      ['Laplace', 'scale $\\Delta_1/\\varepsilon$', 'pure $\\varepsilon$-DP for L1-sensitivity queries'],
      ['Gaussian', '$\\sigma = \\Delta_2\\sqrt{2\\ln(1.25/\\delta)}/\\varepsilon$', '$(\\varepsilon,\\delta)$-DP; composes better'],
      ['Exponential', 'sample proportional to $e^{\\varepsilon u/2\\Delta}$', 'private selection of a discrete choice'],
      ['Randomised response', 'flip the answer with probability $p$', 'local DP — no trusted curator needed at all']
    ])}

${H.lab('dp', 'The privacy/utility trade, measured', 'A real Laplace mechanism on a real histogram. Move ε and watch the noise swamp the signal — then watch what a larger dataset does to the same ε, which is the fact that makes DP practical at scale and hopeless for small groups.')}

<h2><span class="sn">6.8.3</span> DP-SGD, and what ε actually buys</h2>
${H.steps([
      '<b>Compute per-example gradients</b> — not the batch average. This is the expensive part; it defeats the usual batched backward pass and needs microbatching or functorch-style vectorisation.',
      '<b>Clip each one</b> to norm $C$: $g_i \\leftarrow g_i \\cdot \\min(1, C/\\|g_i\\|)$. This bounds one person’s influence, which is what makes the sensitivity finite.',
      '<b>Sum, add Gaussian noise</b> $\\mathcal{N}(0,\\sigma^2C^2I)$, divide by the batch size.',
      '<b>Account for the total budget</b> across all steps, with the moments accountant / Rényi-DP composition — naive composition over 10,000 steps would give a useless bound.'
    ])}
${H.table(['ε', 'Interpretation', 'Where you see it'], [
      ['0.1–1', 'strong; the guarantee is meaningful for an individual', 'census-scale aggregate statistics'],
      ['1–3', 'reasonable; the standard target for a trained model', 'DP-SGD research results'],
      ['3–10', 'weak but not vacuous', 'most deployed industrial systems'],
      ['&gt; 10', '$e^{10} \\approx 22{,}000$ — <b>the bound permits almost anything</b>', 'reported anyway, and worth being sceptical about']
    ], 'num')}
${H.flag('ε is a worst-case bound, not a measurement of realised privacy. A model at ε = 8 may in fact leak nothing detectable; a model at ε = 2 with a mis-specified sensitivity may leak plenty. Report ε <b>and</b> an empirical attack result — membership-inference AUC against a held-out set is the standard companion measure, and the pairing is what a serious privacy review asks for.')}
${H.worked('what DP-SGD costs', `
<p>Typical published numbers, CIFAR-10 from scratch: non-private ≈ 95%, DP-SGD at ε = 3 ≈ 70–80% with careful tuning. On <b>fine-tuning a pretrained model</b> the gap is far smaller — often 1–3 points — because the pretraining did the hard work on public data and only the final adaptation needs protecting.</p>
<p>Compute cost: 2–10× for per-example gradients, depending on implementation. <b>The practical recipe that actually ships is therefore: pretrain on public data, fine-tune with DP on the sensitive data.</b> It is also why large batches help — noise is added once per batch, so the signal-to-noise ratio improves with batch size, and DP-SGD recipes use batches an order of magnitude larger than usual.</p>`)}

${H.table(['Technique', 'Threat it addresses', 'Cost', 'Maturity'], [
      ['<b>DP-SGD</b>', 'memorisation, membership inference', 'accuracy + 2–10× compute', 'production-ready (Opacus, TF-Privacy)'],
      ['<b>Federated learning</b>', 'raw data leaving the device', 'communication, stragglers, non-IID data', 'deployed (keyboards, health)'],
      ['Secure aggregation', 'the server seeing individual updates', 'crypto overhead', 'deployed with federated learning'],
      ['Homomorphic encryption', 'the server seeing anything at all', '<b>100–10,000×</b>', 'narrow use only'],
      ['Secure multi-party computation', 'any single party seeing the data', '10–1000×', 'niche, growing'],
      ['Synthetic data', 'sharing a usable proxy', 'fidelity loss', '<b>only private if generated under DP</b> — otherwise it can memorise']
    ])}
${H.pitfall('Federated learning alone is not privacy. Gradients leak: <i>Deep Leakage from Gradients</i> reconstructed training images from shared updates. Federated learning must be combined with secure aggregation and DP noise before the word "private" is justified — and papers and product pages that use the word without both are making a claim they have not earned.')}
${H.history(`<p>Dwork, McSherry, Nissim and Smith's 2006 paper first formalised differential privacy, motivated directly by a string of re-identification failures like the ones in the table above — the field's founding insight was precisely that privacy needs to be a property of the mechanism, not an afterthought applied to a dataset. Narayanan and Shmatikov's 2008 de-anonymisation of the Netflix Prize dataset, published just as the formal theory was maturing, became the canonical demonstration of why the older approach failed in practice. Google (RAPPOR, 2014) and Apple (2016) were the first large-scale industrial deployments, both using local differential privacy — randomised response at the device, no trusted central curator needed — to collect usage statistics without a server ever seeing an individual's true data. Abadi et al.'s DP-SGD (2016) made differentially private deep learning practical by showing per-example gradient clipping and a tight privacy accountant could be combined into an ordinary training loop, which is the algorithm §6.8.3 walks through in detail.</p>`)}

${H.probe([
      ['State differential privacy.', 'A mechanism is $(\\varepsilon,\\delta)$-DP if for any two datasets differing in one record and any output set, the probabilities differ by at most a factor $e^\\varepsilon$ plus $\\delta$. The promise is to the individual, and it holds against any auxiliary information.'],
      ['Why does DP-SGD clip per-example gradients?', 'To bound each individual’s contribution, which makes the sensitivity finite and lets you calibrate noise to it. Clipping the batch gradient would not bound any single person’s influence.'],
      ['Is ε = 8 private?', 'Formally the bound permits a factor of ~3,000, so it is weak. In practice it may still resist real attacks. Report ε alongside an empirical membership-inference result rather than treating either alone as the answer.'],
      ['Is federated learning private?', 'Not by itself — gradients can be inverted to reconstruct inputs. It needs secure aggregation plus DP noise before the claim holds.'],
      ['Why does DP fine-tuning cost so much less than DP pretraining?', 'The pretraining used public data and required no protection; only the adaptation touches sensitive records, so far less signal has to survive the noise.']
    ])}`,
    labs: {
      dp: function (host) {
        const st = Viz.controls(host, [
          { k: 'eps', label: 'privacy budget ε', min: .05, max: 8, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'population size', min: 100, max: 200000, step: 100, value: 5000, fmt: v => v.toLocaleString() },
          { k: 'queries', label: 'queries answered from the same budget', min: 1, max: 40, step: 1, value: 1, fmt: v => v },
          { k: 'mech', label: 'mechanism', type: 'select', value: 'lap', options: [{ v: 'lap', t: 'Laplace (pure ε-DP)' }, { v: 'gau', t: 'Gaussian (ε,δ)-DP' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'per', label: 'ε per query', cls: 'key' },
          { k: 'noise', label: 'noise scale' },
          { k: 'err', label: 'typical error per bin', cls: 'bad' },
          { k: 'rel', label: 'relative error' },
          { k: 'verdict', label: 'usable?' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(29);
            const K = 10;
            const shape = [.03, .06, .11, .16, .18, .15, .12, .09, .06, .04];
            const truth = shape.map(s => Math.round(s * st.n));
            const epsPer = st.eps / st.queries;
            const sens = 1;  // one person changes one bin by one
            const noisy = truth.map(v => {
              const nz = st.mech === 'lap'
                ? Num.laplaceNoise(sens, epsPer, R)
                : R.normal(0, Num.gaussianSigma(sens, epsPer, 1e-5));
              return Math.max(0, v + nz);
            });
            const P = Viz.plot(ctx, w, h, {
              xd: [-.5, K - .5],
              yd: [0, Math.max.apply(null, truth.concat(noisy)) * 1.2],
              pad: { l: 56, r: 14, t: 16, b: 40 }
            }).frame({ xticks: Num.linspace(0, K - 1, K), xfmt: v => (20 + v * 8) + '', xlabel: 'age bucket', ylabel: 'count' });
            P.clip(() => {
              const bw = P.pw / K * .38;
              truth.forEach((v, i) => {
                ctx.fillStyle = T.c3; ctx.globalAlpha = .8;
                ctx.fillRect(P.x(i) - bw, P.y(v), bw, P.y(0) - P.y(v));
                ctx.fillStyle = T.c2; ctx.globalAlpha = .85;
                ctx.fillRect(P.x(i), P.y(noisy[i]), bw, P.y(0) - P.y(noisy[i]));
                ctx.globalAlpha = 1;
              });
            });
            let err = 0;
            truth.forEach((v, i) => { err += Math.abs(v - noisy[i]); });
            err /= K;
            const relErr = err / (Num.mean(truth) || 1);
            out({
              per: epsPer.toFixed(3),
              noise: st.mech === 'lap' ? 'b = ' + (1 / epsPer).toFixed(1) : 'σ = ' + Num.gaussianSigma(1, epsPer, 1e-5).toFixed(1),
              err: '±' + err.toFixed(1) + ' people',
              rel: (relErr * 100).toFixed(2) + '%',
              verdict: relErr < .01 ? '✓ noise is negligible' : relErr < .08 ? '~ usable for trends' : '✗ noise dominates'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c3)', t: 'true counts' }, { c: 'var(--c2)', t: 'differentially private release' }]);
        Viz.note(host, 'At ε = 1 on 5,000 people the noise is a few individuals per bin — invisible. Drop the population to 200 and the same ε makes the histogram unrecognisable. <b>That is the central asymmetry of differential privacy: the noise is calibrated to one person’s influence, so it is absolute, while the signal grows with $n$.</b> DP is nearly free at census scale and brutal for small subgroups — which is a fairness problem hiding inside a privacy mechanism, since the smallest subgroups are usually the ones that most need accurate statistics. Now raise the query count: the budget divides, and ten questions at ε = 1 total means ε = 0.1 each.');
      }
    },
    quiz: [
      {
        q: 'Differential privacy guarantees that…',
        options: ['the data cannot be decrypted', 'the output is nearly as likely whether or not any one individual participated', 'no identifiers are present', 'the model cannot be inverted'],
        answer: 1,
        why: 'The formal definition $P(M(D)\\in S)\\le e^{\\varepsilon}P(M(D\')\\in S)+\\delta$ is precisely a statement that the output distribution barely changes whether or not any one individual\'s record is in the dataset, which is what §6.8.2 reads as a promise to that individual: whatever gets published would have been almost exactly as likely to be published had they opted out entirely. "No identifiers are present" is the tempting wrong answer because it describes what conventional anonymisation attempts, and §6.8.1\'s table of incidents — the Massachusetts GIC re-identification, the Netflix Prize de-anonymisation — exists specifically to show that removing identifiers is not the same guarantee at all, since auxiliary data can re-identify a record even with no names or IDs left in it. "The data cannot be decrypted" confuses differential privacy with cryptography; DP makes no claim about concealing the data itself; it constrains what a computation on the data is allowed to reveal about any one person\'s presence in it. "The model cannot be inverted" describes a specific attack (model inversion) rather than the general property DP provides, and a DP guarantee actually holds against inversion, membership inference, and any other attack simultaneously, precisely because it does not depend on enumerating attacks in advance. The general principle, stated as the key insight of §6.8.1, is that privacy is a property of the release mechanism, not of the dataset, robust to whatever auxiliary information an attacker might have now or acquire later.'
      },
      {
        q: 'DP-SGD clips gradients per example in order to…',
        options: ['prevent exploding gradients', 'bound each individual’s influence so the noise can be calibrated to it', 'speed up training', 'improve convergence'],
        answer: 1,
        why: 'Clipping each per-example gradient to norm $C$ — $g_i \\leftarrow g_i\\cdot\\min(1,C/\\|g_i\\|)$ — puts a hard, known ceiling on how much any single training example can move the aggregated gradient, and that bounded sensitivity is exactly what lets the noise added afterwards be calibrated to a specific, finite quantity rather than an unbounded one. "Prevent exploding gradients" is a tempting misreading precisely because gradient clipping already has a well-known role elsewhere in this course as a training stabiliser, and it is easy to assume DP-SGD is reusing that same trick — but ordinary clipping bounds the batch-average gradient\'s norm, while DP-SGD clips before averaging, specifically so one person\'s contribution to that average is bounded, a different target entirely. "Speed up training" and "improve convergence" both describe things per-example clipping does not do and in practice actively works against, since computing and clipping gradients individually is the expensive part of DP-SGD, defeating the usual batched backward pass and requiring microbatching, exactly as §6.8.3 states. The general principle behind every differential-privacy mechanism in this section — Laplace noise, Gaussian noise, DP-SGD alike — is that calibrated noise can only protect what has a known, bounded sensitivity; clip first to create that bound, then add noise sized to it, never noise without a bound behind it.'
      },
      {
        q: 'Federated learning without secure aggregation and DP noise is…',
        options: ['fully private', 'not private — gradients can be inverted to reconstruct training inputs', 'private only for images', 'equivalent to DP'],
        answer: 1,
        why: 'Federated learning only ensures raw data never leaves the device — it says nothing about what the shared gradient updates themselves reveal, and Deep Leakage from Gradients demonstrated concretely that training images can be reconstructed directly from those updates, which is exactly why §6.8.3\'s table lists federated learning\'s cost as "communication, stragglers, non-IID data" while listing a completely separate threat, "the server seeing individual updates", against secure aggregation instead. "Fully private" is the tempting default because federated learning is often marketed as a privacy technology in its own right, and moving the data\'s physical location does feel like the main privacy win — but §6.8.3\'s pitfall states this plainly: federated learning must be combined with secure aggregation and DP noise before the word "private" is earned, and a product page using the word without both is making a claim it has not earned. "Private only for images" narrows the vulnerability to one modality, but gradient inversion is a property of gradients carrying information about the inputs that produced them, not of pixels specifically — the same leakage risk applies to any data type a model trains on. "Equivalent to DP" conflates two different kinds of guarantee: federation is an architecture describing where computation happens, while differential privacy is a mathematical bound on what any output can reveal, and one does not imply the other. The general principle is the same one running through this whole section: a technique that changes where data physically sits is not automatically a technique that bounds what an observer can learn, and only the latter is a genuine privacy guarantee.'
      },
      {
        q: 'An ε of 10 means the probability ratio bound is about…',
        options: ['10×', '100×', '22,000×', '2×'],
        answer: 2,
        why: 'The DP bound is literally $e^{\\varepsilon}$, and $e^{10}\\approx 22{,}026$ — a number you can check on any calculator — so at ε = 10 the guarantee only promises that one outcome is at most about 22,000 times more or less likely with a given person\'s data included than without it, which in practice permits almost anything, exactly as the ε-interpretation table in §6.8.3 states outright. "10×" is the tempting linear-thinking error: treating ε itself as the multiplicative bound rather than $e^\\varepsilon$, an easy mistake since ε is the number everyone actually writes down and discusses, while the bound it controls is buried inside an exponential. "100×" repeats the same mistake with $\\varepsilon^2$ instead of ε, a slightly less naive guess but still missing that the relationship is exponential, not polynomial, in ε. "2×" is not close to any natural miscalculation from the formula and understates the bound so drastically that at ε = 10 it would make the guarantee sound strong rather than nearly vacuous — precisely backwards. The general principle, which §6.8.3\'s flag insists on, is that ε is a worst-case bound rather than a measurement of realised privacy, so a serious privacy review reports ε alongside an empirical result like membership-inference AUC rather than treating the number on its own as reassuring.'
      }
    ],
    cards: [
      { q: 'The DP definition', a: '$P(M(D)\\in S)\\le e^\\varepsilon P(M(D\')\\in S)+\\delta$ for datasets differing in one record. A promise to the individual.' },
      { q: 'DP-SGD, four steps', a: 'Per-example gradients → clip to norm C → sum + Gaussian noise → account for the budget with Rényi/moments composition.' },
      { q: 'Why anonymisation fails', a: 'Re-identification via auxiliary data: ZIP + birthdate + sex identifies 87% of Americans; Netflix ratings matched public IMDb reviews.' },
      { q: 'The practical DP recipe', a: 'Pretrain on public data, DP-fine-tune on sensitive data, use very large batches. The accuracy gap drops from ~20 points to 1–3.' },
      { q: 'Federated learning caveat', a: 'Gradients leak. Needs secure aggregation plus DP before the word "private" is earned.' }
    ]
  });
})();
