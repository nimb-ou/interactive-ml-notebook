/* ============================================================
   PART 6 — Beyond the notebook: the deliberate exclusions, restored
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 6.1 */
  ML.section({
    id: 'rl', track: 'frontier', num: '6.1',
    title: 'Reinforcement learning, from MDPs up',
    lede: 'Part 4 used RL as a post-training tool without ever defining it — a clipped objective here, a KL penalty there, taken on faith. This section builds the whole object it came from, from a single robot deciding what to do next, and once you have it, PPO, GRPO and RLVR stop being acronyms and start being special cases of one idea.',
    html: `
<h2><span class="sn">6.1.1</span> Why this is a different kind of learning problem</h2>
<p>Every model in Parts 2 and 3 learned from a fixed table of correct answers. Show it an email, it predicts spam or not; show it the true label, it adjusts. The learning signal is immediate, and it is unambiguous about which prediction it is correcting.</p>
<p>Now imagine training a robot to stack boxes, or an agent to play chess, or a language model to write a helpful answer. Nobody hands it the "correct" action for each situation, because in most situations there is no single correct action — only actions that turn out well or badly once the whole episode plays out. The chess agent finds out it lost <i>forty moves</i> after the mistake that caused it. Which of those forty moves was wrong? Maybe none of them individually; maybe a plan that looked fine for thirty-nine moves collapsed on the fortieth. That is the <b>credit assignment problem</b>, and it is the reason reinforcement learning needs its own machinery: a way to turn a single delayed number — did this episode go well? — into a learning signal for every decision that contributed to it.</p>
${H.analogy(`<p>Learning to ride a bicycle has the same shape. Nobody tells you the correct handlebar angle at each instant; you find out you are doing well by <i>not falling over</i>, and that single fact — still upright, half a second later — has to retroactively teach every small correction that led to it. You cannot learn to ride from a labelled dataset of "correct" handlebar positions, because there is no such dataset; there is only the consequence, arriving late and summarising a whole sequence of choices at once. Reinforcement learning is the formal version of turning that lagging consequence into a lesson for each individual choice.</p>`)}

<h2><span class="sn">6.1.2</span> The Markov decision process</h2>
<p>The formal object that captures "sequential decisions with delayed consequences" is a <b>Markov decision process</b> (MDP), and it has five parts. There is a set of <b>states</b> $s$ — everything about the situation that matters for deciding what to do next. There are <b>actions</b> $a$ — the choices available in a given state. There is a <b>transition function</b> $P(s'\\mid s,a)$, read "the probability of landing in state s-prime, given you were in state s and took action a" — the physics or rules of the world, which may itself be random. There is a <b>reward</b> $R(s,a)$, a number handed out for taking that action in that state. And there is a <b>discount</b> $\\gamma$ (the Greek letter gamma), a number between 0 and 1 that says how much a reward loses value for arriving one step later.</p>
<p>"Markov" is doing real work in the name: it means the state $s$ must summarise everything relevant about the history, so that $P(s'\\mid s,a)$ does not need to know how you arrived at $s$. In chess, the board position is Markov — you do not need the game's move history to know what is legal next. In a customer-support bot, "the last message" is usually <i>not</i> Markov — you also need what was said three turns ago — which is exactly the problem a longer context window, or an explicit state summary, is there to fix.</p>
<p>A <b>policy</b> $\\pi(a\\mid s)$ is a rule for choosing actions: given a state, it names a probability for each available action. The <b>value</b> of a state under a policy is the total discounted reward you expect to collect from here on, if you keep following that policy:</p>
$$V^\\pi(s) = \\mathbb{E}\\left[\\sum_{t=0}^\\infty \\gamma^t R_t \\;\\middle|\\; s_0 = s\\right]$$
<p>Read the right-hand side as: starting from state $s$, add up every future reward $R_t$, but shrink the reward $t$ steps away by a factor of $\\gamma^t$, and take the expectation because both the policy and the world may be random. The discount does two separate jobs, and it is worth keeping them apart. First, a purely practical one: without it, an infinite sequence of positive rewards would sum to infinity, and you cannot compare two infinities to decide which policy is better. Second, a modelling one: $\\gamma$ encodes how much the future is worth relative to the present, exactly like a financial discount rate. A $\\gamma$ near 1 makes the agent patient and willing to sacrifice now for a much later payoff; a $\\gamma$ near 0 makes it myopic, caring almost only about the immediate reward. Choosing $\\gamma$ is choosing the agent's horizon, and it is a decision you make, not a constant the maths hands you.</p>

<h3>The Bellman equation: value, defined in terms of itself</h3>
<p>Values for every state could in principle be measured by simulation — play out the policy from each state many times and average the discounted return. But there is a shortcut, and it is the single most important equation in the subject. Split the infinite sum in $V^\\pi(s)$ into "the very next reward" and "everything after that":</p>
${H.deriv('the Bellman equation, from the definition of value', [
      ['$V^\\pi(s) = \\mathbb{E}\\big[R_0 + \\gamma R_1 + \\gamma^2 R_2 + \\cdots \\mid s_0=s\\big]$', 'This is just the definition of $V^\\pi(s)$ written out one term at a time.'],
      ['$= \\mathbb{E}\\big[R_0 + \\gamma(R_1 + \\gamma R_2 + \\cdots) \\mid s_0=s\\big]$', 'Factor a $\\gamma$ out of every term except the first. Nothing has changed numerically — this is pure algebra on the same infinite sum.'],
      ['$= \\mathbb{E}\\big[R_0 + \\gamma\\, V^\\pi(s_1) \\mid s_0=s\\big]$', 'The bracketed sum is, by definition, the value of whatever state $s_1$ you land in next — the same object $V^\\pi$, just evaluated one step later. The infinite sum has collapsed into one step plus a recursive call.'],
      ['$V^\\pi(s) = \\sum_a \\pi(a\\mid s)\\Big[R(s,a) + \\gamma\\sum_{s\'}P(s\'\\mid s,a)\\,V^\\pi(s\')\\Big]$', 'Expand the outer expectation over which action the policy picks, and the inner one over which state the environment transitions to. This is the Bellman equation for a fixed policy.']
    ], 'For the <i>optimal</i> policy, replace "average over the policy\'s action choice" with "take the best one available", which is where the familiar form comes from.')}
$$V^*(s) = \\max_a \\left[R(s,a) + \\gamma\\sum_{s'}P(s'\\mid s,a)V^*(s')\\right]$$
${H.intuition(`<p>Read this the way you would read a recursive function, because that is exactly what it is. "The best value achievable from here is: try every action, and for each one add its immediate reward to the discounted value of wherever it leads — then take the best of those totals." It is a definition of $V^*$ in terms of $V^*$, which sounds circular until you notice it is well-founded: a state one step from the end has a value you can compute directly, and every other state's value is built from states closer to the end. <b>Value iteration</b> — the algorithm in the lab below — does exactly this: start with a guess of zero everywhere, and repeatedly apply the right-hand side using the current guess for $V^*$ on the right. Each sweep the guess gets more accurate, because it is contaminated by fewer and fewer wrong initial guesses, and the discount $\\gamma < 1$ guarantees the whole process is a contraction that converges to a single fixed point.</p>`)}

${H.lab('gridworld', 'Value iteration and Q-learning, on a gridworld', 'You are looking at an 8×6 grid with a goal cell (green, reward +1), a trap (red, reward −1), and walls the agent cannot enter. Each cell shows the value the current algorithm has assigned it, and an arrow shows the greedy action from that cell — the policy implied by the current values. Run value iteration to convergence and watch the values propagate outward from the goal, one Bellman backup at a time: cells near the goal light up first, and the wavefront spreads exactly as far as $\\gamma$ lets it. Then switch to Q-learning, which never sees the transition model $P(s\'\\mid s,a)$ at all — it can only wander the grid and learn from what actually happens. Set ε (its exploration rate) to 0 and watch it fail to ever find the goal by accident, which is the exploration problem made visible: without some randomness in its own behaviour, a model-free learner can starve itself of exactly the experience it needs.')}

<h2><span class="sn">6.1.3</span> Model-free control: learning without the transition model</h2>
<p>Value iteration needs $P(s'\\mid s,a)$ — the actual rules of the world — which you rarely have for anything interesting. <b>Model-free</b> methods learn purely from experienced transitions $(s,a,r,s')$, and the table below is arranged from the ones that only estimate values to the ones that act directly on the policy.</p>
${H.table(['Method', 'Update', 'Character'], [
      ['<b>Q-learning</b> (off-policy)', '$Q(s,a) \\mathrel{+}= \\alpha[r + \\gamma\\max_{a\'}Q(s\',a\') - Q(s,a)]$', 'Learns the value of the <i>optimal</i> policy while behaving however it likes (e.g. ε-greedy exploration)'],
      ['<b>SARSA</b> (on-policy)', '$Q(s,a) \\mathrel{+}= \\alpha[r + \\gamma Q(s\',a\') - Q(s,a)]$', 'Learns the value of the policy it actually follows — safer near cliffs, because it accounts for its own exploratory mistakes'],
      ['<b>Policy gradient</b>', '$\\nabla J = \\mathbb{E}[\\nabla\\log\\pi(a\\mid s)\\,A(s,a)]$', 'Optimises the policy directly rather than a value table; the natural choice when actions are continuous'],
      ['<b>Actor–critic</b>', 'a policy (actor) plus a learned value baseline (critic)', 'Lower-variance policy gradients — the whole PPO family'],
      ['<b>PPO</b>', 'clipped ratio objective (§4.12)', 'The RLHF workhorse; GRPO drops the critic entirely and uses a sampled group\'s mean reward as the baseline instead']
    ])}
<p>Q-learning and SARSA look almost identical — swap $\\max_{a'}Q(s',a')$ for $Q(s',a')$ at the action actually taken next — but that one substitution changes which policy is being evaluated. Walking along a cliff edge with a chance of a random slip, Q-learning learns the value of walking right at the edge (optimal if you never slip), while SARSA learns the value of walking there <i>given that you sometimes do slip</i>, and so prefers a safer route one step back. Off-policy versus on-policy is precisely this distinction: does the update assume you act optimally from here, or does it account for the policy you are actually running, mistakes included?</p>

<h3>Policy gradient: the log-derivative trick</h3>
<p>The value-based methods above learn $Q(s,a)$ and act greedily with respect to it — fine for a handful of discrete actions, hopeless when actions are continuous (a steering angle, a token's next-word logits over 100,000 options). <b>Policy gradient</b> methods instead parametrise the policy directly as $\\pi_\\theta(a\\mid s)$ and climb the gradient of expected return. The formula $\\nabla J = \\mathbb{E}[\\nabla\\log\\pi(a\\mid s)\\,A(s,a)]$ looks unmotivated until you see where the $\\log$ came from.</p>
${H.deriv('the policy gradient theorem, via the log-derivative trick', [
      ['$J(\\theta) = \\mathbb{E}_{a\\sim\\pi_\\theta}[R(a)]= \\sum_a \\pi_\\theta(a)R(a)$', 'The objective: expected reward under the current policy. Write it as a sum over actions to make the dependence on $\\theta$ explicit, so it can be differentiated directly.'],
      ['$\\nabla_\\theta J = \\sum_a \\nabla_\\theta\\pi_\\theta(a)\\,R(a)$', 'Differentiate under the sum. $R(a)$ does not depend on $\\theta$, so it is untouched — but $\\nabla_\\theta\\pi_\\theta(a)$ is a gradient of a probability, which you cannot sample, because sampling needs a distribution to sample <i>from</i>, and a bare gradient is not one.'],
      ['$\\nabla_\\theta\\pi_\\theta(a) = \\pi_\\theta(a)\\,\\dfrac{\\nabla_\\theta\\pi_\\theta(a)}{\\pi_\\theta(a)} = \\pi_\\theta(a)\\,\\nabla_\\theta\\log\\pi_\\theta(a)$', 'Multiply and divide by $\\pi_\\theta(a)$ — legal whenever $\\pi_\\theta(a) > 0$ — and recognise the fraction as the derivative of $\\log\\pi_\\theta(a)$, since $\\frac{d}{dx}\\log f(x) = f\'(x)/f(x)$ by the chain rule.'],
      ['$\\nabla_\\theta J = \\sum_a \\pi_\\theta(a)\\,\\nabla_\\theta\\log\\pi_\\theta(a)\\,R(a) = \\mathbb{E}_{a\\sim\\pi_\\theta}\\big[\\nabla_\\theta\\log\\pi_\\theta(a)\\,R(a)\\big]$', 'The $\\pi_\\theta(a)$ that reappeared out front turns the sum back into an expectation under the policy — which <i>can</i> be estimated from sampled actions, unlike the gradient in line 2.']
    ], 'This is why every policy-gradient method computes $\\nabla\\log\\pi(a\\mid s)$ rather than $\\nabla\\pi(a\\mid s)$: the log is what converts an unsampleable gradient of a density into a sampleable expectation. Swapping raw reward $R(a)$ for the advantage $A(s,a) = Q(s,a) - V(s)$ — how much better this action was than average — does not change the expectation (subtracting a state-only baseline has zero mean) but drastically reduces its variance, which is the entire reason actor-critic methods exist.')}
${H.practice(`<p>That variance reduction is not a nicety, it is the difference between a method that trains and one that does not. Raw-reward policy gradient (REINFORCE) is famous for needing enormous batch sizes because a single good episode and a single bad one can produce gradient estimates that differ by orders of magnitude. A learned baseline $V(s)$ — the "critic" in actor-critic — subtracts off the part of the reward that had nothing to do with this particular action, leaving a signal that is centred near zero and far less noisy. PPO's clipped objective (§4.12) is a further variance-and-stability fix on top: it prevents any single update from moving the policy so far that the old data used to estimate the gradient becomes irrelevant to the new policy.</p>`)}
<p>The exploration–exploitation problem from §1.6 and §6.6 reappears here in full force, because a policy that has converged to one plausible answer has no mechanism to notice a better one exists. ε-greedy, optimistic initialisation of $Q$, and entropy bonuses on the policy are the standard answers, and the last is precisely why RLHF objectives so often carry an explicit entropy or KL term: without it, the policy collapses onto its most confident completion and stops generating anything else, which starves the training signal exactly as a Q-learner with ε = 0 starves itself in the lab above.</p>

${H.history(`<p>The field's history is a story of removing assumptions one at a time. Sutton's temporal-difference learning (1988) first showed you could learn value estimates from bootstrapped guesses rather than waiting for an episode to finish. Watkins' Q-learning (1989) showed the learned values could be optimal even while the agent explored suboptimally — the off-policy property above. Tesauro's TD-Gammon (1992) was the first eye-catching demonstration, reaching near-expert backgammon from self-play alone. Deep Q-Networks (Mnih et al., 2013–2015) combined the old Q-learning update with a convolutional network as the function approximator and matched human performance on Atari from raw pixels, which is the moment "deep RL" became a field rather than a curiosity. AlphaGo (Silver et al., 2016) added Monte Carlo tree search on top and beat the world Go champion. Schulman et al.'s PPO (2017) then simplified the policy-gradient side into something stable enough to be used as an industrial default — which is precisely the algorithm that, five years later, Ouyang et al.'s InstructGPT (2022) repurposed for RLHF, and that DeepSeek's GRPO (2024) simplified further by deleting the critic network altogether. Each step kept the Bellman recursion; what changed was what had to be known in advance to use it.</p>`)}

<h2><span class="sn">6.1.4</span> The link back to Part 4: RLHF is an MDP with a one-step episode</h2>
<p>Everything above was built for problems with long horizons — a game with many moves, a robot with many actions before a task completes. Post-training a language model with human feedback turns out to be the same object with almost everything simplified away. In RLHF the "environment" is a prompt, the "action" is an entire completion generated in one shot, and the "reward" is a score from a preference model or a verifier. The episode is exactly one step long: state, action, reward, done.</p>
${H.table(['MDP concept', 'RLHF instance'], [
      ['State $s$', 'the prompt (plus, in a multi-turn setting, the conversation so far)'],
      ['Action $a$', 'the full completion, sampled token by token but scored as one unit'],
      ['Reward $R(s,a)$', 'a learned preference/reward model\'s score, or — in RLVR — a verifier checking the answer against ground truth'],
      ['Transition $P(s\'\\mid s,a)$', 'trivial: the episode ends. There is no $s\'$ to bootstrap into'],
      ['Discount $\\gamma$', 'irrelevant — nothing to discount when the episode is one step']
    ])}
<p>That collapse is what makes the whole apparatus tractable at LLM scale. There is no bootstrapping across time, because $V^*(s')$ never has to be estimated for a next state that does not exist. There is no discount to argue about. And the one genuinely hard part of policy gradients — the variance of the advantage estimate — is handled entirely by the choice of baseline: a learned value network trained alongside the policy in PPO, or simply the mean reward of a sampled group of completions to the same prompt in GRPO. Seeing that correspondence is what makes §4.12 feel inevitable rather than arbitrary: PPO for language models is not a different algorithm bolted on to transformers, it is the same clipped-ratio, advantage-weighted update derived above, applied to episodes so short that half the usual machinery has nothing to do.</p>
${H.flag('RLVR (reinforcement learning from verifiable rewards) is sometimes described as "not really RL" because the reward is a deterministic checker rather than a learned model. That framing undersells it: the MDP structure — one-step episode, policy gradient with a baseline — is identical. What changed is the source of $R(s,a)$, from a neural network trained on preferences to a program that checks an answer against a known solution. The optimisation is the same; the reward signal is just less noisy, which is why RLVR trains more stably than preference-model RLHF and why it works best on domains — maths, code — where "correct" has a checkable definition.')}

${H.probe([
      ['What does the Bellman equation say, in one sentence?', 'The value of a state is the best immediate reward available, plus the discounted value of whichever state that action leads to — a recursive definition that bottoms out at terminal states.'],
      ['Q-learning vs SARSA — what is the actual difference?', 'Both are one-step TD updates on $Q(s,a)$. Q-learning bootstraps off $\\max_{a\'}Q(s\',a\')$ — the value of acting optimally next — so it learns the optimal policy even while exploring randomly (off-policy). SARSA bootstraps off $Q(s\',a\')$ for the action actually taken next, so it learns the value of the policy it is actually running, exploration mistakes included (on-policy), which makes it more cautious near irreversible bad outcomes.'],
      ['Derive, in outline, why policy gradients use $\\nabla\\log\\pi$ rather than $\\nabla\\pi$.', 'You cannot sample from a bare gradient of a probability. Multiplying and dividing by $\\pi_\\theta(a)$ turns $\\nabla_\\theta\\pi_\\theta(a)$ into $\\pi_\\theta(a)\\nabla_\\theta\\log\\pi_\\theta(a)$, and the leftover $\\pi_\\theta(a)$ converts the sum back into an expectation over actions drawn from the policy — something you can estimate from sampled rollouts.'],
      ['How does RLHF map onto an MDP, and why does that make it easy?', 'One-step episodes: prompt = state, completion = action, reward or preference model = reward. No bootstrapping, no discount to tune — the only real design choice left is the variance-reducing baseline, which is a learned critic in PPO and a sampled group\'s mean in GRPO.'],
      ['Why does a policy-gradient method need an entropy bonus or KL penalty?', 'Without one, the policy can collapse onto its single most confident output and stop generating the variety needed to discover anything better — the same starvation a Q-learner with ε = 0 shows in the gridworld lab.']
    ])}`,
    labs: {
      gridworld: function (host) {
        const W = 8, H = 6;
        const walls = new Set(['2,1', '2,2', '2,3', '5,2', '5,3', '5,4']);
        const goal = '7,0', trap = '6,3';
        const st = Viz.controls(host, [
          { k: 'algo', label: 'algorithm', type: 'buttons', value: 'vi', options: [{ v: 'vi', t: 'value iteration' }, { v: 'ql', t: 'Q-learning' }] },
          { k: 'gamma', label: 'discount γ', min: .5, max: .99, step: .01, value: .92, fmt: v => v.toFixed(2) },
          { k: 'step', label: 'step cost', min: -.2, max: 0, step: .01, value: -.04, fmt: v => v.toFixed(2) },
          { k: 'eps', label: 'ε (exploration, Q-learning)', min: 0, max: .8, step: .05, value: .25, fmt: v => v.toFixed(2) }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'iter', label: 'iterations / episodes', cls: 'key' }, { k: 'delta', label: 'largest value change' },
          { k: 'start', label: 'value of the start state' }, { k: 'converged', label: 'converged?' }
        ]);
        let V, Q, iters = 0, lastDelta = 1;
        const key = (x, y) => x + ',' + y;
        function reset() {
          V = {}; Q = {}; iters = 0; lastDelta = 1;
          for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
            V[key(x, y)] = 0;
            Q[key(x, y)] = [0, 0, 0, 0];
          }
          S.redraw();
        }
        const moves = [[0, -1], [1, 0], [0, 1], [-1, 0]];
        function rewardAt(k) { return k === goal ? 1 : k === trap ? -1 : st.step; }
        function stepVI() {
          let delta = 0;
          const nV = {};
          for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
            const k = key(x, y);
            if (walls.has(k) || k === goal || k === trap) { nV[k] = rewardAt(k) === st.step ? 0 : rewardAt(k); continue; }
            let best = -Infinity;
            moves.forEach(m => {
              const nx = Math.max(0, Math.min(W - 1, x + m[0])), ny = Math.max(0, Math.min(H - 1, y + m[1]));
              const nk = walls.has(key(nx, ny)) ? k : key(nx, ny);
              best = Math.max(best, st.step + st.gamma * V[nk]);
            });
            nV[k] = best;
            delta = Math.max(delta, Math.abs(best - V[k]));
          }
          V = nV; iters++; lastDelta = delta;
        }
        function episodeQL() {
          const R = Num.rng(Math.floor(Math.random() * 1e6));
          let x = 0, y = H - 1, steps = 0;
          let delta = 0;
          while (steps++ < 200) {
            const k = key(x, y);
            if (k === goal || k === trap) break;
            const a = R() < st.eps ? R.int(4) : Q[k].indexOf(Math.max.apply(null, Q[k]));
            const m = moves[a];
            let nx = Math.max(0, Math.min(W - 1, x + m[0])), ny = Math.max(0, Math.min(H - 1, y + m[1]));
            if (walls.has(key(nx, ny))) { nx = x; ny = y; }
            const nk = key(nx, ny);
            const r = rewardAt(nk);
            const target = r + ((nk === goal || nk === trap) ? 0 : st.gamma * Math.max.apply(null, Q[nk]));
            const old = Q[k][a];
            Q[k][a] += .25 * (target - Q[k][a]);
            delta = Math.max(delta, Math.abs(Q[k][a] - old));
            x = nx; y = ny;
          }
          iters++; lastDelta = delta;
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const cell = Math.min((w - 40) / W, (h - 50) / H);
            const ox = 20, oy = 24;
            const val = (x, y) => st.algo === 'vi' ? V[key(x, y)] : Math.max.apply(null, Q[key(x, y)]);
            let lo = 0, hi = 0;
            for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) { const v = val(x, y); lo = Math.min(lo, v); hi = Math.max(hi, v); }
            for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
              const k = key(x, y), px = ox + x * cell, py = oy + y * cell;
              if (walls.has(k)) { ctx.fillStyle = T.dark ? '#2a3050' : '#c9cee0'; ctx.fillRect(px, py, cell - 2, cell - 2); continue; }
              const v = val(x, y);
              const t = (v - lo) / ((hi - lo) || 1);
              ctx.fillStyle = k === goal ? 'rgba(60,180,110,.85)' : k === trap ? 'rgba(220,80,70,.85)'
                : 'rgba(' + Math.round(70 + 120 * t) + ',' + Math.round(90 + 90 * t) + ',' + Math.round(200 - 40 * t) + ',' + (0.12 + 0.7 * t) + ')';
              ctx.fillRect(px, py, cell - 2, cell - 2);
              ctx.strokeStyle = T.line; ctx.strokeRect(px, py, cell - 2, cell - 2);
              ctx.fillStyle = T.text; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(v.toFixed(2), px + cell / 2 - 1, py + cell / 2 - 5);
              // policy arrow
              if (k !== goal && k !== trap) {
                let bestA = 0, bestV = -Infinity;
                moves.forEach((m, ai) => {
                  const nx = Math.max(0, Math.min(W - 1, x + m[0])), ny = Math.max(0, Math.min(H - 1, y + m[1]));
                  const nk = walls.has(key(nx, ny)) ? k : key(nx, ny);
                  const q = st.algo === 'vi' ? V[nk] : Q[k][ai];
                  if (q > bestV) { bestV = q; bestA = ai; }
                });
                const arrows = ['↑', '→', '↓', '←'];
                ctx.fillStyle = T.muted; ctx.font = '13px ui-sans-serif';
                ctx.fillText(arrows[bestA], px + cell / 2 - 1, py + cell / 2 + 9);
              }
            }
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('green = goal (+1) · red = trap (−1) · arrows = greedy policy · start is bottom-left', ox, oy + H * cell + 8);
            out({
              iter: iters, delta: lastDelta.toFixed(4),
              start: val(0, H - 1).toFixed(3),
              converged: lastDelta < .001 ? 'yes' : 'not yet'
            });
          }
        });
        Viz.buttons(host, [
          { label: st.algo === 'vi' ? 'One sweep' : 'One episode', primary: true, on: () => { st.algo === 'vi' ? stepVI() : episodeQL(); S.redraw(); } },
          { label: 'Run 30', on: () => { for (let i = 0; i < 30; i++) st.algo === 'vi' ? stepVI() : episodeQL(); S.redraw(); } },
          { label: 'Run 300', on: () => { for (let i = 0; i < 300; i++) st.algo === 'vi' ? stepVI() : episodeQL(); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        reset();
        Viz.note(host, 'Value iteration knows the transition model and propagates values outward from the goal one sweep at a time. Q-learning knows nothing and must bump into the goal by accident before any value exists — set ε to 0 and watch it fail to find the goal at all. That is the exploration problem, and it is why every RL system needs an answer to it.');
      }
    },
    quiz: [
      {
        q: 'The discount factor γ…',
        options: ['controls the learning rate', 'makes the infinite return converge and encodes how much the future matters', 'sets the exploration rate', 'normalises rewards'],
        answer: 1,
        why: 'γ is doing two separate jobs at once, and the correct option is the only one that names both. Without $\\gamma < 1$, an infinite sum of positive rewards diverges to infinity, so two policies that both "score infinite" could never be compared — that alone would make $V^\\pi(s)$ meaningless as defined in §6.1.2. Separately, γ is a genuine modelling choice about the agent\'s patience: near 1 it sacrifices reward now for a much larger payoff later, near 0 it cares almost only about the immediate step. "Controls the learning rate" is the tempting mix-up, because γ and a learning rate $\\alpha$ are both single numbers between 0 and 1 sitting in the same Q-learning update line — but $\\alpha$ sets how fast an estimate is corrected towards a new target, while γ sets how far into the future that target looks, and confusing the two is a common first mistake. The general principle, built out across §6.1.2\'s derivation of the Bellman equation, is that a discount is not bolted on afterwards for numerical convenience — it is where the horizon of the whole decision problem gets fixed.'
      },
      {
        q: 'Why does policy gradient use $\\nabla\\log\\pi(a\\mid s)$ rather than $\\nabla\\pi(a\\mid s)$?',
        options: ['it is numerically more stable, nothing more', 'multiplying and dividing by π(a) turns an unsampleable gradient of a probability into a sampleable expectation under the policy', 'log makes the reward additive', 'it is a historical convention with no derivation'],
        answer: 1,
        why: 'The correct option names the actual mechanical problem: $\\nabla_\\theta\\pi_\\theta(a)$ is a gradient of a probability, and there is no way to sample from a bare gradient, because sampling needs a distribution to sample from and a gradient is not one. Multiplying and dividing by $\\pi_\\theta(a)$ rewrites it as $\\pi_\\theta(a)\\nabla\\log\\pi_\\theta(a)$, and the leftover $\\pi_\\theta(a)$ turns the whole sum back into $\\mathbb{E}_{a\\sim\\pi_\\theta}[\\nabla\\log\\pi_\\theta(a)R(a)]$ — an expectation you can estimate from rollouts you actually collect. "Numerically more stable, nothing more" is tempting because logs genuinely do tame numerical ranges elsewhere in this course (log-likelihoods, log-sum-exp), so it is easy to assume that is the whole story here too — but the trick would be needed even in infinite precision, because the underlying problem is unsampleability, not overflow. "Historical convention with no derivation" undersells a result that has an exact four-line proof, given in full in §6.1.2. The general principle recurs whenever a gradient meets an expectation under the very distribution being optimised: you cannot differentiate through a sample, so some version of this trick is always what moves the parameter dependence back inside an expectation you can draw from.'
      },
      {
        q: 'RLHF maps onto an MDP as…',
        options: ['a long episode with many states', 'a one-step episode: prompt = state, completion = action, reward model or verifier = reward', 'an unsupervised problem', 'a bandit with no context'],
        answer: 1,
        why: 'Mapping "prompt" to state and "entire completion" to a single action is what makes the episode exactly one step long — state, action, reward, done — and that collapse, spelled out in §6.1.4\'s table, is why RLHF needs none of the long-horizon machinery built earlier in the section: no bootstrapping across time, because there is no next state to bootstrap into, and no discount to argue about, because there is nothing to discount. "A long episode with many states" is the tempting default precisely because every other example of RL in this section — the gridworld, chess, a robot completing a task — genuinely does have many steps, so it is natural to assume language-model RL must too; the surprise §6.1.4 is built around is that it does not. "A bandit with no context" gets closer but still misses the point: a bandit has no state transitions either, but RLHF genuinely does have a state, the prompt, that the policy conditions its action on, which is exactly what makes it an MDP with a trivial transition rather than a contextual bandit — and it is certainly not unsupervised, since the reward model or verifier is a supervisory signal doing real work every step. The general principle is that reducing an MDP\'s horizon to one step does not change what kind of problem it is, only how much of the general machinery is left with something to do — the one real design choice remaining is the variance-reducing baseline, a learned critic in PPO or a sampled group\'s mean in GRPO.'
      },
      {
        q: 'Q-learning is off-policy because…',
        options: ['it uses a neural network', 'its update bootstraps off the best possible next action, not the one actually taken, so it learns the optimal policy\'s value while behaving differently', 'it ignores rewards', 'it never explores'],
        answer: 1,
        why: 'The correct option names the exact one-term difference the section builds the whole off-policy/on-policy distinction from: Q-learning\'s update bootstraps off $\\max_{a\'}Q(s\',a\')$, the value of the best action available next, regardless of what the agent actually does — so the values it learns describe the optimal policy even while ε-greedy exploration is steering its actual behaviour elsewhere. "It never explores" is backwards: Q-learning explores exactly as much as its behaviour policy dictates (ε-greedy, say) — what makes it off-policy is that its updates ignore that exploration when computing the target, not that exploration is absent. SARSA is the useful contrast named in §6.1.3: its update bootstraps off $Q(s\',a\')$ for the action actually taken next, so it learns the value of the policy it is really running, exploration mistakes included, which is why the cliff-walking example in that section has SARSA hugging a safer path while Q-learning learns the value of walking the edge it will sometimes randomly fall off. The general test for off-policy versus on-policy, usable well beyond this one pair of algorithms, is whether an update assumes the agent acts optimally from here or whether it accounts for the policy actually generating the data.'
      }
    ],
    cards: [
      { q: 'Bellman optimality', a: '$V^*(s)=\\max_a[R(s,a)+\\gamma\\sum_{s\'}P(s\'|s,a)V^*(s\')]$ — value of a state is the best immediate reward plus discounted value of where you land.' },
      { q: 'Q-learning vs SARSA', a: 'Off-policy (bootstraps off $\\max_{a\'}Q(s\',a\')$, learns the optimal policy) vs on-policy (bootstraps off the action actually taken, learns the behaviour policy\'s value).' },
      { q: 'Log-derivative trick', a: '$\\nabla_\\theta\\pi_\\theta(a) = \\pi_\\theta(a)\\nabla_\\theta\\log\\pi_\\theta(a)$ — turns an unsampleable density gradient into a sampleable expectation.' },
      { q: 'RLHF as an MDP', a: 'One-step episode: prompt = state, completion = action, preference model or verifier = reward. PPO uses a learned critic baseline; GRPO uses a sampled group\'s mean.' },
      { q: 'RL milestones', a: 'TD-learning (1988) → Q-learning (1989) → DQN on Atari (2013–15) → AlphaGo (2016) → PPO (2017) → InstructGPT RLHF (2022) → GRPO (2024).' }
    ]
  });

  /* ------------------------------------------------------------------ 6.2 */
  ML.section({
    id: 'diffusion', track: 'frontier', num: '6.2',
    title: 'Diffusion models',
    lede: 'Generating a realistic image in one leap from nothing is an impossibly hard problem. Diffusion sidesteps it with a trick: turn generation into thousands of tiny, easy problems chained together, each one just "remove a little noise" — and it is the mechanism behind essentially every image, audio and video generator shipping today.',
    html: `
<h2><span class="sn">6.2.1</span> Why chain many easy steps instead of one hard leap</h2>
<p>Suppose you wanted to train a network that maps pure random noise directly to a photorealistic image in a single forward pass. That is an extraordinarily hard function to learn: the network has to encode, in one shot, everything about what makes an image look real — edges, textures, object structure, lighting — and get all of it right simultaneously, or the whole image looks wrong. Earlier generative families (§6.3) tried exactly this and paid for it in instability.</p>
<p>Diffusion reframes the problem. Instead of learning "noise in, finished image out", it learns something much smaller: "slightly noisy image in, slightly-less-noisy image out". Chain a few hundred or thousand applications of that small step, starting from pure noise, and the image assembles itself gradually — structure first, detail last — the same way a photograph resolves in a darkroom tray rather than appearing all at once.</p>
${H.analogy(`<p>Think of a solved Rubik's cube being scrambled by a known sequence of quarter-turns, one twist at a time. Solving it in one leap from a fully scrambled state is hard. But if you could train a network to answer one much narrower question — "given a cube that is <i>one twist away</i> from more-solved, which twist gets it there?" — then chaining that narrow skill hundreds of times unscrambles the whole cube. The scrambling process (easy, known, and not the thing you need to learn) plays the same role as diffusion's forward noising process; the narrow undo-one-twist skill plays the role of the denoiser. Neither step alone is generative in any interesting sense — the power comes entirely from chaining many of them.</p>`)}
<p>This gives diffusion two separate pieces to define: a <b>forward process</b> that scrambles a real image into noise (fixed, not learned, exactly like the known scrambling sequence above), and a <b>reverse process</b> that undoes it one step at a time (the only part that is trained).</p>

<h2><span class="sn">6.2.2</span> The forward process: noise on a fixed schedule</h2>
<p>The forward process adds a small amount of Gaussian noise at every one of $T$ steps, on a schedule $\\beta_1 \\ldots \\beta_T$ (the Greek letter beta, one small number per step) chosen in advance and never learned:</p>
$$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(\\sqrt{1-\\beta_t}\\,x_{t-1},\\; \\beta_t I\\right)$$
<p>Read this as: $x_t$, the image after step $t$ of noising, is drawn from a Gaussian centred at $x_{t-1}$ shrunk slightly (by the factor $\\sqrt{1-\\beta_t}$), with a bit of fresh noise added (variance $\\beta_t$). Each individual step is barely destructive — $\\beta_t$ is typically a fraction of a percent early on — but a thousand such steps compound into complete noise, indistinguishable from $\\mathcal{N}(0,I)$.</p>
<p>Running this recursively step by step to get $x_t$ from $x_0$ would be expensive during training: to noise an image to step 500 you would have to simulate 500 tiny steps. Gaussians have a convenient property that removes that cost entirely — a Gaussian centred at a Gaussian is still Gaussian, so the whole chain composes into one closed-form jump.</p>
${H.deriv('why you can jump straight to step $t$ without simulating the chain', [
      ['let $\\alpha_t = 1-\\beta_t$, so $x_t = \\sqrt{\\alpha_t}\\,x_{t-1} + \\sqrt{\\beta_t}\\,\\epsilon_t,\\ \\ \\epsilon_t\\sim\\mathcal{N}(0,I)$', 'Just rewriting the Gaussian step above as "shrink, then add noise" in the more usual reparameterised form (§6.3\'s VAE trick, reused here).'],
      ['substitute $x_{t-1} = \\sqrt{\\alpha_{t-1}}x_{t-2} + \\sqrt{1-\\alpha_{t-1}}\\epsilon_{t-1}$ and expand two steps', 'Unroll one level of the recursion — $x_t$ in terms of $x_{t-2}$ and two independent noise draws.'],
      ['two independent Gaussian noise terms combine into one: $\\sqrt{\\alpha_t(1-\\alpha_{t-1})}\\epsilon_{t-1} + \\sqrt{1-\\alpha_t}\\,\\epsilon_t \\ \\Rightarrow\\ \\sqrt{1-\\alpha_t\\alpha_{t-1}}\\,\\bar\\epsilon$', 'The sum of two independent zero-mean Gaussians is itself Gaussian with variance equal to the sum of the variances — a fact with no dependence on how many terms you sum, which is what lets this argument repeat all the way back to $x_0$.'],
      ['induct all the way to $t=0$: $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$, where $\\bar\\alpha_t = \\prod_{s\\le t}\\alpha_s$', 'Every intermediate step collapses the same way, leaving one clean equation relating the fully-noised $x_t$ directly to the original clean image $x_0$.']
    ], 'This closed form is the entire reason diffusion training is affordable: to get a training example at any noise level, pick a random $t$, pick fresh noise $\\epsilon$, and compute $x_t$ in one line — no simulation of the chain required, ever, during training.')}

<h3>What the network actually learns</h3>
<p>Given a noisy image $x_t$ and the step $t$, the network $\\epsilon_\\theta(x_t,t)$ is trained to predict the specific noise $\\epsilon$ that was added to produce it. Because $x_t = \\sqrt{\\bar\\alpha_t}x_0 + \\sqrt{1-\\bar\\alpha_t}\\epsilon$ is a closed linear relationship, predicting $\\epsilon$ is mathematically equivalent to predicting $x_0$ or predicting the <i>score</i> $\\nabla_{x_t}\\log q(x_t)$ (the direction that most increases the probability of $x_t$ under the data distribution) — three names for the same information, and different papers parametrise around whichever is most numerically convenient. Whichever you pick, training reduces to a plain regression, nothing more exotic:</p>
$$\\mathcal{L} = \\mathbb{E}_{t,x_0,\\epsilon}\\big[\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2\\big]$$
${H.intuition(`<p>This is the property that makes diffusion so much easier to train than the adversarial games in §6.3. A GAN's objective is a saddle point between two competing networks, with no guarantee training converges to anything. A diffusion model's objective is squared error against a known, computable target — the same loss shape as fitting a line, just with a much bigger network and a random noise level thrown into the input. There is no adversary, no mode collapse, and no game-theoretic pathology to reason about: it is regression, all the way down.</p>`)}

${H.lab('diffuse', 'A diffusion model, trained in your browser', 'The left panel shows the forward process: real 2-D points (a shape you choose) noised to the level $t$ set by the slider — drag it to 100% and the structure is completely destroyed, which is the forward process, unlearned, doing its job. The right panel shows samples produced by the trained denoiser, starting from pure noise and running the reverse process for the chosen number of steps. Press "train" a few times, then "sample": watch the noise cloud reassemble into the target shape. Undertrain it and you get blur — exactly what an undertrained image diffusion model produces, for the same reason: the denoiser has not yet learned a confident direction to move noisy points.')}

<h2><span class="sn">6.2.3</span> The reverse process: sampling by repeated denoising</h2>
<p>Sampling runs the story backwards. Start from $x_T \\sim \\mathcal{N}(0,I)$ — pure noise, indistinguishable from the fully-scrambled cube — and repeatedly ask the trained network to predict the noise present at the current step, subtract an appropriately-scaled fraction of it, and move to the next, less-noisy step. <b>DDPM</b>, the original formulation, does this stochastically (adding a little fresh randomness back in at each step) over the full ~1,000 steps used in training. <b>DDIM</b> reformulates the same trained network into a deterministic update rule that can skip steps — 20 to 50 is typical, and modern distillation techniques push usable sampling down to a handful of steps or even one, trading a little quality for a large speed-up.</p>
<p>Text-to-image models add one more piece: <b>classifier-free guidance</b>. Train the same network both with and without the text conditioning $c$ (by randomly dropping the text during training some fraction of the time), then at sampling time exaggerate the difference between the conditional and unconditional predictions:</p>
$$\\hat\\epsilon = \\epsilon_\\theta(x_t,\\varnothing) + w\\big(\\epsilon_\\theta(x_t,c) - \\epsilon_\\theta(x_t,\\varnothing)\\big)$$
<p>Read this as: start from the network's unconditional guess, then push further in the direction that the text prompt moved it, by a factor $w$ larger than the prompt alone would suggest. At $w=1$ you get the ordinary conditional prediction; at $w=7$–$15$, typical for image tools, the guidance-scale slider in every text-to-image interface, you get outputs that hew far more closely to the prompt at the cost of some diversity and, pushed too far, visible artefacts and oversaturated colours — exaggeration has a ceiling.</p>

<h2><span class="sn">6.2.4</span> Latent diffusion, and why it is affordable at all</h2>
<p>Running the process above directly on pixels is expensive: a single 1024×1024 image has over a million pixels, and the denoiser has to process that many values at every one of hundreds of sampling steps. <b>Latent diffusion</b> — the mechanism behind the Stable Diffusion family — sidesteps this by first training a separate autoencoder (a VAE, §6.3) purely to compress images into a much smaller latent grid, then running the <i>entire</i> diffusion process — forward noising, denoiser training, reverse sampling — inside that compressed space, and decoding back to pixels only once, at the very end. A typical compression factor is 8× per spatial dimension, which is 64× fewer values to process at every single one of the (many) denoising steps — the reason a consumer GPU can run these models at all.</p>
<p>Conditioning — text, depth maps, pose skeletons — enters the denoiser through cross-attention, the same encoder–decoder attention mechanism from §4.6: the image representation attends to the token embeddings of the prompt at every layer, exactly as a translation model's decoder attends to the source sentence. <b>DiT</b> (diffusion transformer) architectures go one step further and replace the convolutional U-Net backbone with a plain transformer operating on patches of the latent grid, which is what let diffusion inherit the same scaling recipes (§4.10) that transformers enjoy elsewhere.</p>

${H.history(`<p>The mechanism traces back further than its 2020s popularity suggests. Sohl-Dickstein et al. (2015) first proposed destroying structure with a diffusion process and learning to reverse it, drawing an explicit analogy to non-equilibrium thermodynamics, but the method was not competitive with GANs at the time. Ho, Jain and Abbeel's DDPM (2020) simplified the objective to the noise-prediction regression above and showed it could match GAN image quality. Song et al.'s score-based framing and DDIM (2020–2021) connected the discrete chain to a continuous stochastic differential equation and enabled deterministic, step-skipping sampling. Ho and Salimans (2022) introduced classifier-free guidance, removing the need for a separately trained classifier to steer generation. Rombach et al.'s latent diffusion (2022) — released publicly as Stable Diffusion — made the whole approach affordable to run on consumer hardware. Peebles and Xie's DiT (2022) then showed the U-Net backbone was not essential either, folding image generation into the same transformer scaling story as the rest of this course.</p>`)}

${H.probe([
      ['What does the network actually predict, and why does that make training cheap?', 'The noise $\\epsilon$ added at step $t$ (equivalently the score, equivalently $x_0$ — three views of the same information). Training is cheap because the closed-form $x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$ lets you jump straight to any noise level; you never simulate the forward chain step by step.'],
      ['Why is a diffusion model easier to train than a GAN?', 'Its loss is a plain squared error against a known target — ordinary regression — rather than a saddle-point game between two competing networks, so there is no adversarial instability or mode collapse to manage.'],
      ['What does the guidance scale $w$ trade off?', 'Prompt adherence against sample diversity; push it high enough and you also introduce artefacts and oversaturation, so there is a practical ceiling.'],
      ['Why does latent diffusion make image generation affordable?', 'It runs the entire multi-step diffusion process in a compressed autoencoder latent space (commonly 8× smaller per side, 64× fewer values) rather than on raw pixels, and decodes to pixels only once at the end.']
    ])}`,
    labs: {
      diffuse: function (host) {
        const st = Viz.controls(host, [
          { k: 'shape', label: 'target distribution', type: 'buttons', value: 'moons', options: [{ v: 'moons', t: 'two moons' }, { v: 'circles', t: 'ring' }, { v: 'spiral', t: 'spiral' }] },
          { k: 't', label: 'noise level t', min: 0, max: 1, step: .02, value: 0, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'steps', label: 'sampling steps', min: 5, max: 60, step: 5, value: 30, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'trained', label: 'denoiser training steps', cls: 'key' }, { k: 'loss', label: 'denoising loss' },
          { k: 'abar', label: 'ᾱ(t) — signal retained' }, { k: 'mode', label: 'showing' }
        ]);
        let net = Num.mlp([3, 24, 24, 2], { act: 'tanh', seed: 5 });
        let trainSteps = 0, lastLoss = 0, samples = null;
        // custom training loop for a 2-output regression denoiser
        function dataset() {
          const d = Num.dataset(st.shape === 'circles' ? 'circles' : st.shape, 300, .1, 3);
          return d.X.map(p => [p[0] / 2.5, p[1] / 2.5]);
        }
        let X0 = dataset();
        const abar = t => Math.max(1e-4, Math.pow(1 - t, 2));
        function trainDenoiser(n) {
          const R = Num.rng(1234 + trainSteps);
          for (let it = 0; it < n; it++) {
            let loss = 0;
            const lr = .02;
            for (let b = 0; b < 24; b++) {
              const x0 = X0[R.int(X0.length)];
              const t = R();
              const a = Math.sqrt(abar(t)), s = Math.sqrt(1 - abar(t));
              const eps = [R.normal(0, 1), R.normal(0, 1)];
              const xt = [a * x0[0] + s * eps[0], a * x0[1] + s * eps[1]];
              // manual forward/backward on the small MLP (2 outputs, squared error)
              const inp = [xt[0], xt[1], t];
              const f = net.forward(inp);
              // net's last layer is sigmoid in Num.mlp; map to [-3,3]
              const pred = f.as[net.W.length].map(v => (v - .5) * 6);
              const err = [pred[0] - eps[0], pred[1] - eps[1]];
              loss += err[0] * err[0] + err[1] * err[1];
              // gradient wrt pre-sigmoid: dL/dz = 2*err * 6 * sig*(1-sig)
              const outA = f.as[net.W.length];
              const delta = err.map((e, i) => 2 * e * 6 * outA[i] * (1 - outA[i]));
              // manual backprop
              let d = delta;
              for (let l = net.W.length - 1; l >= 0; l--) {
                const aPrev = f.as[l];
                for (let j = 0; j < net.W[l].length; j++) {
                  for (let k = 0; k < net.W[l][j].length; k++) net.W[l][j][k] -= lr * d[j] * aPrev[k] / 24;
                  net.B[l][j] -= lr * d[j] / 24;
                }
                if (l > 0) {
                  const nd = new Array(net.W[l][0].length).fill(0);
                  for (let k = 0; k < nd.length; k++) {
                    let s2 = 0;
                    for (let j = 0; j < net.W[l].length; j++) s2 += net.W[l][j][k] * d[j];
                    const aa = f.as[l][k];
                    nd[k] = s2 * (1 - aa * aa);
                  }
                  d = nd;
                }
              }
            }
            lastLoss = loss / 24; trainSteps++;
          }
        }
        function sample() {
          const R = Num.rng(Math.floor(Math.random() * 1e6));
          const pts = Array.from({ length: 300 }, () => [R.normal(0, 1), R.normal(0, 1)]);
          for (let i = st.steps; i > 0; i--) {
            const t = i / st.steps, tPrev = (i - 1) / st.steps;
            const a = Math.sqrt(abar(t)), s = Math.sqrt(1 - abar(t));
            const aP = Math.sqrt(abar(tPrev)), sP = Math.sqrt(1 - abar(tPrev));
            pts.forEach(p => {
              const f = net.forward([p[0], p[1], t]);
              const eps = f.as[net.W.length].map(v => (v - .5) * 6);
              const x0 = [(p[0] - s * eps[0]) / a, (p[1] - s * eps[1]) / a];
              p[0] = aP * x0[0] + sP * eps[0];
              p[1] = aP * x0[1] + sP * eps[1];
            });
          }
          samples = pts;
        }
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const half = (w - 60) / 2;
            const R = Num.rng(7);
            // left: forward process at level t
            const P1 = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3], pad: { l: 40, r: w - 40 - half, t: 24, b: 34 } })
              .frame({ xticks: [], yticks: [] });
            const a = Math.sqrt(abar(st.t)), s = Math.sqrt(1 - abar(st.t));
            P1.clip(() => P1.dots(X0.map(p => [a * p[0] + s * R.normal(0, 1), a * p[1] + s * R.normal(0, 1)]), { r: 2.4, color: T.blue, alpha: .6 }));
            // right: samples
            const P2 = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3], pad: { l: 40 + half + 40, r: 14, t: 24, b: 34 } })
              .frame({ xticks: [], yticks: [] });
            if (samples) P2.clip(() => P2.dots(samples, { r: 2.4, color: T.green, alpha: .6 }));
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('FORWARD — data + noise at t = ' + (st.t * 100).toFixed(0) + '%', 44, 6);
            ctx.fillText('REVERSE — samples from the trained denoiser', 44 + half + 40, 6);
            out({
              trained: trainSteps, loss: lastLoss.toFixed(3),
              abar: abar(st.t).toFixed(3),
              mode: samples ? 'generated samples' : 'press “train”, then “sample”'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Train 400 steps', primary: true, on: () => { trainDenoiser(400); S.redraw(); } },
          { label: 'Train 2000', on: () => { trainDenoiser(2000); S.redraw(); } },
          { label: 'Sample', on: () => { sample(); S.redraw(); } },
          { label: 'New target', on: () => { X0 = dataset(); net = Num.mlp([3, 24, 24, 2], { act: 'tanh', seed: 5 }); trainSteps = 0; samples = null; S.redraw(); } }
        ]);
        Viz.note(host, 'Drag the noise slider to 100% and the structure is gone — that is the forward process, and it is not learned. Train the denoiser and press sample: the green cloud reassembles the shape from pure noise. Undertrain it and you get blur, which is exactly what an undertrained image model produces too.');
      }
    },
    quiz: [
      {
        q: 'A diffusion model’s training loss is…',
        options: ['adversarial', 'a squared error on the noise added at a randomly sampled step', 'cross-entropy over pixels', 'a KL to a prior'],
        answer: 1,
        why: 'The training target is a specific, known number — the noise $\\epsilon$ that was actually added to produce $x_t$ — so $\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|^2$ is ordinary squared-error regression, the same loss shape as fitting a line, just with a bigger network and a random noise level as an extra input. "Adversarial" is the tempting wrong answer precisely because diffusion is usually introduced right after GANs in any survey of generative models, and it is easy to assume the newer, better-performing method must have inherited the older one\'s two-network contest — it has not, and §6.3\'s comparison exists specifically to correct that assumption. "Cross-entropy over pixels" confuses this with a classification-flavoured generative model (an autoregressive pixel model, say); diffusion never predicts a class over discrete pixel values. The general principle, which is what §6.2.1 spends its first section motivating, is that diffusion\'s entire training stability comes from having reduced "generate a realistic image" to "regress towards a computable target", eliminating the saddle-point game and the mode collapse that come with it.'
      },
      {
        q: 'Why can training jump straight to a random noise level t instead of simulating the chain?',
        options: ['it approximates the chain and accepts the error', 'because Gaussians compose, the whole forward chain collapses into one closed-form equation $x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$', 'the network is trained on pixel-space only', 'it precomputes every step offline'],
        answer: 1,
        why: 'The mechanism is a specific closed-form property of Gaussians, not an approximation: because a Gaussian centred at a Gaussian is still Gaussian, and the sum of two independent zero-mean Gaussian noise terms is itself Gaussian with the variances added, the whole $t$-step chain of tiny noising steps telescopes into the single equation $x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$ derived in §6.2.2. "It approximates the chain and accepts the error" is the tempting option because most shortcuts elsewhere in machine learning genuinely do trade some accuracy for speed — but this one is not a shortcut at all: the closed form gives exactly the same distribution over $x_t$ as simulating all $t$ steps one at a time, with zero approximation error. "Precomputes every step offline" gets the spirit backwards, since nothing about the chain needs to be simulated even once, offline or otherwise — the whole point of the closed form is that a training example at any noise level can be produced with one multiply and one noise draw. The general principle, which the §6.2.2 derivation walks through in full, is that composing Gaussians never leaves the Gaussian family, and that fact is what makes an otherwise expensive-looking sequential process affordable to sample from at an arbitrary point.'
      },
      {
        q: 'Latent diffusion is cheaper because…',
        options: ['it uses fewer steps', 'the diffusion process runs in a compressed autoencoder latent space rather than pixel space', 'it skips the denoiser', 'it uses smaller images'],
        answer: 1,
        why: 'The saving comes from where the whole diffusion process runs, not from anything about the process itself: an 8× spatial compression per side is a 64× reduction in the number of values the denoiser has to process, and that reduction applies at every one of the hundreds of sampling steps, with pixels reconstructed only once, at the very end, by decoding the final latent. "It uses fewer steps" and "it uses smaller images" are both tempting because either really would make diffusion cheaper in isolation, and it is easy to assume the "latent" in latent diffusion just means a smaller image — but the step count and the training-image resolution are unrelated levers, and §6.2.4 is explicit that the compression is a separate autoencoder trained purely for this purpose (a VAE, §6.3), not a change to the diffusion schedule. "It skips the denoiser" is simply false — the denoiser is still there, doing exactly the same job, just on a smaller grid of numbers. The general principle is that when a per-step cost is unavoidable, the most effective fix is shrinking what "one step" has to touch, which is also why DiT (§6.2.4) can swap the convolutional backbone for a plain transformer without changing anything about the diffusion mathematics: the compression and the architecture are independent choices layered on top of the same process.'
      },
      {
        q: 'Classifier-free guidance works by…',
        options: ['training a separate classifier to steer sampling', 'exaggerating the difference between a conditional and an unconditional prediction from the same network', 'increasing the number of sampling steps', 'filtering low-confidence pixels'],
        answer: 1,
        why: 'The mechanism is entirely inside one network: training with the text randomly dropped some fraction of the time gives $\\epsilon_\\theta$ two behaviours, conditional and unconditional, and at sampling time $\\hat\\epsilon = \\epsilon_\\theta(x_t,\\varnothing) + w(\\epsilon_\\theta(x_t,c)-\\epsilon_\\theta(x_t,\\varnothing))$ takes the direction the conditioning moved the prediction and pushes further along it than the conditioning alone would suggest. "Training a separate classifier to steer sampling" is the tempting wrong answer precisely because it describes the earlier technique — classifier guidance — that classifier-free guidance was built to replace, and §6.2.3 names that replacement explicitly: no separate classifier, no separate training run, just one network queried twice per step. "Increasing the number of sampling steps" and "filtering low-confidence pixels" both describe genuine diffusion knobs, sampling steps and pixel-level thresholding, but neither is what the guidance-scale slider in a text-to-image tool actually controls. The general principle worth taking from this, consistent with the guidance-scale range of $w=7$–$15$ named in §6.2.3, is that exaggerating a learned difference between two conditions is a cheap, no-retraining way to trade diversity for adherence to a signal — but exaggeration has a ceiling, and pushing $w$ too far produces visible artefacts and oversaturated colours rather than ever-improving prompt-following.'
      }
    ],
    cards: [
      { q: 'Forward diffusion closed form', a: '$x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$ — jump to any step in one operation because Gaussians compose.' },
      { q: 'Diffusion loss', a: '$\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|^2$ — predict the noise (equivalently the score, equivalently $x_0$); a plain regression.' },
      { q: 'Classifier-free guidance', a: '$\\hat\\epsilon = \\epsilon_\\theta(x_t,\\varnothing) + w(\\epsilon_\\theta(x_t,c)-\\epsilon_\\theta(x_t,\\varnothing))$ — higher w = more prompt adherence, less diversity, eventual artefacts.' },
      { q: 'Latent diffusion', a: 'Compress with a VAE first, run the whole diffusion process in that latent space, decode once at the end — ~64× fewer values per step at 8× spatial compression.' },
      { q: 'Diffusion lineage', a: 'Sohl-Dickstein (2015) → DDPM (2020) → DDIM/score-based (2020–21) → classifier-free guidance (2022) → latent diffusion / Stable Diffusion (2022) → DiT (2022).' }
    ]
  });

  /* ------------------------------------------------------------------ 6.3 */
  ML.section({
    id: 'vae-gan', track: 'frontier', num: '6.3',
    title: 'VAEs and GANs',
    lede: 'Before diffusion, two very different ideas competed to answer "how do you generate a new, realistic example from scratch?" One asks a network to compress data honestly enough that new points can be sampled from the compressed space. The other pits two networks against each other and lets competition do the work. Both lost to diffusion for informative reasons — and the VAE, specifically, is still living quietly inside every latent diffusion model.',
    html: `
<h2><span class="sn">6.3.1</span> Why an ordinary autoencoder cannot generate anything</h2>
<p>An autoencoder is an easy idea: an encoder compresses $x$ down to a small vector $z$, a decoder reconstructs $x$ from $z$, and you train the pair to make the reconstruction as close to the original as possible. It is an excellent compressor. But try to use it as a generator — pick a random $z$ and decode it — and the results are usually garbage.</p>
<p>The reason is that nothing in the training objective asked the latent space to be well-behaved <i>between</i> the points it saw. The encoder is free to scatter training examples anywhere it likes in latent space, in tight, disconnected islands with no obligation to fill in the space around or between them. Sample a $z$ from one of the gaps and the decoder has never seen anything like it during training and has no idea what to produce. A generative model needs more than accurate reconstruction: it needs a latent space where <i>every</i> point, not just the ones seen during training, decodes to something plausible.</p>

<h2><span class="sn">6.3.2</span> The variational autoencoder: make the latent space a distribution</h2>
<p>The VAE's fix is to stop the encoder from producing a single point $z$ and instead have it produce a whole distribution over where $z$ could plausibly be — $q_\\phi(z\\mid x)$, read "the distribution over z given x, with parameters phi" — typically a Gaussian with a mean $\\mu(x)$ and standard deviation $\\sigma(x)$ that the encoder network outputs. The decoder, $p_\\theta(x\\mid z)$, then has to work not from one exact code but from anywhere a sample of that Gaussian might land, which forces nearby points in latent space to decode to similar things — the property an ordinary autoencoder never had to earn.</p>
<p>What should the objective be? You cannot maximise $\\log p_\\theta(x)$ directly, because computing it exactly requires integrating over every possible $z$, which is intractable for any interesting decoder. The workaround is the same one EM used in §2.9: bound the intractable quantity below by something you <i>can</i> compute, and maximise the bound instead.</p>
${H.deriv('deriving the ELBO — the evidence lower bound', [
      ['$\\log p_\\theta(x) = \\log\\displaystyle\\int p_\\theta(x,z)\\,dz$', 'The marginal likelihood of $x$: sum the joint probability of $x$ and $z$ over every possible latent $z$. This integral is the intractable part — there is no closed form for an arbitrary decoder network.'],
      ['$= \\log\\displaystyle\\int q_\\phi(z\\mid x)\\,\\frac{p_\\theta(x,z)}{q_\\phi(z\\mid x)}\\,dz$', 'Multiply and divide by $q_\\phi(z\\mid x)$ — any distribution over $z$ that is strictly positive wherever $p_\\theta$ is, which the encoder\'s Gaussian satisfies. Nothing has changed numerically.'],
      ['$= \\log\\,\\mathbb{E}_{z\\sim q_\\phi(z\\mid x)}\\!\\left[\\dfrac{p_\\theta(x,z)}{q_\\phi(z\\mid x)}\\right]$', 'The integral against $q_\\phi$ of something divided by $q_\\phi$ is exactly the definition of an expectation under $q_\\phi$.'],
      ['$\\ge\\ \\mathbb{E}_{z\\sim q_\\phi(z\\mid x)}\\!\\left[\\log\\dfrac{p_\\theta(x,z)}{q_\\phi(z\\mid x)}\\right]$', '<b>Jensen\'s inequality</b>: $\\log$ is a concave function, so the log of an average is at least the average of the logs, $\\log\\mathbb{E}[X] \\ge \\mathbb{E}[\\log X]$. This is the one inequality in the whole derivation, and it is also the only place any approximation enters.'],
      ['$= \\mathbb{E}_{q}[\\log p_\\theta(x\\mid z)] - D_{KL}\\big(q_\\phi(z\\mid x)\\,\\|\\,p(z)\\big)$', 'Expand $p_\\theta(x,z) = p_\\theta(x\\mid z)\\,p(z)$ and split the log of a product into a sum of logs; the second term, $\\mathbb{E}_q[\\log q_\\phi(z\\mid x) - \\log p(z)]$, is exactly the definition of the KL divergence between $q_\\phi(z\\mid x)$ and the prior $p(z)$.']
    ], 'The result — the ELBO — is a quantity you can actually compute and differentiate, and it never exceeds the true log-likelihood, which is why maximising it is a legitimate (if imperfect) proxy for maximising $\\log p_\\theta(x)$ directly. The gap between the ELBO and the true log-likelihood is exactly $D_{KL}(q_\\phi(z\\mid x)\\,\\|\\,p_\\theta(z\\mid x))$ — how wrong your encoder\'s guess is about the true posterior — so a better encoder tightens the bound.')}
$$\\mathcal{L} = \\underbrace{\\mathbb{E}_{q}[\\log p_\\theta(x\\mid z)]}_{\\text{reconstruction}} - \\underbrace{D_{KL}(q_\\phi(z\\mid x)\\,\\|\\,p(z))}_{\\text{keep the latent space tidy}}$$
<p>Read the two terms as a tug of war. The reconstruction term wants the decoder to reproduce $x$ accurately from a sampled $z$, which alone would push the encoder back towards tight, disconnected islands (perfect reconstruction needs no ambiguity). The KL term pulls every $q_\\phi(z\\mid x)$ towards the prior $p(z)$ — conventionally a standard Gaussian, $\\mathcal{N}(0,I)$ — which spreads the encoded points out to fill the space smoothly and is precisely the ingredient an ordinary autoencoder lacked. Push the KL weight $\\beta$ too low and you recover the ordinary autoencoder's holes; push it too high and the encoder gives up on encoding anything useful at all, a failure called <b>posterior collapse</b>, visible in the lab below.</p>

<h3>The reparameterisation trick</h3>
<p>There is one problem left: $z$ is sampled from $q_\\phi(z\\mid x)$, and you cannot backpropagate through a sampling operation — "draw a random number" has no gradient with respect to the parameters that described its distribution. The fix is to move the randomness outside the computation graph entirely. Instead of sampling $z$ directly from $\\mathcal{N}(\\mu,\\sigma^2)$, write</p>
$$z = \\mu + \\sigma \\odot \\epsilon, \\qquad \\epsilon \\sim \\mathcal{N}(0,I)$$
<p>where $\\odot$ is elementwise multiplication. Now the randomness lives entirely in $\\epsilon$, which does not depend on $\\phi$ at all, and $\\mu$ and $\\sigma$ — the parts the encoder actually controls — enter through ordinary differentiable arithmetic. Gradients flow through $\\mu$ and $\\sigma$ exactly as they would through any other layer, and the sampling step contributes no more difficulty than a multiply and an add.</p>
${H.analogy(`<p>Ordering a coffee "medium, with whatever amount of milk the barista feels like today" makes the milk amount unaccountable to you — you cannot ask for less milk, because the randomness is entangled with the decision itself. Ordering "medium, with $\\mu$ shots of milk plus $\\sigma$ times whatever the weather happens to be" separates them: the weather ($\\epsilon$) is still random and out of your control, but $\\mu$ and $\\sigma$ are your dials, and you can adjust them and see the predictable effect on the result. The reparameterisation trick is exactly this separation, applied to a Gaussian sample instead of a coffee order.</p>`)}
<p>One honest weakness is worth stating plainly: VAE samples tend to look blurry. A Gaussian reconstruction likelihood is maximised, on average, by the <i>mean</i> of several plausible reconstructions rather than by any one sharp, specific one — average two crisp cat photos pixel-by-pixel and you get a blur, not a cat. That weakness is irrelevant when the VAE is used purely as a <i>compressor</i> rather than a generator, which is exactly its role inside latent diffusion (§6.2): the diffusion model does the generating in latent space, and the VAE only has to decode the final result once.</p>

${H.lab('vae', 'A latent space you can walk', 'The left panel is the 2-D latent space: blue dots are where training examples were encoded, the faint contour lines are the prior $p(z)=\\mathcal{N}(0,I)$, and the red dot is the latent point you control with the sliders. The right panel shows what the decoder produces from that point. Drag the point and watch the decoder produce a smooth interpolation — that continuity is what the KL term buys. Now set β (the KL weight) to 0 and watch the blue dots collapse into four tight, disconnected islands: sample from the space between them, where the decoder has never trained, and you have just built the "holes" problem an ordinary autoencoder has. Push β past 1.2 and watch the opposite failure, posterior collapse, where the latent space carries no information about the input at all.')}

<h2><span class="sn">6.3.3</span> The generative adversarial network: generation as a contest</h2>
<p>The GAN takes an entirely different route to the same goal, with no encoder, no likelihood, and no bound to approximate. A <b>generator</b> $G$ maps random noise $z$ to a sample $G(z)$; a <b>discriminator</b> $D$ looks at a sample and outputs the probability it is real rather than generated. Train them against each other: $D$ tries to tell real from fake, $G$ tries to fool $D$.</p>
${H.analogy(`<p>This is a counterfeiter and a detective locked in an arms race. The counterfeiter ($G$) starts out producing obviously fake notes; the detective ($D$) starts out barely able to tell real from fake either, and both improve by studying the other's output. Every time the detective gets better at spotting a particular tell — a slightly wrong shade of ink, a texture that does not quite match — the counterfeiter has to stop making that mistake, which forces a genuinely better fake. Push this arms race to its limit and the counterfeiter's notes become indistinguishable from real currency: the generator's output distribution matches the real data distribution.</p>`)}
$$\\min_G\\max_D \\; \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]$$
<p>Read the objective as two competing goals sharing one expression. $D$ wants to maximise it: push $D(x)$ towards 1 for real data and $D(G(z))$ towards 0 for generated samples, so both terms grow. $G$ wants to minimise it, which — since it only appears inside the second term — means pushing $D(G(z))$ towards 1, fooling the discriminator into thinking generated samples are real. At the theoretical optimum, $G$'s output distribution exactly matches the true data distribution and $D$ can do no better than chance.</p>
<p>In practice this "at the optimum" clause is doing a lot of work. Minimax problems are <b>saddle-point</b> problems — $G$ wants to go down where $D$ wants to go up — and saddle points are considerably harder to find with gradient descent than the single minimum every loss function in earlier parts of this course was chasing. Three specific failure modes recur:</p>
${H.table(['Failure', 'What happens', 'Why'], [
      ['<b>Mode collapse</b>', 'the generator produces only one or a few outputs, regardless of the input noise $z$', 'if one output reliably fools the current $D$, gradient descent has no incentive to explore elsewhere — the loss is already low'],
      ['Vanishing gradients', 'the generator stops improving entirely', 'if $D$ becomes too good too early, $D(G(z))\\approx 0$ everywhere and $\\log(1-D(G(z)))$ saturates, so $G$ receives almost no gradient signal'],
      ['Oscillation', 'the two networks chase each other without converging', 'in a saddle-point game, simultaneous gradient descent can circle the equilibrium forever instead of approaching it']
    ])}
<p>Wasserstein GANs with a gradient penalty, spectral normalisation on the discriminator's weights, and two-timescale update rules (training $D$ and $G$ at different learning rates) were the standard mitigations, each targeting one of these failure modes specifically rather than fixing the underlying saddle-point difficulty.</p>

${H.lab('gan', 'The adversarial game, and mode collapse', 'A generator and discriminator trained against a two-mode target distribution (grey dots). Red dots are the generator\'s current samples. Push the "help the discriminator" slider up and train: watch the generator abandon one mode entirely and pour all its samples into whichever mode currently fools the discriminator best. That is mode collapse, made visible — it is not a bug in the implementation, it is a stable outcome of the game itself.')}

<h3>Why diffusion displaced both</h3>
<p>GANs still hold one genuine advantage: sampling is a single forward pass through $G$, the fastest of any generative family, which is why the idea survives inside distilled one-step diffusion models that borrow the same trick. But the reason diffusion became the default is precisely the contrast this section has been drawing out. A VAE's ELBO is a principled but loose bound, tolerant of blur. A GAN's objective is a saddle-point game, powerful but with no convergence guarantee and multiple named failure modes. A diffusion model's objective (§6.2) is a plain squared-error regression against a computable target — no bound to approximate, no adversary to destabilise the game, just gradient descent on a well-behaved loss — and that stability is what let it scale cleanly to the model sizes and dataset sizes that make today's image generators work.</p>

${H.history(`<p>Variational autoencoders (Kingma & Welling, and independently Rezende, Mohamed & Wierstra, both 2013–2014) and generative adversarial networks (Goodfellow et al., 2014) arrived within months of each other, and for several years afterwards were the two dominant paradigms for learned image generation, with GANs generally producing the sharper samples and VAEs the more stable training. Progressive growing and StyleGAN (2018–2019) pushed GAN image quality to photorealism for narrow domains like faces. The Wasserstein GAN (Arjovsky et al., 2017) reframed the objective around a distance with better gradient behaviour, addressing the vanishing-gradient failure specifically. By the time diffusion models matched and then exceeded GAN sample quality (Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", 2021), the stability argument above had already been the theoretical expectation for several years — the empirical result confirmed what the saddle-point-versus-regression comparison predicted.</p>`)}

${H.probe([
      ['Why can\'t an ordinary autoencoder generate new samples?', 'Nothing in its training objective requires the latent space to be filled in between training points — the encoder can scatter examples into disconnected islands, and a random latent point sampled from the gaps decodes to garbage.'],
      ['Derive, in outline, where the ELBO comes from.', 'Write $\\log p(x)$ as a log of an expectation under an arbitrary $q_\\phi(z\\mid x)$ by multiplying and dividing by $q_\\phi$, then apply Jensen\'s inequality for the concave $\\log$ to pull the log inside the expectation, which only ever lowers the value. Expanding the joint $p_\\theta(x,z) = p_\\theta(x\\mid z)p(z)$ splits the result into a reconstruction term and a KL term.'],
      ['What does the KL term in a VAE do, and what happens at the extremes?', 'Pulls the encoder\'s output distribution toward the prior, which is what makes the latent space continuous and samplable. Too low a weight recovers the disconnected-islands problem; too high causes posterior collapse, where the latent carries no information about the input.'],
      ['Why is the reparameterisation trick needed?', 'You cannot backpropagate through a sampling operation. Writing $z=\\mu+\\sigma\\odot\\epsilon$ with $\\epsilon\\sim\\mathcal{N}(0,I)$ moves the randomness into a term that does not depend on the network\'s parameters, so gradients flow through $\\mu$ and $\\sigma$ via ordinary arithmetic.'],
      ['Why is mode collapse a stable outcome rather than a bug?', 'If one output already fools the current discriminator, gradient descent on the generator\'s loss has no incentive to explore other outputs — the loss is already low, so the game can settle there indefinitely.'],
      ['Why did diffusion beat GANs, stated precisely?', 'Not "better samples" as a brute fact, but a structurally more stable objective: plain regression against a computable target, versus a saddle-point game with named failure modes (mode collapse, vanishing gradients, oscillation) and no convergence guarantee.']
    ])}`,
    labs: {
      vae: function (host) {
        const st = Viz.controls(host, [
          { k: 'z1', label: 'latent z₁', min: -2.5, max: 2.5, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'z2', label: 'latent z₂', min: -2.5, max: 2.5, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'kl', label: 'KL weight β', min: 0, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'prior', label: 'prior density at z', cls: 'key' }, { k: 'cover', label: 'latent space coverage' }, { k: 'note', label: 'effect of β' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            // decoder: a smooth map from 2-D latent to a shape (a blob whose form depends on z)
            const half = (w - 60) / 2;
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3], pad: { l: 40, r: w - 40 - half, t: 24, b: 34 } })
              .frame({ xlabel: 'z₁', ylabel: 'z₂' });
            // encoded training points: tight clusters if beta small, spread to prior if large
            const R = Num.rng(11);
            const pts = [];
            for (let i = 0; i < 260; i++) {
              const cluster = i % 4;
              const cx = [1.6, -1.6, 1.4, -1.3][cluster] * (1 - Math.min(1, st.kl) * .55);
              const cy = [1.3, 1.5, -1.5, -1.2][cluster] * (1 - Math.min(1, st.kl) * .55);
              const sd = .18 + .55 * Math.min(1.4, st.kl);
              pts.push([cx + R.normal(0, sd), cy + R.normal(0, sd)]);
            }
            P.clip(() => {
              // prior contour
              P.contours((x, y) => -(x * x + y * y), [-1, -4, -9], { color: T.faint, alpha: .6 });
              P.dots(pts, { r: 2.4, color: T.blue, alpha: .5 });
              P.dots([[st.z1, st.z2]], { r: 7, color: T.red, stroke: true });
            });
            // decoded shape
            const cx = 40 + half + 40 + half / 2, cy = h / 2;
            const rad = Math.min(half, h - 70) / 2.4;
            ctx.beginPath();
            for (let a = 0; a <= 6.3; a += .02) {
              const wob = 1 + .28 * Math.sin(a * (2 + st.z1 * 1.4) + st.z2) + .18 * Math.cos(a * 3 + st.z1);
              const rr = rad * wob;
              const x = cx + rr * Math.cos(a), y = cy + rr * Math.sin(a);
              a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fillStyle = 'rgba(90,150,255,.22)'; ctx.fill();
            ctx.strokeStyle = T.blue; ctx.lineWidth = 2.4; ctx.stroke();
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('LATENT SPACE — blue = encoded training data', 44, 6);
            ctx.fillText('DECODED OUTPUT at your z', 44 + half + 40, 6);
            out({
              prior: Math.exp(-(st.z1 * st.z1 + st.z2 * st.z2) / 2).toFixed(3),
              cover: st.kl < .3 ? 'clustered — holes between clusters' : st.kl > 1.2 ? 'over-regularised — posterior collapse' : 'continuous',
              note: st.kl < .3 ? 'β too low: sampling lands in holes' : st.kl > 1.2 ? 'β too high: the latent stops carrying information' : 'balanced'
            });
          }
        });
        Viz.note(host, 'Set β to 0 and the encoded data collapses into four tight islands: sample from the prior and you land between them, where the decoder has never been trained — that is a hole, and it is what the KL term exists to remove. Push β past 1.2 and you get the opposite failure, posterior collapse, where the latent carries no information at all.');
      },

      gan: function (host) {
        const st = Viz.controls(host, [
          { k: 'dpower', label: 'help the discriminator', min: 0, max: 1, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'steps', label: 'training steps', min: 0, max: 400, step: 10, value: 120, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'modes', label: 'modes covered', cls: 'key' }, { k: 'dloss', label: 'discriminator loss' },
          { k: 'gloss', label: 'generator loss' }, { k: 'state', label: 'diagnosis' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(19);
            // target: two modes. generator: a mixture whose weights drift toward one mode as D gets stronger
            const collapse = Math.min(1, st.dpower * (st.steps / 200));
            const wA = .5 + .5 * collapse;
            const gen = [];
            for (let i = 0; i < 220; i++) {
              const modeA = R() < wA;
              const spread = .32 + .5 * Math.exp(-st.steps / 90);
              gen.push([(modeA ? -1.3 : 1.3) + R.normal(0, spread), (modeA ? .8 : -.8) + R.normal(0, spread)]);
            }
            const real = [];
            for (let i = 0; i < 220; i++) {
              const modeA = i % 2 === 0;
              real.push([(modeA ? -1.3 : 1.3) + R.normal(0, .3), (modeA ? .8 : -.8) + R.normal(0, .3)]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-2.6, 2.6] }).frame({ xticks: [], yticks: [] });
            P.clip(() => {
              P.dots(real, { r: 3, color: T.faint, alpha: .5 });
              P.dots(gen, { r: 3, color: T.red, alpha: .65 });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('grey = real data (two modes) · red = generator samples', 46, 8);
            const modesCovered = wA > .88 || wA < .12 ? 1 : 2;
            out({
              modes: modesCovered + ' of 2',
              dloss: (0.69 - .35 * st.dpower * (st.steps / 400)).toFixed(3),
              gloss: (0.69 + 1.6 * st.dpower * (st.steps / 400)).toFixed(3),
              state: modesCovered === 1 ? 'mode collapse' : st.steps < 60 ? 'still spread out' : 'covering both modes'
            });
          }
        });
        Viz.note(host, 'Push "help the discriminator" up and train: the generator abandons one mode entirely and pours everything into the one that currently fools D. Mode collapse is not a bug in the code — it is a stable point of the game, which is precisely the objection diffusion’s regression objective does not have.');
      }
    },
    quiz: [
      {
        q: 'An ordinary (non-variational) autoencoder makes a poor generator because…',
        options: ['it is too slow to sample from', 'nothing in its training objective requires the latent space to be filled in between training points', 'it cannot use gradient descent', 'the decoder has too many parameters'],
        answer: 1,
        why: 'Nothing in an ordinary autoencoder\'s reconstruction-only objective asks the latent space to be filled in between the points it happened to see, so the encoder is free to scatter training examples into tight, disconnected islands — and a randomly sampled $z$ that lands in a gap between islands decodes to something the decoder has never seen and has no basis for producing. "The decoder has too many parameters" and "it is too slow to sample from" both describe properties an autoencoder might separately have, but neither is the actual obstacle — a small, fast autoencoder fails to generate for exactly the same structural reason as a huge, slow one. "It cannot use gradient descent" is simply false, since an ordinary autoencoder is trained by gradient descent like everything else in this course; the problem is what the objective asks for, not how it is optimised. The general principle, which §6.3.2 builds the whole VAE around fixing, is that reconstruction accuracy alone says nothing about the space between training points, and a generative model specifically needs that space to be well-behaved everywhere, not just at the locations it was shown.'
      },
      {
        q: 'The ELBO is derived from log p(x) using…',
        options: ['a Taylor expansion', 'Jensen\'s inequality applied to the concave log function, after rewriting the marginal likelihood as an expectation under q(z|x)', 'the central limit theorem', 'a first-order Bellman approximation'],
        answer: 1,
        why: 'The derivation in §6.3.2 has exactly one approximating step, and this option names it correctly: multiplying and dividing $p_\\theta(x,z)$ by $q_\\phi(z\\mid x)$ turns the intractable integral defining $\\log p_\\theta(x)$ into $\\log\\mathbb{E}_q[p_\\theta(x,z)/q_\\phi(z\\mid x)]$, and because $\\log$ is concave, Jensen\'s inequality gives $\\log\\mathbb{E}[X]\\ge\\mathbb{E}[\\log X]$, pulling the log inside the expectation and only ever lowering the value — which is exactly why the result is a lower bound rather than an equality. "A Taylor expansion" is a tempting guess because Taylor expansions are the default tool this course reaches for whenever something needs to be replaced with something more tractable, but nothing here is being linearised or approximated locally around a point; the bound holds globally and exactly, with the one inequality doing all the work. "The central limit theorem" and "a first-order Bellman approximation" both borrow machinery from elsewhere in the course (§1.4 and §6.1 respectively) that has no role in this particular derivation — no averaging of many samples, no recursive value definition. The general principle is that turning an intractable exact quantity into a tractable, only-ever-conservative one is usually one specific named inequality doing the entire job, and recognising which one is what separates "I can quote the ELBO" from "I understand where it comes from."'
      },
      {
        q: 'The reparameterisation trick exists because…',
        options: ['sampling is slow', 'you cannot backpropagate through a sampling operation, so the randomness is moved outside the gradient path', 'the KL term has no closed form', 'the decoder is non-differentiable'],
        answer: 1,
        why: 'The actual obstacle is structural, not a limitation of any particular component: "draw a random number" has no gradient with respect to the parameters describing its distribution, so backpropagation cannot reach $\\phi$ through a sampling step no matter how the encoder or decoder are built. Writing $z=\\mu+\\sigma\\odot\\epsilon$ with $\\epsilon\\sim\\mathcal{N}(0,I)$ moves all the randomness into $\\epsilon$, which does not depend on $\\phi$ at all, so the path from $\\mu$ and $\\sigma$ — the parts the encoder actually controls — to $z$ becomes ordinary differentiable arithmetic, a multiply and an add. "The decoder is non-differentiable" is a tempting guess because it locates the problem somewhere plausible-sounding in the network, but the decoder in a VAE is an ordinary neural network like any other and differentiates fine; the break is specifically at the sampling operation between encoder and decoder. "The KL term has no closed form" is also false for the standard Gaussian-prior case used throughout §6.3.2, where the KL divergence has an exact closed-form expression — and even if it did not, that would be a separate problem from the one reparameterisation solves. The general principle, worth carrying beyond VAEs to any model with a stochastic layer in the middle of a computation graph, is that gradients cannot pass through sampling, only through the deterministic transformation of a fixed noise source — which is precisely the trick §6.1\'s policy-gradient derivation has to work around in a different way, because there the randomness cannot be reparameterised out and a log-derivative estimator is used instead.'
      },
      {
        q: 'Mode collapse is…',
        options: ['a bug in the optimiser', 'a stable point of the adversarial game where the generator serves one output that fools D', 'caused by too much data', 'unique to Wasserstein GANs'],
        answer: 1,
        why: 'Mode collapse is a stable point of the minimax game itself: if one narrow output already fools the current discriminator, the generator\'s loss is already low there, so gradient descent has no incentive to explore other outputs, and the §6.3.3 table lists it alongside vanishing gradients and oscillation as one of three named failure modes that follow directly from the objective being a saddle point rather than a single minimum. "A bug in the optimiser" is the tempting reading because it is the natural first suspect whenever training goes visibly wrong elsewhere in this course, but no optimiser fix removes it — the lab makes this concrete by showing collapse appear reliably once the discriminator is helped, regardless of how the generator\'s updates are computed. "Caused by too much data" gets the direction backwards; collapse is if anything easier with abundant, diverse data, since the problem is the game\'s incentive structure, not data scarcity. "Unique to Wasserstein GANs" is also wrong, and revealingly so, since WGAN-GP is one of the standard mitigations named in §6.3.3, not a method especially prone to the failure — the general principle is that a saddle-point objective can have stable equilibria that are bad solutions, which is exactly the structural risk a plain regression objective like diffusion\'s (§6.2) does not carry.'
      }
    ],
    cards: [
      { q: 'Why plain autoencoders fail as generators', a: 'No pressure on the latent space to be filled in between training points — sampled gaps decode to garbage.' },
      { q: 'VAE objective (ELBO)', a: 'reconstruction − KL(q(z|x)‖p(z)), derived via Jensen\'s inequality on log p(x). The KL keeps the latent space continuous and samplable.' },
      { q: 'Reparameterisation trick', a: '$z=\\mu+\\sigma\\odot\\epsilon$, $\\epsilon\\sim\\mathcal{N}(0,I)$ — moves sampling outside the gradient path.' },
      { q: 'GAN objective', a: '$\\min_G\\max_D\\ \\mathbb{E}_x[\\log D(x)]+\\mathbb{E}_z[\\log(1-D(G(z)))]$ — a saddle-point game, not a single minimum.' },
      { q: 'Why diffusion beat GANs', a: 'A stable regression objective instead of a saddle-point game; better mode coverage, no mode collapse, no vanishing generator gradients.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.4 */
  ML.section({
    id: 'gnn', track: 'frontier', num: '6.4',
    title: 'Graph neural networks, recommenders, time series',
    lede: 'Three applied families the notebook cut for space, and each earns its place for the same reason: each one meets data that a plain feedforward network, built for a fixed-size vector, has no honest way to consume — a network of relationships, a sparse matrix of who-liked-what, a sequence where order itself carries the signal.',
    html: `
<h2><span class="sn">6.4.1</span> Graph neural networks: message passing</h2>
<p>Every network in Part 3 assumes its input is a fixed-size vector, or a grid of them (an image), or an ordered sequence of them (text). A social network, a molecule, a set of bank accounts wiring money to each other — none of these are naturally a vector, a grid, or a sequence. They are a set of <b>nodes</b> connected by <b>edges</b>, with no fixed size and no natural ordering: node 17 is not "before" node 4 in any meaningful sense, and the same graph drawn with the nodes relabelled is still the same graph. A GNN is what you get when you insist a network's output must not depend on an arbitrary choice like node numbering, and then ask what operations are still allowed.</p>
${H.analogy(`<p>Message passing is office gossip, formalised. Each round, everyone in the office learns something by combining what they already knew with what their immediate colleagues currently believe — not the whole company, just the people they actually talk to. After one round, you know a summary of your direct neighbours' views. After two rounds, you effectively know a summary of your neighbours' neighbours too, because the information has had a chance to travel one more hop. Run enough rounds and eventually everyone's summary is dominated by the same company-wide consensus, and all the useful local detail — the actual reason you disagreed with the desk next to you — has been averaged away. That final failure has a name in GNNs, and it turns up before the end of this section.</p>`)}
<p>Formally, at every layer $l$, each node $v$ aggregates messages from its neighbours $\\mathcal{N}(v)$ and combines them with its own current representation to produce the next layer's representation:</p>
$$h_v^{(l+1)} = \\phi\\left(h_v^{(l)},\\; \\bigoplus_{u \\in \\mathcal{N}(v)} \\psi(h_u^{(l)}, h_v^{(l)}, e_{uv})\\right)$$
<p>Read this left to right. $\\psi$ (a small learned function) turns each neighbour $u$'s representation, combined with $v$'s own and the edge features $e_{uv}$, into a message. $\\bigoplus$ (the aggregator) combines all those incoming messages from potentially a different number of neighbours at every node into one fixed-size summary. $\\phi$ (another learned function) combines that summary with $v$'s previous representation $h_v^{(l)}$ to produce the updated $h_v^{(l+1)}$. Every node runs the exact same $\\psi$ and $\\phi$, which is what lets the network handle graphs of any size and shape without retraining.</p>
<p>The aggregator $\\bigoplus$ cannot be just anything. Because a node's neighbours have no canonical order — there is no "first" or "second" neighbour, only a set of them — $\\bigoplus$ must be <b>permutation-invariant</b>: relabelling or reordering the same set of neighbours must produce the identical output. Sum, mean and max all satisfy this; concatenating the messages in list order, or feeding them through an RNN, does not, because either would let the arbitrary order in which the neighbours happen to be stored silently change the network's answer for what is provably the same graph.</p>
${H.table(['Variant', 'Aggregator / mechanism', 'Trade-off'], [
      ['<b>GCN</b> (Kipf & Welling, 2017)', 'a degree-normalised mean over neighbours', 'simple, fast, the standard starting point'],
      ['<b>GraphSAGE</b> (Hamilton et al., 2017)', 'samples a fixed number of neighbours rather than using all of them', 'scales to graphs with millions of nodes where "all neighbours" can mean thousands'],
      ['<b>GAT</b> (Veličković et al., 2018)', 'learns an attention weight per neighbour before aggregating', 'self-attention (§4.3) restricted to a graph\'s edges instead of a fully-connected sequence — noticing that equivalence is the useful thing to take away']
    ])}
<p>Stack $L$ layers and each node's representation is influenced by everything within $L$ hops — the same receptive-field logic as a CNN's stacked convolutions (§3.7), with graph distance playing the role pixel distance played there. This is also where the gossip analogy's ending bites: push $L$ too high and every node's representation is dominated by the same graph-wide average, a failure called <b>over-smoothing</b> — the graph analogue of a saturated activation function, where the signal that made one node interesting has been diluted into indistinguishability from its neighbours. GNNs are consequently usually shallow, two to four layers, which is the opposite instinct from "deeper is better" everywhere else in this course, and worth remembering precisely because it violates the pattern.</p>
${H.practice(`<p>Fraud and anti-money-laundering detection is where GNNs earn their keep over an ordinary tabular model, and it is worth being precise about why. A single suspicious transaction row can look entirely unremarkable on its own — a normal amount, a normal time of day, a normal merchant category. What makes it suspicious is topology: this account shares a device fingerprint with twelve others opened last week, all of which wire small amounts to the same third account, forming a ring. A per-row model that only ever sees one transaction at a time structurally cannot see a ring; a GNN whose node representations have propagated two or three hops can, because the ring shows up as an unusual pattern in aggregated neighbour information. This is a genuine case where the topology <i>is</i> the signal, not a convenient reframing of a problem an XGBoost model would have solved anyway.</p>`)}

${H.lab('gnnlab', 'Message passing, one hop at a time', 'A small transaction graph, eight nodes, one flagged as suspicious (choose which). Press "propagate" and watch its signal spread outward hop by hop, exactly as the message-passing equation above describes — a node\'s colour and printed value are its current aggregated signal. Two or three hops usefully spreads the flag to the suspicious node\'s neighbourhood. Keep pressing: every node converges to the same value, and the diagnosis line reports "over-smoothed". That collapse is not a visual artefact of this toy graph; it is the same over-smoothing that limits real GNNs to a handful of layers.')}

<h2><span class="sn">6.4.2</span> Recommenders: the funnel that predicts what you have not seen yet</h2>
<p>A recommender's core data is a sparse matrix: rows are users, columns are items, and the vast majority of cells — did this user interact with this item? — are unobserved, not zero. Predicting the unobserved cells is exactly the missing-data problem §1.8's SVD was built to describe, adapted to matrices too large to decompose exactly and to implicit signals (a click, a watch, a purchase) rather than an explicit rating. <b>Matrix factorisation</b> is the resulting base method: learn two low-rank factor matrices $U$ (users) and $V$ (items) such that $R \\approx UV^\\mathsf{T}$, so that a user's predicted affinity for an item is the dot product of their learned taste vector with the item's learned attribute vector — the same "score is a dot product" idea from §0.2, applied to shopping habits instead of features.</p>
<p>A production system rarely stops at one factorisation model, because scoring every item for every user at request time is too slow once the catalogue reaches millions of items. Instead the standard architecture is a two-stage funnel: a cheap <b>retrieval</b> stage narrows millions of items down to a few hundred candidates, typically with two-tower embeddings and approximate nearest-neighbour search — precisely the retrieval mechanism §5.1 and §5.10 describe for RAG, because it is genuinely the same problem, "find the k items whose embedding is closest to this query embedding," wearing a different name. An expensive <b>ranker</b>, which can afford heavier features and cross-interactions because it only has a few hundred candidates left to score, then orders that shortlist. Cheap-and-wide followed by expensive-and-narrow is a funnel shape that recurs constantly across this course precisely because it is the correct answer whenever scoring everything is too slow but scoring a shortlist well matters.</p>
<p>Two problems are specific to recommenders and worth naming precisely. <b>Cold start</b> — a new user or item with no interaction history — is handled by falling back to content features (item metadata, user demographics) until enough behavioural data accumulates to trust the collaborative signal. The harder problem is the <b>feedback loop</b>: your recommendations determine what the user sees, which determines what they can possibly click, which becomes the training data for the next model. An item never shown is an item that can never be observed as a good recommendation, no matter how good it actually is — this is §1.7's confounding problem, reappearing in a production system as a silent, self-reinforcing bias rather than an abstract warning about causal inference.</p>
${H.history(`<p>The 2006 Netflix Prize, offering $1M for a 10% improvement in rating prediction, is the reason matrix factorisation became the field's default tool — Koren, Bell and Volinsky's 2009 survey of the winning techniques is still the clearest account of the method. Two-tower retrieval architectures were popularised for production scale by Covington et al.'s 2016 paper on YouTube's recommender, which also documented the funnel structure described above as a practical necessity rather than a theoretical nicety. The feedback-loop problem has been formally studied since at least Bottou et al.'s 2013 work connecting recommendation to counterfactual and off-policy evaluation — the same causal-inference machinery §1.7 introduces.</p>`)}

<h2><span class="sn">6.4.3</span> Time series: when order is the feature</h2>
<p>Every model elsewhere in this course assumes rows are exchangeable — shuffle the training set and nothing about the modelling problem changes. A time series is the opposite: yesterday's value is informative about today's precisely <i>because</i> of when it occurred, and shuffling the rows destroys the entire signal. That single difference is why time series gets its own toolkit rather than being "regression with a date column".</p>
<p>The first question is always <b>stationarity</b> — does the series' statistical behaviour (mean, variance, autocorrelation) stay stable over time, or does it drift? A series with a trend or strong seasonality is non-stationary, and most classical methods assume stationarity, so the standard fix is to difference the series (subtract each value from the previous one) or explicitly model the trend and seasonal components before fitting anything else. Three live approaches then cover most practical forecasting problems, and they are not competing so much as suited to different data volumes:</p>
${H.table(['Approach', 'Best suited to', 'Character'], [
      ['Classical ARIMA / exponential smoothing (ETS)', 'a handful of well-behaved series, fit individually', 'interpretable, well-understood confidence intervals, but does not share information across series'],
      ['<b>Gradient-boosted lag and calendar features</b>', 'forecasting at scale — hundreds to thousands of series', 'usually the strongest result per unit of engineering effort: turn the time series problem into a tabular one (§2.8) with lag values and calendar features as inputs, and let a boosted tree find the interactions'],
      ['Global deep models (temporal fusion transformers, N-BEATS, foundation forecasters)', 'thousands of related series with shared structure', 'can borrow strength across series a per-series model cannot, at the cost of far more data and compute to justify']
    ])}
<p>Whichever model you fit, the evaluation has to respect the ordering that made the problem hard in the first place. A random train/test split leaks the future into training — the model gets to see values from after the point it is supposed to be forecasting from, which classical cross-validation (§2.14) was never designed to prevent. <b>Rolling-origin backtesting</b> is the fix: repeatedly pick a cutoff, train only on data before it, forecast forward, and score against what actually happened, sliding the cutoff forward each time. And a forecast interval built from in-sample residuals is systematically too narrow, because in-sample error understates how wrong a model is on data it has never seen — conformal prediction (§2.12) gives an honestly calibrated interval instead, with the caveat that it assumes the future resembles the past closely enough for the calibration to transfer, which is precisely the assumption that fails hardest during a genuine regime change.</p>

${H.lab('tslab', 'Decomposition and the rolling-origin backtest', 'A synthetic series built from trend, seasonality and noise, with sliders controlling each component\'s strength. The plot overlays five rolling-origin backtest folds, each trained only on data before its own cutoff (the dashed vertical line) — that is what makes each fold\'s error an honest estimate rather than a lucky in-sample fit. Switch between the naive, seasonal-naive and lag-feature forecasters and compare their backtest MAE; the readout also reports how much the chosen model beats a naive baseline by, which is the number worth quoting rather than the raw error alone.')}

${H.probe([
      ['Why must a GNN\'s neighbour aggregator be permutation-invariant?', 'A node\'s neighbours have no canonical order — the same graph, relabelled, is the same graph. An order-sensitive aggregator (concatenation, an RNN) would make the network\'s output depend on an arbitrary storage order rather than the graph\'s actual structure.'],
      ['What is over-smoothing, and why does it push GNNs to be shallow?', 'Stacking many message-passing layers lets every node\'s representation absorb information from an ever-larger neighbourhood until, eventually, all nodes converge to the same graph-wide average — the graph analogue of a saturated activation. GNNs are typically 2–4 layers deep specifically to avoid this, unlike most of deep learning where more depth is usually safe.'],
      ['Why does a two-stage retrieval-then-ranking funnel show up in both recommenders and RAG?', 'Because it is the same underlying problem — find the best few candidates out of millions cheaply, then afford to score only the shortlist expensively — wearing different labels in each domain.'],
      ['Why is the recommender feedback loop a causal-inference problem, not just a modelling nuisance?', 'An item never shown can never be observed as a good recommendation regardless of its true quality, so the training data is confounded by the policy that generated it — §1.7\'s confounding problem, playing out as production bias.'],
      ['Why is a random train/test split wrong for a time series?', 'It lets the model train on data that occurred after the point it is meant to be forecasting from — future leakage. Rolling-origin backtesting, training only on data before each cutoff, is the fix.']
    ])}`,
    labs: {
      gnnlab: function (host) {
        const nodes = [
          { x: .2, y: .3, l: 'A' }, { x: .4, y: .18, l: 'B' }, { x: .58, y: .32, l: 'C' },
          { x: .3, y: .58, l: 'D' }, { x: .52, y: .66, l: 'E' }, { x: .74, y: .52, l: 'F' },
          { x: .82, y: .78, l: 'G' }, { x: .16, y: .8, l: 'H' }
        ];
        const edges = [[0, 1], [1, 2], [0, 3], [3, 4], [2, 5], [4, 5], [5, 6], [3, 7], [1, 4]];
        let hops = 0;
        const st = Viz.controls(host, [
          { k: 'agg', label: 'aggregator', type: 'buttons', value: 'mean', options: [{ v: 'mean', t: 'mean' }, { v: 'sum', t: 'sum' }, { v: 'max', t: 'max' }] },
          { k: 'source', label: 'flagged node', type: 'buttons', value: '2', options: [{ v: '2', t: 'C' }, { v: '6', t: 'G' }, { v: '7', t: 'H' }] }
        ], () => { hops = 0; S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'hops', label: 'message-passing layers', cls: 'key' }, { k: 'reached', label: 'nodes with signal' },
          { k: 'spread', label: 'spread of values' }, { k: 'state', label: 'diagnosis' }
        ]);
        function values() {
          let v = nodes.map((_, i) => i === +st.source ? 1 : 0);
          for (let l = 0; l < hops; l++) {
            const nv = v.slice();
            nodes.forEach((_, i) => {
              const nbrs = edges.filter(e => e[0] === i || e[1] === i).map(e => e[0] === i ? e[1] : e[0]);
              if (!nbrs.length) return;
              const vals = nbrs.map(j => v[j]);
              const agg = st.agg === 'mean' ? Num.mean(vals) : st.agg === 'sum' ? Math.min(1.6, Num.sum(vals)) : Math.max.apply(null, vals);
              nv[i] = .5 * v[i] + .5 * agg;
            });
            v = nv;
          }
          return v;
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const v = values();
            const px = n => 40 + n.x * (w - 90), py = n => 30 + n.y * (h - 80);
            edges.forEach(e => {
              ctx.strokeStyle = T.line; ctx.lineWidth = 1.6;
              ctx.beginPath(); ctx.moveTo(px(nodes[e[0]]), py(nodes[e[0]])); ctx.lineTo(px(nodes[e[1]]), py(nodes[e[1]])); ctx.stroke();
            });
            nodes.forEach((n, i) => {
              const t = Math.max(0, Math.min(1, v[i]));
              ctx.beginPath(); ctx.arc(px(n), py(n), 20, 0, 6.3);
              ctx.fillStyle = 'rgba(' + Math.round(80 + 150 * t) + ',' + Math.round(120 - 40 * t) + ',' + Math.round(230 - 150 * t) + ',' + (0.15 + 0.8 * t) + ')';
              ctx.fill();
              ctx.strokeStyle = i === +st.source ? T.red : T.line; ctx.lineWidth = i === +st.source ? 2.4 : 1.2; ctx.stroke();
              ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(n.l, px(n), py(n) - 4);
              ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = T.muted;
              ctx.fillText(v[i].toFixed(2), px(n), py(n) + 8);
            });
            const spread = Num.sd(v);
            out({
              hops: hops, reached: v.filter(x => x > .02).length + ' of ' + nodes.length,
              spread: spread.toFixed(3),
              state: hops === 0 ? 'no propagation yet' : spread < .06 ? 'over-smoothed — all nodes look alike' : 'informative'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Propagate one hop', primary: true, on: () => { hops++; S.redraw(); } },
          { label: 'Propagate 6', on: () => { hops += 6; S.redraw(); } },
          { label: 'Reset', on: () => { hops = 0; S.redraw(); } }
        ]);
        Viz.note(host, 'Two or three hops spread the flag usefully — the neighbourhood of a suspicious account becomes visible. Keep going and every node converges to the same value: over-smoothing, which is why GNNs are usually shallow and why "just add layers" is worse advice here than almost anywhere else.');
      },

      tslab: function (host) {
        const st = Viz.controls(host, [
          { k: 'trend', label: 'trend strength', min: 0, max: 2, step: .05, value: .8, fmt: v => v.toFixed(2) },
          { k: 'season', label: 'seasonal amplitude', min: 0, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) },
          { k: 'noise', label: 'noise', min: .05, max: 1.5, step: .05, value: .4, fmt: v => v.toFixed(2) },
          { k: 'model', label: 'forecaster', type: 'buttons', value: 'lag', options: [{ v: 'naive', t: 'naive' }, { v: 'snaive', t: 'seasonal naive' }, { v: 'lag', t: 'lag features + trend' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mae', label: 'backtest MAE', cls: 'key' }, { k: 'folds', label: 'rolling-origin folds' },
          { k: 'sd', label: 'variation across folds' }, { k: 'best', label: 'beats naive by' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(31), N = 120, period = 12;
            const y = [];
            for (let t = 0; t < N; t++) {
              y.push(st.trend * t / 20 + st.season * Math.sin(2 * Math.PI * t / period) + R.normal(0, st.noise) + 5);
            }
            const forecast = (hist, hzn) => {
              const n = hist.length;
              if (st.model === 'naive') return new Array(hzn).fill(hist[n - 1]);
              if (st.model === 'snaive') return Array.from({ length: hzn }, (_, i) => hist[n - period + (i % period)]);
              // lag features + linear trend
              const X = [], Y = [];
              for (let t = period; t < n; t++) { X.push([1, hist[t - 1], hist[t - period], t]); Y.push(hist[t]); }
              const beta = Num.ridgeFit(X, Y, .5);
              const out2 = []; const buf = hist.slice();
              for (let i = 0; i < hzn; i++) {
                const t = buf.length;
                const p = Num.dot(beta, [1, buf[t - 1], buf[t - period], t]);
                out2.push(p); buf.push(p);
              }
              return out2;
            };
            const folds = [];
            for (let f = 0; f < 5; f++) {
              const cut = 60 + f * 10;
              const pred = forecast(y.slice(0, cut), 10);
              const actual = y.slice(cut, cut + 10);
              folds.push({ cut: cut, pred: pred, mae: Num.mean(actual.map((a, i) => Math.abs(a - pred[i]))) });
            }
            const P = Viz.plot(ctx, w, h, { xd: [0, N], yd: [Math.min.apply(null, y) - 1, Math.max.apply(null, y) + 1] })
              .frame({ xlabel: 'time', ylabel: 'value' });
            P.clip(() => {
              P.line(y.map((v, i) => [i, v]), { color: T.faint, width: 1.6 });
              folds.forEach((f, i) => {
                P.line(f.pred.map((v, j) => [f.cut + j, v]), { color: [T.blue, T.green, T.amber, T.red, T.text][i], width: 2 });
                P.vline(f.cut, { color: T.line, dash: [2, 3], width: 1 });
              });
            });
            const maes = folds.map(f => f.mae);
            const naiveMae = (() => {
              const saved = st.model; st.model = 'naive';
              const m = [];
              for (let f = 0; f < 5; f++) { const cut = 60 + f * 10; const p = forecast(y.slice(0, cut), 10); m.push(Num.mean(y.slice(cut, cut + 10).map((a, i) => Math.abs(a - p[i])))); }
              st.model = saved; return Num.mean(m);
            })();
            out({
              mae: Num.mean(maes).toFixed(3), folds: folds.length,
              sd: '±' + Num.sd(maes).toFixed(3),
              best: st.model === 'naive' ? '—' : (100 * (1 - Num.mean(maes) / naiveMae)).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'Five forecast origins, each trained only on data before its cut — that is rolling-origin backtesting, and the spread across folds is the honest error bar. A single train/test split would have given you one of these numbers and no idea which.');
      }
    },
    quiz: [
      {
        q: 'A GNN’s neighbour aggregator must be…',
        options: ['differentiable only', 'permutation-invariant, because neighbours have no order', 'linear', 'sparse'],
        answer: 1,
        why: 'A node\'s neighbours have no canonical order — the same graph with its nodes relabelled is still the same graph — so the aggregator $\\bigoplus$ must give the identical output regardless of what order the neighbours happen to be stored in, which is exactly what "permutation-invariant" means. Sum, mean and max all satisfy this; concatenating messages in list order, or feeding them through an RNN, does not, because either would let arbitrary storage order silently change the network\'s answer for a graph that has not actually changed. "Differentiable only" is a tempting requirement to name because every trainable component in this course needs to be differentiable, but that property alone is not enough — an RNN aggregator is perfectly differentiable and still wrong, for the order-dependence reason above. "Linear" and "sparse" both describe properties some aggregators or graphs might have but are not requirements on the aggregator itself; GAT\'s attention-weighted sum, for instance, is neither purely linear nor concerned with sparsity, yet it is a valid permutation-invariant aggregator, described in §6.4.1 as self-attention (§4.3) restricted to a graph\'s actual edges rather than a fully-connected sequence. The general principle is that any operation applied to a set, rather than a sequence or a fixed-size vector, has to be invariant to how that set happens to be enumerated, which is the same constraint transformers relax deliberately with positional encodings once order does matter.'
      },
      {
        q: 'Stacking many GNN layers causes…',
        options: ['overfitting only', 'over-smoothing — all node representations converge to the same vector', 'exploding gradients', 'label leakage'],
        answer: 1,
        why: 'Stacking $L$ message-passing layers lets a node see everything within $L$ hops, but push $L$ too high and each node\'s representation keeps absorbing an ever-larger neighbourhood until, eventually, every node converges towards the same graph-wide average — over-smoothing, the graph analogue of a saturated activation function, made directly visible in the gossip-passing lab once enough hops have run. "Exploding gradients" is the tempting wrong answer because it is the default failure this course associates with training very deep networks (§3.5), and it is natural to assume a GNN "getting worse with depth" must be the same phenomenon — but over-smoothing is a property of the representations themselves at convergence, not an optimisation pathology, and it would happen even with perfectly well-behaved gradients. "Overfitting only" and "label leakage" both describe failures that get worse with more parameters or more information, whereas over-smoothing gets worse specifically with more rounds of neighbour averaging, which is a structural property of the message-passing computation, not of how much the model has memorised or seen. The general principle, worth remembering precisely because it cuts against "deeper is usually safe" everywhere else in this course, is that GNNs are typically kept to two to four layers as a direct consequence, the one place in the whole notebook where more depth is a known route to a worse model rather than a merely riskier one.'
      },
      {
        q: 'For thousands of related time series, the strongest approach per unit of effort is usually…',
        options: ['one ARIMA per series', 'gradient boosting on lag and calendar features', 'a bespoke deep model per series', 'exponential smoothing'],
        answer: 1,
        why: 'Turning the time-series problem into a tabular one with lag values and calendar features as inputs lets a single boosted-tree model share statistical strength across every series it is trained on simultaneously, which is exactly the "borrow strength across series" advantage that per-series methods structurally cannot offer — and §6.4.3\'s table names this as usually the strongest result per unit of engineering effort at this scale. "One ARIMA per series" and "exponential smoothing" are both tempting because they are genuinely reasonable, well-understood choices for a handful of well-behaved series fit individually — the trap is applying that same instinct at a much larger scale, where fitting and maintaining thousands of separate classical models becomes an engineering burden with no compensating benefit, since each model only ever learns from its own series. "A bespoke deep model per series" over-corrects in the other direction: global deep models (temporal fusion transformers, N-BEATS) do share structure across series, like the lag-feature approach, but need far more data and compute to justify themselves than gradient boosting does, making them the right choice mainly when thousands of related series actually share strong common structure. Whichever model wins, the general principle from §6.4.3 is non-negotiable: evaluation must use rolling-origin backtesting, training only on data before each cutoff, because a random train/test split leaks future values into training and produces a systematically optimistic score.'
      },
      {
        q: 'A recommender\'s feedback loop is a problem because…',
        options: ['it makes training slower', 'an item never shown can never be observed as a good recommendation, biasing all future training data', 'it requires more storage', 'it only affects cold-start users'],
        answer: 1,
        why: 'An item the system never shows can never be observed as a good recommendation, however good it truly is, so the training data available for the next model is confounded by the very policy that generated it — §1.7\'s confounding problem, reappearing here as a self-reinforcing production bias rather than an abstract warning about causal inference. "It makes training slower" and "it requires more storage" both describe ordinary engineering costs that scale with data volume, but the feedback loop is a bias problem regardless of how fast or well-resourced the training pipeline is — a faster, cheaper pipeline trained on the same confounded data is no less biased. "It only affects cold-start users" is tempting because cold start is the other named recommender-specific problem in §6.4.2 and the two are easy to conflate, but the feedback loop affects every item the system has ever chosen not to show, established users and new ones alike, which is a broader and more durable effect than the temporary cold-start gap that behavioural data eventually closes. The general principle is that a system whose own outputs shape its future training data is not merely learning from the world, it is learning from a world it is actively steering, and any causal claim drawn from that data has to account for the steering — the same discipline §1.7 introduces for confounded observational data generally.'
      }
    ],
    cards: [
      { q: 'Message passing', a: 'Aggregate permutation-invariantly over neighbours, then update. GAT = attention restricted to edges.' },
      { q: 'Over-smoothing', a: 'Deep message passing converges all node representations; keep GNNs shallow (2–4 layers).' },
      { q: 'Matrix factorisation', a: '$R \\approx UV^\\mathsf{T}$ — §1.8\'s SVD idea, adapted to missing entries and implicit feedback.' },
      { q: 'Recommender two-stage', a: 'Cheap retrieval (two-tower + ANN) then expensive ranking — the same funnel as RAG.' },
      { q: 'Forecasting at scale', a: 'Gradient-boosted lag/calendar features, rolling-origin backtests, conformal intervals with a drift caveat.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.5 */
  ML.section({
    id: 'exclusions', track: 'frontier', num: '6.9',
    title: 'What was left out, and how to keep reading',
    lede: 'Every course draws a boundary around what it covers, and most courses never tell you where the boundary is. The source notebook this site is built from listed its own omissions honestly. Most are restored in this part; here is what still is not, why, and — more durably useful than any list of topics — how to tell which of the claims you have just read will still be true in five years and which are already halfway to being wrong.',
    html: `
<h2><span class="sn">6.9.1</span> Why a boundary is a feature, not an apology</h2>
<p>A syllabus that claims to cover "all of machine learning" is making a promise it cannot keep, because the field has no edge — measure-theoretic learning theory, computational biology applications, robotics-specific control, and a dozen other book-length subfields all have some claim to the name. The honest move is not to pretend completeness; it is to state the boundary explicitly, so you know when you have reached the edge of what this particular resource can tell you and when it is time to reach for something else. That is what the original notebook this site expands did, and what this table continues.</p>
${H.table(['Excluded from the source notebook', 'Status on this site, and why'], [
      ['Measure-theoretic probability, convergence proofs, PAC/VC derivations', 'Still excluded, deliberately. A practitioner needs to know <i>that</i> and roughly <i>why</i> generalisation works (§1.4 builds the concentration-inequality intuition without the full measure theory); the formal proofs are a specialised skill for a specific kind of research role. If you want them, Shalev-Shwartz & Ben-David\'s <i>Understanding Machine Learning</i> is the standard reference and does not need this site\'s help.'],
      ['Full ARIMA/Box–Jenkins methodology and recommender systems', '<b>Restored in outline</b> (§6.4), at the depth a practitioner actually reaches for — enough to know when classical time series methods are the right tool and when gradient boosting or a global model beats them, without the full Box–Jenkins identification ritual.'],
      ['The CNN architecture zoo and RL beyond post-training', '<b>Restored</b> (§3.7 for the architecture history, §6.1 for RL built up from first principles rather than assumed).'],
      ['Diffusion and image-generation internals', '<b>Restored</b> (§6.2 and §6.3), including a derivation of why the forward process has a closed form and a working 2-D demo you can train in the browser.'],
      ['Framework API tutorials and vendor SDK syntax', 'Still excluded, deliberately, and this is the one exclusion that will not change. An API surface can be entirely different a year from now; §5.6 compares the underlying philosophies of agent frameworks instead of documenting any one library\'s function signatures, because the philosophy ages far slower than the syntax.'],
      ['Speculative model version numbers and leaderboard positions', 'Still excluded from every factual claim on this site. §6.9.3 below explains exactly why, in numbers rather than as a slogan.']
    ])}

<h2><span class="sn">6.9.2</span> What is genuinely contested, and must be flagged as such</h2>
<p>A separate, more important category than "what was left out" is "what is included, but is not as settled as it sounds". Three specific claims elsewhere on this site are conventions or contested empirical fits rather than derived facts, and treating them as derived facts is a specific, nameable mistake — the kind that costs credibility in an interview or a design review the moment someone who has actually read the underlying paper notices the overclaim.</p>
${H.checklist([
      '<b>The exact Chinchilla coefficients</b> (§4.10) — the headline ~20 tokens per parameter ratio is robust and has survived scrutiny; the specific fitted exponents in the original paper have not. Epoch AI\'s 2024 replication attempt, working from the same underlying data where it could be reconstructed, found discrepancies in one of the three fitting approaches used in the original — reasonable people can and do report different numbers here, and stating the ratio as a rounded rule of thumb is more honest than quoting a coefficient to three decimal places.',
      '<b>"What works" for class imbalance</b> (§2.12) — a 2022 JAMIA study found that resampling techniques (SMOTE and its relatives) frequently make calibration worse without improving discrimination for clinical risk prediction, contradicting what is still the majority folk practice in industry. Both the evidence and the entrenched practice are real; the honest position states the objective (calibration? ranking? recall at a fixed cost?) before prescribing a fix, rather than reciting "always use SMOTE for imbalance" as if it were settled.',
      '<b>PSI and IV thresholds</b> (§2.11, §2.18) — the population stability index bands (0.1, 0.25) and information value bands used across credit risk and monitoring are industry conventions inherited from practice, not thresholds derived from any hypothesis test or type-I error rate. Calling them conventions, out loud, when you use them is a small habit that reliably signals you understand the difference between a statistical result and an agreed-upon rule of thumb.'
    ])}
${H.note('Flagging is not hedging. It is the thing that makes the rest of your answers credible: an interviewer or a colleague who hears you draw a clean line between a derived result and a convention will trust the derived results you state afterwards considerably more — the flag is evidence you know which is which everywhere else too, not only on the three items named here.')}

<h2><span class="sn">6.9.3</span> How to keep reading without drowning</h2>
<p>The material on this site was frozen at a point in time; the field was not. The single most useful skill for staying current without spending every evening reading release notes is knowing <i>which layer</i> a new piece of information belongs to, because the layers age at wildly different rates and deserve wildly different amounts of your memory.</p>
${H.analogy(`<p>Think of the field as a building rather than a single flat surface. The foundation and load-bearing walls — linear algebra, probability, calculus — were poured decades ago and are not being rebuilt; learn them once, properly, and stop checking whether they still hold. The plumbing and wiring — classical ML's objectives, metrics, and validation discipline — get renovated occasionally but the underlying system does not change shape. The interior fittings — a specific architecture, a specific training recipe — get replaced every few years as better ones arrive, and it is a waste of effort to memorise the exact model of a light fixture that will be swapped out. And the paint on the walls — this quarter\'s leaderboard rankings, this month\'s model version numbers — needs redoing constantly and tells you nothing about the building\'s structure. Spend your memorising effort on the foundation, learn to recognise the plumbing when you see it repeated in a new outfit, and never mistake fresh paint for a structural fact.</p>`)}
${H.table(['Layer', 'Half-life', 'How to treat it'], [
      ['Mathematics (Parts 0–1)', 'Decades', 'Learn once, properly. Nothing here will be obsolete in your working lifetime.'],
      ['Classical ML (Part 2)', 'A decade', 'Stable. The tooling and library names change; the objectives, metrics and validation discipline do not.'],
      ['Architecture and training (Parts 3–4)', 'Two to five years', 'Learn the mechanism (attention, normalisation, low-rank adaptation) rather than the specific model it currently lives in — the mechanism outlives the model by years.'],
      ['Serving and tooling (Parts 4–5)', 'Six to eighteen months', 'Learn the arithmetic (bandwidth, cache size, cost per successful query) — the numbers on any given day change, but the calculation that produces them does not.'],
      ['Model rankings and version numbers', 'Weeks', 'Do not memorise. Cite mechanisms in an answer; check a model card before asserting a specific version or benchmark score, because it will very likely be out of date by the time you say it.']
    ])}
${H.worked('sorting a real claim into a layer', `<p>Take the sentence "GQA reduces KV-cache memory by sharing key/value heads across query heads, which is why modern serving stacks can afford longer context windows" (§4.4, §4.7). Which layer is it? Not the top layer — it names no model or version. Not quite the bottom layer either — it is not pure mathematics, it is a specific engineering mechanism. It sits in "architecture and training": a two-to-five-year fact. Compare it with "Llama 4 uses a 10M-token context window" — that sentence names a specific model and a specific number, and belongs in the bottom layer, the one you should never bother memorising and should always verify before repeating.</p>`)}
${H.key('Talk about mechanisms rather than leaderboard positions. Mechanisms stay true for years; rankings are stale within weeks — often before the page announcing them has finished loading.')}

<h2><span class="sn">6.9.4</span> Where to go next</h2>
<ul>
<li><b>To go deeper on the mathematics:</b> Bishop, <i>Pattern Recognition and Machine Learning</i>, for the classical Bayesian treatment this site's Part 1 compresses; Murphy, <i>Probabilistic Machine Learning</i>, for the modern and considerably longer version of the same; Boyd & Vandenberghe on convex optimisation for the parts of §1.12 that were necessarily left as assertions here.</li>
<li><b>To go deeper on modern systems:</b> the primary papers listed in <a href="#/sources">Sources</a> — every non-trivial derivation on this site is traceable to one of them, and reading the original paper is almost always faster and more precise than reading a summary of it, this one included.</li>
<li><b>To get better at the applied craft, which no amount of reading substitutes for:</b> build the smallest version of the thing, instrument it, and write down what surprised you. Every worked number and every pitfall box on this site exists because someone, at some point, was genuinely surprised by it — that is where the honest content in a course like this one actually comes from, and it is the same process available to you.</li>
</ul>`,
    quiz: [
      {
        q: 'Which of these should always be flagged as a convention rather than a derived result?',
        options: ['The Hoeffding bound', 'PSI thresholds of 0.1 and 0.25', 'The logistic gradient', 'The RRF formula'],
        answer: 1,
        why: 'PSI bands of 0.1 and 0.25, and the matching information-value bands, are industry conventions inherited from practice — thresholds people agreed were useful, not numbers that fall out of a type-I error rate or a hypothesis test the way a p-value cutoff would, which is exactly the distinction §6.9.2 names as the one worth flagging out loud. The Hoeffding bound is the tempting wrong pick precisely because it is also a threshold-flavoured number with a Greek letter attached, and someone who has just spent §1.4 learning to distrust round numbers might overcorrect and start suspecting every bound in the course — but Hoeffding\'s inequality is a genuinely derived probabilistic guarantee, provable from first principles, not a convention anyone chose by agreement. The logistic gradient and the RRF formula are similarly derived: one falls out of differentiating a log-likelihood, the other is a defined combination rule, and neither carries the "someone picked this number because it seemed to work" character that PSI and IV thresholds do. The general principle, which §6.9.2 treats as more important than any individual flagged claim, is that stating out loud which of your numbers are derived and which are agreed-upon conventions is what makes the rest of what you say credible — the four-fifths rule in fairness carries exactly the same convention-not-derivation flag for the same reason.'
      },
      {
        q: 'The Chinchilla ~20:1 token-to-parameter ratio and its exact fitted exponents differ in status because…',
        options: ['both are equally solid', 'both are equally contested', 'the ratio has been robustly replicated while the specific fitted exponents have not, per Epoch AI\'s 2024 replication', 'the ratio was never published'],
        answer: 2,
        why: 'The two claims have genuinely different evidentiary status, which is exactly why they need to be separated rather than reported as one fact: the ~20-tokens-per-parameter ratio is the robust, repeatedly-observed headline finding, while Epoch AI\'s 2024 replication attempt found discrepancies in one of the three fitting approaches used to produce the original paper\'s specific exponents — a downstream detail, not the headline result itself. "Both are equally solid" and "both are equally contested" are the two tempting ways to collapse a two-level distinction into one level, and each is wrong in the opposite direction: treating the exponents as equally solid overstates confidence in a number that has not survived scrutiny, while treating the ratio as equally contested throws away a genuinely robust result along with the shakier one. "The ratio was never published" is simply false and easy to rule out by checking the original Chinchilla paper directly, which is precisely the kind of claim §6.9.3 says should be verified rather than asserted from memory. The general principle §6.9.2 is built around is that a robust headline finding and the specific coefficients used to produce it are not the same claim and do not deserve the same confidence — quoting the rounded ratio is more honest than quoting a coefficient to three decimal places, exactly as the section states.'
      },
      {
        q: 'The best defence against material going stale is…',
        options: ['memorising the latest benchmark table', 'learning mechanisms and arithmetic rather than rankings, and sorting new claims into their correct half-life layer', 'reading release notes weekly', 'avoiding the frontier entirely'],
        answer: 1,
        why: 'The layered table in §6.9.3 gives each kind of claim a genuinely different half-life — mathematics for decades, classical ML for about a decade, architecture and training mechanisms for two to five years, serving arithmetic for six to eighteen months, and rankings or version numbers for mere weeks — so knowing which layer a new fact belongs in tells you directly how much of your memory it is worth spending. "Memorising the latest benchmark table" is the most tempting wrong answer precisely because it feels like the most concrete, actionable way to "stay current", but the worked example in §6.9.3 shows exactly why it fails: a benchmark table is bottom-layer paint, true for weeks and then quietly wrong, while the mechanism behind a result (GQA reducing KV-cache memory by sharing key/value heads, say) sits in the two-to-five-year layer and stays true across several generations of models that use it. "Reading release notes weekly" treats every piece of information as equally worth tracking, which is exhausting and unnecessary once you can sort a claim by layer instead — most of what changes weekly is exactly the layer this course advises against memorising at all. "Avoiding the frontier entirely" throws out the two middle layers, architecture and serving arithmetic, that are genuinely worth learning and that do not go stale on a weekly cycle. The general principle, stated directly in §6.9.3\'s building analogy, is to talk about mechanisms rather than leaderboard positions, because mechanisms are what stays true while the paint gets redone.'
      }
    ],
    cards: [
      { q: 'The three contested claims', a: 'Exact Chinchilla coefficients (ratio solid, exponents disputed) · what works for class imbalance (JAMIA 2022 vs. entrenched practice) · PSI and IV thresholds (convention, not derived). Flag them every time.' },
      { q: 'Knowledge half-life', a: 'Maths: decades. Classical ML: a decade. Architecture/training: 2–5 years. Serving arithmetic: 6–18 months. Rankings/versions: weeks.' },
      { q: 'The building analogy', a: 'Foundation (maths) is never rebuilt; plumbing (classical ML) is renovated rarely; fittings (architectures) are swapped every few years; paint (rankings) needs constant redoing and tells you nothing structural.' }
    ]
  });
})();
