/* ============================================================
   PART 1 — Foundations (1.12 – 1.15): optimisation theory,
   Monte Carlo and the bootstrap, Bayesian inference in practice,
   and floating point.
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 1.12 */
  ML.section({
    id: 'optimization', track: 'foundations', num: '1.12', level: 3,
    title: 'Optimisation: convexity, conditioning, duality, KKT',
    lede: 'Training is optimisation. Almost every training pathology — the loss that will not move, the one that oscillates, the one that needs a schedule — is a statement about curvature, and almost every classical model with a "dual form" is a statement about Lagrange multipliers.',
    prereq: ['calculus-ml'],
    related: ['optimisers', 'svm', 'regularization'],
    html: `
${H.tldr([
      'Convex means every local minimum is global. Logistic regression, ridge, lasso and SVMs are convex; neural networks are not, and it matters less than you would think (§2.2).',
      'The <b>condition number</b> $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ of the Hessian sets the speed limit: gradient descent needs $O(\\kappa\\log(1/\\epsilon))$ steps, momentum $O(\\sqrt{\\kappa}\\log(1/\\epsilon))$, Newton $O(\\log\\log(1/\\epsilon))$ but at $O(d^3)$ per step.',
      'Every constrained problem has a dual. <b>Strong duality plus complementary slackness</b> is why the SVM depends only on its support vectors, and why the kernel trick is possible at all (§2.6).'
    ])}

<p>§1.9 already gave you the working parts. It showed you the Hessian, proved that a positive semi-definite one rules out bad valleys, derived Newton's step from a quadratic model in five lines, and — inside the PCA derivation — showed you a constrained optimum where the objective's gradient and the constraint's gradient turn out to be parallel. All of that is true, and none of it is the whole story. Three questions were left open on purpose, because each one needs room of its own to answer properly.</p>

<p>First: convexity is a yes-or-no fact about an objective, and most of the objectives that actually matter in this course — a neural network's loss, chief among them — fail the test outright. Does that make them hopeless to optimise? Second: §1.9 proved gradient descent's per-step rate is $(\\kappa-1)/(\\kappa+1)$ and stopped there. Is that the best <i>any</i> method built only from gradients can do, or merely the best that one particular update rule can do? Third — and this is the one worth slowing down for — the PCA example in §1.9.5 was a friendly special case: an equality constraint, solvable by setting a derivative to zero. Real constraints are usually inequalities: a budget you must not exceed, a margin you must maintain, a probability that must stay non-negative. Inequality constraints need a different piece of machinery entirely, and that machinery is the reason the SVM, the constrained form of ridge regression, and the $\\beta$ sitting inside every RLHF objective all work the way they do. That machinery — duality — is the hardest material in Part 1, and §1.12.3 below builds it from a picture you can draw on a napkin, not from the algebra first.</p>

<h2><span class="sn">1.12.1</span> Living with non-convexity</h2>
<p>Recall the test from §1.9.2: a twice-differentiable $f$ is convex exactly when its Hessian is positive semi-definite everywhere, and a convex objective has the one property that makes optimisation trustworthy rather than merely hopeful — every point where the gradient vanishes is <i>the</i> global minimum, not just a candidate. What §1.9 did not do is tell you which models you actually train are convex, and what to do about the ones that are not.</p>
${H.table(['Problem', 'Convex?', 'Consequence'], [
      ['Linear/ridge regression', 'yes, strictly (with $\\lambda>0$)', 'unique solution, closed form'],
      ['Lasso', 'yes, not strictly', 'unique fit, possibly non-unique $w$ under collinearity'],
      ['Logistic regression', 'yes', 'unique optimum unless the data are separable, when $\\|w\\|\\to\\infty$ — which is what regularisation actually fixes'],
      ['SVM (hinge + L2)', 'yes', 'a quadratic program with a dual; global optimum guaranteed'],
      ['k-means objective', 'no (in the assignments)', "Lloyd's algorithm finds a local optimum; restart it (§2.9)"],
      ['Neural networks', 'no', 'many minima — but at scale most are of similar quality, and saddle points, not minima, are the obstacle (§3.5)']
    ])}
<p>The last row is the one worth understanding rather than accepting. A neural network's loss surface has, by construction, entire families of equally-good minima — permute two hidden units and every weight that touches them and you get a different point in parameter space with an identical loss — so "unique global minimum" was never on the table. The real question is whether gradient descent gets stuck somewhere <i>bad</i> before it reaches anywhere good, and the answer turns out to hinge on a distinction worth drawing carefully.</p>
${H.analogy(`<p>A true local minimum is the bottom of a sink: every direction you could step from there leads back uphill, so there is no way out without a deliberate, sustained climb. A saddle point is a mountain pass: it is the low point along the ridge you are walking, but the high point along the trail that crosses it — most directions from a pass lead uphill, yet at least one leads down. A blindfolded hiker taking small random steps escapes a pass eventually, because sooner or later a step lands on that one downhill direction. The same hiker never escapes a genuine sink, no matter how long they wait.</p>
<p>Stochastic gradient descent is that blindfolded hiker, and the "random steps" are the noise that comes from estimating the gradient on a mini-batch rather than the full dataset. That noise, which looks like a nuisance everywhere else, is precisely what lets training walk through a pass it would otherwise sit at for a very long time.</p>`)}
${H.history(`<p>Treating non-convexity as close to disqualifying was the mainstream, reasonable position through the 1980s and into the 1990s: with no guarantee against getting permanently trapped, why trust the answer at all? What actually happens when you train a network with a hundred million parameters anyway is not a theorem anybody proved in advance — it is an empirical surprise, and the evidence only accumulated well into the 2010s, alongside the hardware that made training such networks routine enough to observe.</p>
<p>The picture that emerged has two parts. First, the mountain-pass argument above: a random stationary point in a $d$-dimensional landscape is a true minimum only if <i>every one</i> of $d$ independent curvature directions happens to curve upward, and for $d$ in the millions that coincidence is exponentially unlikely — passes, not sinks, are what SGD typically meets. Second, and more surprising, the passes and shallow sinks that large networks do settle into tend empirically to sit at similar loss to one another, so the landscape behaves more like a rolling plain punctuated by occasional dips than a single mountain range with one true summit worth searching for.</p>
<p>The honest caveat survives all of this: it is an empirical picture, backed by theory only in idealised, simplified regimes, not a proof about the specific network you are about to train. Non-convexity is a real cost, and §2.2 is where you pay it — in the form of run-to-run variance that a convex model never has to report.</p>`)}

<h2><span class="sn">1.12.2</span> Conditioning is the speed limit</h2>
<p>§1.9.4 already derived the number that matters: near a minimum every smooth loss looks like a quadratic bowl, rotating into the Hessian's eigenbasis decouples the coordinates, and gradient descent's best achievable per-step contraction works out to $(\\kappa-1)/(\\kappa+1) \\approx 1-2/\\kappa$, where $\\kappa=\\lambda_{\\max}/\\lambda_{\\min}$. Reaching accuracy $\\epsilon$ therefore costs $O(\\kappa\\log(1/\\epsilon))$ steps. That derivation used nothing but the current gradient at each step, which raises the question this section actually answers: is $O(\\kappa)$ a property of gradient descent, or a property of <i>every</i> method that only ever looks at gradients? If it is the latter, no amount of cleverness inside plain gradient descent will ever beat it — you would need a different algorithm, not a better learning-rate schedule.</p>
${H.key('One learning rate must serve every direction at once. That is the entire problem, and $\\kappa$ measures how badly it fails.')}
${H.table(['Method', 'Iterations to accuracy ε', 'Cost per iteration', 'When it is the right answer'], [
      ['Gradient descent', '$O(\\kappa\\log\\frac1\\epsilon)$', '$O(d)$', 'κ is small, or d is enormous'],
      ['+ Momentum / Nesterov', '$O(\\sqrt{\\kappa}\\log\\frac1\\epsilon)$', '$O(d)$', 'almost always — it is free'],
      ['Adam / RMSProp', 'problem-dependent', '$O(d)$', 'badly scaled coordinates, sparse gradients (§3.5)'],
      ['L-BFGS', 'superlinear', '$O(md)$', 'smooth, deterministic, medium $d$ — full-batch classical models'],
      ['Newton', '$O(\\log\\log\\frac1\\epsilon)$', '$O(d^3)$ solve', '$d$ in the hundreds; the gold standard for GLMs (IRLS)']
    ])}
${H.note('This is why feature scaling is not cosmetic. Standardising inputs shrinks $\\kappa$ directly; batch normalisation (§3.6) is the same idea applied to every hidden layer, and Adam approximates a diagonal preconditioner that undoes per-coordinate scale.')}

<p>The answer is that gradient descent itself is not optimal, and the derivation below shows exactly where it leaves progress on the table. Before running it, here is what the lab beneath it will show you doing that arithmetic in real time.</p>

<p><b>What you are looking at.</b> The left panel is the same eigenbasis picture as §1.9's ravine lab, redrawn: the horizontal axis is the flat direction (curvature $\\lambda=1$), the vertical axis the steep one (curvature $\\lambda=\\kappa$), and the faint closed curves are contours of the quadratic bowl. Three trajectories are drawn from the same starting point — grey-blue for gradient descent, a second colour for momentum, and a marked point for Newton, which needs only one or two dots because it converges in a single step. The right panel plots $\\log_{10}(\\text{loss})$ against iteration for all three, so a straight line downward is geometric (exponential) convergence and the slope of that line is the contraction rate itself, made visible.</p>

<p><b>What to do with it.</b> The sliders set $\\kappa$, the learning rate as a fraction of the largest stable step $2/\\lambda_{\\max}$, and momentum's $\\beta$. Start at the default $\\kappa=20$ and read the two step counts in the readout: gradient descent and momentum. Now push $\\kappa$ to 50 and watch both numbers grow — but not at the same rate. Then push the learning rate fraction past 1.00 and watch gradient descent diverge along the steep direction while it is still barely making progress along the flat one, the two failure modes that make a single shared learning rate so hard to choose in the first place.</p>

<p><b>The thing genuinely worth noticing.</b> Compare the "theory: κ vs √κ" readout against the actual step counts as you vary $\\kappa$. At $\\kappa=50$ the theory predicts roughly a 7-to-1 advantage for momentum ($\\sqrt{50}\\approx7.1$), and the measured step counts track that ratio closely, not exactly — a genuine algorithm has constants the asymptotic notation hides, but the <i>scaling</i> is exactly what the derivation below predicts. Newton, meanwhile, needs one step regardless of $\\kappa$, because a quadratic is exactly the function Newton's method assumes; the price for that indifference to conditioning is the $O(d^3)$ Hessian solve the readout does not show you, and which is why Newton never appears in the race once $d$ is more than a few hundred.</p>

${H.lab('convrace', 'The convergence race — watch the condition number bite', 'A genuine quadratic bowl whose curvature ratio you control. Every optimiser here runs its real update rule; the trajectories are what they actually do. Push κ to 50 and gradient descent zig-zags across the valley while momentum sails down it.')}

<p>Here is the derivation the lab is racing. It starts from exactly the line §1.9.4 ended on.</p>

${H.deriv('why momentum earns the square root', [
      ['$w_{t+1} = w_t - \\eta\\nabla f(w_t)$', 'Plain gradient descent. In the eigenbasis, coordinate $i$ obeys $e_{t+1} = (1-\\eta\\lambda_i)e_t$.'],
      ['$\\text{rate} = \\max_i|1-\\eta\\lambda_i|$', 'The slowest coordinate sets the pace. Minimising the max over $i$ gives $\\eta^\\star = 2/(\\lambda_{\\min}+\\lambda_{\\max})$…'],
      ['$\\text{rate}^\\star = \\dfrac{\\kappa-1}{\\kappa+1} \\approx 1-\\dfrac{2}{\\kappa}$', '…and even at the optimal step, progress per iteration is $O(1/\\kappa)$. This is a hard limit for any method that only looks at the current gradient.'],
      ['$v_{t+1} = \\beta v_t + \\nabla f(w_t),\\; w_{t+1}=w_t-\\eta v_{t+1}$', 'Momentum adds state. The per-coordinate recursion becomes second order — a damped oscillator rather than a leaky bucket.'],
      ['$\\text{rate} = \\dfrac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1}$', 'With $\\beta$ tuned to $\\left(\\frac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1}\\right)^2$, the two roots of that recursion have equal modulus $\\sqrt{\\beta}$ and the rate improves to $O(1/\\sqrt\\kappa)$. At $\\kappa=10^4$ that is 100 iterations instead of 10,000.']
    ], 'Nesterov’s method achieves this rate with a provable lower-bound match — no first-order method can do better on this problem class. That is the real content of "accelerated gradient": not a heuristic that helps, an algorithm that is optimal.')}

${H.history(`<p>The two halves of that story are forty years apart. Boris Polyak proposed the momentum update in 1964, motivated by an analogy to a heavy ball rolling in a valley — heavy enough that it does not react to every small wiggle of the surface, so it damps the oscillation across a ravine while continuing to accumulate speed along it. That is a physically motivated heuristic, and for decades it was understood only as one: it visibly helped, and nobody had proved it could not be beaten.</p>
<p>Yurii Nesterov closed the gap in 1983 with a small but decisive change — evaluate the gradient not at the current point but at a point extrapolated slightly ahead along the current velocity — and proved a matching lower bound: no algorithm that only queries gradients can beat $O(\\sqrt\\kappa\\log(1/\\epsilon))$ on this problem class. That turns "momentum helps" into "momentum is, up to constants, the best that first-order information allows", which is a fundamentally different kind of claim. Deep learning did not discover acceleration; it inherited a closed problem from convex optimisation and kept using the answer even after the guarantees stopped strictly applying.</p>`)}

<h2><span class="sn">1.12.3</span> Inequality constraints: a hill with a fence around it</h2>

<p>§1.9.5 handled an <i>equality</i> constraint — PCA's requirement that $\\|w\\|=1$ exactly — by folding it into a Lagrangian and setting the combined gradient to zero. That worked because an equality constraint is a thin surface you are confined to, with no notion of being "comfortably inside" it. Most real constraints are not like that. A regulariser bounds a weight vector's norm from <i>above</i>. An SVM asks every point to be classified with at least some margin, not exactly some margin. A budget must not be exceeded, but spending less is allowed. These are <b>inequality</b> constraints, $g(x)\\le 0$, and they behave differently in a way worth seeing geometrically before any algebra touches it.</p>

<p>Picture a landscape of hills — this is a <i>maximisation</i> for a moment, because "profit" is more intuitive to want more of than "loss" — representing how good each point $(x,y)$ is. Now fence off part of the map: you are not allowed to leave the enclosed region, say everywhere satisfying $x+y\\le c$ for some budget $c$. Two things can happen. If the true, unconstrained hilltop already sits inside the fence, the fence is irrelevant — you were never going to leave anyway, and the ordinary zero-gradient condition finds the best point on its own. But if the hilltop lies outside the fence, you cannot reach it. You walk toward it, hit the fence, and now face a genuinely new question: given that you must stay inside, which point on the fence itself is best?</p>

<p>Standing on the fence, you have exactly one useful move: slide along it, in either direction. If sliding one way increases how good your position is, slide that way — you are not yet at the constrained optimum, because there was a free improvement available without leaving the fence. You are stuck, in the good sense, only when <i>neither</i> direction along the fence improves your position at all. That happens exactly when the hill's slope, at your feet, has no component running along the fence — the only component of the slope left over is the one pushing directly against the fence, perpendicular to it. <mark>At a constrained optimum, the objective's gradient is parallel to the constraint's gradient — pointing straight into the wall you are pressed against.</mark> This is the same "parallel gradients" fact §1.9.5 proved for an equality constraint; you now have the version for an inequality, and you have it from a picture rather than an assertion.</p>

${H.worked('the exact numbers the lab below starts with', `<p>Take $f(x,y)=(x-2)^2+(y-1.4)^2$ — a bowl whose unconstrained minimum sits at $(2,\\,1.4)$ — subject to the fence $x+y\\le1.2$. Since $2+1.4=3.4$ is well outside the fence, the constraint is active and the true minimum is unreachable.</p>
<p>Differentiating $\\mathcal{L}=(x-2)^2+(y-1.4)^2+\\lambda(x+y-1.2)$ and setting both partials to zero gives $x=2-\\lambda/2$ and $y=1.4-\\lambda/2$. Substituting into the (now-active) constraint $x+y=1.2$: $3.4-\\lambda=1.2$, so $\\lambda=2.2$. That gives $x=2-1.1=0.9$ and $y=1.4-1.1=0.3$ — the point on the fence closest to the unreachable hilltop, in exactly the sense a shadow (§0.2.3) is the closest point a light can reach.</p>
<p>Check the sign: $\\lambda=2.2>0$. That is not incidental — a fence can only ever push you back into the feasible region, never pull you further out, so its multiplier can never be negative. A negative $\\lambda$ would mean the constraint was helping you improve, which is a contradiction: if it were helping, it would not have been holding you back in the first place.</p>`)}

<p>That sign restriction is worth dwelling on, because it is the one genuinely new piece of algebra an inequality constraint adds over an equality one. Fold both kinds of constraint into a single Lagrangian at once:</p>
$$\\mathcal{L}(x,\\lambda,\\nu) = f(x) + \\sum_i \\lambda_i g_i(x) + \\sum_j \\nu_j h_j(x), \\qquad \\lambda_i \\ge 0$$
<p>Read it as bookkeeping for the picture above: $g_i(x)\\le 0$ are the fences, $h_j(x)=0$ are the thin equality surfaces from §1.9.5, and each fence gets its own multiplier $\\lambda_i$, forced non-negative for exactly the reason worked out numerically above. The equality multipliers $\\nu_j$ carry no sign restriction, because a thin surface can push you back from either side.</p>

<p>Now ask a question that has nothing to do with any particular $x$: however good is the best <i>achievable</i> value of $f$, could a cheap, wrong-looking calculation ever tell you something true about it — a lower bound you can compute without knowing the answer? It can, and the argument is three lines.</p>

${H.deriv('weak duality: a lower bound you get for free', [
      ['$d(\\lambda,\\nu) := \\inf_x \\mathcal{L}(x,\\lambda,\\nu)$', 'Define the <b>dual function</b>: for fixed multipliers, minimise the Lagrangian over every $x$, feasible or not. This infimum is always $\\le$ the Lagrangian at any one specific $x$, including a feasible one.'],
      ['$\\mathcal{L}(x^\\star,\\lambda,\\nu) = f(x^\\star) + \\sum_i\\lambda_i g_i(x^\\star)$', 'Evaluate at the true constrained optimum $x^\\star$ (equality terms vanish there by feasibility, so they drop out). Every $g_i(x^\\star)\\le0$ by feasibility, and every $\\lambda_i\\ge0$ by the sign restriction just derived, so every term $\\lambda_i g_i(x^\\star)$ is $\\le 0$.'],
      ['$\\mathcal{L}(x^\\star,\\lambda,\\nu) \\le f(x^\\star)$', 'A sum of non-positive terms added to $f(x^\\star)$ cannot exceed $f(x^\\star)$. Adding the fence-pressure terms can only ever pull the Lagrangian value down, never up, at a feasible point.'],
      ['$d(\\lambda,\\nu) \\le \\mathcal{L}(x^\\star,\\lambda,\\nu) \\le f(x^\\star)$', 'Chain the two inequalities. This holds for <i>every</i> choice of $\\lambda\\ge0,\\nu$ — the dual function is a lower bound on the true optimum $f(x^\\star)$ no matter which multipliers you plug in, computed by a minimisation that never had to know what $x^\\star$ was.']
    ], 'This is weak duality, and notice what it did not assume: nothing about $f$ or $g$ being convex. The bound is true, free, and universal — the only question left is how tight it is, and that is where convexity finally earns its keep.')}

<p>When the primal problem is convex and mildly well-behaved (<b>Slater's condition</b>: some point exists that satisfies every inequality <i>strictly</i>, with room to spare, not merely on the boundary), the gap between the dual bound and the true optimum closes completely — $\\max_{\\lambda\\ge0,\\nu} d(\\lambda,\\nu) = f(x^\\star)$ exactly. That is <b>strong duality</b>, it is the deeper fact that convexity buys you beyond "no bad valleys", and at that point the solution to the constrained problem satisfies four conditions collectively called <b>KKT</b>, three of which you have already met in disguise:</p>
${H.table(['Condition', 'Statement', 'What it means, in the fence picture'], [
      ['Stationarity', '$\\nabla f + \\sum\\lambda_i\\nabla g_i + \\sum\\nu_j\\nabla h_j = 0$', 'the pull of the hill is exactly cancelled by the push of every fence and surface touching you — the worked example above, in general form'],
      ['Primal feasibility', '$g_i(x)\\le 0,\\; h_j(x)=0$', 'you are actually standing inside the fenced region, on the allowed surfaces'],
      ['Dual feasibility', '$\\lambda_i \\ge 0$', 'a fence can only push you back in, never pull you further out — proved numerically above'],
      ['<b>Complementary slackness</b>', '$\\lambda_i\\, g_i(x) = 0$', '<b>a fence you are not touching cannot be pushing on you</b> — either $g_i(x)=0$ (you are pressed against it) or $\\lambda_i=0$ (it has nothing to do with your position)']
    ])}
${H.key('Complementary slackness is the money condition. It says inactive constraints have zero multiplier — which is exactly why an SVM’s solution depends only on the points touching the margin, and why $\\alpha_i > 0$ identifies the support vectors (§2.6).')}

<p><b>What you are looking at.</b> A single 2-D minimisation, drawn live: faint closed contours are the bowl $f(x,y)=(x-a)^2+(y-b)^2$ around its unconstrained optimum (the grey ring), the diagonal line is the fence $x+y=c$, and the red-tinted half-plane beyond it is off-limits. The solid coloured point is the actual constrained solution, computed by the same three-line algebra as the worked example above, for whatever $a$, $b$ and $c$ the sliders currently hold. When the fence is active, two arrows appear at the solution: $-\\nabla f$, pointing back toward the unconstrained hilltop, and $\\lambda\\nabla g$, pointing directly out through the fence — the picture's version of "the two gradients are parallel". The readout reports whether the constraint is active, the multiplier's value, the solution's coordinates, and the product $\\lambda\\cdot g(x^\\star)$.</p>

<p><b>What to do with it.</b> There is no dragging here — three sliders set the fence's position $c$ and the unconstrained optimum's coordinates $(a,b)$ directly, which is arguably clearer, since you can place the true hilltop exactly where you want and watch the solution respond. Start from the default and slide $c$ upward, relaxing the budget. At some point the fence passes behind the unconstrained optimum entirely, the two arrows disappear, and the readout switches to "inactive (slack)" with $\\lambda=0$ — the constraint stopped mattering the instant it stopped binding. Now slide $c$ back down past that point and watch $\\lambda$ climb smoothly from zero: the further the true optimum is pushed outside the fence, the harder the fence has to push back.</p>

<p><b>The thing genuinely worth noticing.</b> Watch the readout's last line, $\\lambda\\cdot g(x^\\star)$, through the entire sweep. It reads zero in <i>every single configuration</i> — not approximately, exactly, to the precision shown — whether the constraint is slack ($\\lambda=0$) or active ($g=0$). That constant zero is complementary slackness made visible rather than asserted, and it is the exact mechanism by which a fitted SVM can discard every training point except the handful touching the margin: for every other point the constraint is slack, so its multiplier is zero, so it contributes nothing to the solution and could be deleted without moving the boundary by a hair (§2.6).</p>

${H.lab('kkt', 'Lagrange multipliers, made visible', 'Set the fence position and the true unconstrained optimum with the sliders. When the fence is slack the unconstrained optimum survives untouched and λ = 0; the moment the fence binds, the solution slides onto the boundary and the two arrows become anti-parallel. That anti-parallelism <i>is</i> the stationarity condition.')}

${H.more('Duality in the three places you will actually meet it', `
<ul>
<li><b>SVM (§2.6).</b> The dual is a quadratic program in $\\alpha$ with only inner products $x_i^\\top x_j$ in it. Replace those with $k(x_i,x_j)$ and you have the kernel trick — impossible in the primal, because the primal contains $w$, which may live in an infinite-dimensional space.</li>
<li><b>Ridge and lasso (§2.3).</b> The penalty form $\\min \\|Xw-y\\|^2 + \\lambda\\|w\\|$ and the constraint form $\\min\\|Xw-y\\|^2$ s.t. $\\|w\\|\\le t$ are Lagrangian duals of each other. Every $\\lambda$ corresponds to some $t$. The famous diamond-vs-circle picture is the constraint form; the thing you implement is the penalty form.</li>
<li><b>Constrained RL and safe fine-tuning (§4.12).</b> "Maximise reward subject to KL $\\le \\delta$ from the reference policy" is solved by dualising into "maximise reward $-\\beta\\,$KL". That is precisely where the $\\beta$ in PPO's penalty and in DPO's objective comes from — it is a Lagrange multiplier, and tuning it is choosing where on the trade-off curve to sit.</li>
</ul>`)}

${H.probe([
      ['What does convexity guarantee, and what does it not?', 'Guarantees: every stationary point is a global minimum, and duality gaps vanish under Slater. Does not guarantee: a unique minimiser (needs strict convexity), fast convergence (that is conditioning), or that the model is any good.'],
      ['Why is momentum $O(\\sqrt\\kappa)$ and plain GD $O(\\kappa)$?', 'GD’s per-coordinate error contracts by $1-\\eta\\lambda_i$ and one $\\eta$ must serve all $\\lambda_i$; momentum makes the recursion second order so both roots can be balanced at modulus $\\sqrt\\beta$.'],
      ['State complementary slackness and give a use.', '$\\lambda_i g_i(x^\\star)=0$: a constraint is either active or has zero multiplier. It identifies SVM support vectors and prunes inactive constraints from the solution.'],
      ['When would you use Newton’s method in 2026?', 'Small-to-medium $d$ with a smooth deterministic objective: GLM fitting via IRLS, calibration fits, hyperparameter inner loops. Never for a large network — the $d\\times d$ solve and the stochastic gradients both kill it.']
    ], 'Saying "neural networks are non-convex so we cannot say anything". You can say a great deal: about conditioning, about saddle points, about why normalisation and Adam help. Non-convexity is not an excuse to stop reasoning.')}`,
    labs: {
      convrace: function (host) {
        const st = Viz.controls(host, [
          { k: 'kappa', label: 'condition number κ', min: 1, max: 60, step: 1, value: 20, fmt: v => v + '×' },
          { k: 'lr', label: 'learning rate (× the stable max)', min: .05, max: 1.0, step: .01, value: .9, fmt: v => v.toFixed(2) },
          { k: 'beta', label: 'momentum β', min: 0, max: .98, step: .01, value: .9, fmt: v => v.toFixed(2) },
          { k: 'show', label: 'view', type: 'select', value: 'both', options: [{ v: 'both', t: 'trajectory + convergence' }, { v: 'traj', t: 'trajectory only' }, { v: 'conv', t: 'convergence only' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'gd', label: 'GD steps to 1e-4', cls: 'bad' },
          { k: 'mom', label: 'momentum steps', cls: 'good' },
          { k: 'nt', label: 'Newton steps', cls: 'key' },
          { k: 'pred', label: 'theory: κ vs √κ' }
        ]);

        function run(method) {
          const k = st.kappa;
          const lmax = k, lmin = 1;
          const eta = st.lr * 2 / lmax;
          let w = [1, 1], v = [0, 0];
          const path = [w.slice()], errs = [];
          const f = p => .5 * (lmin * p[0] * p[0] + lmax * p[1] * p[1]);
          for (let t = 0; t < 400; t++) {
            const g = [lmin * w[0], lmax * w[1]];
            if (method === 'gd') w = [w[0] - eta * g[0], w[1] - eta * g[1]];
            else if (method === 'mom') {
              v = [st.beta * v[0] + g[0], st.beta * v[1] + g[1]];
              w = [w[0] - eta * v[0], w[1] - eta * v[1]];
            } else { // Newton: H⁻¹g exactly solves a quadratic in one step
              w = [w[0] - g[0] / lmin, w[1] - g[1] / lmax];
            }
            if (!isFinite(w[0]) || !isFinite(w[1]) || Math.abs(w[0]) > 1e6) break;
            path.push(w.slice());
            errs.push(Math.max(1e-16, f(w)));
          }
          let hit = errs.findIndex(e => e < 1e-4);
          return { path: path, errs: errs, hit: hit < 0 ? null : hit + 1 };
        }

        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const gd = run('gd'), mo = run('mom'), nt = run('newton');
            const k = st.kappa;
            const both = st.show === 'both';
            const w1 = both ? w * .52 : w, x0 = 0;

            if (st.show !== 'conv') {
              const P = Viz.plot(ctx, w1, h, { xd: [-1.35, 1.35], yd: [-1.2, 1.2], pad: { l: 40, r: 10, t: 16, b: 36 } })
                .frame({ xlabel: 'flat direction (λ = 1)', ylabel: 'steep (λ = κ)' });
              P.clip(() => {
                P.contours((x, y) => .5 * (x * x + k * y * y), [.05, .2, .5, 1, 2], { color: T.faint, alpha: .5 });
                P.line(gd.path, { color: T.c2, width: 1.6 });
                P.dots(gd.path.slice(0, 40), { r: 2, color: T.c2, alpha: .9 });
                P.line(mo.path, { color: T.c3, width: 1.8 });
                P.dots(mo.path.slice(0, 40), { r: 2, color: T.c3, alpha: .9 });
                P.dots(nt.path.slice(0, 3), { r: 4.5, color: T.c1, stroke: true });
                P.dots([[0, 0]], { r: 4, color: T.text });
              });
            }
            if (st.show !== 'traj') {
              const ox = both ? w * .52 : 0, ww = both ? w * .48 : w;
              ctx.save(); ctx.translate(ox, 0);
              const P2 = Viz.plot(ctx, ww, h, {
                xd: [0, 120],
                yd: [-10, 1],
                pad: { l: 44, r: 12, t: 16, b: 36 }
              }).frame({ xlabel: 'iteration', ylabel: 'log₁₀ loss' });
              P2.clip(() => {
                P2.line(gd.errs.map((e, i) => [i, Math.log10(e)]), { color: T.c2, width: 2 });
                P2.line(mo.errs.map((e, i) => [i, Math.log10(e)]), { color: T.c3, width: 2 });
                P2.line(nt.errs.map((e, i) => [i, Math.log10(Math.max(1e-10, e))]), { color: T.c1, width: 2 });
                P2.hline(-4, { color: T.faint, dash: [4, 4], label: '1e-4' });
              });
              ctx.restore();
            }
            out({
              gd: gd.hit == null ? '>400 (or diverged)' : gd.hit,
              mom: mo.hit == null ? '>400' : mo.hit,
              nt: nt.hit == null ? '—' : nt.hit,
              pred: Math.round(k) + ' vs ' + Math.round(Math.sqrt(k))
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c2)', t: 'gradient descent' }, { c: 'var(--c3)', t: 'momentum' }, { c: 'var(--c1)', t: 'Newton' }
        ]);
        Viz.note(host, 'Newton lands in <b>one</b> step, because a quadratic is exactly what Newton’s method assumes — the price is forming and inverting the Hessian, which is $O(d^3)$ and hopeless at $d=10^9$. Now push the learning rate past 1.00 of the stable maximum and gradient descent diverges along the steep direction while still crawling along the flat one: the two failure modes that make one global learning rate so hard to choose.');
      },

      kkt: function (host) {
        const st = Viz.controls(host, [
          { k: 'c', label: 'constraint: x + y ≤ c', min: -1, max: 4, step: .05, value: 1.2, fmt: v => v.toFixed(2) },
          { k: 'cx', label: 'unconstrained optimum x*', min: -1, max: 3, step: .05, value: 2, fmt: v => v.toFixed(2) },
          { k: 'cy', label: 'unconstrained optimum y*', min: -1, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'active', label: 'constraint', cls: 'key' },
          { k: 'lam', label: 'multiplier λ' },
          { k: 'sol', label: 'solution' },
          { k: 'slack', label: 'λ · g(x*)' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const a = st.cx, b = st.cy, c = st.c;
            const f = (x, y) => (x - a) * (x - a) + (y - b) * (y - b);
            // projection onto x+y=c when the constraint binds
            const viol = a + b - c;
            const active = viol > 0;
            const sx = active ? a - viol / 2 : a, sy = active ? b - viol / 2 : b;
            const lam = active ? viol : 0;   // ∇f = -λ∇g ⇒ 2(x-a) = -λ, with x-a = -viol/2

            const P = Viz.plot(ctx, w, h, { xd: [-1.5, 4], yd: [-1.5, 3.6], pad: { l: 42, r: 14, t: 16, b: 36 } })
              .frame({ xlabel: 'x', ylabel: 'y' });
            P.clip(() => {
              // infeasible half-plane
              P.field((x, y) => (x + y > c ? 1 : 0), {
                step: 4, lo: 0, hi: 1,
                colors: t => t > .5 ? [180, 60, 50, 26] : [0, 0, 0, 0]
              });
              P.contours(f, [.15, .6, 1.5, 3, 5, 8], { color: T.faint, alpha: .55 });
              P.line([[-2, c + 2], [5, c - 5]], { color: T.c2, width: 2 });
              P.dots([[a, b]], { r: 4.5, color: T.faint, stroke: true });
              P.text(a, b, ' unconstrained', { dx: 8, dy: -10, color: T.faint, font: '10.5px ui-sans-serif' });
              P.dots([[sx, sy]], { r: 6, color: T.c1, stroke: true, strokeWidth: 2 });
              P.text(sx, sy, ' solution', { dx: 9, dy: 12, color: T.c1, font: '11px ui-sans-serif' });
              if (active) {
                // ∇f at the solution, and λ∇g pointing back
                const gfx = 2 * (sx - a), gfy = 2 * (sy - b);
                const sc = .6;
                P.arrow(sx, sy, sx + gfx * sc, sy + gfy * sc, { color: T.c3, width: 2 });
                P.text(sx + gfx * sc, sy + gfy * sc, ' −∇f', { dx: 6, color: T.c3, font: '11px ui-sans-serif' });
                P.arrow(sx, sy, sx + lam * sc, sy + lam * sc, { color: T.c2, width: 2 });
                P.text(sx + lam * sc, sy + lam * sc, ' λ∇g', { dx: 6, dy: -8, color: T.c2, font: '11px ui-sans-serif' });
              }
            });
            out({
              active: active ? 'ACTIVE (tight)' : 'inactive (slack)',
              lam: lam.toFixed(3),
              sol: '(' + sx.toFixed(2) + ', ' + sy.toFixed(2) + ')',
              slack: (lam * (sx + sy - c)).toFixed(6)
            });
          }
        });
        Viz.note(host, 'Watch the last readout: <b>λ · g(x*) is zero in every configuration</b>. When the constraint is slack, λ = 0; when it binds, g = 0. That product being identically zero <i>is</i> complementary slackness, and it is the reason a fitted SVM can throw away every training point except the handful on the margin.');
      }
    },
    quiz: [
      {
        q: 'The condition number of the Hessian is 10,000. Roughly how much does well-tuned momentum help versus plain gradient descent?',
        options: ['about 2×', 'about 100×', 'about 10,000×', 'not at all'],
        answer: 1,
        why: 'Gradient descent needs $O(\\kappa\\log(1/\\epsilon))$ iterations and momentum needs $O(\\sqrt\\kappa\\log(1/\\epsilon))$, so the improvement is the ratio $\\kappa/\\sqrt\\kappa=\\sqrt\\kappa$: at $\\kappa=10{,}000$ that is $\\sqrt{10{,}000}=100$, taking roughly 10,000 steps down to roughly 100. "About 10,000×" is the tempting wrong answer, because it treats momentum as if it eliminated the condition number entirely rather than merely taking its square root — that would be Newton\'s job, not momentum\'s, and Newton pays for it with an $O(d^3)$ solve momentum never touches. The general lesson: whenever a rate improves from $O(\\kappa)$ to $O(\\sqrt\\kappa)$, the speedup you get is $\\sqrt\\kappa$, not $\\kappa$ itself.'
      },
      {
        q: 'Complementary slackness says…',
        options: ['all constraints are active at the optimum', 'either a constraint is tight or its multiplier is zero', 'the dual equals the primal', 'the gradient is zero'],
        answer: 1,
        why: 'The condition is $\\lambda_i\\,g_i(x^\\star)=0$ for every inequality constraint: a product of two things is zero only when at least one of them is, so either the constraint is tight ($g_i=0$, you are pressed against that fence) or its multiplier is zero ($\\lambda_i=0$, that fence is not touching you and so cannot be pushing). "All constraints are active" is the tempting wrong answer because it sounds like the sharper, more specific claim, but it is not even true in general — most constraints in a real problem are slack, and complementary slackness is precisely the statement that slack ones contribute nothing. This is the fact that makes an SVM\'s solution sparse: every training point comfortably past the margin has a slack constraint and therefore $\\alpha_i=0$, and could be deleted from the dataset without moving the fitted boundary (§2.6).'
      },
      {
        q: 'Logistic regression on perfectly separable data has…',
        options: ['no solution because the loss is non-convex', 'a convex loss whose minimiser runs off to infinite weights', 'a unique bounded optimum', 'multiple disconnected minima'],
        answer: 1,
        why: 'The logistic loss is convex everywhere, so "non-convex" is simply false — that option is there to catch a memorised-but-wrong association between "something is going wrong" and "the problem must be non-convex". On separable data the likelihood keeps improving forever as $\\|w\\|\\to\\infty$ in the separating direction, because pushing the decision boundary\'s confidence toward certainty always lowers the loss a little further; a convex function with no minimiser inside any bounded region is called non-coercive, and that is the correct diagnosis. Any amount of L2 regularisation restores a finite optimum by adding a term that grows with $\\|w\\|$, which is the practical reason scikit-learn regularises logistic regression by default rather than leaving it to the user to discover this failure mode.'
      },
      {
        q: 'Why is Newton’s method impractical for deep networks?',
        options: ['it needs convexity', 'the $d\\times d$ Hessian solve is $O(d^3)$ and gradients are stochastic', 'it converges too slowly', 'it cannot handle constraints'],
        answer: 1,
        why: 'Two independent problems, either one fatal alone. Forming and solving against the $d\\times d$ Hessian costs $O(d^3)$ time and $O(d^2)$ memory, and for a network with a billion parameters that is not merely slow, it is physically impossible on any hardware that exists. Separately, deep learning estimates gradients from mini-batches, and Newton\'s step $-H^{-1}g$ trusts the local quadratic model far more precisely than a noisy stochastic gradient deserves. "It converges too slowly" is backwards and worth noticing as backwards: Newton converges in <i>fewer iterations</i> than any first-order method, which is exactly why quasi-Newton methods and diagonal preconditioners like Adam exist — they are attempts to buy back some of that iteration count cheaply, at $O(d)$ instead of $O(d^3)$.'
      }
    ],
    cards: [
      { q: 'Convexity, in one sentence', a: 'Every local minimum is global, and the Hessian is PSD everywhere. It does not imply uniqueness or speed.' },
      { q: 'Convergence rates by κ', a: 'GD $O(\\kappa)$, momentum/Nesterov $O(\\sqrt\\kappa)$, Newton $O(\\log\\log)$ at $O(d^3)$ per step.' },
      { q: 'The four KKT conditions', a: 'Stationarity, primal feasibility, dual feasibility ($\\lambda\\ge0$), complementary slackness ($\\lambda_i g_i = 0$).' },
      { q: 'Why the SVM has a kernel trick and the primal does not', a: 'The dual contains only inner products $x_i^\\top x_j$, which can be replaced by $k(x_i,x_j)$; the primal contains $w$ itself.' },
      { q: 'Where the β in PPO/DPO comes from', a: 'A Lagrange multiplier: "maximise reward s.t. KL ≤ δ" dualised into "maximise reward − β·KL".' }
    ]
  });

  /* ------------------------------------------------------------------ 1.13 */
  ML.section({
    id: 'sampling', track: 'foundations', num: '1.13', level: 2,
    title: 'Monte Carlo, the bootstrap, and simulation as proof',
    lede: 'When the integral is hard, sample. When the sampling distribution is unknown, resample. Between them these two ideas replace most of the analytic statistics you were told you needed — and they are how you check the analytic answer when you do have one.',
    prereq: ['concentration'],
    related: ['bayesian-inference', 'intervals', 'experimentation'],
    html: `
${H.tldr([
      'Monte Carlo error falls as $\\sigma/\\sqrt{n}$ — <b>independent of dimension</b>. That is why sampling beats quadrature in high dimensions, and why 100× more samples buys only 10× more accuracy.',
      'The bootstrap replaces "what is the sampling distribution of my statistic?" with "resample the data and look". It works for medians, AUCs, ratios and pipeline outputs where no formula exists.',
      'Simulate before you derive. If your closed-form interval and a simulation disagree, the simulation is usually right — and always cheaper to trust.'
    ])}

<p>Suppose you want the average value of some quantity $f(X)$ where $X$ follows a distribution $p$ too complicated to integrate against by hand — a posterior over a model's parameters, say, which is exactly the problem §1.14 spends an entire section wrestling with. In principle the answer is one line, $\\mathbb{E}[f(X)] = \\int f(x)\\,p(x)\\,dx$, and in practice that integral has no closed form for almost any $p$ worth having: no antiderivative exists, and a fine grid over the space is out of the question for a reason this section makes precise below.</p>

<p>But there is something you very often <i>can</i> do even when you cannot integrate: draw samples from $p$. A posterior you cannot write down algebraically, you can frequently still simulate from. And §1.4.1 already proved the fact this whole section leans on: an average of independent draws concentrates around its true expectation at a rate of $1/\\sqrt n$, whatever is being averaged. Put those two facts together and an integral you cannot do becomes an average you can — draw enough samples, average $f$ over them, and the law of large numbers does the integration for you.</p>

<h2><span class="sn">1.13.1</span> Monte Carlo integration</h2>
<p>To compute $I = \\mathbb{E}_{x\\sim p}[f(x)] = \\int f(x)p(x)\\,dx$, draw $x_1,\\dots,x_n \\sim p$ independently and average:</p>
$$\\hat I_n = \\frac1n\\sum_{i=1}^n f(x_i)$$
${H.deriv('why the average is unbiased, in one line', [
      ['$\\mathbb{E}[\\hat I_n] = \\mathbb{E}\\Big[\\frac1n\\sum_{i=1}^n f(x_i)\\Big]$', 'Take the expectation of the estimator exactly as defined above.'],
      ['$= \\frac1n\\sum_{i=1}^n \\mathbb{E}[f(x_i)]$', 'Expectation is linear, so it distributes over the sum. This step uses nothing about independence — it works even if the $x_i$ were correlated.'],
      ['$= \\frac1n\\sum_{i=1}^n I = I$', 'Every $x_i$ is drawn from the same $p$, so every term in the sum equals $I$ by the definition of $I$ at the top. This holds for any $n\\ge1$: a single sample is already an unbiased, if very noisy, estimate.']
    ], 'Unbiasedness describes the average of infinitely many repeated experiments, not any one run of this one. §1.4.1 supplies the missing half — how tightly one particular $\\hat I_n$ actually clusters around $I$ — and the answer is $\\mathrm{sd}(\\hat I_n)=\\sigma_f/\\sqrt n$, by exactly the argument that governs any sample mean.')}

<p>That $\\sqrt n$ in the denominator is the entire reason Monte Carlo earns a name of its own rather than being a curiosity. Compare it with the obvious alternative: lay a grid over the space and evaluate $f$ at every grid point. Resolving each of $d$ input dimensions to a modest 10 levels needs $10^d$ grid points. At $d=1$ that is 10 points — trivial. At $d=6$, which is nothing exotic for a feature set, it is already a million. At $d=20$, smaller than a single layer of a tiny neural network, it is $10^{20}$ — a number with no physical intuition attached to it at all. The grid's accuracy improves with more points, but the number of points needed for a <i>fixed</i> resolution explodes exponentially in $d$. Monte Carlo's error, $\\sigma_f/\\sqrt n$, has no $d$ in it whatsoever: the same $n$ samples buy the same accuracy whether $f$ has 3 inputs or 3,000. Past about $d=4$, quadrature has already lost, and it never recovers.</p>
${H.key('$\\sqrt{n}$ is the tax on everything: every extra digit of accuracy costs a hundredfold increase in samples. Variance reduction is therefore not an optimisation, it is the only lever you have.')}

<p><b>What you are looking at.</b> One genuinely awkward integral, $\\int_0^1 e^x\\,dx = e-1 \\approx 1.71828$, estimated four different ways at once: plain sampling, antithetic pairs, stratification and importance sampling. Each dot is the absolute error of one independent repeat of an estimate at a given sample size; the horizontal bar through each column of dots is the median error across repeats for that method. The dashed line is the theoretical $\\sigma/\\sqrt n$ that plain Monte Carlo should trace out.</p>

<p><b>What to do with it.</b> Drag the sample-size slider from $10^1$ toward $10^4$ and watch every method's dots march downward along roughly parallel lines — a straight line on these log-scaled axes is exactly what a power law like $n^{-1/2}$ produces. Increase the repeats slider and each column's dot swarm fills in, making the median line more trustworthy without moving where it sits.</p>

<p><b>The thing genuinely worth noticing.</b> Stratification wins here, by a wide margin, because a smooth one-dimensional integrand lets forcing the samples to cover $[0,1]$ evenly remove almost all of the randomness plain sampling leaves on the table — its error can fall faster than $n^{-1/2}$ altogether. That advantage is real and it is also fragile: it depends on smoothness and on living in one dimension, and it evaporates as $d$ grows, because there is no cheap way to stratify a high-dimensional cube into evenly-covering cells. Plain Monte Carlo, the worst performer in this lab, is the only one of the four whose rate does not care how many dimensions you have — which is exactly why it, not the smarter-looking alternatives, is the method you can always fall back on.</p>

${H.lab('mc', 'Monte Carlo convergence, and the three ways to cheat it', 'Estimate the same integral four ways: plain sampling, antithetic pairs, stratification and importance sampling. All four are unbiased; the error bands are wildly different. The dashed line is the theoretical $\\sigma/\\sqrt n$.')}

${H.table(['Variance reduction', 'Idea', 'Typical gain', 'Where you meet it'], [
      ['Antithetic variates', 'pair each $u$ with $1-u$; the errors cancel if $f$ is monotone', '2–10×', 'pricing, simulation studies'],
      ['Stratification', 'force the samples to cover the space evenly', '2–20×', 'stratified train/test splits are the same idea (§2.14)'],
      ['Importance sampling', 'sample from $q$, reweight by $p/q$; concentrate effort where $f$ is large', '10–1000×, or catastrophic', 'off-policy RL (§6.1), rare-event estimation'],
      ['Control variates', 'subtract a correlated quantity whose mean you know', '2–50×', 'CUPED in A/B testing is exactly this (§2.25)'],
      ['Common random numbers', 'reuse the same seed across arms', 'large', 'comparing two policies or two model versions']
    ])}

<p>Importance sampling deserves a moment longer than the table gives it, because unlike the other three rows it lets you sample from a distribution <i>other than</i> the one in the problem — which sounds like it should introduce bias, and does not.</p>
${H.deriv('why reweighting by p/q stays unbiased', [
      ['$I = \\int f(x)\\,p(x)\\,dx$', 'The quantity wanted, unchanged from the top of this section.'],
      ['$= \\int f(x)\\,\\dfrac{p(x)}{q(x)}\\,q(x)\\,dx$', 'Multiply and divide by $q(x)$, valid wherever $q(x)>0$ everywhere $f(x)p(x)\\ne0$. Nothing has changed numerically — this is the identity trick, the same move §1.10.6 uses to keep log-sum-exp exact.'],
      ['$= \\mathbb{E}_{x\\sim q}\\Big[f(x)\\,\\dfrac{p(x)}{q(x)}\\Big]$', 'Read the rewritten integral as an expectation under $q$ instead of $p$: $q$ generates the samples, and the weight $w(x)=p(x)/q(x)$ corrects for having sampled from the wrong distribution.'],
      ['$\\hat I_n = \\dfrac1n\\sum_i f(x_i)\\,w(x_i), \\;\\; x_i\\sim q$', 'Estimate the expectation the ordinary way, under $q$. By the identical one-line argument used for plain Monte Carlo above, this is unbiased for $I$ for <i>any</i> valid $q$ — the choice of $q$ changes only the variance, never the mean.']
    ], 'The freedom to choose $q$ is the entire point: put more samples where $f(x)p(x)$ is large and the estimate can have dramatically lower variance than sampling from $p$ directly. This is how rare-event probabilities get estimated at all, since sampling straight from $p$ would almost never land in the rare region to begin with. Chosen badly, that same freedom is exactly what the pitfall below describes.')}
${H.pitfall('Importance sampling has an infinite-variance failure mode: if $q$ has lighter tails than $p\\,|f|$, a single sample with an enormous weight dominates the estimate and the sample standard error <i>understates</i> the true error without bound. Always report the effective sample size $\\text{ESS} = (\\sum w_i)^2/\\sum w_i^2$; if it has collapsed to a handful, your estimate is one number wearing a crowd’s clothing.')}

<h2><span class="sn">1.13.2</span> The bootstrap</h2>
<p>Monte Carlo assumed you could already sample from $p$. Now assume something both harder and far more common: you have exactly one sample of real data, size $n$, and a statistic $T$ computed from it — a median, an AUC, the output of an entire fitted pipeline. You want to know how much $T$ would vary had you collected a different sample of the same size, so you can put a confidence interval around it. For a mean, §1.4 handed you a formula. For almost anything else, no formula exists, and deriving one by hand, statistic by statistic, is not realistic.</p>
${H.analogy(`<p>The name is a joke with real content packed into it. "Pulling yourself up by your own bootstraps" is proverbially impossible, and estimating the uncertainty of a statistic using nothing but the single sample that produced it sounds like the same kind of impossible. Bradley Efron's 1979 resolution was to notice that it is not. Your one sample <i>is</i> your best available estimate of the entire population's distribution — for every value you have not seen, you have no evidence one way or the other — so treat the sample itself as a stand-in population, and draw new samples from <i>that</i>, with replacement, as many times as you like.</p>`)}
${H.steps([
      'Draw $n$ observations from your data <b>with replacement</b>. This is one bootstrap sample; about 63.2% of the original rows appear at least once ($1-e^{-1}$), which is also where the out-of-bag estimate in random forests comes from (§2.7).',
      'Compute $T^{*}$ on it.',
      'Repeat $B$ times, typically 1,000–10,000. The spread of $\\{T^*_b\\}$ estimates the sampling variability of $T$; the 2.5th and 97.5th percentiles give a 95% interval.'
    ])}

<p><b>What you are looking at.</b> A histogram of $T^*$ computed across every one of the bootstrap resamples — the bootstrap distribution itself, drawn directly rather than approximated by a formula. The solid vertical line marks the point estimate $T$ computed once on the real data; the two dashed lines mark the 2.5th and 97.5th percentiles of the histogram, which together are the percentile confidence interval, and the bars between the dashed lines are shaded differently from the tails so the interval reads as a region rather than two isolated marks.</p>

<p><b>What to do with it.</b> Choose the median or the 90th percentile from the statistic selector and watch a histogram build that has no reason to be symmetric, unlike the tidy bell curve a formula would silently have assumed. Raise the population-skew slider and watch that asymmetry appear directly in the sampling distribution. Then compare the readout's percentile interval against the naive Gaussian interval computed from the same bootstrap standard error: on a skewed statistic the two visibly disagree, and the percentile interval — built from the actual shape of the resampled distribution rather than an assumed bell curve — is the one telling the truth.</p>

<p><b>The thing genuinely worth noticing.</b> Select <b>maximum</b> from the statistic dropdown. The bootstrap distribution collapses onto a handful of discrete spikes, because a resample can never produce a value larger than the single largest value already sitting in your data — the observed maximum is a hard ceiling on every possible bootstrap maximum. The resulting interval is a narrow band pinned suspiciously close to that observed value, and it is wrong: the bootstrap has failed, confidently, with nothing in the output flagging that anything went wrong. It is the sharpest illustration in this section of a fact worth carrying everywhere else — a method unbiased in its ordinary use can still be the wrong tool for a specific statistic, and the failure does not announce itself.</p>

${H.lab('boot', 'Bootstrap a statistic that has no formula', 'The median, the trimmed mean, the 90th percentile, the correlation, the AUC. The histogram is the bootstrap distribution; the shaded band is the percentile interval; the dashed line is the Gaussian interval you would have written down if the statistic had been a mean.')}

${H.table(['Bootstrap flavour', 'When', 'Caveat'], [
      ['Percentile', 'default; simple and adequate', 'slightly under-covers for skewed statistics'],
      ['BCa (bias-corrected, accelerated)', 'skewed statistics, small $n$', 'more computation, better coverage'],
      ['Block bootstrap', '<b>time series</b> — resample contiguous blocks', 'block length must exceed the autocorrelation range'],
      ['Cluster bootstrap', 'grouped data (users, hospitals, sessions)', 'resample <b>clusters</b>, not rows, or you understate variance badly'],
      ['Bayesian bootstrap', 'want a posterior-flavoured version', 'Dirichlet weights instead of counts']
    ])}
${H.pitfall('The bootstrap fails for extremes, exactly as the lab above shows. It also fails silently when observations are dependent and you resample rows — the single most common bootstrap error in industry, and the reason per-user metrics must be bootstrapped by user, not by impression.')}

<h2><span class="sn">1.13.3</span> Simulate first</h2>
${H.intuition(`<p>Any statistical claim you are about to make can be tested in twenty lines: generate data where you <i>know</i> the truth, run your procedure a thousand times, and count how often it is right. Does your 95% interval cover 95% of the time? Does your p-value come out uniform under the null? Does your early-stopping rule inflate the false-positive rate (§2.25 says it does)? This is not a substitute for theory; it is how you find out that you applied the theory to the wrong problem — and it is the same "sample many times, look at the empirical spread" idea this whole section has been building, now aimed at your <i>method</i> instead of at an unknown quantity.</p>`)}
${H.code(`# Does my confidence interval actually cover 95% of the time?
import numpy as np
rng, cover = np.random.default_rng(0), 0
for _ in range(2000):
    x  = rng.exponential(scale=2.0, size=30)       # skewed, not normal
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(30)
    lo, hi = m - 1.96*se, m + 1.96*se              # the interval we teach
    cover += (lo <= 2.0 <= hi)
print(cover / 2000)     # ≈ 0.91, not 0.95 — the CLT has not arrived at n=30`)}
${H.flag('That 0.91 is the honest answer for skewed data at $n=30$, and it is why the bootstrap or a log transform is worth the trouble. The textbook "$n>30$ and the CLT applies" rule is a rule of thumb about symmetric distributions, not a theorem.')}

${H.probe([
      ['Why does Monte Carlo beat numerical integration in high dimensions?', 'Its error is $O(n^{-1/2})$ regardless of $d$; a product grid needs $m^d$ points. Past about four dimensions there is no contest.'],
      ['You bootstrap a click-through rate from 100,000 rows and get an implausibly tight interval. What is wrong?', 'Almost certainly the rows are not independent — many rows per user. Bootstrap at the user level; the effective sample size is the number of users, not impressions.'],
      ['How many bootstrap replicates?', '1,000 for a standard error, 10,000 for a percentile interval you intend to quote. The Monte Carlo error of the interval itself falls as $1/\\sqrt B$ and is free to shrink.'],
      ['What is the 0.632 in the .632 bootstrap?', 'The probability a given row appears in a bootstrap sample: $1-(1-1/n)^n \\to 1-e^{-1}=0.632$. The complement is the out-of-bag set.']
    ])}`,
    labs: {
      mc: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'samples (log₁₀)', min: 1, max: 4, step: .1, value: 3, fmt: v => Math.round(Math.pow(10, v)).toLocaleString() },
          { k: 'reps', label: 'independent repeats', min: 5, max: 60, step: 1, value: 25, fmt: v => v },
          { k: 'seed', label: 'seed', min: 1, max: 40, step: 1, value: 7, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'truth', label: 'true value', cls: 'key' },
          { k: 'plain', label: 'plain MC error' },
          { k: 'anti', label: 'antithetic' },
          { k: 'strat', label: 'stratified' },
          { k: 'imp', label: 'importance' }
        ]);
        /* I = ∫₀¹ e^x dx = e − 1, with an importance density q ∝ 1 + x */
        const TRUTH = Math.E - 1;
        const f = x => Math.exp(x);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const N = Math.max(4, Math.round(Math.pow(10, st.n)));
            const R = Num.rng(st.seed);
            const errs = { plain: [], anti: [], strat: [], imp: [] };
            for (let r = 0; r < st.reps; r++) {
              let sp = 0, sa = 0, ss = 0, si = 0;
              const half = Math.max(1, Math.floor(N / 2));
              for (let i = 0; i < N; i++) sp += f(R());
              for (let i = 0; i < half; i++) { const u = R(); sa += f(u) + f(1 - u); }
              for (let i = 0; i < N; i++) { const u = (i + R()) / N; ss += f(u); }
              for (let i = 0; i < N; i++) {
                // q(x) = (1+x)/1.5 on [0,1]; inverse-CDF sample
                const u = R();
                const x = Math.sqrt(1 + 3 * u) - 1;
                const q = (1 + x) / 1.5;
                si += f(x) / q;
              }
              errs.plain.push(Math.abs(sp / N - TRUTH));
              errs.anti.push(Math.abs(sa / (2 * half) - TRUTH));
              errs.strat.push(Math.abs(ss / N - TRUTH));
              errs.imp.push(Math.abs(si / N - TRUTH));
            }
            const med = a => Num.quantile(a.slice().sort((x, y) => x - y), .5);
            const P = Viz.plot(ctx, w, h, { xd: [-.6, 3.6], yd: [-7, -0.4], pad: { l: 50, r: 14, t: 16, b: 40 } })
              .frame({ xticks: [0, 1, 2, 3], xfmt: v => '10^' + v, xlabel: 'absolute error (repeats jittered)', ylabel: 'log₁₀ |error|' });
            const cols = [T.c1, T.c3, T.c4, T.c5];
            const keys = ['plain', 'anti', 'strat', 'imp'];
            P.clip(() => {
              keys.forEach((k, j) => {
                errs[k].forEach((e, i) => {
                  P.dots([[j + (i / st.reps - .5) * .55, Math.log10(Math.max(1e-9, e))]], { r: 2.6, color: cols[j], alpha: .75 });
                });
                const m = Math.log10(Math.max(1e-9, med(errs[k])));
                ctx.strokeStyle = cols[j]; ctx.lineWidth = 2.4;
                ctx.beginPath(); ctx.moveTo(P.x(j - .32), P.y(m)); ctx.lineTo(P.x(j + .32), P.y(m)); ctx.stroke();
              });
              const sig = Math.sqrt(Num.variance(Array.from({ length: 400 }, (_, i) => f((i + .5) / 400))));
              P.hline(Math.log10(sig / Math.sqrt(N)), { color: T.faint, dash: [5, 4], label: 'σ/√n' });
            });
            ctx.fillStyle = T.muted; ctx.font = '10.5px ui-sans-serif';
            ctx.textAlign = 'center';
            ['plain', 'antithetic', 'stratified', 'importance'].forEach((lbl, j) => ctx.fillText(lbl, P.x(j), h - 22));
            out({
              truth: TRUTH.toFixed(6),
              plain: med(errs.plain).toExponential(2),
              anti: med(errs.anti).toExponential(2),
              strat: med(errs.strat).toExponential(2),
              imp: med(errs.imp).toExponential(2)
            });
          }
        });
        Viz.note(host, 'Stratification wins here by a wide margin because the integrand is smooth and one-dimensional — its error falls like $n^{-1}$ or better rather than $n^{-1/2}$. That advantage evaporates in high dimensions, which is precisely why plain Monte Carlo survives: it is the only one of the four whose rate does not care how many dimensions you have.');
      },

      boot: function (host) {
        const st = Viz.controls(host, [
          { k: 'stat', label: 'statistic', type: 'select', value: 'median', options: [
            { v: 'median', t: 'median' }, { v: 'mean', t: 'mean' }, { v: 'p90', t: '90th percentile' },
            { v: 'trim', t: '10% trimmed mean' }, { v: 'max', t: 'maximum (watch it fail)' }
          ] },
          { k: 'n', label: 'sample size n', min: 15, max: 400, step: 5, value: 60, fmt: v => v },
          { k: 'B', label: 'bootstrap replicates B', min: 100, max: 3000, step: 100, value: 1000, fmt: v => v.toLocaleString() },
          { k: 'skew', label: 'population skew', min: 0, max: 1, step: .05, value: .8, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'est', label: 'point estimate', cls: 'key' },
          { k: 'ci', label: '95% percentile CI' },
          { k: 'se', label: 'bootstrap SE' },
          { k: 'gauss', label: 'naive ±1.96·SE' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(11);
            const data = Array.from({ length: st.n }, () =>
              st.skew > 0 ? R.gamma(1 + 6 * (1 - st.skew)) * (2 + 6 * st.skew) : R.normal(10, 3));
            const stats = {
              median: a => Num.quantile(a.slice().sort((x, y) => x - y), .5),
              mean: a => Num.mean(a),
              p90: a => Num.quantile(a.slice().sort((x, y) => x - y), .9),
              trim: a => { const s = a.slice().sort((x, y) => x - y); const k = Math.floor(a.length * .1); return Num.mean(s.slice(k, s.length - k)); },
              max: a => Math.max.apply(null, a)
            };
            const fn = stats[st.stat];
            const bs = Num.bootstrap(data, fn, st.B, 5);
            const point = fn(data);
            const hist = Num.hist(bs.samples, 34);
            const P = Viz.plot(ctx, w, h, {
              xd: [hist.lo, hist.hi],
              yd: [0, Math.max.apply(null, hist.bins) * 1.18],
              pad: { l: 46, r: 14, t: 16, b: 40 }
            }).frame({ xlabel: 'value of the statistic across ' + st.B.toLocaleString() + ' resamples', ylabel: 'count' });
            P.clip(() => {
              P.bars(hist.bins, {
                gap: .1,
                color: (v, i) => (hist.centers[i] >= bs.lo && hist.centers[i] <= bs.hi) ? T.c1 : T.line
              });
              P.vline(point, { color: T.c2, width: 2, dash: false, label: 'estimate' });
              P.vline(bs.lo, { color: T.c1, dash: [4, 3] });
              P.vline(bs.hi, { color: T.c1, dash: [4, 3] });
            });
            out({
              est: point.toFixed(3),
              ci: '[' + bs.lo.toFixed(2) + ', ' + bs.hi.toFixed(2) + ']',
              se: bs.se.toFixed(3),
              gauss: '[' + (point - 1.96 * bs.se).toFixed(2) + ', ' + (point + 1.96 * bs.se).toFixed(2) + ']'
            });
          }
        });
        Viz.note(host, 'Choose <b>maximum</b>: the bootstrap distribution piles up on a few discrete values and the upper tail is truncated at the observed max — the distribution is wrong and the interval is nonsense. That is the bootstrap’s known failure mode for extremes. Then compare the percentile interval with the naive Gaussian one on a skewed statistic: the percentile interval is asymmetric because the sampling distribution is, and the symmetric one silently mis-covers.');
      }
    },
    quiz: [
      {
        q: 'Monte Carlo error scales as…',
        options: ['$1/n$', '$1/\\sqrt n$, independent of dimension', '$1/n^{d}$', '$\\log n / n$'],
        answer: 1,
        why: 'The estimator is an ordinary sample mean, so its standard deviation falls as $\\sigma_f/\\sqrt n$ by the same concentration argument §1.4.1 proved for any average of independent draws — and crucially, that formula never mentions the dimension $d$ of $x$. "$1/n^{d}$" is the tempting wrong answer because it correctly describes how a <i>quadrature grid</i> behaves as $d$ grows, and mixing the two up is the single most common confusion in this material: quadrature degrades with dimension, Monte Carlo does not, and that asymmetry is the entire reason to prefer sampling once $d$ is more than about four. The $1/\\sqrt n$ rate is simultaneously the blessing (no curse of dimensionality) and the curse (a full extra digit of accuracy costs a hundredfold more samples).'
      },
      {
        q: 'Roughly what fraction of the original rows appears in one bootstrap sample?',
        options: ['50%', '63.2%', '95%', '100%'],
        answer: 1,
        why: 'The probability a specific row is never drawn in $n$ draws with replacement from $n$ rows is $(1-1/n)^n$, which converges to $e^{-1}\\approx0.368$ as $n$ grows; the complement, the probability it appears at least once, is $1-e^{-1}\\approx0.632$. "50%" is the tempting answer if you picture drawing without replacement or reason about it as a coin flip per row, but resampling is with replacement and some rows are drawn two or three times while others are skipped entirely — that unevenness is exactly what produces the 63.2/36.8 split rather than an even 50/50. The excluded 36.8% is the out-of-bag set random forests use for a built-in validation estimate (§2.7).'
      },
      {
        q: 'You have 5 million impressions from 40,000 users. To bootstrap CTR you should resample…',
        options: ['impressions', 'users', 'either — it makes no difference', 'days'],
        answer: 1,
        why: 'Impressions from the same user are correlated — one user\'s clicking habits do not reset between their impressions — so the true amount of independent information is closer to 40,000 users than to 5 million impressions. Resampling impressions treats 5 million as the effective sample size, and the resulting interval is far too tight: it reports confidence the data does not actually contain. "Either — it makes no difference" is the dangerous wrong answer precisely because the bootstrap will run without complaint either way and hand back a number that looks perfectly reasonable; nothing about the procedure warns you that you resampled at the wrong level. Resample whole users, keeping each user\'s full block of impressions together, and the interval widens to reflect the real 40,000-unit sample size.'
      },
      {
        q: 'Importance sampling with a proposal that has lighter tails than the target can…',
        options: ['only be slow', 'have infinite variance, with a sample SE that hides it', 'introduce bias', 'never converge to the right answer'],
        answer: 1,
        why: 'The estimator $\\hat I_n=\\frac1n\\sum f(x_i)p(x_i)/q(x_i)$ stays unbiased for any valid $q$, by the same identity-multiplication argument that derives it — so "introduce bias" is wrong on the mechanics, even though it is the intuitively scary-sounding option. The real danger is variance: if $q$\'s tails are lighter than $p\\,|f|$\'s, an occasional sample lands where $q$ badly under-covers $p$, producing a weight $p(x)/q(x)$ that is enormous, and a single such sample can dominate the whole average. Worse, the ordinary sample standard error is computed from the same lopsided weights and understates how bad the situation is — the estimate looks confidently precise and is not. This is why the effective sample size $\\mathrm{ESS}=(\\sum w_i)^2/\\sum w_i^2$ is reported alongside the estimate rather than trusting the standard error alone.'
      }
    ],
    cards: [
      { q: 'Monte Carlo error rate', a: '$\\sigma/\\sqrt n$, independent of dimension. Ten times more accuracy costs a hundred times more samples.' },
      { q: 'The bootstrap in one line', a: 'Resample your data with replacement $B$ times, recompute the statistic, and read the spread as its sampling distribution.' },
      { q: 'Where the bootstrap fails', a: 'Extremes (max, min, high quantiles), dependent data resampled at the wrong level, and very small $n$.' },
      { q: 'Effective sample size for weights', a: '$\\mathrm{ESS} = (\\sum w_i)^2 / \\sum w_i^2$. A collapsed ESS means one sample is doing all the work.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.14 */
  ML.section({
    id: 'bayesian-inference', track: 'foundations', num: '1.14', level: 3,
    title: 'Bayesian inference in practice: conjugacy, MCMC, variational',
    lede: 'Bayes’ rule (§1.1) is a line of algebra. Doing Bayesian inference on a real model means computing a posterior you cannot write down — and there are exactly three strategies for that, each with a distinct failure mode.',
    prereq: ['bayes', 'mle-map'],
    related: ['gp-bayesopt', 'sampling', 'vae-gan'],
    html: `
${H.tldr([
      'Three routes to a posterior: <b>conjugacy</b> (exact, rare, instant), <b>MCMC</b> (asymptotically exact, slow, diagnosable), <b>variational</b> (fast, biased, scales — and is what a VAE does, §6.3).',
      'The obstacle is always the same: the evidence $p(D) = \\int p(D\\mid\\theta)p(\\theta)d\\theta$. Conjugacy makes it analytic, MCMC avoids it, VI replaces it with a bound.',
      'A Bayesian answer is a <i>distribution</i>. If you are going to collapse it to a point anyway, you have paid for something you did not use — except that the width is usually the part you needed (§2.12, §2.21).'
    ])}

<p>§1.1 built Bayes' rule from ordinary counting, and §1.5 showed that maximising the resulting posterior gives you MAP: a single best point $\\hat\\theta$, obtained by turning inference into an optimisation problem. That point is often exactly what you came for, but it quietly throws away most of what Bayes' rule actually promised. The full statement of Bayesian inference is not "find the best $\\theta$"; it is "tell me the whole distribution of how plausible every $\\theta$ is, given the data" — a shape, not a point. Getting the shape rather than its peak means actually computing $p(\\theta\\mid D)$, and for almost any model with more than a handful of parameters, that computation runs straight into an integral with no closed form. This section is the three ways of getting the shape anyway.</p>

<h2><span class="sn">1.14.1</span> The obstacle, stated once</h2>
$$p(\\theta\\mid D) = \\frac{p(D\\mid\\theta)\\,p(\\theta)}{p(D)},\\qquad p(D)=\\int p(D\\mid\\theta)p(\\theta)\\,d\\theta$$
<p>The numerator is easy: it is a likelihood times a prior, both of which you wrote down yourself, and evaluating it at any specific $\\theta$ is a single arithmetic computation. The denominator, the <b>evidence</b> $p(D)$, is an integral over the <i>entire</i> parameter space — every possible $\\theta$, weighted by how well it would have explained the data. §1.13.1 already showed what a grid over a $d$-dimensional space costs: roughly $10^d$ points for a modest resolution, a number that is merely large at $d=6$ and physically absurd by $d=20$. A model with a few thousand parameters, which is a small one by modern standards, puts that integral entirely out of reach of any method that tries to evaluate it directly. Three strategies get you the posterior without ever forming that integral: dodge it algebraically, avoid needing it at all, or replace it with something you can compute instead.</p>

<h2><span class="sn">1.14.2</span> Conjugacy: when the algebra closes</h2>
<p>The cheapest escape is for the integral to never come up in the first place. A prior is <b>conjugate</b> to a likelihood if the posterior lands back in the same family the prior came from — updating is then arithmetic on the family's parameters, fast enough to do in your head.</p>
${H.table(['Likelihood', 'Conjugate prior', 'Posterior', 'Read it as'], [
      ['Bernoulli / Binomial', 'Beta$(\\alpha,\\beta)$', 'Beta$(\\alpha+s,\\ \\beta+n-s)$', 'α−1 pseudo-successes, β−1 pseudo-failures'],
      ['Poisson', 'Gamma$(\\alpha,\\beta)$', 'Gamma$(\\alpha+\\sum x_i,\\ \\beta+n)$', 'α pseudo-events in β pseudo-periods'],
      ['Normal (known σ²)', 'Normal$(\\mu_0,\\tau^2)$', 'Normal, precision-weighted mean', 'precisions add; means average by precision'],
      ['Multinomial', 'Dirichlet$(\\alpha)$', 'Dirichlet$(\\alpha + \\text{counts})$', 'the smoothing in Naive Bayes (§2.5)'],
      ['Normal (unknown σ²)', 'Normal-Inverse-Gamma', 'Normal-Inverse-Gamma', 'the origin of the $t$ distribution']
    ])}
${H.worked('the sentence to have ready', `
<p>You observe 7 conversions in 10 visits. With a uniform prior Beta(1,1), the posterior is Beta(8,4): mean $8/12 = 0.667$, mode $7/10 = 0.700$ (that is the MLE), and a 95% credible interval of roughly $[0.39, 0.89]$.</p>
<p><b>The posterior mean is the MLE shrunk toward the prior mean</b>, and the amount of shrinkage is the prior's pseudo-count divided by the total count. With Beta(1,1) that is 2 pseudo-observations against 10 real ones — noticeable. At $n=1000$ it is invisible. That single sentence — <i>the prior is worth $\\alpha+\\beta$ observations and its influence decays as $1/n$</i> — answers most "how much does the prior matter" questions, and it is the distributional version of the point §1.5 already made: a prior is a regulariser, and here you can watch exactly how much regularisation it applies, and watch that amount decay as data accumulates.</p>`)}

<p><b>What you are looking at.</b> Two curves over the same axis, the possible conversion rate $\\theta$ from 0 to 1: a dashed line for the prior density and a solid filled curve for the posterior density, both Beta distributions. A solid vertical line marks the MLE — the raw observed rate, successes over trials — and two dotted vertical lines bracket the 95% credible interval, the region containing the middle 95% of the posterior's area.</p>

<p><b>What to do with it.</b> Start from the flat Beta(1,1) prior and click <b>+ success</b> twice in a row. The MLE readout jumps straight to 100%, but the posterior mean reads 75% with a credible interval running from about 0.19 to 0.99 — both are correct answers to different questions, and only one of them belongs on a dashboard after two data points. Now click <b>+10 at 70%</b> several times and watch prior and likelihood swap roles: the posterior narrows around the true generating rate and the shrinkage readout, which is exactly $(\\alpha_0+\\beta_0)/(\\alpha_0+\\beta_0+n)$, drops toward zero.</p>

<p><b>The thing genuinely worth noticing.</b> Drag the prior sliders to something opinionated — $\\alpha_0=20,\\ \\beta_0=2$, say, a strong prior belief that the rate is high — before adding any data at all, then click <b>+ failure</b> a single time. The posterior barely moves, because one observation is being weighed against 22 pseudo-observations already baked into the prior. This is the mechanism behind Thompson sampling (§6.6): an arm that starts with an optimistic, uncertain prior keeps getting tried until enough real data has actually accumulated to override it, and it is exactly why a 2-out-of-2 conversion rate on a brand-new page should never be reported as "100%".</p>

${H.lab('conjugate', 'Beta–Binomial updating, one observation at a time', 'The prior is your belief about a conversion rate before any data. Click to add a success or a failure and watch the posterior narrow. This is the entire mathematics behind Thompson sampling (§6.6) and behind why a 2-out-of-2 conversion rate should not be reported as 100%.')}

${H.history(`<p>Conjugate families were not a curiosity for most of the 20th century — for a working Bayesian statistician they were close to the entire toolkit. Thomas Bayes' original 1763 essay and Laplace's more thorough, independent development of the same idea a decade later both worked with problems simple enough to solve by hand. By the mid-20th century, Howard Raiffa and Robert Schlaifer had catalogued conjugate pairs systematically precisely because, without one, a Bayesian analysis of anything but a toy problem was computationally out of reach — there was no way to evaluate $p(D)$ for an arbitrary prior and likelihood, so you were restricted to the small list of combinations where the algebra happened to close.</p>
<p>That restriction is what the rest of this section exists to remove. Gibbs sampling and the Metropolis algorithm had existed since the 1950s in statistical physics, applied to entirely different problems, but it took until the methods were connected to Bayesian statistics around 1990 for the field to stop being limited to whichever prior made the algebra convenient and start using whichever prior actually represented its beliefs.</p>`)}

<h2><span class="sn">1.14.3</span> MCMC: build a chain whose stationary distribution is the posterior</h2>
${H.analogy(`<p>You are a hiker on a mountain range in thick fog: you can never see the whole landscape or measure your absolute elevation, which is exactly the position "you cannot compute $p(D)$" puts you in. But you carry an altimeter that compares two specific points at a time and reports only their <i>ratio</i> of unnormalised heights — which is exactly what evaluating $p(D\\mid\\theta)p(\\theta)$ at two values of $\\theta$ gives you, since the missing $p(D)$ is the same unknown constant at both points and cancels in the ratio.</p>
<p>The strategy: propose a nearby step. If it is higher, take it. If it is lower, take it anyway with a probability proportional to how much lower it is — never refuse a downhill step outright, or you would get stuck circling the first hill you found. Wander long enough this way and the <i>fraction of time you spend at each location</i> converges to that location's true, normalised elevation, even though you never once computed the mountain's absolute height. That fraction-of-time-spent is the posterior, and the wandering is Markov chain Monte Carlo.</p>`)}
<p>Metropolis–Hastings turns that analogy into an algorithm:</p>
${H.steps([
      'Propose $\\theta^{*} \\sim q(\\cdot\\mid\\theta_t)$ — usually a Gaussian step from where you are.',
      'Accept with probability $\\min\\!\\left(1, \\dfrac{p(D\\mid\\theta^*)p(\\theta^*)}{p(D\\mid\\theta_t)p(\\theta_t)}\\right)$ for a symmetric proposal. <b>The intractable $p(D)$ cancels in the ratio.</b> That cancellation is the whole trick.',
      'If accepted, move; otherwise stay put and record the current point again. The chain’s stationary distribution is the posterior.'
    ])}
<p>That last claim — that this particular accept/reject rule makes the posterior the chain's long-run resting distribution — is not obvious on sight, and it is worth deriving once rather than taking on faith.</p>
${H.deriv('why the acceptance rule makes the posterior a stationary distribution', [
      ['$\\pi(\\theta)\\,T(\\theta\\to\\theta^{*}) = \\pi(\\theta^{*})\\,T(\\theta^{*}\\to\\theta)$', 'A sufficient condition for a random walk with transition rule $T$ to leave a distribution $\\pi$ unchanged is <b>detailed balance</b>: the flow of probability from $\\theta$ to $\\theta^*$ exactly matches the flow back. If this holds for every pair of points, $\\pi$ is automatically a stationary distribution of the walk.'],
      ['$T(\\theta\\to\\theta^{*}) = q(\\theta^{*}\\mid\\theta)\\,A(\\theta\\to\\theta^{*})$', 'Split the transition into proposing a move, with density $q$, and then accepting it with probability $A$. Detailed balance must hold for this product.'],
      ['$A(\\theta\\to\\theta^{*}) = \\min\\!\\Big(1,\\ \\dfrac{\\pi(\\theta^{*})\\,q(\\theta\\mid\\theta^{*})}{\\pi(\\theta)\\,q(\\theta^{*}\\mid\\theta)}\\Big)$', 'This is the Metropolis–Hastings acceptance rule. Whichever side of the underlying ratio is larger has its acceptance probability capped at 1, and the other direction\'s acceptance probability is exactly that ratio (or its reciprocal) — arranged so both sides of the detailed-balance equation reduce to the same quantity, $\\min\\big(\\pi(\\theta)q(\\theta^*|\\theta),\\ \\pi(\\theta^*)q(\\theta|\\theta^*)\\big)$.'],
      ['$\\pi(\\theta) := p(D\\mid\\theta)\\,p(\\theta)$', 'Choose $\\pi$ to be the <i>unnormalised</i> numerator of Bayes\' rule, not the true posterior. Because $\\pi$ only ever appears as the ratio $\\pi(\\theta^*)/\\pi(\\theta)$ inside $A$, the missing constant $p(D)$ — the same for every $\\theta$ — appears in both the numerator and denominator of that ratio and cancels exactly. You never had to compute it.'],
      ['symmetric proposal: $q(\\theta^{*}\\mid\\theta) = q(\\theta\\mid\\theta^{*})$', 'The steps above use a Gaussian random-walk step, which proposes $\\theta^*$ and $\\theta$ from each other with equal density. The proposal ratio in $A$ is then exactly 1, and the acceptance rule collapses to the plain ratio of unnormalised posteriors given in step 2 of the algorithm.']
    ], 'Detailed balance only proves the posterior is a stationary distribution of the walk — that once you are distributed according to it, you stay that way. It does not by itself prove the chain gets there from an arbitrary starting point, which additionally needs the chain to be able to reach every region of nonzero posterior probability (irreducibility) and not get trapped in a cycle (aperiodicity). A Gaussian step on a connected parameter space provides both for free, which is why the algorithm above needs no further conditions to work in practice.')}

<p><b>What you are looking at.</b> The left panel is a 2-D posterior you cannot integrate in closed form — shaded by density, with contour lines at equal log-posterior height — overlaid with the actual path the Metropolis chain walked, drawn as a faint trail with individual visited points marked. The right panel is a trace plot: $\\theta_1$'s value plotted against iteration number, which is the standard first diagnostic every practitioner looks at before anything more sophisticated.</p>

<p><b>What to do with it.</b> Leave the shape at the default "banana" — a posterior that is curved and correlated, the kind no conjugate family could ever represent — and watch the acceptance-rate readout as you drag the step-size slider. Too small a step and the chain accepts nearly everything (a rate near 90%) but each step covers almost no ground, so the trace looks like a slow, heavily autocorrelated drift. Too large a step and the chain proposes moves so far into low-density territory that it rejects almost all of them, and the trace looks like a flat line interrupted by rare jumps. Somewhere around 25–40% acceptance the chain is doing the most useful exploring per proposal, which is why that range is the standard tuning target.</p>

<p><b>The thing genuinely worth noticing.</b> Switch the shape to <b>bimodal</b> and set a small step size. The chain finds one of the two modes, explores it thoroughly, reports a beautifully tight-looking posterior mean — and the acceptance rate, the trace plot, every single-chain diagnostic you have, looks completely healthy throughout. It never discovers that the other half of the true posterior exists. This is precisely why you run several chains from dispersed starting points and compare them with $\\hat R$: no diagnostic computed from one chain alone can detect a mode that chain never visited, because from inside that one chain there is nothing to detect.</p>

${H.lab('mcmc', 'Metropolis on a posterior you cannot integrate', 'A genuinely awkward "banana" posterior. Adjust the step size and watch the classic trade: too small and the chain crawls with a 90% acceptance rate and enormous autocorrelation; too large and it rejects nearly everything and stands still. The sweet spot is around 25–40% acceptance, which is the standard tuning target.')}

${H.table(['Diagnostic', 'What it detects', 'Threshold people use'], [
      ['Trace plot', 'the chain stuck, drifting, or not mixing', 'should look like a fuzzy caterpillar, not a wandering line'],
      ['Acceptance rate', 'proposal scale wrong', '≈0.234 asymptotically optimal for random-walk MH; 0.6–0.8 for HMC/NUTS'],
      ['$\\hat R$ (Gelman–Rubin)', 'chains from different starts disagreeing', '$\\hat R < 1.01$'],
      ['Effective sample size', 'autocorrelation eating your samples', 'ESS > 400 per parameter'],
      ['Divergences (HMC)', 'geometry the sampler cannot follow', 'any at all is a warning; reparameterise']
    ])}
${H.flag('Modern practice is not hand-rolled Metropolis. It is Hamiltonian Monte Carlo with the No-U-Turn Sampler (Stan, PyMC, NumPyro), which uses gradients of the log posterior to propose distant, high-acceptance moves. The lab above is Metropolis because it is the version you can watch; the version you would ship is NUTS.')}

<h2><span class="sn">1.14.4</span> Variational inference: turn integration into optimisation</h2>
<p>MCMC solves the problem completely, given enough time — and "given enough time" is doing a lot of work in that sentence when the likelihood must be evaluated over a large dataset at every single one of thousands of iterations. §1.12 spent an entire section building machinery for a different kind of problem: not "sample from a hard distribution" but "minimise an objective efficiently, at scale, with gradients". Variational inference is the move of converting the first kind of problem into the second.</p>
<p>Pick a tractable family $q_\\phi$, parameterised by $\\phi$ — a diagonal Gaussian, say — and search for the member of that family closest to the true posterior. "Closest" is measured in KL divergence (§1.10), specifically $\\mathrm{KL}(q_\\phi\\|p(\\theta\\mid D))$, and the obstacle is that this quantity contains $\\log p(D)$ buried inside it — the same intractable evidence this whole section has been dodging. The way around it is not to approximate the KL directly, but to find an exact identity hiding next to it.</p>
${H.deriv('the ELBO identity, exact rather than approximate', [
      ['$\\mathrm{KL}(q\\|p(\\theta\\mid D)) = \\mathbb{E}_q\\big[\\log q(\\theta) - \\log p(\\theta\\mid D)\\big]$', 'The definition of KL divergence (§1.10), applied to comparing the chosen family $q$ against the true posterior.'],
      ['$= \\mathbb{E}_q\\big[\\log q(\\theta) - \\log p(D,\\theta) + \\log p(D)\\big]$', 'Expand $p(\\theta\\mid D) = p(D,\\theta)/p(D)$ by Bayes\' rule (§1.1), and split the log of that quotient into a difference of logs.'],
      ['$= \\log p(D) - \\mathbb{E}_q\\big[\\log p(D,\\theta) - \\log q(\\theta)\\big]$', 'Pull $\\log p(D)$ outside the expectation, since it does not depend on $\\theta$, and rearrange the remaining terms. The bracketed expression is, by definition, the ELBO.'],
      ['$\\log p(D) = \\text{ELBO} + \\mathrm{KL}(q\\,\\|\\,p(\\theta\\mid D))$', 'Rearrange the previous line. This is an <i>identity</i>, exact for any $q$ whatsoever — not an approximation, and not a bound at this stage. The left side is a fixed number that does not depend on $q$ at all.']
    ], 'Because $\\log p(D)$ is fixed and $\\mathrm{KL}\\ge0$ always, the ELBO can never exceed $\\log p(D)$ — an evidence <i>lower</i> bound, which is where the name comes from — and since the two terms on the right must always sum to the same fixed number, pushing the ELBO up is mechanically the same act as pushing the KL term down. Maximising the ELBO over $\\phi$ is therefore silently minimising the exact divergence you wanted to minimise, and it never requires touching $p(D)$ directly: the ELBO itself, $\\mathbb{E}_q[\\log p(D,\\theta)-\\log q(\\theta)]$, is built entirely from quantities you can evaluate.')}
<p>The same identity, with $\\theta$ renamed to a latent $z$ and $q$ produced by an encoder network rather than fit per-datapoint, <i>is</i> the variational autoencoder (§6.3) — this derivation is not a preview of that one, it is that one, with different names on the symbols.</p>
${H.vs('MCMC', [
      'Asymptotically exact — given enough time it is the posterior',
      'Has honest diagnostics ($\\hat R$, ESS, divergences)',
      'Slow; scales badly in dimension and dataset size',
      'Hard to run inside a training loop'
    ], 'Variational inference', [
      'Fast, batched, differentiable — runs on a GPU',
      'Biased by the family you chose, with no way to measure the gap',
      'Mean-field $q$ <b>systematically under-estimates variance</b> — the mode-seeking direction of reverse KL',
      'The default when inference must be amortised over millions of data points'
    ])}
${H.key('Reverse KL $\\mathrm{KL}(q\\|p)$ is mode-seeking: it pays an infinite price for putting mass where $p$ has none, so $q$ hides inside one mode and comes out too narrow. Forward KL $\\mathrm{KL}(p\\|q)$ is mass-covering and over-disperses. Which one you optimise decides which way you will be wrong (§1.10).')}

${H.probe([
      ['Why does Metropolis–Hastings not need the normalising constant?', 'Only the ratio of unnormalised posteriors enters the acceptance probability, and $p(D)$ cancels.'],
      ['Your MCMC has $\\hat R = 1.3$. What does that mean and what do you do?', 'The chains have not converged to a common distribution. Run longer, reparameterise (non-centred parameterisation for hierarchical models), or check for an unidentified parameter.'],
      ['Why is mean-field VI over-confident?', 'It optimises reverse KL, which penalises $q$ having mass where $p$ has none but not the reverse — so $q$ contracts onto one mode. The posterior variance it reports is a lower bound.'],
      ['When is Bayesian inference actually worth the cost?', 'Small data, hierarchical structure with partial pooling, genuine need for calibrated uncertainty in a decision, or a sequential decision problem where the posterior drives exploration (§6.6). Not for a well-fed classifier where a validation set answers the question.']
    ], 'Claiming "Bayesian methods prevent overfitting". They regularise via the prior and average over uncertainty, but a badly specified model overfits Bayesianly and reports narrow intervals around the wrong answer.')}`,
    labs: {
      conjugate: function (host) {
        const el = ML.el;
        let a = 1, b = 1, s = 0, f = 0;
        const st = Viz.controls(host, [
          { k: 'a0', label: 'prior α', min: .5, max: 20, step: .5, value: 1, fmt: v => v },
          { k: 'b0', label: 'prior β', min: .5, max: 20, step: .5, value: 1, fmt: v => v }
        ], () => S.redraw());
        Viz.buttons(host, [
          { label: '+ success', primary: true, on: () => { s++; S.redraw(); } },
          { label: '+ failure', on: () => { f++; S.redraw(); } },
          { label: '+10 at 70%', on: () => { for (let i = 0; i < 10; i++) (Math.random() < .7 ? s++ : f++); S.redraw(); } },
          { label: 'clear data', on: () => { s = 0; f = 0; S.redraw(); } }
        ]);
        const out = Viz.readout(host, [
          { k: 'data', label: 'data', cls: 'key' },
          { k: 'mle', label: 'MLE (mode)' },
          { k: 'post', label: 'posterior mean' },
          { k: 'ci', label: '95% credible' },
          { k: 'shrink', label: 'shrinkage' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            a = st.a0 + s; b = st.b0 + f;
            const grid = Num.linspace(0.001, 0.999, 240);
            const post = grid.map(x => Num.betaPdf(x, a, b));
            const prior = grid.map(x => Num.betaPdf(x, st.a0, st.b0));
            const mx = Math.max(Math.max.apply(null, post), Math.max.apply(null, prior));
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, mx * 1.12], pad: { l: 44, r: 14, t: 16, b: 38 } })
              .frame({ xlabel: 'conversion rate θ', ylabel: 'density' });
            // credible interval by cumulative trapezoid
            let tot = 0; const cdf = [];
            for (let i = 0; i < grid.length; i++) { tot += post[i]; cdf.push(tot); }
            const qAt = p => { const target = p * tot; for (let i = 0; i < cdf.length; i++) if (cdf[i] >= target) return grid[i]; return 1; };
            const lo = qAt(.025), hi = qAt(.975);
            P.clip(() => {
              P.area(grid.map((x, i) => [x, post[i]]), { color: T.c1, alpha: .16 });
              P.line(grid.map((x, i) => [x, prior[i]]), { color: T.faint, width: 1.8, dash: [5, 4] });
              P.line(grid.map((x, i) => [x, post[i]]), { color: T.c1, width: 2.8 });
              if (s + f > 0) P.vline(s / (s + f), { color: T.c2, dash: false, width: 1.8, label: 'MLE' });
              P.vline(lo, { color: T.c1, dash: [3, 3], width: 1 });
              P.vline(hi, { color: T.c1, dash: [3, 3], width: 1 });
            });
            const mle = (s + f) ? s / (s + f) : NaN;
            const pm = a / (a + b);
            out({
              data: s + ' / ' + (s + f),
              mle: (s + f) ? mle.toFixed(3) : '—',
              post: pm.toFixed(3),
              ci: '[' + lo.toFixed(3) + ', ' + hi.toFixed(3) + ']',
              shrink: (s + f) ? ((st.a0 + st.b0) / (st.a0 + st.b0 + s + f) * 100).toFixed(0) + '% toward prior' : '100%'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--faint)', t: 'prior' }, { c: 'var(--c1)', t: 'posterior' }, { c: 'var(--c2)', t: 'MLE' }]);
        Viz.note(host, 'Click <b>+ success</b> twice from a uniform prior. The MLE says 100%; the posterior mean says 75% with a credible interval from 0.19 to 0.99. Both are correct answers to different questions, and only one of them is safe to put on a dashboard. Now add fifty observations and watch prior and likelihood swap importance — the shrinkage readout is exactly $(\\alpha_0+\\beta_0)/(\\alpha_0+\\beta_0+n)$.');
      },

      mcmc: function (host) {
        const st = Viz.controls(host, [
          { k: 'step', label: 'proposal step size', min: .05, max: 3, step: .05, value: .6, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'iterations', min: 200, max: 8000, step: 200, value: 3000, fmt: v => v.toLocaleString() },
          { k: 'burn', label: 'discard as burn-in', min: 0, max: 1000, step: 50, value: 200, fmt: v => v },
          { k: 'shape', label: 'posterior', type: 'select', value: 'banana', options: [{ v: 'banana', t: 'banana (correlated)' }, { v: 'bimodal', t: 'bimodal (hard)' }, { v: 'gauss', t: 'Gaussian (easy)' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'acc', label: 'acceptance rate', cls: 'key' },
          { k: 'verdict', label: 'tuning' },
          { k: 'ess', label: 'rough ESS' },
          { k: 'mean', label: 'posterior mean x' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const logp = st.shape === 'banana'
              ? p => { const x = p[0], y = p[1]; return -0.5 * (x * x / 4 + Math.pow(y - 0.25 * x * x + 1, 2) / 0.25); }
              : st.shape === 'bimodal'
                ? p => { const d1 = Math.pow(p[0] + 1.6, 2) + Math.pow(p[1], 2), d2 = Math.pow(p[0] - 1.6, 2) + Math.pow(p[1], 2);
                         return Num.logsumexp([-0.5 * d1 / 0.16, -0.5 * d2 / 0.16]); }
                : p => -0.5 * (p[0] * p[0] + p[1] * p[1]) / 0.6;

            const chain = Num.metropolis(logp, [0, 0], { step: st.step, seed: 4 });
            chain.draw(st.n);
            const pts = chain.chain.slice(st.burn);
            const xs = pts.map(p => p[0]);

            const w1 = w * .58;
            const P = Viz.plot(ctx, w1, h, { xd: [-4, 4], yd: [-3, 2.5], pad: { l: 40, r: 8, t: 16, b: 36 } })
              .frame({ xlabel: 'θ₁', ylabel: 'θ₂' });
            P.clip(() => {
              P.field((x, y) => Math.exp(logp([x, y])), { step: 4, colors: t => [90, 120, 200, Math.round(70 * t)] });
              P.contours((x, y) => logp([x, y]), [-8, -4, -2, -1], { color: T.faint, alpha: .5 });
              pts.forEach((p, i) => { if (i % Math.max(1, Math.floor(pts.length / 900)) === 0) P.dots([[p[0], p[1]]], { r: 1.7, color: T.c2, alpha: .55 }); });
              P.line(pts.slice(0, 120), { color: T.c2, width: .8, alpha: .55 });
            });

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, {
              xd: [0, xs.length], yd: [-4, 4], pad: { l: 36, r: 12, t: 16, b: 36 }
            }).frame({ xlabel: 'iteration', ylabel: 'θ₁ trace' });
            P2.clip(() => {
              P2.line(xs.map((v, i) => [i, v]), { color: T.c1, width: .9 });
              P2.hline(Num.mean(xs), { color: T.c2, dash: [4, 3], label: 'mean' });
            });
            ctx.restore();

            // crude ESS via lag-1 autocorrelation
            const m = Num.mean(xs);
            let num = 0, den = 0;
            for (let i = 1; i < xs.length; i++) num += (xs[i] - m) * (xs[i - 1] - m);
            for (let i = 0; i < xs.length; i++) den += (xs[i] - m) * (xs[i] - m);
            const r1 = den ? num / den : 0;
            const ess = Math.max(1, Math.round(xs.length * (1 - r1) / (1 + r1)));
            const acc = chain.rate;
            out({
              acc: (acc * 100).toFixed(1) + '%',
              verdict: acc > .6 ? 'steps too small' : acc < .12 ? 'steps too large' : '≈ well tuned',
              ess: ess.toLocaleString() + ' of ' + xs.length.toLocaleString(),
              mean: Num.mean(xs).toFixed(3)
            });
          }
        });
        Viz.note(host, 'Select <b>bimodal</b> and set a small step. The chain finds one mode, reports a beautifully tight posterior, and never discovers that the other half of the answer exists — with an acceptance rate that looks perfectly healthy. This is why you run several chains from dispersed starts and check $\\hat R$: no single-chain diagnostic can detect a mode you never visited.');
      }
    },
    quiz: [
      {
        q: 'The Metropolis acceptance ratio avoids the evidence $p(D)$ because…',
        options: ['it is assumed to be 1', 'it cancels in the ratio of unnormalised posteriors', 'it is estimated by sampling', 'the prior is conjugate'],
        answer: 1,
        why: 'The acceptance probability is built entirely from a ratio $\\pi(\\theta^*)/\\pi(\\theta)$ where $\\pi=p(D\\mid\\theta)p(\\theta)$, and the true posterior differs from $\\pi$ only by the constant factor $1/p(D)$ — a factor that appears in both the numerator and the denominator of that ratio and divides out exactly, as the detailed-balance derivation in §1.14.3 shows. "It is estimated by sampling" is the tempting wrong answer because estimating hard quantities by sampling is the theme of the whole surrounding chapter, but this is the one case where nothing needs estimating at all: the constant does not merely become easy to approximate, it vanishes from the arithmetic completely. Conjugacy is unrelated — MH works on any posterior, conjugate or not, which is the entire reason it exists.'
      },
      {
        q: 'You observe 2 conversions in 2 trials with a uniform Beta(1,1) prior. The posterior mean is…',
        options: ['1.00', '0.75', '0.67', '0.50'],
        answer: 1,
        why: 'The Beta–Binomial update gives Beta$(1+2,\\ 1+0)=$ Beta$(3,1)$, whose mean is $3/(3+1)=0.75$. The MLE — successes over trials, $2/2=1.00$ — is the tempting first answer, and it is precisely the number the worked example in §1.14.2 warns against reporting: it is the mode of an extremely wide posterior, not a trustworthy point estimate, and treating two lucky trials as proof of a 100% conversion rate is the exact mistake conjugate updating is built to prevent. The gap between 1.00 and 0.75 <i>is</i> the prior\'s two pseudo-observations pulling the estimate back from an overconfident edge case.'
      },
      {
        q: 'Mean-field variational inference typically reports posterior variances that are…',
        options: ['too large', 'too small', 'unbiased', 'not computable at all'],
        answer: 1,
        why: 'VI minimises reverse KL, $\\mathrm{KL}(q\\|p)$, which — as §1.10 establishes generally — penalises $q$ for placing mass where $p$ has none but not the reverse, so the fitted $q$ contracts to hide safely inside a single mode or a narrow region of the true posterior. The ELBO identity derived in §1.14.4 explains why this goes undetected: maximising the ELBO guarantees the KL term is being minimised, but it gives no information about <i>how much residual KL remains</i>, so a fitted $q$ can be confidently narrow with nothing in the optimisation objective to flag it. "Not computable at all" is wrong outright — $q$\'s variance is whatever parameter you fit it to be, it is simply an underestimate of the truth.'
      },
      {
        q: 'Which diagnostic would catch a sampler that never visited a second mode?',
        options: ['acceptance rate', 'multiple chains from dispersed starts, compared with $\\hat R$', 'the trace plot of one chain', 'the ELBO'],
        answer: 1,
        why: 'A chain that has spent its entire run inside one mode has nothing in its own history to suggest another mode exists — its acceptance rate can sit right in the healthy 25–40% range and its trace plot can look like a perfectly fuzzy caterpillar, exactly as the bimodal setting in the MCMC lab above demonstrates. The only way to expose a missed mode is a diagnostic that compares <i>across</i> chains: start several chains from widely dispersed points, and if they disagree about where the posterior mass is, $\\hat R$ — the Gelman–Rubin statistic — flags it, because a single chain literally cannot report evidence about territory it never visited.'
      }
    ],
    cards: [
      { q: 'Beta–Binomial update', a: 'Beta$(\\alpha,\\beta)$ + $s$ successes in $n$ trials → Beta$(\\alpha+s,\\beta+n-s)$. Prior worth $\\alpha+\\beta$ observations.' },
      { q: 'Why MH does not need $p(D)$', a: 'The acceptance probability is a ratio of unnormalised posteriors; the constant cancels.' },
      { q: 'The ELBO identity', a: '$\\log p(D) = \\text{ELBO} + \\mathrm{KL}(q\\|p(\\theta|D))$. Maximising the ELBO minimises the KL because the left side is fixed.' },
      { q: 'MCMC health checks', a: '$\\hat R<1.01$, ESS > 400/parameter, acceptance ≈0.23 (RWMH) or 0.6–0.8 (NUTS), zero divergences.' },
      { q: 'Reverse vs forward KL', a: 'Reverse (VI) is mode-seeking and under-disperses; forward is mass-covering and over-disperses.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.15 */
  ML.section({
    id: 'numerics', track: 'foundations', num: '1.15', level: 2,
    title: 'Floating point, stability, and the epsilon in Adam',
    lede: 'Real numbers do not exist in a computer. Almost every "mysterious NaN", every loss that becomes infinity at step 4,000, and every model that trains in float32 and breaks in float16 is one of about five well-understood floating-point failures.',
    related: ['optimisers', 'distributed', 'compression'],
    html: `
${H.tldr([
      'A float has finite <b>range</b> (the exponent) and finite <b>precision</b> (the mantissa). fp16 fails on range; bf16 trades precision to keep fp32’s range, which is why it won for training.',
      'Three moves fix most instability: work in <b>log space</b>, subtract the max before <code>exp</code> (<b>log-sum-exp</b>), and never subtract two nearly-equal large numbers.',
      'The $\\varepsilon$ in Adam, the $\\varepsilon$ in LayerNorm and the clipping in cross-entropy are all the same defensive move: stop a denominator or a logarithm from reaching zero.'
    ])}

<p>Open any Python prompt and type <code>(1e20 + 1) - 1e20</code>. Algebraically the answer is obviously 1 — add one to a number, subtract the same number back off, and you are left with the one you added. The computer returns <code>0.0</code>.</p>

<p>Nothing is broken. The computer did exactly what floating-point arithmetic specifies. To see why, push the same experiment slightly: <code>(1e15 + 1) - 1e15</code> correctly returns <code>1.0</code>, but <code>(1e16 + 1) - 1e16</code> already returns <code>0.0</code>. Somewhere between $10^{15}$ and $10^{16}$ the number 1 stopped being representable as an offset from a number that size, and that boundary is not fuzzy — it sits at exactly $2^{53} \\approx 9.007\\times10^{15}$, the point past which a 64-bit float runs out of digits to distinguish $n$ from $n+1$. Below that line, every integer is stored exactly. Above it, the gap between two consecutive representable numbers is itself bigger than 1, so adding 1 can do nothing at all — the addition is not wrong, it rounds to the only value nearby that the format can actually store.</p>

<h2><span class="sn">1.15.1</span> What a float actually is</h2>
<p>A value is stored as $\\pm\\, m \\times 2^{e}$ — a sign, a fixed number of bits for the <b>mantissa</b> $m$ (the significant digits) and a fixed number of bits for the <b>exponent</b> $e$ (how far to shift the decimal point). The exponent sets how big or small a number you can represent at all; the mantissa sets how many significant digits you keep once you are there — and that second part is the direct explanation of the experiment above.</p>
${H.analogy(`<p>A float is a ruler whose tick marks get further apart the further out you look, always keeping the same <i>number</i> of ticks between each power of two. Near 1, the ticks are separated by about $2^{-52}$, closer together than you could ever need. Near $10^{20}$, the same 52 bits of mantissa have to cover a vastly larger span, so the ticks there are separated by roughly 16,384 — anything smaller than that gap, added to a number out there, lands on the same tick it started from and vanishes.</p>
<p>This is why floating point gives you constant <i>relative</i> precision (about 15–17 significant decimal digits, always) rather than constant <i>absolute</i> precision. It is a deliberate, correct design for a world where quantities span enormous ranges — but it means the question "is this number precise enough?" only has an answer once you know how big the number next to it is.</p>`)}
${H.table(['Format', 'Bits (s/e/m)', 'Max', 'Smallest normal', 'Decimal digits', 'Used for'], [
      ['fp64', '1/11/52', '~1.8e308', '~2.2e-308', '~15.9', 'scientific computing, gradient checks'],
      ['fp32', '1/8/23', '~3.4e38', '~1.2e-38', '~7.2', 'the default everywhere until 2018'],
      ['<b>bf16</b>', '1/8/7', '~3.4e38', '~1.2e-38', '~2.4', '<b>training</b> — same range as fp32, no loss scaling needed'],
      ['fp16', '1/5/10', '65,504', '~6.1e-5', '~3.3', 'training with loss scaling; inference'],
      ['fp8 (E4M3)', '1/4/3', '448', '~2e-3', '~1.5', 'inference, and forward passes on H100-class hardware'],
      ['int8', 'integer', '127', '1', 'exact, 256 levels', 'post-training quantisation (§3.13)']
    ], 'num')}
${H.key('bf16 has the same exponent as fp32 and only 8 bits of mantissa. It sacrificed precision — which training tolerates, because gradients are noisy anyway — to keep range, which training does not tolerate losing. That single design choice is why bf16 replaced fp16 for pretraining.')}
${H.history(`<p>Before 1985 every hardware vendor's floating point rounded, overflowed and lost precision according to its own conventions, so the same formula could give a different answer on a different machine. IEEE 754, driven largely by William Kahan, standardised the format this section describes — including the exact rounding rule that makes <code>(1e16+1)-1e16</code> return zero identically on every conforming processor built since. That predictability is the entire reason "compute it and see" is a reliable debugging technique at all: the behaviour is specified, not accidental.</p>
<p>bf16 — "brain float", 16 bits — is much more recent, developed at Google Brain and built into the first generation of TPUs around 2018 for exactly one reason: fp32's exponent kept training stable, fp16's exponent did not, and nobody had previously needed a format that kept fp32's range while giving up its precision, because outside neural-network training almost nothing tolerates losing that many significant digits.</p>`)}

<p><b>What you are looking at.</b> A single value, set by the two sliders — its order of magnitude and how many significant digits past that magnitude it carries — shown as a marker on four stacked number lines, one per format, each shaded across the span that format can represent without overflowing or flushing to zero. The readout below reports exactly what each format would round your value to, and how far off that rounding is in relative terms.</p>

<p><b>What to do with it.</b> There is nothing to drag or type here; both controls are sliders, one for the magnitude ($10^x$) and one for how many extra significant digits to add on top of it. Set the magnitude to $10^{-6}$ — an entirely ordinary gradient value deep in a network — and read the four rows: fp32 and bf16 both hold it comfortably, but fp16 reports <b>underflow → 0</b>. That single row is the entire reason loss scaling exists, and the entire reason bf16 made loss scaling unnecessary.</p>

<p><b>The thing genuinely worth noticing.</b> Now push the magnitude up past $10^{6}$ and watch fp16 and fp8 both report <b>overflow → inf</b> while fp32 and bf16 carry on unaffected — the same asymmetry as before, mirrored to the other end of the number line. bf16 never leaves the "working" state anywhere you can push this slider, for a value that would defeat fp16 twice over, in opposite directions. That is what "same exponent as fp32" buys you, concretely, one slider position at a time.</p>

${H.lab('formats', 'What happens to your number in each format', 'Drag the magnitude and precision sliders and see exactly what each format stores, what it rounds to, and where it overflows or flushes to zero. The gradient magnitudes typical of a deep network are marked.')}

<h2><span class="sn">1.15.2</span> The five failures</h2>
<p>The vanishing 1 at the top of this section is one instance of a small, closed list of ways floating point breaks. Once you can name which of the five you are looking at, the fix is usually already known.</p>
${H.table(['Failure', 'Symptom', 'Cause', 'Fix'], [
      ['Overflow', 'inf, then NaN everywhere', '$e^{z}$ with $z=800$; fp16 gradients past 65,504', 'log-sum-exp; loss scaling; bf16'],
      ['Underflow', 'silent zeros, gradient stops', 'product of 500 probabilities; fp16 gradients below 6e-5', 'work in log space; loss scaling'],
      ['Catastrophic cancellation', 'wrong answer, no warning', 'subtracting nearly equal numbers destroys significant digits', 'algebraic rearrangement; Welford; two-pass'],
      ['Division by ~zero', 'inf, NaN, exploding update', 'normalising by a variance that is 0', 'the $\\varepsilon$ in Adam and LayerNorm'],
      ['Non-associativity', 'results differ run to run', '$(a+b)+c \\ne a+(b+c)$ in float; GPU reduction order varies', 'accept it, or force deterministic kernels and pay ~10–20%']
    ])}

<p>Overflow is the most dramatic of the five, and it is worth deriving its standard fix once rather than memorising the formula. Every softmax and every cross-entropy loss needs $\\log\\sum_j e^{z_j}$, the normaliser of a set of scores. Feed it a perfectly reasonable logit, $z=1000$ say, and $e^{1000}$ overflows fp64 outright — the true value of the whole expression is only about 1000, and the naive computation cannot even get started.</p>
${H.deriv('log-sum-exp: the most useful three lines in numerical ML', [
      ['$\\log\\sum_j e^{z_j}$', 'Wanted. With $z_j = 1000$, every term overflows and the answer is <code>inf</code> — even though the true value is about 1000.'],
      ['$= \\log\\sum_j e^{z_j - c}\\,e^{c}$', 'Insert $e^{c}e^{-c}=1$ for any constant $c$ whatsoever, multiplying and dividing each term by it. Nothing has changed mathematically — this is the identity trick, the same move §1.13.1 uses to derive importance sampling.'],
      ['$= c + \\log\\sum_j e^{z_j-c}$', 'Pull the constant $e^c$ out of the sum (it does not depend on $j$) and then out of the log, turning a product into a sum. Still an exact rearrangement, true for any $c$.'],
      ['$c := \\max_j z_j$', 'Now choose the max. The largest shifted exponent is exactly $z_{\\max}-c=0$, so $e^0=1$ — every other term is in $(0,1]$, nothing can overflow, and the worst that happens is a tiny term underflows to zero, where it was contributing nothing to the sum anyway.']
    ], 'This is inside every softmax, every cross-entropy, every HMM forward pass and every mixture-model E-step you will ever use. Frameworks do it for you <i>only if you use the fused op</i>: <code>cross_entropy(logits, y)</code> is stable, <code>log(softmax(logits))</code> then NLL is not. (If this derivation felt familiar, §1.10.6 walks the identical three lines in the specific context of a softmax normaliser — this version is the same trick, generalised.)')}

<p><b>What you are looking at.</b> A small table, recomputed live as you move the sliders, running the same three calculations two ways each: a softmax probability, a product of many probabilities, and a variance — each once the naive way and once the numerically stable way described above and in §1.15.3 below. A result that has actually failed — an <code>inf</code>, a <code>NaN</code>, or a silent flush to zero — is coloured red rather than reported as a plausible-looking number.</p>

<p><b>What to do with it.</b> Start with the magnitude slider low, where naive and stable agree to full precision, and raise it slowly. Watch the naive softmax column specifically: it stays finite and correct right up to a magnitude of 709, and at 710 it flips straight to <code>NaN</code>. There is no gradual degradation in between — nothing warns you as you approach the edge.</p>

<p><b>The thing genuinely worth noticing.</b> Push the "mean offset for the variance" slider up toward $10^8$ and watch the naive one-pass variance column, not merely fail, but return a <i>negative</i> number — a variance, which is a sum of squares, reported as less than zero. That silence is the dangerous part: an <code>inf</code> or a <code>NaN</code> at least announces itself and typically crashes something downstream loudly enough to be noticed; a confidently wrong finite number does not, and it is why cancellation is treated separately from overflow and underflow in the table above even though all three are floating-point failures at heart.</p>

${H.lab('stability', 'Break it, then fix it', 'The same three computations done naively and stably, on inputs you control. Push the magnitude slider and watch the naive column produce inf, NaN or a negative variance while the stable column keeps working.')}

<h2><span class="sn">1.15.3</span> Cancellation, and the negative variance</h2>
<p>The opening example of this section subtracted two numbers that were supposed to be nearly equal and lost the small difference entirely — that is <b>catastrophic cancellation</b> by name, and it is not confined to contrived examples. The textbook one-pass variance formula, $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$, is algebraically exact and numerically dangerous whenever the mean is large relative to the spread, for exactly the same reason $10^{20}+1$ could not be told apart from $10^{20}$. With values clustered near $10^9$ and a standard deviation of 1, $\\mathbb{E}[X^2]$ and $\\mathbb{E}[X]^2$ agree to roughly eighteen digits while fp64 carries only about sixteen — so their difference is not a small true value plus a little rounding noise, it is <i>almost entirely</i> rounding noise, and it can and does come out negative, a variance the algebra guarantees can never be less than zero.</p>
${H.code(`# Welford's online algorithm: one pass, numerically stable, and it
# streams — which is why it is what every metrics library actually uses.
n = 0; mean = 0.0; M2 = 0.0
for x in stream:
    n += 1
    d = x - mean
    mean += d / n
    M2 += d * (x - mean)      # note: the *updated* mean. That is the trick.
var = M2 / (n - 1)`)}
<p>The trick is visible in the comment: by centring each new point against a running mean <i>before</i> squaring anything, Welford's update never forms the two enormous, nearly-equal quantities $\\mathbb{E}[X^2]$ and $\\mathbb{E}[X]^2$ in the first place, so there is nothing large for the cancellation to eat into. It computes the same mathematical quantity as the one-pass formula, by a route that never manufactures the danger.</p>
${H.note('The same disease appears in $\\log(1+x)$ for tiny $x$ (use <code>log1p</code>), in $e^{x}-1$ (use <code>expm1</code>), and in the quadratic formula when $b^2 \\gg 4ac$. Every language ships the fixed versions; almost nobody reaches for them until they have been bitten.')}

<h2><span class="sn">1.15.4</span> Where the epsilons live</h2>
${H.table(['Place', 'Typical value', 'What it prevents', 'What happens if you change it'], [
      ['Adam denominator $\\sqrt{\\hat v}+\\varepsilon$', '1e-8', 'division by zero for a coordinate with no gradient history', 'larger ε damps the adaptivity — 1e-3 makes Adam behave more like SGD; some LLM recipes use 1e-6 for stability in bf16'],
      ['LayerNorm / BatchNorm $\\sqrt{\\sigma^2+\\varepsilon}$', '1e-5', 'a constant feature has zero variance', 'too small → inf on constant inputs; too large → the normalisation stops normalising'],
      ['Cross-entropy clipping', '1e-7…1e-12', '$\\log 0 = -\\infty$', 'use the fused logits version and you do not need it at all'],
      ['Softmax temperature floor', '~1e-3', 'divide-by-zero at $T\\to0$', 'greedy decoding should be a branch, not a limit (§4.15)'],
      ['Gradient clipping norm', '1.0', 'a single bad batch destroying the weights', 'the standard defence against loss spikes in LLM pretraining (§4.11)']
    ])}
${H.pitfall('Adam’s $\\varepsilon$ sits <i>outside</i> the square root in the standard formulation: $\\hat m/(\\sqrt{\\hat v}+\\varepsilon)$. Putting it inside, $\\hat m/\\sqrt{\\hat v+\\varepsilon}$, is a different algorithm with different behaviour for small gradients — and both appear in real codebases. If you are debugging a reproduction that will not match, this is a place to look.')}

${H.probe([
      ['Why did bf16 replace fp16 for training?', 'Same 8-bit exponent as fp32, so activations and gradients cannot overflow or flush to zero, and no loss scaling is required. It pays with mantissa bits, which stochastic training barely notices.'],
      ['Write down the stable softmax.', 'Subtract $\\max_j z_j$ from every logit first: $\\mathrm{softmax}(z)_k = e^{z_k-c}/\\sum_j e^{z_j-c}$ with $c=\\max z$. Exact, and nothing can overflow.'],
      ['Your variance came out negative. Explain.', 'One-pass $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$ with a large mean: catastrophic cancellation. Use Welford or a two-pass computation.'],
      ['Why is the same training run not bit-identical on two GPUs?', 'Floating-point addition is not associative, and reduction order depends on the kernel, the block size and the number of devices. Determinism is available and costs throughput.'],
      ['What is loss scaling and why does fp16 need it?', 'Multiply the loss by a large constant before the backward pass so gradients land inside fp16’s representable range, then unscale before the optimiser step. bf16 does not need it because its range already matches fp32.']
    ], 'Treating NaN as a mystery. NaN has exactly a few sources — 0/0, inf−inf, log of a negative, sqrt of a negative, and inf*0. Find the first NaN, not the hundredth: hooks that check each tensor after every op will localise it in one run.')}`,
    labs: {
      formats: function (host) {
        const st = Viz.controls(host, [
          { k: 'exp', label: 'magnitude (10^x)', min: -12, max: 12, step: .25, value: -4, fmt: v => '1e' + v.toFixed(2) },
          { k: 'mant', label: 'mantissa digits', min: 1, max: 9, step: 1, value: 4, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'val', label: 'your value', cls: 'key' },
          { k: 'f32', label: 'fp32' }, { k: 'bf', label: 'bf16' },
          { k: 'f16', label: 'fp16' }, { k: 'f8', label: 'fp8 e4m3' }
        ]);

        function round(v, mantBits, maxV, minNormal) {
          if (!isFinite(v) || Math.abs(v) > maxV) return Infinity * Math.sign(v);
          if (Math.abs(v) < minNormal) return 0;
          const e = Math.floor(Math.log2(Math.abs(v)));
          const q = Math.pow(2, e - mantBits);
          return Math.round(v / q) * q;
        }
        const FMT = {
          f32: { m: 23, max: 3.4e38, min: 1.2e-38 },
          bf: { m: 7, max: 3.4e38, min: 1.2e-38 },
          f16: { m: 10, max: 65504, min: 6.1e-5 },
          f8: { m: 3, max: 448, min: 2e-3 }
        };
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const v = Math.pow(10, st.exp) * (1 + 0.1 * st.mant);
            const P = Viz.plot(ctx, w, h, { xd: [-14, 14], yd: [0, 5], pad: { l: 60, r: 16, t: 20, b: 40 } })
              .frame({ grid: false, yticks: [], xticks: [-12, -8, -4, 0, 4, 8, 12], xfmt: x => '1e' + x, xlabel: 'magnitude' });
            const rows = [['fp32', FMT.f32, T.c1], ['bf16', FMT.bf, T.c3], ['fp16', FMT.f16, T.c4], ['fp8 e4m3', FMT.f8, T.c2]];
            rows.forEach((r, i) => {
              const y = 4 - i;
              const lo = Math.log10(r[1].min), hi = Math.log10(r[1].max);
              ctx.fillStyle = r[2]; ctx.globalAlpha = .22;
              ctx.fillRect(P.x(lo), P.y(y) - 11, P.x(hi) - P.x(lo), 22);
              ctx.globalAlpha = 1;
              ctx.strokeStyle = r[2]; ctx.lineWidth = 1.4;
              ctx.strokeRect(P.x(lo), P.y(y) - 11, P.x(hi) - P.x(lo), 22);
              ctx.fillStyle = T.text; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText(r[0], P.x(-14) - 6, P.y(y));
            });
            P.vline(st.exp, { color: T.text, width: 2, dash: false, label: 'your value' });
            // the band where deep-net gradients live
            ctx.fillStyle = T.faint; ctx.globalAlpha = .16;
            ctx.fillRect(P.x(-8), P.y(4.7), P.x(-2) - P.x(-8), P.y(0.2) - P.y(4.7));
            ctx.globalAlpha = 1;
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
            ctx.fillText('typical gradient magnitudes', P.x(-5), P.y(4.75));

            const fmtOut = (key) => {
              const r = round(v, FMT[key].m, FMT[key].max, FMT[key].min);
              if (!isFinite(r)) return 'overflow → inf';
              if (r === 0) return 'underflow → 0';
              const relErr = Math.abs(r - v) / v;
              return r.toExponential(3) + '  (' + (relErr * 100).toFixed(relErr > .01 ? 1 : 3) + '% off)';
            };
            out({ val: v.toExponential(6), f32: fmtOut('f32'), bf: fmtOut('bf'), f16: fmtOut('f16'), f8: fmtOut('f8') });
          }
        });
        Viz.note(host, 'Set the magnitude to 1e-6 — a perfectly ordinary gradient. fp32 and bf16 hold it; <b>fp16 flushes it to zero</b>, and the parameter it belonged to stops learning. That single failure is what loss scaling exists to prevent, and what bf16 makes unnecessary. Now go to 1e6 and watch fp16 and fp8 overflow instead.');
      },

      stability: function (host) {
        const st = Viz.controls(host, [
          { k: 'mag', label: 'input magnitude', min: 1, max: 900, step: 1, value: 40, fmt: v => v },
          { k: 'nprob', label: 'probabilities to multiply', min: 10, max: 2000, step: 10, value: 400, fmt: v => v },
          { k: 'offset', label: 'mean offset for the variance', min: 0, max: 10, step: .5, value: 0, fmt: v => '1e' + v }
        ], () => draw());
        const host2 = ML.el('div');
        host.appendChild(host2);

        function draw() {
          const z = [st.mag, st.mag - 1, st.mag - 3, st.mag - 7];
          // 1. softmax
          const naiveDen = z.reduce((a, v) => a + Math.exp(v), 0);
          const naive = z.map(v => Math.exp(v) / naiveDen);
          const c = Math.max.apply(null, z);
          const stableDen = z.reduce((a, v) => a + Math.exp(v - c), 0);
          const stable = z.map(v => Math.exp(v - c) / stableDen);
          // 2. product of probabilities
          let prod = 1, logsum = 0;
          for (let i = 0; i < st.nprob; i++) { prod *= 0.7; logsum += Math.log(0.7); }
          // 3. variance with a large offset
          const R = Num.rng(2);
          const off = Math.pow(10, st.offset);
          const xs = Array.from({ length: 200 }, () => off + R.normal(0, 1));
          let s1 = 0, s2 = 0;
          xs.forEach(x => { s1 += x; s2 += x * x; });
          const onepass = s2 / xs.length - Math.pow(s1 / xs.length, 2);
          const m = s1 / xs.length;
          let ss = 0; xs.forEach(x => { ss += (x - m) * (x - m); });
          const twopass = ss / xs.length;

          const fmt = v => !isFinite(v) ? '<span style="color:var(--red)">' + (v > 0 ? 'inf' : '-inf') + '</span>'
            : isNaN(v) ? '<span style="color:var(--red)">NaN</span>'
              : (v === 0 ? '<span style="color:var(--red)">0 (underflow)</span>' : v.toPrecision(6));

          host2.innerHTML = H.table(
            ['Computation', 'Naive', 'Stable', 'Why'],
            [
              ['softmax, largest probability',
                fmt(naive[0]), fmt(stable[0]),
                '<code>exp(' + st.mag + ')</code> overflows fp64 past 709'],
              ['product of ' + st.nprob + ' × 0.7',
                fmt(prod), 'exp(' + logsum.toFixed(1) + ') — kept in logs',
                'underflows below ~1e-308'],
              ['variance, values near 1e' + st.offset,
                fmt(onepass), fmt(twopass),
                'E[X²] − E[X]² cancels away every significant digit']
            ], 'num');
        }
        draw();
        Viz.note(host, 'Push the magnitude past 709 and the naive softmax returns NaN — <code>inf/inf</code>. The stable one is unmoved, because after subtracting the max the largest term is exactly 1. Push the mean offset to 1e8 and the one-pass variance goes negative: an impossible answer, returned confidently, with no warning of any kind. <b>That silence is what makes cancellation the dangerous one.</b>');
      }
    },
    quiz: [
      {
        q: 'bf16 is preferred to fp16 for training because…',
        options: ['it has more mantissa bits', 'it has the same exponent range as fp32, so gradients neither overflow nor flush to zero', 'it is faster on all hardware', 'it is more accurate'],
        answer: 1,
        why: 'bf16 allocates the same 8 bits to its exponent as fp32, so it can represent the same enormous range of magnitudes — an ordinary gradient never overflows and never flushes silently to zero, as the formats lab demonstrates directly. "It has more mantissa bits" is exactly backwards: bf16 has <i>fewer</i> mantissa bits than fp16 (7 against 10) and is measurably less precise number-for-number. It wins anyway, because range is the property training cannot survive losing, while stochastic gradient noise already swamps the precision it gives up — a trade that would be a poor one for, say, scientific computing, and is a good one specifically for training.'
      },
      {
        q: 'The log-sum-exp trick works by…',
        options: ['taking logs of the inputs', 'subtracting the maximum before exponentiating', 'clipping large values', 'using float64'],
        answer: 1,
        why: 'The identity $\\log\\sum_j e^{z_j} = c + \\log\\sum_j e^{z_j-c}$ holds exactly for any constant $c$, and choosing $c=\\max_j z_j$ makes the largest shifted exponent exactly $e^0=1$, so every term is bounded in $(0,1]$ and overflow becomes structurally impossible. "Clipping large values" is the tempting wrong answer because it also prevents overflow, but clipping changes the actual numbers going into the computation and returns an approximation; the max-subtraction trick is an algebraic rearrangement that returns the mathematically exact answer, just computed by a route that never forms a number too large to store.'
      },
      {
        q: 'Your one-pass variance returns −3.2e-7. The cause is…',
        options: ['a bug in the data', 'catastrophic cancellation between $\\mathbb{E}[X^2]$ and $\\mathbb{E}[X]^2$', 'integer overflow', 'the sample being too small'],
        answer: 1,
        why: 'The one-pass formula subtracts two quantities, $\\mathbb{E}[X^2]$ and $\\mathbb{E}[X]^2$, that are enormous and nearly identical whenever the data\'s mean is large relative to its spread — precisely the $(10^{20}+1)-10^{20}$ situation this section opens with, wearing a statistical formula\'s clothes. The true variance is a small residual sitting past the digits fp64 can actually resolve, so the computed difference is dominated by rounding error and can land on either side of zero. "Integer overflow" cannot be the cause because none of these quantities are integers or anywhere near the integer overflow range; the fix, as for the opening example, is to never form the two large near-equal numbers in the first place — which is exactly what Welford\'s running update accomplishes.'
      },
      {
        q: 'Adam’s $\\varepsilon = 10^{-8}$ exists to…',
        options: ['add regularisation', 'prevent division by zero when the second-moment estimate is tiny', 'set the learning rate floor', 'stabilise the momentum'],
        answer: 1,
        why: 'Adam divides the momentum-smoothed gradient by $\\sqrt{\\hat v}$, an estimate of that coordinate\'s recent squared-gradient magnitude, and for a parameter with little or no gradient history $\\hat v$ can sit at or near zero — without $\\varepsilon$ added to the denominator, that division is exactly the "division by ~zero" failure mode in this section\'s table of five, producing <code>inf</code> or <code>NaN</code> updates. It is a numerical guard, not a modelling choice: it does not penalise any particular weight value the way an L2 term would, it only stops one specific arithmetic operation from blowing up. Raising it toward 1e-3 has a real, secondary effect worth knowing — it damps Adam\'s per-coordinate adaptivity and pushes its behaviour toward plain SGD with momentum, because the denominator stops being dominated by the (now swamped) gradient-history term.'
      }
    ],
    cards: [
      { q: 'fp16 vs bf16, in one line', a: 'Same 16 bits: fp16 spends them on precision, bf16 on range. Training needs range, so bf16 won.' },
      { q: 'Stable softmax', a: 'Subtract $\\max_j z_j$ from every logit before exponentiating. Exact, and overflow becomes impossible.' },
      { q: 'Welford’s update', a: '$d = x-\\mu$; $\\mu \\mathrel{+}= d/n$; $M_2 \\mathrel{+}= d\\,(x-\\mu_{\\text{new}})$. One pass, stable, streaming.' },
      { q: 'fp16 representable range', a: 'Max 65,504; smallest normal ≈6e-5. Gradients below that flush to zero — hence loss scaling.' },
      { q: 'Why runs are not bit-reproducible', a: 'Float addition is not associative and GPU reduction order varies with kernel, block size and device count.' }
    ]
  });
})();
