/* ============================================================
   PART 2 — Core & classical ML (2.7 – 2.11)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.7 */
  ML.section({
    id: 'trees', track: 'classical', num: '2.7',
    title: 'Trees, bagging, random forests',
    lede: 'Rests on §1.3’s variance algebra and §1.10’s entropy; sets up §2.8 by contrast — bagging attacks variance with independent parallel models, boosting attacks bias with dependent sequential ones.',
    html: `
<p>Put yourself in front of a stack of loan files with a simple job: sort each one into "approve" or "decline" by asking questions about the applicant, one at a time, in whatever order seems useful. You are not allowed a formula — no weighted sum, no probability, just a sequence of yes/no questions, like the weather game in §1.10. <i>Is utilisation above 0.6? If yes, have there been more than two enquiries in the last six months?</i> Follow enough of these chains and you have built, without knowing the word for it, a <b>decision tree</b>: a flowchart of questions that ends in a verdict.</p>

<p>The one real design decision in that game is which question to ask first, and second, and so on. Some questions barely help — asking about a feature that has nothing to do with default leaves you exactly as unsure as before. Some questions help enormously — asking about a feature that neatly separates the good files from the bad ones. You need a number that scores a question by how much it clears things up. That number is called <b>impurity</b>, and a tree is nothing more than the greedy strategy of always asking the question that reduces it the most.</p>

<h2><span class="sn">2.7.1</span> Measuring how mixed a group is</h2>

<p>Start with the group you are trying to split. Say 100 applicants sit in front of you, 60 who went on to repay and 40 who defaulted. Guess "repay" for every one of them and you are right 60% of the time — not a terrible guess, but a long way from confident. This node is <i>impure</i>: it contains a real mix of both outcomes, and no single label describes it well.</p>

<p>Now try a candidate question: <i>is utilisation at most 0.5?</i> Seventy applicants say yes, and of those, 55 repaid and only 15 defaulted — a node that is now mostly one thing. Thirty applicants say no, and of those only 5 repaid against 25 who defaulted — the other node has swung almost as hard the other way. Asking this one question has taken a 60/40 muddle and produced two much more confident groups. That is what a good split does, and everything below is just putting a number on how "confident" each group is.</p>

<p><b>Gini impurity</b> asks a direct question about that confidence: if you picked two applicants from this node at random, with replacement, what is the chance their labels disagree? Write $p_k$ for the fraction of the node in class $k$; the chance the first pick lands in class $k$ and the second does not is $p_k(1-p_k)$, and summing over classes and simplifying gives</p>

$$\\text{Gini} = 1-\\sum_k p_k^2$$

<p>Read it as: take the fraction in each class, square it, add the squares, and subtract from 1. Squaring rewards a node for being lopsided — $0.5^2 + 0.5^2 = 0.5$ is the worst a two-class node can score, while $0.9^2+0.1^2=0.82$, giving Gini $=0.18$, is close to the floor of 0. A node with only one class present scores exactly 0, because $1^2=1$ and there is nothing left to subtract.</p>

<p><b>Entropy</b> asks the same question in the vocabulary §1.10 already built: how many yes/no questions, on average, would it take to learn one applicant's true label if you had to guess it from these proportions? $-\\sum_k p_k\\log_2 p_k$ is exactly the entropy formula from that section, applied to the label distribution inside a node rather than to tomorrow's weather. A pure node needs zero questions because you already know the answer; a 50/50 node needs one full bit, the most any two-outcome distribution can demand.</p>

${H.worked('the split above, scored both ways', `<p>Parent: 60 repay, 40 default out of 100. $\\text{Gini} = 1-(0.6^2+0.4^2) = 1-0.52 = 0.48$. $\\text{Entropy} = -(0.6\\log_2 0.6+0.4\\log_2 0.4) = 0.971$ bits.</p>
<p>Left child (utilisation ≤ 0.5, $n=70$): 55 repay, 15 default. $\\text{Gini} = 1-(0.786^2+0.214^2) = 0.337$.</p>
<p>Right child ($n=30$): 5 repay, 25 default. $\\text{Gini} = 1-(0.167^2+0.833^2) = 0.278$.</p>
<p>Weight each child by its share of the 100 rows and add: $\\tfrac{70}{100}(0.337) + \\tfrac{30}{100}(0.278) = 0.319$. The impurity fell from 0.48 to 0.319 — a <b>Gini reduction of 0.161</b>, and this is exactly the number a real implementation compares across every candidate split at this node before picking a winner. The same arithmetic on entropy gives a weighted child entropy of 0.720 against a parent of 0.971, an <b>information gain of 0.251</b> bits: over a quarter of a bit of genuine uncertainty removed by one question.</p>`)}

<p>Notice the two criteria agreed on the direction — this split helps — and would very likely agree on which of several candidate splits helps most, because both are just different ways of penalising a 50/50 mix more than a 90/10 one. They disagree only at the margins, which is why the table below calls the choice between them a rounding error next to the depth and leaf-size knobs.</p>

${H.table(['Criterion', 'Formula', 'Character'], [
      ['Gini', '$1-\\sum_k p_k^2$', 'Slightly faster; the sklearn default; prefers the largest class purity'],
      ['Entropy', '$-\\sum_k p_k \\log_2 p_k$', 'Information gain (§1.10); marginally more balanced splits, rarely different in practice'],
      ['Variance (regression)', '$\\frac1n\\sum (y-\\bar y)^2$', 'The squared-error criterion — this is what boosting’s trees fit'],
      ['MAE (regression)', '$\\frac1n\\sum|y-\\text{median}|$', 'Robust, slower, no closed-form leaf update']
    ])}

${H.analogy(`<p>Impurity is a jar-of-marbles question. A jar of 100 marbles, 60 red and 40 blue, is more "mixed" than a jar of 90 red and 10 blue, and both are more mixed than a jar that is all one colour. Gini and entropy are just two different rulers for that same mixedness, and both put the all-one-colour jar at zero and the 50/50 jar at the top.</p>
<p>A tree is a person sorting one big jar into smaller jars, over and over, always choosing the pour that leaves the resulting jars as unmixed as possible. Nobody tells the sorter to stop; left alone, they will keep pouring until every jar holds one marble each, which is a perfect but useless description of the original jar. That is exactly what an unconstrained tree does to training data, and it is why <code>max_depth</code> and <code>min_samples_leaf</code> exist — they tell the sorter when good enough is good enough.</p>`)}

<p>Trees pay for this greedy, question-at-a-time strategy with two real strengths and one real weakness. The strengths: they capture interactions automatically, because "utilisation ≤ 0.5 <i>and then</i> enquiries ≤ 2" is just a path through two splits, with no cross-term to engineer by hand; and they need no feature scaling at all, because a split only ever asks "is this value above a threshold", a question that could not care less whether the threshold is 0.5 or 500,000. The weakness is variance. Change a handful of rows near the top of the tree and the single best split there can flip to a different feature entirely, and because every split below inherits the region its parent carved out, one flipped decision near the root rewrites every leaf beneath it. A tree is not a slightly-wrong model of your data; it is a highly confident, easily-toppled one.</p>

${H.history(`<p>Two lineages converge on the same idea from different directions. Ross Quinlan's ID3 (1986) and its successor C4.5 grew out of machine learning research and used information gain — Shannon's entropy, imported wholesale. Leo Breiman, Jerome Friedman, Richard Olshen and Charles Stone's CART (1984) came out of statistics and used Gini impurity, chosen partly because it needs no logarithm and is marginally cheaper to compute at every candidate split on every feature. The two communities were solving the identical problem and arrived at two impurity measures that, as the worked example above shows, almost always rank splits the same way. CART also introduced cost-complexity pruning and, not incidentally, the regression tree — variance in place of a class-probability impurity — which is the version that gradient boosting (§2.8) builds on.</p>`)}

<p><b>What you are looking at.</b> The plot draws the decision boundary a real CART implementation grows on your chosen dataset, coloured by predicted class, alongside the actual points. Switching the view to "the tree" instead draws the same tree as a flowchart, root at the top, so you can see the literal sequence of questions the boundary encodes. The readout reports training accuracy, the number of leaves, and the depth actually reached — which can be less than the cap if the tree runs out of impurity to remove first.</p>

<p><b>What to do with it.</b> Start shallow, at depth 1 or 2, and watch the partition stay coarse and the accuracy readout stay well under 100%. Now push the depth slider up one notch at a time. The rectangles multiply, hugging individual points ever more tightly, and accuracy climbs — often all the way to 100% — while the boundary itself starts doing something no sane rule should do: carving out a single-point island around one noisy example. Switch to the XOR or circles datasets, where a diagonal or curved true boundary has to be approximated by a staircase of axis-aligned rectangles, and watch how many splits that staircase costs.</p>

<p><b>The thing genuinely worth noticing.</b> Press <i>Reseed</i> at a high depth and watch the partition change shape substantially even though the underlying data distribution has not — only the particular 90-odd sample points have. That instability, visible in real time, is the variance problem stated in words two paragraphs up, made concrete. It is also the reason the next subsection exists: if one tree is unreliable, average many of them.</p>

${H.lab('tree', 'Grow a tree, watch it partition the plane', 'A real CART implementation, live. Depth and leaf size control how far the sorter is allowed to keep pouring; watch training accuracy hit 100% while the shapes become obviously unreasonable — that is overfitting, visible as geometry rather than as a gap between two numbers.')}

<h2><span class="sn">2.7.2</span> Bagging and random forests</h2>

<p>If a single tree is a high-variance estimate, the fix suggested by §1.3 is not to build a better tree, it is to build many mediocre ones and average them. <b>Bagging</b> — bootstrap aggregating — draws $B$ bootstrap resamples of the training rows (sample $n$ rows with replacement from a dataset of $n$ rows), fits one full-depth tree to each, and averages their predictions. Averaging is the operation, so the variance formula §1.3 derived for averaging $B$ correlated models applies directly and without modification:</p>

$$\\mathrm{Var}\\!\\left(\\frac1B\\sum_b f_b\\right) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$$

<p>where $\\sigma^2$ is each tree's own variance and $\\rho$ is the correlation between any pair of them. §1.3 walks the derivation of this in full and puts real numbers on it — at $\\rho=0.6$, adding trees pulls the variance down towards a floor of $0.6\\sigma^2$ and stalls there — so it is not repeated here. What matters for a random forest is what that floor implies: bagging alone attacks only the second term, the one with $B$ in the denominator. However many trees you bag, the variance cannot fall below $\\rho\\sigma^2$, and bagged trees on the same rows, splitting on the same features, tend to be quite similar to one another — $\\rho$ around 0.5–0.7 is typical on tabular data.</p>

<p><b>Random forests</b> attack the floor directly. At every split, instead of considering every feature, consider only a random subset of them (typically $\\sqrt{d}$ for classification). This deliberately weakens each individual tree — the split it is forced to choose from a smaller menu is sometimes not the best one available — which raises $\\sigma^2$ a little. But it also means two trees are far less likely to make the identical sequence of choices, which drives $\\rho$ down a lot. §1.3's own worked numbers make the trade-off explicit: halving $\\rho$ from 0.6 to 0.3 at $B=100$ cuts the ensemble's variance from 0.604 to 0.307, a 49% reduction bought for free at training time, against the much smaller gain available from simply adding more trees at fixed $\\rho$. That asymmetry — decorrelation dominates volume — is the entire justification for random forests, and it is worth being able to reproduce from the formula rather than recite as a slogan.</p>

<p>Bootstrap sampling has a second, free benefit. Because each tree is trained on a resample rather than the full dataset, a chunk of rows is left out of any given tree's training set entirely — the <b>out-of-bag</b> rows for that tree. Predicting those left-out rows with the trees that never saw them gives a validation estimate without spending a single row on a held-out set.</p>

${H.deriv('why roughly 37% of rows are left out of each bootstrap', [
      ['$P(\\text{row } i \\text{ never drawn}) = \\left(1-\\tfrac1n\\right)^n$', 'A bootstrap draws $n$ rows with replacement from $n$ rows. Each individual draw misses row $i$ with probability $1-1/n$, and the $n$ draws are independent, so the probability every one of them misses row $i$ is that factor raised to the $n$-th power.'],
      ['$\\ln P = n\\ln\\!\\left(1-\\tfrac1n\\right)$', 'Take logs to turn the power into a product, which is easier to take a limit of.'],
      ['$\\ln\\!\\left(1-\\tfrac1n\\right) \\approx -\\tfrac1n$ for large $n$', 'The first-order Taylor approximation of $\\ln(1-x)$ near $x=0$ is $-x$ (§0.3’s linear approximation, applied here to $\\ln$). Since $1/n$ is small for any dataset worth bootstrapping, this is an excellent approximation.'],
      ['$n\\ln\\!\\left(1-\\tfrac1n\\right) \\to n\\times\\left(-\\tfrac1n\\right) = -1$', 'Substitute the approximation. The $n$ cancels exactly, leaving a constant that does not depend on $n$ at all — the reason the "roughly 37%" figure barely moves whether you have 500 rows or 5 million.'],
      ['$P \\to e^{-1} \\approx 0.368$', 'Undo the logarithm. At $n=1{,}000$ the exact value already agrees with $e^{-1}$ to three decimal places, which is how quickly this asymptotic result kicks in.']
    ], 'So on average 36.8% of rows are excluded from any one tree\'s training set — call it "about 37%" or "about $e^{-1}$" interchangeably. Averaging each row\'s prediction over only the trees that omitted it gives the out-of-bag estimate, which behaves like a free, nearly-unbiased validation score baked into the training procedure itself.')}

${H.note('Bagging attacks variance with independent parallel models. Boosting (§2.8) attacks bias with dependent sequential ones. Same ingredient — many weak trees — opposite recipe, and interviewers love this contrast because it tests whether you understand why each ensemble exists rather than merely that both use trees.')}

<p><b>What you are looking at.</b> Both a single deep tree and a $B$-tree forest are trained on the identical bootstrap-resampled data, with the single tree's boundary drawn as a thin amber contour line over the forest's shaded region. The readout separately reports the single tree's training accuracy, the ensemble's training accuracy, the out-of-bag estimate, and a qualitative read on "boundary instability" derived from how many trees are in the forest.</p>

<p><b>What to do with it.</b> Leave $B$ low, around 1–3, and press <i>Reseed the data</i> repeatedly. Both the amber line and the shaded ensemble region jump around on every press — with only a handful of trees, the "ensemble" barely differs from a single tree, because there has not been enough averaging yet to matter. Now raise $B$ to 50 or more and reseed again. The amber single-tree contour still lurches on every press, exactly as before, because nothing about a lone tree's variance has changed. But the shaded ensemble boundary now holds nearly still.</p>

<p><b>The thing genuinely worth noticing.</b> Toggle the feature-sampling control between "both" (plain bagging) and "random 1" (a forest) at a fixed, generous $B$. The ensemble boundary changes shape somewhat, but the more telling number is in the out-of-bag readout and in how much less it jitters between reseeds under the random-feature setting — that reduced jitter is $\\rho$ falling, made visible rather than asserted. You are watching the two terms of the §1.3 formula separately: raising $B$ shrinks the second term, and switching on feature sampling lowers the floor the second term is falling towards.</p>

${H.lab('forest', 'One tree versus a forest, on the same data', 'Both are trained here, live. Watch the single tree’s amber boundary jump every time you reseed the data while the forest’s shaded boundary barely moves — that stability <i>is</i> the $\\frac{1-\\rho}{B}\\sigma^2$ term shrinking, and switching on random feature sampling lowers the $\\rho\\sigma^2$ floor it is shrinking towards.')}

<h2><span class="sn">2.7.3</span> Reading a tree aloud</h2>

<p>None of the ensemble machinery above survives being explained to a lending committee, and this is precisely why a single shallow tree still has a job. Leaves hold predictions; edges hold conditions a person can read out loud without translation: <i>utilisation ≤ 0.62 → enquiries ≤ 2 → predicted default probability 0.03</i>. A logistic regression coefficient requires the listener to trust an abstraction — "a one-unit increase in standardised utilisation raises the log-odds by 0.4" — while a tree path is just a sentence in English with numbers in it. That readability is why a shallow, heavily-pruned tree still appears in regulated credit and healthcare settings as a challenger model, a segmentation device, or the human-readable explanation layered on top of a production ensemble that a regulator will never be shown directly.</p>

${H.probe([
      ['Why is a random forest better than plain bagging?', 'Feature subsampling decorrelates the trees, lowering ρ and therefore the variance floor ρσ² that more trees alone can never remove — see §1.3 for the algebra and this section for the numbers.'],
      ['What is out-of-bag error?', 'Each bootstrap omits about 37% of rows ($e^{-1}$, derived above); predicting those with only the trees that never saw them gives a free, nearly unbiased validation estimate.'],
      ['Gini or entropy?', 'They rarely disagree on which split to take, because both simply penalise a mixed node more than a lopsided one. Say that, then say the depth cap and min-samples-per-leaf matter far more to the model you end up with.']
    ], 'Reporting impurity-based feature importance as if it were unbiased — it favours high-cardinality and continuous features (§2.17), because they offer more candidate thresholds to search over and so are more likely to find a spuriously good split by chance alone.')}`,
    labs: {
      tree: function (host) {
        let data = Num.dataset('moons', 90, .35, 15);
        const st = Viz.controls(host, [
          { k: 'depth', label: 'max depth', min: 1, max: 8, step: 1, value: 3, fmt: v => v },
          { k: 'leaf', label: 'min samples per leaf', min: 1, max: 20, step: 1, value: 1, fmt: v => v },
          { k: 'crit', label: 'criterion', type: 'buttons', value: 'gini', options: [{ v: 'gini', t: 'Gini' }, { v: 'entropy', t: 'entropy' }] },
          { k: 'view', label: 'view', type: 'buttons', value: 'space', options: [{ v: 'space', t: 'partition' }, { v: 'tree', t: 'the tree' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'acc', label: 'training accuracy', cls: 'key' }, { k: 'leaves', label: 'leaves' }, { k: 'depth', label: 'depth reached' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const t = Num.tree(data.X, data.y, { maxDepth: st.depth, minLeaf: st.leaf, criterion: st.crit });
            let leaves = 0, maxd = 0;
            (function walk(n, d) { maxd = Math.max(maxd, d); if (!n.left) { leaves++; return; } walk(n.left, d + 1); walk(n.right, d + 1); })(t.root, 0);
            if (st.view === 'space') {
              const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
              P.clip(() => Labs.boundary(P, (x, y) => t.predict([x, y]), { step: 3 }));
              P.clip(() => Labs.points(P, data.X, data.y));
            } else {
              Labs.drawTree(ctx, t.root, w / 2, 16, w, T, ['x₁', 'x₂'], st.depth);
            }
            const acc = Num.mean(data.X.map((x, i) => ((t.predict(x) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            out({ acc: (acc * 100).toFixed(1) + '%', leaves: leaves, depth: maxd });
          }
        });
        Viz.buttons(host, [
          { label: 'Moons', on: () => { data = Num.dataset('moons', 90, .35, 15); S.redraw(); } },
          { label: 'XOR', on: () => { data = Num.dataset('xor', 90, .3, 5); S.redraw(); } },
          { label: 'Circles', on: () => { data = Num.dataset('circles', 90, .2, 8); S.redraw(); } },
          { label: 'Reseed', on: () => { data = Num.dataset('moons', 90, .35, Math.floor(Math.random() * 1000)); S.redraw(); } }
        ]);
        Viz.note(host, 'Note the boundaries are always axis-aligned staircases. A diagonal boundary needs many splits to approximate — which is exactly why trees love interactions and dislike rotations, and why PCA before a tree is usually a bad idea.');
      },

      forest: function (host) {
        let seed = 11;
        const st = Viz.controls(host, [
          { k: 'B', label: 'trees in the forest', min: 1, max: 80, step: 1, value: 30, fmt: v => v },
          { k: 'depth', label: 'depth of each tree', min: 1, max: 8, step: 1, value: 6, fmt: v => v },
          { k: 'feat', label: 'features per split (decorrelation)', type: 'buttons', value: '1', options: [{ v: '2', t: 'both (bagging)' }, { v: '1', t: 'random 1 (forest)' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'single', label: 'single-tree accuracy' }, { k: 'ens', label: 'ensemble accuracy', cls: 'good' },
          { k: 'oob', label: 'out-of-bag estimate', cls: 'key' }, { k: 'var', label: 'boundary instability' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const data = Num.dataset('moons', 120, .42, seed);
            const R = Num.rng(seed + 1);
            const trees = [], oobPred = data.X.map(() => []);
            for (let b = 0; b < st.B; b++) {
              const idx = data.X.map(() => R.int(data.X.length));
              const inbag = new Set(idx);
              const Xb = idx.map(i => data.X[i]), yb = idx.map(i => data.y[i]);
              const featMask = st.feat === '1' ? (R() < .5 ? 0 : 1) : null;
              const Xm = featMask === null ? Xb : Xb.map(p => featMask === 0 ? [p[0], p[1] * 0] : [p[0] * 0, p[1]]);
              const t = Num.tree(Xm, yb, { maxDepth: st.depth, minLeaf: 2 });
              t._mask = featMask;
              trees.push(t);
              data.X.forEach((p, i) => { if (!inbag.has(i)) oobPred[i].push(predOne(t, p)); });
            }
            function predOne(t, p) {
              const q = t._mask === null ? p : (t._mask === 0 ? [p[0], 0] : [0, p[1]]);
              return t.predict(q);
            }
            const ens = (x, y) => Num.mean(trees.map(t => predOne(t, [x, y])));
            const single = Num.tree(data.X, data.y, { maxDepth: st.depth, minLeaf: 2 });
            const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            P.clip(() => {
              Labs.boundary(P, ens, { step: 4 });
              P.contours((x, y) => single.predict([x, y]), [.5], { color: T.amber, width: 1.6, alpha: .9 });
            });
            P.clip(() => Labs.points(P, data.X, data.y));
            const accS = Num.mean(data.X.map((x, i) => ((single.predict(x) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            const accE = Num.mean(data.X.map((x, i) => ((ens(x[0], x[1]) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            const oobOK = oobPred.map((ps, i) => ps.length ? (((Num.mean(ps) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0) : null).filter(v => v !== null);
            out({
              single: (accS * 100).toFixed(1) + '%', ens: (accE * 100).toFixed(1) + '%',
              oob: oobOK.length ? (Num.mean(oobOK) * 100).toFixed(1) + '%' : '—',
              var: st.B < 5 ? 'high' : st.B < 20 ? 'moderate' : 'low'
            });
          }
        });
        Viz.buttons(host, [{ label: 'Reseed the data', primary: true, on: () => { seed = Math.floor(Math.random() * 10000); S.redraw(); } }]);
        Viz.legend(host, [{ c: Viz.theme().amber, t: 'single deep tree’s boundary' }, { c: Viz.theme().text, t: 'ensemble boundary' }]);
        Viz.note(host, 'Press <i>reseed</i> repeatedly with B = 1 and then with B = 50. The amber single-tree line lurches every time; the ensemble surface barely moves. That difference is $\\frac{1-\\rho}{B}\\sigma^2$ shrinking in front of you — and switching feature sampling on lowers ρ, dropping the floor as well.');
      }
    },
    quiz: [
      {
        q: 'Random forests differ from plain bagging mainly by…',
        options: ['using deeper trees', 'sampling a random subset of features at each split to decorrelate the trees', 'weighting misclassified points', 'using a lower learning rate'],
        answer: 1,
        why: 'Feature subsampling lowers ρ, the pairwise correlation between trees, which lowers the variance floor ρσ² in the §1.3 formula that adding more trees alone can never touch. "Deeper trees" is the tempting wrong answer because both bagging and forests do usually use deep, low-bias trees — but that choice is common to both methods, not the thing that distinguishes them, and depth alone does nothing about correlation between trees. Weighting misclassified points and lowering a learning rate are both boosting ideas (§2.8), smuggled in to test whether you keep the two families straight: bagging never reweights examples and has no learning rate, because there is no sequential fitting for one to control.'
      },
      {
        q: 'Out-of-bag error is available because…',
        options: ['trees are pruned', 'each bootstrap sample omits about 37% of the rows, which those trees never saw', 'forests use cross-validation internally', 'the trees are shallow'],
        answer: 1,
        why: '$(1-1/n)^n \\to e^{-1} \\approx 0.368$ as $n$ grows, derived above from a first-order log approximation — so on average about 37% of rows are absent from any one tree\'s bootstrap sample, and predicting a row using only the trees that never trained on it gives a nearly unbiased validation score for free. "Forests use cross-validation internally" is a plausible-sounding confusion, since both ideas produce a validation-style estimate without a held-out set, but OOB requires no explicit fold structure at all — it is a byproduct of bootstrap sampling that exists whether or not you asked for it. Pruning and shallow trees are unrelated: OOB rows exist regardless of how deep the trees are grown.'
      }
    ],
    cards: [
      { q: 'Bagging vs boosting, in one line', a: 'Bagging: independent parallel models averaged, attacks variance. Boosting: dependent sequential models added, attacks bias.' },
      { q: 'Out-of-bag fraction', a: '≈ 37% ($e^{-1}$) of rows are omitted by each bootstrap — a free validation estimate.' },
      { q: 'Impurity measures', a: 'Gini $1-\\sum p_k^2$, entropy $-\\sum p_k\\log p_k$, variance for regression — all ways to score how mixed a node is.' },
      { q: 'Why decorrelation beats volume', a: 'At B=100, halving ρ from 0.6 to 0.3 cuts variance 49% ($0.604\\to0.307$); adding trees alone saturates at the floor $\\rho\\sigma^2$.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.8 */
  ML.section({
    id: 'boosting', track: 'classical', num: '2.8',
    title: 'Gradient boosting, in depth',
    lede: 'The centrepiece of Part 2 and the tool you will be asked about most; rests on §1.9’s Newton step and §1.5’s loss derivation.',
    html: `
<p>A single tree, kept shallow so it does not memorise the training set, is not very accurate — that is the price of low variance from §2.7. Say a two-level tree gets the broad shape of the problem right but leaves a pattern in its mistakes: it predicts too low for one kind of example and too high for another. Nothing stops you from fitting a <i>second</i> shallow tree whose job is only to predict those leftover mistakes, and adding its output to the first tree's. If the leftovers still have a pattern, fit a third tree to what remains after that. Repeat enough times and you have traded one weak model for a committee of weak models, each cleaning up specifically what the ones before it got wrong. That iterative "predict, look at what's left over, fit the leftovers" loop is boosting, and the entire section is about making that loop precise enough to implement.</p>

<h2><span class="sn">2.8.1</span> The functional-gradient view</h2>
<p>Build the ensemble one tree at a time, $F_m(x) = F_{m-1}(x) + \\eta f_m(x)$: take whatever the ensemble predicts so far, $F_{m-1}$, and add a small multiple $\\eta$ of a new weak learner $f_m$. Read $\\eta$ (eta) as the same learning rate you already know from §0.3 and §1.9 — it controls how much of each new tree's opinion actually gets added, not how the tree itself is grown. The one design question is what $f_m$ should be fit to, and the answer that makes this gradient descent rather than a heuristic is: <b>the negative gradient of the loss, evaluated at the current predictions</b>.</p>

${H.analogy(`<p>Ordinary gradient descent (§0.3, §1.9) adjusts a fixed list of numbers — the parameters $\\theta$ — by stepping each one against its own partial derivative. Gradient boosting adjusts a <i>function</i> the same way: think of the current prediction $F_{m-1}(x_i)$ for every training row $i$ as one enormous vector of numbers, exactly $n$ of them, that you are free to move in any direction. The "gradient" of the loss with respect to that vector is computable one coordinate at a time, exactly like any other gradient, and stepping against it would reduce the loss — except you are not allowed to touch those $n$ numbers directly at prediction time on new data. So instead of taking the step, you train a small model, a tree, to <i>approximate</i> what that step would have been, and you get to reuse that approximation on inputs you have never seen. Boosting is gradient descent that has to earn a living by generalising.</p>`)}

<p>For squared-error loss $\\tfrac12(y-F)^2$, the derivative with respect to $F$ is $F-y$, so the negative gradient is $y-F$: the ordinary residual. This is where the folk explanation "each tree fits the previous errors" comes from, and it is not wrong, only incomplete — it is true for exactly one loss function. For logistic loss the negative gradient works out to $y - p$ where $p$ is the current predicted probability, a residual on the probability scale rather than the label scale, and for a different loss again — quantile loss, say, or Huber — the target each tree chases is different again. <mark>"Fit the residuals" is the squared-error special case of "fit the negative gradient", and the general statement is the one that survives changing the loss.</mark></p>

<h2><span class="sn">2.8.2</span> XGBoost's second-order objective, and the leaf weight as a Newton step</h2>
<p>Fitting a tree to the gradient alone throws away information you have already paid to compute. §1.9.3 built Newton's method precisely to use it: instead of a linear model of the loss, use a quadratic one, and read off the step that minimises the quadratic exactly rather than guessing a step length by hand. XGBoost applies exactly that idea, tree by tree. Taylor-expand the loss to second order around the current prediction, with $g_i$ the first derivative and $h_i$ the second derivative of the loss at row $i$ — precisely the $g$ and $H$ of §1.9.3, just computed per example rather than once for the whole parameter vector:</p>
$$\\text{obj}^{(t)} \\approx \\sum_i \\left[g_i f_t(x_i) + \\tfrac12 h_i f_t(x_i)^2\\right] + \\Omega(f_t), \\qquad \\Omega = \\gamma T + \\tfrac12\\lambda\\|w\\|^2$$
<p>Read the sum left to right: for every training row, a linear term that rewards moving the prediction in the direction the gradient wants, plus a quadratic term whose coefficient is the curvature $h_i$ — exactly the two terms of the second-order Taylor expansion from §1.9.1, added up over every row instead of expanded around one point. $\\Omega$ is new: a penalty of $\\gamma$ per leaf $T$ the tree is allowed to have, plus an $L2$ penalty $\\tfrac12\\lambda$ on the leaf values $w$ themselves, both there to stop the tree from chasing the training data too precisely.</p>

<p>A tree assigns one constant value to every row that lands in the same leaf, so the sum above can be regrouped leaf by leaf rather than row by row. Collect every row that falls into leaf $j$ and sum their gradients and curvatures into two single numbers, $G_j = \\sum_{i \\in j} g_i$ and $H_j = \\sum_{i \\in j} h_i$. Leaf $j$'s entire contribution to the objective is then $G_j w_j + \\tfrac12(H_j+\\lambda)w_j^2$ — an ordinary one-variable quadratic in the single unknown $w_j$, the value that leaf will predict.</p>

${H.deriv('the leaf weight as Newton\'s step, specialised to one leaf', [
      ['$\\theta \\leftarrow \\theta - H^{-1}g$', 'The general Newton step from §1.9.3: given a quadratic model of the loss with gradient $g$ and curvature $H$, this is the step that lands exactly on the quadratic\'s minimum.'],
      ['leaf $j$\'s local objective: $L_j(w) = G_j w + \\tfrac12(H_j+\\lambda) w^2$', 'Every row in leaf $j$ shares the same predicted value $w$, so the whole leaf behaves like a single scalar parameter. Summing the per-row linear and quadratic terms over the rows in the leaf gives exactly this one-variable quadratic — a gradient $G_j$ and a curvature $H_j$, aggregated from many rows into one number each.'],
      ['$\\dfrac{dL_j}{dw} = G_j + (H_j+\\lambda)w$', 'Differentiate the scalar quadratic in $w$, the one-dimensional case of the Taylor-model derivative worked in §1.9.3.'],
      ['set to zero: $w_j^* = -\\dfrac{G_j}{H_j+\\lambda}$', 'This is $-H^{-1}g$ from the first line, with $H$ replaced by the damped curvature $H_j+\\lambda$ rather than $H_j$ alone. §1.9\'s own caution about Newton\'s method — that a raw $H^{-1}$ can be unstable and needs damping — is answered here structurally: $\\lambda$ is added to the curvature before it is ever inverted, so a leaf with almost no curvature ($H_j \\approx 0$) still gets a bounded, sensible weight instead of one that explodes.']
    ], 'This is not an analogy to Newton\'s method; it is Newton\'s method, applied one scalar parameter at a time, once per leaf, with a ridge-style damping term built permanently into the curvature. Recognising $-G/(H+\\lambda)$ as $-H^{-1}g$ rather than an arbitrary boosting formula is precisely the distinction §1.9.3 flags as separating reciting from understanding.')}

<p>Substituting the optimal $w_j^*$ back into the objective gives a single number per leaf, the <b>structure score</b> $-\\tfrac12\\,G_j^2/(H_j+\\lambda) + \\gamma$, summed over all leaves — a way of scoring an entire tree shape without ever fitting the leaf values first and checking afterwards. A candidate split's <b>gain</b> is simply how much that score improves by turning one leaf into two:</p>
$$\\text{gain} = \\tfrac12\\left[\\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right] - \\gamma$$
<p>Read the bracket as: the score of the left child, plus the score of the right child, minus the score the parent already had. If splitting genuinely separates the gradients into two more homogeneous groups, the sum of the two child terms exceeds the single parent term and the bracket is positive; the $-\\gamma$ at the end then charges a fixed entry fee for the extra leaf the split creates, so a split has to clear that bar, not just be non-negative, to survive.</p>

${H.worked('worked split — by hand', `
<p>Logloss, so $g_i = p_i - y_i$ and $h_i = p_i(1-p_i)$. Start from $p = 0.5$ for everyone, so every $g = \\pm 0.5$ and every $h = 0.25$. Four samples with labels [1, 1, 0, 0]; the candidate split sends the two positives left and the two negatives right. Take $\\lambda = 1$, $\\gamma = 0$.</p>
${H.code(`LEFT            RIGHT           PARENT
G_L = −1.0      G_R = +1.0      G = 0
H_L =  0.5      H_R =  0.5      H = 1.0`)}
$$\\text{gain} = \\tfrac12\\left[\\tfrac{1}{1.5}+\\tfrac{1}{1.5}-\\tfrac{0}{2}\\right] = \\tfrac12(0.667+0.667) = 0.667 > 0$$
<p>The split is kept. Leaf weights are $w^* = -G/(H+\\lambda) = \\mp 0.667$ — positive leaf up, negative leaf down — then scaled by the learning rate $\\eta$ before being added to the ensemble. Note the parent term vanished because $G = 0$: <mark>a node whose gradients cancel is exactly a node with nothing left to learn.</mark></p>`)}

<p><b>What you are looking at.</b> Three panels — LEFT, RIGHT, PARENT — each showing that group's summed gradient $G$, summed curvature $H$, and (for the children) the resulting optimal leaf weight $w^*$. Underneath, the gain formula is spelled out with your current numbers substituted in, and a verdict in green or red states whether the split clears zero.</p>

<p><b>What to do with it.</b> Press "reproduce the worked example" to see the by-hand calculation above land exactly on the same 0.667. Then move $\\lambda$ up from 1 towards 10 and watch both leaf weights shrink towards zero and the gain fall with them — $\\lambda$ is damping the curvature in the denominator exactly as the derivation above describes, and a large enough $\\lambda$ can turn a genuinely useful split into one that no longer clears $\\gamma$. Move $p_0$, the current prediction before this tree, away from 0.5 towards 0.9 or 0.1 and watch the curvature term $h=p(1-p)$ shrink — logistic curvature is largest at maximum uncertainty and smallest when the model is already confident, so late trees in a well-fit ensemble see smaller Hessians and, other things equal, smaller leaf weights.</p>

<p><b>The thing genuinely worth noticing.</b> Press "a useless split (gradients cancel)" and watch $G_L$ and $G_R$ both land on numbers that make the parent's own $G$ zero-ish while the split still separates positives from negatives physically. Depending on exactly how the gradients balance, the gain can come out at or near zero even though the split "looks" informative by eye — because gain is a statement about gradients, not about label counts, and a split that reshuffles two already-balanced groups has nothing left to teach the next tree. This is the mechanism, not merely the assertion, behind the boxed line in the worked example above.</p>

${H.lab('gain', 'The split-gain calculator', 'Every quantity in the worked example, live and editable. Move the group sizes, λ and γ and watch the gain and both leaf weights respond in step with the formula — including the moment γ makes a genuinely separating split not worth taking, which is pre-pruning built directly into the score rather than applied afterwards.')}

<p>Zoom out from one split to the whole ensemble. Each new tree is grown to greedily maximise gain summed over its splits, its leaf values are fixed by $w^*_j$, and the whole tree is then shrunk by $\\eta$ and added to the running prediction — after which the gradients and curvatures for every row are recomputed against the new, slightly-improved predictions, and the process repeats for the next tree.</p>

<p><b>What you are looking at.</b> A one-dimensional regression problem, the true generating curve drawn as a dashed green line, the training points as faint dots, and the ensemble's current prediction as a solid blue curve built from however many trees you have added so far. Switching the view to residuals redraws the same x-axis but plots, per point, exactly what the <i>next</i> tree will be asked to fit — the leftover gap between the true label and the current prediction.</p>

<p><b>What to do with it.</b> Press "add a tree" once and watch the blue curve take one small step towards the green one; press it a dozen more times and watch it settle in. Switch to the residual view and repeat: the vertical lines shrink tree by tree, which is the "fit the leftovers" story from the top of this section made literal. Now push the learning rate up towards 1 and reset — the fit lurches on every added tree and starts fitting individual noisy points rather than the underlying sine; pull it down to 0.05 and the same number of trees barely dents the residuals, but the eventual fit, given enough trees, is visibly smoother.</p>

<p><b>The thing genuinely worth noticing.</b> At a very low learning rate, add trees ten at a time and watch how much steadier the residual pattern shrinks compared with a single large step at high $\\eta$ — small, damped steps average out the sampling noise in any one tree's gradient estimate the same way a small step size stabilises ordinary gradient descent near a ravine (§1.9.4). Slow and steady is not merely a metaphor here; it is the same conditioning argument, one abstraction layer up.</p>

${H.lab('boost', 'Boosting, one tree at a time', 'Press <i>add a tree</i> and watch the ensemble crawl toward the target while the residuals — literally, what the next tree will be fitted to — visibly shrink. Lower the learning rate and you need more trees for the same fit; that trade is the first row of the tuning table below.')}

<h2><span class="sn">2.8.3</span> The three libraries</h2>
<p><b>LightGBM</b> changes three things relative to the XGBoost recipe above: histogram binning of continuous features into a few hundred buckets before searching for splits (a large constant-factor speedup, since gain no longer has to be recomputed at every distinct value); <i>leaf-wise</i> best-first growth, which always expands whichever leaf in the whole tree currently offers the largest gain, instead of expanding every leaf at the current depth before going deeper — lower loss per tree for the same leaf budget, at the cost of trees that can grow unevenly deep and overfit if <code>num_leaves</code> and depth are left uncapped; and GOSS (Gradient-based One-Side Sampling) — keep every row with a large gradient, since those are the rows the model is still getting wrong, subsample the well-fit small-gradient rows, and reweight the sample to keep the gradient estimate unbiased. <b>CatBoost</b> changes two things: ordered target statistics, which compute a categorical feature's target-mean encoding using only the rows that appear earlier in a random permutation, so a row's own label can never leak into its own encoding — the principled version of the leakage §2.11 warns against by hand; and oblivious (symmetric) trees, which use the identical split condition at every node of a given level, producing a weaker individual tree but one that scores extremely fast and is strongly regularised by construction, since the same threshold is being reused across an entire level rather than tuned per node.</p>

${H.history(`<p>The lineage runs through three re-framings of the same idea. Freund and Schapire's AdaBoost (1995) reweighted misclassified examples after every round — a specific, clever heuristic that worked before anyone had a general theory for why. Jerome Friedman's Gradient Boosting Machine (1999, 2001) showed AdaBoost was a special case of fitting to a loss function's negative gradient, which is the §2.8.1 framing above and the reason the family is called "gradient" boosting at all — reweighting misclassified points and fitting negative gradients turn out to be the same operation under exponential loss. Tianqi Chen and Carlos Guestrin's XGBoost paper (2016) added the second-order objective this section derives, plus the systems engineering — sparsity-aware splitting, out-of-core computation, cache-aware access patterns — that made boosting fast enough for Kaggle and production alike. LightGBM (Microsoft, 2017) and CatBoost (Yandex, 2017) then optimised different bottlenecks in that same second-order recipe: LightGBM for raw throughput on very large tabular datasets, CatBoost for categorical features and leakage safety without manual encoding.</p>`)}

<h2><span class="sn">2.8.4</span> Tuning, in the order that matters</h2>
<p>Interviewers ask "how would you tune it" to find out whether you have actually done it. The answer is an ordered strategy, not a parameter list: fix a low learning rate and let early stopping choose the number of trees, then control complexity, then subsample, then regularise, and only then touch anything else.</p>
${H.table(['Knob', 'Start at', 'What it trades'], [
      ['<code>learning_rate</code>', '0.05', 'Lower is always better for accuracy and always slower. Halve it and roughly double the trees.'],
      ['<code>n_estimators</code>', 'large + early stop', 'Never tune by hand; let a validation window pick it (50-round patience).'],
      ['<code>max_depth</code> / <code>num_leaves</code>', '6 / 31', 'Interaction depth against variance. The single most important complexity knob.'],
      ['<code>min_child_weight</code>', '1 → raise', 'Minimum Hessian mass per leaf ($H_j$ in the derivation above). The cleanest fix for noisy, thin leaves.'],
      ['<code>subsample</code>, <code>colsample</code>', '0.8, 0.8', 'Adds decorrelation (§2.7) and speed; below ~0.5 you start losing signal.'],
      ['<code>reg_lambda</code> / <code>gamma</code>', '1 / 0', 'λ damps the curvature in $w^*$ (this section); γ is the minimum gain to split at all.'],
      ['<code>scale_pos_weight</code>', '1', 'Recall against calibration. Read §2.12 before touching it.']
    ])}

<h3>Monotonic constraints — say this unprompted in a credit interview</h3>
<p>All three libraries let you force a feature's effect to be monotone (<code>monotone_constraints</code>), so higher utilisation can never <i>lower</i> predicted risk. Mechanically this restricts which splits the gain search is even allowed to consider: a split that would make the prediction move the wrong way as the feature increases is simply excluded from the candidate set, whatever gain it would otherwise have scored. That buys three things at once: a model that agrees with domain knowledge, a defensible story for a validator, and free regularization — it removes exactly the wiggles that noise creates, since a non-monotone wiggle is, by construction, no longer an option the tree can express. Cost is a point or two of AUC, usually less. Pair it with <code>interaction_constraints</code> when a validator needs to see that two features never combine.</p>

<h3>One caution about early stopping</h3>
<p>The validation set that chose your tree count has been used for a decision, so it is no longer an unbiased estimate — that is what the nested scheme in §2.14 exists for. And under an out-of-time split, early stopping on a random validation fold will overshoot: stop on the OOT window instead, or you have tuned the number of trees to a distribution you will not be serving.</p>

${H.probe([
      ['Why second-order?', 'It is a Newton step (§1.9.3): curvature $h_i$, summed per leaf into $H_j$ and damped by λ, gives better leaf values and better split scores than the gradient alone, and it makes the gain formula loss-agnostic — swap in any twice-differentiable loss and $g_i$, $h_i$ change but the machinery does not.'],
      ['LightGBM vs XGBoost?', 'Histogram binning + leaf-wise growth + GOSS make LightGBM faster on large data, at the cost of more tuning (num_leaves, depth) to avoid overfitting from the uneven, best-first tree shapes leaf-wise growth produces.'],
      ['Where is pruning?', 'The −γ in the gain: a split must beat γ to be taken, so pre-pruning is built into the score rather than applied by trimming a fully-grown tree afterwards.']
    ], 'Running LightGBM leaf-wise with no <code>num_leaves</code> or depth cap, then blaming the library for overfitting — leaf-wise growth is more accurate per tree precisely because it is less constrained, and that same freedom is what runs away if you do not fence it in yourself.')}`,
    labs: {
      gain: function (host) {
        const st = Viz.controls(host, [
          { k: 'nL1', label: 'positives sent LEFT', min: 0, max: 6, step: 1, value: 2, fmt: v => v },
          { k: 'nL0', label: 'negatives sent LEFT', min: 0, max: 6, step: 1, value: 0, fmt: v => v },
          { k: 'nR1', label: 'positives sent RIGHT', min: 0, max: 6, step: 1, value: 0, fmt: v => v },
          { k: 'nR0', label: 'negatives sent RIGHT', min: 0, max: 6, step: 1, value: 2, fmt: v => v },
          { k: 'lam', label: 'λ (reg_lambda)', min: 0, max: 10, step: .25, value: 1, fmt: v => v.toFixed(2) },
          { k: 'gam', label: 'γ (min gain to split)', min: 0, max: 2, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'p0', label: 'current prediction p', min: .05, max: .95, step: .05, value: .5, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'gain', label: 'gain', cls: 'key' }, { k: 'wL', label: 'left leaf w*' },
          { k: 'wR', label: 'right leaf w*' }, { k: 'keep', label: 'split kept?' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const p = st.p0, g1 = p - 1, g0 = p - 0, hh = p * (1 - p);
            const GL = st.nL1 * g1 + st.nL0 * g0, HL = (st.nL1 + st.nL0) * hh;
            const GR = st.nR1 * g1 + st.nR0 * g0, HR = (st.nR1 + st.nR0) * hh;
            const term = (G, H) => (G * G) / (H + st.lam);
            const gain = .5 * (term(GL, HL) + term(GR, HR) - term(GL + GR, HL + HR)) - st.gam;
            const wL = -GL / (HL + st.lam), wR = -GR / (HR + st.lam);
            const box = (x, y, wd, ht, title, lines, col) => {
              Labs.roundRect(ctx, x, y, wd, ht, 8);
              ctx.fillStyle = T.panel; ctx.fill(); ctx.strokeStyle = col || T.line; ctx.lineWidth = 1.4; ctx.stroke();
              ctx.fillStyle = col || T.blue; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(title, x + 10, y + 8);
              ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace';
              lines.forEach((L, i) => ctx.fillText(L, x + 10, y + 26 + i * 17));
            };
            const bw = Math.min(150, (w - 60) / 3);
            box(14, 20, bw, 92, 'LEFT', ['G_L = ' + GL.toFixed(2), 'H_L = ' + HL.toFixed(2), 'w* = ' + (isFinite(wL) ? wL.toFixed(3) : '—')], T.blue);
            box(24 + bw, 20, bw, 92, 'RIGHT', ['G_R = ' + GR.toFixed(2), 'H_R = ' + HR.toFixed(2), 'w* = ' + (isFinite(wR) ? wR.toFixed(3) : '—')], T.red);
            box(34 + 2 * bw, 20, bw, 92, 'PARENT', ['G = ' + (GL + GR).toFixed(2), 'H = ' + (HL + HR).toFixed(2), 'score = ' + (-.5 * term(GL + GR, HL + HR)).toFixed(3)], T.muted);
            ctx.fillStyle = T.text; ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('gain = ½[ ' + term(GL, HL).toFixed(3) + ' + ' + term(GR, HR).toFixed(3) + ' − ' + term(GL + GR, HL + HR).toFixed(3) + ' ] − γ = ' + gain.toFixed(4), 14, 132);
            ctx.fillStyle = gain > 0 ? T.green : T.red; ctx.font = 'bold 14px ui-sans-serif';
            ctx.fillText(gain > 0 ? '→ split taken' : '→ split rejected (gain ≤ 0)', 14, 160);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.fillText('Leaf values are then scaled by the learning rate η before being added to the ensemble.', 14, 188);
            out({
              gain: gain.toFixed(4), wL: isFinite(wL) ? wL.toFixed(3) : '—',
              wR: isFinite(wR) ? wR.toFixed(3) : '—', keep: gain > 0 ? 'yes' : 'no'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Reproduce the worked example', primary: true, on: () => { st.$set('nL1', 2); st.$set('nL0', 0); st.$set('nR1', 0); st.$set('nR0', 2); st.$set('lam', 1); st.$set('gam', 0); st.$set('p0', .5); S.redraw(); } },
          { label: 'A useless split (gradients cancel)', on: () => { st.$set('nL1', 2); st.$set('nL0', 2); st.$set('nR1', 2); st.$set('nR0', 2); S.redraw(); } }
        ]);
      },

      boost: function (host) {
        const data = Num.regressionData(60, 'sine', .28, 12);
        const X = data.X.map(x => [x]);
        let model = Num.boosting(X, data.y, { lr: .3, maxDepth: 2 });
        const st = Viz.controls(host, [
          { k: 'lr', label: 'learning rate η', min: .02, max: 1, step: .02, value: .3, fmt: v => v.toFixed(2) },
          { k: 'depth', label: 'depth per tree', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'show', label: 'show', type: 'buttons', value: 'fit', options: [{ v: 'fit', t: 'the fit' }, { v: 'res', t: 'residuals the next tree sees' }] }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'trees', label: 'trees', cls: 'key' }, { k: 'mse', label: 'train MSE' }, { k: 'resid', label: 'mean |residual|' }
        ]);
        function reset() { model = Num.boosting(X, data.y, { lr: st.lr, maxDepth: st.depth }); S.redraw(); }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: st.show === 'fit' ? [-3, 3] : [-1.6, 1.6] })
              .frame({ xlabel: 'x', ylabel: st.show === 'fit' ? 'y' : 'residual y − F(x)' });
            if (st.show === 'fit') {
              P.clip(() => {
                P.dots(data.X.map((x, i) => [x, data.y[i]]), { r: 3.4, color: T.faint, alpha: .8 });
                P.fn(x => Math.sin(x * 1.4) * 1.6, { color: T.green, width: 1.6, dash: [6, 4] });
                P.fn(x => model.predict([x]), { color: T.blue, width: 2.8, n: 400 });
              });
            } else {
              P.clip(() => {
                P.hline(0, { color: T.faint, dash: [3, 3] });
                data.X.forEach((x, i) => {
                  const r = data.y[i] - model.predict([x]);
                  P.line([[x, 0], [x, r]], { color: r > 0 ? T.blue : T.red, width: 1.4 });
                  P.dots([[x, r]], { r: 3, color: r > 0 ? T.blue : T.red });
                });
              });
            }
            out({
              trees: model.trees.length, mse: model.loss().toFixed(4),
              resid: Num.mean(data.X.map((x, i) => Math.abs(data.y[i] - model.predict([x])))).toFixed(4)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Add a tree', primary: true, on: () => { model.addTree(); S.redraw(); } },
          { label: 'Add 10', on: () => { for (let i = 0; i < 10; i++) model.addTree(); S.redraw(); } },
          { label: 'Add 100', on: () => { for (let i = 0; i < 100; i++) model.addTree(); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        Viz.note(host, 'Switch to the residual view and step one tree at a time: each new stump is fitted to what is left over, and the leftovers shrink in a pattern you can watch. At η = 1 the fit lurches and starts chasing noise; at η = 0.05 it creeps but ends up smoother — the accuracy/patience trade in one control.');
      }
    },
    quiz: [
      {
        q: 'The optimal leaf weight in XGBoost is…',
        options: ['the mean of the residuals in the leaf', '$-G/(H+\\lambda)$', '$G/(H-\\lambda)$', 'the median of the labels'],
        answer: 1,
        why: 'Minimising the per-leaf quadratic $Gw + \\frac12(H+\\lambda)w^2$ by setting its derivative to zero gives $w^* = -G/(H+\\lambda)$ — the one-dimensional case of Newton\'s step $-H^{-1}g$ from §1.9.3, with the curvature damped by λ before it is inverted. "The mean of the residuals" is the tempting answer because it is exactly correct for squared-error loss with $\\lambda=0$ — there, $g_i = F(x_i)-y_i$ and $h_i=1$, so $-G/H$ collapses to the mean residual — but it silently assumes both a specific loss and no regularization, and the question is asking about the general XGBoost formula, which holds for any twice-differentiable loss. $G/(H-\\lambda)$ has the right shape and the wrong sign on λ: subtracting λ would make the leaf weight blow up as $H\\to\\lambda$, the opposite of what a damping term should do.'
      },
      {
        q: 'A candidate split has $G_L = -1$, $H_L = 0.5$, $G_R = +1$, $H_R = 0.5$, λ = 1, γ = 0. The gain is…',
        options: ['0', '0.667', '1.333', 'negative'],
        answer: 1,
        why: '½[1/1.5 + 1/1.5 − 0/2] = ½(1.333) = 0.667. The parent term vanishes because $G_L+G_R=0$: the gradients cancel exactly, so the parent leaf\'s own structure score is zero and the entire gain comes from the two children. "0" is the tempting wrong answer for exactly that reason — a test-taker who notices the parent term vanishing might assume there is nothing left to gain, when in fact a vanishing parent term is the best possible case for a split, not the worst: it means the split alone captured the full separation. This is the worked example in the derivation above, reproducible directly in the split-gain lab.'
      },
      {
        q: 'You must tune a boosted model with limited compute. The correct order is…',
        options: ['n_estimators, then learning_rate, then everything else', 'fix a low learning rate + early stopping, then depth/leaves, then subsampling, then regularization', 'gamma first, then lambda, then depth', 'tune all parameters jointly with grid search'],
        answer: 1,
        why: 'Tree count should never be tuned by hand — a validation window with early stopping picks it far more reliably than a grid search would; complexity (depth or num_leaves) is the dominant remaining knob because it sets how much interaction the model can express; subsampling and regularization are refinements layered on top of a complexity level that is already sensible. "Tune all parameters jointly with grid search" is the answer that sounds most careful and is the most wasteful: the parameters are not independent, so a joint grid explodes combinatorially, and most of that space is wasted exploring learning-rate/tree-count combinations that early stopping would have picked for free.'
      }
    ],
    cards: [
      { q: 'XGBoost leaf weight and gain', a: '$w^*=-G/(H+\\lambda)$; gain $=\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}]-\\gamma$.' },
      { q: 'Why second-order boosting?', a: 'The leaf weight is Newton\'s step $-H^{-1}g$ (§1.9.3) applied per leaf, with λ damping the curvature before inversion.' },
      { q: 'LightGBM’s three changes', a: 'Histogram binning, leaf-wise growth, GOSS. Faster on large data; needs num_leaves/depth caps.' },
      { q: 'CatBoost’s two changes', a: 'Ordered target statistics (prevents encoding leakage) and oblivious symmetric trees.' },
      { q: 'Monotonic constraints — why say it', a: 'Domain agreement + validator story + free regularization, at a cost of a point or two of AUC.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.9 */
  ML.section({
    id: 'unsupervised', track: 'classical', num: '2.9',
    title: 'Unsupervised: k-means, GMM/EM, DBSCAN, anomaly detection',
    lede: 'The EM derivation here is the same bound-tightening argument used for variational objectives generally — and the anomaly half is the part that pays for itself in fraud and AML.',
    html: `
<p>Every model in Parts 0 through 2.6 shared one assumption: someone handed you a label. A default flag, a price, a class. Now imagine you are handed a table of a million customer transactions with no such column — nobody has told you which customers are "similar", because nobody knows yet. All you have is the rows themselves and a distance between any two of them, courtesy of §0.2. The question this section answers is what you can possibly do with that alone, and the surprising answer is: quite a lot, provided you are honest about what you are actually optimising.</p>

<h2><span class="sn">2.9.1</span> k-means: clustering as an optimisation problem</h2>

<p>The simplest version of "group similar rows together" is to pick a number of groups $k$, guess a representative point — a <b>centroid</b> — for each one, and then judge the guess by how far every row sits from the centroid it has been assigned to. Squaring those distances and summing them gives the objective k-means minimises, the <b>within-cluster sum of squares</b>:</p>

$$\\text{WCSS} = \\sum_{j=1}^{k}\\sum_{x \\in C_j} \\|x - \\mu_j\\|^2$$

<p>Read it as: for each cluster $C_j$, add up the squared distance from every point in it to that cluster's centroid $\\mu_j$, then add the $k$ cluster totals together. A small WCSS means every point sits close to whatever centroid it was assigned, which is a reasonable formalisation of "the groups are tight". The catch is that minimising WCSS exactly, over every possible assignment of $n$ points to $k$ groups, is combinatorially hopeless for any real $n$ — so instead of solving it exactly, k-means solves it by alternation, a strategy called <b>Lloyd's algorithm</b>: freeze the centroids and optimise the assignments, then freeze the assignments and optimise the centroids, and repeat.</p>

${H.deriv('why each half of Lloyd\'s algorithm can only lower WCSS, never raise it', [
      ['<b>assignment step</b>: for each point $x$, choose the cluster $j$ minimising $\\|x-\\mu_j\\|^2$', 'This is the definition of "assign to the nearest centroid". Since every point independently picks whichever of the $k$ fixed centroids is closest to it, no point\'s own contribution to WCSS can be made worse by this step — the previous assignment is always still an available option, and this step only ever switches to something at least as good.'],
      ['<b>update step</b>: for each cluster, set $\\mu_j$ to the mean of the points currently assigned to it', 'For a fixed set of points, the single point that minimises the sum of squared distances to all of them is exactly their mean — differentiating $\\sum_x\\|x-\\mu\\|^2$ with respect to $\\mu$ and setting it to zero gives $\\mu = \\tfrac1{|C_j|}\\sum_{x\\in C_j} x$, the ordinary arithmetic mean. So re-centring on the mean can only lower or hold each cluster\'s contribution to WCSS, never raise it.'],
      ['WCSS after both steps $\\le$ WCSS before them', 'Both halves individually cannot increase the objective, so the pair cannot either. WCSS is bounded below by zero and decreases (weakly) every iteration, so it must converge — to <i>some</i> fixed point, though not necessarily the global optimum, since the sequence of alternations depends entirely on where it started.']
    ], 'This is exactly why the algorithm is guaranteed to stop rather than oscillate forever, and exactly why it can still stop somewhere bad: "monotonically decreasing and bounded below" only promises convergence to <i>a</i> local optimum, and Lloyd\'s alternation has no mechanism for noticing a better one elsewhere. That is what k-means++ is for — seed the centroids intelligently instead of leaving the search to find its own way out of a bad starting point.')}

${H.worked('one full Lloyd step, with the numbers', `<p>Four points: $(0,0)$, $(1,0)$, $(5,5)$, $(6,5)$. Start with two badly-placed centroids, $\\mu_1=(0,0)$ and $\\mu_2=(6,5)$ — deliberately sitting exactly on two of the data points rather than anywhere sensible.</p>
<p><b>Assign.</b> $(0,0)$ and $(1,0)$ are closer to $\\mu_1$; $(5,5)$ and $(6,5)$ are closer to $\\mu_2$. WCSS at these centroids: $0 + 1 + 1 + 0 = 2$.</p>
<p><b>Update.</b> The new centroids are the cluster means: $\\mu_1 = (0.5, 0)$, $\\mu_2 = (5.5, 5)$.</p>
<p>WCSS at the new centroids: $(0.25+0.25) + (0.25+0.25) = \\mathbf{1.0}$ — exactly half what it was, after a single assign-then-update pass, and never above the value it started at, exactly as the derivation above guarantees.</p>`)}

<p>k-means requires you to choose $k$ in advance (§2.9.4 comes back to how) and it implicitly assumes clusters that are roughly round and roughly equal in spread, because "nearest centroid" always partitions space into convex polygons — a <b>Voronoi diagram</b> — and a convex boundary simply cannot wrap around a crescent or a ring, however many iterations you run.</p>

<p><b>What you are looking at.</b> The plane, shaded into Voronoi regions by whichever centroid is currently nearest, with the data points coloured by their current assignment and the centroids drawn as larger ringed dots. The inset chart traces WCSS across every step taken so far — a line that, per the derivation above, only ever goes down or holds flat.</p>

<p><b>What to do with it.</b> Press "assign step" and "update step" alternately, one at a time, and watch the WCSS inset fall on every single press without exception. Then press "random restart" several times on the crescent-shaped dataset and watch how differently the algorithm partitions the same data depending purely on where the centroids happened to start — sometimes cleanly, sometimes visibly wrong, with one crescent's tip stolen by the other cluster.</p>

<p><b>The thing genuinely worth noticing.</b> Switch initialisation from "random" to "k-means++" and repeat the same run of restarts. Bad partitions become noticeably rarer, not because the alternation itself changed — it is identical Lloyd's-algorithm arithmetic either way — but because k-means++ spreads the <i>starting</i> centroids apart deliberately, biased towards picking a new seed far from the ones already chosen, so the search begins closer to a sensible answer instead of gambling on it.</p>

${H.lab('kmeans', 'Lloyd’s algorithm, step by step', 'Press <i>assign</i> and <i>update</i> alternately and watch WCSS fall monotonically — never up, exactly as the derivation guarantees. Then press <i>random restart</i> a few times on the crescents: the local optima are real, and switching to k-means++ visibly reduces how often you land in a bad one.')}

<h2><span class="sn">2.9.2</span> GMM and EM: clustering with a probability attached</h2>

<p>k-means makes a hard choice for every point — it belongs to exactly one cluster — and it never says how confident that choice was. A point sitting exactly between two centroids gets assigned to one of them with total, unearned certainty. A <b>Gaussian mixture model</b> replaces the hard assignment with a soft one: every point gets a probability of belonging to each of $k$ Gaussian components, each with its own mean <i>and</i> its own covariance, so clusters can be elongated or tilted rather than forced into k-means' round, equally-sized Voronoi cells.</p>

<p>Fitting a mixture by maximum likelihood runs immediately into a wall. The likelihood of one point is a mixture — a weighted sum over which component it might have come from — and you need $\\log p(x) = \\log\\sum_z p(x,z)$, where $z$ is the unobserved label saying which component generated $x$. The sum sits stubbornly inside the logarithm, and unlike a product inside a log, a sum inside a log does not simplify into anything you can differentiate cleanly and solve in closed form.</p>

${H.deriv('the ELBO: turning an intractable log-sum into a tractable lower bound', [
      ['$\\log p(x) = \\log\\sum_z p(x,z)$', 'The quantity you actually want to maximise: the log-likelihood of the observed data, marginalising over the unobserved component label $z$.'],
      ['$= \\log\\sum_z q(z)\\,\\dfrac{p(x,z)}{q(z)}$', 'Multiply and divide by an arbitrary distribution $q(z)$ over the same unobserved variable — legal for any $q$ that is nowhere zero where $p(x,z)$ is not, and chosen purely to set up the next step. Nothing about the value has changed yet.'],
      ['$= \\log\\,\\mathbb{E}_q\\!\\left[\\dfrac{p(x,z)}{q(z)}\\right]$', 'Recognise the sum, weighted by $q(z)$, as an expectation under $q$ — the definition of $\\mathbb{E}_q[\\cdot]$ from §1.2.'],
      ['$\\ge \\mathbb{E}_q[\\log p(x,z)] - \\mathbb{E}_q[\\log q(z)]$', 'Apply Jensen\'s inequality: $\\log$ is concave, so the log of an average is at least the average of the logs — $\\log\\mathbb{E}[Y] \\ge \\mathbb{E}[\\log Y]$ for any positive random variable $Y$. Here $Y = p(x,z)/q(z)$, and splitting $\\log(p(x,z)/q(z))$ into $\\log p(x,z) - \\log q(z)$ inside the expectation gives the two terms on the right.']
    ], 'The right-hand side is the ELBO — the Evidence Lower BOund, $\\mathcal{L}(q,\\theta)$ — and it holds for <i>any</i> choice of $q(z)$, which is exactly what makes it useful: you have replaced one hard problem (maximise the true log-likelihood directly) with a family of easier ones (maximise a lower bound, for a $q$ you get to pick).')}

<p>The gap between $\\log p(x)$ and the ELBO is exactly $D_{KL}(q\\,\\|\\,p(z\\mid x))$, the KL divergence from §1.10 applied to the true posterior over the hidden label. So the <b>E-step</b> sets $q = p(z\\mid x)$ — computing each point's <b>responsibilities</b>, the posterior probability it belongs to each component given the current parameters — which drives that KL gap to exactly zero and makes the bound touch the true likelihood. The <b>M-step</b> then maximises the now-tight bound over the parameters $\\theta$, which is easy because the awkward log-of-a-sum has become, inside the expectation, a sum of logs. Because the E-step tightens the bound to the true likelihood and the M-step can only raise a tight bound, the true likelihood itself can never fall across a full E–M cycle. <mark>"Tighten the bound, then maximise it" is the whole of EM in five words.</mark></p>

${H.intuition(`<p>k-means is what EM looks like in the limit of total confidence. Shrink every Gaussian component in a GMM down to a fixed, tiny, identical spherical covariance and let it approach zero: the E-step's soft responsibilities collapse towards 0 or 1, since the component nearest a point completely dominates the others once the covariance is small enough to matter. A responsibility that can only be 0 or 1 <i>is</i> a hard assignment, and the M-step's update — the weighted mean, with weights that are now just 0s and 1s — <i>is</i> exactly "move each centroid to the mean of its assigned points". k-means is not a different algorithm standing beside EM; it is the special case of EM where every component shares one fixed, spherical, shrinking-to-a-point covariance. That is also why k-means cannot express elongated clusters and a full GMM can — the covariance k-means implicitly assumes was never allowed to be anything but a sphere.</p>`)}

<p><b>What you are looking at.</b> A histogram of one-dimensional data with the current mixture density overlaid: each component's individual bell curve dashed in its own colour, and their weighted sum — the model's actual density estimate — drawn as a solid blue curve. Vertical dashed lines mark each component's current mean.</p>

<p><b>What to do with it.</b> Press "E-step" once and note the log-likelihood readout appear; press "M-step" and watch the dashed component curves shift to better straddle the data they are currently responsible for. Run twenty rounds at once and watch the mixture settle: the blue sum curve tightens around the two visible bumps in the histogram, and the log-likelihood readout climbs on every round without exception.</p>

<p><b>The thing genuinely worth noticing.</b> Reset with the true separation between the two underlying groups set low, so the histogram shows one blurred hump rather than two distinct ones, and run EM to convergence. The two fitted components frequently end up nearly on top of each other rather than splitting the data cleanly — there was not enough separation in the data for the likelihood surface to prefer two distinct components over one component counted twice. EM converging is a statement about the optimisation; it is not a promise that what it converged to is the clustering you were hoping to find.</p>

${H.lab('em', 'EM on a one-dimensional mixture', 'Watch responsibilities (the E-step, closing the KL gap) and parameter updates (the M-step, raising the now-tight bound) alternate, with the log-likelihood printed each round. It never goes down — that is the guarantee the ELBO derivation above buys, not a property of this particular dataset.')}

<h2><span class="sn">2.9.3</span> DBSCAN and hierarchical clustering</h2>

<p>Both k-means and GMM share a structural limit: every point must belong to some cluster, and every cluster is, at heart, a blob around a centre. Neither can say "this point does not belong anywhere" and neither can natively trace a shape like a crescent or a ring, because centroid-based membership only ever produces convex regions.</p>

${H.analogy(`<p>Think about how you would mark neighbourhoods on a map of a city using nothing but population density. A neighbourhood is wherever houses are packed closely enough together that you can walk from one to the next without a long empty stretch in between; keep walking through dense areas and the neighbourhood's boundary grows organically, wrapping around a river bend or a park however it needs to. A lone farmhouse three miles from anything is not "the smallest neighbourhood" — it is not a neighbourhood at all, and a sensible map marks it as empty countryside rather than forcing it into the nearest town.</p>
<p>That is <b>DBSCAN</b>: define "close enough to keep walking" as being within a radius <code>eps</code> of at least <code>minPts</code> other points, and grow clusters by chaining together everything reachable that way. A point with too few neighbours within <code>eps</code> to start or extend a chain is labelled <b>noise</b> rather than assigned to whichever cluster happens to be nearest — the algorithm is explicitly allowed to say "I don't know" about a point, which neither k-means nor a GMM can do.</p>`)}

<p>The strength this buys is real: DBSCAN traces clusters of essentially any shape and hands you an explicit noise label for the outliers, which is often exactly what a fraud or anomaly workflow wants. The weakness is the same parameter that makes it work: <code>eps</code> is one fixed radius for the whole dataset, so DBSCAN struggles the moment clusters genuinely differ in density — a radius tuned for a sparse cluster swallows a dense one whole, and a radius tuned for the dense one calls the sparse one entirely noise. HDBSCAN generalises the radius into something adaptive per region precisely to fix this.</p>

<p><b>Hierarchical</b> clustering sidesteps choosing $k$ a different way: start with every point as its own cluster, repeatedly merge the two closest clusters, and record every merge in order. The result is a <b>dendrogram</b> — a tree of merges you can cut at any height to get any number of clusters after the fact, rather than having committed to $k$ before seeing the data. What "closest" means between two whole clusters, not two points, is the <b>linkage</b> choice: single linkage (closest pair of points) chains through narrow bridges and can produce long, straggly clusters; complete linkage (farthest pair) produces tight, compact ones; average and Ward's linkage sit between the two, with Ward specifically minimising the increase in within-cluster variance at each merge, which makes it the linkage most similar in spirit to k-means' own objective.</p>

<p><b>What you are looking at.</b> The same crescent-shaped dataset run through k-means on one setting and DBSCAN on the other, so the comparison is apples to apples. k-means draws its centroids as ringed dots; DBSCAN marks noise points with an × instead of colouring them into any cluster.</p>

<p><b>What to do with it.</b> Run k-means on the crescents first. It reliably slices each crescent in half along a straight boundary — sometimes putting the top of one crescent with the bottom of the other — because a Voronoi cell simply cannot bend around a curve. Switch to DBSCAN with the default <code>eps</code> and <code>minPts</code> and the two crescents come back intact, correctly separated, because density-based growth follows the curve of the data rather than cutting straight lines through it.</p>

<p><b>The thing genuinely worth noticing.</b> Push <code>eps</code> down in small steps from its default. At some point the two crescents each fracture into several smaller fragments plus a scatter of × marks, because the neighbourhood radius has become too small to bridge naturally sparser patches within a single true cluster — DBSCAN's single global density threshold cannot always fit every region of the data. That fragility is the honest cost of the flexibility above, and it is exactly what HDBSCAN exists to remove.</p>

${H.lab('dbscan', 'Where k-means fails and density-based clustering does not', 'The same crescents through both algorithms, computed live. k-means slices them in half — it can only produce convex Voronoi cells. DBSCAN follows the curve and gets to say "noise" for anything that does not belong.')}

<h2><span class="sn">2.9.4</span> Choosing k without fooling yourself</h2>

<p>Every method above except DBSCAN needs $k$ handed to it, which pushes the real question one level up: how do you choose $k$ without simply picking the number that flatters your existing intuition? Three diagnostics, in increasing order of honesty.</p>

<p>The <b>elbow method</b> plots WCSS against $k$ and looks for where the curve visibly bends. It is quick, and it is frequently ambiguous, because WCSS decreases monotonically as $k$ grows by sheer construction — more clusters can only ever tighten the fit, all the way to WCSS $=0$ at $k=n$, one cluster per point — so "the bend" is often a matter of squinting rather than a sharp, unambiguous feature of the curve.</p>

<p>The <b>silhouette</b> score fixes the ambiguity by giving each point a bounded, interpretable number instead of contributing to one global curve. For a point with mean distance $a$ to others in its own cluster and mean distance $b$ to the points of the nearest <i>other</i> cluster, the silhouette is $(b-a)/\\max(a,b)$, which lands in $[-1, 1]$: close to 1 when a point sits deep inside its own cluster and far from the next one, near 0 when it sits right on a boundary between two clusters, and negative when it is, on average, closer to a different cluster than to its own — a sign the clustering itself made a mistake, not merely an ambiguous case.</p>

${H.worked('a silhouette score, computed by hand', `<p>Two tiny clusters: $A = \\{(0,0), (0,1)\\}$, $B = \\{(4,0), (4,1)\\}$. Take the point $(0,0)$.</p>
<p>$a$ = its distance to the only other point in its own cluster: $\\mathrm{dist}((0,0),(0,1)) = 1$.</p>
<p>$b$ = its mean distance to the points of the nearest other cluster: $\\tfrac12\\big(\\mathrm{dist}((0,0),(4,0)) + \\mathrm{dist}((0,0),(4,1))\\big) = \\tfrac12(4 + 4.123) = 4.062$.</p>
<p>$s = (b-a)/\\max(a,b) = (4.062-1)/4.062 = \\mathbf{0.754}$ — close to 1, correctly reporting that these two clusters are cleanly separated relative to how tight each one is internally.</p>`)}

<p>The <b>gap statistic</b> asks a more fundamental question than either of the above: is there any cluster structure here at all? It compares your actual WCSS at a given $k$ against the WCSS you would get clustering the same number of points drawn <i>uniformly at random</i> with no structure whatsoever, over many random draws. It is the only one of the three that can meaningfully answer $k=1$ — that the data has no real clusters, and any partition you drew would be splitting noise. That is the question to ask first, because a great deal of published clustering work skips it and reports structure that a uniform random cloud would have "found" just as confidently.</p>

<h2><span class="sn">2.9.5</span> Anomaly detection — the half that pays for itself</h2>
${H.table(['Method', 'Definition of "unusual"', 'Notes'], [
      ['<b>Isolation Forest</b>', 'Few random splits isolate the point', 'Fast, high-dimensional, almost no tuning beyond contamination — the default'],
      ['<b>One-class SVM</b>', 'Outside a boundary fitted around the bulk', 'RBF kernel (§2.6); sensitive to scaling and to ν'],
      ['<b>Local Outlier Factor</b>', 'Local density much lower than neighbours’', 'Catches locally-anomalous points: a normal-looking transaction in an abnormal neighbourhood'],
      ['<b>Autoencoder</b>', 'High reconstruction error', 'Right when the "normal" manifold is genuinely nonlinear and you have a lot of it']
    ])}
<p>Isolation Forest deserves a sentence on the intuition, since its name undersells it: build ordinary random trees exactly as in §2.7, but instead of splitting to reduce impurity, split on a uniformly random feature at a uniformly random threshold, and keep going until every point sits alone in its own leaf. An anomalous point — one that already looks different along some feature — tends to get separated from the bulk of the data in very few splits, because it rarely needs to be distinguished from many similar neighbours first. A typical point, surrounded by near-duplicates, takes many more splits to isolate. Average the depth needed to isolate a point across many random trees, and a short average depth is the anomaly score.</p>
<p>The hard part with any of these four is not fitting the model, it is evaluating it without labels for what "anomalous" actually meant in this dataset. Three practical moves: score every row, rank them, and have investigators review only the top-$k$, which both catches the highest-value cases immediately and starts building a genuinely labelled set for later; measure <b>precision@k</b> at the review capacity you actually have, since nobody can investigate 5,000 daily alerts regardless of how the model scores the other 4,995; and watch stability over time, because an anomaly detector whose alert volume triples overnight has almost always detected a change in the input pipeline, not a sudden crime wave. Once a few thousand labels exist from that review process, a supervised model trained on them will almost always beat the original unsupervised score on the metric that matters — <mark>treat anomaly detection as a bootstrap into supervision, not a destination.</mark></p>

${H.probe([
      ['Derive EM in one sentence.', 'Jensen\'s inequality on the log-sum gives an ELBO whose gap to the true likelihood is $D_{KL}(q\\|p(z|x))$; the E-step sets $q=p(z|x)$ to close the gap, the M-step raises the now-tight bound, so the true likelihood never decreases across a cycle.'],
      ['Why does k-means fail on crescents?', 'Nearest-centroid assignment always partitions space into convex Voronoi cells, and a crescent is not a convex shape — no straight-line boundary can separate it from its neighbour without cutting through it.'],
      ['How do you choose k?', 'Silhouette over elbow, because it gives a bounded per-point score with a meaningful zero rather than a curve you have to eyeball for a bend; reach for the gap statistic first if you genuinely need to test k = 1, i.e. whether there is any structure at all.'],
      ['What is k-means, in terms of EM?', 'The limiting case where every Gaussian component shares one fixed, spherical covariance shrinking towards zero — responsibilities collapse to hard 0/1 assignments and the M-step becomes "move each centroid to its cluster\'s mean".']
    ], 'Clustering unscaled features. k-means and DBSCAN both minimise or threshold Euclidean distance, so an income column measured in pounds will dominate a utilisation ratio measured in fractions of one, and you will have clustered on income alone without ever deciding to.')}`,
    labs: {
      kmeans: function (host) {
        let data = Num.dataset('blobs', 180, .55, 3);
        let km = null, iters = 0, hist = [];
        const st = Viz.controls(host, [
          { k: 'k', label: 'k (clusters)', min: 2, max: 6, step: 1, value: 3, fmt: v => v },
          { k: 'init', label: 'initialisation', type: 'buttons', value: 'pp', options: [{ v: 'rand', t: 'random' }, { v: 'pp', t: 'k-means++' }] }
        ], restart);
        const out = Viz.readout(host, [
          { k: 'wcss', label: 'WCSS (objective)', cls: 'key' }, { k: 'it', label: 'iterations' }, { k: 'sil', label: 'silhouette' }
        ]);
        function restart(seed) {
          km = Num.kmeans(data.X, st.k, typeof seed === 'number' ? seed : Math.floor(Math.random() * 1e5), st.init === 'pp');
          km.assignStep(); iters = 0; hist = [km.wcss()]; S.redraw();
        }
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4.5, 4.5], yd: [-3.4, 3.4] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            const cols = Labs.palette(st.k);
            P.clip(() => {
              // Voronoi shading
              P.field((x, y) => {
                let bi = 0, bd = 1e9;
                km.centers.forEach((c, j) => { const d = (x - c[0]) ** 2 + (y - c[1]) ** 2; if (d < bd) { bd = d; bi = j; } });
                return bi / Math.max(1, st.k - 1);
              }, {
                step: 5, lo: 0, hi: 1,
                colors: t => {
                  const idx = Math.round(t * (st.k - 1));
                  const c = cols[idx] || cols[0];
                  const r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
                  return [r, g, b, 34];
                }
              });
              data.X.forEach((p, i) => P.dots([p], { r: 3.2, color: cols[km.assign[i]], alpha: .95 }));
              km.centers.forEach((c, j) => {
                P.dots([c], { r: 8, color: cols[j], stroke: true, strokeWidth: 2.5 });
                P.dots([c], { r: 2.5, color: '#fff' });
              });
            });
            // objective inset
            if (hist.length > 1) {
              const ix = w - 150, iy = 16, iw = 130, ih = 48;
              ctx.fillStyle = T.paper; ctx.globalAlpha = .9; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
              ctx.strokeStyle = T.line; ctx.strokeRect(ix, iy, iw, ih);
              const mx = hist[0], mn = Math.min.apply(null, hist);
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6; ctx.beginPath();
              hist.forEach((v, i) => {
                const X = ix + iw * i / Math.max(1, hist.length - 1);
                const Y = iy + ih - ih * ((v - mn) / ((mx - mn) || 1)) * .8 - 4;
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              });
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('WCSS — monotone', ix + 4, iy + 3);
            }
            // silhouette
            let sil = 0;
            const sample = data.X.slice(0, 80);
            sample.forEach((p, i) => {
              const own = km.assign[i];
              const dists = {};
              data.X.forEach((q, j) => {
                if (i === j) return;
                const c = km.assign[j], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
                dists[c] = dists[c] || []; dists[c].push(d);
              });
              const a = dists[own] ? Num.mean(dists[own]) : 0;
              let b = 1e9;
              Object.keys(dists).forEach(c => { if (+c !== own) b = Math.min(b, Num.mean(dists[c])); });
              sil += (b - a) / Math.max(a, b);
            });
            out({ wcss: km.wcss().toFixed(1), it: iters, sil: (sil / sample.length).toFixed(3) });
          }
        });
        Viz.buttons(host, [
          { label: 'Assign step', primary: true, on: () => { km.assignStep(); hist.push(km.wcss()); S.redraw(); } },
          { label: 'Update step', on: () => { km.updateStep(); iters++; hist.push(km.wcss()); S.redraw(); } },
          { label: 'Run to convergence', on: () => { for (let i = 0; i < 40; i++) { const ch = km.assignStep(); km.updateStep(); iters++; hist.push(km.wcss()); if (!ch) break; } S.redraw(); } },
          { label: 'Random restart', on: () => restart() },
          { label: 'Crescents', on: () => { data = Num.dataset('moons', 180, .3, 4); restart(); } },
          { label: 'Blobs', on: () => { data = Num.dataset('blobs', 180, .55, 3); restart(); } }
        ]);
        restart(3);
      },

      em: function (host) {
        let xs = [], g = null, ll = [];
        const st = Viz.controls(host, [
          { k: 'k', label: 'components', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'sep', label: 'true separation', min: .5, max: 6, step: .1, value: 3, fmt: v => v.toFixed(1) },
          { k: 'n', label: 'samples', min: 60, max: 800, step: 20, value: 250, fmt: v => v }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'll', label: 'log-likelihood', cls: 'key' }, { k: 'it', label: 'EM rounds' },
          { k: 'mu', label: 'means' }, { k: 'pi', label: 'weights' }
        ]);
        function reset() {
          const R = Num.rng(29);
          xs = [];
          for (let i = 0; i < st.n; i++) xs.push(R() < .45 ? R.normal(-st.sep / 2, .8) : R.normal(st.sep / 2, 1.1));
          g = Num.gmm1d(xs, st.k, 7); ll = []; S.redraw();
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const hh = Num.hist(xs, 40, -8, 8);
            const dens = hh.bins.map(c => c / (xs.length * hh.w));
            const mix = x => Num.sum(g.mu.map((m, j) => g.pi[j] * Num.normPdf(x, m, g.sg[j])));
            const P = Viz.plot(ctx, w, h, { xd: [-8, 8], yd: [0, Math.max(Math.max.apply(null, dens), .4) * 1.25] })
              .frame({ xlabel: 'x', ylabel: 'density' });
            P.clip(() => {
              hh.centers.forEach((c, i) => {
                ctx.fillStyle = T.faint; ctx.globalAlpha = .35;
                const x0 = P.x(c - hh.w / 2), x1 = P.x(c + hh.w / 2);
                ctx.fillRect(x0, P.y(dens[i]), Math.max(1, x1 - x0 - 1), P.y(0) - P.y(dens[i]));
                ctx.globalAlpha = 1;
              });
              const cols = Labs.palette(st.k);
              g.mu.forEach((m, j) => P.fn(x => g.pi[j] * Num.normPdf(x, m, g.sg[j]), { color: cols[j], width: 1.8, dash: [5, 3] }));
              P.fn(mix, { color: T.blue, width: 2.8 });
              g.mu.forEach((m, j) => P.vline(m, { color: cols[j], dash: [2, 3], width: 1 }));
            });
            out({
              ll: ll.length ? ll[ll.length - 1].toFixed(4) : '—', it: ll.length,
              mu: '[' + g.mu.map(v => v.toFixed(2)).join(', ') + ']',
              pi: '[' + g.pi.map(v => v.toFixed(2)).join(', ') + ']'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'E-step', primary: true, on: () => { ll.push(g.eStep()); S.redraw(); } },
          { label: 'M-step', on: () => { g.mStep(); S.redraw(); } },
          { label: 'Run 20 rounds', on: () => { for (let i = 0; i < 20; i++) { ll.push(g.eStep()); g.mStep(); } S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        reset();
        Viz.note(host, 'Watch the log-likelihood readout after every round: it never decreases. That monotonicity is not luck — it follows directly from the ELBO argument, and it is the property that makes EM safe to run to convergence.');
      },

      dbscan: function (host) {
        let data = Num.dataset('moons', 200, .18, 6);
        const st = Viz.controls(host, [
          { k: 'algo', label: 'algorithm', type: 'buttons', value: 'db', options: [{ v: 'km', t: 'k-means' }, { v: 'db', t: 'DBSCAN' }] },
          { k: 'k', label: 'k (for k-means)', min: 2, max: 6, step: 1, value: 2, fmt: v => v },
          { k: 'eps', label: 'eps (DBSCAN radius)', min: .08, max: 1.2, step: .02, value: .34, fmt: v => v.toFixed(2) },
          { k: 'min', label: 'minPts', min: 2, max: 20, step: 1, value: 5, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'clusters', label: 'clusters found', cls: 'key' }, { k: 'noise', label: 'points labelled noise' }, { k: 'shape', label: 'recovers the crescents?' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            let labels, k, noise = 0;
            if (st.algo === 'km') {
              const km = Num.kmeans(data.X, st.k, 5, true);
              for (let i = 0; i < 40; i++) { const ch = km.assignStep(); km.updateStep(); if (!ch) break; }
              km.assignStep();
              labels = km.assign; k = st.k;
              P.clip(() => {
                const cols = Labs.palette(k);
                km.centers.forEach((c, j) => P.dots([c], { r: 8, color: cols[j], stroke: true, strokeWidth: 2.5 }));
              });
            } else {
              const db = Num.dbscan(data.X, st.eps, st.min);
              labels = db.labels; k = db.k; noise = labels.filter(l => l === -1).length;
            }
            const cols = Labs.palette(Math.max(1, k));
            P.clip(() => {
              data.X.forEach((p, i) => {
                const l = labels[i];
                P.dots([p], { r: 3.4, color: l < 0 ? 'transparent' : cols[l % cols.length], stroke: l < 0 ? T.faint : true, strokeWidth: 1.4 });
                if (l < 0) { P.text(p[0], p[1], '×', { color: T.faint, font: '12px ui-monospace', align: 'center' }); }
              });
            });
            out({
              clusters: k, noise: st.algo === 'db' ? noise : '—',
              shape: st.algo === 'db' && k === 2 && noise < data.X.length * .25 ? 'yes' : (st.algo === 'km' ? 'no — sliced in half' : 'tune eps/minPts')
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Crescents', on: () => { data = Num.dataset('moons', 200, .18, 6); S.redraw(); } },
          { label: 'Blobs', on: () => { data = Num.dataset('blobs', 200, .5, 2); S.redraw(); } },
          { label: 'Circles', on: () => { data = Num.dataset('circles', 200, .12, 9); S.redraw(); } }
        ]);
        Viz.note(host, '× marks noise. DBSCAN’s ability to <i>refuse to cluster</i> a point is the feature — in fraud work those refusals are frequently the interesting rows.');
      }
    },
    quiz: [
      {
        q: 'EM is guaranteed not to decrease the likelihood because…',
        options: ['it uses gradient descent with a small step', 'the E-step makes the ELBO tight and the M-step raises it', 'the clusters are convex', 'it re-initialises each round'],
        answer: 1,
        why: 'The gap between $\\log p(x)$ and the ELBO is exactly $D_{KL}(q\\|p(z|x))$; setting $q=p(z|x)$ in the E-step closes that gap to zero, and the M-step then maximises a bound that is, at that instant, equal to the true likelihood — so any increase in the bound is an increase in the likelihood too. "Gradient descent with a small step" is a tempting distractor because EM does feel iterative and incremental like gradient descent, but nothing about it involves a learning rate or a step size, and its guarantee is exact rather than asymptotic. "The clusters are convex" is simply irrelevant to the argument — it is a geometric property of k-means, not of EM, and EM\'s monotonicity holds for mixtures with arbitrarily shaped, non-convex responsibility surfaces.'
      },
      {
        q: 'You need to know whether your data has any cluster structure at all. Which diagnostic answers that?',
        options: ['The elbow method', 'Silhouette score', 'The gap statistic', 'WCSS at k = 10'],
        answer: 2,
        why: 'Only the gap statistic compares your actual WCSS against the WCSS you would get clustering uniform random noise of the same size, which is the only one of the three with an honest null hypothesis to test against — and it is exactly that comparison that lets it support k = 1. The elbow method is a tempting first guess because it is the diagnostic most people reach for by habit, but WCSS decreases monotonically with k by construction whether or not the data has any real structure, so an "elbow" can appear in pure noise. Silhouette score is a genuinely better diagnostic than the elbow for choosing among k ≥ 2, but it has no way to express "there should be only one cluster" — it always compares against some other cluster, even when no other cluster should exist.'
      },
      {
        q: 'An anomaly detector’s alert volume triples overnight. The first hypothesis should be…',
        options: ['a fraud ring', 'a data-pipeline or upstream schema change', 'model drift in the labels', 'the contamination parameter is wrong'],
        answer: 1,
        why: 'Sudden, large volume changes are overwhelmingly input distribution changes rather than a genuine change in the underlying phenomenon — a column that started arriving in cents instead of pounds, a null-filling default that changed upstream, a new device type flooding one feature\'s range. Check the inputs (§2.18\'s PSI is built for exactly this) before opening a fraud investigation. "A fraud ring" is the answer that feels urgent and is rarely right at this scale: genuine fraud waves are usually a fraction of a percent shift, not a tripling overnight. "Model drift in the labels" presupposes you have labels to drift, which an unsupervised detector by definition does not.'
      }
    ],
    cards: [
      { q: 'EM in five words', a: '"Tighten the bound, then maximise it."' },
      { q: 'Why k-means fails on crescents', a: 'Nearest-centroid assignment produces convex Voronoi cells; the clusters are not convex.' },
      { q: 'Choosing k', a: 'Silhouette (meaningful zero) over elbow (monotone by construction); gap statistic to test k = 1.' },
      { q: 'Anomaly detection’s hard part', a: 'Evaluation without labels: rank, review top-k, measure precision@k at real capacity, and watch stability over time.' },
      { q: 'k-means as a limit of EM', a: 'Shrink every GMM component to a shared, spherical, zero covariance: responsibilities collapse to hard 0/1, and the M-step becomes "recentre on the mean".' }
    ]
  });

  /* ------------------------------------------------------------------ 2.10 */
  ML.section({
    id: 'pca', track: 'classical', num: '2.10',
    title: 'Dimensionality reduction: PCA, SVD, t-SNE, UMAP',
    lede: 'Builds geometrically on §1.8’s eigenvectors and points to §1.9.5 for the Lagrangian derivation rather than repeating it; the same eigen-argument reappears whenever someone asks what a low-rank approximation is.',
    html: `
<p>Suppose you have measured fifty things about each customer — spend across fifty product categories, say — and most of those fifty numbers move together: a customer who spends more on groceries tends to spend more on household goods too. Fifty numbers is too many to plot, too many to eyeball for structure, and mostly redundant besides, since knowing forty-nine of them already tells you most of what the fiftieth will say. You want fewer numbers that still capture what is actually happening. Throwing away columns at random is obviously wrong — you might drop the one feature carrying real signal and keep forty-nine that mostly repeat each other. What you want instead is to find the <i>directions</i> in which your data genuinely varies, keep the ones that vary a lot, and discard the ones that barely vary at all. That is the problem PCA solves, and the tools to solve it precisely — eigenvectors — you have already built in §1.8.</p>

<h2><span class="sn">2.10.1</span> PCA, built from what a matrix already means</h2>

<p>Recall the picture from §1.8: a matrix is a machine that moves arrows, and most directions get rotated by it, except for a special few — the eigenvectors — which the matrix only stretches or shrinks, never turns. §1.8's own worked example was $A = \\begin{bmatrix}2&1\\\\1&2\\end{bmatrix}$, whose eigenvectors are $(1,1)$ with $\\lambda=3$ and $(1,-1)$ with $\\lambda=1$, and which — being symmetric — the spectral theorem describes as "a pure stretch along a set of mutually perpendicular axes, no shear and no rotation, once you look along the right axes."</p>

<p>Now reuse that exact matrix for a different job. Suppose $A$ is not an arbitrary linear map but the <b>covariance matrix</b> $\\Sigma$ of two features, and suppose — this is the fact §1.8.3 already established — that covariance matrices are always symmetric and positive semi-definite. Being symmetric, $\\Sigma$ is that same pure stretch along perpendicular axes, no shear, no rotation, once you look along the right axes. So picture your data cloud as roughly elliptical, and ask which axis of that ellipse is longest. §1.8 already answers this without a single new idea: the ellipse's long axis is the eigenvector with the larger eigenvalue, because "stretch by $\\lambda$" along a direction is exactly what a larger eigenvalue means, and a covariance matrix's version of "stretch" is "spread the data out". For $\\Sigma = \\begin{bmatrix}2&1\\\\1&2\\end{bmatrix}$, the data is stretched three times as far along $(1,1)$ as it is along $(1,-1)$ — meaning the two original features rise and fall together (a positive off-diagonal entry), and the direction that captures the most of that joint movement is the diagonal $(1,1)$, not either original axis on its own.</p>

<p>That is the entire geometric content of PCA: <b>the principal components are the eigenvectors of the covariance matrix, and each eigenvalue is exactly how much variance the data has along its own direction.</b> With eigenvalues 3 and 1, keeping only the first component retains $3/(3+1) = 75\\%$ of the total variance in the data using a single number per row instead of two.</p>

<p>Stating it geometrically like this is deliberately the reverse of most textbook orderings, which start from an optimisation problem. The optimisation problem is real and worth having precisely once: find the unit direction $w$ of maximum projected variance, i.e. maximise $w^\\mathsf{T}\\Sigma w$ subject to $\\|w\\|=1$. §1.9.5 works this Lagrangian through in full, five lines from the constrained maximisation to the eigenvector equation $\\Sigma w = \\lambda w$ and back to the fact that the maximum variance achieved is exactly the largest eigenvalue — it is not repeated here. What matters for this section is what that equation buys you once you already know what an eigenvector is: it is the algebraic proof that "the direction of maximum spread" and "the eigenvector of the covariance matrix" were never two different facts requiring two different pieces of evidence. They are the same sentence, and the picture above is what that sentence looks like drawn.</p>

${H.key('The principal directions are the eigenvectors of the covariance matrix; each eigenvalue is the variance captured along its own direction. This is not an approximation of what PCA does — it is what PCA is.')}

${H.history(`<p>Karl Pearson introduced the idea in 1901, framing it exactly as this section does: given a cloud of points, find the line that fits them best in the sense of minimising the sum of squared perpendicular distances to it — the reconstruction-error framing, not the variance-maximisation one. Harold Hotelling reintroduced the same construction independently in 1933, this time from the variance-maximisation side and under the name that stuck, "principal components". That two mathematicians arrived at the identical eigenvector equation from what looked like two different optimisation problems is not a coincidence of history — it is the same fact §1.8.2's Eckart–Young theorem states formally: minimising reconstruction error and maximising retained variance are the same objective, seen from either end, for exactly the same reason that the two halves of a Pythagorean triangle's legs must trade off against a fixed hypotenuse. Total variance in the data is fixed; whatever a component fails to capture, the reconstruction error absorbs instead, so minimising one term is maximising the other by arithmetic necessity, not by additional assumption.</p>`)}

<p>Equivalently, PCA is the SVD (§1.8.2) of the centred data matrix: the right singular vectors of $X$ after subtracting the column means are exactly the eigenvectors of $X^\\mathsf{T}X \\propto \\Sigma$, which is the identity §1.8.2 proved directly. This is the numerically stable way to compute PCA in practice — nobody forms $\\Sigma$ and eigendecomposes it by hand — and it is the reason "PCA" and "truncated SVD" are used almost interchangeably in the field. One practical consequence follows immediately from what §0.2's pitfall box already said about distance: PCA always centres the data first, because variance is measured around the mean, and it should usually be run on <i>scaled</i> features too, unless every column genuinely shares the same units — otherwise "the direction of maximum variance" quietly becomes "the direction of whichever column happens to be measured in the smallest units", exactly the failure mode §1.3's own correlation lab demonstrates when you push two features' standard deviations apart instead of their correlation.</p>

<p><b>What you are looking at.</b> A cloud of points generated with a chosen anisotropy (how much more spread along one axis than the other) and rotation, with the two principal components drawn as arrows from the data's mean — green for PC1, amber for PC2 — each arrow's length set by the square root of its own eigenvalue, so a longer arrow means more variance captured along that direction. Switching the view mode overlays either the projection onto PC1 (each point connected by a faint line to its shadow on the green axis, exactly the projection construction from §0.2) or the full reconstruction using only PC1, showing what the data would look like if PC2 were simply discarded.</p>

<p><b>What to do with it.</b> Leave the defaults — anisotropy $4\\times$, rotated $30°$ — and read the eigenvalues off the readout: $\\lambda_1 \\approx 4$, $\\lambda_2 \\approx 1$, matching the construction exactly, since the generator's own variances along its two axes were set to 4 and 1 before rotating — rotation moves the eigen<i>vectors</i> but never touches the eigen<i>values</i>. PC1 explains close to $4/5 = 80\\%$ of the variance. Now rotate the cloud with the slider and watch the two arrows turn in lockstep with it while the eigenvalues themselves do not move at all — direct visual confirmation that a rotation changes where the axes point without changing how much each one stretches.</p>

<p><b>The thing genuinely worth noticing.</b> Drag anisotropy down to exactly $1\\times$ and watch the arrows start jittering unpredictably as you nudge any other control, sometimes swapping which one is labelled PC1. With equal eigenvalues, every direction in the plane is stretched identically, so there is no longer a unique "direction of maximum variance" to find — any pair of perpendicular axes is an equally valid eigenbasis, and the software's choice of which one to return is an accident of its numerical routine, not a fact about the data. This is the practical form of the warning in §1.3's own PCA lab: a nearly-round data cloud makes PCA loadings meaningless to interpret, however confidently the numbers print out to three decimal places.</p>

${H.lab('pca', 'PCA on data you can shape', 'Drag anisotropy and rotation and watch the two component arrows track the data — PC1 in green always the longer axis of the cloud, PC2 in amber always perpendicular to it. The explained-variance readout is $\\lambda_i/\\sum_j\\lambda_j$; switching to the reconstruction view shows exactly what you throw away by keeping only PC1.')}

<h2><span class="sn">2.10.2</span> What PCA is not for</h2>

<p>Compression always costs something — the discarded components carried some variance — and Eckart–Young (§1.8.2) says PCA's compression is the best possible rank-$r$ approximation in the least-squares sense, so if compression is what you want, PCA is provably the right shape of answer. But "provably optimal at minimising squared reconstruction error" and "the right tool for your actual downstream task" are different claims, and the table below is really about which of your goals PCA's specific optimality target happens to serve.</p>

${H.table(['Use', 'Verdict'], [
      ['Compressing correlated numeric features before a linear model', 'Good — collinearity handled, variance retained'],
      ['Visualising structure in 2-D', 'Fine, with the caveat that PC1–PC2 may hide the interesting direction'],
      ['Feature selection', 'No — components are mixtures of all features, so interpretability is lost'],
      ['Before a tree ensemble', 'Usually harmful — trees split on axes, and PCA rotates the axes away from the meaningful ones'],
      ['Removing noise', 'Sometimes, if noise is genuinely low-variance — which is an assumption, not a fact']
    ])}

<p>The tree-ensemble row deserves a sentence, because it is the row people get wrong most often having just learned PCA is "good practice". §2.7's own lab note observes that trees produce axis-aligned staircases and pay dearly, in extra splits, for any boundary that runs diagonally across the original feature axes. PCA's whole purpose is to <i>rotate</i> the axes to align with variance rather than with your original, meaningful columns — which is precisely the rotation a tree cannot exploit and can only be hurt by, since "utilisation" and "tenure" become unrecoverable blends inside PC1 and PC2, and every split has to be phrased in that mixed, uninterpretable coordinate system instead.</p>

<h2><span class="sn">2.10.3</span> t-SNE and UMAP</h2>
<p>PCA is a <i>linear</i> compression: it can only ever find flat, straight-line directions of variance, so data that lies on a curved surface — a spiral, a swiss roll, a manifold that only makes sense locally — defeats it, because no single straight direction captures a curve. t-SNE and UMAP are non-linear alternatives built for exactly this case, but they buy that flexibility by giving up something PCA never promised to have in the first place: a fixed, reusable coordinate system. Both work by trying to preserve <b>local</b> neighbourhoods — which points were near which other points in the original high-dimensional space — while being free to stretch, compress, and rearrange the overall layout however the optimisation finds convenient. They are visualisation tools, not compression or feature-engineering tools; there is no meaningful way to add a new point to an existing t-SNE plot the way you can project a new row through PCA's already-fitted axes.</p>
<p>That freedom to rearrange is exactly what makes the result dangerous to over-read. Cluster sizes, inter-cluster distances and empty space in a t-SNE plot are artefacts of the optimisation and the perplexity setting — perplexity being, roughly, how many neighbours each point tries to stay close to — and <b>not properties of your data</b>. Two genuinely distant clusters can be drawn close together or far apart depending on nothing but the random seed the optimiser started from, because nothing in the objective function is asked to preserve those global distances; only the local neighbour lists are. UMAP preserves somewhat more global structure than t-SNE and is considerably faster on large datasets, but the same warning applies to it for the same reason: neither algorithm's loss function contains a term that cares about far-apart relationships.</p>
${H.flag('Reading global geometry — cluster distances, relative sizes — off a t-SNE or UMAP plot is the classic mistake. Say "the neighbourhoods are meaningful, the distances are not" and move on.')}

<p><b>What you are looking at.</b> Three projections of the identical five-dimensional synthetic dataset with three genuine clusters, computed live: PCA (a linear projection onto the top two eigenvectors), a neighbour embedding built in the spirit of t-SNE, and a random projection included purely as a control — a projection with no notion of "preserve structure" at all. The readout reports what fraction of each point's true nearest neighbours survive into the 2-D layout, and, for PCA specifically, what fraction of the original variance the two axes shown actually capture.</p>

<p><b>What to do with it.</b> Compare all three at once. The random projection typically preserves fewer than half of any point's true neighbours — it was never asked to preserve anything — while both PCA and the neighbour embedding do noticeably better. Then press "re-run with a new seed" repeatedly while watching only the neighbour embedding: the three clusters visibly rotate, translate, and change their apparent size and separation on every run, while the "neighbours preserved" readout barely moves.</p>

<p><b>The thing genuinely worth noticing.</b> That combination — layout changes wildly, neighbour-preservation score barely changes — is the whole warning above made into something you watched happen rather than were told about. The clusters staying separated across reseeds is real information: those points really were each other's nearest neighbours in the original five dimensions. Their changing position, size and spacing on the page is not information about anything; it is the optimiser's arbitrary choice of where to place an arbitrary local minimum, made fresh each run.</p>

${H.lab('tsne', 'The same data, three projections', 'PCA, a t-SNE-style neighbour embedding, and a random projection — all computed live on the same five-dimensional data. Re-run the embedding with a different seed and watch the cluster positions and apparent sizes change while the neighbourhood-preservation score stays stable. That contrast is exactly what you must, and must not, read into the picture.')}

${H.probe([
      ['What does PC1 maximise?', 'Projected variance $w^\\mathsf{T}\\Sigma w$ subject to $\\|w\\|=1$; the Lagrangian (§1.9.5) shows the maximiser is the top eigenvector of the covariance matrix, and the maximum value achieved is its eigenvalue.'],
      ['PCA or SVD?', 'PCA is the SVD of the centred data matrix — the same object under two names, and the SVD route is the numerically stable way real software computes it rather than ever forming Σ explicitly.'],
      ['Would you PCA before boosting?', 'Rarely — trees split on axes (§2.7), and PCA rotates away from the interpretable original axes, trading away explainability without a reliable accuracy gain in return.']
    ], 'Reading global geometry — cluster distances, relative sizes, empty space — off a t-SNE or UMAP plot. Neither algorithm\'s objective contains a term for preserving anything beyond local neighbourhoods, so anything you read off the plot beyond "these points were near each other" is reading noise as signal.')}`,
    labs: {
      pca: function (host) {
        const st = Viz.controls(host, [
          { k: 'rot', label: 'rotate the cloud', min: 0, max: 180, step: 1, value: 30, fmt: v => v + '°' },
          { k: 'aniso', label: 'anisotropy (λ₁/λ₂)', min: 1, max: 12, step: .2, value: 4, fmt: v => v.toFixed(1) + '×' },
          { k: 'show', label: 'show', type: 'buttons', value: 'pcs', options: [{ v: 'pcs', t: 'components' }, { v: 'proj', t: 'projection onto PC1' }, { v: 'rec', t: 'reconstruction' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'l1', label: 'λ₁', cls: 'key' }, { k: 'l2', label: 'λ₂' },
          { k: 'ev1', label: 'PC1 explains' }, { k: 'err', label: 'error if you drop PC2' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(43), th = st.rot * Math.PI / 180;
            const pts = [];
            for (let i = 0; i < 320; i++) {
              const a = R.normal(0, Math.sqrt(st.aniso)), b = R.normal(0, 1);
              pts.push([a * Math.cos(th) - b * Math.sin(th), a * Math.sin(th) + b * Math.cos(th)]);
            }
            const pc = Num.pca(pts);
            const P = Viz.plot(ctx, w, h, { xd: [-6, 6], yd: [-4.5, 4.5] }).frame({ xlabel: 'feature 1', ylabel: 'feature 2' });
            P.clip(() => {
              if (st.show === 'pcs') {
                P.dots(pts, { r: 2.6, color: T.blue, alpha: .55 });
              } else {
                const v = pc.vectors[0];
                pts.forEach(p => {
                  const c = [p[0] - pc.mean[0], p[1] - pc.mean[1]];
                  const t = c[0] * v[0] + c[1] * v[1];
                  const proj = [pc.mean[0] + t * v[0], pc.mean[1] + t * v[1]];
                  if (st.show === 'proj') {
                    P.line([p, proj], { color: T.faint, width: .8, alpha: .5 });
                    P.dots([p], { r: 2.2, color: T.blue, alpha: .4 });
                    P.dots([proj], { r: 2.6, color: T.red, alpha: .8 });
                  } else {
                    P.dots([proj], { r: 2.6, color: T.red, alpha: .7 });
                  }
                });
              }
              pc.vectors.forEach((v, i) => {
                const s = 2 * Math.sqrt(Math.max(0, pc.values[i]));
                P.arrow(pc.mean[0], pc.mean[1], pc.mean[0] + v[0] * s, pc.mean[1] + v[1] * s, { color: i ? T.amber : T.green, width: 2.6 });
                P.text(pc.mean[0] + v[0] * s, pc.mean[1] + v[1] * s, ' PC' + (i + 1) + ' · λ=' + pc.values[i].toFixed(2), { color: i ? T.amber : T.green, font: '11px ui-sans-serif' });
              });
            });
            out({
              l1: pc.values[0].toFixed(3), l2: pc.values[1].toFixed(3),
              ev1: (pc.explained[0] * 100).toFixed(1) + '%',
              err: (pc.explained[1] * 100).toFixed(1) + '% of variance'
            });
          }
        });
        Viz.note(host, 'Set anisotropy to 1 and the components become arbitrary — with equal eigenvalues there is no preferred direction, and the "principal" component is whatever the numerics happened to return. That degeneracy is worth knowing before you interpret loadings.');
      },

      tsne: function (host) {
        let seed = 3;
        const st = Viz.controls(host, [
          { k: 'method', label: 'projection', type: 'buttons', value: 'ne', options: [{ v: 'pca', t: 'PCA' }, { v: 'ne', t: 'neighbour embedding' }, { v: 'rand', t: 'random projection' }] },
          { k: 'perp', label: 'neighbourhood size', min: 3, max: 40, step: 1, value: 12, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kept', label: 'neighbours preserved', cls: 'key' }, { k: 'var', label: 'variance kept (PCA)' }, { k: 'stable', label: 'layout stable across seeds?' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            // 5-D data with three genuine clusters
            const R = Num.rng(seed);
            const D = 5, K = 3, n = 150;
            const centers = Array.from({ length: K }, () => Array.from({ length: D }, () => R.normal(0, 2.4)));
            const X = [], lab = [];
            for (let i = 0; i < n; i++) {
              const c = i % K;
              X.push(centers[c].map(v => v + R.normal(0, .8)));
              lab.push(c);
            }
            let Y;
            if (st.method === 'pca') {
              const pc = Num.pca(X);
              Y = X.map(p => {
                const cc = p.map((v, j) => v - pc.mean[j]);
                return [Num.dot(cc, pc.vectors[0]), Num.dot(cc, pc.vectors[1])];
              });
            } else if (st.method === 'rand') {
              const RP = Num.rng(seed + 9);
              const a = Array.from({ length: D }, () => RP.normal(0, 1)), b = Array.from({ length: D }, () => RP.normal(0, 1));
              Y = X.map(p => [Num.dot(p, a) / Math.sqrt(D), Num.dot(p, b) / Math.sqrt(D)]);
            } else {
              // simple force-directed neighbour embedding (t-SNE flavoured, not identical)
              const RE = Num.rng(seed + 21);
              Y = X.map(() => [RE.normal(0, .6), RE.normal(0, .6)]);
              const kNN = X.map((p, i) => X.map((q, j) => [j, Num.dot(p.map((v, t) => v - q[t]), p.map((v, t) => v - q[t]))])
                .sort((u, v) => u[1] - v[1]).slice(1, st.perp + 1).map(u => u[0]));
              for (let it = 0; it < 220; it++) {
                const lr = 0.12 * (1 - it / 260);
                for (let i = 0; i < n; i++) {
                  let fx = 0, fy = 0;
                  kNN[i].forEach(j => { fx -= (Y[i][0] - Y[j][0]); fy -= (Y[i][1] - Y[j][1]); });
                  for (let s = 0; s < 14; s++) {
                    const j = RE.int(n); if (j === i) continue;
                    const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
                    const d2 = dx * dx + dy * dy + .05;
                    fx += 2.2 * dx / d2; fy += 2.2 * dy / d2;
                  }
                  Y[i][0] += lr * fx / st.perp; Y[i][1] += lr * fy / st.perp;
                }
              }
            }
            const xs = Y.map(p => p[0]), ys = Y.map(p => p[1]);
            const pad = .15;
            const xr = [Math.min.apply(null, xs), Math.max.apply(null, xs)], yr = [Math.min.apply(null, ys), Math.max.apply(null, ys)];
            const dx = (xr[1] - xr[0]) * pad || 1, dy = (yr[1] - yr[0]) * pad || 1;
            const P = Viz.plot(ctx, w, h, { xd: [xr[0] - dx, xr[1] + dx], yd: [yr[0] - dy, yr[1] + dy] })
              .frame({ xlabel: 'component 1', ylabel: 'component 2' });
            const cols = Labs.palette(K);
            P.clip(() => Y.forEach((p, i) => P.dots([p], { r: 3.6, color: cols[lab[i]], stroke: true })));
            // neighbour preservation
            const nn = (pts, i, k) => pts.map((q, j) => [j, (pts[i][0] - q[0]) ** 2 + (pts[i][1] - q[1]) ** 2 + (pts[i].length > 2 ? 0 : 0)])
              .sort((u, v) => u[1] - v[1]).slice(1, k + 1).map(u => u[0]);
            let kept = 0;
            for (let i = 0; i < n; i += 3) {
              const hi = X.map((q, j) => [j, Num.sum(X[i].map((v, t) => (v - q[t]) ** 2))]).sort((u, v) => u[1] - v[1]).slice(1, 11).map(u => u[0]);
              const lo = nn(Y, i, 10);
              kept += hi.filter(j => lo.indexOf(j) >= 0).length / 10;
            }
            const pc = Num.pca(X);
            out({
              kept: (100 * kept / Math.ceil(n / 3)).toFixed(0) + '%',
              var: ((pc.explained[0] + pc.explained[1]) * 100).toFixed(0) + '%',
              stable: st.method === 'ne' ? 'no — re-run and see' : 'yes'
            });
          }
        });
        Viz.buttons(host, [{ label: 'Re-run with a new seed', primary: true, on: () => { seed = Math.floor(Math.random() * 1e4); S.redraw(); } }]);
        Viz.note(host, 'Re-run the neighbour embedding a few times: the three clusters stay separated (that is real) while their positions, orientations and apparent sizes move (that is not). PCA’s layout is deterministic but may hide structure that lives outside the top two variance directions.');
      }
    },
    quiz: [
      {
        q: 'PCA’s first component maximises…',
        options: ['the correlation with the target', 'the projected variance, subject to unit norm', 'the number of non-zero loadings', 'the reconstruction error'],
        answer: 1,
        why: 'The Lagrangian derived in §1.9.5 gives $\\Sigma w=\\lambda w$: PC1 is the top eigenvector of the covariance matrix, and the variance it captures is that eigenvector\'s own eigenvalue. "Correlation with the target" is the tempting distractor because it sounds like the kind of thing a "principal" direction should care about — but PCA is entirely unsupervised, computed from $X$ alone with no target column in sight, which is exactly why §2.10.2\'s table calls feature selection a bad use of it: a component can carry huge variance while having nothing to do with what you are trying to predict. "Minimises the reconstruction error" is very nearly true and is a fact about PCA — Eckart–Young (§1.8.2) — but it describes the same optimum from the opposite direction, as a minimisation rather than the variance-maximisation the question actually asked about.'
      },
      {
        q: 'In a t-SNE plot, two clusters appear far apart and one looks much larger. What can you conclude?',
        options: ['They are genuinely far apart and one is bigger', 'Neither — inter-cluster distance and cluster size are artefacts of the optimisation', 'The perplexity is too low', 'The data is Gaussian'],
        answer: 1,
        why: 'Neighbourhood membership is meaningful — t-SNE\'s objective is built entirely around preserving which points were near which other points — but nothing in that objective rewards preserving global distance or density, so inter-cluster gaps and apparent cluster sizes are simply wherever the optimisation happened to settle. Re-running with a different random seed typically changes both substantially while barely touching which points cluster with which, exactly as the neighbour-embedding lab shows. "The perplexity is too low" is a tempting technical-sounding answer, but changing perplexity changes the <i>local</i> neighbourhood size being preserved, not whether the resulting global layout becomes trustworthy — no perplexity setting makes inter-cluster distance meaningful, because the objective never asked for that.'
      }
    ],
    cards: [
      { q: 'PCA derivation', a: 'Maximise $w^\\mathsf{T}\\Sigma w$ s.t. $\\|w\\|=1$ → $\\Sigma w=\\lambda w$ (§1.9.5): eigenvectors of the covariance.' },
      { q: 'Explained variance', a: '$\\lambda_i/\\sum_j\\lambda_j$ — the share of total variance along component $i$.' },
      { q: 't-SNE caveat', a: 'Local neighbourhoods are meaningful; cluster sizes, distances and empty space are optimisation artefacts.' },
      { q: 'PCA geometrically', a: 'Covariance is symmetric PSD, so it is a pure stretch along perpendicular axes (§1.8) — those axes are the components.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.11 */
  ML.section({
    id: 'features', track: 'classical', num: '2.11',
    title: 'Feature engineering, encoding, missing data, leakage',
    lede: 'The most practically load-bearing section in Part 2, and the one where interviews are actually lost.',
    html: `
<p>Every model so far in Part 2 has assumed the features already arrived as clean numbers. Real tables do not. A merchant category column holds 500 distinct strings, not one meaningful order. A quarter of the income column is blank. A "date of last payment" field exists for every row, including the ones that never made a payment at all, because someone filled it with the date the row was created instead. None of these problems are solved by choosing a better model — a better random forest cannot invent a merchant-category ordering that does not exist, and no amount of regularization un-leaks a feature that already contains the answer. This section is about the decisions that happen before a single model is fit, and it is, in practice, where more real projects fail than in any choice of algorithm.</p>

<h2><span class="sn">2.11.1</span> Encoding categories into numbers</h2>
<p><b>One-hot</b> encoding — one binary column per category — is the safe default: it invents no ordering and no numeric relationship between categories that was not there. Its failure mode is arithmetic rather than statistical: a merchant-category column with 500 distinct values becomes 500 new columns, most of them almost entirely zero, and a tree or a linear model now has to search a much wider, sparser space to find anything.</p>

<p><b>Target (mean) encoding</b> replaces a category with the average outcome seen for that category in the training data — merchant category "electronics" becomes, say, 0.08, its historical default rate — which is compact, informative, and dangerous for exactly the reason it is powerful: the encoding is computed <i>from the very labels the model is about to be trained on</i>. Fit it on the whole dataset before cross-validation and every row's own label has quietly contributed to its own feature value.</p>

${H.worked('how much a single row leaks into its own encoding', `<p>Category "electronics" has 10 training rows, 3 of them bad. The full-sample mean encoding is $3/10 = 0.300$, and every row of that category — including the 3 bad ones — receives exactly that value as a feature.</p>
<p>Now ask what a <i>bad</i> row's encoding would have been had it never been allowed to see its own label: leave it out and recompute from the remaining 9 rows, $2/9 = 0.222$. A good row's leave-one-out encoding, by contrast, barely moves: $3/9=0.333$ against $3/10=0.300$.</p>
<p>The bad row's full-sample encoding (0.300) sits noticeably below its honest, leave-one-out encoding (0.222) — its own presence in the average has pulled the number towards "less risky" than the other 9 rows alone would say. That gap is small for one row, but it is systematic across every row in the dataset, and a model with enough capacity will learn to exploit it, producing a validation score that never survives contact with genuinely new categories.</p>`)}

<p>The fix is the same one used everywhere else a statistic is estimated from data that will also be scored: fit the encoding <i>inside</i> each training fold only, or use an out-of-fold scheme, and never touch the row you are currently encoding. CatBoost's <b>ordered target statistics</b> (§2.8.3) are the principled, systematic version of this — using only rows earlier in a random permutation, so no row's label can ever reach its own encoding, by construction rather than by discipline. <b>Ordinal</b> encoding — mapping categories to integers 1, 2, 3… — is harmless for trees, which only ever ask "is this value above a threshold" and can recover any monotone-enough split regardless of which integers you chose, but it actively misleads a linear model unless the assigned order is a real fact about the world, because a linear model will happily fit a coefficient to "category 3 is worth one and a half times category 2", a statement that was never true.</p>

<h2><span class="sn">2.11.2</span> WOE and IV — the credit-risk vocabulary</h2>
<p>Bin a feature, then per bin:</p>
$$\\mathrm{WOE}_i = \\ln\\frac{\\%\\text{good}_i}{\\%\\text{bad}_i}, \\qquad \\mathrm{IV} = \\sum_i (\\%\\text{good}_i - \\%\\text{bad}_i)\\cdot \\mathrm{WOE}_i$$
<p>Read WOE as a log-odds statement about one bin: positive when a bin has more than its fair share of goods relative to bads, negative when the reverse, and exactly zero when a bin's mix matches the overall population — which is precisely the scale a logistic scorecard wants, since a logistic model's whole output is built from log-odds (§2.4). IV then sums each bin's WOE, weighted by how much that bin's good/bad mix actually differs from the rest, into one number summarising the feature's total separating power. Conventional bands: &lt;0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, &gt;0.3 strong.</p>
${H.flag('Flag when you say this: those bands are industry rules of thumb, not test-derived thresholds — and IV is algebraically the symmetric KL (Jeffreys) divergence between the good and bad distributions, the same object as PSI in §2.18 with different inputs.')}

${H.deriv('why IV is exactly the Jeffreys divergence between the good and bad distributions', [
      ['$D_{KL}(\\text{good}\\,\\|\\,\\text{bad}) = \\sum_i \\%\\text{good}_i\\, \\ln\\dfrac{\\%\\text{good}_i}{\\%\\text{bad}_i}$', 'The ordinary KL divergence of §1.10, applied to the two distributions across bins: how the goods are spread out, and how the bads are.'],
      ['$D_{KL}(\\text{bad}\\,\\|\\,\\text{good}) = -\\sum_i \\%\\text{bad}_i\\, \\ln\\dfrac{\\%\\text{good}_i}{\\%\\text{bad}_i}$', 'The same divergence with the two roles swapped, then rewritten using $\\ln(1/x)=-\\ln x$ so both lines share the identical log term $\\ln(\\%\\text{good}_i/\\%\\text{bad}_i)$ — the point of this rewrite is to make the next line possible.'],
      ['sum: $\\sum_i (\\%\\text{good}_i - \\%\\text{bad}_i)\\,\\ln\\dfrac{\\%\\text{good}_i}{\\%\\text{bad}_i}$', 'Add the two divergences. Because both now carry the same log factor, the two sums combine into a single sum with $(\\%\\text{good}_i-\\%\\text{bad}_i)$ out front — ordinary algebraic factoring, nothing more.'],
      ['$= \\sum_i (\\%\\text{good}_i-\\%\\text{bad}_i)\\cdot\\mathrm{WOE}_i = \\mathrm{IV}$', 'Recognise the log term as $\\mathrm{WOE}_i$ by its own definition. The result is exactly the IV formula, so $\\mathrm{IV} = D_{KL}(\\text{good}\\|\\text{bad}) + D_{KL}(\\text{bad}\\|\\text{good})$, the symmetrised — Jeffreys — divergence.']
    ], 'Check it against the worked example below: $D_{KL}(\\text{good}\\|\\text{bad}) \\approx 0.332$ and $D_{KL}(\\text{bad}\\|\\text{good}) \\approx 0.372$ for those three bins, and $0.332+0.372=0.704$ — exactly the IV computed by hand there. IV is not merely "similar in spirit" to a divergence; it is one, which is also why it inherits KL\'s property of blowing up when a bin has bads but essentially no goods or vice versa — a single near-empty bin can make IV enormous without the feature being genuinely predictive anywhere else.')}

${H.lab('woe', 'WOE and IV, computed on an editable table', 'Move the counts and watch WOE, the IV contributions and the monotonicity check respond live. The suspicious-IV warning fires above 0.5 for the reason the worked box explains — and the derivation above explains why that threshold is not arbitrary.')}

${H.worked('worked WOE and IV — one feature, three bins', `
${H.table(['Bin', 'good', 'bad', '%good', '%bad', 'WOE', 'contrib'], [
      ['util &lt; 0.3', '4,500', '100', '0.50', '0.20', '+0.916', '0.275'],
      ['0.3–0.7', '3,150', '150', '0.35', '0.30', '+0.154', '0.008'],
      ['util &gt; 0.7', '1,350', '250', '0.15', '0.50', '−1.204', '0.421']
    ])}
<p>WOE for the top bin: $\\ln(0.15/0.50) = \\ln 0.30 = -1.204$. Its IV contribution: $(0.15-0.50)\\times(-1.204) = 0.421$ — positive, because both factors flip sign together. Summing: <b>IV = 0.275 + 0.008 + 0.421 = 0.704</b>.</p>
<p>By the conventional bands that is "very strong" — and <b>above 0.5 you should be suspicious rather than pleased</b>. Check that utilisation was measured strictly before the outcome window opened. Note also that the middle bin contributes almost nothing (0.008): it separates nobody, so merging it into a neighbour simplifies the scorecard at no cost. That is what binning decisions actually look like.</p>`)}

<h2><span class="sn">2.11.3</span> How to bin</h2>
<p>Equal-width bins slice the feature's numeric range into equal segments, and on skewed financial data that is almost always the wrong cut: if utilisation ranges from 0 to 50 because a handful of accounts carry balances far beyond their limit, an equal-width scheme puts 95% of the population inside the first of ten bins and leaves the other nine describing a sparse handful of extreme outliers — every bin is supposed to say something about a meaningfully-sized group, and nine of the ten now say almost nothing. Equal-frequency (quantile) binning fixes this by construction, choosing cut points so that each bin holds the same <i>number</i> of rows rather than the same span of values, and it is the safe default for exactly the reason equal-width fails.</p>
<p>Supervised binning goes one step further and chooses cuts using the target itself — a shallow decision tree grown on the single feature (§2.7's splitting logic, applied to one column), or ChiMerge, which starts from many fine bins and merges neighbours whose good/bad mix is not statistically different. Either way, the next step is to <b>enforce monotonicity</b>: merge any adjacent bins that break the trend, because a scorecard whose risk goes up, then down, then up again as utilisation rises is not a policy anyone can defend to a validator, whatever the data technically supports. Two further rules keep you out of trouble: every bin needs a minimum population (5% of the base is the usual floor) and a minimum count of bads, or its WOE is estimated from too few events to mean anything; and the binning itself must be fitted inside the fold like any other learned transformation — it is exactly as capable of leaking the target as the encodings in §2.11.1, for the identical reason.</p>

<h2><span class="sn">2.11.4</span> Missing data</h2>
<p>The fix for a missing value depends entirely on <i>why</i> it is missing, so name the mechanism before reaching for a default. <b>MCAR</b> — missing completely at random — means the chance a value is absent has nothing to do with anything, observed or not: a system glitch drops 2% of rows' income field at random. It is safe to impute, because the missing rows are, in expectation, an unbiased sample of all rows. <b>MAR</b> — missing at random, despite the confusing name — means the chance of absence is explained by <i>other, observed</i> features: younger applicants are less likely to have filled in an income field, but conditional on age the missingness has no further pattern. Model-based imputation, which predicts the missing value from the other columns, is principled here because those other columns genuinely carry the information needed to fill the gap. <b>MNAR</b> — missing not at random — is the dangerous case: the chance of absence depends on the unobserved value itself, as when the wealthiest and the most financially distressed applicants are both disproportionately likely to leave an income field blank, each for their own reason. There, the fact of missingness <i>is</i> informative and imputing a plausible number actively destroys signal, so the standard fix is to add an explicit missing-indicator column alongside whatever imputed value you use, letting the model learn from the flag itself rather than losing that information to a filled-in guess.</p>

${H.analogy(`<p>A survey with a "prefer not to say" option is the whole taxonomy in one form. If the option is skipped purely because a page loaded slowly and people gave up before reaching it, that is MCAR — annoying, but unbiased. If younger respondents skip the income question far more often simply because they check phones less carefully and rush through forms, and you can see their age, that is MAR — the skip pattern is explained by something you already know. If people skip the income question specifically <i>because</i> their income is unusually high or unusually low and they would rather not say, that is MNAR, and no amount of clever imputation recovers what "prefer not to say" was actually telling you: the silence itself was the data point.</p>`)}

<p>Trees handle missing values natively, learning a default direction to send a missing row at each split rather than requiring the value to be filled in beforehand; linear models have no such mechanism and need the gap filled explicitly before they can run at all. Median imputation plus a missing-indicator column is a defensible, low-effort baseline across mechanisms — it does not require correctly diagnosing MCAR versus MAR versus MNAR before you can start, and the indicator column recovers most of what MNAR would otherwise cost you.</p>

<h2><span class="sn">2.11.5</span> Leakage</h2>
<p>Leakage is any feature computed using information that will not exist at prediction time: post-outcome fields recorded after the event you are trying to predict, target-derived encodings fitted on everything rather than inside the fold (§2.11.1), IDs that happen to be correlated with time, aggregates computed over a feature's full history rather than strictly up to the decision date, a "days since last payment" field that was in fact recorded after the account had already defaulted. None of these are algorithm failures — every model in this course will happily, faithfully learn whatever the leaked feature is telling it, which is exactly the problem. It shows up as an implausibly good validation score, because the model has, in effect, been handed the answer sheet disguised as a feature.</p>
${H.key('If a result is too good, look for leakage before you celebrate.')}

<p><b>What you are looking at.</b> Five short scenarios, one at a time, each posing the same underlying question in a different guise: given two candidate ways of building a feature, which one is actually knowable at the moment the model would need to make its prediction? Choosing an option locks in your answer, reveals which one was point-in-time safe, and explains why in one sentence.</p>

<p><b>What to do with it.</b> Work through all five before checking how you did overall. Each scenario is drawn from a real category of mistake — a post-outcome field, an encoding fitted outside the fold, an aggregate that ignores the cutoff date, an ID standing in for time, and a suspiciously good validation score — so getting one wrong tells you something specific about which category of leakage is easiest for you to miss, not just that you missed a question.</p>

<p><b>The thing genuinely worth noticing.</b> The running score readout states plainly that a single unsafe answer is enough to invalidate a validation result — there is no partial credit, because leakage does not average out across features the way ordinary noise does. One leaked column can inflate an entire model's validation AUC regardless of how carefully every other feature was built, which is exactly why a validator's checklist runs through every feature individually rather than accepting a good aggregate score as reassurance.</p>

${H.lab('leak', 'The leakage checklist, as a live audit', 'Five questions, one worked scenario each. Answer them for a feature and the verdict assembles — this is the same interrogation a model validator will run on your work, and getting even one wrong is enough to invalidate the result.')}

${H.probe([
      ['How do you prevent target-encoding leakage?', 'Fit the encoding within each training fold only, or use out-of-fold / ordered statistics (§2.8.3) — never let a row\'s own label reach its own encoded value, as the worked leave-one-out example above shows directly.'],
      ['A feature has IV 0.5 — good news?', 'Nominally "strong" by the conventional bands, but suspiciously high IV is a classic leakage signature precisely because IV is a divergence (derived above) and divergences blow up on near-empty, near-perfectly-separating bins. Investigate before shipping.'],
      ['Which missingness mechanism needs an indicator?', 'MNAR — the fact of missingness carries information about the unobserved value itself, so imputing a plausible number destroys exactly the signal an indicator column would have preserved.']
    ], 'Imputing before splitting. The imputer has then seen the validation rows, so its estimate of the median or the model it fits to predict missing values has already learned something about data it is about to be scored on — the identical mistake as fitting a target encoding on the full dataset, one step earlier in the pipeline.')}`,
    labs: {
      woe: function (host) {
        const st = Viz.controls(host, [
          { k: 'g1', label: 'bin 1 · goods', min: 100, max: 6000, step: 50, value: 4500, fmt: v => v.toLocaleString() },
          { k: 'b1', label: 'bin 1 · bads', min: 10, max: 600, step: 10, value: 100, fmt: v => v },
          { k: 'g2', label: 'bin 2 · goods', min: 100, max: 6000, step: 50, value: 3150, fmt: v => v.toLocaleString() },
          { k: 'b2', label: 'bin 2 · bads', min: 10, max: 600, step: 10, value: 150, fmt: v => v },
          { k: 'g3', label: 'bin 3 · goods', min: 100, max: 6000, step: 50, value: 1350, fmt: v => v.toLocaleString() },
          { k: 'b3', label: 'bin 3 · bads', min: 10, max: 600, step: 10, value: 250, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'iv', label: 'information value', cls: 'key' }, { k: 'band', label: 'conventional band' },
          { k: 'mono', label: 'monotone?' }, { k: 'flag', label: 'suspicious?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const goods = [st.g1, st.g2, st.g3], bads = [st.b1, st.b2, st.b3];
            const G = Num.sum(goods), B = Num.sum(bads);
            const rows = goods.map((g, i) => {
              const pg = g / G, pb = bads[i] / B;
              const woe = Math.log(Math.max(1e-9, pg) / Math.max(1e-9, pb));
              return { bin: ['util < 0.3', '0.3 – 0.7', 'util > 0.7'][i], g: g, b: bads[i], pg: pg, pb: pb, woe: woe, contrib: (pg - pb) * woe };
            });
            const IV = Num.sum(rows.map(r => r.contrib));
            // table
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const cols = ['bin', 'good', 'bad', '%good', '%bad', 'WOE', 'contrib'];
            const cw = Math.min(92, (w - 24) / 7);
            ctx.fillStyle = T.blue;
            cols.forEach((c, i) => { ctx.textAlign = i === 0 ? 'left' : 'right'; ctx.fillText(c, 14 + i * cw + (i ? cw - 8 : 0), 20); });
            rows.forEach((r, ri) => {
              const y = 44 + ri * 24;
              const vals = [r.bin, r.g.toLocaleString(), r.b.toLocaleString(), r.pg.toFixed(3), r.pb.toFixed(3), (r.woe >= 0 ? '+' : '') + r.woe.toFixed(3), r.contrib.toFixed(3)];
              vals.forEach((v, i) => {
                ctx.textAlign = i === 0 ? 'left' : 'right';
                ctx.fillStyle = i === 5 ? (r.woe >= 0 ? T.green : T.red) : T.text;
                ctx.fillText(v, 14 + i * cw + (i ? cw - 8 : 0), y);
              });
            });
            // WOE profile
            const by = 130, bh = 96, bw = Math.min(w - 40, 420);
            const mx = Math.max.apply(null, rows.map(r => Math.abs(r.woe))) || 1;
            ctx.strokeStyle = T.line; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(14, by + bh / 2); ctx.lineTo(14 + bw, by + bh / 2); ctx.stroke(); ctx.setLineDash([]);
            rows.forEach((r, i) => {
              const x = 14 + (i + .5) * (bw / 3) - 26;
              const hgt = (r.woe / mx) * (bh / 2 - 6);
              ctx.fillStyle = r.woe >= 0 ? T.blue : T.red;
              ctx.fillRect(x, by + bh / 2 - Math.max(0, hgt), 52, Math.abs(hgt));
              ctx.fillStyle = T.muted; ctx.font = '10px ui-sans-serif'; ctx.textAlign = 'center';
              ctx.fillText(r.bin, x + 26, by + bh + 12);
            });
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
            ctx.fillText('WOE > 0 = better odds (good)', 14 + bw + 10, by + 16);
            ctx.fillText('WOE < 0 = worse odds (bad)', 14 + bw + 10, by + 34);
            const mono = (rows[0].woe > rows[1].woe && rows[1].woe > rows[2].woe) || (rows[0].woe < rows[1].woe && rows[1].woe < rows[2].woe);
            out({
              iv: IV.toFixed(3),
              band: IV < .02 ? 'useless' : IV < .1 ? 'weak' : IV < .3 ? 'medium' : 'strong',
              mono: mono ? 'yes' : 'no — rebin',
              flag: IV > .5 ? 'yes — check for leakage' : 'no'
            });
          }
        });
        Viz.note(host, 'A non-monotone WOE profile usually means the binning is wrong, not that the feature is wrong. And an IV above 0.5 on a behavioural feature is a leakage smell — verify the measurement date before you celebrate.');
      },

      leak: function (host) {
        const items = [
          { q: 'Could this field exist at decision time?', bad: '"days_since_last_payment" recorded after the default event', good: 'Balance as of the application timestamp', why: 'Anything stamped after the outcome window opens is not a feature, it is the answer.' },
          { q: 'Was any encoding fitted outside the fold?', bad: 'Target-mean encoding computed on the full dataset before CV', good: 'Out-of-fold target encoding, or CatBoost ordered statistics', why: 'The encoding carries the label; fitting it on everything lets each row see its own target.' },
          { q: 'Do aggregates respect the cutoff date?', bad: 'Customer mean transaction size over the whole panel', good: 'Mean over the 90 days strictly before the decision', why: 'Point-in-time correctness. The offline store must reconstruct what was known then (§2.18).' },
          { q: 'Is an ID standing in for time?', bad: 'account_id, which increments with signup date', good: 'Explicit tenure, with the ID dropped', why: 'Monotone IDs let the model read the calendar and exploit cohort effects that will not repeat.' },
          { q: 'Does the score look too good?', bad: 'AUC 0.98 on a problem where the industry ships 0.75', good: 'AUC in the plausible band, with an OOT check', why: 'Implausible performance is the loudest leakage signal there is.' }
        ];
        let idx = 0, answers = new Array(items.length).fill(null);
        const box = ML.el('div');
        host.appendChild(box);
        function render() {
          const it = items[idx];
          box.innerHTML = '<p class="boxtitle">leakage checklist · question ' + (idx + 1) + ' of ' + items.length + '</p>' +
            '<p style="font-size:1.05em;font-weight:600;font-family:var(--sans)">' + it.q + '</p>' +
            '<div class="qopts">' +
            '<button class="qopt" data-v="bad"><span class="k">A</span><span>' + it.bad + '</span></button>' +
            '<button class="qopt" data-v="good"><span class="k">B</span><span>' + it.good + '</span></button>' +
            '</div><p class="qwhy" style="display:none">' + it.why + '</p>' +
            '<div class="btnrow"><button class="btn" data-nav="prev">← previous</button><button class="btn primary" data-nav="next">next →</button></div>' +
            '<p class="small" id="leakScore" style="margin-top:10px"></p>';
          box.querySelectorAll('.qopt').forEach(b => b.addEventListener('click', () => {
            answers[idx] = b.getAttribute('data-v');
            box.querySelectorAll('.qopt').forEach(x => {
              x.classList.add('locked');
              if (x.getAttribute('data-v') === 'good') x.classList.add('right');
              else x.classList.add('wrong');
            });
            box.querySelector('.qwhy').style.display = '';
            score();
          }));
          box.querySelector('[data-nav="next"]').addEventListener('click', () => { idx = (idx + 1) % items.length; render(); });
          box.querySelector('[data-nav="prev"]').addEventListener('click', () => { idx = (idx - 1 + items.length) % items.length; render(); });
          score();
        }
        function score() {
          const done = answers.filter(a => a !== null).length;
          const ok = answers.filter(a => a === 'good').length;
          const el2 = box.querySelector('#leakScore');
          if (el2) el2.innerHTML = done ? ('<b>' + ok + ' of ' + done + '</b> answered with the point-in-time-safe option. A single "A" answer is enough to invalidate a validation score.') : 'Pick the option that is safe at decision time.';
        }
        render();
      }
    },
    quiz: [
      {
        q: 'You compute target-mean encoding on the full dataset, then run 5-fold CV. What is the consequence?',
        options: ['Nothing — the encoding is not a model', 'Each validation row influenced its own encoding, so CV is optimistically biased', 'The model will underfit', 'It only matters for high-cardinality features'],
        answer: 1,
        why: 'The encoding for a category is an average that includes every row of that category, so a validation row\'s own label has already leaked, in small part, into the very feature the model uses to predict it — exactly the arithmetic the worked leave-one-out example in §2.11.1 makes concrete. Fit it inside the fold, or use out-of-fold / ordered statistics instead. "Nothing — the encoding is not a model" is the tempting rationalisation, treating feature engineering as somehow exempt from the fold discipline that obviously applies to a model — but a fitted statistic is a fitted statistic regardless of whether it comes with a `.predict()` method, and the same leakage the imputer commits at the end of this section is committed here one step earlier in the pipeline. It is not restricted to high-cardinality features either; the leak per row shrinks as category counts grow, but it is present at any cardinality.'
      },
      {
        q: 'A behavioural feature scores IV = 0.62. The right response is…',
        options: ['Ship it — anything above 0.3 is strong', 'Investigate for leakage: verify the measurement date precedes the outcome window', 'Bin it more finely', 'Drop it — IV that high is always noise'],
        answer: 1,
        why: 'Suspiciously high IV is a classic leakage signature, and there is a mechanical reason for it: IV is exactly the Jeffreys (symmetrised KL) divergence between the good and bad distributions, derived above, and a divergence like that grows sharply, not gently, as a bin\'s good/bad mix approaches a near-perfect split — which is what leakage typically produces. "Ship it — anything above 0.3 is strong" applies the conventional band literally and misses the point the flag box makes explicitly: those bands are rules of thumb about genuinely predictive features, not a calibrated statistical test, and they say nothing about the far more common failure of a feature that is "strong" because it secretly encodes the outcome. "Drop it — IV that high is always noise" overcorrects in the other direction: a very high IV is a reason to investigate, not an automatic verdict, since a small number of features genuinely are this predictive.'
      },
      {
        q: 'Missingness that depends on the unobserved value itself is called…',
        options: ['MCAR', 'MAR', 'MNAR — and the missing-indicator is informative', 'Censoring'],
        answer: 2,
        why: 'MNAR: the fact of missingness carries signal about the unobserved value — the survey-question analogy above, where people with unusually high or low incomes disproportionately decline to answer — so silently imputing a plausible number actively throws away information, and an explicit indicator column preserves it instead. MCAR and MAR are the tempting near-misses because all three names sound alike and differ only in exactly what the missingness is allowed to depend on: MCAR depends on nothing at all, and MAR depends only on other, already-observed columns — neither describes missingness that depends on the hidden value itself. "Censoring" is a related but distinct concept from survival analysis, where a value is known only to lie above or below some threshold rather than being absent outright.'
      }
    ],
    cards: [
      { q: 'WOE and IV formulas', a: '$\\mathrm{WOE}_i=\\ln(\\%good_i/\\%bad_i)$; $\\mathrm{IV}=\\sum_i(\\%good_i-\\%bad_i)\\mathrm{WOE}_i$. Bands <0.02 / 0.02–0.1 / 0.1–0.3 / >0.3 are conventions.' },
      { q: 'Leakage, defined', a: 'Any feature computed from information unavailable at prediction time. Signature: an implausibly good validation score.' },
      { q: 'Missingness mechanisms', a: 'MCAR (safe to impute), MAR (model-based imputation), MNAR (add an indicator — missingness is informative).' },
      { q: 'Binning rules', a: 'Quantile bins by default; enforce monotone WOE; minimum 5% population and a minimum bad count per bin; fit inside the fold.' },
      { q: 'IV as a divergence', a: '$\\mathrm{IV}=D_{KL}(\\text{good}\\|\\text{bad})+D_{KL}(\\text{bad}\\|\\text{good})$ — the Jeffreys divergence, which is why it blows up on near-empty bins.' }
    ]
  });
})();
