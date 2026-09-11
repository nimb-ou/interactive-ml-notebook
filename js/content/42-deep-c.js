/* ============================================================
   PART 3 — Deep learning, continued: autodiff (3.11), reading a
   training curve (3.12), compression (3.13), robustness (3.14).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 3.11 */
  ML.section({
    id: 'autodiff', track: 'deep', num: '3.11', level: 2,
    title: 'Automatic differentiation and the computation graph',
    lede: 'Backpropagation (§3.2) is one instance of a general algorithm. Understanding the general version explains why activations dominate your memory bill, what gradient checkpointing actually trades, and why <code>.detach()</code> and <code>torch.no_grad()</code> are different things.',
    prereq: ['backprop', 'matrix-calculus'],
    related: ['backprop', 'distributed', 'optimisers'],
    html: `
<p>Section 3.2 derived backpropagation for one fixed shape of network: a stack of dense layers, ReLU in between, four numbered steps you could carry out with a pencil. That was the right way to learn what a gradient <i>is</i>. It is not, however, what happens when you write a transformer block, or a residual connection, or any of the increasingly strange architectures the rest of this course describes, and then call a single method, <code>loss.backward()</code>. Nobody re-derives the four steps of §3.2 by hand for every new architecture. Something more general is running underneath that call — a mechanism that does not know or care whether your forward pass was four dense layers or four hundred different operations wired together in whatever order your Python function happens to call them.</p>
<p>That mechanism is <b>automatic differentiation</b> (autodiff, or AD), and it is worth building properly, because three things about it are routinely misunderstood even by people who use it daily: it is not symbolic differentiation, it is not numerical approximation, and the reason it needs so much memory is not a bug — it is the direct, unavoidable price of the one design decision that makes it fast.</p>

${H.tldr([
      'Autodiff is neither symbolic differentiation nor finite differences. It evaluates the chain rule numerically on a graph of primitive operations, and it is <b>exact</b> to floating-point precision.',
      '<b>Forward mode</b> costs one sweep per input; <b>reverse mode</b> one per output. A loss has one output and $10^9$ inputs, so reverse mode wins by a factor of a billion — and pays for it with memory.',
      'Reverse mode must keep every intermediate activation until its gradient is used. That is why activation memory, not parameter memory, is usually what makes a batch not fit — and why checkpointing (recompute instead of store) is the standard escape.'
    ])}

<h2><span class="sn">3.11.1</span> Three ways to get a derivative, only one of which is used</h2>
<p>Given a function written as code, there are exactly three ways to get its derivative, and it is worth seeing why two of them fail before trusting that the third is not just "the one everyone happens to use".</p>
<p>The first instinct, if you know calculus, is <b>symbolic differentiation</b>: treat the code as a formula and manipulate it algebraically, the way you would by hand on paper, using a computer algebra system to keep track of the bookkeeping. This is exact, and for a short formula it is fine — it is exactly what §0.7 does for a two-parameter linear model. It stops being fine the moment the formula is built by composing many small functions, because differentiating a product does not just differentiate each factor, it also has to write down every cross-term the product rule generates, and those cross-terms are themselves formulas that get differentiated again at the next layer out. The size of the resulting expression can grow exponentially with the depth of the composition — a phenomenon with the deservedly ugly name <b>expression swell</b> — so that the symbolic derivative of a ten-line program can run to pages, and of a network with a million operations, never finishes at all.</p>
<p>The second instinct, if you do not trust the algebra, is to sidestep it entirely: <b>numerical differentiation</b>, nudge one input by a small $\\varepsilon$, measure how much the output moved, and divide. This works on absolutely any function, including ones you cannot see inside — a black box, a physical experiment, someone else's undifferentiable code. Its cost is what rules it out for training: get one partial derivative this way and you have paid for one extra function evaluation; get the gradient with respect to a billion parameters and you have paid for a billion of them, one nudge at a time. Its accuracy is worse than it looks, too — §0.7 shows the error is $O(\\varepsilon^2)$ at best, and shrinking $\\varepsilon$ to fight that error eventually runs into floating-point cancellation and the error grows again. Numerical differentiation earns a permanent, narrow role: checking that an analytic gradient is correct (§0.7), never computing the gradient you actually train with.</p>
${H.table(['Method', 'How', 'Error', 'Cost', 'Verdict'], [
      ['Symbolic', 'manipulate the expression algebraically', 'exact', 'expression swell — the formula can grow exponentially', 'fine for a paper, useless for a network'],
      ['Numerical', 'finite differences', '$O(\\varepsilon^2)$ at best, and it never vanishes', '$O(d)$ function evaluations', 'only for checking (§0.7)'],
      ['<b>Automatic</b>', 'chain rule applied numerically, op by op', '<b>exact</b> up to floating point', '$O(1)$ passes for reverse mode', 'what every framework does']
    ])}
<p>Autodiff is the third way, and the trick that makes it both exact and cheap is refusing to do either of the other two things. It never builds the symbolic expression for the whole function — so there is no expression to swell — and it never approximates anything with a nudge — so there is no truncation error. Instead it takes the one thing every framework already has for free, the sequence of primitive operations your code actually executed, and applies the chain rule to that sequence directly, one small exact local derivative at a time, evaluated at the specific numbers your forward pass produced. It is exact because each of those local derivatives is a known closed-form rule (the derivative of a multiply, of a sine, of a matrix product), not an approximation. It is cheap because, as the next two subsections show, the chain rule can be organised so the whole gradient costs about one extra pass over the graph, however many operations that graph contains.</p>
${H.key('Autodiff is often mistaken for one of the other two. It is exact like symbolic differentiation and cheap like numerical differentiation, because it never builds the symbolic expression — it just evaluates the derivative at the point it is standing on.')}
${H.history(`<p>Reverse-mode autodiff predates deep learning by decades and was reinvented more than once before anyone thought to apply it to a neural network. Seppo Linnainmaa's 1970 master's thesis described what is now called reverse accumulation; Paul Werbos proposed applying essentially the same idea to neural networks in his 1974 PhD thesis; and it was Rumelhart, Hinton and Williams's 1986 paper that finally connected it to training multi-layer networks and gave it the name that stuck, backpropagation. What §3.2 taught you as "the algorithm neural networks use" is really one application of a much older and more general numerical technique — the graph-based engine this section builds works identically whether the function on top is a two-layer perceptron or a physics simulator.</p>`)}

<h2><span class="sn">3.11.2</span> The graph, forwards and backwards</h2>
<p>Building the general engine starts from a single observation about how your forward-pass code is actually structured. Whatever framework you use, the moment you write <code>c = a * b</code> or <code>y = torch.sin(x)</code>, the framework is not just computing a number — it is quietly recording <i>what just happened</i>: which operation ran, and which values fed into it. Do that for every line of a forward pass and you have built, without asking for it explicitly, a directed graph: one node per intermediate quantity, one edge for every value that flowed from one operation into the next. This is the <b>computation graph</b>, and it is the entire data structure autodiff needs.</p>
${H.analogy(`<p>Think of the forward pass as a bucket brigade passing water from a well to a fire, one person handing a bucket to the next. Each person's only job is two-fold: take what you're handed, do your one small thing to it (fill it fuller, or tip some out), and pass it on. Nobody in the middle needs to know where the water started or where the fire is — they only need their neighbours. The computation graph is a record of who handed what to whom.</p>
<p>The backward pass runs the same brigade in reverse, passing not water but <i>blame</i>: starting from the fire ("the loss changed by this much"), each person in turn works out how much of that blame belongs to what they were handed, and passes exactly that much backwards to whoever handed it to them. A person who diluted the water a lot passes back less blame per unit than one who barely touched it — which is precisely the local derivative each node contributes.</p>`)}
<p>Every operation you write becomes a node. Each node knows two things: how to compute its output from its inputs (forward), and how to turn "the gradient of the loss with respect to my output" into "the gradient with respect to each of my inputs" (backward, the <b>vector-Jacobian product</b>). Nothing else is needed. A framework's entire library of layers is, underneath, a catalogue of nodes each implementing exactly these two methods, and building a custom layer means writing exactly these two methods and nothing more.</p>

<p><b>What you are looking at.</b> Seven boxes are the nodes of a tiny computation graph for $L = (a\\cdot b + \\sin c)^2$: the three leaves $a$, $b$, $c$ on the left, the two products of them, $u = a\\cdot b$ and $v=\\sin c$, in the middle, their sum $w = u+v$ after that, and the final loss $L = w^2$ on the right. Arrows carry values forward, left to right, through exactly the operations that expression describes. The three sliders above the diagram set the actual numbers $a$, $b$ and $c$ feed in.</p>
<p><b>What to do with it.</b> Press play and watch the animation step through two phases. In the forward phase, boxes fill in with their computed value $v$ one at a time, left to right, exactly the bucket brigade handing water forward. In the backward phase, the same boxes now fill in a second number, $\\bar g = \\partial L/\\partial(\\text{that node})$, starting from $L$ itself (whose own adjoint is trivially 1) and working right to left. The readout beneath confirms the three gradients you actually want, $\\partial L/\\partial a$, $\\partial L/\\partial b$, $\\partial L/\\partial c$, and a <i>numerical check</i> column recomputes $\\partial L/\\partial a$ by central differences (§0.7) so you can watch the two methods agree to four decimals.</p>
<p><b>The thing genuinely worth noticing.</b> Node $w$ receives its adjoint from a single node, $L$, but node $u$ and node $v$ each hand their adjoint off to only one place too — except that when you trace where $w$'s own contribution came from, it split and travelled backward into <i>both</i> $u$ and $v$ at once, because both fed $w$ on the way forward. Watch the engine accumulate with <code>+=</code> rather than overwrite at exactly that point. A node that is reused by two downstream computations receives two separate contributions to its adjoint during the backward pass, and forgetting to add them — overwriting instead of accumulating — is the single most common bug in a hand-rolled autodiff engine, and the reason every real one guards against it explicitly.</p>
${H.lab('graph', 'A computation graph, differentiated in front of you', 'A real reverse-mode engine, ten lines of it, running on the expression shown. Step forward to fill in the values; step backward to watch adjoints accumulate from the loss end. Every number here is computed, not annotated.')}

<p>Now make the pattern from the lab precise enough to code. Backpropagation's $\\delta^{(l)}$ from §3.2 — "the gradient of the loss with respect to this layer's output" — generalises directly: call it $\\bar v$, the <b>adjoint</b> of any node $v$ in the graph, and the entire backward pass is nothing but computing every node's adjoint from the adjoints of the nodes that used it.</p>
${H.deriv('the vector-Jacobian product, and why it is the right primitive', [
      ['$\\bar{v} := \\dfrac{\\partial \\mathcal{L}}{\\partial v}$', 'Define the <i>adjoint</i> of every intermediate $v$: how much the loss moves when $v$ moves. It has the same shape as $v$.'],
      ['$\\bar{u} = \\bar{v}^\\top \\dfrac{\\partial v}{\\partial u}$', 'The chain rule for a node $v = f(u)$. This is a row vector times a Jacobian — <b>a vector-Jacobian product</b>, written <code>vjp</code>.'],
      ['$\\bar{u} \\mathrel{+}= \\bar{v}^\\top J_f$', 'Accumulate with <code>+=</code>, because $u$ may feed several nodes and each contributes. Forgetting to accumulate is the classic hand-rolled-autodiff bug.'],
      ['never form $J_f$', 'For $v = Wu$ the Jacobian is $W$ and $\\bar u = W^\\top\\bar v$ — a matrix-vector product, never an explicit Jacobian. For an elementwise $\\sigma$, $\\bar u = \\bar v \\odot \\sigma\'(u)$. <b>Every primitive ships its vjp; none ships a Jacobian.</b>']
    ], 'This is why writing a custom layer means implementing <code>forward</code> and <code>backward</code> and nothing else: the framework only ever asks a node to do a vjp.')}

<p>The vjp only says how to move adjoints across <i>one</i> node. It leaves open a real choice: in what order do you visit the whole graph? There are exactly two consistent answers, and they are not equally good for the same job.</p>
<p>You could start at the <i>inputs</i> and carry derivative information forward alongside every value, the same direction the forward pass already runs — this is <b>forward mode</b>. Or you could run the forward pass first to build the graph, then start at the <i>output</i> and walk backward exactly as the lab above just did — this is <b>reverse mode</b>. Both are legitimate applications of the chain rule; they differ only in which end of the graph they start from, and that single choice changes the cost by orders of magnitude depending on the graph's shape.</p>
${H.vs('Forward mode', [
      'Carries a derivative alongside every value (dual numbers)',
      'One sweep gives $\\partial(\\text{all outputs})/\\partial(\\text{one input})$',
      'Cost: $O(n_{\\text{inputs}})$ sweeps',
      'No memory of the forward pass needed',
      'Right for: few inputs, many outputs — Jacobian-vector products, sensitivity analysis'
    ], 'Reverse mode', [
      'Records the graph, then walks it backwards',
      'One sweep gives $\\partial(\\text{one output})/\\partial(\\text{all inputs})$',
      'Cost: $O(n_{\\text{outputs}})$ sweeps ≈ 2× a forward pass',
      '<b>Must store every activation</b> until its gradient is consumed',
      'Right for: training anything with a scalar loss'
    ])}
${H.intuition(`<p>The reason reverse mode wins for training is entirely about the shape of the function being differentiated, and it is worth stating as bluntly as possible: a loss function takes in everything — every parameter, potentially billions of them — and hands back exactly one number. Forward mode pays per input, so differentiating a billion-input function costs a billion sweeps. Reverse mode pays per output, so differentiating a one-output function costs one sweep, regardless of how many inputs fed it. Training a neural network is, structurally, always this shape: many-in, one-out. That is not a coincidence you need to check for each new architecture; it is guaranteed the instant you define a loss as a single scalar to minimise, and it is the entire reason every deep learning framework is built around reverse mode and treats forward mode as a niche tool for the rare occasion you have few inputs and many outputs instead — computing how sensitive an entire output vector is to one parameter, say, or certain Hessian-vector products.</p>`)}

<h2><span class="sn">3.11.3</span> Memory: the real cost, and checkpointing</h2>
<p>Reverse mode's bill comes due on the last row of the "reverse mode" list above: it must store every activation until its gradient is consumed. Trace back through why. Step 4 of §3.2's algorithm computes a weight gradient as $\\delta^{(l)}a^{(l-1)\\mathsf{T}}$ — the adjoint arriving at a layer, outer-producted with whatever that layer <i>received</i> on the way forward. If that received value was not kept around, computing this one gradient would mean re-running the forward pass from scratch to regenerate it. So reverse mode's entire graph — every intermediate node's value — has to survive from the moment it is computed in the forward pass to the moment its gradient is used in the backward pass, which for the earliest layers is nearly the whole length of training that one example. This is not a quirk of any one framework's implementation. It is what "one sweep per output" actually requires: you cannot walk the graph backwards using values you threw away on the way forward.</p>
<p>Concretely, for a transformer, the memory during training splits roughly into four buckets:</p>
${H.table(['Bucket', 'Size (bf16 weights, Adam)', 'Scales with'], [
      ['Parameters', '2 bytes × $N$', 'model size'],
      ['Gradients', '2 bytes × $N$', 'model size'],
      ['Optimiser state (Adam m, v, fp32 master)', '~12 bytes × $N$', 'model size'],
      ['<b>Activations</b>', '$O(L \\cdot B \\cdot S \\cdot d)$', '<b>batch × sequence × depth</b>']
    ])}
<p>The first three buckets are fixed once the model is chosen — they do not care how big a batch you feed it. The fourth does not care about the model's size in the same way at all; it grows with how much data you push through in one go and how deep the network is. This is exactly why increasing batch size is often what tips a training run from fitting to not fitting, even though the parameter count on disk has not moved an inch — a fact worth having ready, because "we didn't change the model, why did it OOM?" is one of the most common confusions in the field.</p>
${H.worked('why a 7B model needs ~112 GB to train, and 14 GB to serve', `
<p>7B parameters in bf16: weights 14 GB, gradients 14 GB, Adam state (fp32 momentum + variance + master weights) 84 GB → <b>112 GB before a single activation</b>. That is why a 7B model does not fine-tune on one 80 GB card without LoRA (§4.13), ZeRO sharding (§4.11) or 8-bit optimiser states.</p>
<p>Inference needs only the 14 GB of weights plus the KV cache (§4.7). <b>The 8× gap between training and serving memory is entirely optimiser state and gradients</b> — a number worth having ready, because it is the first thing asked when someone says "can we fine-tune this".</p>`)}

<p>Activation memory looks unavoidable — store everything, or cannot compute the gradient, seemingly full stop. <b>Gradient checkpointing</b> is the observation that this is a false choice, because there is a third option nobody considers at first: store <i>some</i> activations and recompute the rest on demand. Instead of caching every layer's output, cache only every $k$-th layer — the checkpoints — and when the backward pass needs an activation that was not kept, replay a short stretch of forward computation from the nearest surviving checkpoint to regenerate it, use it, then discard it again. You are deliberately re-introducing part of the cost that caching exists to avoid, in exchange for a memory bill that no longer grows with the network's full depth.</p>
<p><b>What you are looking at.</b> The plot traces peak activation memory, in megabytes, as a function of the checkpoint interval $k$ you choose, for a network of $L$ layers each costing a fixed amount of activation memory per layer — both set by the sliders. The dashed horizontal line marks the memory cost of storing everything ($k=1$, no checkpointing at all); the dashed vertical line marks $k=\\sqrt L$, the value that minimises the curve. The dot sits at whatever $k$ you have currently dialled in.</p>
<p><b>What to do with it.</b> Drag $k$ from 1 upward and watch memory fall steeply at first, then bottom out, then start climbing again as $k$ gets so large that each recomputed segment itself becomes long and expensive to hold live. Move the vertical dashed line by changing $L$ and confirm the optimum always sits at $\\sqrt L$, never anywhere else — that is not a rule of thumb, it falls straight out of calculus once you write memory as a function of $k$ and minimise it, which the note below the lab does explicitly.</p>
<p><b>The thing genuinely worth noticing.</b> At $k=1$ you are back to storing every activation — no recomputation, maximum memory. At $k=L$ you store almost nothing and recompute almost the entire forward pass for every single backward step — minimum memory, maximum compute. The curve between those two extremes is not linear; it drops fast near $k=1$ and flattens out, which is why even a modest $k$ — nowhere near the true optimum — already recovers most of the possible saving. You do not need to tune this precisely to benefit from it.</p>
${H.lab('ckpt', 'Gradient checkpointing: the memory/compute trade', 'Store every activation, or store only every $k$-th and recompute the rest during the backward pass. The optimum is at $k=\\sqrt{L}$, which turns $O(L)$ memory into $O(\\sqrt{L})$ for about 30% extra compute.')}

<p>Checkpointing is the general-purpose tool, but it is one entry in a family of tricks that all attack the same activation-memory bill from different angles — some by storing less, some by never storing at all, some by not needing the gradient in the first place. Knowing which one applies to which situation is worth more than knowing any single one in depth:</p>
${H.table(['Trick', 'Saves', 'Costs', 'Where'], [
      ['Gradient checkpointing', 'activation memory → $O(\\sqrt L)$', '~30% more compute', 'the standard first move when OOM'],
      ['Gradient accumulation', 'peak activation memory', 'nothing but wall-clock', 'simulating a large batch on small hardware'],
      ['Mixed precision (bf16)', 'half the activation and weight bytes', 'needs fp32 master weights', 'universal since 2020 (§1.15)'],
      ['<code>torch.no_grad()</code>', '<b>all</b> activation storage', 'no gradients at all', 'inference and evaluation'],
      ['<code>.detach()</code>', 'the history behind one tensor', 'that path gets no gradient', 'stop-gradient in BYOL, target networks in RL'],
      ['Fused / flash kernels', 'materialising the attention matrix', 'kernel complexity', 'FlashAttention (§4.7)']
    ])}
${H.pitfall('<code>no_grad()</code> and <code>detach()</code> are not interchangeable. <code>no_grad()</code> is a <i>context</i>: nothing inside it records history at all, so no memory is spent. <code>detach()</code> returns a <i>tensor</i> cut off from its history, but the graph that produced it still exists and is still holding memory if anything else references it. Using <code>detach()</code> in an evaluation loop and wondering why memory still climbs is a rite of passage.')}

${H.probe([
      ['Why does reverse-mode autodiff use so much memory?', 'The backward pass needs each node’s inputs to compute its vjp, so every activation must survive from the forward pass until its gradient is consumed. Memory scales with depth × batch × sequence.'],
      ['What exactly does gradient checkpointing trade?', 'Compute for memory: it stores activations only at checkpoint boundaries and recomputes the segments in between during the backward pass. $O(L)\\to O(\\sqrt L)$ memory for roughly one extra forward pass.'],
      ['When is forward mode the right choice?', 'When you want a Jacobian-vector product — directional derivatives, one-input sensitivity, some Hessian-vector product formulations. Rarely for training.'],
      ['A model trains at batch 8 and OOMs at 16, but the parameter count did not change. Why?', 'Activation memory scales with batch size; parameters do not. Checkpointing, accumulation or a shorter sequence are the fixes.']
    ], 'Saying autodiff is "just symbolic differentiation" or "just finite differences". It is neither, and the distinction is exactly what makes it work.')}`,
    labs: {
      graph: function (host) {
        /* a genuine ten-line reverse-mode engine */
        function V(val, children, op, back) {
          return { v: val, g: 0, children: children || [], op: op || 'leaf', back: back || function () {}, name: '' };
        }
        function mul(a, b) { const o = V(a.v * b.v, [a, b], '×'); o.back = () => { a.g += b.v * o.g; b.g += a.v * o.g; }; return o; }
        function add(a, b) { const o = V(a.v + b.v, [a, b], '+'); o.back = () => { a.g += o.g; b.g += o.g; }; return o; }
        function sin(a) { const o = V(Math.sin(a.v), [a], 'sin'); o.back = () => { a.g += Math.cos(a.v) * o.g; }; return o; }
        function sq(a) { const o = V(a.v * a.v, [a], '( )²'); o.back = () => { a.g += 2 * a.v * o.g; }; return o; }

        const st = Viz.controls(host, [
          { k: 'a', label: 'a', min: -3, max: 3, step: .1, value: 2, fmt: v => v.toFixed(1) },
          { k: 'b', label: 'b', min: -3, max: 3, step: .1, value: -1.5, fmt: v => v.toFixed(1) },
          { k: 'c', label: 'c', min: -3, max: 3, step: .1, value: 0.6, fmt: v => v.toFixed(1) }
        ], () => build());

        const stage = ML.el('div');
        host.appendChild(stage);
        let phase = 0;

        function graph() {
          const a = V(st.a); a.name = 'a';
          const b = V(st.b); b.name = 'b';
          const c = V(st.c); c.name = 'c';
          const ab = mul(a, b); ab.name = 'u = a·b';
          const sc = sin(c); sc.name = 'v = sin(c)';
          const s = add(ab, sc); s.name = 'w = u + v';
          const L = sq(s); L.name = 'L = w²';
          return { a, b, c, ab, sc, s, L };
        }

        function backward(L) {
          const order = [], seen = new Set();
          (function visit(n) { if (seen.has(n)) return; seen.add(n); n.children.forEach(visit); order.push(n); })(L);
          order.forEach(n => { n.g = 0; });
          L.g = 1;
          for (let i = order.length - 1; i >= 0; i--) order[i].back();
          return order;
        }

        let S = null;
        function build() {
          stage.innerHTML = '';
          const g = graph();
          backward(g.L);
          const nodes = [
            { id: 'a', x: .02, y: .05, w: .17, h: .2, label: 'a', ref: g.a },
            { id: 'b', x: .02, y: .40, w: .17, h: .2, label: 'b', ref: g.b },
            { id: 'c', x: .02, y: .75, w: .17, h: .2, label: 'c', ref: g.c },
            { id: 'u', x: .28, y: .22, w: .19, h: .2, label: 'u = a·b', ref: g.ab },
            { id: 'v', x: .28, y: .75, w: .19, h: .2, label: 'v = sin c', ref: g.sc },
            { id: 'w', x: .55, y: .45, w: .18, h: .2, label: 'w = u+v', ref: g.s },
            { id: 'L', x: .81, y: .45, w: .17, h: .2, label: 'L = w²', ref: g.L }
          ];
          const edges = [
            { from: 'a', to: 'u' }, { from: 'b', to: 'u' }, { from: 'c', to: 'v' },
            { from: 'u', to: 'w' }, { from: 'v', to: 'w' }, { from: 'w', to: 'L' }
          ];
          const order = ['a', 'b', 'c', 'u', 'v', 'w', 'L'];

          S = Viz.surface(stage, {
            height: 250,
            draw: function (ctx, w2, h, T) {
              const fwd = phase <= 6;
              const step = phase;
              const shown = fwd ? order.slice(0, step + 1) : order.slice();
              const gradShown = fwd ? [] : order.slice().reverse().slice(0, step - 6);
              const nd = nodes.map(n => {
                const on = shown.indexOf(n.id) >= 0;
                const gon = gradShown.indexOf(n.id) >= 0;
                return {
                  id: n.id, x: n.x, y: n.y, w: n.w, h: n.h,
                  label: n.label,
                  sub: on ? (gon ? 'v=' + n.ref.v.toFixed(2) + '  ḡ=' + n.ref.g.toFixed(2) : 'v=' + n.ref.v.toFixed(2)) : '—',
                  fill: gon ? 'color-mix(in oklab, ' + T.c2 + ' 22%, ' + T.panel + ')'
                    : on ? 'color-mix(in oklab, ' + T.c1 + ' 15%, ' + T.panel + ')' : T.panel,
                  accent: gon ? T.c2 : on ? T.c1 : T.line
                };
              });
              const ed = edges.map(e => ({
                from: e.from, to: e.to,
                color: (!fwd && gradShown.indexOf(e.to) >= 0) ? T.c2 : (shown.indexOf(e.to) >= 0 ? T.c1 : T.line),
                width: 1.8
              }));
              Viz.flow(ctx, w2, h, nd, ed, { pad: 12 });
              ctx.fillStyle = T.muted; ctx.font = '11.5px ui-sans-serif'; ctx.textAlign = 'left';
              ctx.fillText(fwd ? '→ forward pass: computing values' : '← backward pass: accumulating adjoints ḡ = ∂L/∂node', 12, 14);
            }
          });
          const out = Viz.readout(stage, [
            { k: 'L', label: 'L', cls: 'key' },
            { k: 'ga', label: '∂L/∂a' }, { k: 'gb', label: '∂L/∂b' }, { k: 'gc', label: '∂L/∂c' },
            { k: 'chk', label: 'numerical check' }
          ]);
          const f = (A, B, C) => Math.pow(A * B + Math.sin(C), 2);
          const h = 1e-5;
          out({
            L: g.L.v.toFixed(4),
            ga: g.a.g.toFixed(4), gb: g.b.g.toFixed(4), gc: g.c.g.toFixed(4),
            chk: ((f(st.a + h, st.b, st.c) - f(st.a - h, st.b, st.c)) / (2 * h)).toFixed(4) + ' (∂a)'
          });
          Viz.player(stage, {
            frames: 13, fps: 1.4, repeat: true,
            label: i => i <= 6 ? 'forward ' + (i + 1) + '/7' : 'backward ' + (i - 6) + '/6',
            onFrame: i => { phase = i; if (S) S.redraw(); }
          });
        }
        build();
        Viz.note(host, 'The <b>numerical check</b> readout recomputes ∂L/∂a by central differences and matches the autodiff answer to four decimals — the same check as §0.7, now verifying an engine rather than a formula. Note that the backward pass touches each node exactly once, in reverse topological order, and that node <code>w</code> pushes its adjoint into <i>both</i> <code>u</code> and <code>v</code> with <code>+=</code>. That accumulation is the whole reason a node can be reused.');
      },

      ckpt: function (host) {
        const st = Viz.controls(host, [
          { k: 'L', label: 'layers L', min: 8, max: 128, step: 4, value: 48, fmt: v => v },
          { k: 'perLayer', label: 'activation memory per layer (MB)', min: 20, max: 800, step: 20, value: 220, fmt: v => v + ' MB' },
          { k: 'k', label: 'checkpoint every k layers', min: 1, max: 24, step: 1, value: 7, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'none', label: 'no checkpointing', cls: 'bad' },
          { k: 'now', label: 'at your k', cls: 'key' },
          { k: 'best', label: 'optimum k = √L', cls: 'good' },
          { k: 'compute', label: 'extra compute' },
          { k: 'save', label: 'memory saved' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const L = st.L, M = st.perLayer;
            // store L/k checkpoints, plus one segment of k activations live at a time
            const mem = k => (L / k + k) * M;
            const kOpt = Math.max(1, Math.round(Math.sqrt(L)));
            const P = Viz.plot(ctx, w, h, {
              xd: [1, Math.min(24, L)], yd: [0, mem(1) * 1.05],
              pad: { l: 62, r: 14, t: 16, b: 40 }
            }).frame({ xlabel: 'checkpoint interval k', ylabel: 'peak activation memory (MB)' });
            P.clip(() => {
              P.area(Num.linspace(1, Math.min(24, L), 90).map(k => [k, mem(k)]), { color: T.c1, alpha: .12 });
              P.fn(mem, { color: T.c1, width: 2.6, n: 160, from: 1, to: Math.min(24, L) });
              P.hline(mem(1), { color: T.c2, dash: [5, 4], label: 'store everything' });
              P.vline(kOpt, { color: T.c3, dash: [4, 4], label: 'k = √L' });
              P.dots([[st.k, mem(st.k)]], { r: 5.5, color: T.c4, stroke: true, strokeWidth: 2 });
            });
            out({
              none: (mem(1) / 1024).toFixed(1) + ' GB',
              now: (mem(st.k) / 1024).toFixed(2) + ' GB',
              best: 'k=' + kOpt + ' → ' + (mem(kOpt) / 1024).toFixed(2) + ' GB',
              compute: '+' + ((1 - 1 / st.k) * 33).toFixed(0) + '% (≈1 extra fwd)',
              save: ((1 - mem(st.k) / mem(1)) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.note(host, 'The curve is $\\left(\\frac{L}{k}+k\\right)M$ — checkpoints stored plus one live segment — and it is minimised at $k=\\sqrt L$ by AM–GM. At 48 layers that is $k=7$: <b>10.5 GB becomes 2.98 GB, a 71.7% saving, for roughly one extra forward pass.</b> This single trick is what lets a long-context model train on the hardware you have.');
      }
    },
    quiz: [
      {
        q: 'Reverse-mode autodiff is memory-hungry because…',
        options: ['it stores the symbolic expression', 'every activation must survive until its gradient is consumed', 'it uses float64', 'it recomputes the forward pass'],
        answer: 1,
        why: 'Every node’s backward step needs its forward inputs to compute the vector-Jacobian product — for $v=Wu$, forming $\\bar u = W^\\top\\bar v$ needs $W$, and for gradients further back it needs $u$ itself — so nothing can be discarded on the way forward until the matching gradient has actually been consumed on the way back. That is a structural property of reverse mode, not an implementation detail, which is why every framework pays it. "It stores the symbolic expression" is the tempting wrong answer because it describes real behaviour of the other method from §3.11.1, symbolic differentiation, whose expressions can grow exponentially with composition depth (expression swell) — autodiff was built precisely to avoid ever constructing that expression, so crediting it with the other method’s cost is an easy but backwards mistake. Float64 and recomputing the forward pass are both false: autodiff runs at whatever precision your tensors already use, and reverse mode is expensive exactly because it does not recompute — it caches, which is what §3.11.3 goes on to trade away deliberately via checkpointing.'
      },
      {
        q: 'The primitive every autodiff node implements is…',
        options: ['the full Jacobian', 'a vector-Jacobian product', 'a Hessian-vector product', 'a finite difference'],
        answer: 1,
        why: 'Every node’s backward method computes $\\bar u = \\bar v^\\top \\partial v/\\partial u$ — the adjoint arriving from downstream, multiplied against the local Jacobian — and for $v=Wu$ that multiplication is literally $\\bar u = W^\\top\\bar v$, a matrix-vector product that never needs the Jacobian written out as its own object. "The full Jacobian" is the tempting wrong answer because it sounds like the more thorough, complete thing to compute, and for a hand-derived toy example it may be what you would actually write on paper — but for a layer mapping thousands of inputs to thousands of outputs the Jacobian is a matrix with millions of entries that is never needed, only its product with one vector. A Hessian-vector product is a genuine object elsewhere (curvature estimation, some second-order optimisers) but it is not what a single node’s backward method computes. §3.11.2’s derivation exists to make this concrete: "never form $J_f$" is stated as a rule because every primitive in every framework ships a vjp, and none ships a Jacobian.'
      },
      {
        q: 'Gradient checkpointing with $k=\\sqrt{L}$ turns activation memory from $O(L)$ into…',
        options: ['$O(1)$', '$O(\\sqrt L)$', '$O(\\log L)$', '$O(L/2)$'],
        answer: 1,
        why: 'Peak activation memory as a function of checkpoint interval is $(L/k + k)M$ — checkpoints stored plus one live recomputation segment — and by AM–GM that sum is minimised exactly when the two terms are equal, at $k=\\sqrt L$, giving $2\\sqrt L$ segments rather than $L$: that is $O(\\sqrt L)$, not merely some fixed fraction of $L$. "$O(L/2)$" is the tempting wrong answer because "trading memory for compute" sounds like it should mean cutting the bill in half, the intuitive reading of any trade-off — but the actual curve is not linear in $k$, and a fixed fraction of $L$ stays $O(L)$ regardless of which fraction, so it buys no improvement as models get deeper. The gap is not academic: at $L=48$, $2\\sqrt{48}\\approx13.9$ against $L/2=24$, and the advantage of $\\sqrt L$ over any fixed fraction only grows as $L$ does. This is the calculus §3.11.3 walks through explicitly — minimise the memory function, do not guess at the trade — and it is the same discipline behind the section’s worked 112 GB training versus 14 GB serving split: know which term in a resource bill actually scales with what.'
      },
      {
        q: 'Roughly how much memory does Adam add on top of bf16 weights?',
        options: ['none', 'about 2 bytes per parameter', 'about 12 bytes per parameter', 'about 100 bytes per parameter'],
        answer: 2,
        why: 'Adam keeps two extra fp32 tensors per parameter — momentum and the second moment — plus an fp32 master copy of the weight for the update to accumulate into safely, and $4+4+4=12$ bytes is exactly the optimiser-state row in §3.11.3’s table, which is why a 7B model’s Adam state alone comes to roughly $7\\times10^9\\times12\\approx84$ GB. "About 2 bytes per parameter" is the tempting wrong answer because 2 bytes is the size of one bf16 tensor, the same size already charged separately for the weights and again for the gradients in the table’s first two rows — mistaking Adam’s addition for "one more copy like the ones you already have" undercounts it sixfold, since Adam is three fp32 tensors, not one bf16 one. "About 100 bytes" overshoots by nearly an order of magnitude, and "none" ignores that Adam has to track statistics at all. The general point, and the reason the 112 GB training versus 14 GB serving split matters, is that parameter, gradient and optimiser memory are all fixed the instant you choose a model, while only activation memory moves with batch size — so an OOM after changing only the batch is never explained by any of these three buckets.'
      }
    ],
    cards: [
      { q: 'Forward vs reverse mode', a: 'Forward: one sweep per input. Reverse: one per output. A scalar loss with many parameters makes reverse mode a billion times cheaper.' },
      { q: 'What a node must implement', a: 'Forward, and a vector-Jacobian product. Never an explicit Jacobian.' },
      { q: 'Training memory for a 7B model', a: '14 GB weights + 14 GB grads + ~84 GB Adam ≈ 112 GB, before activations. Serving needs 14 GB + KV cache.' },
      { q: 'Checkpointing trade', a: '$O(L)\\to O(\\sqrt L)$ activation memory for roughly one extra forward pass (~30% compute).' },
      { q: 'no_grad vs detach', a: '<code>no_grad</code> stops history being recorded at all; <code>detach</code> cuts one tensor loose but the graph behind it may still be held.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.12 */
  ML.section({
    id: 'training-dynamics', track: 'deep', num: '3.12', level: 2,
    title: 'Reading a training curve',
    lede: 'The most valuable diagnostic skill in deep learning, and the one least often taught: looking at a loss curve and naming the cause. Every shape here has one, and interviewers show these plots because a candidate who can read them has actually trained something.',
    prereq: ['optimisers'],
    related: ['optimisers', 'normalisation', 'distributed'],
    html: `
<p>You launch a training run, walk away, and come back three hours later to a plot. Nobody hands you the model's internals at that point — no debugger, no stack trace, just one wobbling line going, roughly, down. And yet an experienced practitioner can look at that line for about five seconds and say something specific: "your learning rate is too high", or "that's a bad shard around step 40,000", or "nothing is wrong, that's just the schedule". None of that comes from re-deriving optimisation theory on the spot. It comes from having seen each of these shapes before, with its cause already confirmed, enough times that the shape and the cause have fused into one piece of knowledge. This section is an attempt to hand you that fusion directly, for the ten shapes that account for nearly everything you will ever see on a loss curve.</p>
${H.analogy(`<p>A cardiologist reading an ECG strip is doing the same kind of pattern-matching. They are not deriving the electrophysiology of the heart from first principles at the bedside; they learned, once, which waveform goes with which condition, and now the shape <i>is</i> the diagnosis, read in seconds. A loss curve is your model's vital signs, plotted over time instead of space, and this section is the equivalent of learning to read the strip: flat is not "nothing is happening", it is a specific and nameable failure, exactly the way a flat line on a heart monitor is not silence but a very specific piece of bad news.</p>`)}
${H.history(`<p>The single most load-bearing habit in this section — overfit a tiny batch before you do anything else — is not folklore; it has a clear modern source. Andrej Karpathy's 2019 essay <i>A Recipe for Training Neural Networks</i> crystallised it as step one of a disciplined training process, after observing that "neural net training fails silently" in ways ordinary software failures do not: a badly wired network does not usually crash, it just quietly trains to a mediocre number, and every symptom looks plausible on its own. Deliberately trying to overfit a handful of examples turns a silent failure into a loud one — if the network cannot memorise ten examples, something upstream of the model is broken, and that is far cheaper to learn in the first two minutes than after a three-hour run.</p>`)}

${H.tldr([
      'Before anything else: <b>overfit a batch of ten examples to zero loss</b>. If you cannot, the bug is in the data, the loss or the label alignment — not in the model.',
      'Flat loss = learning rate too low, dead activations, or no gradient path. Exploding loss = learning rate too high, no clipping, or fp16 overflow. Sawtooth = the learning-rate schedule. Step change = a data problem.',
      'Train and validation curves separating is <i>not</i> automatically bad. What matters is whether validation is still improving.'
    ])}

<h2><span class="sn">3.12.1</span> The shapes, and what causes them</h2>
<p>Two questions do almost all of the diagnostic work, and it is worth asking them before reaching for the table below. First: <b>is the curve bending, or is it stepping?</b> A bend — a gradual change in slope — is what an optimisation process does: the learning rate, the loss landscape, the schedule are all continuous things, and continuous causes produce continuous-looking effects on the curve. A step — a sudden, near-vertical jump with no ramp either side — is what a discrete event does: a new data shard starting, a corrupted batch, a code deploy mid-run. Optimisation problems bend curves; data and infrastructure problems step them. Second: <b>does it recover, or is it fatal?</b> A spike that comes back down within a few dozen steps is very different from one that plateaus at the new, worse level forever, even though both start out looking identical for the first several steps. Keep both questions in mind while reading the table, because they are what turns "the loss did something weird" into "the loss did this specific, named thing".</p>
${H.table(['Shape', 'Most likely cause', 'Confirm by', 'Fix'], [
      ['<b>Flat from step 0</b>', 'LR too low, or gradients are not reaching the parameters', 'print grad norms per layer; check <code>requires_grad</code>', 'raise LR; check the graph is connected'],
      ['<b>Loss = NaN at step k</b>', 'LR too high, fp16 overflow, or log(0)', 'find the <i>first</i> NaN with per-op hooks', 'clip gradients; bf16; stabilise the loss (§1.15)'],
      ['<b>Rises then plateaus high</b>', 'LR above the stability threshold', 'LR range test', 'reduce LR ~10×; add warmup'],
      ['<b>Falls, then spikes, then recovers</b>', 'a bad batch, or a loss-scaling event', 'log the batch index of the spike', 'gradient clipping; skip-on-spike; data cleaning'],
      ['<b>Sawtooth</b>', 'cyclic or restarting LR schedule', 'plot the LR alongside', 'nothing — this is intended'],
      ['<b>Staircase down</b>', 'step LR decay', 'plot the LR', 'nothing'],
      ['<b>Train ↓, val ↑</b>', 'overfitting', 'gap widens with epochs', 'regularise, augment, early-stop'],
      ['<b>Val below train</b>', 'dropout/augmentation active in training only, or an easier val split', 'evaluate train in eval mode', 'usually benign — verify the split'],
      ['<b>Sudden step change</b>', 'a data pipeline event: shard boundary, corrupt file, distribution change', 'plot loss against data index, not step', 'audit the shard'],
      ['<b>Loss falls, metric flat</b>', 'optimising the wrong thing', 'compare loss and metric per epoch', 'change the loss or the threshold (§2.13)']
    ])}

${H.pitfall('The table above resolves each shape to a single "most likely cause", and that phrasing is doing real work: it is the most likely cause, not the only one that can produce that shape. A flat loss almost always means the learning rate is too low or the gradient path is broken, but it can also mean the loss function itself has a plateau built in — a threshold-based metric standing in for a loss, say. Treat the table as where to look first, not as a lookup table you can stop thinking after consulting.')}

<p><b>What you are looking at.</b> A single loss curve, generated by a real underlying mechanism rather than drawn by hand — the code that produces "diverge" genuinely diverges, the code that produces "NaN" genuinely divides by something that goes to zero. Some cases plot a second curve in a different colour: validation, laid over training, exactly the pairing the table's last few rows describe.</p>
<p><b>What to do with it.</b> Before clicking anything, say out loud which cause you think produced the curve on screen — bending or stepping, recovering or not — using the two questions above. Only then click one of the three answer choices. Right or wrong, the explanation beneath tells you not just the correct cause but what to check to <i>confirm</i> it, which is the more useful skill: naming a plausible cause is easy, confirming it against alternatives is where the diagnostic actually pays for itself.</p>
<p><b>The thing genuinely worth noticing.</b> Work through all ten and you will find the cases cluster into families faster than you expect: three or four different-looking curves all resolve to "check the learning rate", two resolve to "this is fine, look at the schedule", and the data-pipeline cases are the ones with the sharpest, most step-like discontinuity, every time. That clustering — a handful of underlying causes producing a wide variety of surface shapes — is the actual content of this section, more than any individual curve.</p>
${H.lab('diagnose', 'Diagnose the run', 'Ten real pathologies, generated with their actual mechanism, shown one at a time. Name the cause before you reveal the answer. This is the exercise, in miniature, that interviewers run when they show you a plot.')}

<h2><span class="sn">3.12.2</span> The learning-rate range test</h2>
<p>Reading a curve after the fact tells you what went wrong. The learning-rate range test is the corresponding tool for <i>before</i> the fact: rather than guess a learning rate and watch three hours later whether it was too high, run a short, cheap experiment that answers the question in a couple of minutes. The idea, due to Leslie Smith's 2017 <i>Cyclical Learning Rates</i> paper, is disarmingly simple. Take the network you are about to train, and instead of picking one learning rate and committing to it, sweep the learning rate exponentially upward over a few hundred steps — start absurdly small, end absurdly large — training for just a handful of steps at each value and recording the loss. Plot loss against the learning rate that produced it, on a log axis, and a strikingly consistent shape appears: flat while the rate is too small to move anything, then a descending region as the rate becomes useful, then a sharp, unmistakable rise as the rate becomes large enough to destabilise training.</p>
<p><b>Pick roughly one order of magnitude below the minimum</b> of that curve — the steepest-descent region, not the very bottom — and that is your learning rate, chosen in minutes rather than guessed and discovered wrong three hours in.</p>
<p><b>What you are looking at.</b> The x-axis is learning rate on a log scale, spanning several orders of magnitude; the y-axis is the loss reached after a short burst of training at that rate, starting fresh each time from the same initialisation. The solid line traces loss against rate. Two dashed verticals mark the rate at the curve's literal minimum and the suggested rate — steepest-descent, roughly a decade below the minimum — with the shaded region past the right-hand dashed line marking where training diverged outright.</p>
<p><b>What to do with it.</b> Watch the curve trace out live as it sweeps: flat on the left, a clear descending slope, then a cliff. Switch the network from shallow to deep and watch the whole usable window — the width of the descending region — narrow considerably; switch the activation from ReLU to tanh and watch the entire curve shift left, tolerating only smaller rates before the cliff arrives.</p>
<p><b>The thing genuinely worth noticing.</b> The suggested rate is never at the bottom of the curve, and that is not a conservative safety margin tacked on afterwards — it is the point. Training for thousands of steps sharpens the loss landscape in ways a hundred-step burst cannot show you, so a rate that looks optimal on this short sweep is already living dangerously close to the instability cliff once training has run for a while.</p>
${H.lab('lrfind', 'The LR range test, on a network that really trains', 'A genuine MLP, genuinely trained at each learning rate for a short burst. The suggested value is the point of steepest descent, which is the standard heuristic.')}
${H.note('Why one order of magnitude below the minimum? Because at the minimum of this curve you are already close to the instability threshold, and the curve is measured over a short burst with a fixed initialisation. As training progresses the loss surface sharpens, and a rate that was fine at step 100 diverges at step 10,000. Warmup exists for the same reason.')}

<h2><span class="sn">3.12.3</span> The instruments to log, in order of usefulness</h2>
<p>The loss curve is the instrument everyone watches, and it is also the last one to show trouble. By the time the loss itself moves, the underlying cause — a layer whose gradients have gone quiet, an activation distribution that has drifted, a learning rate the schedule has pushed too high — has usually been building for a while. The instruments below are ordered by how early they give warning, not by how easy they are to compute, and the gap between "gradient norms look wrong" and "loss looks wrong" can be thousands of steps, which is thousands of steps of wasted compute if you are only watching the loss.</p>
${H.steps([
      '<b>Loss, train and validation, on the same axes.</b> Log-scale the y-axis. Most of the information is in the first 5% of steps.',
      '<b>Gradient norm, globally and per layer.</b> A layer whose grad norm is 1e-9 is not learning; one at 1e3 is about to break everything. The ratio between the largest and smallest layer norms is the single best early-warning signal.',
      '<b>Learning rate.</b> On the same plot as the loss. Half of all "mysterious" curve features are the schedule.',
      '<b>Update-to-weight ratio</b> $\\|\\Delta w\\|/\\|w\\|$ per layer. Healthy is around $10^{-3}$. Much smaller means nothing is moving; much larger means the layer is being rewritten each step.',
      '<b>Activation statistics.</b> Mean and standard deviation per layer, plus the fraction of dead ReLUs. Drift here precedes loss problems by thousands of steps.',
      '<b>Throughput and the loss-vs-tokens curve.</b> Loss against <i>tokens seen</i>, not steps — it is the only comparison that survives a change of batch size.'
    ])}
${H.key('Plot loss against tokens (or examples) seen, never against steps, whenever the batch size might change. Two runs with different batch sizes are not comparable on a step axis, and half of all "this optimiser is better" claims dissolve when replotted.')}
${H.intuition(`<p>The update-to-weight ratio is worth dwelling on because it answers a question none of the other instruments can: is this specific layer actually learning anything, right now? A healthy ratio of roughly $10^{-3}$ says each step nudges the weight by about a tenth of a percent of its own size — small enough to be a refinement, large enough to add up over thousands of steps. A ratio many orders below that means the layer is frozen in every practical sense, even though its <code>requires_grad</code> flag says otherwise and its gradient norm might look non-zero on a log plot. A ratio many orders above it means the layer is being reinvented at every step, which is a symptom shared by a learning rate that is too high and by a layer whose activations have exploded, and the ratio is often the fastest way to tell which layer is the source when only the aggregate loss is misbehaving.</p>`)}

${H.pitfall('The loss spike that "recovers" often has not. A large spike can knock the optimiser into a worse basin, and Adam’s second-moment estimate takes thousands of steps to forget an enormous gradient. Large pretraining runs keep periodic checkpoints precisely so they can rewind past a spike and skip the offending data — this is standard operating procedure at frontier labs, not an exotic measure (§4.11).')}

${H.probe([
      ['Your loss is flat at exactly $\\ln(C)$ for a $C$-class problem. Diagnose it.', 'The model is outputting a uniform distribution — it has learned the prior and nothing else. Either the LR is far too low, the labels are shuffled relative to the inputs, or the head is disconnected. Check the label alignment first; it is the most common cause.'],
      ['Validation loss is <i>below</i> training loss. Is something wrong?', 'Usually not. Training loss is measured with dropout and augmentation on, validation without. Confirm by evaluating the training set in eval mode; if the gap persists, your split is not random.'],
      ['What is the first thing you do on a new model?', 'Overfit ten examples to near-zero loss with regularisation off. It takes two minutes and eliminates the entire class of data and plumbing bugs.'],
      ['Loss falls but your metric does not move. What is happening?', 'The loss and the metric are not aligned: often a threshold problem on an imbalanced task, or a proxy loss that does not track the objective. Look at the metric’s decomposition, not the loss.']
    ])}`,
    labs: {
      diagnose: function (host) {
        const el = ML.el;
        const CASES = [
          { name: 'flat', gen: (t) => 2.3 + 0.0002 * Math.sin(t / 9), cause: 'Learning rate far too low (or the head is disconnected)',
            why: 'The loss sits at ln(10) ≈ 2.30 — a uniform distribution over 10 classes. The model has learned the prior and nothing else. Check that gradients are non-zero before you touch the learning rate.' },
          { name: 'diverge', gen: (t) => t < 40 ? 2.3 - 0.01 * t : Math.min(9, 1.9 + 0.09 * (t - 40)), cause: 'Learning rate above the stability threshold',
            why: 'A brief improvement while the parameters are still small, then monotone divergence. Classic too-high LR. Warmup plus a 10× reduction usually fixes it.' },
          { name: 'nan', gen: (t) => t < 62 ? 2.3 * Math.exp(-t / 45) + .3 : NaN, cause: 'NaN — fp16 overflow or log(0)',
            why: 'Everything is healthy until it is not. Find the *first* NaN with per-tensor hooks; the usual culprits are an unclipped gradient, an fp16 overflow, or an unstabilised log/exp (§1.15).' },
          { name: 'spike', gen: (t) => { const b = 2.3 * Math.exp(-t / 40) + .25; return (t > 55 && t < 62) ? b + 2.6 * Math.exp(-Math.abs(t - 58)) : b; }, cause: 'A bad batch — one pathological example or a corrupt shard',
            why: 'A single sharp excursion that recovers. Log the batch index at the spike. Gradient clipping keeps it from being fatal; the data still needs cleaning, and Adam’s second moment takes a long time to forget the event.' },
          { name: 'saw', gen: (t) => 2.3 * Math.exp(-t / 55) + .28 + .16 * Math.abs(((t % 25) / 25) - .5), cause: 'A cyclic (or restarting) learning-rate schedule',
            why: 'Nothing is wrong. Plot the learning rate on the same axes and the mystery evaporates. Half of all confusing curve features are the schedule.' },
          { name: 'overfit', gen: (t) => 2.3 * Math.exp(-t / 22) + .04, gen2: (t) => 2.3 * Math.exp(-t / 30) + .35 + Math.max(0, (t - 45) * .010), cause: 'Overfitting',
            why: 'Training keeps falling; validation bottoms out around step 45 and then climbs. Early stopping at the minimum, more regularisation, more data or more augmentation.' },
          { name: 'step', gen: (t) => (t < 55 ? 2.3 * Math.exp(-t / 30) + .3 : 2.3 * Math.exp(-t / 30) + .95), cause: 'A data pipeline event — shard boundary or distribution change',
            why: 'A discontinuity, not a slope change. Optimisation problems bend the curve; data problems step it. Plot loss against data index rather than step and the boundary will be obvious.' },
          { name: 'plateau', gen: (t) => t < 30 ? 2.3 - .02 * t : (t < 75 ? 1.7 : 1.7 - .02 * (t - 75)), cause: 'A saddle or a dead-ReLU plateau; the schedule later escapes it',
            why: 'Long flat region then renewed progress. Check the fraction of dead ReLUs and the per-layer gradient norms during the plateau. Warmup, better initialisation (§3.4) or a nonzero-gradient activation prevent it.' },
          { name: 'noisy', gen: (t) => 2.3 * Math.exp(-t / 40) + .3 + .38 * Math.sin(t * 2.1) * Math.exp(-t / 120), cause: 'Batch size too small (or LR too high for this batch size)',
            why: 'The trend is fine; the variance is enormous. Gradient noise scales as $1/\\sqrt{B}$. Increase the batch, accumulate gradients, or lower the LR — the linear scaling rule ties the two together.' },
          { name: 'gap', gen: (t) => 2.3 * Math.exp(-t / 32) + .18, gen2: (t) => 2.3 * Math.exp(-t / 34) + .21, cause: 'Healthy',
            why: 'Both curves fall together, validation slightly above training, still improving at the end. Keep training — and note that a small persistent gap is normal, not a problem to be fixed.' }
        ];

        let idx = 0, revealed = false, score = 0, attempted = 0;
        const wrap = el('div');
        host.appendChild(wrap);

        function paint() {
          wrap.innerHTML = '';
          const c = CASES[idx];
          const cvHost = el('div');
          wrap.appendChild(cvHost);
          Viz.surface(cvHost, {
            height: 230,
            draw: function (ctx, w, h, T) {
              const ts = Num.linspace(0, 100, 101);
              const tr = ts.map(t => [t, c.gen(t)]).filter(p => isFinite(p[1]));
              const va = c.gen2 ? ts.map(t => [t, c.gen2(t)]).filter(p => isFinite(p[1])) : null;
              const all = tr.concat(va || []).map(p => p[1]);
              const P = Viz.plot(ctx, w, h, {
                xd: [0, 100], yd: [0, Math.min(9.5, Math.max.apply(null, all) * 1.15)],
                pad: { l: 48, r: 14, t: 16, b: 38 }
              }).frame({ xlabel: 'step', ylabel: 'loss' });
              P.clip(() => {
                if (va) P.line(va, { color: T.c2, width: 2.2 });
                P.line(tr, { color: T.c1, width: 2.4 });
                const brk = ts.find(t => !isFinite(c.gen(t)));
                if (brk != null) {
                  P.vline(brk, { color: T.c2, width: 2, dash: [3, 3], label: 'NaN' });
                }
              });
              if (va) {
                ctx.fillStyle = T.muted; ctx.font = '10.5px ui-sans-serif'; ctx.textAlign = 'right';
                ctx.fillText('— train', w - 20, 22); ctx.fillStyle = T.c2; ctx.fillText('— validation', w - 20, 36);
              }
            }
          });

          const opts = ML.shuffle(CASES.map(x => x.cause)).slice(0, 3);
          if (opts.indexOf(c.cause) < 0) opts[0] = c.cause;
          const shuffled = ML.shuffle(opts);
          const optHost = el('div', { class: 'qopts', style: 'margin-top:12px' });
          const explain = el('p', { class: 'qwhy', style: 'display:none' });
          shuffled.forEach(o => {
            const b = el('button', { class: 'qopt', type: 'button' }, [el('span', { class: 'k', text: '›' }), el('span', { text: o })]);
            b.addEventListener('click', () => {
              if (revealed) return;
              revealed = true; attempted++;
              const ok = o === c.cause;
              if (ok) score++;
              b.classList.add(ok ? 'right' : 'wrong');
              Array.prototype.forEach.call(optHost.children, x => {
                x.classList.add('locked');
                if (x.textContent.indexOf(c.cause) >= 0) x.classList.add('right');
              });
              explain.innerHTML = '<b>' + c.cause + '.</b> ' + c.why;
              explain.style.display = '';
              tally.textContent = score + ' / ' + attempted + ' correct';
            });
            optHost.appendChild(b);
          });
          wrap.appendChild(optHost);
          wrap.appendChild(explain);
          const row = el('div', { class: 'btnrow' });
          row.appendChild(el('button', {
            class: 'btn primary', type: 'button', text: 'Next curve →',
            onclick: () => { idx = (idx + 1) % CASES.length; revealed = false; paint(); }
          }));
          row.appendChild(tally);
          wrap.appendChild(row);
          ML.typeset(explain);
        }
        const tally = el('span', { class: 'small', style: 'font-family:var(--mono)', text: '0 / 0 correct' });
        paint();
        Viz.note(host, 'Ten curves, each generated by its actual mechanism rather than drawn by hand. Work through all of them; the distinctions that matter are <b>bend versus step</b> (optimisation versus data) and <b>recovers versus does not</b> (transient versus fatal).');
      },

      lrfind: function (host) {
        const st = Viz.controls(host, [
          { k: 'arch', label: 'network', type: 'select', value: 'mid', options: [{ v: 'small', t: '2–8–1 (small)' }, { v: 'mid', t: '2–16–12–1' }, { v: 'deep', t: '2–16–16–16–1 (deep)' }] },
          { k: 'act', label: 'activation', type: 'select', value: 'relu', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }] },
          { k: 'burst', label: 'steps per LR', min: 3, max: 25, step: 1, value: 8, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sugg', label: 'suggested LR', cls: 'good' },
          { k: 'min', label: 'LR at minimum loss', cls: 'key' },
          { k: 'div', label: 'diverges above' },
          { k: 'note', label: 'rule' }
        ]);
        const data = Num.dataset('circles', 200, .22, 5);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const sizes = st.arch === 'small' ? [2, 8, 1] : st.arch === 'mid' ? [2, 16, 12, 1] : [2, 16, 16, 16, 1];
            const pts = [];
            for (let e = -5; e <= 1; e += .1) {
              const lr = Math.pow(10, e);
              const net = Num.mlp(sizes, { seed: 3, act: st.act });
              let L = NaN;
              for (let s = 0; s < st.burst; s++) L = net.trainBatch(data.X, data.y, lr);
              pts.push([e, isFinite(L) ? Math.min(6, L) : 6]);
            }
            const finite = pts.filter(p => p[1] < 5.9);
            let best = finite.reduce((a, b) => b[1] < a[1] ? b : a, finite[0] || [0, 1]);
            // steepest descent: largest negative slope over a small window
            let steep = best, sv = 0;
            for (let i = 3; i < pts.length - 1; i++) {
              const d = pts[i - 3][1] - pts[i][1];
              if (d > sv && pts[i][1] < best[1] * 2.2) { sv = d; steep = pts[i]; }
            }
            const divIdx = pts.findIndex(p => p[1] >= 5.9);
            const P = Viz.plot(ctx, w, h, {
              xd: [-5, 1], yd: [0, 3], pad: { l: 48, r: 14, t: 16, b: 40 }
            }).frame({ xticks: [-5, -4, -3, -2, -1, 0, 1], xfmt: v => '1e' + v, xlabel: 'learning rate', ylabel: 'loss after ' + st.burst + ' steps' });
            P.clip(() => {
              P.line(pts, { color: T.c1, width: 2.6 });
              P.vline(steep[0], { color: T.c3, width: 2, dash: false, label: 'suggested' });
              P.vline(best[0], { color: T.c4, dash: [4, 3], label: 'minimum' });
              if (divIdx > 0) {
                ctx.fillStyle = T.c2; ctx.globalAlpha = .1;
                ctx.fillRect(P.x(pts[divIdx][0]), P.pad.t, P.x(1) - P.x(pts[divIdx][0]), P.ph);
                ctx.globalAlpha = 1;
              }
            });
            out({
              sugg: Math.pow(10, steep[0]).toExponential(1),
              min: Math.pow(10, best[0]).toExponential(1),
              div: divIdx > 0 ? Math.pow(10, pts[divIdx][0]).toExponential(1) : 'not in range',
              note: 'take ~10× below the minimum'
            });
          }
        });
        Viz.note(host, 'Switch the activation to <b>tanh</b> and watch the whole curve shift left — saturating activations tolerate a smaller learning rate than ReLU. Then switch to the <b>deep</b> network: the usable window narrows, which is exactly the problem residual connections and normalisation were introduced to solve (§3.6).');
      }
    },
    quiz: [
      {
        q: 'Your 10-class classifier’s loss is pinned at 2.303. The most likely cause is…',
        options: ['overfitting', 'the model is predicting a uniform distribution — no learning is happening', 'the learning rate is too high', 'the batch size is too large'],
        answer: 1,
        why: '$\\ln 10 \\approx 2.303$ is exactly the cross-entropy of a uniform guess over ten classes, so a loss pinned there is not noise — it is the network outputting the prior and never moving off it, which §3.12.1’s table attributes to a learning rate too low to update anything, or a gradient path that is broken outright (a disconnected head, mismatched labels). "The learning rate is too high" is the tempting wrong answer because a bad learning rate is everyone’s first reflex for anything that goes wrong in training, and it is the correct diagnosis for the neighbouring row in that same table — but that row describes a curve that rises then plateaus, or diverges outright, the opposite direction from being stuck exactly at the prior. Overfitting and an oversized batch would both still show some early progress before their symptoms appear; neither produces a loss frozen at a value that happens to equal the exact entropy of guessing. The general skill this section trains is recognising that specific numbers on a loss curve are not decoration — $\\ln(C)$ for $C$ classes is a signature precise enough to name the failure mode on sight.'
      },
      {
        q: 'A sudden vertical step in the loss (not a bend) usually indicates…',
        options: ['a learning-rate change', 'a data pipeline event such as a shard boundary', 'overfitting', 'vanishing gradients'],
        answer: 1,
        why: 'A vertical, ramp-free jump in the loss value itself cannot come from anything optimisation-related, because a learning rate — however it is scheduled — only changes how fast the parameters move from here, never where they already are; nothing about adjusting a hyperparameter can teleport the current loss to a new level with no transition. A discontinuity of that kind means whatever is being fed to the network changed abruptly: a new data shard, a corrupted batch, a mid-run change to the input pipeline. "A learning-rate change" is the tempting wrong answer precisely because LR schedules genuinely do produce two of the other named shapes in §3.12.1’s table, sawtooth and staircase — but both of those are changes in the curve’s slope, a bend, never an instantaneous jump in its value, and conflating "the schedule causes curve features" with "the schedule causes this particular feature" is the trap. Overfitting and vanishing gradients are both gradual, cumulative effects that build up over many steps and would never produce a single sharp discontinuity. The general habit §3.12.1 is training is asking bend or step before reaching for a specific cause, because that one question sorts every named shape into something about how you are optimising versus something about what you are feeding it.'
      },
      {
        q: 'The LR range test suggests taking a rate…',
        options: ['at the loss minimum', 'about an order of magnitude below the minimum, in the steepest-descent region', 'at the divergence point', 'as high as possible'],
        answer: 1,
        why: 'The suggested rate sits about a decade below the curve’s literal minimum, in the steepest-descent region, because that minimum is already living dangerously close to the instability cliff on the right — and §3.12.2’s note explains why picking it outright is worse than it looks: the sweep trains for only a short burst, but a real run continues for thousands of steps, and the loss landscape sharpens as training progresses, so a rate that looked fine at step 100 can be the rate that diverges at step 10,000. "At the loss minimum" is the tempting wrong answer because it is the naive reading of what a sweep like this is for — you ran the experiment to find the best-performing rate, so why not take the best-performing point — but the lab’s own note calls this out directly: the margin below the minimum is not conservative caution tacked on afterwards, it is the entire reason the heuristic works. "At the divergence point" and "as high as possible" both ignore that the curve’s right-hand rise marks outright training failure, not merely reduced performance. The general lesson, which also explains why deeper networks show a narrower usable window in the lab, is that a rate measured safe over a hundred steps is not the same claim as a rate safe over an entire training run.'
      },
      {
        q: 'Comparing two runs with different batch sizes, you should plot loss against…',
        options: ['optimiser steps', 'tokens or examples seen', 'wall-clock time', 'epochs'],
        answer: 1,
        why: 'Loss against tokens (or examples) seen is the only axis that survives a change in batch size, because it measures the one thing both runs actually have in common — how much data has passed through the model — rather than an artefact of how that data happened to be grouped. "Optimiser steps" is the tempting wrong answer, and it is the default x-axis in almost every training-logging tool, which is exactly the trap: a step at batch size 128 consumes four times the data of a step at batch size 32, so at any given step count the larger-batch run has silently seen far more examples, making it look more sample-efficient purely because of how its data was bucketed, not because its optimisation is actually better. Wall-clock time carries its own confound — hardware, parallelism and I/O all drag on it — and epochs hide the same batch-size effect one level up. §3.12.3’s key box states this as a hard rule for exactly this reason: many "this optimiser is better" claims dissolve when replotted against tokens, which is the general principle every batch-size comparison in this course has to respect.'
      }
    ],
    cards: [
      { q: 'First thing to do with a new model', a: 'Overfit ten examples to near-zero loss with regularisation off. It rules out every data and plumbing bug in two minutes.' },
      { q: 'Loss stuck at ln(C)', a: 'Uniform predictions — the model learned the prior. Check label alignment and gradient flow.' },
      { q: 'Bend vs step', a: 'A bend is an optimisation event; a vertical step is a data event.' },
      { q: 'Update-to-weight ratio', a: '$\\|\\Delta w\\|/\\|w\\| \\approx 10^{-3}$ is healthy. Orders of magnitude off in either direction is the earliest reliable warning.' },
      { q: 'LR range test', a: 'Sweep LR exponentially, plot loss; pick the steepest-descent point, roughly 10× below the minimum.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.13 */
  ML.section({
    id: 'compression', track: 'deep', num: '3.13', level: 2,
    title: 'Quantisation, pruning, and distillation',
    lede: 'Three ways to make a trained model smaller and faster. They compose, they have very different risk profiles, and the arithmetic that decides which one you need is the same memory-bandwidth arithmetic as §4.14.',
    prereq: ['numerics'],
    related: ['serving', 'lora', 'numerics'],
    html: `
<p>You have a model that works. Someone else now needs it to run on hardware smaller than the one it was trained on, or to answer in milliseconds rather than seconds, or to cost a fraction of what it currently costs per query. The model's accuracy is fine; its <i>footprint</i> is the problem. Three genuinely different levers exist for shrinking that footprint, and they correspond to three different questions you could ask about a trained network: can each number be described more cheaply? Are all of these numbers actually necessary? And could a much smaller network be taught to reach almost the same answers? Those questions are quantisation, pruning and distillation, in that order, and this section builds each from its motivating question rather than its acronym.</p>
${H.analogy(`<p>Think of the three as three different answers to "this video file is too large". Quantisation is lowering the bitrate: every frame is still there, but each pixel is described with fewer bits, and you choose that trade knowing exactly how much space each bit costs you. Pruning is deleting frames: if ninety per cent of a static shot is identical to the frame before it, most of those frames are redundant, and removing them shrinks the file — but only if your player actually skips the deleted frames rather than storing blank ones and playing them anyway, which is precisely the distinction between structured and unstructured pruning below. Distillation is asking someone who watched the whole film to write you a much shorter summary that still lets you follow the plot — a different, smaller artefact, produced by learning from the original rather than editing it.</p>`)}

${H.tldr([
      '<b>Quantisation</b> is nearly always the first move: INT8 is roughly free, 4-bit costs 1–3 points of quality and quarters the memory, and LLM inference is memory-bandwidth-bound so fewer bytes means proportionally more tokens per second.',
      '<b>Pruning</b>: unstructured sparsity compresses well but rarely speeds anything up without special kernels; structured pruning (whole heads, channels, layers) is slower to recover but actually runs faster.',
      '<b>Distillation</b> transfers the teacher’s <i>soft</i> predictions. The information is in the wrong answers’ relative probabilities — which is why the temperature matters and why the gradient is scaled by $T^2$.'
    ])}

<h2><span class="sn">3.13.1</span> Quantisation</h2>
<p>A trained weight is stored as a 32- or 16-bit float, which is a far more precise description than the weight's actual role in the network needs — nobody is relying on the fifteenth decimal digit of a single attention weight. Quantisation exploits that slack directly: pick a small integer grid, typically 256 points for 8-bit or 16 points for 4-bit, and describe every weight by the nearest grid point plus one shared conversion recipe for the whole group of weights. Map a float tensor to that grid with $q = \\mathrm{round}(x/s) + z$, where $s$ is a <b>scale</b> — how much real value one integer step represents — and $z$ is a <b>zero-point</b>, an offset used when the values are not centred on zero. Recovering an approximate float back out is <b>dequantisation</b>, $\\hat x = s(q-z)$, and the whole scheme lives or dies on how good an approximation that round-trip is. Two decisions determine everything about that quality.</p>
${H.table(['Decision', 'Options', 'Trade'], [
      ['Symmetric or affine', '$z=0$ (symmetric) vs a learned zero-point', 'symmetric is cheaper; affine handles skewed activation ranges (post-ReLU)'],
      ['Granularity', 'per-tensor / per-channel / per-group (e.g. 64 or 128 weights)', 'finer granularity costs a few bits of overhead and recovers most of the quality — <b>group-wise is why 4-bit works at all</b>'],
      ['<b>PTQ or QAT</b>', 'quantise after training vs simulate quantisation during it', 'PTQ needs ~128 calibration samples and an afternoon; QAT needs a training run and buys 1–2 points back at ≤4 bits'],
      ['What to quantise', 'weights only / weights + activations / + KV cache', 'weight-only is easiest and helps most for LLMs; activation quantisation is where outliers bite']
    ])}
<p>The granularity row deserves a second look, because it is the row that decides whether 4-bit is usable at all. A single scale for an entire weight matrix has to serve every value in that matrix, including whichever one happens to be the largest in magnitude — and one unusually large weight forces the whole grid to stretch to accommodate it, wasting most of the grid's resolution on a range the typical weight never visits. Shrink the group a single scale has to cover, down to a block of 64 or 128 weights, and each group can set its own scale tuned to its own range. The overhead is a handful of extra numbers stored per group; the payoff, visible directly in the lab below, is the difference between 4-bit quantisation being usable and 4-bit quantisation being useless.</p>
<p><b>What you are looking at.</b> A real multilayer perceptron, trained to convergence on a two-class problem, then quantised at the bit width and granularity you choose. The left panel is a histogram of the original fp32 weights, in outline, with the quantised values overlaid as coloured spikes — each spike is one surviving grid level, and its height is how many weights landed on it. The right panel is the network's decision boundary after quantisation, on the same held-out points used to score accuracy.</p>
<p><b>What to do with it.</b> Start at 8 bits and note the two panels barely change from the unquantised baseline — the grid is fine enough to be invisible. Walk the bit width down toward 1 and watch the histogram collapse onto a handful of spikes, each one absorbing a wide range of original values, while the decision boundary visibly coarsens and the accuracy readout drops. Then, holding bits fixed at 4, switch granularity from per-channel back to per-tensor and watch accuracy fall again with nothing else changed.</p>
<p><b>The thing genuinely worth noticing.</b> The granularity switch is the whole argument for group-wise quantisation made visible: the same number of bits, describing the same weights, gives a noticeably worse result the moment one scale has to serve a wider range of values. That is not a subtle effect requiring careful measurement — it shows up as a clearly worse decision boundary on the same two-line change. Every production 4-bit format ships a group size for exactly this reason.</p>
${H.lab('quant', 'Quantise a trained network and watch the accuracy fall', 'A real MLP, really trained, whose weights are really quantised at the bit width you choose. Everything below the plot is measured, including the accuracy — this is not a lookup table.')}
<h3>How much error a grid actually costs, and who pays it</h3>

<p>"Wastes the grid" has been doing heavy lifting for two paragraphs, and it is worth replacing with a number, because the number explains both why 8 bits is usually free and why the outlier problem is as severe as it is. Round-to-nearest onto a grid of spacing $s$ leaves a residual error $x - \\hat x$ that is at most $s/2$ in magnitude and, for weights that are not themselves sitting suspiciously close to grid points, spread roughly uniformly across $[-s/2,\\, s/2]$. A uniform distribution on an interval of width $s$ has variance $s^2/12$ (§1.2), so the quantisation noise power is $s^2/12$ and its RMS size is $s/\\sqrt{12} \\approx 0.289\\,s$. That is the entire error model, and everything else follows from what sets $s$.</p>

${H.deriv('the effective bit width, after outliers have taken their cut', [
      ['$s = \\dfrac{2\\max_j |w_j|}{2^b}$', 'A symmetric $b$-bit grid must span from $-\\max|w|$ to $+\\max|w|$, because a value outside the grid cannot be represented at all. Divide that span by the $2^b$ available levels. Note immediately what sets the step size: the single largest weight in the group, not the typical one.'],
      ['$\\sigma_{\\text{noise}}^2 = \\dfrac{s^2}{12}$', 'The uniform-error variance from above. This is the noise the round trip injects into every weight in the group, large or small, since all of them are rounded onto the same grid.'],
      ['$\\mathrm{SNR} = \\dfrac{\\sigma_w}{\\sigma_{\\text{noise}}} = \\dfrac{\\sigma_w\\sqrt{12}}{s} = \\dfrac{\\sigma_w\\sqrt{12}\\,2^b}{2\\max_j|w_j|} = \\sqrt3\\;\\cdot\\;\\dfrac{\\sigma_w}{\\max_j|w_j|}\\;\\cdot\\;2^{b}$', 'Signal-to-noise ratio: the typical weight magnitude $\\sigma_w$ over the RMS error. Substitute $s$ from line 1 and collect. The result factors into three independent pieces — a constant, a property of the weight distribution, and the bit width.'],
      ['$\\log_2 \\mathrm{SNR} = b - \\log_2\\!\\left(\\dfrac{\\max_j|w_j|}{\\sigma_w}\\right) + \\log_2\\sqrt3$', 'Take logs base 2, so that each unit is one bit. The middle term is the whole story: the ratio of the largest weight to the typical one is <b>subtracted directly from the bit width</b>, in bits.'],
      ['$b_{\\text{eff}} \\;\\approx\\; b - \\log_2\\!\\left(\\dfrac{\\max_j|w_j|}{\\sigma_w}\\right)$', 'Define the effective bit width as the precision the typical weight actually receives, dropping the constant. An outlier does not merely waste part of the grid — it consumes a precisely computable number of bits, and every weight in its group pays.']
    ], 'The subtracted term is sometimes called the crest factor of the group, and reading it as "bits the outlier took" makes the entire quantisation literature legible at a glance. For a well-behaved Gaussian block of 4,096 weights, the largest of them sits around $4$ to $5$ standard deviations out — the leading-order estimate is $\\sqrt{2\\ln n} \\approx 4.1$ for $n = 4{,}096$ — so $\\log_2 4.5 \\approx 2.2$ bits go to the tail and INT8 delivers roughly 5.8 effective bits to the typical weight. That is comfortably enough, which is exactly why weight-only INT8 is close to lossless and why nobody writes papers about it.')}

${H.worked('what a 100× activation outlier actually costs, in bits', `<p>Take the number quoted in every LLM quantisation paper: a handful of activation channels carrying values about $100\\times$ the typical magnitude. Put it through the formula.</p>
<p>$\\log_2 100 = 6.64$ bits. Against INT8's $b = 8$, the effective precision delivered to an ordinary channel is $8 - 6.64 = \\mathbf{1.36}$ bits.</p>
<p>Not "degraded", not "somewhat coarser" — <mark>fewer than two bits</mark>, which is to say the typical activation is being described by something closer to a sign and a rough magnitude than by a number. Compare the well-behaved case in the derivation above, where the same 8 bits delivered 5.8. The outlier did not take a slice of the grid; it took three quarters of it.</p>
<p>Now read the four named fixes as four different ways of attacking that one subtracted term. <b>LLM.int8()</b> removes the outlier channels from the group entirely and computes them in fp16, so the remaining group's $\\max/\\sigma$ collapses back to the Gaussian case and the full 8 bits become available again. <b>SmoothQuant</b> divides the activations by a per-channel constant and multiplies the corresponding weight rows by the same constant — mathematically a no-op on the layer's output, but it moves magnitude from a tensor where outliers are catastrophic into one where they are mild, lowering $\\max/\\sigma$ on the side that matters. <b>Group-wise scaling</b>, from the table above, shrinks the group until the outlier is alone with a handful of neighbours, so only those few weights pay the 6.64 bits instead of all 4,096. <b>AWQ</b> notes that not all channels deserve equal protection and spends the available range on the ones whose error propagates furthest.</p>
<p>Four techniques, one term. Whenever a new quantisation method appears, the useful first question is which factor of $\\max_j|w_j| / \\sigma_w$ it is trying to reduce, and that question almost always has a clean answer.</p>`)}

${H.flag('The hard part of LLM quantisation is <b>activation outliers</b>: a handful of channels carry values 100× the rest, and per-tensor scaling then wastes the whole grid on them — costing $\\log_2 100 \\approx 6.6$ of INT8\'s 8 bits, as derived above. LLM.int8() handles those channels in fp16; SmoothQuant migrates the difficulty from activations into weights; AWQ protects the 1% of weight channels that matter most; GPTQ solves a layerwise reconstruction problem with second-order information. All four exist because of the same outlier phenomenon.')}
${H.worked('why 4-bit quantisation buys throughput, not just memory', `
<p>Decoding one token from a 70B model reads every weight once. At bf16 that is 140 GB per token; on a card with 2 TB/s of memory bandwidth the floor is $140/2000 = 70$ ms per token, or about 14 tokens/second — <b>before any computation at all</b>.</p>
<p>Quantise to 4-bit and the read becomes 35 GB, so the floor drops to 17.5 ms, about 57 tokens/second. <b>The speedup is 4×, and it comes from bandwidth, not arithmetic.</b> This is also why batching is nearly free for decode: the weights are read once for the whole batch (§4.14).</p>`)}

<h2><span class="sn">3.13.2</span> Pruning</h2>
<p>Quantisation keeps every weight and describes each one more cheaply. <b>Pruning</b> asks a different question: does every weight need to be there at all? The idea is old — LeCun, Denker and Solla's 1990 paper <i>Optimal Brain Damage</i> showed that a substantial fraction of a trained network's weights could be deleted, with the smallest-magnitude ones removed first, and the network fine-tuned back to nearly its original accuracy. The magnitude heuristic behind that result is still the default today: a weight close to zero is contributing almost nothing to the output, so zeroing it out and removing it from the parameter count should cost little.</p>
<p>What that 1990 result does not tell you, and what trips up a great many otherwise-careful engineers, is whether removing those weights makes the network faster to run. Those are two separate claims — smaller on disk, and quicker at inference — and pruning delivers the first far more reliably than the second.</p>
${H.table(['Kind', 'What is removed', 'Compression', 'Actual speedup', 'Recovery'], [
      ['Unstructured magnitude', 'individual weights below a threshold', '10–20× on disk', '<b>none</b> on dense hardware', 'easy, retrain a little'],
      ['2:4 semi-structured', '2 of every 4 weights', '2×', '~1.5–1.7× on Ampere+ tensor cores', 'moderate'],
      ['Structured (channels, heads)', 'whole units', '2–4×', '<b>real, proportional</b>', 'harder; needs fine-tuning'],
      ['Layer dropping', 'whole blocks', 'proportional', 'proportional', 'surprisingly viable for deep LLMs — the middle layers are the redundant ones'],
      ['Movement pruning', 'weights moving toward zero during fine-tuning', 'high', 'as above', 'best for transfer settings']
    ])}
${H.pitfall('"We pruned 90% of the weights" almost never means "it runs 10× faster". A dense GEMM does not care that most of its inputs are zero. Unless you are targeting a sparse kernel, 2:4 sparsity on recent NVIDIA hardware, or actually deleting structural units, unstructured pruning buys disk space and nothing else. Say this in an interview and you will separate yourself from a large number of candidates.')}
${H.intuition(`<p>The <b>lottery ticket hypothesis</b> is the interesting theoretical claim here: a randomly initialised dense network contains a sparse subnetwork that, trained <i>from the same initialisation</i>, matches the full network. It is a genuine and reproducible finding at small scale. What it has not delivered is a way to find the ticket without training the dense network first — so it remains a statement about what exists rather than a practical method.</p>`)}

<h2><span class="sn">3.13.3</span> Distillation</h2>
<p>Quantisation and pruning both start from a trained network and shrink it in place. <b>Distillation</b> does something stranger: it trains a second, smaller network from scratch, and the thing that makes the smaller network good is not the original training data at all — it is the first network's <i>predictions</i>. Call the large, already-trained network the <b>teacher</b> and the small one being trained the <b>student</b>. The naive way to teach the student would be to hand it the same hard labels the teacher was trained on — this image is a "7", full stop — and that works, but it throws away something the teacher has that the raw labels never had: an opinion about which wrong answers are <i>less</i> wrong than others.</p>
${H.analogy(`<p>Picture a strict exam graded pass/fail against a marked answer key, next to a thoughtful tutor who tells you not just the right answer but how close each of your alternatives was — "your '7' looks a bit like a '1', which makes sense, they share a vertical stroke; it looks nothing like an '8'". The pass/fail exam is a hard label: right or wrong, nothing else. The tutor's running commentary is what a teacher network's full output distribution carries, and it is strictly more informative per example, which is the entire case for distillation over training the small network from labels alone.</p>`)}
<p>To use that commentary you first have to make it legible, because a well-trained classifier's raw output is usually close to one-hot — 99.9% on the correct class, with all the interesting "this looks a little like a 1" information compressed into probabilities near zero that are hard to learn from. <b>Temperature</b> softens the softmax to bring that information back out: divide the logits by $T > 1$ before the softmax, and the output distribution spreads out, so that the small probabilities the teacher assigned to plausible wrong answers become large enough to actually train on. The full distillation loss blends this softened teacher-matching term with an ordinary hard-label term:</p>
$$\\mathcal{L} = \\alpha\\, T^2\\,\\mathrm{KL}\\!\\left(\\sigma(z_t/T)\\,\\|\\,\\sigma(z_s/T)\\right) + (1-\\alpha)\\,\\mathrm{CE}(y, \\sigma(z_s))$$
<p>Read the two terms as the tutor and the answer key working together: the first term, weighted by $\\alpha$, pulls the student's softened output towards the teacher's softened output — the KL divergence (§1.10) between the two distributions, at temperature $T$. The second term is the ordinary cross-entropy against the true hard label, unchanged. The $T^2$ sitting in front of the first term looks like it could be an arbitrary tuning knob. It is not, and the derivation below shows exactly why it has to be there.</p>
${H.deriv('why the $T^2$ is there, and why it is not a fudge', [
      ['$\\dfrac{\\partial}{\\partial z_s}\\mathrm{KL}\\big(\\sigma(z_t/T)\\,\\|\\,\\sigma(z_s/T)\\big) = \\dfrac{1}{T}\\left(\\sigma(z_s/T)-\\sigma(z_t/T)\\right)$', 'The softmax-plus-cross-entropy gradient from §0.7, with the temperature carried through the chain rule. Each logit gradient picks up a factor $1/T$.'],
      ['gradient magnitude $\\sim 1/T^2$', 'The <i>differences</i> between the softened probabilities are themselves $O(1/T)$ for large $T$, so the product shrinks quadratically.'],
      ['multiply by $T^2$', 'Restores the gradient to the same scale as the hard-label term, so that $\\alpha$ actually controls the balance and you can change $T$ without re-tuning the learning rate. <b>It is a normalisation, not a hyperparameter.</b>']
    ])}
${H.key('The value of distillation is the <i>dark knowledge</i> in the wrong answers: a teacher that says "7" with probability 0.9 and "1" with 0.08 has told the student that this 7 looks a bit like a 1. A hard label destroys that. Raising $T$ amplifies exactly those small probabilities, which is why $T$ is typically 2–10.')}
<p>The formula above matches output probabilities directly, and it was the original 2015 formulation, but "what is matched between teacher and student" is itself a choice with several workable answers, and the one that dominates in large language models today is not the classifier version at all.</p>
${H.table(['Variant', 'What is matched', 'Note'], [
      ['Response / logit KD', 'output distribution', 'the original; still a strong baseline'],
      ['Feature / hint KD', 'intermediate activations', 'needs a projection when widths differ'],
      ['Attention transfer', 'attention maps', 'effective for transformers'],
      ['Sequence-level KD', 'the teacher’s generated sequences as training data', '<b>the dominant LLM method</b>: generate from the teacher, fine-tune on the output'],
      ['Self-distillation', 'the model’s own earlier checkpoint', 'improves accuracy with no teacher at all, which is odd and reproducible']
    ])}
${H.flag('Distilling from a commercial API to train a competitor is prohibited by most providers’ terms of service and is a live legal question, not merely a technical one. It is also, separately, how a great many open models were actually trained. Mention the constraint if the topic comes up in an interview — awareness of it is part of the job.')}

${H.probe([
      ['You need 4× less memory on a 13B model tomorrow. What do you do?', 'Weight-only 4-bit group quantisation (AWQ or GPTQ, group size 128) with a small calibration set. It is hours of work, costs a point or two, and buys ~4× on both memory and decode throughput.'],
      ['Why does INT8 quantisation of LLMs fail without special handling?', 'Activation outliers: a few channels have values two orders of magnitude larger, and per-tensor scaling then wastes the grid. LLM.int8(), SmoothQuant and AWQ are three different answers.'],
      ['Why is unstructured pruning disappointing in production?', 'Dense kernels do not exploit scattered zeros. You get disk compression, not latency. Structured pruning or 2:4 sparsity is what actually runs faster.'],
      ['Why multiply the distillation loss by $T^2$?', 'The gradient through a temperature-$T$ softmax scales as $1/T^2$; the factor restores it so that $\\alpha$ balances the two terms independently of $T$.']
    ])}`,
    labs: {
      quant: function (host) {
        const st = Viz.controls(host, [
          { k: 'bits', label: 'weight bits', min: 1, max: 8, step: 1, value: 4, fmt: v => v + '-bit' },
          { k: 'group', label: 'quantisation granularity', type: 'select', value: 'row', options: [{ v: 'tensor', t: 'per tensor' }, { v: 'row', t: 'per output channel' }] },
          { k: 'sym', label: 'symmetric', type: 'toggle', value: true },
          { k: 'prune', label: 'also prune smallest weights', min: 0, max: .9, step: .05, value: 0, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'fp', label: 'fp32 accuracy', cls: 'key' },
          { k: 'q', label: 'quantised accuracy', cls: 'good' },
          { k: 'drop', label: 'drop', cls: 'bad' },
          { k: 'size', label: 'model size' },
          { k: 'rmse', label: 'weight RMSE' }
        ]);
        const train = Num.dataset('moons', 260, .22, 4);
        const test = Num.dataset('moons', 400, .22, 71);
        const net = Num.mlp([2, 24, 16, 1], { seed: 5, act: 'relu' });
        for (let i = 0; i < 900; i++) net.trainBatch(train.X, train.y, .05);
        const W0 = net.W.map(l => l.map(r => r.slice()));

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            let sqErr = 0, nW = 0;
            const Wq = W0.map(layer => {
              if (st.group === 'tensor') {
                const flat = layer.flat();
                const kept = st.prune > 0 ? pruneThr(flat, st.prune) : null;
                const q = Num.quantize(flat.map(v => (kept && Math.abs(v) < kept) ? 0 : v), st.bits, { symmetric: st.sym });
                let k = 0;
                return layer.map(row => row.map(() => q.deq[k++]));
              }
              return layer.map(row => {
                const kept = st.prune > 0 ? pruneThr(row, st.prune) : null;
                const q = Num.quantize(row.map(v => (kept && Math.abs(v) < kept) ? 0 : v), st.bits, { symmetric: st.sym });
                return q.deq;
              });
            });
            W0.forEach((l, li) => l.forEach((r, ri) => r.forEach((v, ci) => { sqErr += (v - Wq[li][ri][ci]) ** 2; nW++; })));

            const accOf = (Wset) => {
              W0.forEach((l, li) => l.forEach((r, ri) => r.forEach((_, ci) => { net.W[li][ri][ci] = Wset[li][ri][ci]; })));
              return test.X.filter((p, i) => (net.predict(p) > .5 ? 1 : 0) === test.y[i]).length / test.X.length;
            };
            const accFp = accOf(W0);
            const accQ = accOf(Wq);

            const w1 = w * .5;
            const flatW = W0.flat(2), flatQ = Wq.flat(2);
            const hs = Num.hist(flatW, 40);
            const P = Viz.plot(ctx, w1, h, {
              xd: [hs.lo, hs.hi], yd: [0, Math.max.apply(null, hs.bins) * 1.15],
              pad: { l: 42, r: 8, t: 16, b: 38 }
            }).frame({ xlabel: 'weight value', ylabel: 'count' });
            P.clip(() => {
              P.bars(hs.bins, { gap: .08, color: T.line });
              const hq = Num.hist(flatQ, 40);
              // quantised weights land on a grid: draw them as spikes
              const levels = {};
              flatQ.forEach(v => { levels[v.toFixed(4)] = (levels[v.toFixed(4)] || 0) + 1; });
              Object.keys(levels).forEach(k => {
                const v = +k;
                if (v < hs.lo || v > hs.hi) return;
                ctx.strokeStyle = T.c2; ctx.lineWidth = 1.6; ctx.globalAlpha = .8;
                ctx.beginPath(); ctx.moveTo(P.x(v), P.y(0)); ctx.lineTo(P.x(v), P.y(Math.min(levels[k], hs.bins.reduce((a, b) => Math.max(a, b), 0)))); ctx.stroke();
                ctx.globalAlpha = 1;
              });
            });

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, { xd: [-2.6, 2.9], yd: [-1.9, 2.3], pad: { l: 8, r: 8, t: 16, b: 30 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'decision boundary after quantisation' });
            accOf(Wq);
            P2.clip(() => {
              Labs.boundary(P2, (x, y) => net.predict([x, y]), { step: 4, lo: 0, hi: 1 });
              Labs.points(P2, test.X.slice(0, 140), test.y.slice(0, 140), { r: 2.6 });
            });
            ctx.restore();
            accOf(W0);

            const nParams = flatW.length;
            out({
              fp: (accFp * 100).toFixed(1) + '%',
              q: (accQ * 100).toFixed(1) + '%',
              drop: ((accFp - accQ) * 100).toFixed(1) + ' pts',
              size: (nParams * st.bits / 8).toFixed(0) + ' B vs ' + (nParams * 4) + ' B (' + (32 / st.bits).toFixed(1) + '×)',
              rmse: Math.sqrt(sqErr / nW).toFixed(4)
            });
          }
        });
        function pruneThr(arr, frac) {
          const s = arr.map(Math.abs).sort((a, b) => a - b);
          return s[Math.floor(frac * s.length)];
        }
        Viz.legend(host, [{ c: 'var(--line)', t: 'original fp32 weights' }, { c: 'var(--c2)', t: 'quantisation grid' }]);
        Viz.note(host, 'At 8 bits the grid is dense enough that the two distributions are indistinguishable and accuracy is unchanged. At 3 bits the weights collapse onto eight levels and the decision boundary visibly coarsens. Now switch granularity to <b>per tensor</b> at 4 bits: accuracy falls further, because one scale must serve channels with very different magnitudes. <b>That is the entire argument for group-wise quantisation</b>, and it is why every 4-bit LLM format has a group size.');
      }
    },
    quiz: [
      {
        q: '4-bit weight quantisation speeds up LLM decoding mainly because…',
        options: ['integer arithmetic is faster', 'decoding is memory-bandwidth-bound and there are 4× fewer bytes to read', 'the model has fewer parameters', 'it enables larger batches'],
        answer: 1,
        why: 'Decoding one token reads every weight exactly once, and at 2 TB/s of memory bandwidth that read time, not the arithmetic performed once the bytes arrive, is the floor: 140 GB of bf16 weights force a $140/2000=70$ ms wait per token regardless of how fast the multiply-adds are, while 35 GB at 4-bit cuts that same wait to 17.5 ms — exactly the 4× the byte count shrank by. "Integer arithmetic is faster" is the tempting wrong answer because it is true in general — narrower types often do execute faster — but it names the wrong bottleneck for this workload: decode is memory-bandwidth-bound, not compute-bound, so even a faster multiply would not shorten the wait for the weights to arrive from memory unless the weights themselves got smaller. "Fewer parameters" is simply false, since quantisation removes no weight, and larger batches are a separate benefit noted in §3.13.1’s worked example (the weights are read once per batch, not once per example) rather than the reason 4-bit itself is fast. The general principle, which also explains why doubling FLOPs on a decode-only workload barely moves latency, is to ask whether an operation is bandwidth-bound or compute-bound before assuming a precision win comes from faster arithmetic.'
      },
      {
        q: 'Unstructured magnitude pruning to 90% sparsity typically gives what speedup on standard GPU kernels?',
        options: ['10×', '2×', 'essentially none', '4×'],
        answer: 2,
        why: 'A dense GEMM kernel multiplies every entry in the matrix regardless of its value, so scattering zeros through an otherwise-dense array does not change the amount of work the hardware performs — you get disk compression from the sparsity, and essentially nothing at inference, unless you switch to a sparse-aware kernel, 2:4 semi-structured sparsity on supported tensor cores, or remove whole structural units so the matrix genuinely shrinks. "10×" is the tempting wrong answer because it is exactly the compression ratio 90% sparsity implies, and it feels natural that a number which shrinks the model on disk should shrink its runtime by the same factor — but §3.13.2’s pitfall exists precisely to break that assumption: "we pruned 90% of the weights" and "it runs 10× faster" are two entirely different claims, and pruning delivers the first far more reliably than the second. "2×" and "4×" undershoot in the other direction by assuming some partial speedup leaks through, when standard dense kernels have no mechanism to exploit unstructured zeros at all. The general test, which the table in §3.13.2 lays out across five pruning variants, is whether the compression is structured enough for hardware to actually skip the removed work — magnitude alone earns you a smaller file, not a faster one.'
      },
      {
        q: 'The $T^2$ factor in the distillation loss…',
        options: ['is a tunable hyperparameter', 'compensates for the $1/T^2$ shrinkage of the gradient so $\\alpha$ stays meaningful', 'sharpens the teacher distribution', 'prevents overfitting'],
        answer: 1,
        why: 'Softening a softmax by temperature $T$ shrinks the gradient flowing back through it by roughly $1/T^2$ — one factor of $1/T$ from the chain rule through the division inside the softmax, another because the softened probability differences themselves shrink as $O(1/T)$ — so multiplying the KL term by $T^2$ exactly undoes that shrinkage and restores the two loss terms to a comparable scale, which is what lets $\\alpha$ actually control their balance rather than having it silently retuned every time $T$ changes. "A tunable hyperparameter" is the tempting wrong answer because $T^2$ sits in the formula in exactly the position a free coefficient would occupy, right next to $\\alpha$ and $T$, both of which genuinely are choices you sweep — but §3.13.3’s derivation shows the exponent is forced by the same $T$ already inside the softmax, not chosen independently. "Sharpens the teacher distribution" has the direction backwards, since raising $T$ softens the distribution, and "prevents overfitting" describes no mechanism the term actually has. The general habit this question tests, developed across every `H.deriv` box in this course, is checking whether a term in a formula is a design choice or a consequence forced by algebra someone already did for you.'
      },
      {
        q: 'The main obstacle to INT8 quantisation of LLM activations is…',
        options: ['insufficient training data', 'a few channels with outlier values 100× the rest', 'lack of hardware support', 'the softmax'],
        answer: 1,
        why: 'A handful of activation channels in a trained LLM carry values roughly 100× the typical magnitude, and because INT8 quantisation of activations usually uses one shared scale per tensor, that scale has to stretch to cover the outliers, wasting almost the entire 256-point grid on a range the ordinary channels never visit — exactly the granularity problem §3.13.1 already showed for weights, now hitting activations where per-channel scales are harder to apply cleanly. "Lack of hardware support" is the tempting wrong answer because INT8 throughput genuinely did depend on newer tensor-core generations at one point, so it sounds like the plausible bottleneck — but INT8 arithmetic itself has been broadly supported for years, and the four named fixes in §3.13.1’s flag (LLM.int8() isolating outlier channels into fp16, SmoothQuant migrating the difficulty into weights, AWQ protecting the weight channels that matter most, GPTQ using second-order reconstruction) all attack the same numerical outlier problem, not a missing hardware feature. Insufficient training data and the softmax are both unrelated to how a fixed, already-trained set of activation values gets rounded onto an integer grid. The general principle is the one the granularity row in §3.13.1’s table already established for weights: a single scale is only as good as the widest range it has to cover, and a handful of extreme values can ruin the whole grid for everyone else.'
      }
    ],
    cards: [
      { q: 'Quantisation formula', a: '$q=\\mathrm{round}(x/s)+z$, $\\hat x = s(q-z)$. Group-wise scales are what make 4-bit viable.' },
      { q: 'Why 4-bit is 4× faster to decode', a: 'Decode is memory-bandwidth-bound: 70B at bf16 reads 140 GB/token, at 4-bit 35 GB/token.' },
      { q: 'Pruning that actually speeds things up', a: 'Structured (heads, channels, layers) or 2:4 semi-structured on supported hardware. Unstructured buys disk only.' },
      { q: 'Dark knowledge', a: 'The relative probabilities the teacher assigns to wrong answers. Temperature amplifies them; $T^2$ rescales the gradient.' },
      { q: 'Modern LLM distillation', a: 'Sequence-level: generate from the teacher, fine-tune the student on the generations. Subject to provider terms of service.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.14 */
  ML.section({
    id: 'robustness', track: 'deep', num: '3.14', level: 3,
    title: 'Adversarial examples and robustness',
    lede: 'A perturbation too small to see flips a confident prediction. Ten years after the discovery this is still not solved, and the honest state of the field — a measurable robustness/accuracy trade-off, and no defence that survives an adaptive attacker — is itself the thing worth knowing.',
    prereq: ['backprop'],
    related: ['safety', 'production', 'fairness'],
    html: `
<p>In 2013, Szegedy and colleagues took a photograph a network classified correctly, added a change so small no human looking at the two images side by side could tell them apart, and watched the same network confidently call the result something else entirely — and not something plausible, but a bizarre, high-confidence misclassification. That result did not describe a rare, exotic failure mode. It described something present, reliably and reproducibly, in every image classifier tested since, and the version of it most people have actually seen is Goodfellow, Shlens and Szegedy's 2015 panda: a photograph correctly classified as "panda" at 57.7% confidence, a barely-perceptible pattern added to it, and the same photograph now classified "gibbon" at 99.3% confidence. This section builds the attack that finds such a perturbation, the leading explanation for why it exists at all, and an honest account of what does and does not defend against it — because a decade on, this is still not a solved problem, and the field's own track record of confidently declaring it solved and then being proven wrong is itself part of what you need to know.</p>
${H.tldr([
      'FGSM: $x\' = x + \\epsilon\\,\\mathrm{sign}(\\nabla_x \\mathcal{L})$. One gradient step <i>with respect to the input</i>, in the direction that most increases the loss.',
      'Adversarial examples are not bugs in a particular model — they <b>transfer</b> between architectures and training sets, which suggests they exploit genuinely predictive but non-robust features in the data.',
      'Adversarial training is the only defence that has consistently survived. It costs 3–30× the training compute and measurably reduces clean accuracy. Almost everything else has been broken by adaptive attacks.'
    ])}

<h2><span class="sn">3.14.1</span> The attack</h2>
<p>Every optimisation problem in this course so far has moved the same kind of object: the weights, to lower the loss. An adversarial attack takes the identical machinery — a loss function and its gradient — and points it at a different variable entirely. Freeze the weights $\\theta$, and instead move the <i>input</i> $x$, in the direction that <b>raises</b> the loss rather than lowers it, subject to a budget that keeps the change small enough to be imperceptible: $\\|\\delta\\|_p \\le \\epsilon$, where $\\delta$ is the perturbation added to $x$ and $\\epsilon$ caps its size under some norm $p$. Written as an optimisation problem, an attack is:</p>
$$\\max_{\\|\\delta\\|_\\infty \\le \\epsilon} \\mathcal{L}(f_\\theta(x+\\delta), y)$$
<p>Read this the same way you read gradient descent's minimisation, just with the sign and the variable both flipped: instead of $\\min_\\theta$, it is $\\max_\\delta$; instead of moving the parameters that define the function, it moves the input the function is evaluated at. The label $y$ is held fixed at the <i>true</i> label throughout — the attack is defined as making the network wrong about what the input actually is, not merely making it uncertain.</p>
<p>The simplest way to (approximately) solve that maximisation is <b>FGSM</b>, the Fast Gradient Sign Method: take one step, of size $\\epsilon$, in the direction of the sign of the input gradient. It is a single backward pass, cheap enough to run on every training example if you wanted to, and it is deliberately crude — a linear approximation to a problem that is not actually linear. The table below places it among the family of attacks that trade cost for how faithfully they solve the true maximisation.</p>
${H.table(['Attack', 'Idea', 'Strength', 'Cost'], [
      ['FGSM', 'one signed gradient step', 'weak; useful as a diagnostic', '1 backward pass'],
      ['<b>PGD</b>', 'many small steps, projected back into the ball, random start', '<b>the standard benchmark</b>', '10–100 passes'],
      ['C&amp;W', 'optimisation with a margin objective', 'very strong, finds minimal perturbations', 'expensive'],
      ['AutoAttack', 'an ensemble of four parameter-free attacks', 'the current reporting standard', 'very expensive'],
      ['Square attack', 'query-only, no gradients', 'black-box baseline', 'many queries'],
      ['Transfer attack', 'attack your own surrogate, use it on theirs', 'works without any access at all', 'cheap — and the realistic threat model']
    ])}
<p>PGD is FGSM run iteratively and kept honest: take a small signed-gradient step, then project the result back inside the $\\epsilon$-ball around the original input if the step wandered outside it, repeat for tens or hundreds of iterations, and start from a random point inside the ball rather than the clean input itself so the attack is not accidentally stuck in one local direction. Where FGSM answers "roughly where is the gradient pointing", PGD actually searches, and it is PGD — not FGSM — that every serious robustness claim in §3.14.3 is measured against.</p>

<p><b>What you are looking at.</b> A genuinely trained two-layer network on a two-dimensional classification problem, drawn as a coloured decision surface with the training points scattered over it. The black dot is the point you are attacking; the dashed square around it is the $\\epsilon$-ball — every point the attack is allowed to move to. A coloured trail traces the attack's actual path from the original point toward wherever it ends up, and the second dot marks that final, adversarial point.</p>
<p><b>What to do with it.</b> Drag the point to sit safely in the middle of one colour's territory, well clear of the boundary, and watch how large $\\epsilon$ has to grow before the trail crosses into the other colour. Now drag the same point to sit just barely on the correct side of the boundary and repeat: a far smaller $\\epsilon$ suffices, often small enough that the dashed square looks tiny next to the one from a moment ago. Switch the step count from 1 (plain FGSM) up to a genuine multi-step PGD run and watch the trail bend to actually hug the boundary rather than taking one straight, sometimes wasteful, leap.</p>
<p><b>The thing genuinely worth noticing.</b> The readout's <i>ε needed to flip</i> field is, in this two-dimensional toy, exactly the distance from the point to the decision boundary — and that identity is the whole story of adversarial vulnerability in higher dimensions too. A point is only as safe as it is far from the boundary, and a network trained purely to minimise average error has no reason to push that distance up for every point; it only needs the boundary to be roughly in the right place on average. Adversarial training, in §3.14.3, is precisely the fix that makes "far from the boundary" part of the objective, not an accident of it.</p>
${H.lab('fgsm', 'Flip a real classifier with a perturbation you can measure', 'A genuinely trained network on a genuine 2-D problem. The attack computes $\\nabla_x\\mathcal{L}$ by central differences and steps along it. Watch the point cross a boundary it was nowhere near — and watch how much smaller ε needs to be near the boundary than far from it.')}

${H.intuition(`<p>Why does a tiny perturbation matter so much? In high dimensions, a small per-pixel budget is a large total budget: an $\\epsilon = 8/255$ change to every one of 150,528 pixels has L2 norm around 12, which is not small at all. The linear explanation is that $w^\\top\\delta$ can be large when $\\delta$ aligns with $w$ and $d$ is big, even with $\\|\\delta\\|_\\infty$ tiny. Deep networks are locally close to linear, so the attack needs one gradient rather than a search.</p>`)}

<h2><span class="sn">3.14.2</span> Why they transfer, and what that implies</h2>
<p>A fact about adversarial examples is more unsettling, on reflection, than the existence of the attack itself: a perturbation crafted against one network, using that network's own gradients, very often fools a <i>second</i>, independently trained network too — different architecture, different random initialisation, sometimes even a different training set. If adversarial examples were simply bugs particular to one model's idiosyncratic weights, this should not happen; two models with different bugs should not share a blind spot. The fact that they reliably do share one is the observation any credible account has to explain, and it is also, incidentally, why a "transfer attack" in the table above needs no access to the model being attacked at all — attack your own copy, and the result usually works on theirs.</p>
${H.key('The most credible account (Ilyas et al., 2019): datasets contain features that are genuinely predictive but not robust — real statistical signal that a small perturbation destroys. Models learn them because they work. Adversarial examples are therefore a property of the <i>data</i> as much as of the model, which is why they transfer across architectures.')}
<p>Read that account slowly, because it reframes the whole problem. It is not saying networks are picking up on noise or spurious correlations in the usual sense — a "non-robust feature" in this account is real signal, genuinely predictive of the label on the actual data distribution, that simply happens to be fragile: a small, human-imperceptible push in feature space flips it, even though it was informative right up until that push. Any model trained to minimise average error, on any architecture, has an incentive to use that signal, because it works, right up until someone perturbs exactly the direction it depends on. That is why the vulnerability is shared: it lives in the data's structure, not in one network's particular mistakes.</p>
${H.flag('This reframing is well supported but not universally accepted, and the field has a long history of defences that looked sound and were broken within months — usually because they obscured the gradient rather than removing the vulnerability. Any claim of a new defence should be read alongside whether it was evaluated against an <i>adaptive</i> attacker who knows the defence exists.')}
${H.history(`<p>The decade between Szegedy et al. (2013) and today has been a genuine arms race, not a steady march toward a solution, and it is worth knowing the shape of it rather than just the current scoreboard. Goodfellow et al.'s FGSM (2015) was proposed alongside the linear explanation still used above. A wave of defences followed through 2017–2018 — gradient masking, input transformations, detection networks — each published with an evaluation showing it worked. Athalye, Carlini and Wagner's 2018 paper <i>Obfuscated Gradients Give a False Sense of Security</i> then examined a batch of these ICLR-accepted defences and broke the large majority of them, showing that most were not removing the vulnerability but merely hiding the gradient an attacker needs — a distinction the field had not, until then, been rigorous about testing for. Madry et al.'s adversarial training (2018) is the defence that has best survived that scrutiny since, which is precisely why it anchors §3.14.3.</p>`)}

<h2><span class="sn">3.14.3</span> Defences, honestly assessed</h2>
<p>Given that history, the right way to read the table below is not "which of these works" but "which of these has actually been tested against an attacker who knows the defence is there and adapts to it" — the adaptive-attack standard the previous history box exists to justify.</p>
${H.table(['Defence', 'Idea', 'Status'], [
      ['<b>Adversarial training</b>', 'train on PGD examples generated on the fly', '<b>works</b>; 3–30× compute, and a real clean-accuracy cost'],
      ['TRADES', 'explicitly trade clean accuracy against robustness with a tunable term', 'works; makes the trade-off a dial'],
      ['Randomised smoothing', 'add Gaussian noise, take a majority vote', '<b>certified</b> robustness in L2 — a guarantee, at a modest accuracy cost'],
      ['Gradient masking / obfuscation', 'make gradients uninformative', '<b>broken</b> — repeatedly, by attacks that estimate the gradient another way'],
      ['Input preprocessing (JPEG, blur, quantise)', 'destroy the perturbation', 'broken by BPDA (approximate the non-differentiable step)'],
      ['Detection', 'classify inputs as adversarial', 'mostly broken; the attacker can attack the detector too'],
      ['Ensembles', 'attack must fool all members', 'raises the cost, does not solve it — perturbations transfer within the ensemble']
    ])}
${H.worked('the robustness/accuracy trade-off, in numbers', `
<p>On CIFAR-10 at $\\epsilon = 8/255$ (L∞), the picture has been stable for years:</p>
<ul>
<li>A standard model: ~95% clean, <b>~0%</b> robust under AutoAttack.</li>
<li>Adversarially trained: ~85% clean, ~50–60% robust — using extra generated data, the current leaders sit around 70%.</li>
<li>Randomised smoothing gives a <i>certificate</i>: a provable radius, at lower clean accuracy.</li>
</ul>
<p><b>Ten points of clean accuracy for fifty points of robustness</b> is the trade, and whether it is worth paying is a question about your threat model, not about machine learning. For a photo-tagging feature: no. For a content-moderation classifier with a motivated adversary: yes.</p>`)}

${H.more('Natural distribution shift is the more common problem', `
<p>Adversarial robustness gets the attention; ordinary distribution shift causes far more production failures. The taxonomy worth having:</p>
<ul>
<li><b>Covariate shift</b>: $p(x)$ changes, $p(y\\mid x)$ does not. New camera, new user segment. Detectable from inputs alone — this is what PSI monitoring catches (§2.18).</li>
<li><b>Label shift</b>: $p(y)$ changes, $p(x\\mid y)$ does not. Fraud rate triples. Fixable by reweighting if you can estimate the new prior.</li>
<li><b>Concept drift</b>: $p(y\\mid x)$ itself changes. The relationship you learned is no longer true. <b>Only labels can detect this</b>, which is why outcome monitoring cannot be replaced by input monitoring.</li>
</ul>
<p>The practical consequence: a model that is 95% accurate in the lab and 78% in production has almost certainly met covariate shift, not an adversary. Check the input distribution first.</p>`)}

${H.probe([
      ['Write down FGSM.', '$x\' = x + \\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L}(f(x),y))$ — one signed gradient step on the input, clipped to the valid range.'],
      ['Why does a defence that "hides the gradient" fail?', 'It removes the attacker’s easiest path, not the vulnerability. Adaptive attacks estimate the gradient by finite differences, transfer from a surrogate, or replace the non-differentiable step with a differentiable approximation (BPDA).'],
      ['What is the cost of adversarial training?', '3–30× training compute (each step needs a PGD inner loop) and roughly ten points of clean accuracy on CIFAR-10 at the standard budget.'],
      ['You see accuracy drop in production. How do you tell adversarial attack from distribution shift?', 'Shift moves the whole input distribution and is visible in PSI/KS on features. An attack leaves the marginal distribution nearly unchanged but produces high-confidence errors concentrated near boundaries. Check input drift first — it is far more likely.']
    ], 'Presenting adversarial robustness as solved, or as a niche curiosity. It is neither: it is unsolved, and it matters exactly as much as your threat model says it does.')}`,
    labs: {
      fgsm: function (host) {
        const st = Viz.controls(host, [
          { k: 'eps', label: 'perturbation budget ε', min: 0, max: 1.2, step: .02, value: .35, fmt: v => v.toFixed(2) },
          { k: 'steps', label: 'attack steps (1 = FGSM, >1 = PGD)', min: 1, max: 20, step: 1, value: 1, fmt: v => v },
          { k: 'px', label: 'point x', min: -2.2, max: 2.6, step: .05, value: -1.0, fmt: v => v.toFixed(2) },
          { k: 'py', label: 'point y', min: -1.6, max: 2.2, step: .05, value: 0.45, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'orig', label: 'original prediction', cls: 'key' },
          { k: 'adv', label: 'after attack', cls: 'bad' },
          { k: 'flip', label: 'flipped?' },
          { k: 'dist', label: 'perturbation ‖δ‖₂' },
          { k: 'need', label: 'ε needed to flip' }
        ]);
        const data = Num.dataset('moons', 240, .2, 8);
        const net = Num.mlp([2, 20, 14, 1], { seed: 6, act: 'tanh' });
        for (let i = 0; i < 1400; i++) net.trainBatch(data.X, data.y, .05);

        function loss(p, y) {
          const q = Math.min(1 - 1e-9, Math.max(1e-9, net.predict(p)));
          return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
        }
        function attack(p, eps, steps) {
          const y = net.predict(p) > .5 ? 1 : 0;
          let cur = p.slice();
          const path = [cur.slice()];
          const alpha = steps > 1 ? eps / steps * 2.2 : eps;
          for (let s = 0; s < steps; s++) {
            const g = Num.inputGrad(q => loss(q, y), cur, 1e-4);
            cur = cur.map((v, i) => v + alpha * Math.sign(g[i]));
            // project back into the L∞ ball around p
            cur = cur.map((v, i) => Math.max(p[i] - eps, Math.min(p[i] + eps, v)));
            path.push(cur.slice());
          }
          return { adv: cur, path: path, y: y };
        }

        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const p = [st.px, st.py];
            const a = attack(p, st.eps, st.steps);
            const p0 = net.predict(p), p1 = net.predict(a.adv);
            const P = Viz.plot(ctx, w, h, { xd: [-2.6, 3.0], yd: [-1.9, 2.4], pad: { l: 40, r: 14, t: 16, b: 34 } })
              .frame({ xlabel: 'x₁', ylabel: 'x₂' });
            P.clip(() => {
              Labs.boundary(P, (x, y) => net.predict([x, y]), { step: 4, lo: 0, hi: 1, alpha: 80 });
              Labs.points(P, data.X, data.y, { r: 2.4 });
              // the L∞ ball
              ctx.strokeStyle = T.text; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
              ctx.strokeRect(P.x(p[0] - st.eps), P.y(p[1] + st.eps),
                P.x(p[0] + st.eps) - P.x(p[0] - st.eps), P.y(p[1] - st.eps) - P.y(p[1] + st.eps));
              ctx.setLineDash([]);
              P.line(a.path, { color: T.c4, width: 1.8 });
              P.dots([p], { r: 6, color: T.text, stroke: true, strokeWidth: 2 });
              P.dots([a.adv], { r: 6, color: T.c2, stroke: true, strokeWidth: 2 });
              P.text(p[0], p[1], ' original', { dx: 9, dy: -11, color: T.text, font: '11px ui-sans-serif' });
              P.text(a.adv[0], a.adv[1], ' adversarial', { dx: 9, dy: 12, color: T.c2, font: '11px ui-sans-serif' });
            });
            // the smallest ε that flips it
            let need = null;
            for (let e = .02; e <= 1.6; e += .02) {
              const r = attack(p, e, Math.max(st.steps, 6));
              if ((net.predict(r.adv) > .5 ? 1 : 0) !== a.y) { need = e; break; }
            }
            out({
              orig: (p0 > .5 ? 'class 1' : 'class 0') + ' @ ' + (Math.max(p0, 1 - p0) * 100).toFixed(1) + '%',
              adv: (p1 > .5 ? 'class 1' : 'class 0') + ' @ ' + (Math.max(p1, 1 - p1) * 100).toFixed(1) + '%',
              flip: ((p1 > .5 ? 1 : 0) !== a.y) ? 'YES — attack succeeded' : 'no',
              dist: Math.hypot(a.adv[0] - p[0], a.adv[1] - p[1]).toFixed(3),
              need: need ? need.toFixed(2) : '> 1.6'
            });
          }
        });
        Viz.note(host, 'Move the point deep into a cluster: the ε needed to flip it grows, because the distance to the boundary is the thing the attack must cross. Move it near the boundary and a budget of 0.05 suffices. <b>Distance to the decision boundary is the local robustness</b>, and adversarial training works by explicitly pushing that distance up for every training point — which is also, unavoidably, why it moves the boundary away from where clean accuracy would have put it.');
      }
    },
    quiz: [
      {
        q: 'FGSM perturbs the input by…',
        options: ['random noise of magnitude ε', '$\\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L})$', 'the negative gradient of the loss', 'the model’s weights'],
        answer: 1,
        why: 'FGSM takes one step of a fixed size $\\epsilon$ in the direction of the sign of $\\nabla_x\\mathcal{L}$ — not the gradient’s magnitude, just which way each coordinate points — because the attack solves $\\max_\\delta$ under an $L_\\infty$ budget, and the sign is exactly what maximises the dot product with a fixed per-coordinate budget. "The negative gradient of the loss" is the tempting wrong answer because it is the direction every optimisation problem elsewhere in this course actually moves in — gradient descent on the weights always goes negative, to shrink the loss — and carrying that reflex over here gets the sign of the whole attack backwards: §3.14.1 is explicit that an attack freezes the weights and instead moves the input in the direction that raises the loss, the opposite of every training step you have written so far. "Random noise" ignores that the direction is chosen deliberately from the gradient rather than sampled, and "the model’s weights" confuses what is being attacked with what is being perturbed. The general principle worth carrying forward is that an adversarial attack is gradient ascent on the input holding the parameters fixed — the same machinery as training, pointed at a different variable with the opposite sign.'
      },
      {
        q: 'Adversarial examples transfer between independently trained models, which suggests…',
        options: ['all models share a bug', 'they exploit genuinely predictive but non-robust features of the data', 'the attacks are random', 'the models were trained on the same seed'],
        answer: 1,
        why: 'A perturbation crafted against one network reliably fools a second, differently architected one because both networks learned the same genuinely predictive but fragile statistical signal from the data — the Ilyas et al. account §3.14.2 builds around — and any model trained to maximise accuracy has an incentive to use that signal, since it works, right up until it is perturbed. "All models share a bug" is the tempting wrong answer because it is the intuitively satisfying explanation for two independent systems failing identically — shared symptom, shared cause, so it must be a shared flaw — but "bug" implies something accidental and architecture-specific, like a shared implementation mistake, when the section’s whole point is that the vulnerability lives in the data’s structure, not in any one network’s idiosyncratic weights, which is exactly why it survives a change of architecture, initialisation, and even training set. "The attacks are random" is directly contradicted by FGSM’s construction, which is a deliberate gradient step, and "same seed" is not a precondition transfer needs at all, since transfer is observed across genuinely different training runs. The general lesson, which also explains why a transfer attack needs no access to the victim model, is that adversarial vulnerability is best understood as a property of the dataset’s own statistics, not a defect to be patched in any particular network.'
      },
      {
        q: 'A defence that makes gradients uninformative is…',
        options: ['a strong defence', 'gradient masking — historically broken by adaptive attacks', 'certified robustness', 'equivalent to adversarial training'],
        answer: 1,
        why: 'Gradient masking makes the standard first-order attack fail to find a useful direction, but it does so by hiding the signal an attacker would use, not by moving the decision boundary further from any input — which is why Athalye, Carlini and Wagner’s 2018 paper found that the large majority of a whole wave of ICLR-accepted defences relying on this idea were broken once an adaptive attacker estimated the gradient another way, by finite differences, by attacking a substitute model and transferring the result, or by replacing a non-differentiable step with a differentiable approximation (BPDA). "A strong defence" is the tempting wrong answer because that is exactly how these defences looked when evaluated only against the standard gradient-based attack they were designed to defeat — the attack genuinely fails, the reported robustness number genuinely looks good — and the whole point of §3.14.2’s history box is that a defence evaluated only this way has not really been tested, only tested against an attacker too naive to route around it. Certified robustness (randomised smoothing) is a real, different thing that provides a mathematical guarantee rather than an empirically fragile obstacle, and gradient masking is not equivalent to adversarial training, which changes what the model actually learns rather than what an attacker can see. The general standard this section insists on, stated explicitly in §3.14.3, is that any defence claim should be read alongside whether it was tested against an attacker who already knows the defence exists.'
      },
      {
        q: 'Production accuracy dropped 15 points. The most likely cause is…',
        options: ['an adversarial attack', 'covariate shift — the input distribution moved', 'a bug in the loss function', 'label leakage'],
        answer: 1,
        why: 'A double-digit, across-the-board accuracy drop is the signature of covariate shift — the input distribution $p(x)$ moving while the labelling function stays the same, as §3.14.3’s taxonomy describes — and it is detectable directly from the inputs alone, via PSI or KS statistics, without ever needing a label. "An adversarial attack" is the tempting wrong answer for a very specific reason: you have just spent this entire section reading about adversarial examples, so reaching for that explanation the moment something goes wrong in production is the most available hypothesis, not necessarily the most likely one — and the section’s own probe warns against exactly this, pointing out that a real attack leaves the marginal input distribution nearly unchanged while producing high-confidence errors concentrated near decision boundaries, a narrower and rarer signature than a broad fifteen-point drop. A bug in the loss function would typically show up during training, not as a sudden field-only drop, and label leakage inflates offline metrics rather than degrading live ones. The general habit the whole section is built to install is to check the boring, common explanation — has the input distribution simply moved — before reaching for the rare, dramatic one, because ordinary distribution shift causes far more production failures than adversaries ever do.'
      }
    ],
    cards: [
      { q: 'FGSM', a: '$x\' = x + \\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L})$. PGD is the iterative, projected, random-start version — the benchmark.' },
      { q: 'Why perturbations transfer', a: 'They exploit non-robust but genuinely predictive features in the data, so independently trained models learn the same vulnerability.' },
      { q: 'The robustness trade-off', a: 'CIFAR-10 at ε=8/255: ~95% clean / ~0% robust, vs ~85% clean / ~55% robust after adversarial training.' },
      { q: 'Three kinds of shift', a: 'Covariate ($p(x)$), label ($p(y)$), concept ($p(y|x)$). Only the last needs labels to detect.' }
    ]
  });
})();
