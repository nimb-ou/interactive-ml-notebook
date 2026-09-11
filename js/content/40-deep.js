/* ============================================================
   PART 3 — Neural networks & deep learning (3.1 – 3.6)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 3.1 */
  ML.section({
    id: 'nn-fundamentals', track: 'deep', num: '3.1',
    title: 'Neural network fundamentals',
    lede: 'The bridge into modern AI: every training detail in this part recurs at ten thousand times the scale in Part 4.',
    rests: 'Rests on §0.2 (matrices as functions on space), §0.3 (chain rule), §1.5 (the loss), §1.9 (conditioning).',
    html: `
<p>Take four points sitting at the corners of a unit square: $(0,0)$ and $(1,1)$ belong to class 0, while $(1,0)$ and $(0,1)$ belong to class 1. This is the XOR pattern, and it looks like the most harmless dataset imaginable — four points, two classes, perfectly balanced. Now try to separate the two classes the way §2.4's logistic regression does: draw one straight line and put every 0 on one side of it and every 1 on the other.</p>

<p>There is no such line. Walk around the square corner by corner and the label flips every single time, so any line you draw has at least one point stranded on the wrong side of it — try the diagonals, try anything through the middle, it does not matter. A linear model is exactly one cut across the whole feature space, and this pattern cannot be described by one cut. This is not a contrived example chosen to be difficult; it is the simplest possible case of a much larger fact, which is that most of the structure worth modelling in images, audio and language is not linearly separable either. Logistic regression, on its own, is stuck.</p>

<h2><span class="sn">3.1.1</span> The tempting fix that does nothing</h2>

<p>The obvious next move is to stack two linear models: run the input through one linear layer, then feed the result into a second linear layer, hoping the extra step buys extra power. It is worth checking that idea in full before reaching for anything more complicated, because the check is short and the answer is completely decisive.</p>

${H.deriv('why two linear layers are exactly as powerful as one', [
      ['$h = W_1 x + b_1$', 'The first layer: an ordinary linear layer from §0.2, producing a new list of numbers from the input $x$.'],
      ['$\\hat y = W_2 h + b_2$', 'The second layer, fed the first layer\'s output $h$ instead of the raw input.'],
      ['$\\hat y = W_2(W_1 x + b_1) + b_2$', 'Substitute the first line into the second. Nothing clever has happened yet — this is just writing out what "feed the output into the next layer" means.'],
      ['$= (W_2 W_1) x + (W_2 b_1 + b_2)$', 'Expand and regroup. Matrix multiplication is associative, so $W_2(W_1 x)$ equals $(W_2 W_1)x$ (§0.2), and the two bias terms combine into one constant vector by ordinary vector addition.'],
      ['$= Wx + b, \\quad W := W_2W_1, \\;\\; b := W_2b_1 + b_2$', 'Name the combined matrix and the combined bias. Whatever $W_1$, $W_2$, $b_1$ and $b_2$ happened to be, this final line has exactly the shape of a single linear layer.']
    ], 'Nothing in this argument used any property of $W_1$ or $W_2$ beyond their being matrices, so it holds for any two linear layers, and by induction for any chain of them — two, twenty, two hundred. Stack a hundred linear layers and you still have, at most, the expressive power of one. Depth bought you nothing: more parameters to store and more arithmetic to do, computing a function a single layer could already represent.')}

<p>Put numbers through it so the claim is not just symbolic. Take $W_1 = \\begin{bmatrix}2 & 0\\\\ 1 & 1\\end{bmatrix}$ and $W_2 = \\begin{bmatrix}1 & 1\\\\ 0 & 3\\end{bmatrix}$, both with zero bias, and feed in $x = (1, 2)$. Going layer by layer: $h = W_1x = (2\\times1+0\\times2,\\; 1\\times1+1\\times2) = (2, 3)$, and then $\\hat y = W_2 h = (1\\times2+1\\times3,\\; 0\\times2+3\\times3) = (5, 9)$. Now collapse first: $W = W_2W_1 = \\begin{bmatrix}3 & 1\\\\ 3 & 3\\end{bmatrix}$, and $Wx = (3\\times1+1\\times2,\\; 3\\times1+3\\times2) = (5, 9)$ — the identical answer, reached without ever computing $h$. The two-layer network and the single collapsed matrix are not merely similar. They are the same function.</p>

${H.history(`<p>This is not a modern observation. It is essentially the objection Marvin Minsky and Seymour Papert raised in their 1969 book <i>Perceptrons</i>, which showed that a single-layer perceptron — no hidden layer at all — cannot represent XOR, and cast serious doubt on whether stacking would help, since a stack of purely linear units collapses to one. The book is widely credited with cooling funding and enthusiasm for neural networks for most of the 1970s, a period sometimes called the first "AI winter". The irony is that the fix was already implicit in the objection: it is not depth that was missing, it is a nonlinearity between the layers. Once backpropagation made training multi-layer networks with nonlinearities practical — popularised by Rumelhart, Hinton and Williams in 1986 — the XOR problem became a textbook first example rather than an open question, and it still is one (§0.3 walks a full backprop derivation on a network like the one below).</p>`)}

<h2><span class="sn">3.1.2</span> The nonlinearity, and what it actually buys you</h2>

<p>The fix is to put a nonlinear function — an <b>activation</b> — between the two linear steps, applied to every entry of $h$ separately: $\\hat y = W_2\\,\\phi(W_1x + b_1) + b_2$, where $\\phi$ (the Greek letter phi) might be as simple as $\\phi(z) = \\max(0, z)$, which zeroes out every negative entry and leaves positive entries untouched. This is called a <b>ReLU</b>, short for rectified linear unit, and it is deliberately the least exotic nonlinear function imaginable — its whole job is to <i>not</i> be linear, and it does that job with a single comparison.</p>

<p>The moment $\\phi$ sits between the two matrices, the collapsing argument above breaks. $W_2\\phi(W_1x+b_1)+b_2$ cannot in general be rewritten as $Wx+b$ for any fixed matrix $W$, because $\\phi$ treats different regions of the input differently — it keeps some coordinates and discards others, and which coordinates get discarded changes as $x$ moves. A single linear map can never do that; every linear map treats the whole of space the same way, stretching and rotating it uniformly (§0.2). A nonlinearity is what lets one layer's output depend on <i>which region</i> the input landed in, not merely on a fixed weighted sum of it, and that is precisely the extra freedom XOR needs.</p>

${H.worked('an exact, hand-built network that solves XOR — no training required', `<p>Take two hidden units with weights $W_1 = \\begin{bmatrix}1&1\\\\1&1\\end{bmatrix}$ and biases $b_1 = (0, -1)$, a ReLU activation, and an output layer $w_2 = (1, -2)$ with $b_2 = 0$. Check all four corners.</p>
<p>$(0,0)$: both units see $x_1+x_2=0$, so $h_1=\\mathrm{ReLU}(0)=0$ and $h_2=\\mathrm{ReLU}(0-1)=0$. Output $= 1(0) - 2(0) = 0$. Correct.</p>
<p>$(1,0)$ and $(0,1)$: both see $x_1+x_2=1$, so $h_1=\\mathrm{ReLU}(1)=1$ and $h_2=\\mathrm{ReLU}(1-1)=0$. Output $=1(1)-2(0)=1$. Correct on both corners, by the symmetry of $x_1+x_2$.</p>
<p>$(1,1)$: both see $x_1+x_2=2$, so $h_1=\\mathrm{ReLU}(2)=2$ and $h_2=\\mathrm{ReLU}(2-1)=1$. Output $=1(2)-2(1)=0$. Correct.</p>
<p>All four labels come out exactly right, with weights chosen by hand rather than learned. The two hidden units carve the plane into three bands along the direction $x_1+x_2$ — below 0, between 0 and 1, above 1 — and the output layer reads off which band a point fell into. That is the entire mechanism: each ReLU unit contributes a fold in the surface, and the output layer is once again linear, just linear <i>on top of a surface that has already been bent</i>.</p>`)}

${H.analogy(`<p>A single linear layer is like a pane of flat glass: whatever pattern of light passes through it comes out uniformly stretched, sheared or dimmed, but a straight edge going in is always a straight edge coming out. Stacking flat panes of glass, however many you like, still leaves you with flat glass — the combined effect of any number of uniform stretches is itself a uniform stretch (that is exactly the matrix-multiplication argument above).</p>
<p>An activation function is a crease pressed into the glass at every point. One crease lets you fold the light differently on either side of it. Stack creased panes and the folds compound: the second pane can bend what the first pane already bent, in a way a single sheet of glass, however you stretched it, never could. Depth is valuable only once there is something to crease.</p>`)}

<h2><span class="sn">3.1.3</span> From perceptron to universal approximator</h2>

<p>The building block above — a linear combination of inputs, thresholded or squashed by a nonlinearity — is called a <b>perceptron</b> when the nonlinearity is a hard step: $\\hat y = \\mathbb{1}[w^\\mathsf{T}x+b>0]$, where $\\mathbb{1}[\\cdot]$ is 1 when the condition holds and 0 otherwise. A single perceptron is exactly the linear classifier that could not solve XOR. A <b>multilayer perceptron</b>, or MLP, is what you get by stacking several layers of perceptron-like units with a smooth nonlinearity between them and training the whole stack with backpropagation (§3.2) rather than hand-deriving the weights as in the worked example above.</p>

<p>Push the width of a single hidden layer high enough and something strong becomes true: a network with one sufficiently wide hidden layer can approximate any continuous function on a bounded region of input space to any desired accuracy. This is the <b>universal approximation theorem</b>, and it is the formal statement behind the informal claim that neural networks can "learn anything".</p>

${H.flag('That theorem is weaker than it sounds, and the gap between what it promises and what people assume it promises is worth stating precisely. It says a good approximation <i>exists</i> somewhere among the possible settings of the weights. It says nothing about how wide "sufficiently wide" needs to be for a given function — the required width can grow astronomically for functions with fine detail. It says nothing about whether gradient descent, starting from a random initialisation, will ever find that setting rather than getting stuck somewhere worse (§1.12). And it says nothing about how much data you would need to pin the right function down among all the ones the network could represent, which is a generalisation question entirely separate from an approximation one. Depth is what makes the approximation efficient in practice: composing many small nonlinear bends, layer after layer, reaches functions that a single enormous layer could only reach with far more units, because each new layer can reuse and recombine the features the previous layer already built.')}

<h2><span class="sn">3.1.4</span> Watching a network build its own features</h2>

<p><b>What you are looking at.</b> The lab below trains a real multilayer network — forward pass, backpropagation, Adam optimiser — entirely in your browser, on one of five two-dimensional datasets. The main panel shows the data points, coloured by class, with the network's current decision boundary drawn as a shaded region behind them: blue where it predicts one class, red the other, with the intensity of the colour showing how confident the prediction is. To the right, a small grid of tiny heatmaps shows the first hidden layer, one panel per unit. Each panel colours every point in the plane by how strongly that single hidden unit activates there, so you are looking directly at one learned feature at a time — literally the function $\\phi(w^\\mathsf{T}x+b)$ for one unit's own weights. Below those panels, a small line plot tracks the training loss as it falls.</p>

<p><b>What to do with it.</b> Start on <i>circles</i> with one hidden layer of two units and press <i>Train 200 epochs</i>. Watch the two unit panels: each one settles into a crude half-plane, shaded on one side and not the other, because with a linear-in-effect first layer and only two of them, a half-plane cut is all a single unit can represent. Then switch to <i>spiral</i> with the same tiny network and train again — the loss plateaus, the boundary stays a rough wedge, and it never resolves the spiral no matter how long you train it. Now raise the width to 8 and the depth to 3 and train again on the spiral: the unit panels stop looking like clean half-planes and start looking like curved bands, and the boundary follows the spiral's arms.</p>

<p><b>The thing genuinely worth noticing.</b> The failure on the two-unit spiral is not a training failure — running it for longer does not fix it, because the problem is capacity, not optimisation. Two units can produce at most two folds in the decision surface, and a spiral needs many. Once you add width or depth, the same Adam optimiser, on the same data, finds a boundary that actually wraps around the arms, and the individual unit panels show <i>why</i>: later units build curved features out of combinations of the cruder half-plane features the first layer produced. That progressive elaboration — features built out of features — is what "depth" buys once there is a nonlinearity to make it real, and it is the thing every much larger network in Part 4 is also doing, just with millions of units instead of a handful you can watch directly.</p>

${H.lab('playground', 'Build and train a neural network', 'Choose a dataset, a shape and an activation, then train. The small panels are the individual hidden units — each one is a feature the network invented. Watch them specialise: on the spiral, units first learn crude half-planes, then bend.')}

<h2><span class="sn">3.1.5</span> What each knob actually does</h2>
<p>Every control in that lab maps onto a real modelling decision, and the same knobs reappear, scaled up, throughout the rest of this course.</p>
${H.table(['Knob', 'Effect', 'Failure mode when wrong'], [
      ['<b>Width</b>', 'How many features per layer — how many folds one layer can add', 'Too narrow: underfits, loss plateaus high, as the two-unit spiral above showed'],
      ['<b>Depth</b>', 'How composed those features can be — later layers combine earlier ones', 'Too deep without residuals/normalisation: gradients vanish (§3.3)'],
      ['<b>Activation</b>', 'The shape of the fold', 'Saturating activations kill gradients through depth'],
      ['<b>Learning rate</b>', 'Step size in gradient descent (§0.3)', 'Too high: loss oscillates or NaNs. Too low: never arrives'],
      ['<b>Batch size</b>', 'Gradient noise per step', 'Too large: wasted compute past the critical batch size (§4.11)'],
      ['<b>Weight decay</b>', 'Pull toward small weights', 'Too high: underfits; too low: memorises']
    ])}

<h2><span class="sn">3.1.6</span> The features are the point</h2>
<p>Classical ML asks you to engineer features by hand and then fits a simple model on top of them (§2.11): someone decides that "ratio of two columns" or "distance to the nearest cluster centre" is a useful quantity, computes it once, and hands it to a linear model or a tree. A neural network collapses that two-stage process into one. The early layers <i>are</i> the feature engineering — the hidden-unit panels in the lab above are literally features being invented — and the final layer is a simple linear or logistic model sitting on top of them, trained jointly with the features it depends on rather than after them.</p>

<p>That joint training is why neural networks win wherever the useful features are hard to state in advance — pixels, waveforms, raw text — and why they usually lose on small-to-medium tabular data, where a human already knows the meaningful quantities (age, income, days-since-last-purchase) and a boosted tree (§2.8) can exploit that structure directly, with less data and far less tuning. The network is not spending its capacity discovering "income divided by household size matters" when a person could simply have told it so.</p>

${H.probe([
      ['Why do you need a nonlinearity?', 'Composed linear maps collapse to a single linear map, as the associativity argument in §3.1.1 shows directly: $W_2(W_1x+b_1)+b_2 = Wx+b$ for some fixed $W$ and $b$. Without a nonlinearity between layers, depth adds parameters and computation but no new functions the network can represent.'],
      ['What does the universal approximation theorem actually guarantee?', 'Existence of an approximating network somewhere in weight space — nothing about how wide it must be, whether gradient descent can find it, or how much data would be needed to identify it. Depth, not the theorem itself, is what makes approximation practical.'],
      ['When would you not use a neural net?', 'Small-to-medium tabular data with features a person already understands — gradient boosting is usually more accurate, needs less data, and is far easier to explain than a network spending its capacity rediscovering feature interactions a domain expert could have named directly.']
    ])}`,
    labs: {
      playground: function (host) {
        let data = Num.dataset('circles', 220, .22, 5);
        let net = null, running = false, hist = [], epoch = 0;
        const st = Viz.controls(host, [
          { k: 'dataset', label: 'dataset', type: 'select', value: 'circles', options: [
            { v: 'circles', t: 'circles' }, { v: 'moons', t: 'moons' }, { v: 'xor', t: 'xor' }, { v: 'spiral', t: 'spiral' }, { v: 'blobs', t: 'blobs (linear)' }] },
          { k: 'depth', label: 'hidden layers', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'width', label: 'units per layer', min: 2, max: 12, step: 1, value: 6, fmt: v => v },
          { k: 'act', label: 'activation', type: 'buttons', value: 'tanh', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }, { v: 'sigmoid', t: 'sigmoid' }] },
          { k: 'lr', label: 'learning rate', min: -3.3, max: -.3, step: .1, value: -1.3, fmt: v => Math.pow(10, v).toFixed(3) },
          { k: 'l2', label: 'weight decay', min: 0, max: .05, step: .001, value: 0, fmt: v => v.toFixed(3) },
          { k: 'noise', label: 'data noise', min: .05, max: .6, step: .05, value: .22, fmt: v => v.toFixed(2) }
        ], rebuild);
        const out = Viz.readout(host, [
          { k: 'loss', label: 'train loss', cls: 'key' }, { k: 'acc', label: 'train accuracy' },
          { k: 'tacc', label: 'test accuracy', cls: 'good' }, { k: 'ep', label: 'epochs' }, { k: 'params', label: 'parameters' }
        ]);
        let test = null;
        function rebuild() {
          data = Num.dataset(st.dataset, 220, st.noise, 5);
          test = Num.dataset(st.dataset, 220, st.noise, 999);
          const sizes = [2];
          for (let i = 0; i < st.depth; i++) sizes.push(st.width);
          sizes.push(1);
          net = Num.mlp(sizes, { act: st.act, seed: 3 });
          hist = []; epoch = 0; S.redraw();
        }
        function trainSteps(n) {
          for (let i = 0; i < n; i++) { const l = net.trainBatch(data.X, data.y, Math.pow(10, st.lr), st.l2); hist.push(l); epoch++; }
        }
        const S = Viz.surface(host, {
          height: 380,
          draw: function (ctx, w, h, T) {
            const panelW = Math.min(190, w * .32);
            const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6], pad: { l: 36, r: panelW + 24, t: 14, b: 34 } })
              .frame({ xlabel: 'x₁', ylabel: 'x₂' });
            P.clip(() => Labs.boundary(P, (x, y) => net.predict([x, y]), { step: 4 }));
            P.clip(() => Labs.points(P, data.X, data.y, { r: 3.2 }));
            // hidden unit panels (first hidden layer)
            const nUnits = Math.min(st.width, 8);
            const cols = 2, cell = Math.min(60, (panelW - 12) / cols);
            const ox = w - panelW - 6, oy = 22;
            ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('hidden layer 1 — learned features', ox, oy - 5);
            for (let u = 0; u < nUnits; u++) {
              const cx = ox + (u % cols) * (cell + 6), cy = oy + Math.floor(u / cols) * (cell + 6);
              const img = ctx.createImageData(Math.round(cell), Math.round(cell));
              for (let py = 0; py < cell; py++) for (let px = 0; px < cell; px++) {
                const x = -3.4 + 6.8 * px / cell, y = 2.6 - 5.2 * py / cell;
                const a = net.forward([x, y]).as[1][u];
                const t = Math.max(0, Math.min(1, (a + 1) / 2));
                const i4 = (py * Math.round(cell) + px) * 4;
                img.data[i4] = Math.round(90 + 130 * t);
                img.data[i4 + 1] = Math.round(110 + 60 * (1 - Math.abs(t - .5) * 2));
                img.data[i4 + 2] = Math.round(220 - 140 * t);
                img.data[i4 + 3] = 190;
              }
              Viz.blit(ctx, img, Math.round(cx), Math.round(cy), cell, cell);
              ctx.strokeStyle = T.line; ctx.strokeRect(cx, cy, cell, cell);
            }
            // loss curve
            if (hist.length > 2) {
              const lx = ox, ly = oy + Math.ceil(nUnits / cols) * (cell + 6) + 14, lw = panelW - 6, lh = 52;
              ctx.fillStyle = T.paper; ctx.fillRect(lx, ly, lw, lh);
              ctx.strokeStyle = T.line; ctx.strokeRect(lx, ly, lw, lh);
              const mx = Math.max.apply(null, hist.slice(0, 20));
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6; ctx.beginPath();
              const step = Math.max(1, Math.floor(hist.length / lw));
              for (let i = 0; i < hist.length; i += step) {
                const X = lx + lw * i / (hist.length - 1), Y = ly + lh - lh * Math.min(1, hist[i] / (mx || 1));
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              }
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('loss', lx + 4, ly + 3);
            }
            const acc = Num.mean(data.X.map((x, i) => ((net.predict(x) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            const tacc = Num.mean(test.X.map((x, i) => ((net.predict(x) > .5 ? 1 : 0) === test.y[i]) ? 1 : 0));
            let params = 0;
            net.W.forEach(l => l.forEach(r => params += r.length + 1));
            out({
              loss: hist.length ? hist[hist.length - 1].toFixed(4) : '—',
              acc: (acc * 100).toFixed(1) + '%', tacc: (tacc * 100).toFixed(1) + '%',
              ep: epoch, params: params
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Train 200 epochs', primary: true, on: () => { trainSteps(200); S.redraw(); } },
          { label: 'Train / pause', on: () => {
            running = !running;
            const t = setInterval(() => { if (!running) return clearInterval(t); trainSteps(6); S.redraw(); }, 30);
            ML.onCleanup(() => clearInterval(t));
          } },
          { label: 'Reset weights', on: rebuild }
        ]);
        rebuild();
        Viz.note(host, 'Try the spiral with 1 hidden layer of 2 units — it cannot be done; the model does not have enough features. Add width, then depth, and watch the boundary become possible. That progression is what "capacity" means, made visible.');
      }
    },
    quiz: [
      {
        q: 'You stack three linear layers with no activation between them. The result is…',
        options: ['a deep model with more capacity', 'exactly equivalent to a single linear layer', 'a model that cannot be trained', 'a convex problem with better conditioning'],
        answer: 1,
        why: 'Composition of linear maps is linear, and the proof is nothing more than matrix associativity: $W_3(W_2(W_1x)) = (W_3W_2W_1)x$ is one matrix applied once, exactly as §3.1.1 derives for two layers and induction extends to any number. Option A is the tempting one, because more layers really do mean more parameters and more arithmetic — but neither of those is the same thing as more <i>representable functions</i>, and here the extra parameters buy nothing: every setting of $W_3, W_2, W_1$ collapses to some single matrix $W$, so the family of functions three linear layers can express is identical to the family one layer can express. The nonlinearity between layers is what breaks that collapse and is the only thing that makes depth mean anything.'
      },
      {
        q: 'The universal approximation theorem guarantees…',
        options: ['that gradient descent will find a good network', 'that a sufficiently wide network can approximate any continuous function on a compact set', 'good generalisation', 'that depth beats width'],
        answer: 1,
        why: 'It is a pure existence result about approximation: somewhere among all the possible weight settings for a wide-enough single hidden layer, one of them gets arbitrarily close to any continuous function on a bounded region. Option A is the trap, because "a good network exists" and "gradient descent will find it starting from a random initialisation" are entirely different claims — the theorem is silent on optimisation, and §1.12 is where the difficulty of actually finding good minima gets treated properly. It says nothing about how much data identifying the right function requires (generalisation) and nothing about whether width or depth is the more efficient route to it — in practice depth usually wins, which is the opposite of what the theorem, read carelessly, might suggest.'
      }
    ],
    cards: [
      { q: 'Why do neural nets need nonlinearity?', a: 'Composed linear maps collapse to one linear map; without it, depth adds nothing.' },
      { q: 'Universal approximation — the caveat', a: 'It guarantees existence, not learnability, efficiency or generalisation. Depth is what makes it practical.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.2 */
  ML.section({
    id: 'backprop', track: 'deep', num: '3.2',
    title: 'Backpropagation, by hand',
    lede: 'Nothing more than the chain rule applied layer by layer in reverse, reusing cached forward activations so the whole gradient costs about one forward pass. That is the entire reason deep learning is computationally feasible.',
    rests: 'This section leans directly on two derivations done in full elsewhere: §0.3.5 derives the chain rule and applies it to a one-unit network; §0.7.6 costs both bracketing orders of the resulting matrix product and shows reverse mode wins by a factor of hundreds on a toy network, millions at real scale. Neither is rederived here — this section is those two results, applied layer by layer to a real multi-layer network, with the bookkeeping made concrete.',
    html: `
<p>A network with three hidden layers computes a composition $L = \\ell(f_3(f_2(f_1(x))))$: the input goes through a layer, then another, then another, and finally a loss compares the result with a label. Suppose you need $\\partial L/\\partial W^{(1)}$, the gradient of the loss with respect to the very first layer's weights, buried three compositions deep. You already know, from §0.3.5, that this is a product of local derivatives along the path from the loss back to that parameter. The question this section answers is not <i>what</i> to multiply — that was settled in §0.3 — but <i>how to organise the multiplication</i> so that computing the gradient of a million-parameter network costs about the same as running it once forward, rather than a million times.</p>

<h2><span class="sn">3.2.1</span> Why you cannot just write the derivative out</h2>

<p>The naive plan is to expand $L$ symbolically as one long formula in terms of every weight, then differentiate that formula with respect to each parameter in turn. This is exactly what §0.7.1 did for a two-parameter linear model, and it worked fine there. It stops working the moment the network has any depth to it, for a reason that has nothing to do with calculus being hard: the expanded formula for $L$ in terms of a deep network's weights has a term for every path from an input to the output, and the number of such paths multiplies at every layer. Differentiating that expanded mess by hand, one weight at a time, is the same combinatorial explosion §0.3.1 used to rule out grid-searching a loss — asking one question at a time about a problem whose size grows exponentially in its depth.</p>

<p>What rescues the situation is the same thing that rescued gradient descent in §0.3: never expand the formula. Keep the network as a sequence of small, simple steps, and apply the chain rule <i>mechanically</i>, one layer at a time, reusing intermediate results rather than recomputing them. That reuse is backpropagation, and it is worth being precise about what is genuinely new here versus what is a direct application of results already earned. The chain rule itself, and the observation that a gradient is a product of local derivatives along a path, is §0.3.5 in full. The observation that this product should be evaluated starting from the scalar loss and working backwards, rather than starting from the many parameters and working forwards, is §0.7.6's cost-and-shape argument, with the concrete tally of 535,040 multiply-adds one way against 104,775,328 the other on a four-layer example. Backpropagation is the discipline of doing exactly that, layer after layer, with one extra ingredient: caching.</p>

${H.history(`<p>Reverse-mode differentiation predates neural networks by more than a decade. Seppo Linnainmaa's 1970 thesis described the general technique — propagate a Jacobian backwards through a chain of computations rather than forwards — as a piece of numerical analysis, with no connection to learning at all. Paul Werbos applied essentially the same idea to neural networks specifically in his 1974 PhD thesis, and it went almost unread outside a small circle.</p>
<p>The idea did not become the default way to train a network until Rumelhart, Hinton and Williams's 1986 paper popularised it under the name "backpropagation" and, crucially, demonstrated it training networks to do things a linear model provably could not — solving problems like XOR that need at least one hidden layer, which §3.1 shows is exactly where a single layer's linear ceiling bites. The sixteen-year gap between "the algorithm exists" and "the algorithm is the standard tool" was not a gap in the mathematics, which §0.3.5 and §0.7.6 both show is nothing more than the chain rule organised sensibly. It was a gap in believing multi-layer networks were worth training at all, and in having datasets and hardware where the answer to "is it worth it" was clearly yes. The lesson generalises past this one algorithm: a correct derivation sitting unused is usually a sign that the surrounding ecosystem has not caught up yet, not a sign that the idea was wrong.</p>`)}

<h2><span class="sn">3.2.2</span> A worked step, by hand</h2>

<p>Before the general machinery, work one concrete case all the way through, forward and back, small enough to check by hand at every line.</p>

${H.worked('worked backprop — one full step, by hand', `
<p>A two-layer net with one unit per layer: $z_1 = w_1x + b_1$, $a_1 = \\mathrm{ReLU}(z_1)$, $\\hat y = w_2a_1 + b_2$, loss $L = \\tfrac12(\\hat y - y)^2$. Take $x=1$, $w_1=0.5$, $w_2=1.0$, biases 0, target $y=1$.</p>
<p><b>Forward.</b> $z_1 = 0.5(1) = 0.5$. $a_1 = \\mathrm{ReLU}(0.5) = 0.5$, since $0.5 > 0$ the gate is open. $\\hat y = 1.0(0.5) = 0.5$. $L = \\tfrac12(0.5-1)^2 = \\tfrac12(0.25) = 0.125$.</p>
<p><b>Backward, one link of the chain at a time.</b> Start where the loss is defined: $\\partial L/\\partial\\hat y = \\hat y - y = 0.5 - 1 = -0.5$, exactly the residual, by the power rule applied to the square (§0.3.2). Now step back through the second layer: $\\partial L/\\partial w_2 = (\\partial L/\\partial\\hat y)\\cdot(\\partial\\hat y/\\partial w_2) = (-0.5)(a_1) = (-0.5)(0.5) = -0.25$, and $\\partial L/\\partial a_1 = (\\partial L/\\partial\\hat y)\\cdot(\\partial\\hat y/\\partial a_1) = (-0.5)(w_2) = (-0.5)(1.0) = -0.5$. Step back through the ReLU: since $z_1 = 0.5 > 0$ the gate's local derivative is 1 (§0.3.5), so $\\partial L/\\partial z_1 = (-0.5)(1) = -0.5$, unchanged. Step back through the first layer: $\\partial L/\\partial w_1 = (\\partial L/\\partial z_1)\\cdot x = (-0.5)(1) = -0.5$.</p>
<p><b>Update</b> at $\\eta = 0.1$: $w_2 \\leftarrow 1.0 - 0.1(-0.25) = 1.025$; $w_1 \\leftarrow 0.5 - 0.1(-0.5) = 0.55$.</p>
<p><b>Re-forward, to check the step actually helped.</b> $z_1 = 0.55$, $a_1 = 0.55$, $\\hat y = 1.025 \\times 0.55 = 0.56375$, $L = \\tfrac12(0.56375-1)^2 = \\tfrac12(0.43625)^2 \\approx 0.0952$ — down from 0.125, exactly as gradient descent promises (§0.3.2).</p>
<p>Two things about this arithmetic generalise directly to networks a million times larger. First, every gradient here was a product of local factors read straight off the forward computation — $x$, $w_2$, and the ReLU gate — multiplied along the single path from the loss back to each parameter, with nothing rederived from scratch at each step. Second, $a_1$ was needed again during the backward pass, to compute $\\partial L/\\partial w_2$, well after the forward pass had already produced it and moved on. That reuse is not incidental; it is the entire reason backward passes need a cache of forward values, and §3.2.4 below makes the memory cost of that cache explicit.</p>`)}

<h3>Watching numbers flow both ways</h3>

<p><b>What you are looking at.</b> Five boxes laid left to right are the stages of the worked example above: the input $x$, the pre-activation $z_1 = w_1x+b_1$, the activation $a_1 = \\phi(z_1)$, the prediction $\\hat y = w_2a_1+b_2$, and the loss $L$. The bold number inside each box is that stage's forward <i>value</i>, computed left to right along the blue arrows. The red number beneath each box is $\\partial L/\\partial\\,\\cdot$ for that stage, computed right to left along the red arrows — the backward pass. The line beneath the diagram spells out the two products you actually want, $\\partial L/\\partial w_2$ and $\\partial L/\\partial w_1$, exactly as multiplied out in the worked box above.</p>

<p><b>What to do with it.</b> Four controls are yours to set directly: the input $x$, the target $y$, the learning rate $\\eta$, and the activation function. The weights $w_1, w_2$ (and the biases, held at 0) are the network's own parameters — you do not set them directly, you only change them by training, exactly as in real backpropagation. Press <i>Reset to the worked example</i> to land on $x=1$, $y=1$, $\\eta=0.1$ with ReLU, matching the worked box above line for line. Then press <i>Take one step</i> and check the printed $\\partial L/\\partial w_2$ and $\\partial L/\\partial w_1$ against the arithmetic you just did by hand — they should match to three decimal places. Press it again, or press <i>Run 50 steps</i>, and watch the loss curve in the corner fall.</p>

<p><b>The thing genuinely worth noticing.</b> Switch the activation to sigmoid and watch $\\partial L/\\partial w_1$ shrink relative to the ReLU case, even though $\\partial L/\\partial w_2$ barely changes. The difference is entirely the local gate: a ReLU that is active contributes a factor of exactly 1 to the product passing through it, while a sigmoid contributes at most $0.25$ (§0.3.2's rule table). With only two layers the effect is a modest shrink you can read off the readout. Stack forty layers of sigmoid instead of two and that same factor, multiplied forty times, is the vanishing gradient — the subject of the next section, and something you are already watching happen in miniature here.</p>

${H.lab('bp', 'Backprop, step by step, with your numbers', 'Every intermediate value and every gradient, recomputed as you move the inputs. Press <i>step</i> to apply the update and watch the loss fall — the same arithmetic as the worked box, under your control.')}

<h2><span class="sn">3.2.3</span> The general algorithm</h2>
<p>The worked example above had one unit per layer, so every product was a product of scalars. A real layer has many units, so each local derivative becomes a vector or a matrix, and the chain rule becomes the matrix chain rule of §0.7.6: multiply Jacobians instead of numbers, in the same right-to-left order, for the same reason. Written as an algorithm, layer by layer:</p>
${H.steps([
      '<b>Forward:</b> compute and cache $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$ and $a^{(l)} = \\phi(z^{(l)})$ for every layer, keeping every $a^{(l)}$ in memory rather than discarding it.',
      '<b>Output error:</b> $\\delta^{(L)} = \\nabla_a L \\odot \\phi\'(z^{(L)})$, where $\\odot$ means multiply entry by entry — for sigmoid output with cross-entropy loss this pair of local derivatives cancels down to the clean $\\hat y - y$ derived in full in §0.7.6.',
      '<b>Propagate:</b> $\\delta^{(l)} = (W^{(l+1)\\mathsf{T}}\\delta^{(l+1)}) \\odot \\phi\'(z^{(l)})$ — exactly the linear-layer backward rule $\\partial\\mathcal{L}/\\partial x = W^\\top g$ derived from first principles in §0.7.6, applied once per layer, with the activation\'s local derivative folded in.',
      '<b>Gradients:</b> $\\partial L/\\partial W^{(l)} = \\delta^{(l)}a^{(l-1)\\mathsf{T}}$ and $\\partial L/\\partial b^{(l)} = \\delta^{(l)}$ — the outer product $\\partial\\mathcal{L}/\\partial W = g\\,x^\\top$ from the same derivation, with $\\delta^{(l)}$ playing the role of $g$ and $a^{(l-1)}$ the role of the layer\'s input.'
    ])}
<p>Read step 4 as a sentence, because it is the one line every framework's <code>backward()</code> implements: <b>the gradient of a weight matrix is the error arriving at this layer, outer-producted with the activation that fed it.</b> A unit that received a large incoming activation gets a proportionally large weight gradient, which is the same "features that are always large dominate the update" fact §0.2 and §0.3.4 raised about scaling — it now shows up as a property of every single layer, not just the input.</p>

${H.code(`# backprop in numpy, for a 2-layer net — this is the whole idea
z1 = X @ W1 + b1;  a1 = np.maximum(0, z1)      # cache a1 — needed again below
z2 = a1 @ W2 + b2; p  = 1 / (1 + np.exp(-z2))

d2 = (p - y) / len(X)                          # dL/dz2 for BCE+sigmoid — the clean p - y
dW2 = a1.T @ d2;        db2 = d2.sum(0)        # step 4: error outer-producted with the input it saw
d1  = (d2 @ W2.T) * (z1 > 0)                   # step 3: propagate through W2ᵀ, then gate through ReLU
dW1 = X.T  @ d1;        db1 = d1.sum(0)        # step 4 again, one layer earlier`)}

<h3>Shapes, written out</h3>
<p>Before trusting that code, or any implementation of it, make every shape in the four steps explicit — §0.2.5's "shapes are the debugging tool" habit, applied to the one place in this course where a shape bug is most likely to run silently rather than crash. Write $n_l$ for the width of layer $l$, so $n_0$ is the number of input features and $n_L$ the number of outputs, and suppose a real implementation processes $B$ examples at once, stacked into a batch. Every quantity below then carries an extra leading dimension of size $B$ in practice; the single-example column is the shape the algebra in §3.2.3 was written in.</p>

${H.table(['Quantity', 'Shape (one example)', 'Shape (batch of B)', 'Read as'], [
      ['$a^{(l-1)}$, the input this layer receives', '$n_{l-1}$', '$B\\times n_{l-1}$', 'one row per example, one column per incoming feature'],
      ['$W^{(l)}$', '$n_l\\times n_{l-1}$', 'unchanged — weights are never batched', 'row $i$ holds the weights feeding output unit $i$'],
      ['$b^{(l)}$', '$n_l$', '$n_l$, broadcast down the batch', 'one bias per output unit, added to every row'],
      ['$z^{(l)},\\ a^{(l)}$', '$n_l$', '$B\\times n_l$', 'one pre- and post-activation value per output unit, per example'],
      ['$\\delta^{(l)}=\\partial L/\\partial z^{(l)}$', '$n_l$', '$B\\times n_l$', 'exactly the shape of $z^{(l)}$ — the backward pass never invents a new shape, it only reuses forward ones'],
      ['$\\partial L/\\partial W^{(l)}=\\delta^{(l)}a^{(l-1)\\mathsf T}$', '$n_l\\times n_{l-1}$', 'same, summed over the batch', 'always exactly $W^{(l)}$\'s own shape, which is the point of the outer product']
    ])}

${H.pitfall('That last row is worth treating as a standing sanity check rather than a fact to memorise: a weight gradient that does not come out the same shape as the weight it belongs to is not a subtle numerical issue, it is a wiring mistake — a transpose left out, or a matrix multiplied on the wrong side. Frameworks will not necessarily raise an error for this; broadcasting rules happily produce a wrong-shaped or silently-broadcast result that runs, and even trains, badly, without complaint. Confirming $\\partial L/\\partial W^{(l)}$\'s shape against $W^{(l)}$\'s shape before trusting a single number out of a backward pass is the single highest-value debugging habit this section can hand you.')}

<h2><span class="sn">3.2.4</span> Why forward activations must be cached — and what caching them costs</h2>
<p>Look again at step 4: $\\partial L/\\partial W^{(l)}$ needs $a^{(l-1)}$, the activation the layer <i>received</i> during the forward pass, not anything computed during the backward pass. If that value had not been kept in memory, computing this one gradient would mean re-running the forward pass from the input up to layer $l-1$ all over again — for every layer, at every training step. Caching is not an optimisation added on top of backpropagation; it is the thing that makes the reverse-mode cost analysis of §0.7.6 actually hold. Without the cache, "one backward pass costs about one forward pass" becomes false, because part of the forward pass gets paid for twice.</p>
<p>This is also, precisely, the trade activation checkpointing makes in the other direction (§4.11): it deliberately discards some of the cache to save memory, accepting that a partial forward recomputation will be needed later. Seen from here, checkpointing is not a separate trick — it is choosing, layer by layer, to pay the cost that caching exists to avoid, in exchange for fitting a bigger model in the same GPU memory.</p>

<p>Caching buys back time, but it is not free — it trades exactly the recomputation §0.7.6 wants to avoid for memory instead, and it is worth putting a number on that trade rather than leaving "the cache costs memory" as a vague warning. Every one of the $L$ layers has to keep its own $a^{(l-1)}$ alive from the moment the forward pass produces it until the moment the backward pass consumes it — the entire duration of one full forward-then-backward step. Sum that over every layer and every example in the batch and you get the total size of the cache, and it is worth comparing that total directly against the one other big consumer of memory in the same step: the weights themselves.</p>

${H.deriv('when the activation cache outweighs the weights', [
      ['activation memory (fp32, bytes) $= 4B\\sum_{l=0}^{L-1} n_l$', 'One cached value per unit, per example, at every layer a later layer needs to look back at — that is layers $0$ through $L-1$, since layer $L$\'s output is the network\'s prediction, consumed immediately rather than cached. Four bytes per number in single precision, $B$ examples in the batch.'],
      ['parameter memory (fp32, bytes) $= 4\\sum_{l=1}^{L} n_l n_{l-1}$', 'Every weight matrix, once, regardless of batch size — biases are a lower-order term here and left out for a clean comparison.'],
      ['activations exceed parameters once $B > B^\\star := \\dfrac{\\sum_{l=1}^{L} n_l n_{l-1}}{\\sum_{l=0}^{L-1} n_l}$', 'Set the two expressions equal and solve for $B$: the factor of 4 cancels from both sides, leaving a ratio of a sum of products to a sum of widths — a single crossover batch size, fixed entirely by the network\'s shape.']
    ], 'Notice what does and does not appear in $B^\\star$: it depends only on the widths $n_l$, never on the learning rate, the data, or how long training runs. Parameter memory is a fixed cost, paid once, the instant the model is built. Activation memory is a running cost that grows in direct proportion to the batch size, because every additional example needs its own complete copy of every layer\'s cache. A network can be arbitrarily cheap to store and arbitrarily expensive to train, or the other way round, purely as a function of which side of $B^\\star$ you are training on.')}

${H.practice(`<p>Put real numbers through $B^\\star$ for a network with layer widths $784, 1024, 1024, 1024, 10$ — four weight matrices, roughly the shape of a small classifier with generously wide hidden layers. Summing $n_ln_{l-1}$ over the four weight matrices gives $784\\times1024+1024\\times1024+1024\\times1024+1024\\times10\\approx2{,}910{,}208$ parameters; summing $n_l$ over the four layers a later layer needs to look back at gives $784+1024+1024+1024=3{,}856$. The crossover is $B^\\star=2{,}910{,}208/3{,}856\\approx755$.</p>
<p>Below that batch size, parameter memory dominates outright: at $B=128$, the cache is $4\\times128\\times3{,}856\\approx2.0$ MB against a fixed $4\\times2{,}910{,}208\\approx11.6$ MB of weights — the weights are the memory bill you are actually paying, nearly six times over. Push the batch to $B=1{,}024$, comfortably past $B^\\star$, and the cache alone is $4\\times1{,}024\\times3{,}856\\approx15.8$ MB, already larger than every weight in the network combined.</p>
<p>The lesson generalises past this one toy network. A wide, shallow network — few layers, each one enormous — has a large $\\sum n_ln_{l-1}$ relative to its $\\sum n_l$, so $B^\\star$ is large and parameters usually win. A deep network built from many layers of comparatively modest width — closer to a real transformer's shape, and closer still once you count every attention and feed-forward sub-layer as its own entry in the sum — pushes $\\sum n_l$ up much faster than $\\sum n_ln_{l-1}$, so $B^\\star$ falls, and activation memory can dominate at batch sizes that would look modest for a shallow network. §4.11's activation-checkpointing lab is this exact trade-off, measured on a real training run rather than a toy MLP: memory bought back by discarding cache and paying recomputation instead, worthwhile precisely because that run is deep enough, and its batch large enough, to be sitting well past its own $B^\\star$.</p>`)}

${H.analogy(`<p>Caching forward activations is memoisation, the identical trick you would reach for in any recursive computation that revisits the same sub-problem more than once: compute a value once, write it down, and read the note back rather than recomputing it. The forward pass computes $a^{(l-1)}$ once, on its way to computing everything after it; the backward pass needs that exact same number again, unchanged, to compute a weight gradient. Throwing it away and recomputing it is not a bug, in the way a memory leak is a bug — it is a genuine choice on a real trade-off, exactly the choice activation checkpointing makes deliberately in the other direction, and exactly why "memoise or recompute" is a question that shows up wherever the same intermediate value is needed more than once, backpropagation included.</p>`)}

${H.key('The forward pass and the backward pass are not two independent costs. The backward pass is cheap in compute precisely because the forward pass already paid for the numbers it needs — caching is what banks that payment, and the bill for banking it is memory that scales with batch size and depth, not with the number of parameters.')}

${H.probe([
      ['Why reverse-mode rather than forward-mode?', 'One scalar loss, many parameters. §0.7.6 costs both bracketing orders of the same Jacobian product on a concrete four-layer example and finds reverse mode cheaper by a factor of 196 there, growing without bound as the network grows — reverse mode costs about one sweep per output, forward mode about one per input, and a loss has one output against millions of inputs.'],
      ['Why must activations be cached?', 'The weight gradient at layer $l$ is $\\delta^{(l)}a^{(l-1)\\mathsf{T}}$ (§3.2.3, step 4) — it needs the activation the layer received on the way forward. Without the cache you would have to recompute part of the forward pass during the backward pass, which is exactly the cost activation checkpointing (§4.11) knowingly re-introduces to save memory.'],
      ['Where do vanishing gradients come from, in this picture?', 'Every $\\delta^{(l)}$ in step 3 is the previous $\\delta^{(l+1)}$ multiplied by a local activation derivative. Multiply enough factors below 1 along a long chain — sigmoid contributes at most 0.25 per layer — and the product compounds toward zero geometrically, which §3.3 develops in full.']
    ])}`,
    labs: {
      bp: function (host) {
        let w1 = .5, w2 = 1.0, b1 = 0, b2 = 0, steps = 0, lossHist = [];
        const st = Viz.controls(host, [
          { k: 'x', label: 'input x', min: -2, max: 2, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'y', label: 'target y', min: -2, max: 3, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'lr', label: 'learning rate η', min: .01, max: .8, step: .01, value: .1, fmt: v => v.toFixed(2) },
          { k: 'act', label: 'activation', type: 'buttons', value: 'relu', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }, { v: 'sig', t: 'sigmoid' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'loss', label: 'loss', cls: 'key' }, { k: 'dw1', label: '∂L/∂w₁' }, { k: 'dw2', label: '∂L/∂w₂' },
          { k: 'w', label: 'weights' }, { k: 'steps', label: 'steps taken' }
        ]);
        function fwd() {
          const z1 = w1 * st.x + b1;
          const a1 = st.act === 'relu' ? Math.max(0, z1) : st.act === 'tanh' ? Math.tanh(z1) : Num.sigmoid(z1);
          const da = st.act === 'relu' ? (z1 > 0 ? 1 : 0) : st.act === 'tanh' ? 1 - a1 * a1 : a1 * (1 - a1);
          const yh = w2 * a1 + b2;
          const L = .5 * (yh - st.y) * (yh - st.y);
          const dL = yh - st.y;
          return { z1, a1, da, yh, L, dL, dw2: dL * a1, db2: dL, dz1: dL * w2 * da, dw1: dL * w2 * da * st.x, db1: dL * w2 * da };
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const f = fwd();
            const nodes = [
              { l: 'x', v: st.x, g: null },
              { l: 'z₁ = w₁x + b₁', v: f.z1, g: f.dz1 },
              { l: 'a₁ = φ(z₁)', v: f.a1, g: f.dL * w2 },
              { l: 'ŷ = w₂a₁ + b₂', v: f.yh, g: f.dL },
              { l: 'L = ½(ŷ−y)²', v: f.L, g: 1 }
            ];
            const bw = Math.min(122, (w - 40) / nodes.length - 8), cy = 74;
            nodes.forEach((n, i) => {
              const cx = 20 + i * (bw + 8) + bw / 2;
              Labs.roundRect(ctx, cx - bw / 2, cy - 24, bw, 48, 8);
              ctx.fillStyle = T.panel; ctx.fill(); ctx.strokeStyle = T.line; ctx.stroke();
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(n.l, cx, cy - 10);
              ctx.fillStyle = T.blue; ctx.font = 'bold 14px ui-monospace, monospace';
              ctx.fillText(n.v.toFixed(3), cx, cy + 10);
              if (n.g !== null) {
                ctx.fillStyle = T.red; ctx.font = '11px ui-monospace, monospace';
                ctx.fillText('∂L/∂· = ' + n.g.toFixed(3), cx, cy + 40);
              }
              if (i < nodes.length - 1) {
                const nx = 20 + (i + 1) * (bw + 8) + bw / 2;
                ctx.strokeStyle = T.blue; ctx.lineWidth = 1.5;
                ctx.beginPath(); ctx.moveTo(cx + bw / 2 + 1, cy - 8); ctx.lineTo(nx - bw / 2 - 3, cy - 8); ctx.stroke();
                ctx.strokeStyle = T.red;
                ctx.beginPath(); ctx.moveTo(nx - bw / 2 - 3, cy + 8); ctx.lineTo(cx + bw / 2 + 1, cy + 8); ctx.stroke();
              }
            });
            ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('∂L/∂w₂ = (ŷ−y)·a₁ = ' + f.dw2.toFixed(3), 20, cy + 66);
            ctx.fillText('∂L/∂w₁ = (ŷ−y)·w₂·φ′(z₁)·x = ' + f.dw1.toFixed(3), 20, cy + 88);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.fillText('every gradient is a product of local derivatives along the path back to it', 20, cy + 112);
            // loss history
            if (lossHist.length > 1) {
              const lx = w - 160, ly = 20, lw = 140, lh = 52;
              ctx.fillStyle = T.paper; ctx.fillRect(lx, ly, lw, lh);
              ctx.strokeStyle = T.line; ctx.strokeRect(lx, ly, lw, lh);
              const mx = Math.max.apply(null, lossHist);
              ctx.strokeStyle = T.green; ctx.lineWidth = 1.8; ctx.beginPath();
              lossHist.forEach((v, i) => {
                const X = lx + lw * i / Math.max(1, lossHist.length - 1), Y = ly + lh - lh * (v / (mx || 1)) * .9 - 3;
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              });
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('loss over steps', lx + 4, ly + 3);
            }
            out({
              loss: f.L.toFixed(4), dw1: f.dw1.toFixed(4), dw2: f.dw2.toFixed(4),
              w: 'w₁=' + w1.toFixed(3) + ' w₂=' + w2.toFixed(3), steps: steps
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Take one step', primary: true, on: () => { const f = fwd(); w1 -= st.lr * f.dw1; w2 -= st.lr * f.dw2; b1 -= st.lr * f.db1; b2 -= st.lr * f.db2; steps++; lossHist.push(fwd().L); S.redraw(); } },
          { label: 'Run 50 steps', on: () => { for (let i = 0; i < 50; i++) { const f = fwd(); w1 -= st.lr * f.dw1; w2 -= st.lr * f.dw2; b1 -= st.lr * f.db1; b2 -= st.lr * f.db2; steps++; lossHist.push(fwd().L); } S.redraw(); } },
          { label: 'Reset to the worked example', on: () => { w1 = .5; w2 = 1; b1 = b2 = 0; steps = 0; lossHist = []; st.$set('x', 1); st.$set('y', 1); st.$set('lr', .1); S.redraw(); } }
        ]);
        Viz.note(host, 'Switch the activation to sigmoid and watch ∂L/∂w₁ shrink: the gate now passes at most 0.25 instead of 1. Two layers hides the effect; forty layers does not, and that single factor is most of the story of why deep networks were hard to train before ReLU.');
      }
    },
    quiz: [
      {
        q: 'In the worked example, $\\partial L/\\partial w_1 = -0.5$. Which chain produces it?',
        options: ['$(\\hat y - y)\\cdot a_1$', '$(\\hat y-y)\\cdot w_2\\cdot \\phi\'(z_1)\\cdot x$', '$(\\hat y - y)\\cdot x^2$', '$w_1 \\cdot x$'],
        answer: 1,
        why: 'Four local derivatives multiplied along the single path from $L$ back to $w_1$: the residual $\\hat y - y$ at the loss, the output weight $w_2$ crossed on the way back through the second layer, the activation gate $\\phi\'(z_1)$ (1 for an active ReLU), and finally $x$, the input the first layer actually saw. Option A is the tempting distractor because it is the <i>correct</i> gradient for $w_2$, one layer later in the chain — mixing up which layer\'s local factors belong in which product is the single most common backprop-by-hand mistake, and it is exactly what the worked derivation in §3.2.2 walks past step by step so you cannot skip a link.'
      },
      {
        q: 'Activation checkpointing trades…',
        options: ['accuracy for speed', 'memory for recomputation — it discards cached activations and recomputes them in the backward pass', 'batch size for depth', 'precision for range'],
        answer: 1,
        why: 'The weight-gradient rule $\\partial L/\\partial W^{(l)} = \\delta^{(l)}a^{(l-1)\\mathsf{T}}$ needs the activation a layer received on the forward pass, so a standard backward pass keeps every one of those activations in memory for the whole training step. Checkpointing deliberately throws most of that cache away and, when the backward pass later needs a discarded activation, recomputes it with a small extra forward pass from the nearest surviving checkpoint. It genuinely costs compute — typically one extra forward pass\'s worth — in exchange for fitting a model that would not otherwise fit in memory at all, which is a trade worth making far more often than it sounds.'
      }
    ],
    cards: [
      { q: 'Backprop in one sentence', a: 'The chain rule evaluated right-to-left with cached forward activations, so the full gradient costs about one forward pass.' },
      { q: 'The weight-gradient rule', a: '$\\partial L/\\partial W^{(l)} = \\delta^{(l)}a^{(l-1)\\mathsf{T}}$ — error at this layer times activation from below.' },
      { q: 'Worked example numbers', a: 'Forward L = 0.125; ∂L/∂w₂ = −0.25, ∂L/∂w₁ = −0.5; after one step at η=0.1, L = 0.095.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.3 */
  ML.section({
    id: 'activations', track: 'deep', num: '3.3',
    title: 'Activations and the vanishing gradient',
    lede: 'One plot explains a decade of architecture history: what a function’s derivative does to a product of forty factors.',
    html: `
<p>§3.2 ended on a small, worrying observation: switching the one-unit network's activation from ReLU to sigmoid visibly shrank $\\partial L/\\partial w_1$, even with only two layers between the parameter and the loss. Every gradient in a network is a product of local derivatives along the path back from the loss (§0.3.5, §3.2.3), and a sigmoid's local derivative is never larger than a ReLU's. Two layers barely notice. Forty do. This section makes that difference precise, walks through why it decided a decade of architecture choices, and works through the three fixes that let networks train at real depth today.</p>

<h2><span class="sn">3.3.1</span> The derivative that started it</h2>

<p>The sigmoid, $\\sigma(z) = 1/(1+e^{-z})$, was the default nonlinearity for most of the 1990s and 2000s, chosen because it squashes any real number into $(0,1)$ and reads naturally as a probability. Its derivative is worth deriving once rather than just quoting, because the shape of the result — not just its formula — is what causes the trouble.</p>

${H.deriv('the sigmoid derivative, $\\sigma\'(z) = \\sigma(z)(1-\\sigma(z))$', [
      ['$\\sigma(z) = \\dfrac{1}{1+e^{-z}} = (1+e^{-z})^{-1}$', 'Rewrite as a power so the power rule and chain rule (§0.3.2) apply directly.'],
      ['$\\sigma\'(z) = -(1+e^{-z})^{-2}\\cdot(-e^{-z}) = \\dfrac{e^{-z}}{(1+e^{-z})^2}$', 'Chain rule: differentiate the outer power, then multiply by the derivative of the inner expression $1+e^{-z}$, which is $-e^{-z}$ (§0.3.2\'s exponential rule); the two minus signs cancel.'],
      ['$= \\dfrac{1}{1+e^{-z}}\\cdot\\dfrac{e^{-z}}{1+e^{-z}}$', 'Split the fraction into two factors, both built from the same expression $1+e^{-z}$.'],
      ['$= \\sigma(z)\\cdot\\dfrac{(1+e^{-z})-1}{1+e^{-z}} = \\sigma(z)\\,(1-\\sigma(z))$', 'Rewrite the second factor as $1 - \\dfrac{1}{1+e^{-z}} = 1-\\sigma(z)$, using the definition of $\\sigma$ again.']
    ], 'The result is a product $\\sigma(1-\\sigma)$ of two numbers that always sum to 1 — the product of two numbers summing to a constant is maximised when they are equal, at $\\sigma=0.5$, giving $0.5\\times0.5=0.25$. That is the ceiling: no matter what $z$ is, $\\sigma\'(z) \\le 0.25$, and it falls toward zero as $z$ moves away from 0 in either direction. A unit whose pre-activation sits far from zero — which is most units, most of the time, once training has moved weights away from their random start — contributes almost nothing to any gradient product that passes through it.')}

<p>Tanh, the zero-centred alternative that largely replaced sigmoid through the 2000s, has the same shape of problem in a milder form: its derivative $1-\\tanh^2z$ tops out at 1 rather than 0.25, but still collapses to 0 as $|z|$ grows. Both functions are <b>saturating</b>: flat on both sides, with a derivative that vanishes away from the origin by construction, because "squash everything into a bounded range" and "have a derivative that survives everywhere" are directly in tension — a function that is bounded above and below cannot keep rising forever, so its slope has to fall off somewhere.</p>

${H.table(['Activation', 'Definition', 'Derivative range', 'Character'], [
      ['<b>Sigmoid</b>', '$1/(1+e^{-z})$', '(0, 0.25]', 'Saturates both sides; historical; still the output layer for binary probability'],
      ['<b>Tanh</b>', '$\\tanh z$', '(0, 1]', 'Zero-centred sigmoid; better than sigmoid, still saturates'],
      ['<b>ReLU</b>', '$\\max(0,z)$', '{0, 1}', 'The default for a decade; cheap, non-saturating on the positive side; can "die"'],
      ['<b>Leaky ReLU</b>', '$\\max(\\alpha z, z)$', '{α, 1}', 'Fixes dead units at the cost of a hyperparameter'],
      ['<b>GELU</b>', '$z\\,\\Phi(z)$', 'smooth', 'Smooth ReLU; the transformer default'],
      ['<b>SiLU / Swish</b>', '$z\\,\\sigma(z)$', 'smooth', 'Very close to GELU; the gate inside SwiGLU (§4.6)']
    ])}

<p><b>What you are looking at.</b> The lab below plots each activation $\\phi$ on top and its derivative $\\phi'$ underneath, on the same horizontal axis $z$. A faint dashed sigmoid is drawn in both panels for reference whenever the comparison toggle is on. The green horizontal line in the lower panel marks derivative $=1$: the threshold that decides, layer after layer, whether a gradient product grows or shrinks.</p>

<p><b>What to do with it.</b> Start on sigmoid and read the lower panel: the curve never reaches the green line, peaking at exactly the $0.25$ the derivation above predicts, and falling to nearly nothing within a few units of the origin in either direction. Switch to ReLU and the lower panel becomes two flat segments, at exactly 0 and exactly 1 — no gentle taper, a hard switch. Switch to GELU or SiLU and note that the derivative panel is smooth, mostly hovering near 1, but dips slightly below zero just to the left of the origin, a property plain ReLU does not have.</p>

<p><b>The thing genuinely worth noticing.</b> The readout's <i>after 10 layers (max)</i> field raises the peak derivative to the tenth power — a rough proxy for what ten stacked layers of this activation do to a gradient passing through all of them at their most favourable point. On sigmoid that number is of order $10^{-6}$ even under the best case, where every single unit happens to sit exactly at $z=0$; in practice units drift away from zero during training and the true figure is worse. On ReLU it stays exactly 1, because $1^{10}=1$ regardless of depth. That one exponent, applied to a number below versus at 1, is the entire reason the choice of activation decided whether networks past a handful of layers could train at all.</p>

${H.lab('act', 'Activations and their derivatives, side by side', 'The lower panel is the one that matters: it is the factor each layer contributes to the gradient product. Note how sigmoid never exceeds 0.25 and how ReLU passes exactly 1 wherever it is active.')}

<h2><span class="sn">3.3.2</span> Why depth used to be impossible</h2>
<p>Backpropagation multiplies one such factor per layer, exactly as §3.2.3's step 3 does: $\\delta^{(l)} = (W^{(l+1)\\mathsf{T}}\\delta^{(l+1)})\\odot\\phi'(z^{(l)})$, one activation-derivative gate at every single layer on the way back. With sigmoid, each factor is at most $0.25$, so after 10 layers the gradient reaching layer 1 is scaled by at most $0.25^{10} \\approx 9.5\\times10^{-7}$; after 40 layers, at most $0.25^{40} \\approx 8\\times10^{-25}$ — a number with no meaningful representation in the floating-point arithmetic any framework uses. The early layers of such a network receive an update indistinguishable from zero and simply never move, no matter how long training runs or how good the data is. This is the <b>vanishing gradient</b>, and it is not a bug in any particular implementation; it is what the chain rule does, mechanically, to a long product of small numbers. Exploding gradients are the mirror image: a typical local factor above 1, compounding upward instead of downward, usually surfacing as a loss that turns to NaN within the first few hundred steps.</p>

${H.history(`<p>This is the reason very deep networks were considered impractical for most of the 1990s and 2000s, well after the theory of backpropagation itself (§0.3.5's history box) was settled. Networks of two or three hidden layers trained tolerably; ten or more, with sigmoid or tanh units, routinely failed to learn anything in their early layers, and the community's working assumption for years was that depth simply did not help much in practice, whatever the universal approximation theorem (§3.1) said about the theory.</p>
<p>The turning point was less a single paper than a sequence of small mechanical fixes to exactly this multiplication problem. Xavier Glorot and Yoshua Bengio's 2010 analysis of why deep networks were hard to train pointed squarely at activation saturation and initial-variance mismatch (§3.4 derives the second half of that argument). Glorot, Bordes and Bengio's 2011 paper then showed that simply replacing sigmoid and tanh with ReLU — a function whose derivative is exactly 1 wherever it is active, discussed below — measurably improved deep network training. AlexNet's 2012 ImageNet result, which used ReLU throughout an eight-layer network, is usually credited as the moment the field noticed, and ReLU became close to a universal default within two or three years. GELU (Hendrycks and Gimpel, 2016) and SiLU/Swish (Ramachandran, Zoph and Le, 2017) came later, as smoother variants motivated by slightly different arguments, and GELU in particular is now the default inside transformer feed-forward blocks (§4.5, §4.6). None of these are exotic ideas. Each one is a direct response to the single number $0.25$ derived above.</p>`)}

${H.lab('vanish', 'The gradient product, across depth', 'Set the activation and the depth, and watch the gradient magnitude reaching layer 1. This is computed by running an actual backward pass through a randomly initialised network — the collapse is not a cartoon.')}

<p><b>What you are looking at.</b> This lab builds an actual randomly-initialised network of the width and depth you choose, runs one real forward pass, then one real backward pass from a random gradient at the output, and plots $\\log_{10}$ of the gradient's magnitude at every layer from the last back to the first. The horizontal axis is layer number, counting from 1 at the input; the vertical axis is a logarithm because the quantities span so many orders of magnitude that a linear scale would show nothing but a flat line at zero.</p>

<p><b>What to do with it.</b> Start on sigmoid at depth 20 with residual connections off. The line falls almost as a straight ramp on this logarithmic scale — a straight line here means the underlying quantity is shrinking geometrically, exactly as the powers-of-0.25 argument above predicts — and by layer 1 the readout reports a verdict of <i>vanished</i>. Switch to ReLU with everything else unchanged: the line stays close to flat, because a factor of 1 raised to any power is still 1. Push the depth control to 40 on sigmoid and watch the readout's ratio between the last layer and layer 1 grow by roughly ten orders of magnitude for every doubling of depth.</p>

<p><b>The thing genuinely worth noticing.</b> Turn on residual connections while still using sigmoid. The collapse largely disappears, even though every individual sigmoid gate is exactly as saturating as it was a moment ago. Nothing about the activation changed; what changed is that the gradient now has an additional path back to layer 1 that does not pass through any activation gate at all, contributing a term of exactly 1 at every layer regardless of what the sigmoid units are doing. That is precisely the mechanism §3.3.3 names as the second of three fixes, and seeing it rescue an otherwise-doomed sigmoid network is a stronger argument for residual connections than any amount of prose.</p>

<h2><span class="sn">3.3.3</span> The three fixes, and what each one does</h2>
<p>Every fix that actually solved the vanishing-gradient problem attacks the same product of local derivatives from a different angle, and it is worth naming all three together because a real modern architecture uses all three at once rather than choosing between them.</p>
${H.table(['Fix', 'Mechanism', 'Where you meet it'], [
      ['<b>ReLU family</b>', 'Local derivative is exactly 1 on the active side, so the product does not decay', 'Everywhere since ~2012'],
      ['<b>Residual connections</b>', '$y = x + F(x)$ gives a path whose local derivative is exactly 1, so gradients reach layer 1 unattenuated', 'Every transformer block (§4.5)'],
      ['<b>Normalisation</b>', 'Keeps the distribution each layer sees stable, so activations do not drift into saturation', 'BatchNorm, LayerNorm, RMSNorm (§3.6, §4.6)']
    ])}
${H.key('Every gradient is a product of local derivatives along a path. ReLU makes the factors 1, residuals add a path of pure 1s, and normalisation stops the inputs drifting into the flat regions. Three fixes, one problem.')}

${H.intuition(`<p>It is worth noticing that these three fixes are not competing solutions to be ranked and chosen among — they attack three different multiplicative culprits, and a modern transformer block genuinely needs all three. ReLU-family activations keep the <i>per-unit gate</i> near 1 whenever a unit is active. Residual connections give the gradient an escape route that bypasses the gates entirely, so even a unit that is currently saturated or switched off does not block the whole network's training. Normalisation keeps the raw numbers flowing into each activation in the range where its gate is actually favourable in the first place, rather than drifting into the saturated tails where even a "good" activation like GELU starts to behave like a bad one.</p>
<p>Remove any one of the three from a genuinely deep network and training degrades measurably, which is the practical proof that they are solving different pieces of the same underlying multiplication.</p>`)}

<h3>Dying ReLU</h3>
<p>ReLU trades the slow, graceful decay of a saturating activation for a sharper failure mode. Its derivative is a clean $1$ on the active side, but exactly $0$ on the inactive side — not small, exactly zero. If a unit's pre-activation $z$ turns out negative for every single example in the training set, that unit's output is $0$ everywhere, its local derivative is $0$ everywhere, and by §3.2.3's propagation step, every gradient passing back through it is multiplied by that $0$ and vanishes on the spot. The unit is not merely slow to learn; it is permanently cut off from the loss, because a gradient of exactly zero at every step never nudges its incoming weights anywhere that would reopen it. This is a <b>dead ReLU</b>, and the four-slider lab in §3.2 already showed you building one deliberately by dragging $z$ negative.</p>
<p>The usual causes are a learning rate large enough that one bad update pushes a unit's weights into a region where it is negative on the whole training distribution, or an initial bias set too far negative to begin with. Leaky ReLU (a small nonzero slope on the negative side) and GELU or SiLU (smooth rather than hard-zero on the negative side) all remove the failure mode by construction, since none of them has a region with a local derivative of exactly zero. In practice, with the initialisation schemes of §3.4 and an adaptive optimiser such as Adam (§3.5), dying units are rarely the binding constraint on a real training run — but the mechanism is worth being able to name precisely, because "my accuracy plateaued and some fraction of units never activate on any input" is a specific, diagnosable symptom with this specific, mechanical cause.</p>

${H.probe([
      ['Why did ReLU matter so much?', 'Its derivative is exactly 1 on the active side, so the depth-wise product of local derivatives stops decaying geometrically the way sigmoid\'s does — §3.3.2 puts the contrast at 40 layers as $1^{40}=1$ against $0.25^{40}\\approx8\\times10^{-25}$.'],
      ['What do residual connections do to gradients?', 'They add a path whose local derivative is exactly 1 regardless of what the wrapped function $F$ is doing, so the gradient reaches early layers unattenuated even when every activation gate along the main path is saturated — the vanish lab in §3.3.2 shows this rescuing an otherwise-collapsing sigmoid network directly.'],
      ['GELU vs ReLU?', 'GELU is a smooth, probabilistically-motivated ReLU that dips slightly negative just left of zero instead of hard-clipping to exactly 0, which avoids the dead-unit failure mode above at a small extra compute cost; it is the default inside transformer feed-forward blocks (§4.5, §4.6).']
    ])}`,
    labs: {
      act: function (host) {
        const st = Viz.controls(host, [
          { k: 'f', label: 'activation', type: 'select', value: 'relu', options: [
            { v: 'sigmoid', t: 'sigmoid' }, { v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' },
            { v: 'leaky', t: 'Leaky ReLU' }, { v: 'gelu', t: 'GELU' }, { v: 'silu', t: 'SiLU / Swish' }] },
          { k: 'cmp', label: 'compare with sigmoid', type: 'toggle', value: true }
        ], () => S.redraw());
        const F = {
          sigmoid: [z => Num.sigmoid(z), z => Num.sigmoid(z) * (1 - Num.sigmoid(z))],
          tanh: [z => Math.tanh(z), z => 1 - Math.tanh(z) ** 2],
          relu: [z => Math.max(0, z), z => z > 0 ? 1 : 0],
          leaky: [z => z > 0 ? z : .1 * z, z => z > 0 ? 1 : .1],
          gelu: [z => z * Num.normCdf(z), z => Num.normCdf(z) + z * Num.normPdf(z)],
          silu: [z => z * Num.sigmoid(z), z => Num.sigmoid(z) * (1 + z * (1 - Num.sigmoid(z)))]
        };
        const out = Viz.readout(host, [
          { k: 'max', label: 'max derivative', cls: 'key' }, { k: 'at0', label: 'derivative at 0' },
          { k: 'sat', label: 'saturates?' }, { k: 'd10', label: 'after 10 layers (max)' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const [f, df] = F[st.f];
            const half = (h - 20) / 2;
            const P1 = Viz.plot(ctx, w, h, { xd: [-5, 5], yd: [-1.2, 2.2], pad: { l: 44, r: 14, t: 12, b: h - half + 6 } })
              .frame({ ylabel: 'φ(z)' });
            P1.clip(() => {
              if (st.cmp) P1.fn(F.sigmoid[0], { color: T.faint, width: 1.4, dash: [5, 4] });
              P1.fn(f, { color: T.blue, width: 2.8 });
            });
            const P2 = Viz.plot(ctx, w, h, { xd: [-5, 5], yd: [0, 1.15], pad: { l: 44, r: 14, t: half + 18, b: 34 } })
              .frame({ xlabel: 'z', ylabel: "φ'(z)" });
            P2.clip(() => {
              if (st.cmp) P2.fn(F.sigmoid[1], { color: T.faint, width: 1.4, dash: [5, 4] });
              P2.fn(df, { color: T.red, width: 2.8 });
              P2.hline(1, { color: T.green, dash: [3, 3], width: 1 });
            });
            const grid = []; for (let z = -6; z <= 6; z += .01) grid.push(df(z));
            const mx = Math.max.apply(null, grid);
            out({
              max: mx.toFixed(3), at0: df(0).toFixed(3),
              sat: (st.f === 'sigmoid' || st.f === 'tanh') ? 'both sides' : (st.f === 'relu' ? 'negative side (dies)' : 'softly'),
              d10: Math.pow(mx, 10).toExponential(2)
            });
          }
        });
        Viz.note(host, 'The green line at 1 in the lower panel is the threshold that matters: a factor above it grows the gradient with depth, below it shrinks. Sigmoid never reaches 0.25 — ten layers of it multiply the gradient by at most one in a million.');
      },

      vanish: function (host) {
        const st = Viz.controls(host, [
          { k: 'depth', label: 'layers', min: 2, max: 40, step: 1, value: 20, fmt: v => v },
          { k: 'act', label: 'activation', type: 'buttons', value: 'sigmoid', options: [{ v: 'sigmoid', t: 'sigmoid' }, { v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' }] },
          { k: 'init', label: 'initialisation scale', min: .3, max: 2.5, step: .05, value: 1, fmt: v => '×' + v.toFixed(2) },
          { k: 'resid', label: 'residual connections', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'g1', label: 'gradient norm at layer 1', cls: 'key' },
          { k: 'gl', label: 'at the last layer' }, { k: 'ratio', label: 'ratio' }, { k: 'verdict', label: 'verdict' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(17), width = 24;
            const act = { sigmoid: Num.sigmoid, tanh: Math.tanh, relu: z => Math.max(0, z) }[st.act];
            const dact = {
              sigmoid: (z, a) => a * (1 - a), tanh: (z, a) => 1 - a * a, relu: z => z > 0 ? 1 : 0
            }[st.act];
            const gain = (st.act === 'relu' ? Math.sqrt(2 / width) : Math.sqrt(1 / width)) * st.init;
            const W = [];
            for (let l = 0; l < st.depth; l++) W.push(Array.from({ length: width }, () => Array.from({ length: width }, () => R.normal(0, gain))));
            // forward
            let a = Array.from({ length: width }, () => R.normal(0, 1));
            const acts = [a], zs = [];
            for (let l = 0; l < st.depth; l++) {
              const z = W[l].map(row => Num.dot(row, a));
              const na = z.map(act);
              a = st.resid ? na.map((v, i) => v + acts[l][i]) : na;
              zs.push(z); acts.push(a);
            }
            // backward
            let delta = Array.from({ length: width }, () => R.normal(0, 1));
            const norms = [];
            for (let l = st.depth - 1; l >= 0; l--) {
              const nd = new Array(width).fill(0);
              for (let k = 0; k < width; k++) {
                let s = 0;
                for (let j = 0; j < width; j++) s += W[l][j][k] * delta[j];
                nd[k] = s * dact(zs[l][k], acts[l + 1][k]);
                if (st.resid) nd[k] += delta[k];
              }
              delta = nd;
              norms.unshift(Math.sqrt(Num.dot(delta, delta)));
            }
            const logs = norms.map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [1, st.depth], yd: [Math.min(-12, Math.min.apply(null, logs) - 1), Math.max(3, Math.max.apply(null, logs) + 1)] })
              .frame({ xlabel: 'layer (1 = closest to the input)', ylabel: 'log₁₀ ‖gradient‖' });
            P.clip(() => {
              P.line(norms.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: T.blue, width: 2.6 });
              P.hline(0, { color: T.faint, dash: [4, 4] });
              P.dots([[1, logs[0]]], { r: 5, color: logs[0] < -6 ? T.red : T.green, stroke: true });
            });
            out({
              g1: norms[0].toExponential(2), gl: norms[norms.length - 1].toExponential(2),
              ratio: (norms[0] / (norms[norms.length - 1] || 1)).toExponential(2),
              verdict: logs[0] < -6 ? 'vanished' : logs[0] > 6 ? 'exploded' : 'healthy'
            });
          }
        });
        Viz.note(host, 'Sigmoid at depth 20 vanishes by construction. Switch to ReLU and it survives; turn on residual connections and even sigmoid survives, because the identity path contributes a factor of exactly 1 at every layer. That is why a 100-layer network is trainable at all.');
      }
    },
    quiz: [
      {
        q: 'The maximum derivative of the sigmoid is 0.25. After 12 sigmoid layers, the gradient is scaled by at most…',
        options: ['0.25', '3', '$0.25^{12} \\approx 6\\times10^{-8}$', '12 × 0.25'],
        answer: 2,
        why: 'Backpropagation multiplies one local activation derivative per layer (§3.2.3\'s propagation step), so the product decays <i>geometrically</i> with depth, not linearly. Option D is the tempting wrong answer because it applies the right intuition — "the effect compounds with the number of layers" — with the wrong operation: twelve factors of at most 0.25 multiply together rather than add, giving $0.25^{12}\\approx5.96\\times10^{-8}$, not $12\\times0.25=3$. That gap between multiplying and adding twelve small numbers is the entire content of the vanishing gradient, and it is why doubling the depth of a saturating network does not double the damage — it squares it.'
      },
      {
        q: 'Residual connections help gradients because…',
        options: ['they reduce the number of parameters', 'they add a path whose local derivative is exactly 1', 'they normalise activations', 'they increase the learning rate'],
        answer: 1,
        why: 'Differentiate $y=x+F(x)$ with respect to $x$ and the chain rule gives $\\partial y/\\partial x = 1 + \\partial F/\\partial x$ — the "1" is there regardless of what $F$ computes or how saturated its internal activations are, because it comes from the identity branch of the sum, not from anything learned. That constant term is a gradient superhighway straight back to early layers: even if $\\partial F/\\partial x$ is nearly zero because every unit inside $F$ is saturated, the total local derivative is still at least 1, not 0. The vanish lab in §3.3.2 shows this directly — flipping residual connections on rescues an otherwise-collapsing 20-layer sigmoid network without changing a single activation.'
      }
    ],
    cards: [
      { q: 'Why ReLU changed things', a: 'Derivative exactly 1 on the active side, so the depth-wise gradient product stops decaying.' },
      { q: 'Three fixes for vanishing gradients', a: 'ReLU-family activations, residual connections (a path of derivative 1), and normalisation to prevent saturation.' },
      { q: 'Dying ReLU', a: 'A unit negative for all inputs receives zero gradient forever; caused by too-large steps or large negative bias.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.4 */
  ML.section({
    id: 'initialisation', track: 'deep', num: '3.4',
    title: 'Initialisation: why the starting scale decides whether training happens',
    lede: 'Get the variance of the initial weights wrong by a factor of two per layer and a forty-layer network is dead before the first step.',
    html: `
<p>Before a network sees a single example, someone has to fill its weight matrices with numbers. The tempting shortcut is "small random numbers" and move on — and for a two- or three-layer network, that shortcut is usually harmless. It is not harmless at forty layers, and the reason is exactly the same multiplicative mechanism §3.3 spent an entire section on, applied to the forward pass instead of the backward one. A layer does not merely apply a function; it also rescales however much signal was already in its input. Do that forty times with even a mild, consistent rescaling and the compounding is the whole story: multiply activation variance by $1.5$ at every layer and forty layers multiply it by $1.5^{40}\\approx10^7$, which overflows or saturates every downstream unit; multiply it by $0.7$ instead and forty layers give $0.7^{40}\\approx10^{-6}$, which is indistinguishable from the network computing zero. Neither number looks dangerous on its own. Raised to the fortieth power, both are.</p>

<h2><span class="sn">3.4.1</span> The requirement, stated precisely</h2>
<p>The goal, then, is to choose the initial weight variance so that a layer neither shrinks nor grows the variance of the signal passing through it — so that $\\mathrm{Var}(\\text{output}) \\approx \\mathrm{Var}(\\text{input})$, layer after layer, at the moment training begins. This is not a heuristic to be tuned by trial; it can be derived directly from how a layer's arithmetic mixes its inputs.</p>

${H.deriv('the LeCun rule: preserving variance through a linear layer', [
      ['$z = \\sum_{j=1}^{n_{\\text{in}}} w_j x_j$', 'One output unit of a layer with $n_{\\text{in}}$ inputs (§0.2). Assume the weights $w_j$ are drawn independently with mean 0 and variance $\\mathrm{Var}(w)$, and are independent of the inputs $x_j$, which themselves have mean 0 and variance $\\mathrm{Var}(x)$.'],
      ['$\\mathrm{Var}(z) = \\sum_{j=1}^{n_{\\text{in}}}\\mathrm{Var}(w_jx_j)$', 'Variance of a sum of <i>independent</i> terms is the sum of their variances — no cross terms survive, because independence makes every covariance zero.'],
      ['$\\mathrm{Var}(w_jx_j) = \\mathrm{Var}(w)\\,\\mathrm{Var}(x)$', 'For independent zero-mean $w$ and $x$: $\\mathrm{Var}(wx) = E[w^2x^2] - (E[wx])^2 = E[w^2]E[x^2] - 0 = \\mathrm{Var}(w)\\mathrm{Var}(x)$, using $E[w^2]=\\mathrm{Var}(w)$ and $E[x^2]=\\mathrm{Var}(x)$ because both means are zero.'],
      ['$\\mathrm{Var}(z) = n_{\\text{in}}\\,\\mathrm{Var}(w)\\,\\mathrm{Var}(x)$', 'Substitute: $n_{\\text{in}}$ identical terms, each contributing $\\mathrm{Var}(w)\\mathrm{Var}(x)$.'],
      ['Set $\\mathrm{Var}(z) = \\mathrm{Var}(x)$: $\\quad\\mathrm{Var}(w) = \\dfrac{1}{n_{\\text{in}}}$', 'Demand that the layer preserve variance rather than shrink or grow it, and solve for the one free quantity, $\\mathrm{Var}(w)$. This is <b>LeCun initialisation</b>: more inputs feeding a unit means each individual weight must be proportionally smaller, so their combined contribution stays the same size.']
    ], 'This derivation assumed nothing about the activation function — it is purely about how a weighted sum of independent variables mixes their variances. It is exact for a linear layer, and a close approximation for tanh or sigmoid near the origin, where both behave almost like the identity. It is not accurate for ReLU, because ReLU is not close to linear near zero: it deletes exactly half of its input\'s variance by construction, and that deletion has to be compensated separately.')}

${H.worked('checking the LeCun rule on four numbers', `<p>Take $n_{\\text{in}}=4$ and $\\mathrm{Var}(x)=1$. LeCun\'s rule gives $\\mathrm{Var}(w)=1/4=0.25$. Check the claim directly: $\\mathrm{Var}(z) = n_{\\text{in}}\\,\\mathrm{Var}(w)\\,\\mathrm{Var}(x) = 4 \\times 0.25 \\times 1 = 1$ — exactly the input variance, preserved. Halve $\\mathrm{Var}(w)$ to $0.125$ by mistake and $\\mathrm{Var}(z)=4\\times0.125\\times1=0.5$: the signal has already lost half its variance after a single layer, and forty such layers would multiply it by $0.5^{40}\\approx9\\times10^{-13}$.</p>`)}

<h2><span class="sn">3.4.2</span> The correction ReLU forces</h2>
<p>ReLU zeroes every negative pre-activation, which is precisely the "deletes half the variance" behaviour the derivation above set aside. Redo the calculation with a ReLU sitting between two linear layers, and the factor it introduces falls straight out of the algebra rather than needing to be guessed.</p>

${H.deriv('the He/Kaiming rule: correcting for ReLU\'s deleted half', [
      ['$a = \\mathrm{ReLU}(z), \\quad z \\text{ symmetric about } 0$', 'Assume the pre-activation $z$ has a distribution symmetric around zero — true for a freshly-initialised network, where $z$ is a sum of many independent zero-mean terms and is approximately Gaussian by the central limit theorem.'],
      ['$E[a^2] = \\int_0^\\infty z^2\\,p(z)\\,dz$', 'ReLU passes positive values through unchanged and maps every negative value to exactly 0, so $a^2$ equals $z^2$ on the positive half-line and 0 on the negative half-line.'],
      ['$= \\tfrac12\\int_{-\\infty}^{\\infty} z^2\\,p(z)\\,dz = \\tfrac12 E[z^2]$', 'By symmetry, the positive half-line carries exactly half of the total probability mass of $z^2$ integrated over all of $z$ — the negative and positive tails contribute equally to $E[z^2]$ since $p(z)=p(-z)$.'],
      ['$E[a^2] = \\tfrac12\\,\\mathrm{Var}(z)$', 'Since $E[z]=0$, $E[z^2]=\\mathrm{Var}(z)$. ReLU keeps exactly half of the incoming variance, on average — the other half was in the discarded negative tail.'],
      ['$\\mathrm{Var}(z_{\\text{next}}) = n_{\\text{in}}\\,\\mathrm{Var}(w)\\,E[a^2] = n_{\\text{in}}\\,\\mathrm{Var}(w)\\cdot\\tfrac12\\,\\mathrm{Var}(z)$', 'Apply the LeCun-style sum-of-independent-terms argument again, this time to the <i>next</i> layer, whose inputs are $a$ rather than $x$: the mean-square of $a$ plays the role $\\mathrm{Var}(x)$ played before.'],
      ['Set $\\mathrm{Var}(z_{\\text{next}}) = \\mathrm{Var}(z)$: $\\quad\\mathrm{Var}(w) = \\dfrac{2}{n_{\\text{in}}}$', 'Demand variance preservation across the ReLU layer and solve. The factor of 2 is not a tuned constant; it is precisely the reciprocal of the one-half that ReLU deletes.']
    ], 'This is He (or Kaiming) initialisation, named for Kaiming He, who with Zhang, Ren and Sun derived exactly this correction in 2015 while training networks far deeper than had previously worked reliably. The whole content of the result is one factor of 2, and that factor has a completely mechanical origin: it exists to put back the variance a ReLU throws away.')}

${H.worked('the same four numbers, now through a ReLU', `<p>Take $n_{\\text{in}}=4$ and $\\mathrm{Var}(z)=4$, so $z\\sim$ something symmetric with that variance. He\'s rule gives $\\mathrm{Var}(w) = 2/4 = 0.5$. First, $E[a^2] = \\tfrac12\\mathrm{Var}(z) = \\tfrac12(4)=2$. Then $\\mathrm{Var}(z_{\\text{next}}) = n_{\\text{in}}\\,\\mathrm{Var}(w)\\,E[a^2] = 4 \\times 0.5 \\times 2 = 4$ — exactly $\\mathrm{Var}(z)$, preserved through the nonlinearity. Use the plain LeCun value $\\mathrm{Var}(w)=1/4=0.25$ instead, forgetting the ReLU correction, and $\\mathrm{Var}(z_{\\text{next}}) = 4\\times0.25\\times2 = 2$ — exactly half of 4, losing a further factor of two at every subsequent layer, which is the collapse the lab below lets you watch happen over forty layers.</p>`)}

<h2><span class="sn">3.4.3</span> Xavier, and why the formula has two subscripts</h2>
<p>The derivations above only protect the forward pass — the signal reaching the output. Backpropagation runs the same layers in reverse (§3.2), and by an identical argument applied to gradients instead of activations, preserving <i>gradient</i> variance on the way back requires $\\mathrm{Var}(w) = 1/n_{\\text{out}}$, with the layer's fan-out $n_{\\text{out}}$ playing the role $n_{\\text{in}}$ played going forward, because the backward pass multiplies by $W^\\mathsf{T}$ rather than $W$ (§0.7.6). In general $n_{\\text{in}} \\ne n_{\\text{out}}$, so no single variance satisfies both requirements exactly. Xavier Glorot and Yoshua Bengio's 2010 compromise, aimed at the symmetric, near-linear activations of their time (tanh, sigmoid), was to average the two demands: $\\mathrm{Var}(w) = 2/(n_{\\text{in}}+n_{\\text{out}})$, which reduces to LeCun's plain $1/n_{\\text{in}}$ whenever a layer happens to be square, $n_{\\text{in}}=n_{\\text{out}}$.</p>

${H.table(['Scheme', 'Variance', 'For'], [
      ['<b>Xavier / Glorot</b>', '$2/(n_{in}+n_{out})$', 'Symmetric activations (tanh, sigmoid)'],
      ['<b>He / Kaiming</b>', '$2/n_{in}$', 'ReLU — the factor 2 compensates for zeroing half the inputs'],
      ['<b>LeCun</b>', '$1/n_{in}$', 'SELU and linear layers'],
      ['<b>Orthogonal</b>', 'orthogonal matrix × gain', 'RNNs, where repeated multiplication makes spectral radius critical'],
      ['<b>Zeros</b>', '—', 'Biases only. Zero weights make every unit identical and the network cannot break symmetry.']
    ])}

<p><b>What you are looking at.</b> The lab below builds a real 40-layer network — every layer square, so $n_{\\text{in}}=n_{\\text{out}}=$ the width you choose — fills it with random weights at the variance your chosen scheme prescribes, and pushes one batch of random unit-variance inputs through all forty layers, plotting $\\log_{10}$ of the activation variance at every depth. The green dashed line marks the target: variance exactly 1, meaning the signal neither grew nor shrank.</p>

<p><b>What to do with it.</b> Start on ReLU with the He scheme and watch the blue line hug the green target line all the way to layer 40. Switch the scheme to <i>Xavier (1/n_in)</i> while staying on ReLU — this is exactly the "forgot the factor of 2" mistake the worked example above computed by hand — and watch the line fall steadily away from the target, losing very close to a factor of 2 at every single layer, until the readout reports a collapsed verdict and the top-layer variance is unmeasurably small. Then switch the activation to tanh with the same Xavier scheme: now the line tracks the target closely, because tanh near the origin behaves almost linearly and does not delete half its input's variance the way ReLU does, so the plain LeCun-style formula this lab labels "Xavier" is the correct one here.</p>

<p><b>The thing genuinely worth noticing.</b> Try the <i>too big</i> preset on ReLU and watch the <i>dead units at the top</i> readout climb into double digits, even though the variance line initially looks like it is merely exploding rather than collapsing. Large initial weights push many units' pre-activations deep into either the positive or the negative extreme; the ones that land negative are ReLU-dead from the very first forward pass, before a single gradient has been computed (§3.3.3). Initialisation does not only set the starting loss — it can silently decide, before training even begins, what fraction of the network's units will ever be able to learn at all.</p>

${H.lab('init', 'Activation variance across depth', 'A real forward pass through 40 layers at the initialisation scale you choose. Watch the variance track 1.0 with the correct scheme and explode or collapse when it is wrong — the y-axis is logarithmic for a reason.')}

<h2><span class="sn">3.4.4</span> Why the weights have to be random at all</h2>
<p>Everything above chose a variance; none of it explains why the weights need to be random rather than all fixed at some single non-zero value that happens to have the right variance in aggregate — say, every weight in a layer set to exactly $\\sqrt{1/n_{\\text{in}}}$. The reason is a symmetry argument with nothing to do with variance. If every incoming weight to two different units in the same layer is identical, those two units compute the identical function of the input, receive the identical gradient during backpropagation (§3.2.3), and are updated identically forever — no number of training steps ever makes them different, because the update rule treats identical inputs identically. The layer, however wide it is drawn, behaves as if it had exactly one unit. Randomness is not a heuristic nicety here; it is the only thing that lets units specialise into different features at all. Biases, by contrast, can safely start at exactly zero, because the incoming weights already differ from unit to unit, which is enough to break the symmetry on its own.</p>

<h2><span class="sn">3.4.5</span> At scale</h2>
<p>Modern large models add two refinements on top of the He/Xavier machinery above. Residual branches are often initialised near zero, or scaled down by $1/\\sqrt{2L}$ for a network of $L$ residual blocks, so that at step zero the network is close to the identity function and adding depth costs nothing before training has had a chance to use it — echoing §3.3.3's observation that a residual path already contributes a derivative of exactly 1 regardless of what its wrapped function computes. And <b>μP</b> (maximal update parameterisation, developed further in §4.11) rescales both the initialisation variance and the learning rate as a function of width, using variance-propagation arguments of exactly this kind extended to cover the optimiser's update itself, so that hyperparameters tuned on a small, cheap proxy model transfer directly to a much larger run. That turns hyperparameter search on a training run costing millions of dollars into a search on something that costs very little, which is the practical payoff of taking the algebra in this section seriously rather than treating "small random numbers" as good enough.</p>

${H.probe([
      ['Why He rather than Xavier for ReLU?', 'ReLU zeroes roughly half of a symmetric pre-activation\'s mass, which the derivation in §3.4.2 shows keeps exactly half the incoming variance on average, $E[a^2]=\\tfrac12\\mathrm{Var}(z)$; doubling the weight variance to $2/n_{in}$ exactly compensates.'],
      ['What happens if you initialise all weights to zero?', 'Every unit in a layer computes the identical function of the input, receives the identical gradient during backpropagation, and is updated identically forever — the layer behaves as if it had one unit no matter how wide it is drawn, because nothing in the training dynamics can break the symmetry once it starts perfectly symmetric.'],
      ['What does μP buy you?', 'Hyperparameters found on a small, cheap proxy model transfer to a large one, because initialisation scale and learning rate are both rescaled as a function of width using the same variance-propagation logic derived here, extended to the optimiser update.']
    ])}`,
    labs: {
      init: function (host) {
        const st = Viz.controls(host, [
          { k: 'scheme', label: 'initialisation', type: 'select', value: 'he', options: [
            { v: 'he', t: 'He (2/n_in)' }, { v: 'xavier', t: 'Xavier (1/n_in)' }, { v: 'small', t: 'too small (0.1/n_in)' }, { v: 'big', t: 'too big (6/n_in)' }] },
          { k: 'act', label: 'activation', type: 'buttons', value: 'relu', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }, { v: 'linear', t: 'linear' }] },
          { k: 'depth', label: 'depth', min: 5, max: 60, step: 1, value: 40, fmt: v => v },
          { k: 'width', label: 'width', min: 16, max: 256, step: 16, value: 128, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'v1', label: 'variance at layer 1', cls: 'key' }, { k: 'vl', label: 'at the last layer' },
          { k: 'dead', label: 'dead units at the top' }, { k: 'verdict', label: 'verdict' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(37);
            const n = st.width;
            const varW = { he: 2 / n, xavier: 1 / n, small: .1 / n, big: 6 / n }[st.scheme];
            const act = { relu: z => Math.max(0, z), tanh: Math.tanh, linear: z => z }[st.act];
            let a = Array.from({ length: n }, () => R.normal(0, 1));
            const vars = [Num.variance(a)];
            let deadFrac = 0;
            for (let l = 0; l < st.depth; l++) {
              const z = [];
              for (let i = 0; i < n; i++) {
                let s = 0;
                for (let j = 0; j < n; j++) s += R.normal(0, Math.sqrt(varW)) * a[j];
                z.push(s);
              }
              a = z.map(act);
              vars.push(Num.variance(a));
              if (l === st.depth - 1) deadFrac = a.filter(v => Math.abs(v) < 1e-9).length / n;
            }
            const logs = vars.map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [0, st.depth], yd: [Math.min(-14, Math.min.apply(null, logs) - 1), Math.max(6, Math.max.apply(null, logs) + 1)] })
              .frame({ xlabel: 'layer', ylabel: 'log₁₀ variance of activations' });
            P.clip(() => {
              P.line(vars.map((v, i) => [i, Math.log10(Math.max(1e-30, v))]), { color: T.blue, width: 2.6 });
              P.hline(0, { color: T.green, dash: [4, 4], label: 'variance = 1, the target' });
            });
            const last = vars[vars.length - 1];
            out({
              v1: vars[1].toExponential(2), vl: last.toExponential(2),
              dead: (deadFrac * 100).toFixed(0) + '%',
              verdict: last < 1e-6 ? 'collapsed — no signal reaches the output' : last > 1e6 ? 'exploded' : 'stable'
            });
          }
        });
        Viz.note(host, 'Select ReLU with Xavier (1/n) instead of He (2/n) and watch the variance halve every layer — a factor of $2^{-40}$ by the top. That single factor of two is the whole content of the He initialisation paper.');
      }
    },
    quiz: [
      {
        q: 'He initialisation uses variance $2/n_{in}$ rather than $1/n_{in}$ because…',
        options: ['ReLU is faster', 'ReLU zeroes roughly half the inputs, halving the variance; the 2 compensates', 'it prevents overfitting', 'gradients are twice as large'],
        answer: 1,
        why: 'The derivation in §3.4.2 shows exactly where the factor comes from: for a symmetric pre-activation $z$, ReLU keeps only the positive half, so $E[\\mathrm{ReLU}(z)^2] = \\tfrac12\\mathrm{Var}(z)$ — half the variance is thrown away at every layer, on average. Doubling $\\mathrm{Var}(w)$ from $1/n_{in}$ to $2/n_{in}$ exactly cancels that one-half factor, so a layer\'s output variance matches its input variance again. Option D sounds plausible only because "the fix involves a 2" — but the mechanism is compensating for a deletion, not doubling anything that was already there.'
      },
      {
        q: 'Initialising every weight to the same non-zero constant means…',
        options: ['faster convergence', 'all units in a layer stay identical forever — symmetry is never broken', 'exploding gradients', 'nothing, as long as biases are random'],
        answer: 1,
        why: 'Two units fed by identical incoming weights compute the identical function of the input, so they produce identical outputs, receive identical gradients during backpropagation (§3.2.3), and are updated identically at every subsequent step — nothing in ordinary gradient descent can break a symmetry that starts perfectly intact. Option D is the tempting wrong answer because it is true that biases are usually safe at exactly zero, but the question asks about the weights, and randomising only the biases while every weight into a unit is identical still leaves those units seeing the identical bias-shifted input and computing the identical function.'
      }
    ],
    cards: [
      { q: 'He vs Xavier', a: 'He $2/n_{in}$ for ReLU (compensates for zeroing half the inputs); Xavier $2/(n_{in}+n_{out})$ for symmetric activations.' },
      { q: 'Why not initialise to zero?', a: 'Symmetry: all units compute the same function and receive the same gradient forever.' },
      { q: 'μP', a: 'Width-aware parameterisation so hyperparameters transfer from a small proxy model to a large run.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.5 */
  ML.section({
    id: 'optimisers', track: 'deep', num: '3.5',
    title: 'Optimisers: SGD, momentum, Adam, AdamW, schedules',
    lede: 'Everything here is a response to the conditioning problem in §1.9 — and AdamW versus Adam is the most-asked one-line distinction in the field.',
    rests: 'Builds directly on §0.3.4\'s elongated bowl and the boxed rule that the largest stable learning rate is 2 divided by curvature.',
    html: `
<p>§0.3.4 built one small, elongated bowl by hand — $f(w_1, w_2) = \\tfrac12(w_1^2 + 100w_2^2)$, curvature 1 along $w_1$ and 100 along $w_2$ — and showed that plain gradient descent, forced to use a learning rate small enough for the steep direction, needs about 240 steps to make the same progress a well-conditioned bowl would make in one. That example is not a curiosity. It is the shape of almost every real loss surface, because real features rarely arrive on matched scales (§0.2), and it is worth re-opening that exact bowl here to watch the specific failure mode plain gradient descent produces, before fixing it.</p>

<h2><span class="sn">3.5.1</span> The zig-zag, made concrete</h2>

<p>Take that same bowl and the same learning rate, $\\eta = 0.019$, chosen deliberately close to the stability limit $2/100 = 0.02$ for the steep direction. Gradient descent's update multiplies $w_1$ by $1-\\eta = 0.981$ every step — small, steady progress, same sign throughout — and multiplies $w_2$ by $1-100\\eta = 1-1.9 = -0.9$ every step. That second multiplier is negative. Starting from $w_2=1$, the sequence runs $1,\\, -0.9,\\, 0.81,\\, -0.729,\\, 0.6561,\\ldots$ — the sign of $w_2$ flips at <i>every single step</i>, while its magnitude shrinks by 10% each time. Plotted as a path in the $(w_1,w_2)$ plane, that sign-flipping is exactly the zig-zag: the optimiser bounces back and forth across the narrow direction of the valley while creeping, almost imperceptibly, along its long axis.</p>

<p>The gradient itself flips sign in lockstep with $w_2$, since $\\partial f/\\partial w_2 = 100w_2$ is just a positive multiple of $w_2$. That single fact — <i>consecutive gradients disagree in sign along the steep direction, and agree in sign along the shallow one</i> — is the entire structural weakness plain gradient descent has no way to notice. It treats every step in isolation, reacting only to the current gradient, and has no memory of the fact that it was just told to go the opposite way a moment ago.</p>

<h2><span class="sn">3.5.2</span> Momentum: giving the optimiser a memory</h2>

<p>The fix is to stop reacting to the raw gradient and instead accumulate a running average of it, then step in the direction of the average. Formally, introduce a <b>velocity</b> $v$ and update it before updating the parameters:</p>
$$v \\leftarrow \\beta v - \\eta g, \\qquad \\theta \\leftarrow \\theta + v$$
<p>with $\\beta$ (a number just below 1, typically $0.9$) controlling how much of the previous velocity survives each step. This single change has opposite effects on the two directions of the ravine above, and both effects can be derived exactly rather than asserted.</p>

${H.deriv('momentum under a gradient that keeps the same sign forever', [
      ['$v_t = \\beta v_{t-1} - \\eta g, \\quad g \\text{ constant}$', 'Model the shallow direction of the ravine: the gradient $g$ keeps the same sign and roughly the same size at every step, as $w_1$\'s update above did.'],
      ['$v_t = -\\eta g\\,(1 + \\beta + \\beta^2 + \\cdots + \\beta^{t-1})$', 'Unroll the recurrence from $v_0=0$: each past gradient survives into the present velocity, discounted by a power of $\\beta$.'],
      ['$v_t \\to -\\dfrac{\\eta g}{1-\\beta} \\quad \\text{as } t\\to\\infty$', 'The bracket is a geometric series (§0.3\'s tools extend directly here), summing to $1/(1-\\beta)$ as the number of terms grows.']
    ], 'Compare this steady-state velocity to plain gradient descent\'s step, which is always exactly $-\\eta g$. Momentum\'s eventual step is $1/(1-\\beta)$ times larger. At the standard $\\beta = 0.9$, that factor is $1/0.1 = 10$: momentum lets the shallow, consistently-signed direction move ten times faster than plain gradient descent would allow it to, using the very same learning rate.')}

${H.deriv('momentum under a gradient that flips sign every single step', [
      ['$v_t = \\beta v_{t-1} - \\eta g_t, \\quad g_t = -g\\,(-1)^t$', 'Model the steep direction: the gradient has constant magnitude $g$ but alternates in sign at every step, exactly as $w_2$\'s gradient did above.'],
      ['guess a periodic solution $v_t = A(-1)^t$', 'Since the forcing term is itself perfectly periodic with period 2, look for a velocity that settles into the same rhythm rather than unrolling the whole sum.'],
      ['$A(-1)^t = \\beta A(-1)^{t-1} + \\eta g(-1)^t = -\\beta A(-1)^t + \\eta g(-1)^t$', 'Substitute the guess into the recurrence and use $(-1)^{t-1} = -(-1)^t$ to collect every term over the same power of $(-1)^t$.'],
      ['$A(1+\\beta) = \\eta g \\quad\\Rightarrow\\quad A = \\dfrac{\\eta g}{1+\\beta}$', 'Divide out $(-1)^t$ from both sides and solve for the one unknown amplitude $A$.']
    ], 'The steady oscillation amplitude is $\\eta g/(1+\\beta)$, smaller than plain gradient descent\'s raw step $\\eta g$ by a factor $1/(1+\\beta)$. At $\\beta=0.9$ that factor is $1/1.9 \\approx 0.53$: momentum roughly <i>halves</i> the amplitude of the back-and-forth bouncing in the steep direction, because two gradients of opposite sign, averaged together with the recent past, largely cancel.')}

<p>Put the two results side by side and the mechanism is exactly what a "memory" of past gradients should do: the same $\\beta=0.9$ multiplies the shallow direction's progress by 10 and divides the steep direction's oscillation by roughly 2, using nothing but the sign pattern each direction happened to produce. Nobody told momentum which direction was steep and which was shallow — it discovered the distinction purely from whether consecutive gradients agreed or disagreed, which is visible in the raw numbers without knowing anything about the Hessian.</p>

${H.analogy(`<p>A bobsled on a long, straight, gently-sloped run accelerates the whole way down, because every push it feels points the same direction as the last. The same bobsled entering a narrow, twisting canyon does not instantly reverse its velocity every time the wall nudges it sideways — its own momentum smooths the nudge into a gentler curve, because inertia averages a sideways force against whatever direction the sled was already carrying. A gradient step with no memory is a sled with no mass: every gradient, however contradictory, is obeyed instantly and completely. Adding velocity gives the optimiser exactly the inertia that turns instant obedience into an average.</p>`)}

<h2><span class="sn">3.5.3</span> RMSProp: scaling each direction by its own size</h2>

<p>Momentum fixes the zig-zag using only the <i>sign</i> pattern of the gradient. A second, independent idea fixes it using the gradient's <i>magnitude</i> instead: keep a running average of each parameter's squared gradient, $s \\leftarrow \\beta_2 s + (1-\\beta_2)g^2$, and divide that parameter's step by $\\sqrt{s}$. A direction with a habitually large gradient — the steep $w_2$ above, where $g=100w_2$ dwarfs $g=w_1$ — gets divided down; a direction with a habitually small gradient gets divided down far less, so its relative step grows. This is a per-coordinate rescaling toward equal step sizes in every direction, which is precisely what dividing by curvature would do exactly if you could afford to compute the Hessian (§0.3.4's aside on Newton's method makes that connection explicit): RMSProp is a way of buying part of that benefit using only quantities gradient descent was already computing, at first-order cost.</p>

<h2><span class="sn">3.5.4</span> Adam: both fixes at once, with one correction</h2>

<p><b>Adam</b> runs momentum and RMSProp simultaneously, using an exponential moving average for each rather than the raw accumulator above:</p>
$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t, \\quad s_t = \\beta_2 s_{t-1} + (1-\\beta_2)g_t^2$$
$$\\hat m_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat s_t = \\frac{s_t}{1-\\beta_2^t}, \\quad \\theta \\leftarrow \\theta - \\eta\\frac{\\hat m_t}{\\sqrt{\\hat s_t}+\\epsilon}$$
<p>The $\\hat{\\phantom{m}}$ hats mark <b>bias correction</b>, and the need for it is not a minor footnote — without it, Adam's very first few steps are systematically too small, and it is worth seeing exactly how much.</p>

${H.deriv('why $m_t$ starts out biased toward zero, and why dividing by $1-\\beta_1^t$ fixes it exactly', [
      ['$m_t = (1-\\beta_1)\\sum_{i=1}^{t}\\beta_1^{t-i}g_i, \\quad m_0=0$', 'Unroll the exponential moving average from a zero start — every past gradient contributes, weighted by how recently it arrived.'],
      ['assume $g_i = g$ for all $i$ (the calibration case)', 'To isolate the bias, ask what $m_t$ reports if the true gradient has been perfectly constant — the honest answer should be exactly $g$.'],
      ['$m_t = (1-\\beta_1)g\\sum_{i=1}^{t}\\beta_1^{t-i} = (1-\\beta_1)g\\cdot\\dfrac{1-\\beta_1^t}{1-\\beta_1}$', 'The sum is a finite geometric series with $t$ terms and ratio $\\beta_1$, summing to $(1-\\beta_1^t)/(1-\\beta_1)$.'],
      ['$m_t = g\\,(1-\\beta_1^t)$', 'The $(1-\\beta_1)$ factors cancel. Even though the true gradient is exactly $g$ at every step, the raw moving average $m_t$ reports only a fraction $(1-\\beta_1^t)$ of it.'],
      ['$\\hat m_t = m_t / (1-\\beta_1^t) = g$', 'Divide by exactly the missing factor, and the bias vanishes completely — not approximately, exactly, for this calibration case.']
    ], 'At $\\beta_1=0.9$ and $t=1$, the uncorrected estimate is $m_1=(1-0.9)g=0.1g$ — the very first step would move barely a tenth as far as the true gradient warrants, before recovering over the next several steps as $\\beta_1^t$ decays. The bias is worst exactly when it matters most: at the start of training, when the running averages have seen the fewest examples and are most likely to be wrong in whatever direction the first few gradients happened to point.')}

${H.table(['Optimiser', 'Update', 'What it fixes'], [
      ['<b>SGD</b>', '$\\theta \\leftarrow \\theta - \\eta g$', 'Nothing; the baseline'],
      ['<b>Momentum</b>', '$v \\leftarrow \\beta v - \\eta g;\\; \\theta \\leftarrow \\theta + v$', 'Accumulates the consistent direction, cancels oscillation across ravines'],
      ['<b>RMSProp</b>', '$\\theta \\leftarrow \\theta - \\eta g/\\sqrt{\\hat s}$', 'Per-coordinate scaling — an approximation to dividing by curvature'],
      ['<b>Adam</b>', 'momentum + RMSProp with bias correction', 'Both at once; the workhorse'],
      ['<b>AdamW</b>', 'Adam with decoupled weight decay', 'Makes weight decay actually decay (see below)'],
      ['<b>Muon / Shampoo / second-order</b>', 'preconditioned updates', 'Better conditioning still, at higher cost per step; increasingly used at scale']
    ])}

<h2><span class="sn">3.5.5</span> Adam versus AdamW — the answer to have ready</h2>
<p>In plain Adam, L2 regularisation added to the loss becomes part of $g$, so it gets divided by the same adaptive denominator $\\sqrt{\\hat s}$ as everything else. A weight with a large history of gradients — which is exactly the weight the regulariser most wants to shrink — gets its penalty divided down along with its gradient, and ends up decayed <i>less</i> than a weight with small, quiet gradients. That is the opposite of what "pull every weight toward zero, proportionally" was supposed to mean, and it stops behaving like weight decay at all. <b>AdamW decouples it</b>: the decay is applied directly to the weights, outside the adaptive step, so it never passes through the $\\sqrt{\\hat s}$ denominator in the first place.</p>
$$\\theta \\leftarrow \\theta - \\eta\\left(\\frac{\\hat m}{\\sqrt{\\hat s}+\\epsilon} + \\lambda\\theta\\right)$$
${H.key('AdamW decouples weight decay from the adaptive denominator, so decay actually decays. It generalises better and it is the modern default.')}

<p><b>What you are looking at.</b> The lab below runs four real optimisers — SGD, momentum, Adam and AdamW — step by step on one of three surfaces, starting from the same point, drawn as trajectories over contour lines of the loss. The quadratic ravine is exactly the elongated bowl this section has been computing by hand; the Rosenbrock surface is a curved valley with a much harder shape than any straight-line example can show; the saddle has a flat region an optimiser can stall in before the surface drops away. Contours closer together mean the surface is steeper there.</p>

<p><b>What to do with it.</b> Start on the quadratic ravine with the condition number control turned up past 20 and the learning rate left at its default. Watch the faint SGD path visibly zig-zag across the narrow axis exactly as the hand-computed $w_2$ sequence did above, while the amber momentum path cuts a straighter line by damping that same oscillation. Then switch to the saddle surface: SGD can stall for many steps near the centre, where the gradient in the escape direction is tiny, while momentum, having accumulated velocity on the way in, carries through the flat region under its own inertia.</p>

<p><b>The thing genuinely worth noticing.</b> Push the learning rate control well past its default on any surface and watch every single trajectory — including Adam's — blow up or refuse to converge. No optimiser in this lab, or in practice, rescues a step size above the stability limit derived in §0.3.4; momentum and adaptive scaling change how gracefully an optimiser handles an <i>awkwardly shaped</i> surface, not whether it can survive a learning rate that is simply too large for the steepest direction present.</p>

${H.lab('opt', 'Four optimisers on the same surface', 'Real trajectories, computed step by step. Raise the condition number and watch SGD zig-zag, momentum smooth it out, and Adam march almost straight down the valley. Then use the ill-conditioned + high-learning-rate combination to make each one fail.')}

<h2><span class="sn">3.5.6</span> Learning-rate schedules</h2>
<p>Everything above treats $\\eta$ as fixed, but the best $\\eta$ genuinely changes over the course of training. Early on, Adam's second-moment estimate $\\hat s$ has seen almost no gradients and is a poor estimate of anything, which makes the adaptive denominator $\\sqrt{\\hat s}$ unreliable in exactly the way the bias-correction derivation above quantified — an early step can be far too large, in a direction the optimiser has not yet earned any confidence in, and land the model somewhere later training never fully recovers from. <b>Warmup</b> — ramping $\\eta$ up linearly over the first few hundred to few thousand steps — simply refuses to take that gamble, keeping steps small until the running averages have gathered enough evidence to be trusted.</p>
<p>Once warmup ends, the rate still should not stay constant for the rest of training: a rate that was safe early, when the loss surface was far from any minimum, is often larger than ideal once the model is close to one. <b>Cosine decay</b> is the classic answer, smoothly reducing $\\eta$ along a cosine curve down to (near) zero by a predetermined final step. <b>WSD</b> (warmup–stable–decay) instead holds the rate flat for most of training and decays sharply only at the very end, which reaches essentially the same final loss as cosine while not requiring you to commit to a total step count in advance — you can checkpoint at the end of the stable phase and run several different short decays from that one trunk, for instance to compare different final data mixtures (§4.9, §4.11).</p>

<p><b>What you are looking at.</b> The lab below plots learning rate against training step for four schedules on the same axes: constant (faint dashed), step decay (amber), cosine (blue) and WSD (green). The red vertical line marks where warmup ends and the peak rate is reached; the green dashed vertical line marks where WSD's decay phase begins.</p>

<p><b>What to do with it.</b> Drag the <i>WSD stable fraction</i> control down and watch the green decay line start earlier and stretch out longer, while the peak plateau shortens — you are directly controlling how much of the training budget is spent at full speed versus winding down. Compare the areas under the cosine and WSD curves at a similar stable fraction: they are not identical curves, but both spend the great majority of the run near the peak rate and both end near zero, which is the shared property that lets them reach comparable final losses.</p>

<p><b>The thing genuinely worth noticing.</b> Cosine's curve is entirely determined the moment you fix the total step count — change your mind about how long to train, and the whole curve from very early on has to be redrawn to still land at zero by the new end point. WSD's stable phase does not know or care when the run will end; the decay is a short, local decision made only once you decide to stop. That difference in <i>when the schedule has to be committed</i>, not the shape of the curve itself, is why WSD has become the default for very large pretraining runs where the total budget is often revised mid-flight.</p>

${H.lab('sched', 'Schedules, drawn', 'Compare cosine, WSD, step decay and constant. The annotation marks where the "anneal" phase begins — the part of training the model is most sensitive to what it is shown.')}

${H.probe([
      ['Adam vs AdamW in one line?', 'AdamW decouples weight decay from the adaptive denominator, so decay actually decays regardless of a weight\'s gradient history; it generalises better and is the default.'],
      ['Why warmup?', 'Adam\'s second-moment estimate is provably biased toward zero in its first steps (§3.5.4\'s derivation), so early adaptive step sizes can be unreliable; a large step then can move the model somewhere it never recovers from.'],
      ['Why does momentum help a zig-zagging ravine specifically?', 'Momentum accumulates a running average of the gradient. Where consecutive gradients disagree in sign — the steep, oscillating direction — the average partially cancels, damping the bounce by a factor of roughly $1/(1+\\beta)$. Where they agree — the shallow, consistent direction — the average compounds, amplifying progress by roughly $1/(1-\\beta)$, both derived exactly in §3.5.2.'],
      ['When would you still use plain SGD with momentum?', 'Vision models trained long with heavy augmentation often generalise slightly better with SGD+momentum than with Adam.']
    ])}`,
    labs: {
      opt: function (host) {
        const st = Viz.controls(host, [
          { k: 'kappa', label: 'condition number κ', min: 1, max: 60, step: 1, value: 20, fmt: v => v },
          { k: 'lr', label: 'learning rate', min: -3, max: -.4, step: .05, value: -1.4, fmt: v => Math.pow(10, v).toFixed(3) },
          { k: 'surface', label: 'surface', type: 'buttons', value: 'quad', options: [{ v: 'quad', t: 'quadratic ravine' }, { v: 'rosen', t: 'Rosenbrock' }, { v: 'saddle', t: 'saddle' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sgd', label: 'SGD final loss' }, { k: 'mom', label: 'momentum' },
          { k: 'adam', label: 'Adam', cls: 'good' }, { k: 'adamw', label: 'AdamW (λ=0.05)' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const k = st.kappa;
            const surf = {
              quad: { f: (x, y) => .5 * (x * x + k * y * y), g: (x, y) => [x, k * y], start: [-2.3, 1.1], xd: [-2.6, 2.6], yd: [-1.5, 1.5] },
              rosen: { f: (x, y) => Math.pow(1 - x, 2) + 10 * Math.pow(y - x * x, 2), g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)], start: [-1.4, 1.6], xd: [-2, 2], yd: [-.6, 2.4] },
              saddle: { f: (x, y) => x * x - .6 * y * y + .12 * y * y * y * y, g: (x, y) => [2 * x, -1.2 * y + .48 * y * y * y], start: [-1.8, .05], xd: [-2.4, 2.4], yd: [-2, 2] }
            }[st.surface];
            const lr = Math.pow(10, st.lr);
            function run(kind) {
              let x = surf.start[0], y = surf.start[1], vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0;
              const path = [[x, y]];
              for (let t = 1; t <= 300; t++) {
                const [gx, gy] = surf.g(x, y);
                if (kind === 'sgd') { x -= lr * gx; y -= lr * gy; }
                else if (kind === 'mom') { vx = .9 * vx - lr * gx; vy = .9 * vy - lr * gy; x += vx; y += vy; }
                else {
                  mx = .9 * mx + .1 * gx; my = .9 * my + .1 * gy;
                  sx = .999 * sx + .001 * gx * gx; sy = .999 * sy + .001 * gy * gy;
                  const mhx = mx / (1 - Math.pow(.9, t)), mhy = my / (1 - Math.pow(.9, t));
                  const shx = sx / (1 - Math.pow(.999, t)), shy = sy / (1 - Math.pow(.999, t));
                  const step = lr * 3;
                  x -= step * (mhx / (Math.sqrt(shx) + 1e-8) + (kind === 'adamw' ? .05 * x : 0));
                  y -= step * (mhy / (Math.sqrt(shy) + 1e-8) + (kind === 'adamw' ? .05 * y : 0));
                }
                if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 20 || Math.abs(y) > 20) break;
                path.push([x, y]);
              }
              return path;
            }
            const P = Viz.plot(ctx, w, h, { xd: surf.xd, yd: surf.yd })
              .frame({ xlabel: 'θ₁', ylabel: 'θ₂' });
            const base = surf.f(surf.start[0], surf.start[1]);
            P.contours(surf.f, [base * .01, base * .05, base * .15, base * .35, base * .6, base, base * 1.6], { color: T.faint, alpha: .5 });
            const runs = { sgd: run('sgd'), mom: run('mom'), adam: run('adam'), adamw: run('adamw') };
            const cols = { sgd: T.faint, mom: T.amber, adam: T.blue, adamw: T.green };
            P.clip(() => {
              Object.keys(runs).forEach(kk => P.line(runs[kk], { color: cols[kk], width: kk === 'adam' ? 2.6 : 1.8 }));
              P.dots([surf.start], { r: 5, color: T.text, stroke: true });
            });
            const fin = kk => { const p = runs[kk][runs[kk].length - 1]; return surf.f(p[0], p[1]); };
            out({
              sgd: fin('sgd').toExponential(1), mom: fin('mom').toExponential(1),
              adam: fin('adam').toExponential(1), adamw: fin('adamw').toExponential(1)
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().faint, t: 'SGD' }, { c: Viz.theme().amber, t: 'momentum' },
          { c: Viz.theme().blue, t: 'Adam' }, { c: Viz.theme().green, t: 'AdamW' }
        ]);
        Viz.note(host, 'On the saddle surface, plain SGD stalls near the origin where the gradient is tiny in the escape direction; momentum carries through. On the ravine, Adam’s per-coordinate scaling is the visible win. Push the learning rate up and every method fails — no optimiser rescues a step size above the stability limit (§1.9).');
      },

      sched: function (host) {
        const st = Viz.controls(host, [
          { k: 'total', label: 'total steps', min: 1000, max: 100000, step: 1000, value: 20000, fmt: v => (v / 1000) + 'k' },
          { k: 'warm', label: 'warmup steps', min: 0, max: 5000, step: 100, value: 1000, fmt: v => v },
          { k: 'peak', label: 'peak LR', min: -4.3, max: -2, step: .1, value: -3.5, fmt: v => Math.pow(10, v).toExponential(1) },
          { k: 'stable', label: 'WSD stable fraction', min: .3, max: .95, step: .05, value: .8, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const peak = Math.pow(10, st.peak);
            const warm = t => Math.min(1, t / Math.max(1, st.warm));
            const cosine = t => peak * warm(t) * (t <= st.warm ? 1 : .5 * (1 + Math.cos(Math.PI * (t - st.warm) / (st.total - st.warm))));
            const wsd = t => {
              const decayStart = st.warm + (st.total - st.warm) * st.stable;
              if (t <= st.warm) return peak * warm(t);
              if (t <= decayStart) return peak;
              return peak * Math.max(0, 1 - (t - decayStart) / (st.total - decayStart));
            };
            const step = t => peak * warm(t) * Math.pow(.1, Math.floor(3 * t / st.total));
            const P = Viz.plot(ctx, w, h, { xd: [0, st.total], yd: [0, peak * 1.15] })
              .frame({ xlabel: 'training step', ylabel: 'learning rate', yfmt: v => v.toExponential(0), xfmt: v => (v / 1000).toFixed(0) + 'k' });
            P.clip(() => {
              P.fn(t => peak, { color: T.faint, width: 1.4, dash: [4, 4] });
              P.fn(step, { color: T.amber, width: 1.8, n: 400 });
              P.fn(cosine, { color: T.blue, width: 2.6, n: 400 });
              P.fn(wsd, { color: T.green, width: 2.6, n: 400 });
              P.vline(st.warm, { color: T.red, label: 'warmup ends' });
              const ds = st.warm + (st.total - st.warm) * st.stable;
              P.vline(ds, { color: T.green, dash: [3, 3], label: 'anneal begins' });
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().blue, t: 'cosine' }, { c: Viz.theme().green, t: 'WSD (warmup–stable–decay)' },
          { c: Viz.theme().amber, t: 'step decay' }, { c: Viz.theme().faint, t: 'constant' }
        ]);
        Viz.note(host, 'Cosine requires committing to a total step count in advance — awkward when you might extend the run. WSD does not: you can stop the stable phase whenever you like, decay from a checkpoint, and branch several short decays from one trunk. That property is why it has become common for large pretraining runs.');
      }
    },
    quiz: [
      {
        q: 'In plain Adam, adding L2 to the loss behaves badly because…',
        options: ['it is applied twice', 'the penalty gradient is divided by the same adaptive denominator, so high-gradient weights decay less', 'it conflicts with momentum', 'it makes the loss non-convex'],
        answer: 1,
        why: 'An L2 penalty added to the loss becomes part of the gradient $g$ that Adam feeds through $\\hat m/(\\sqrt{\\hat s}+\\epsilon)$, so it gets divided down by exactly the same adaptive denominator as everything else — a weight with a large history of gradients has a large $\\hat s$ and so its penalty term shrinks along with its ordinary gradient, ending up decayed <i>less</i> than a quiet weight. That is backwards from the intent of weight decay, which is meant to pull every weight toward zero in proportion to its own size, not in inverse proportion to how active it has been. AdamW (§3.5.5) fixes this by applying $\\lambda\\theta$ directly to the weights, outside the $\\sqrt{\\hat s}$ division entirely.'
      },
      {
        q: 'Warmup exists mainly to…',
        options: ['save compute', 'avoid large early steps when Adam’s second-moment estimate is still unreliable', 'increase the batch size', 'prevent overfitting'],
        answer: 1,
        why: 'The bias-correction derivation in §3.5.4 shows exactly how unreliable the first few steps are: with a constant true gradient $g$, Adam\'s raw moving average reports only $m_t = g(1-\\beta_1^t)$, a tenth of the true value at $t=1$ with the standard $\\beta_1=0.9$. The correction $\\hat m_t = m_t/(1-\\beta_1^t)$ fixes the average, but $\\hat s_t$ is built from even less data at that point and can still be a poor, noisy estimate — dividing by a badly-estimated denominator can produce a huge first step in a direction the optimiser has not yet earned confidence in. Warmup simply keeps $\\eta$ small until enough steps have passed for both moving averages to settle.'
      },
      {
        q: 'WSD schedules are preferred over cosine at scale because…',
        options: ['they reach lower loss', 'they do not require committing to a total step count and allow branching from one stable trunk', 'they need no warmup', 'they are simpler to implement'],
        answer: 1,
        why: 'Cosine decay is shaped by the total step count from the very first step — change your mind about how long to train and the entire curve has to be redrawn so it still reaches zero at the new end point. WSD\'s long stable phase does not depend on knowing the end point at all; the decision to decay is made locally, once, whenever you choose to stop. That lets a single stable trunk serve several different short decay phases — for instance, comparing final data mixtures — a workflow cosine\'s single committed curve cannot support, while landing at essentially the same final loss.'
      }
    ],
    cards: [
      { q: 'Adam update', a: '$m,s$ EMAs of gradient and squared gradient, bias-corrected; $\\theta \\leftarrow \\theta - \\eta\\hat m/(\\sqrt{\\hat s}+\\epsilon)$.' },
      { q: 'AdamW vs Adam', a: 'Decoupled weight decay applied outside the adaptive denominator, so decay actually decays. The modern default.' },
      { q: 'WSD schedule', a: 'Warmup, long stable phase, sharp final decay — matches cosine and allows branching mid-run.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.6 */
  ML.section({
    id: 'normalisation', track: 'deep', num: '3.6',
    title: 'Normalisation and regularisation for networks',
    lede: 'The stabilisers that make depth trainable, and the four ways to stop a network memorising.',
    html: `
<p>§3.4 solved a problem at exactly one moment in time: it chose an initial weight variance so that activations neither explode nor vanish <i>at step zero</i>. That guarantee has an expiry date. The moment training begins, weights move away from their careful initial values in response to the gradient, and nothing forces them to keep obeying the variance arithmetic §3.4 derived. A hundred steps in, one layer's outputs might have drifted to twice the scale they started at; a hundred steps later, that drift has been multiplied through several more layers, and the same saturation or collapse §3.3 and §3.4 diagnosed at initialisation can reappear mid-training, self-inflicted by the very updates that are supposed to be improving the model.</p>

<p>The fix normalisation offers is more aggressive than picking a good starting point and hoping it holds: recompute the right scale <i>at every forward pass, at every layer</i>, directly from whatever activations actually arrived, rather than trusting an initial choice to survive unsupervised.</p>

<h2><span class="sn">3.6.1</span> One formula, four choices of axis</h2>
<p>Every normalisation layer does the same three things to some collection of numbers $z$: compute a mean $\\mu$ and a variance $\\sigma^2$ over some chosen set of entries, subtract and divide to get $\\hat z = (z-\\mu)/\\sqrt{\\sigma^2+\\epsilon}$, and then apply a learned scale and shift, $\\gamma\\hat z + \\beta$. The small $\\epsilon$ just stops a division by zero when a batch of numbers happens to have almost no spread. The learned $\\gamma$ and $\\beta$ matter more than they look: without them, every layer would be forced to output something with exactly mean 0 and variance 1, which throws away information a later layer might have needed — a unit that genuinely benefits from a large or skewed output has no way to express that. With $\\gamma$ and $\\beta$ present and learnable, the network can always recover the un-normalised version exactly, by setting $\\gamma=\\sigma$ and $\\beta=\\mu$, so normalisation can never make the network strictly less expressive; it only changes what the <i>default</i> distribution looks like before the network chooses to override it.</p>

<p>The only real design decision left is <i>which axis</i> the mean and variance are computed over, and that single choice has enormous practical consequences.</p>

${H.table(['Method', 'Normalises over', 'Depends on batch?', 'Used in'], [
      ['<b>BatchNorm</b>', 'the batch, per channel', 'Yes — and that is its weakness', 'CNNs at reasonable batch sizes'],
      ['<b>LayerNorm</b>', 'the features of one example', 'No', 'Transformers, RNNs'],
      ['<b>RMSNorm</b>', 'the features, no mean subtraction', 'No', 'Modern LLMs (§4.6) — cheaper, one less reduction'],
      ['<b>GroupNorm</b>', 'groups of channels within one example', 'No', 'Vision at small batch sizes']
    ])}

<p>BatchNorm computes $\\mu$ and $\\sigma^2$ across every example currently in the batch, for one channel at a time. That is fine as long as a batch is large and every example is processed the same way at train and test time — but neither holds up in general. At inference you often see one example at a time, with no batch to average over, so BatchNorm has to fall back on running statistics accumulated during training, meaning the layer genuinely computes something different depending on whether it thinks it is training or serving. With small batches the estimated mean and variance are noisy, adding an unwanted source of randomness to every forward pass. And in a sequence model, different positions in a batch of variable-length sequences carry different statistics, so averaging across the batch at a fixed position mixes together quantities that were never comparable in the first place.</p>

<p>LayerNorm sidesteps all three problems by normalising across the <i>features of one example</i> instead of across examples: mean and variance are computed from a single example's own activations, so the computation is identical whether that example arrives alone or inside a batch of a thousand, and nothing about training versus inference behaves differently. That is why transformers, which routinely serve one sequence at a time and process variable-length inputs, use LayerNorm rather than BatchNorm.</p>

<p><b>RMSNorm</b> goes one step further by asking whether the mean-subtraction step is pulling its weight. It computes $\\hat z = z / \\sqrt{\\overline{z^2}+\\epsilon}$ — dividing by the root-mean-square of $z$ directly, with no $\\mu$ subtracted at all — and then applies the same learned $\\gamma$. Skipping the mean is not merely a simplification for its own sake: computing a variance ordinarily requires the mean first (variance is defined relative to it), so LayerNorm needs two passes over the data, one to find $\\mu$ and one to find $\\sigma^2$ around it, while RMSNorm needs only the single pass required to compute $\\overline{z^2}$. One fewer reduction per normalisation layer, repeated across every layer of a network processing trillions of tokens, is a real and measurable saving, and in practice RMSNorm performs comparably to LayerNorm on the activations a modern transformer actually produces — which is why it has become the default (§4.6).</p>

${H.analogy(`<p>Think of a factory assembly line where each station is supposed to receive parts of a standard size, regardless of what happened at the stations before it. Without any correction, small variations at station 1 compound through stations 2, 3, 4 — a part that arrives a fraction too large gets built on by the next station, which builds on that, until by station 40 nothing fits together and something jams (§3.3, §3.4's collapse and explosion). A normalisation layer is a recalibration step bolted onto every single station: whatever arrives, regardless of its history, gets measured and rescaled to the standard size before the station does its own work. The station downstream never has to know or care whether the drift happened one step back or thirty.</p>`)}

<p><b>What you are looking at.</b> The lab below builds an 8-layer network with a fixed, deliberately aggressive weight scale, and draws one column of histograms per layer: each column shows the distribution of that layer's activations across 256 units, with darker cells meaning more units landed in that value bin. The standard deviation printed under each column is the same quantity §3.4's variance-propagation derivation tracked, now measured empirically layer by layer rather than predicted algebraically.</p>

<p><b>What to do with it.</b> Start with normalisation set to <i>none</i> and the activation on tanh. Watch the histogram columns widen and then pile up at the extreme top and bottom bins by around layer 4 — tanh saturating exactly as §3.3.1 described, with the standard deviation reading confirming the activations have spread far past a sensible range. Now switch normalisation to LayerNorm with everything else unchanged: every column looks essentially identical to every other column, all centred and the same width, regardless of how aggressive the raw weight scale is.</p>

<p><b>The thing genuinely worth noticing.</b> Switch from LayerNorm to RMSNorm and compare the columns — they are close to indistinguishable, even though RMSNorm never explicitly centres the distribution around zero the way LayerNorm does. For activations that are already roughly symmetric, which most pre-normalisation activations in a well-behaved network are, the mean-subtraction step LayerNorm performs turns out to correct for very little, and the cheaper computation buys almost the same stability. That is the empirical justification behind the algebraic saving described above.</p>

${H.lab('norm', 'What normalisation does to the distribution each layer sees', 'Activation histograms at increasing depth, with and without normalisation. Without it the distribution drifts and widens until the activation saturates; with it, every layer sees something stable.')}

<h2><span class="sn">3.6.2</span> What normalisation actually buys you</h2>
<p>The lab just above shows normalisation <i>working</i> — the distribution holds steady instead of drifting and saturating — but "the histogram looks stable" is a description of the symptom, not an explanation of why that stability helps an optimiser. The original BatchNorm paper offered one: normalisation, it argued, reduces <b>internal covariate shift</b> (ICS) — the way the distribution of a layer's inputs keeps changing during training as every earlier layer's weights keep updating, forcing each layer to continually re-adapt to a moving target rather than solving a fixed problem. It is a genuinely intuitive story, and it is worth knowing that it is not the explanation the evidence actually supports.</p>

${H.flag('The internal-covariate-shift explanation for why BatchNorm works is contested, and a better-supported alternative exists. Santurkar, Tsipras, Ilyas and Madry (2018) tested it directly: they trained networks with BatchNorm and then <i>deliberately injected noise after the normalisation step</i>, specifically re-introducing the kind of distributional drift ICS blames for slow training. If ICS reduction were the mechanism, reintroducing the drift should have erased BatchNorm\'s benefit — it barely moved the needle, and the noisy, still-drifting networks trained about as well as the clean ones. What they found instead does correlate with the benefit: BatchNorm measurably smooths the loss landscape the optimiser sees, in a sense made precise below, and that smoothing — not any reduction in how much the input distribution moves around — is the better-supported account of why it helps. Cite the historical name and the smoothing mechanism separately: "internal covariate shift" is the term you will see in nearly every older paper and blog post, including BatchNorm\'s own, but treat it as the field\'s working hypothesis at the time rather than as the settled reason.')}

<p>The smoothing argument is worth deriving in a simplified form, because the mechanism is genuinely a two-line piece of algebra once §3.2's ordinary weight-gradient rule is put next to it, and it explains something the ICS story never quite pins down: <i>why</i> a smoother landscape lets you train faster in the first place.</p>

${H.deriv('why normalising a layer\'s input decouples its gradient from an upstream scale that keeps drifting', [
      ['a downstream weight reads $\\hat z=(z-\\mu)/\\sigma$ rather than $z$ directly: $y=w\\hat z$', 'Any layer sitting after a normalisation step is, from its own point of view, an ordinary linear layer (§3.1) — the only difference from the un-normalised case is which quantity it happens to be looking at.'],
      ['$\\partial L/\\partial w = (\\partial L/\\partial y)\\,\\hat z$', 'The identical weight-gradient rule §3.2.3 derived for every layer — nothing special has to be invented for a normalised input specifically.'],
      ['$\\hat z$ has variance essentially 1 at every training step, by construction, however much the raw $z$\'s own mean and variance have drifted since the last step', 'That fixed-variance property is normalisation\'s entire job (§3.6.1): whatever distribution $z$ actually has this step, $\\hat z=(z-\\mu)/\\sigma$ is rescaled back to the same standard shape before $w$ ever sees it.'],
      ['so $\\partial L/\\partial w$\'s scale is controlled by $\\partial L/\\partial y$ and a quantity of known, fixed size — never by the raw scale of $z$ itself', 'Contrast the un-normalised case directly: $\\partial L/\\partial w=(\\partial L/\\partial y)\\,z$ there, and if $z$\'s own scale happens to have grown by a factor of 10 since the last step — exactly the drift this section opened with — the weight gradient silently inherits that same factor of 10, even though nothing about how good $w$ currently is has changed at all.']
    ], 'A learning rate that was safe for a given weight one step ago stays safe the next step only if the gradient\'s scale at that weight is predictable from step to step. Without normalisation, that scale rides on whatever the upstream activation distribution happens to be doing, which §3.4 and this section\'s opening paragraph both already showed is not fixed. With normalisation, the term multiplying $\\partial L/\\partial y$ is pinned to a known size by construction, so the same learning rate that worked last step keeps working this step — which is precisely the informal content of a landscape being "smoother": nearby points in weight space produce gradients of comparable, predictable scale, rather than a scale that can swing wildly depending on what upstream layers happened to do. This is a simplified sketch of Santurkar et al.\'s actual result, which formalises "predictable gradient scale" as a bound on the loss\'s Lipschitz constant — but the mechanism in miniature is exactly this: normalisation puts a ceiling on how much a layer\'s gradient can be dragged around by scale changes it did not cause.')}

<h3>Why BatchNorm's batch dependence breaks at inference</h3>
<p>§3.6.1 already flagged that BatchNorm computes its mean and variance from the batch currently passing through it, and that this becomes a liability the moment training ends. It is worth seeing exactly how it breaks, because the failure is not merely "no batch exists at inference" — it is stranger than that, and it is a real, production-grade bug source rather than a theoretical curiosity.</p>

${H.worked('the same input, two different answers, purely from who else is in the batch', `<p>Suppose one unit's pre-normalisation activation for the example you care about is $z=2.0$. If a naive implementation kept computing BatchNorm's statistics from whatever batch happens to be passing through at serving time — exactly what happens if a production system batches concurrent requests together for GPU throughput, a completely standard practice — the answer for that same $z=2.0$ depends on which other requests happened to arrive alongside it.</p>
<p>Batched with $\\{2.0,-1.0,3.0\\}$: mean $\\approx1.333$, standard deviation $\\approx1.700$, giving a normalised value $(2.0-1.333)/1.700\\approx0.392$. Batched instead with $\\{2.0,5.0,5.0,5.0\\}$ — a batch that simply happened to contain a run of large values from other requests — mean $=4.25$, standard deviation $\\approx1.299$, giving $(2.0-4.25)/1.299\\approx-1.732$. The identical input, the identical trained weights, and the model's answer for this one example has moved by more than two full standard deviations, sign and all, purely as a function of three completely unrelated requests it happened to be batched with. Nothing about the example itself changed.</p>`)}

${H.pitfall('This is exactly the bug dynamic batching creates if BatchNorm is left in training mode at serving time: a request\'s prediction becomes a function of whatever other requests happen to share its batch, which is both nondeterministic — the same input can return a different answer on two separate calls — and a genuine correctness defect, since nothing about the request itself changed. It has shipped in real systems. The fix a framework\'s "eval mode" applies is exactly the one below: stop computing statistics from the current batch at all, and use a fixed pair of numbers accumulated during training instead.')}

<p>Those fixed numbers are the <b>running statistics</b>: alongside the batch mean and variance actually used to normalise activations during training, BatchNorm separately accumulates a running mean and running variance, updated at every training step by blending in the current batch's statistics with an <b>exponential moving average</b> — a running estimate that leans on recent batches more than old ones, without needing to store every batch that has ever passed through:</p>
$$\\text{running} \\leftarrow (1-m)\\cdot\\text{running} + m\\cdot\\text{batch statistic}$$
<p>Read $m$, a small number typically around 0.1, as how much weight each single new batch gets against everything accumulated so far. A larger $m$ tracks recent batches more closely but is noisier; a smaller $m$ is more stable but slower to reflect a real shift in the data. At inference, BatchNorm switches to using this running mean and running variance in place of whatever the current batch's own statistics would have been — the same two fixed numbers for every request, computed once during training and then frozen, so the pathological batch-composition dependence in the worked example above cannot occur: an example's normalised value depends only on the example itself and the model's own accumulated history, never on whoever else happened to be riding alongside it in the same forward pass.</p>

${H.practice(`<p>The exponential-moving-average update means the running estimate does not track the truth instantly — it takes several steps to catch up from wherever it started, and how many depends entirely on $m$. Starting from a running mean of 0 with $m=0.1$ and a training run whose batch means happen to sit around 3.0: after one step the running mean is only $0.1\\times3.0=0.3$; after five steps of batch means near 3.0 it has climbed only to roughly 1.2, still well under half of the true level, because each step blends in only 10% of the new information and keeps 90% of a running average that started at zero. A network evaluated on a validation set partway through a short training run, or right after a change to the momentum $m$, can genuinely see running statistics that still lag the true activation distribution by a wide margin — not a bug in the mechanism, but a direct, predictable consequence of how slowly an exponential moving average catches up from a poor starting point. This is why some frameworks track an unbiased correction for the first few steps, and why restarting training from a checkpoint with freshly zeroed running statistics is a real, reproducible way to quietly wreck a model's validation accuracy without touching a single trained weight.</p>`)}

<h2><span class="sn">3.6.3</span> Regularisation, four ways</h2>
<p>Normalisation keeps a network <i>trainable</i>. It says nothing about whether the network, once trained, generalises to data it has not seen — a separate concern that §1.5 frames as the bias–variance trade-off and this section makes concrete for networks specifically. A network with enough parameters can, in principle, memorise its training set exactly, achieving a training loss of nearly zero while learning nothing that transfers to a new example. Regularisation is any deliberate choice that trades away some training-set fit in exchange for a model that generalises better, and there are four standard levers.</p>

${H.table(['Method', 'Mechanism', 'Note'], [
      ['<b>Weight decay</b>', 'Pull toward small weights (a Gaussian prior, §1.5)', 'Use AdamW so it is decoupled (§3.5)'],
      ['<b>Dropout</b>', 'Randomly zero units during training; scale at inference', 'Prevents co-adaptation. Largely replaced by other methods in large transformers, still standard in smaller nets'],
      ['<b>Early stopping</b>', 'Halt at the best validation epoch', 'The cheapest regulariser there is'],
      ['<b>Data augmentation</b>', 'Expand the effective dataset with label-preserving transforms', 'The most effective of all when the invariances are known']
    ])}

<h3>Dropout, worked through</h3>
<p>At training time, dropout independently zeroes each unit's output with probability $p$, and rescales every surviving unit by $1/(1-p)$ — a detail worth checking rather than taking on faith. A unit's expected output under this scheme is $(1-p)\\times\\big[\\text{value}\\times\\tfrac{1}{1-p}\\big] + p\\times 0 = \\text{value}$: the rescaling exactly cancels the effect of the dropping, in expectation, so the layer's average output matches what it would have been with no dropout at all. That is what lets inference skip dropout entirely and simply use every unit at full strength — the training-time rescaling has already made the two cases match on average, so no separate correction is needed at test time.</p>

<p>Every individual forward pass during training, though, is using a different random subset of units — a different <i>thinned sub-network</i> each time, with a different random 20% or 30% of units forced to zero. Training on many different draws of that random subset is, in effect, training a huge ensemble of overlapping smaller networks (§2.7 covers ensembles directly), and using the full network at inference approximates averaging their predictions. That averaging is one half of why dropout regularises. The other half is what it prevents during training: without dropout, two units can specialise into fixing each other's specific mistakes, a fragile co-adapted pair that only works because both happen to be present together. Dropout makes that co-adaptation unreliable, since either partner might be zeroed on any given step, and so it pushes every unit toward being independently useful rather than useful only in combination with particular others.</p>

<p><b>What you are looking at.</b> The lab below trains two networks side by side on the same noisy, moon-shaped dataset — one with dropout disabled, one with the dropout rate you set — and draws each network's decision boundary in its own panel, with the training points overlaid. The readout below reports training and test accuracy for both networks.</p>

<p><b>What to do with it.</b> Leave the label noise control at its default and press <i>Train 1500 steps</i>. Watch the left panel's boundary bend itself around individual mislabelled points, producing a jagged, overconfident shape, while its training accuracy climbs higher than the right panel's. Then compare the test-accuracy readouts: the right network, with dropout on, very often posts a lower training accuracy and a <i>higher</i> test accuracy than the left one.</p>

<p><b>The thing genuinely worth noticing.</b> That combination — worse on the data it saw, better on the data it did not — is not a contradiction to explain away. It is the single clearest signature a regulariser is doing its job, because it demonstrates directly that some of the left network's extra training accuracy was purchased by fitting noise specific to those exact training points, exactly the failure mode §1.5's bias–variance framing predicts and exactly what dropout's forced independence, described above, is designed to prevent.</p>

${H.lab('dropout', 'Dropout, and what it does to the boundary', 'Train the same network with and without dropout on noisy data. Watch the decision boundary go from jagged and confident to smoother and better-calibrated — and watch the training loss get <i>worse</i> while the test accuracy improves. That gap is the entire point.')}

${H.probe([
      ['LayerNorm or BatchNorm for a transformer?', 'LayerNorm (or RMSNorm) — no batch dependence, and per-position statistics are meaningless in a batch of variable-length sequences; BatchNorm additionally behaves differently at train and inference because it falls back on running statistics.'],
      ['What does dropout actually average over?', 'An exponential family of randomly thinned sub-networks, one different subset per forward pass during training; running the full network at inference approximates averaging their predictions, and the $1/(1-p)$ rescaling during training is what makes that approximation exact in expectation.'],
      ['Which regulariser is cheapest?', 'Early stopping — one validation curve you were already computing, and no extra compute spent during training itself.']
    ])}`,
    labs: {
      norm: function (host) {
        const st = Viz.controls(host, [
          { k: 'norm', label: 'normalisation', type: 'buttons', value: 'none', options: [{ v: 'none', t: 'none' }, { v: 'layer', t: 'LayerNorm' }, { v: 'rms', t: 'RMSNorm' }] },
          { k: 'act', label: 'activation', type: 'buttons', value: 'tanh', options: [{ v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' }] },
          { k: 'scale', label: 'weight scale', min: .5, max: 2.5, step: .05, value: 1.6, fmt: v => '×' + v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sd1', label: 'sd at layer 1', cls: 'key' }, { k: 'sd8', label: 'sd at layer 8' },
          { k: 'sat', label: 'saturated units at layer 8' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(53), n = 256, depth = 8;
            const act = st.act === 'tanh' ? Math.tanh : (z => Math.max(0, z));
            let a = Array.from({ length: n }, () => R.normal(0, 1));
            const layers = [];
            for (let l = 0; l < depth; l++) {
              const z = [];
              for (let i = 0; i < n; i++) {
                let s = 0;
                for (let j = 0; j < 64; j++) s += R.normal(0, st.scale / 8) * a[(j * 7 + i) % n];
                z.push(s);
              }
              let zz = z;
              if (st.norm === 'layer') { const m = Num.mean(z), sd = Num.sd(z) || 1; zz = z.map(v => (v - m) / sd); }
              else if (st.norm === 'rms') { const rms = Math.sqrt(Num.mean(z.map(v => v * v))) || 1; zz = z.map(v => v / rms); }
              a = zz.map(act);
              layers.push(a.slice());
            }
            const colW = (w - 40) / depth;
            layers.forEach((la, l) => {
              const hh = Num.hist(la, 26, -2.2, 2.2);
              const mx = Math.max.apply(null, hh.bins) || 1;
              const x0 = 24 + l * colW;
              hh.bins.forEach((c, i) => {
                const y = 40 + (i / 26) * (h - 90);
                ctx.fillStyle = T.blue; ctx.globalAlpha = .25 + .6 * (c / mx);
                ctx.fillRect(x0, y, Math.max(2, colW * .78 * (c / mx)), (h - 90) / 26 - 1);
                ctx.globalAlpha = 1;
              });
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
              ctx.fillText('L' + (l + 1), x0, 34);
              ctx.fillStyle = T.faint; ctx.textBaseline = 'top';
              ctx.fillText('σ=' + Num.sd(la).toFixed(2), x0, h - 44);
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('distribution of activations at each depth (each column is a histogram)', 24, h - 26);
            const sat = layers[depth - 1].filter(v => Math.abs(v) > .99).length / n;
            out({
              sd1: Num.sd(layers[0]).toFixed(3), sd8: Num.sd(layers[depth - 1]).toFixed(3),
              sat: (sat * 100).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'With weight scale 1.6 and no normalisation, tanh activations pile up at ±1 by layer 4 — every unit saturated, every derivative near zero, nothing learns. LayerNorm and RMSNorm both hold the distribution steady, and RMSNorm does it without computing a mean, which is the cheaper reduction modern LLMs prefer.');
      },

      dropout: function (host) {
        let netA = null, netB = null, trained = 0;
        const st = Viz.controls(host, [
          { k: 'p', label: 'dropout rate', min: 0, max: .6, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'noise', label: 'label noise in the data', min: 0, max: .3, step: .02, value: .12, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'width', label: 'width', min: 4, max: 24, step: 2, value: 16, fmt: v => v }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'trA', label: 'train acc · no dropout' }, { k: 'teA', label: 'test acc · no dropout' },
          { k: 'trB', label: 'train acc · dropout' }, { k: 'teB', label: 'test acc · dropout', cls: 'good' }
        ]);
        let data, test;
        function reset() {
          const R = Num.rng(9);
          data = Num.dataset('moons', 120, .28, 4);
          data.y = data.y.map(v => R() < st.noise ? 1 - v : v);
          test = Num.dataset('moons', 300, .28, 77);
          netA = Num.mlp([2, st.width, st.width, 1], { act: 'tanh', seed: 5 });
          netB = Num.mlp([2, st.width, st.width, 1], { act: 'tanh', seed: 5 });
          trained = 0; S.redraw();
        }
        function trainBoth(steps) {
          for (let i = 0; i < steps; i++) {
            netA.trainBatch(data.X, data.y, .05, 0);
            netB.trainBatch(data.X, data.y, .05, 0, st.p);   // real inverted dropout on every hidden unit — §3.6.2's exact scheme
            trained++;
          }
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const half = (w - 50) / 2;
            [[netA, 40, 'no dropout'], [netB, 40 + half + 22, 'dropout p = ' + st.p.toFixed(2)]].forEach(([net, x0, title]) => {
              const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-2.4, 2.4], pad: { l: x0, r: w - x0 - half, t: 26, b: 34 } })
                .frame({ xticks: [], yticks: [] });
              P.clip(() => Labs.boundary(P, (x, y) => net.predict([x, y]), { step: 5 }));
              P.clip(() => Labs.points(P, data.X, data.y, { r: 2.8 }));
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(title, x0, 8);
            });
            const accOf = (net, d) => Num.mean(d.X.map((x, i) => ((net.predict(x) > .5 ? 1 : 0) === d.y[i]) ? 1 : 0));
            out({
              trA: (accOf(netA, data) * 100).toFixed(1) + '%', teA: (accOf(netA, test) * 100).toFixed(1) + '%',
              trB: (accOf(netB, data) * 100).toFixed(1) + '%', teB: (accOf(netB, test) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Train 300 steps', primary: true, on: () => { trainBoth(300); S.redraw(); } },
          { label: 'Train 1500 steps', on: () => { trainBoth(1500); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        reset();
        Viz.note(host, 'Train both to 1500 steps with label noise on: the left network reaches higher training accuracy by bending around the mislabelled points, and pays for it on test. The right one refuses to. Higher training loss with better test accuracy is what a working regulariser looks like.');
      }
    },
    quiz: [
      {
        q: 'Transformers use LayerNorm/RMSNorm rather than BatchNorm because…',
        options: ['it is more accurate', 'batch statistics are unstable across variable-length sequences and differ between train and inference', 'BatchNorm is slower', 'LayerNorm has fewer parameters'],
        answer: 1,
        why: 'BatchNorm computes its mean and variance across the examples currently in the batch, which forces it to fall back on running statistics at inference (when a batch may not exist) and mixes together per-position statistics that are not comparable across a batch of variable-length sequences. LayerNorm computes its statistics from a single example\'s own features, so the same computation runs identically whether that example arrives alone or inside a batch of a thousand — nothing about train versus inference changes its behaviour. RMSNorm keeps that property and additionally skips the mean-subtraction step (§3.6.1), which is a genuine compute saving but is not the reason transformers moved away from BatchNorm in the first place.'
      },
      {
        q: 'Your training loss rises when you add dropout but test accuracy improves. This means…',
        options: ['dropout is misconfigured', 'the regulariser is working as intended', 'the learning rate is too high', 'the model is underfitting'],
        answer: 1,
        why: 'A regulariser\'s entire purpose is to trade away some training-set fit for better generalisation (§1.5\'s bias–variance framing), so a higher training loss alongside a higher test accuracy is the direct signature of that trade succeeding — some of the fit the unregularised model achieved was fitting noise specific to the training points, not signal, and §3.6.2 shows exactly this pattern in the dropout lab. Option D, underfitting, is the tempting distractor because "worse training performance" sounds like underfitting on its own — but underfitting means the model is too weak to fit even the true signal, and it would show up as <i>both</i> training and test performance being poor, not as training loss rising while test accuracy simultaneously improves.'
      }
    ],
    cards: [
      { q: 'LayerNorm vs BatchNorm vs RMSNorm', a: 'Per-example features / per-channel over the batch / per-example features with no mean subtraction (cheapest, modern LLM default).' },
      { q: 'Why dropout works', a: 'Each pass trains a thinned sub-network; inference averages the ensemble and units cannot co-adapt.' },
      { q: 'The four regularisers', a: 'Weight decay (AdamW), dropout, early stopping, data augmentation — augmentation is usually the strongest when invariances are known.' }
    ]
  });
})();
