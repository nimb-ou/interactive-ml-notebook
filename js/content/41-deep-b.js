/* ============================================================
   PART 3 — Deep learning (3.7 – 3.10): CNNs, RNNs, embeddings
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 3.7 */
  ML.section({
    id: 'cnn', track: 'deep', num: '3.7',
    title: 'Convolutional networks',
    lede: 'The architecture that made an inductive bias explicit: locality and translation invariance. An excellent bet for images, a poor one for long-range syntax — which is exactly the sentence §4.1 builds on.',
    rests: 'Rests on §0.2 (the dot product, and a matrix as a function on space).',
    html: `
<p>Try applying §3.1's ordinary layer, $h = Wx+b$, directly to an image. A modest $224\\times224$ colour photograph is $224\\times224\\times3 = 150{,}528$ numbers, so connecting it to a hidden layer of even 1000 units needs a weight matrix with $150{,}528\\times1{,}000 \\approx 150$ million entries — for one layer, before the network has done anything with them. Worse than the raw size is what those weights fail to encode. Every one of those 150 million connections is tied to an absolute pixel position: the weight linking pixel $(10,10)$ to hidden unit 7 is a completely different number from the weight linking pixel $(11,10)$ to that same unit, learned entirely independently. Shift every object in the photo one pixel to the right and the network has to relearn the whole pattern from scratch, because nothing in a plain dense layer knows that "an edge" is the same kind of feature wherever in the image it happens to appear.</p>

<p>What is missing is a way to say, in the architecture itself, two things a person would take for granted about images: a pixel's neighbours matter far more than a pixel on the opposite side of the frame (<b>locality</b>), and a useful feature detector — an edge, a corner, a patch of texture — is worth applying identically everywhere rather than relearning at every position (<b>parameter sharing</b>, which gives you <b>translation equivariance</b>: shift the input and the detected features shift with it). A <b>convolution</b> is the operation that encodes both at once.</p>

<h2><span class="sn">3.7.1</span> A convolution is a dot product, run over every window</h2>

<p>Take a small grid of learned numbers, the <b>kernel</b> — $3\\times3$ is typical — and slide it over the image one position at a time. At each position, flatten the $3\\times3$ patch of the image underneath it into 9 numbers, flatten the kernel into the same 9 slots, and take their dot product exactly as §0.2 defined it: multiply corresponding entries and sum. That single number becomes one entry of the output. Move the kernel one pixel over and repeat. The same 9 kernel weights (plus one bias) are reused at <i>every</i> position in the image — that reuse is the parameter sharing, and it is also why a convolution never has to relearn "edge" separately for the top-left corner and the bottom-right corner of the image.</p>

${H.worked('one kernel, two patches, verified by hand', `<p>Take the vertical-edge kernel $K = \\begin{bmatrix}-1&0&1\\\\-2&0&2\\\\-1&0&1\\end{bmatrix}$ (a Sobel filter) and a $3\\times3$ image patch that is dark on the left and bright on the right — a genuine vertical edge — with values $\\begin{bmatrix}0.2&0.2&0.9\\\\0.2&0.2&0.9\\\\0.2&0.2&0.9\\end{bmatrix}$.</p>
<p>The dot product, row by row: row 1 gives $(-1)(0.2)+(0)(0.2)+(1)(0.9)=0.7$; row 2 gives $(-2)(0.2)+(0)(0.2)+(2)(0.9)=1.4$; row 3 gives $0.7$ again by the same arithmetic as row 1. Total: $0.7+1.4+0.7=2.8$ — a strong, positive response, exactly what a detector built to fire on left-to-right brightening should produce.</p>
<p>Now try a patch with no edge at all, flat at $0.5$ everywhere. Row 1: $(-1)(0.5)+(0)(0.5)+(1)(0.5)=0$; every row gives 0 by the identical cancellation. Total: exactly 0. The kernel is completely silent on a flat region and fires only where there is a genuine left-to-right brightness change — which is precisely the behaviour "edge detector" promises, and it falls straight out of the dot product with no special-casing anywhere.</p>`)}

<p><b>What you are looking at.</b> The lab below shows a small procedural test image on the left — a bright block with a hole in it, and a textured band across the bottom — with the current $3\\times3$ kernel drawn to its right, each cell coloured blue for positive and red for negative and labelled with its number. The panel furthest right is the output: every pixel there is one dot product between the kernel and the image patch underneath it, coloured on a signed scale since a kernel like Sobel can produce negative as well as positive responses.</p>

<p><b>What to do with it.</b> Start on the vertical-edge kernel and look at the output: it lights up brightly along the block's left and right edges, exactly where brightness changes sharply left-to-right, and stays near zero across the flat interior and the flat background — the same behaviour the two hand-computed patches above predicted. Switch to the horizontal-edge kernel and the bright response rotates to the block's top and bottom edges instead. Try blur and watch every sharp boundary in the image soften; try Laplacian, which responds to edges in <i>any</i> direction at once because it does not privilege a row or a column the way Sobel does.</p>

<p><b>The thing genuinely worth noticing.</b> The readout compares <i>parameters used</i> — a fixed 10, nine kernel weights plus one bias, whatever the image size — against what a fully-connected layer producing an output of the same size would need, which scales with the image's area squared and reaches the hundreds of thousands even for this tiny 28-pixel test image. That ratio is not a minor efficiency gain; it is the entire reason convolutional networks were trainable on 1998-era hardware at all, and it holds regardless of how large the input image grows, because the kernel's size never depends on the image's size.</p>

${H.lab('conv', 'Convolution, kernel by kernel', 'Pick a kernel and watch it applied to the image, one window at a time. The classic edge detectors are not magic — they are the numbers you can read in the grid, and the network learns numbers like these rather than being given them.')}

<h2><span class="sn">3.7.2</span> Shapes, stride, padding</h2>
<p>Sliding a $k\\times k$ kernel across an input of size $H_{\\text{in}}$ produces an output that is smaller, unless you compensate for it, and the exact output size is worth deriving rather than memorising.</p>

${H.deriv('the convolution output-shape formula', [
      ['no padding, stride 1: valid starts $0, 1, \\ldots, H_{\\text{in}}-k$', 'A kernel window starting at position $i$ covers indices $i$ through $i+k-1$. For the window to fit entirely inside the input, its last index must not exceed $H_{\\text{in}}-1$, i.e. $i \\le H_{\\text{in}}-k$.'],
      ['number of valid positions $= H_{\\text{in}}-k+1$', 'The starting positions run from 0 up to $H_{\\text{in}}-k$ inclusive, which is $H_{\\text{in}}-k+1$ integers.'],
      ['stride $s$: keep only every $s$-th start', 'A stride of $s$ uses starting positions $0, s, 2s, \\ldots$ instead of every integer, so the count of valid positions becomes $\\lfloor(H_{\\text{in}}-k)/s\\rfloor + 1$ — the floor accounts for a final partial step that does not reach a valid position.'],
      ['padding $p$ on each side: replace $H_{\\text{in}}$ with $H_{\\text{in}}+2p$', 'Padding adds $p$ extra positions on both ends before the sliding starts, so the effective input length used in every step above is $H_{\\text{in}}+2p$.'],
      ['$H_{\\text{out}} = \\left\\lfloor\\dfrac{H_{\\text{in}}+2p-k}{s}\\right\\rfloor + 1$', 'Combine all three adjustments into the general formula.']
    ], '"Same" padding, meaning the output stays the same size as the input, is the special case $p=(k-1)/2$ with $s=1$: check it by substitution and the $-k$ and $+2p=(k-1)$ very nearly cancel, leaving output size equal to input size (exactly, once the floor is accounted for). Parameters in a conv layer are $k^2 \\times C_{\\text{in}} \\times C_{\\text{out}} + C_{\\text{out}}$ — note that $H_{\\text{in}}$ never appears in that count, which is the same independence-from-image-size observation the lab above made numerically.')}

<h2><span class="sn">3.7.3</span> The receptive field: how far back one output unit can see</h2>
<p>A single convolution layer only looks at a $k\\times k$ neighbourhood, but a <i>stack</i> of layers sees further, because a unit in layer 2 depends on several units in layer 1, each of which depends on its own $k\\times k$ neighbourhood in the input. The <b>receptive field</b> is the size of that indirect neighbourhood, and it grows by a fixed amount at every added layer: each additional $3\\times3$, stride-1 convolution extends how far a unit can see by exactly one pixel on every side, because the layer beneath it needed its own kernel radius of reach to produce each of the pixels the new layer combines. After $L$ such layers, the receptive field has radius $L$, so its full width is $2L+1$ — which is why two stacked $3\\times3$ convolutions (receptive field $2(2)+1=5$, at a cost of $9+9=18$ parameters per input–output channel pair) see exactly as far as one $5\\times5$ convolution (also receptive field 5, but $25$ parameters), while adding an extra nonlinearity between them for free.</p>

<p><b>What you are looking at.</b> The lab below plots receptive field size, in pixels, against the number of stacked convolution layers, for the kernel size, stride and dilation you choose. The red dashed line marks 224 pixels, the size of a typical input image, as a concrete benchmark for "sees the whole picture".</p>

<p><b>What to do with it.</b> Leave kernel size at 3, stride at 1, and dilation off, and watch the readout's <i>layers to cover 224px</i> figure: it takes around 112 stacked $3\\times3$ layers before a single output unit's receptive field reaches edge to edge on a realistic image, growing by only 2 pixels of reach per layer. Now switch dilation growth on, which spaces each successive kernel's taps further apart rather than keeping them adjacent, and watch the same 224-pixel threshold get crossed in around 7 layers instead of 112 — the receptive field is now roughly doubling at every layer rather than merely creeping upward by a constant amount.</p>

<p><b>The thing genuinely worth noticing.</b> A hundred and twelve layers of plain $3\\times3$ convolution to see a whole image is not a hypothetical inconvenience; it is close to why very deep plain CNNs were considered impractical before residual connections (§3.3.3) made such depth trainable at all, and dilation is the CNN-side answer to a question that reappears, in a completely different guise, in every transformer: how do you let a unit see far away without paying for it one adjacent step at a time? A standard self-attention layer answers the same question by making the path length between <i>any</i> two positions exactly 1, at quadratic rather than linear cost in sequence length — the direct trade-off dilation refuses to make, and §4.1 opens from precisely this comparison.</p>

${H.lab('rf', 'Receptive field growth, computed', 'Add layers, change stride and dilation, and watch how much of the input one output neuron sees. The stride column is what makes deep CNNs see globally without quadratic cost — and the analogous question for transformers is §4.4’s sliding-window attention.')}

<h2><span class="sn">3.7.4</span> The architecture lineage, and what each one contributed</h2>
${H.table(['Model', 'Year', 'The idea it added'], [
      ['LeNet-5', '1998', 'Convolution + pooling + fully connected, trained end to end'],
      ['AlexNet', '2012', 'ReLU, dropout, GPUs, augmentation — the moment deep learning became practical'],
      ['VGG', '2014', 'Depth from stacked 3×3 kernels only'],
      ['Inception', '2014', 'Multiple kernel sizes in parallel; 1×1 convolutions as cheap channel mixing'],
      ['ResNet', '2015', '<b>Residual connections</b> — the single most transferable idea in the list, and the reason transformers train at depth (§4.5)'],
      ['DenseNet / EfficientNet', '2017–19', 'Feature reuse; compound scaling of depth, width and resolution'],
      ['ConvNeXt', '2022', 'A CNN modernised with transformer-era training recipes — competitive again, which is a useful corrective to "attention is all you need" as a slogan']
    ])}

${H.history(`<p>LeNet-5, trained by Yann LeCun and collaborators on handwritten digit recognition, demonstrated the whole recipe above — shared kernels, pooling, end-to-end gradient training — as early as 1998, on hardware that made anything larger impractical. The idea sat mostly unused at scale for over a decade, not because it was wrong, but because the deep-network training problems §3.3 and §3.4 diagnose (saturating activations, badly scaled initialisation) made anything beyond a handful of convolutional layers unreliable to train, and the datasets and compute available were far smaller than a hungry model needed.</p>
<p>AlexNet's 2012 ImageNet result changed the calculation on every one of those fronts at once: ReLU activations to sidestep saturation (§3.3), dropout to regularise a network with far more parameters than any prior vision model (§3.6), and, critically, training on GPUs, which made a network of AlexNet's size feasible to train in days rather than months. The years directly afterward were mostly refinements of one architecture at a time — VGG pushed depth using nothing but stacked $3\\times3$ kernels, Inception mixed several kernel sizes in parallel — until ResNet's 2015 residual connections solved the depth-training problem directly enough that networks over 100 layers deep became routine. That single idea outlived the convolutional architectures it was invented for: every transformer block built in Part 4 wraps its attention and feed-forward sub-layers in exactly the same residual pattern.</p>`)}

<h2><span class="sn">3.7.5</span> Pooling, and its decline</h2>
<p>Max pooling summarises a small window — commonly $2\\times2$ — by keeping only its strongest response and discarding the rest, which halves the spatial size for free and gives a small amount of genuine translation invariance: if the strongest feature in a window shifts by a pixel but stays inside the same window, the pooled output does not change at all. Modern architectures increasingly replace it with two alternatives that turned out to work at least as well. A <b>strided convolution</b> downsamples using a kernel that is itself learned, rather than a fixed max operation, folding the downsampling decision into the same weights that already do feature detection. And <b>global average pooling</b>, used just before the final classifier, collapses each channel to a single number — its average over the whole spatial map — with no learned parameters at all, and empirically generalises better than flattening the last feature map into a large dense layer, because it cannot overfit to the exact spatial arrangement of a training image the way a dense layer's per-position weights can.</p>

${H.probe([
      ['Why two 3×3 rather than one 5×5?', 'Both reach a receptive field of 5 pixels — §3.7.3 derives the $2L+1$ growth rule directly — but two $3\\times3$ kernels cost $9+9=18$ parameters per channel pair against one $5\\times5$\'s 25, and add an extra nonlinearity between them at no cost.'],
      ['What inductive bias does a CNN encode?', 'Locality (each output depends only on a small neighbourhood) and translation equivariance (the same detector is applied everywhere via shared weights), which together are what let a $3\\times3$ kernel cost 10 parameters regardless of image size, as the lab in §3.7.1 shows directly.'],
      ['Why do transformers beat CNNs on language?', 'Language dependencies are long-range and not organised around spatial adjacency the way image features are; a CNN needs many stacked layers (§3.7.3\'s 112-layer figure, without dilation) to connect distant positions, while attention gives every pair of positions a path length of 1 directly (§4.1).']
    ])}`,
    labs: {
      conv: function (host) {
        const N = 28;
        // procedural "image": a shape with edges and texture
        const img = [];
        for (let i = 0; i < N; i++) {
          const row = [];
          for (let j = 0; j < N; j++) {
            const cx = j - N / 2, cy = i - N / 2;
            let v = 0.15;
            if (Math.abs(cx) < 7 && Math.abs(cy) < 9) v = 0.85;             // block
            if (cx * cx + cy * cy < 16) v = 0.35;                            // hole
            if (i > 20) v = 0.6 + 0.25 * Math.sin(j * 1.4);                  // texture
            row.push(v);
          }
          img.push(row);
        }
        const kernels = {
          edgeV: { k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], t: 'vertical edges (Sobel)' },
          edgeH: { k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], t: 'horizontal edges' },
          blur: { k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]], t: 'blur (box)' },
          sharp: { k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], t: 'sharpen' },
          lap: { k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]], t: 'Laplacian (all edges)' },
          id: { k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], t: 'identity' }
        };
        const st = Viz.controls(host, [
          { k: 'kern', label: 'kernel', type: 'select', value: 'edgeV', options: Object.keys(kernels).map(k => ({ v: k, t: kernels[k].t })) },
          { k: 'stride', label: 'stride', min: 1, max: 3, step: 1, value: 1, fmt: v => v },
          { k: 'relu', label: 'apply ReLU after', type: 'toggle', value: true },
          { k: 'pool', label: 'then 2×2 max pool', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'in', label: 'input shape', cls: 'key' }, { k: 'outsh', label: 'output shape' },
          { k: 'params', label: 'parameters used' }, { k: 'dense', label: 'a dense layer would need' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const K = kernels[st.kern].k;
            const s = st.stride, outN = Math.floor((N - 3) / s) + 1;
            let outImg = [];
            for (let i = 0; i < outN; i++) {
              const row = [];
              for (let j = 0; j < outN; j++) {
                let acc = 0;
                for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) acc += K[a][b] * img[i * s + a][j * s + b];
                row.push(st.relu ? Math.max(0, acc) : acc);
              }
              outImg.push(row);
            }
            if (st.pool) {
              const pN = Math.floor(outN / 2), pooled = [];
              for (let i = 0; i < pN; i++) {
                const row = [];
                for (let j = 0; j < pN; j++) {
                  row.push(Math.max(outImg[2 * i][2 * j], outImg[2 * i][2 * j + 1], outImg[2 * i + 1][2 * j], outImg[2 * i + 1][2 * j + 1]));
                }
                pooled.push(row);
              }
              outImg = pooled;
            }
            const cell = Math.min((h - 70) / N, (w * .28) / N);
            const drawGrid = (m, ox, oy, cs, title, signed) => {
              const flat = m.flat();
              const mx = Math.max.apply(null, flat.map(Math.abs)) || 1;
              m.forEach((row, i) => row.forEach((v, j) => {
                let c;
                if (signed) {
                  const t = v / mx;
                  c = t >= 0 ? 'rgb(' + Math.round(30 + 200 * t) + ',' + Math.round(50 + 120 * t) + ',' + Math.round(90 + 40 * t) + ')'
                             : 'rgb(' + Math.round(30 - 20 * t) + ',' + Math.round(40 - 40 * t) + ',' + Math.round(90 - 140 * t) + ')';
                } else {
                  const g = Math.round(255 * Math.max(0, Math.min(1, v)));
                  c = 'rgb(' + g + ',' + g + ',' + Math.min(255, g + 12) + ')';
                }
                ctx.fillStyle = c;
                ctx.fillRect(ox + j * cs, oy + i * cs, cs + .5, cs + .5);
              }));
              ctx.strokeStyle = T.line; ctx.strokeRect(ox, oy, m[0].length * cs, m.length * cs);
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
              ctx.fillText(title, ox, oy - 5);
            };
            drawGrid(img, 14, 30, cell, 'input ' + N + '×' + N, false);
            // kernel
            const kx = 14 + N * cell + 22, kcell = 24;
            ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('kernel 3×3', kx, 25);
            K.forEach((row, i) => row.forEach((v, j) => {
              ctx.fillStyle = v > 0 ? 'rgba(90,160,255,.28)' : v < 0 ? 'rgba(230,90,80,.28)' : T.panel;
              ctx.fillRect(kx + j * kcell, 30 + i * kcell, kcell - 2, kcell - 2);
              ctx.strokeStyle = T.line; ctx.strokeRect(kx + j * kcell, 30 + i * kcell, kcell - 2, kcell - 2);
              ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(Math.abs(v) < .2 && v !== 0 ? v.toFixed(2) : String(v), kx + j * kcell + kcell / 2 - 1, 30 + i * kcell + kcell / 2 - 1);
            }));
            ctx.fillStyle = T.muted; ctx.font = '18px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText('∗', kx + 1.5 * kcell, 30 + 3 * kcell + 22);
            const outCell = Math.min((h - 70) / outImg.length, (w * .28) / outImg.length);
            drawGrid(outImg, kx + 3 * kcell + 30, 30, outCell, 'output ' + outImg.length + '×' + outImg[0].length + (st.relu ? ' (ReLU)' : '') + (st.pool ? ' + pool' : ''), true);
            out({
              in: N + '×' + N, outsh: outImg.length + '×' + outImg[0].length,
              params: '9 + 1 bias',
              dense: (N * N * outN * outN).toLocaleString()
            });
          }
        });
        Viz.note(host, 'The readout compares 10 parameters against the ~600,000 a fully-connected layer of the same output size would need. That ratio — parameter sharing — is why convolutions made image models trainable in 1998 and why they remain the cheapest way to process a grid.');
      },

      rf: function (host) {
        const st = Viz.controls(host, [
          { k: 'layers', label: 'conv layers', min: 1, max: 20, step: 1, value: 6, fmt: v => v },
          { k: 'k', label: 'kernel size', min: 3, max: 7, step: 2, value: 3, fmt: v => v + '×' + v },
          { k: 'stride', label: 'stride (every layer)', min: 1, max: 2, step: 1, value: 1, fmt: v => v },
          { k: 'dil', label: 'dilation growth', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rf', label: 'receptive field', cls: 'key' }, { k: 'params', label: 'params per channel pair' },
          { k: 'need', label: 'layers to cover 224px' }, { k: 'attn', label: 'attention would need' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const rfs = []; let rf = 1, jump = 1;
            for (let l = 1; l <= st.layers; l++) {
              const d = st.dil ? Math.pow(2, l - 1) : 1;
              rf = rf + (st.k - 1) * d * jump;
              jump = jump * st.stride;
              rfs.push(rf);
            }
            const P = Viz.plot(ctx, w, h, { xd: [1, Math.max(2, st.layers)], yd: [0, Math.max(30, rfs[rfs.length - 1] * 1.1)] })
              .frame({ xlabel: 'layer', ylabel: 'receptive field (pixels)' });
            P.clip(() => {
              P.line(rfs.map((v, i) => [i + 1, v]), { color: T.blue, width: 2.6 });
              P.dots(rfs.map((v, i) => [i + 1, v]), { r: 3.6, color: T.blue });
              P.hline(224, { color: T.red, dash: [4, 4], label: 'a 224px image' });
            });
            let need = 0, r2 = 1, j2 = 1;
            while (r2 < 224 && need < 500) { const d = st.dil ? Math.pow(2, need) : 1; r2 += (st.k - 1) * d * j2; j2 *= st.stride; need++; }
            out({
              rf: rfs[rfs.length - 1] + ' px',
              params: (st.k * st.k * st.layers).toLocaleString(),
              need: need + ' layers', attn: '1 layer (O(n²) cost)'
            });
          }
        });
        Viz.note(host, 'Turn on dilation growth: the receptive field doubles per layer instead of growing linearly, covering an image in a handful of layers. That is the CNN answer to long-range dependency — and it is exactly the trade attention removes by making the path length 1 at quadratic cost (§4.1).');
      }
    },
    quiz: [
      {
        q: 'Two stacked 3×3 convolutions are usually preferred to one 5×5 because…',
        options: ['they are faster to compute', 'same receptive field, fewer parameters, and an extra nonlinearity', 'they have a larger receptive field', 'they avoid padding'],
        answer: 1,
        why: 'The receptive-field growth rule derived in §3.7.3 gives a stack of $L$ stride-1 $3\\times3$ layers a receptive field of $2L+1$, so two of them reach exactly 5 pixels, the same as one $5\\times5$ kernel. Where they differ is parameter count — $9+9=18$ per channel pair against $25$ — and the two-layer version inserts an extra activation between the layers, which the single $5\\times5$ kernel cannot do since it is one linear operation applied once. Option C is the tempting wrong answer because "stacking more layers" sounds like it should reach further, but the arithmetic in §3.7.3 shows the two options land on the identical receptive field, not a larger one.'
      },
      {
        q: 'A CNN’s inductive bias is…',
        options: ['sequential dependence', 'locality and translation equivariance with parameter sharing', 'permutation invariance', 'no bias at all'],
        answer: 1,
        why: 'A convolution kernel only ever looks at a small neighbourhood (locality) and the identical kernel is reused at every position in the image (parameter sharing), which is what makes the map translation-equivariant — shift the input and the output feature map shifts by the same amount. §3.7.1\'s opening calculation is the concrete case against the alternative: a plain dense layer on a $224\\times224$ image needs on the order of 150 million weights for one layer and ties every one of them to an absolute pixel position, so it encodes no such bias at all. That bet is excellent for images, where nearby pixels genuinely are more related than distant ones, and poor for language, where a dependency between two words does not care how far apart they sit in the sentence — the observation §4.1 builds its opening argument from.'
      }
    ],
    cards: [
      { q: 'Conv output shape', a: '$\\lfloor (H+2p-k)/s\\rfloor + 1$; parameters $k^2C_{in}C_{out}+C_{out}$, independent of image size.' },
      { q: 'Receptive field of L stacked 3×3 layers', a: '$2L+1$ with stride 1; dilation makes it grow geometrically.' },
      { q: 'ResNet’s contribution', a: 'Residual connections — a gradient path of derivative 1, which is what makes very deep stacks (including transformers) trainable.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.8 */
  ML.section({
    id: 'rnn', track: 'deep', num: '3.8',
    title: 'Recurrent networks, LSTM, and why they lost',
    lede: 'The right prior for language and the wrong computational shape for a GPU. Understanding both halves is what makes §4.1 land.',
    rests: 'Rests on §3.2.3\'s propagation step and §3.3\'s vanishing/exploding gradient mechanism, both extended from depth to time.',
    html: `
<p>§3.7 built weight sharing <i>across space</i>: the same $3\\times3$ kernel is reused at every position in an image, because a useful feature detector should not have to be relearned separately for each pixel. A sentence poses the analogous question about a different axis. A useful pattern for predicting the next word — "after a comma, expect a clause"; "a name introduced earlier probably gets referred to again" — should not have to be relearned separately for every position in the sentence either, and a sentence can be arbitrarily long, so whatever processes it cannot have a size that grows with sequence length. What is needed is weight sharing <i>across time</i>: one set of weights, applied identically at every timestep, carrying forward a summary of everything seen so far in a fixed-size state.</p>

<h2><span class="sn">3.8.1</span> The recurrence</h2>
<p>A <b>recurrent neural network</b> maintains a hidden state $h_t$ and updates it at every timestep with the same rule:</p>
$$h_t = \\phi(W_h h_{t-1} + W_x x_t + b), \\qquad y_t = W_y h_t$$
<p>Read this as an ordinary layer from §3.1 — a linear combination followed by a nonlinearity — with one addition: alongside the current input $x_t$, the layer also receives its <i>own previous output</i> $h_{t-1}$, fed back in as a second input. $W_h$, $W_x$, $b$ and $\\phi$ are the same four objects at every single timestep; nothing about them changes as $t$ increases, which is exactly the parameter-sharing move §3.7 made across space, now made across time instead. Because the update rule does not care how large $t$ gets, the same fixed-size network can be run over a sequence of length 5 or a sequence of length 5,000 without adding a single parameter — an elegant inductive bias for language, where the past genuinely conditions the present but a fixed context window would arbitrarily cut that conditioning off.</p>

<h2><span class="sn">3.8.2</span> Backpropagation through time — the same propagation rule, applied to the wrong axis</h2>
<p>Training an RNN means differentiating a loss with respect to $W_h$, and the standard way to do it is to <b>unroll</b> the recurrence: write $h_1, h_2, \\ldots, h_T$ out as a chain of $T$ ordinary layers, each one feeding the next, and run backpropagation across that chain exactly as §3.2.3 described. The propagation step from that section, $\\delta^{(l)} = (W^{(l+1)\\mathsf{T}}\\delta^{(l+1)})\\odot\\phi'(z^{(l)})$, applies here completely unchanged — an RNN unrolled over time is not a different algorithm from a deep feedforward network, it is the identical algorithm applied to a network whose "layers" happen to be timesteps. There is exactly one difference, and it turns out to be decisive: in an ordinary deep network, layer 1 and layer 40 have <i>different</i> weight matrices $W^{(1)}$ and $W^{(40)}$. In an unrolled RNN, every single timestep reuses the identical matrix $W_h$, because that reuse is the whole point of the architecture.</p>

${H.deriv('why reusing one matrix makes the vanishing/exploding problem sharper', [
      ['$\\delta_t = (W_h^\\mathsf{T}\\delta_{t+1}) \\odot \\phi\'(z_t)$', 'The propagation rule from §3.2.3, one link per timestep, applied backwards from the end of the sequence.'],
      ['$\\delta_1 \\approx \\big(\\textstyle\\prod_{k=1}^{T-1} W_h^\\mathsf{T}\\mathrm{diag}(\\phi\'_k)\\big)\\,\\delta_T$', 'Unroll the recurrence all the way back to step 1. Approximate by folding the activation gates into diagonal matrices between each application of $W_h^\\mathsf{T}$, so the whole chain is a product of $T-1$ near-identical factors.'],
      ['dominant behaviour set by the spectral radius $\\rho$ of $W_h$', 'For a matrix applied repeatedly to a vector, the component along the eigenvector with the largest eigenvalue magnitude — the spectral radius $\\rho$ — eventually dominates every other component, exactly as in power iteration. Ignore the activation gates for a moment (they only make the effective factor smaller, never larger) and the product above behaves like $W_h^{\\mathsf{T}(T-1)}$.'],
      ['$\\|\\delta_1\\| \\sim \\rho^{\\,T-1}\\,\\|\\delta_T\\|$', 'Repeated multiplication by a matrix with spectral radius $\\rho$ scales a generic vector\'s norm by very nearly $\\rho$ at every application, so after $T-1$ applications the scaling is $\\rho^{T-1}$ — a direct matrix generalisation of the scalar $0.25^{\\text{depth}}$ argument §3.3.2 made for a stack of sigmoid layers.']
    ], 'If $\\rho<1$, the gradient reaching early timesteps vanishes geometrically in sequence length, exactly as depth caused vanishing in §3.3 — except here the base of the exponent is fixed by $W_h$ alone, so no number of added parameters elsewhere in the network changes it, only the recurrent weights themselves and the activation gates riding along with them. If $\\rho>1$, the gradient explodes the same way. And because it is the identical $W_h$ at every step, there is no equivalent of "make the deeper layers behave differently from the shallow ones" — every timestep is stuck living under the same spectral radius.')}

${H.worked('the same matrix, two spectral radii, both computed', `<p>Idealise the recurrence down to one hidden unit, so $W_h$ is a single number $\\rho$ rather than a matrix, and ignore the activation gate for clarity. Over $T=40$ steps, $\\rho=0.7$ gives $0.7^{40}$: $\\ln 0.7 \\approx -0.357$, times 40 is $-14.27$, so $0.7^{40} \\approx e^{-14.27} \\approx 6\\times10^{-7}$ — a gradient reaching step 1 attenuated to seven parts in ten million, indistinguishable from zero in ordinary floating point. Over the same 40 steps, $\\rho=1.1$ gives $1.1^{40}$: $\\ln 1.1 \\approx 0.0953$, times 40 is $3.81$, so $1.1^{40}\\approx e^{3.81}\\approx45$ — a modest-looking $10\\%$ excess per step compounding to a gradient over 40 times too large by the time it reaches the first timestep. Ten percent either side of 1, held constant over forty repetitions of the identical operation, is the entire gap between "vanished" and "unstable".</p>`)}

<p><b>What you are looking at.</b> The lab below builds a genuine 16-dimensional recurrent network with a random weight matrix scaled to have the spectral radius you choose, runs it forward over a sequence of the length you set, then runs a real backward pass and plots $\\log_{10}$ of the gradient magnitude reaching each timestep, from most recent back to earliest.</p>

<p><b>What to do with it.</b> Leave the spectral radius at its default, just under 1, and watch the line slope steadily downward from right to left on the logarithmic axis — a straight line on a log scale is exactly the signature of geometric decay the derivation above predicts. Push the spectral radius past 1 and the slope flips: the line now rises toward the earliest steps, and the verdict reports exploded. Try a long sequence, 50 or 60 steps, at a spectral radius comfortably below 1, and watch the readout's <i>effective memory</i> figure — the number of steps back the gradient stays within a factor of 10 of its value at the end — settle at a small number regardless of how much longer the sequence runs.</p>

<p><b>The thing genuinely worth noticing.</b> Switch on the LSTM-style additive path while keeping the spectral radius exactly where it was. The steep slope that dominated the plot a moment ago flattens out to something close to horizontal across the whole sequence, even though nothing about $W_h$'s spectral radius changed at all. That flattening is not a different optimiser or a different initialisation — it is an architectural change to what the recurrence itself computes, and §3.8.3 below derives exactly why it works.</p>

${H.lab('bptt', 'Gradient decay through time, measured', 'A real unrolled recurrence, with the gradient magnitude reaching each earlier step computed by backpropagation. Move the spectral radius past 1 and watch explosion; below and watch a 40-step memory disappear by step 10.')}

<h2><span class="sn">3.8.3</span> LSTM and GRU: additive memory</h2>
<h2><span class="sn">3.8.3</span> LSTM and GRU: making the escape route explicit</h2>
<p>§3.3.3 fixed the feedforward version of this exact problem with a residual connection, $y=x+F(x)$, whose derivative $\\partial y/\\partial x = 1+\\partial F/\\partial x$ always contains a hard-coded term of exactly 1, giving the gradient a route past the layer's own multiplicative machinery. The <b>LSTM</b> applies the identical idea to time instead of depth, with one refinement: rather than a fixed contribution of exactly 1, it makes the strength of that escape route a <i>learned, input-dependent</i> quantity. Alongside the ordinary hidden state $h_t$, it maintains a separate <b>cell state</b> $c_t$, updated by three gates — numbers in $(0,1)$ produced by a sigmoid, so they behave as soft, differentiable on/off switches:</p>
$$f_t = \\sigma(W_f[h_{t-1},x_t]),\\quad i_t = \\sigma(W_i[\\cdot]),\\quad o_t = \\sigma(W_o[\\cdot])$$
$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tanh(W_c[\\cdot]), \\qquad h_t = o_t \\odot \\tanh(c_t)$$

${H.deriv('why the cell state gives gradients a near-identity path, gate by gate', [
      ['$c_t = f_t \\odot c_{t-1} + i_t \\odot g_t, \\quad g_t := \\tanh(W_c[\\cdot])$', 'Name the candidate update $g_t$ so the recurrence reads cleanly: the new cell state is the old one, scaled entrywise by the forget gate, plus a fresh contribution scaled by the input gate.'],
      ['$\\dfrac{\\partial c_t}{\\partial c_{t-1}} = f_t$', 'Differentiate directly. Unlike the plain RNN\'s $h_t = \\phi(W_hh_{t-1}+\\cdots)$, there is no weight matrix and no activation function sitting between $c_t$ and $c_{t-1}$ — the local derivative is simply the forget gate\'s own value, an entrywise multiplication, nothing more.'],
      ['$\\dfrac{\\partial c_t}{\\partial c_{t-k}} = \\displaystyle\\prod_{j=t-k+1}^{t} f_j$', 'Chain the single-step result across $k$ timesteps (§0.3.5\'s chain rule, applied along the cell-state path specifically). Compare this directly to §3.8.2\'s $\\rho^{\\,T-1}$: there, the base was the spectral radius of one fixed matrix $W_h$, baked in at initialisation and identical at every step; here, the base at each step is a separately learned number $f_j \\in (0,1)$, chosen by the network itself, differently at every timestep and for every example.']
    ], 'A large positive bias on the forget gate pushes $f_j$ close to 1 at every step by default — $\\sigma(2)\\approx0.88$, $\\sigma(4)\\approx0.98$ — which keeps the product $\\prod f_j$ from collapsing geometrically the way a fixed sub-1 spectral radius must. Crucially, this is not a hard-wired identity like a residual connection\'s constant 1; it is a soft, input-dependent decision the network can override when it genuinely wants to forget something, which is exactly the flexibility "always remember everything" would not give it.')}

<p>The output gate $o_t$ then decides how much of the (tanh-squashed) cell state actually surfaces in $h_t$, the value the rest of the network sees — so the cell state can retain information the current output does not need to expose yet. The <b>GRU</b> is a lighter variant that merges the forget and input gates into a single update gate, trading away a small amount of flexibility for fewer parameters, usually with little loss of quality in practice.</p>

<p><b>What you are looking at.</b> The lab below runs a small LSTM-style cell over a 40-step sequence you control. The faint vertical bars are the raw input $x_t$ at each step. The solid blue line is the cell state $c_t$ — the memory itself. The dashed green line is the forget gate's value at each step, and the dashed amber line is the input gate's.</p>

<p><b>What to do with it.</b> Start on the "one pulse, then silence" input pattern with the forget-gate bias at its default of 2, which corresponds to $\\sigma(2)\\approx0.88$ once the gate's other input is near zero. Watch the pulse arrive around step 3, the cell state jump in response as the input gate briefly opens, and then — this is the point — the cell state <i>holds</i> nearly that same value for the remaining 35-odd steps, decaying only slowly, with the forget gate sitting near its high default throughout. Now drag the forget-gate bias down to around $-3$, where $\\sigma(-3)\\approx0.047$, and rerun: the memory readout at step 30 collapses toward zero, because a forget gate that small multiplies the cell state down by roughly 95% at every single step.</p>

<p><b>The thing genuinely worth noticing.</b> Nothing about the network's <i>size</i> changed between those two runs — only one learned bias. That is the real content of this section: memory in an LSTM is not a passive side effect of the architecture the way it partly was for a plain RNN's hidden state; it is a quantity the network actively decides to keep or discard at every single timestep, and that decision is itself learned from data rather than fixed by an initialisation choice the way §3.8.2's spectral radius was.</p>

${H.lab('lstm', 'The gates, watched on a real sequence', 'A small LSTM-style cell run over a sequence you control. Watch the forget gate hold a value across many steps and the input gate decide when to overwrite — memory as an explicit, learnable decision.')}

<h2><span class="sn">3.8.4</span> Why they lost</h2>
<p>Gating fixed the gradient problem well enough for practical sequence lengths — an LSTM trained on sequences of a few hundred tokens works fine, and did for years. What gating could not touch is the shape of the computation itself: <mark>recurrence cannot be parallelised across time, because computing $h_t$ requires already having $h_{t-1}$ in hand.</mark> However many GPU cores are sitting idle, step 500 of a sequence simply cannot start before step 499 has finished, because it is a direct input to it — not a scheduling inconvenience to be optimised around, but a mathematical dependency built into the recurrence itself. Training on the trillions of tokens a modern language model sees requires keeping a GPU saturated with one enormous, embarrassingly parallel matrix multiply; a sequential loop over timesteps is close to the opposite of that, however cheap each individual step is on its own. Attention computes a representation for every position using operations that do not depend on one another the way consecutive RNN steps do, so an entire sequence can be processed as one large parallel matrix operation rather than a loop (§4.1). That is the whole story, and it is worth stating plainly because it runs against the instinct that the "better" architecture should win: the architecture that displaced RNNs did so on parallelism, not on having a more elegant model of language.</p>
${H.note('The wheel turns: state-space models (§4.8) are recurrences again, engineered so the recurrence is associative and therefore parallelisable by a scan. Same prior, different computational shape.')}

${H.probe([
      ['Why do RNN gradients vanish differently from feedforward ones?', 'The identical weight matrix $W_h$ is applied at every timestep, so §3.8.2\'s derivation shows the gradient scales like $\\rho^{T-1}$, the spectral radius of one fixed matrix raised to a power set by sequence length — you cannot fix it by adding parameters elsewhere, because the base of the exponent is fixed by $W_h$ alone, unlike a feedforward net where each layer has its own separate weights.'],
      ['What does the LSTM cell state buy?', 'A near-identity path across time, exactly like a residual connection\'s but softer: $\\partial c_t/\\partial c_{t-1}=f_t$ is a learned, entrywise gate value in $(0,1)$ rather than a hard-coded 1, so the network can push it close to 1 by default and override it when it genuinely needs to forget (§3.8.3).'],
      ['Why did transformers replace RNNs?', 'Parallelism across time, not modelling superiority — computing $h_t$ in an RNN strictly requires $h_{t-1}$ already computed, so training cannot saturate a GPU regardless of gating; attention gives every position a path length of 1 to every other position using operations that do not chain sequentially the same way.']
    ])}`,
    labs: {
      bptt: function (host) {
        const st = Viz.controls(host, [
          { k: 'len', label: 'sequence length', min: 5, max: 60, step: 1, value: 40, fmt: v => v },
          { k: 'rho', label: 'spectral radius of Wₕ', min: .5, max: 1.5, step: .01, value: .85, fmt: v => v.toFixed(2) },
          { k: 'act', label: 'activation', type: 'buttons', value: 'tanh', options: [{ v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' }] },
          { k: 'lstm', label: 'LSTM-style additive path (forget≈1)', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'g1', label: 'gradient reaching step 1', cls: 'key' }, { k: 'half', label: 'effective memory (10× decay)' },
          { k: 'verdict', label: 'verdict' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(31), n = 16;
            const W = Array.from({ length: n }, () => Array.from({ length: n }, () => R.normal(0, st.rho / Math.sqrt(n))));
            const act = st.act === 'tanh' ? Math.tanh : (z => Math.max(0, z));
            const dact = st.act === 'tanh' ? (a => 1 - a * a) : ((a, z) => z > 0 ? 1 : 0);
            let hh = Array.from({ length: n }, () => R.normal(0, .4));
            const hs = [hh], zs = [];
            for (let t = 0; t < st.len; t++) {
              const z = W.map(row => Num.dot(row, hh) + R.normal(0, .1));
              const nh = z.map(act);
              hh = st.lstm ? nh.map((v, i) => 0.92 * hs[t][i] + 0.25 * v) : nh;
              zs.push(z); hs.push(hh);
            }
            let delta = Array.from({ length: n }, () => R.normal(0, 1));
            const norms = [];
            for (let t = st.len - 1; t >= 0; t--) {
              const nd = new Array(n).fill(0);
              for (let k = 0; k < n; k++) {
                let s = 0;
                for (let j = 0; j < n; j++) s += W[j][k] * delta[j] * dact(hs[t + 1][j], zs[t][j]);
                nd[k] = st.lstm ? 0.92 * delta[k] + 0.25 * s : s;
              }
              delta = nd;
              norms.unshift(Math.sqrt(Num.dot(delta, delta)));
            }
            const logs = norms.map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [1, st.len], yd: [Math.min(-12, Math.min.apply(null, logs) - .5), Math.max(2, Math.max.apply(null, logs) + .5)] })
              .frame({ xlabel: 'time step (1 = earliest)', ylabel: 'log₁₀ ‖gradient‖ reaching this step' });
            P.clip(() => {
              P.line(norms.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: st.lstm ? T.green : T.blue, width: 2.6 });
              P.hline(logs[logs.length - 1] - 1, { color: T.faint, dash: [3, 3], label: '10× decay from the last step' });
            });
            let halfIdx = st.len;
            for (let i = norms.length - 1; i >= 0; i--) if (norms[i] < norms[norms.length - 1] / 10) { halfIdx = norms.length - i; break; }
            out({
              g1: norms[0].toExponential(2),
              half: halfIdx + ' steps',
              verdict: logs[0] < -8 ? 'vanished — no long-range learning' : logs[0] > 6 ? 'exploded — clip the gradient' : 'usable'
            });
          }
        });
        Viz.note(host, 'Turn on the additive path: the same recurrence now carries gradient across 40 steps. That single change — a near-identity route through time — is the LSTM’s entire contribution, and it is the same mechanism as a residual connection.');
      },

      lstm: function (host) {
        const st = Viz.controls(host, [
          { k: 'seq', label: 'input pattern', type: 'buttons', value: 'pulse', options: [{ v: 'pulse', t: 'one pulse, then silence' }, { v: 'noise', t: 'noisy stream' }, { v: 'switch', t: 'two pulses' }] },
          { k: 'forget', label: 'forget-gate bias', min: -3, max: 4, step: .1, value: 2, fmt: v => v.toFixed(1) },
          { k: 'input', label: 'input-gate bias', min: -4, max: 3, step: .1, value: 0, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'retain', label: 'memory retained at t=30', cls: 'key' }, { k: 'f', label: 'typical forget gate' }, { k: 'i', label: 'typical input gate' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(3), Tn = 40;
            const xs = [];
            for (let t = 0; t < Tn; t++) {
              if (st.seq === 'pulse') xs.push(t === 3 ? 1 : 0);
              else if (st.seq === 'switch') xs.push(t === 3 ? 1 : (t === 20 ? -1 : 0));
              else xs.push(R.normal(0, .35) + (t === 3 ? 1 : 0));
            }
            let c = 0, hh = 0;
            const cs = [], fs = [], is = [], os = [];
            xs.forEach(x => {
              const f = Num.sigmoid(st.forget + .5 * hh);
              const i = Num.sigmoid(st.input + 2.2 * x);
              const o = Num.sigmoid(.5 + hh);
              const g = Math.tanh(2.4 * x);
              c = f * c + i * g;
              hh = o * Math.tanh(c);
              cs.push(c); fs.push(f); is.push(i); os.push(o);
            });
            const P = Viz.plot(ctx, w, h, { xd: [0, Tn - 1], yd: [-1.2, 1.2] })
              .frame({ xlabel: 'time step', ylabel: 'value' });
            P.clip(() => {
              xs.forEach((v, t) => P.line([[t, 0], [t, v]], { color: T.faint, width: 3, alpha: .7 }));
              P.line(cs.map((v, t) => [t, v]), { color: T.blue, width: 2.8 });
              P.line(fs.map((v, t) => [t, v]), { color: T.green, width: 1.6, dash: [4, 3] });
              P.line(is.map((v, t) => [t, v]), { color: T.amber, width: 1.6, dash: [4, 3] });
              P.hline(0, { color: T.faint, dash: [2, 3], width: 1 });
            });
            out({
              retain: (Math.abs(cs[30]) / Math.max(1e-9, Math.abs(cs[4]))).toFixed(3),
              f: Num.mean(fs).toFixed(3), i: Num.mean(is).toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().faint, t: 'input xₜ' }, { c: Viz.theme().blue, t: 'cell state cₜ (the memory)' },
          { c: Viz.theme().green, t: 'forget gate' }, { c: Viz.theme().amber, t: 'input gate' }
        ]);
        Viz.note(host, 'With a high forget-gate bias the cell holds the pulse for the whole sequence; drop the bias to −1 and the memory decays within a few steps. The gate is a learned decision about what to keep, and that is the concept transformers replace with "attend to everything and let the softmax decide".');
      }
    },
    quiz: [
      {
        q: 'The decisive reason transformers replaced RNNs is…',
        options: ['RNNs cannot model long dependencies at all', 'recurrence cannot be parallelised across time, so it cannot saturate a GPU on trillions of tokens', 'RNNs have more parameters', 'attention is more accurate on short sequences'],
        answer: 1,
        why: 'Gating (§3.8.3) largely solved the gradient problem for practical sequence lengths — the LSTM cell state gives gradients a near-identity path across time, controllable by a learned forget gate. What gating cannot touch is that computing $h_t$ strictly requires $h_{t-1}$ already in hand, a mathematical dependency rather than an implementation inconvenience, so no amount of parallel hardware lets step 500 start before step 499 finishes. Option A is the tempting wrong answer because it sounds like the sort of thing that would matter, but a gated RNN genuinely can model long dependencies reasonably well within practical sequence lengths — it simply cannot be trained fast enough at the scale modern datasets demand, which is a computational-shape problem, not a modelling one.'
      },
      {
        q: 'The LSTM cell state helps because…',
        options: ['it is larger than the hidden state', 'its update is additive and gated, giving a near-identity gradient path across time', 'it uses ReLU', 'it removes the need for backpropagation through time'],
        answer: 1,
        why: 'Differentiating $c_t=f_t\\odot c_{t-1}+i_t\\odot g_t$ with respect to $c_{t-1}$ gives exactly $f_t$ (§3.8.3\'s derivation) — an entrywise multiplication with no weight matrix and no activation function in between, unlike the plain RNN\'s hidden-state update. Pushed close to 1 by a learned forget-gate bias, that local derivative lets gradients survive many timesteps largely intact, the same mechanism a residual connection uses across depth (§3.3.3), except here the "1" is a soft, learned, input-dependent value the network can deliberately lower when it wants to forget, rather than a constant baked into the architecture.'
      }
    ],
    cards: [
      { q: 'RNN gradient problem', a: 'BPTT multiplies the same $W_h^\\mathsf{T}\\mathrm{diag}(\\phi\')$ per step — geometric vanishing or explosion in sequence length.' },
      { q: 'LSTM in one line', a: 'Additive gated cell state: $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t$ — a residual path through time.' },
      { q: 'Why RNNs lost', a: 'Step t needs step t−1, so training cannot be parallelised across time; attention can.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.9 */
  ML.section({
    id: 'embeddings', track: 'deep', num: '3.9',
    title: 'Embeddings and representation learning',
    lede: 'The idea that carries from word2vec to modern retrieval: put meaning in a geometry, then let dot products do the reasoning.',
    rests: 'Rests directly on §0.2\'s dot product, norm and "columns are images of the basis vectors" rule.',
    html: `
<p>Suppose you need to feed a word into a neural network, which by §3.1 only knows how to consume lists of numbers. The direct-seeming choice is a <b>one-hot vector</b>: for a vocabulary of size $V$, represent word $i$ as a length-$V$ vector of zeros with a single 1 in position $i$. Every word gets its own dedicated slot, nothing is ambiguous, and it costs nothing to construct. Now ask §0.2's question of any two distinct one-hot vectors: how similar are they? Their dot product is always exactly 0 — every pair of distinct one-hot vectors is orthogonal, with no exceptions — and their distance is always exactly $\\sqrt2$, computed identically whether the two words are "cat" and "dog" or "cat" and "refrigerator". One-hot vectors carry no similarity structure whatsoever, by construction; every pair of words is equally, maximally unrelated to the geometry, regardless of what the words actually mean.</p>

<h2><span class="sn">3.9.1</span> From one-hot to dense</h2>
<p>An <b>embedding</b> replaces that rigid, meaningless geometry with a learned one. Instead of a length-$V$ vector with a single 1 in it, each word gets a dense vector of typically 256 to 4096 real numbers, trained so that words used in similar ways end up with vectors that are close together and words used differently end up far apart — turning "how similar are these two words?" from a question with one fixed, useless answer into a question the dot product from §0.2 can meaningfully answer.</p>
<p>Mechanically, the whole embedding table is just a matrix $E \\in \\mathbb{R}^{V\\times d}$, one row per vocabulary word, and "looking up word $i$" is nothing more exotic than multiplying $E$ by the one-hot vector for word $i$: by §0.2's boxed rule that the columns of a matrix are the images of the basis vectors, feeding in a one-hot vector always returns exactly one row (or column, depending on which side you multiply from) of the matrix, untouched. An embedding lookup is a linear layer in disguise, with a very particular and very sparse kind of input — which is why frameworks implement it as an indexed memory read rather than an actual matrix multiplication, even though the two are mathematically identical.</p>

<h2><span class="sn">3.9.2</span> How the geometry gets there</h2>
${H.table(['Method', 'Objective', 'What it captures'], [
      ['<b>word2vec (skip-gram)</b>', 'Predict context words from a centre word, with negative sampling', 'Distributional similarity — "words in similar contexts get similar vectors"'],
      ['<b>GloVe</b>', 'Factorise the log co-occurrence matrix', 'The same signal, as an explicit matrix factorisation'],
      ['<b>Contextual (BERT-style)</b>', 'Masked language modelling', 'One vector <i>per occurrence</i>, so "bank" differs by sentence'],
      ['<b>Sentence embeddings</b>', 'Contrastive: pull matched pairs together, push mismatched apart', 'Whole-passage meaning — the objective behind retrieval models (§5.1)'],
      ['<b>CLIP</b>', 'Contrastive across modalities', 'A shared image–text space (§4.16)']
    ])}
<p>Every row of that table looks different, but the underlying move is always the same: define a task that can only be solved well if similar things end up with similar vectors, then let gradient descent find the geometry that solves it. word2vec's task is predicting a word's neighbours; a contrastive sentence embedding's task is telling a matching pair of passages apart from a mismatched one. Neither method ever states a rule like "king is a kind of royalty" — the geometry that captures relations like that falls out as a side effect of solving a much more mundane prediction task at enormous scale.</p>

<p><b>What you are looking at.</b> The lab below is a small, hand-built 4-dimensional embedding space for twenty everyday words, projected down to its top two principal components (§2.10) for display, so what you see on screen is the closest flat picture of a shape that actually lives in more dimensions. Each dot is one word's vector; the controls let you query nearest neighbours by cosine similarity and run vector-arithmetic analogies directly on the underlying 4-dimensional vectors, not the 2-D projection.</p>

<p><b>What to do with it.</b> Query the nearest neighbours of "bank·money" and note it sits right next to "loan" — the two share a coordinate axis in this hand-built space, mimicking what a real trained embedding does automatically. Then query "bank·river": it lands nowhere near "loan" and instead sits beside "river", <i>despite starting with the identical word</i> "bank". This is polysemy resolved by giving each sense of an ambiguous word its own vector rather than forcing one vector to average across every possible meaning — precisely what §3.9.3 below discusses contextual embeddings doing automatically, sentence by sentence, rather than by hand as this lab does.</p>

<p><b>The thing genuinely worth noticing.</b> Toggle the "use cosine" switch off, so the readout reports the raw dot product instead of the cosine similarity, and watch the ranked list of nearest neighbours shift — not randomly, but systematically toward whichever vectors happen to have larger magnitude, exactly as §0.2's distinction between "alignment" and "alignment scaled by length" predicts. Direction and magnitude are answering two different questions, and which one a retrieval system should be asking is a real design decision, not a detail to default past — §5.1 returns to this exact choice at the scale of a production vector database.</p>

${H.lab('embed', 'An embedding space you can interrogate', 'A small hand-built space with real geometry. Search by cosine similarity, run analogies with vector arithmetic, and watch what happens when you normalise — the same operations a vector database performs at a billion-vector scale.')}

<h2><span class="sn">3.9.3</span> The arithmetic, and its limits</h2>
<p>The famous claim $\\text{king} - \\text{man} + \\text{woman} \\approx \\text{queen}$ is a statement about vector arithmetic: subtracting one vector and adding another, then asking which word's vector the result lands closest to. It works when a training objective happens to arrange certain relations as approximately parallel shifts across many pairs — the "royal" shift from man to king and from woman to queen pointing in nearly the same direction, so that subtracting off "royal-maleness" and adding "royal-femaleness" lands you near "queen" purely as vector algebra.</p>

${H.worked('the analogy, computed exactly on this lab\'s toy vectors', `<p>This lab\'s hand-built vectors give $\\text{king}=(0.9,0.8,0.9,0.1)$, $\\text{man}=(0.2,0.3,0.9,0.1)$, $\\text{woman}=(0.2,0.3,0.1,0.9)$, $\\text{queen}=(0.9,0.8,0.1,0.9)$. Compute the target directly: $\\text{king}-\\text{man}+\\text{woman} = (0.9-0.2+0.2,\\; 0.8-0.3+0.3,\\; 0.9-0.9+0.1,\\; 0.1-0.1+0.9) = (0.9,\\,0.8,\\,0.1,\\,0.9)$ — which is exactly $\\text{queen}$\'s vector, coordinate for coordinate, because this toy space was built so that the "royal" shift and the "gender" shift are perfectly parallel by construction. Cosine similarity between the target and queen is exactly 1. The nearest real competitor is princess, at cosine similarity $0.995$ — close, because "prince/princess" shares the same two structural axes as "king/queen" in this space, but not identical, because the toy vectors were not built to make every royal pair coincide exactly.</p>`)}

<p>Real trained embeddings show the same pattern far more approximately. It is a genuine property and a routinely over-sold one: it holds reasonably well for frequent, well-attested relations like this one, and fails outright for rarer or more irregular relations, where no consistent "shift direction" exists across enough examples for training to have discovered one. Standard evaluations of this arithmetic also exclude the query words themselves from the candidate answers, which is a reasonable thing to do — otherwise a word summed directly into the arithmetic, like "woman" above, can appear high in the results purely because it was one of the ingredients, not because the geometry discovered anything — but excluding them does quietly flatter the reported accuracy relative to an unconstrained search.</p>

<h2><span class="sn">3.9.4</span> What matters in practice</h2>
<ul>
<li><b>Cosine or dot product?</b> Cosine if you normalise (pure direction); dot product if magnitude carries meaning, such as term importance. Whichever the model was trained with is the one to use (§5.1).</li>
<li><b>Dimensionality</b> is a memory/quality trade: §5.1 sizes an index and shows why Matryoshka embeddings — trained so that a truncated prefix still works — let you pick the trade after training.</li>
<li><b>Tied embeddings</b>: sharing the input embedding with the output projection saves parameters and usually costs nothing (§4.5), which matters most at small scale where the vocabulary dominates the parameter count.</li>
<li><b>Bias is inherited.</b> Embeddings reproduce the associations in their training corpus, including harmful ones; that is a data property, and mitigating it in the geometry is a research problem rather than a config flag.</li>
</ul>

${H.pitfall(`<p>Two embeddings from two different models are not comparable, even when they happen to have the same dimension $d$. There is nothing that pins "the direction meaning royalty" to the same coordinates across two independently trained models — each training run finds <i>some</i> geometry that solves its objective, not a canonical one, so the same word can land at completely different coordinates in two different embedding spaces. Mixing vectors from two models in the same similarity search, or upgrading a production embedder without re-embedding the whole corpus, silently produces meaningless comparisons rather than an error, because a dot product always returns some number — it just stops meaning anything. §5.1 treats swapping an embedding model as a full migration for exactly this reason.</p>`)}

${H.probe([
      ['Why not one-hot?', 'Every pair of distinct one-hot vectors has dot product exactly 0 and distance exactly $\\sqrt2$, regardless of what the words mean, so no similarity structure exists at all; an embedding replaces that rigid geometry with one learned so that similar usage lands close together (§3.9.1).'],
      ['Cosine or dot product?', 'Match whatever the model was trained with; cosine for normalised direction-only similarity, dot product when magnitude means something. §3.9.2\'s lab shows the ranked list of nearest neighbours actually change when you switch between them, not just the numbers attached to it.'],
      ['What does "contextual embedding" add?', 'A different vector per occurrence rather than one vector per vocabulary entry, so polysemy — "bank" the financial institution versus "bank" the riverside — is resolved by the sentence itself rather than averaged into one compromise vector that serves neither sense well.']
    ])}`,
    labs: {
      embed: function (host) {
        // hand-built 2-D-ish semantic space, then lifted to 8-D with structure
        const words = {
          king: [0.9, 0.8, 0.9, 0.1], queen: [0.9, 0.8, 0.1, 0.9], man: [0.2, 0.3, 0.9, 0.1], woman: [0.2, 0.3, 0.1, 0.9],
          prince: [0.8, 0.6, 0.85, 0.15], princess: [0.8, 0.6, 0.15, 0.85], boy: [0.15, 0.15, 0.9, 0.1], girl: [0.15, 0.15, 0.1, 0.9],
          london: [0.1, 0.9, 0.5, 0.5], paris: [0.12, 0.92, 0.5, 0.5], england: [0.05, 0.75, 0.5, 0.5], france: [0.07, 0.77, 0.5, 0.5],
          bank_money: [0.6, 0.15, 0.5, 0.5], bank_river: [0.15, 0.2, 0.5, 0.5], loan: [0.62, 0.12, 0.5, 0.5], river: [0.13, 0.22, 0.5, 0.5],
          cat: [0.3, 0.05, 0.5, 0.45], dog: [0.32, 0.06, 0.55, 0.45], kitten: [0.28, 0.04, 0.5, 0.47], puppy: [0.3, 0.05, 0.55, 0.47]
        };
        const names = Object.keys(words);
        const st = Viz.controls(host, [
          { k: 'query', label: 'nearest neighbours of', type: 'select', value: 'king', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'a', label: 'analogy: a', type: 'select', value: 'king', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'b', label: '− b', type: 'select', value: 'man', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'c', label: '+ c', type: 'select', value: 'woman', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'norm', label: 'use cosine (normalise)', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nn', label: 'nearest neighbour', cls: 'key' }, { k: 'sim', label: 'similarity' },
          { k: 'ana', label: 'analogy result', cls: 'good' }, { k: 'anasim', label: 'analogy similarity' }
        ]);
        function sim(u, v) {
          const d = Num.dot(u, v);
          return st.norm ? d / ((Num.norm(u) * Num.norm(v)) || 1) : d;
        }
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            // project to 2-D with PCA for display
            const mat = names.map(n => words[n]);
            const pc = Num.pca(mat);
            const proj = mat.map(v => {
              const c = v.map((x, i) => x - pc.mean[i]);
              return [Num.dot(c, pc.vectors[0]), Num.dot(c, pc.vectors[1])];
            });
            const xs = proj.map(p => p[0]), ys = proj.map(p => p[1]);
            const pad = .25;
            const P = Viz.plot(ctx, w, h, {
              xd: [Math.min.apply(null, xs) - pad, Math.max.apply(null, xs) + pad],
              yd: [Math.min.apply(null, ys) - pad, Math.max.apply(null, ys) + pad]
            }).frame({ xlabel: 'PC1', ylabel: 'PC2' });
            const qi = names.indexOf(st.query);
            const sims = names.map((n, i) => ({ n: n, s: sim(words[st.query], words[n]), i: i })).filter(x => x.n !== st.query).sort((a, b) => b.s - a.s);
            // analogy
            const target = words[st.a].map((v, i) => v - words[st.b][i] + words[st.c][i]);
            const anas = names.map(n => ({ n: n, s: sim(target, words[n]) }))
              .filter(x => [st.a, st.b, st.c].indexOf(x.n) < 0).sort((a, b) => b.s - a.s);
            P.clip(() => {
              proj.forEach((p, i) => {
                const isQ = i === qi;
                const isTop = sims.slice(0, 3).some(s => s.i === i);
                P.dots([p], { r: isQ ? 6 : isTop ? 4.6 : 3, color: isQ ? T.red : isTop ? T.blue : T.faint, stroke: true });
                P.text(p[0], p[1], ' ' + names[i].replace('_', '·'), { color: isQ ? T.red : isTop ? T.text : T.faint, font: (isQ ? 'bold ' : '') + '10px ui-sans-serif' });
              });
              // analogy arrows
              const ia = names.indexOf(st.a), ib = names.indexOf(st.b), ic = names.indexOf(st.c), ir = names.indexOf(anas[0].n);
              P.arrow(proj[ib][0], proj[ib][1], proj[ia][0], proj[ia][1], { color: T.green, width: 1.8 });
              if (ir >= 0) P.arrow(proj[ic][0], proj[ic][1], proj[ir][0], proj[ir][1], { color: T.amber, width: 1.8 });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText(st.a + ' − ' + st.b + ' + ' + st.c + '  ≈  ' + anas[0].n + '  (' + anas[0].s.toFixed(3) + ')', 46, 8);
            ctx.fillStyle = T.faint;
            ctx.fillText('top matches: ' + sims.slice(0, 3).map(s => s.n + ' ' + s.s.toFixed(2)).join(' · '), 46, 24);
            out({
              nn: sims[0].n.replace('_', ' · '), sim: sims[0].s.toFixed(3),
              ana: anas[0].n.replace('_', ' · '), anasim: anas[0].s.toFixed(3)
            });
          }
        });
        Viz.note(host, 'Note "bank·money" sits next to "loan" while "bank·river" sits next to "river" — polysemy resolved by giving each sense its own vector, which is exactly what a contextual model does automatically. The analogy evaluation excludes the three query words from the candidate list — not because king would otherwise steal first place (queen was built to match the target exactly, and wins regardless), but because woman, one of the three words summed directly into the arithmetic, would otherwise sit in third place purely by construction, cluttering the result with an "answer" that shows nothing about the geometry.');
      }
    },
    quiz: [
      {
        q: 'Why are dense embeddings preferred to one-hot vectors?',
        options: ['They are easier to store', 'One-hot vectors make every pair equidistant, so no similarity structure exists', 'They avoid the need for a vocabulary', 'They are always more accurate'],
        answer: 1,
        why: 'The dot product of any two distinct one-hot vectors is exactly 0 and their distance is exactly $\\sqrt2$, computed identically no matter which two words are compared (§3.9.1) — similarity has to live somewhere in the geometry for a dot product or a distance to mean anything, and one-hot vectors were never built with any such structure. An embedding replaces that fixed, meaningless geometry with a learned one where usage determines distance. Option A is backwards if anything: a one-hot vector needs no storage at all (an index suffices), while a dense embedding genuinely does cost $d$ numbers per word — the trade is for meaningful geometry, not for convenience.'
      },
      {
        q: 'You switch an embedding model but keep the same vector index. What breaks?',
        options: ['Nothing', 'Everything — the geometry is different, so the whole corpus must be re-embedded and re-indexed', 'Only the reranker', 'Only long documents'],
        answer: 1,
        why: 'Nothing pins "the direction meaning royalty" or any other concept to the same coordinates across two independently trained models — each training run finds some geometry that solves its own objective, not a canonical one shared with every other model of the same dimension. A dot product between a vector from the old model and a vector from the new one still returns a number; it simply stops meaning anything, silently rather than with an error. That silence is what makes it dangerous, and why §5.1 treats an embedder change as a full re-embedding migration of the whole corpus rather than a configuration flag to flip.'
      }
    ],
    cards: [
      { q: 'Embedding matrix', a: '$E\\in\\mathbb{R}^{V\\times d}$ — a lookup table, equivalently a linear layer applied to a one-hot vector.' },
      { q: 'Cosine vs dot product', a: 'Cosine for normalised direction-only similarity; dot product when magnitude carries meaning. Match the training objective.' },
      { q: 'Contextual embeddings', a: 'One vector per occurrence rather than per type, so polysemy is resolved by context.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.10 */
  ML.section({
    id: 'part3-recall', track: 'deep', num: '3.15',
    title: 'Rapid recall — Part 3',
    lede: 'The neural-network essentials, in the form you would be asked for them.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', 'Composed linear layers collapse; the nonlinearity is what makes depth mean anything.', '<a href="#/nn-fundamentals">3.1</a>'],
      ['2', 'Backprop = chain rule right-to-left with cached activations; full gradient ≈ one forward pass.', '<a href="#/backprop">3.2</a>'],
      ['3', 'Worked backprop: L = 0.125 → 0.095 after one step at η = 0.1.', '<a href="#/backprop">3.2</a>'],
      ['4', 'Sigmoid derivative ≤ 0.25 ⇒ vanishing gradients; ReLU passes 1.', '<a href="#/activations">3.3</a>'],
      ['5', 'Residuals add a path of derivative exactly 1 — why 100-layer stacks train.', '<a href="#/activations">3.3</a>'],
      ['6', 'He init 2/n_in for ReLU; Xavier for symmetric activations; never all-zero.', '<a href="#/initialisation">3.4</a>'],
      ['7', 'AdamW decouples weight decay from the adaptive denominator — the modern default.', '<a href="#/optimisers">3.5</a>'],
      ['8', 'Warmup then cosine, or WSD to allow branching mid-run.', '<a href="#/optimisers">3.5</a>'],
      ['9', 'LayerNorm/RMSNorm for transformers; BatchNorm depends on the batch.', '<a href="#/normalisation">3.6</a>'],
      ['10', 'Dropout = training an ensemble of thinned sub-networks.', '<a href="#/normalisation">3.6</a>'],
      ['11', 'Conv output $\\lfloor(H+2p-k)/s\\rfloor+1$; two 3×3 beat one 5×5.', '<a href="#/cnn">3.7</a>'],
      ['12', 'RNNs lost on parallelism, not on modelling; LSTM’s cell state is a residual through time.', '<a href="#/rnn">3.8</a>'],
      ['13', 'Embeddings put meaning in a geometry; match the trained similarity metric.', '<a href="#/embeddings">3.9</a>']
    ])}

${H.lab('drill3', 'Part 3 drill', 'Thirteen prompts, shuffled.')}`,
    labs: {
      drill3: function (host) {
        const cards = [
          ['Why is a nonlinearity necessary?', 'Composed linear maps are linear; without one, depth adds no capacity.'],
          ['Backprop in one sentence.', 'Chain rule evaluated right-to-left with cached forward activations.'],
          ['Why reverse-mode rather than forward-mode?', 'One scalar output, many parameters — reverse costs one pass per output.'],
          ['Max derivative of sigmoid, and the consequence.', '0.25; ten layers scale the gradient by at most $10^{-6}$.'],
          ['What do residual connections do to the gradient?', 'Add a path with local derivative exactly 1, so it reaches layer 1 unattenuated.'],
          ['He vs Xavier initialisation.', '$2/n_{in}$ for ReLU (half the inputs are zeroed) vs $2/(n_{in}+n_{out})$ for symmetric activations.'],
          ['Adam vs AdamW.', 'AdamW applies weight decay outside the adaptive denominator, so decay actually decays.'],
          ['Why warmup?', 'Adam’s second-moment estimate is unreliable early; large first steps can be unrecoverable.'],
          ['WSD schedule and its advantage.', 'Warmup, stable, sharp decay — no need to fix total steps up front; branchable.'],
          ['LayerNorm vs BatchNorm.', 'Per-example features vs per-channel over the batch; transformers use LayerNorm/RMSNorm.'],
          ['Why does dropout regularise?', 'It trains an ensemble of thinned sub-networks and prevents co-adaptation.'],
          ['Convolution output shape.', '$\\lfloor(H+2p-k)/s\\rfloor+1$.'],
          ['Receptive field of L stacked 3×3 convs.', '$2L+1$ at stride 1.'],
          ['Why did transformers replace RNNs?', 'Recurrence cannot parallelise across time; attention gives O(1) path length and one big matmul.'],
          ['What is the LSTM cell state for?', 'An additive, gated memory — a residual path through time.'],
          ['Cosine or dot product for embeddings?', 'Whichever the model was trained with; cosine after normalisation.']
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
        host.appendChild(nav);
        draw();
      }
    },
    quiz: [
      {
        q: 'Which two ideas from Part 3 reappear unchanged inside every transformer block?',
        options: ['Dropout and BatchNorm', 'Residual connections and normalisation', 'Convolution and pooling', 'Momentum and warmup'],
        answer: 1,
        why: 'A transformer block is attention and an FFN wrapped in exactly those two stabilisers — §4.5 draws it with shapes.'
      }
    ]
  });
})();
