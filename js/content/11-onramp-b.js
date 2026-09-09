/* ============================================================
   PART 0 — Start here (0.6 – 0.8): notation, matrix calculus,
   and one complete supervised workflow end to end.
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 0.6 */
  ML.section({
    id: 'notation', track: 'start', num: '0.6', level: 1,
    title: 'Reading the notation',
    lede: 'Most people who "can’t do the maths" can do the maths. What stops them is that nobody ever read a formula out loud for them, left to right, saying what each mark is for. This section does that, and then makes you do it.',
    related: ['matrix-calculus', 'probability-basics', 'formulas'],
    html: `
${H.tldr([
      'A formula is a compressed program. Sums are <code>for</code> loops, subscripts are indices, $\\mathbb{E}$ is a weighted average, and $\\arg\\min$ returns <i>the input</i>, not the value.',
      'Read every formula in three passes: <b>shapes</b> (what is a scalar, a vector, a matrix?), then <b>the loop</b> (what is being summed, over what index?), then <b>the story</b> (what would make this big or small?).',
      'Every convention is the answer to a specific confusion. Learn what a mark is defending against and you stop having to memorise it.',
      'The literature genuinely disagrees with itself about notation. Fix the shapes first and most of the disagreements stop mattering.'
    ])}

<p>Start with something you can already read. Here are four lines of Python, and you need no preparation at all to follow them:</p>

${H.code(`total = 0.0
for i in range(n):
    total += (y[i] - yhat[i]) ** 2
mse = total / n`)}

<p>Nothing there is mysterious. You loop over the examples, take the gap between the truth and the prediction for each one, square it, add it to a running total, and divide by how many there were. If someone asked you to explain that code to a colleague you would do it without thinking.</p>

<p>Now here is the same instruction, written the way a textbook or a paper would write it:</p>

$$\\mathcal{L}(w) = \\frac{1}{n}\\sum_{i=1}^{n}\\left(y_i - \\hat{y}_i\\right)^2$$

<p>This is not a harder idea. It is the <i>same</i> idea, and it computes the same number. Every piece of the code has a counterpart in the formula, and nothing has been added. What has changed is the packaging: four lines and about sixty characters have become one line and about twenty marks, and the machinery that made the code readable — the word <code>for</code>, the brackets on <code>y[i]</code>, the explicit variable called <code>total</code> — has been folded away into marks that take up no space.</p>

<p>So why would anyone prefer the second form? Because the first one cannot be argued about. You cannot differentiate a <code>for</code> loop, and differentiating this loss with respect to $w$ is precisely what training a model requires (§0.7). You cannot see at a glance that the loop body is symmetric in a way that lets you swap two terms. You cannot substitute one expression into another and simplify. <b>A formula is a program written in a notation that is designed to be manipulated on paper</b>, which is a completely different design goal from being executed by a machine, and it is why the two look so different.</p>

<p>Here is where your existing reading habit runs out. Code is read top to bottom, one statement at a time, in the order things happen. A formula is not laid out in that order at all. The $\\sum$ sits to the left of the thing it sums, the range of the loop is written above and below it in small type, and the thing that <i>varies</i> across the loop is signalled only by a subscript. Reading it the way you read code — left to right, taking each mark as it comes — produces nonsense.</p>

<p>What you need instead is a small, explicit procedure for decoding a formula, plus an understanding of why each convention is the way it is. Almost nobody is ever taught either. That is the entire content of this section, and it is the reason many people who believe they cannot do the mathematics can in fact do the mathematics perfectly well.</p>

<h2><span class="sn">0.6.1</span> A formula is a sentence, so read it out loud</h2>

<p>The single most useful habit is to say a formula aloud in English before doing anything else with it. The mean squared error above becomes:</p>

<p><i>"The loss, which depends on the parameters $w$, is the average over all $n$ examples of the squared gap between the true answer and our prediction."</i></p>

<p>Notice that the sentence is not much longer than the formula and contains exactly the same information. Every mark did a specific job, and here they are laid out one at a time against their equivalent in code:</p>

${H.table(['Mark', 'Out loud', 'In code'], [
      ['$\\mathcal{L}(w)$', 'a number that depends on $w$ — the thing we push down', '<code>def loss(w): ...</code>'],
      ['$\\frac{1}{n}\\sum_{i=1}^{n}$', 'average over the examples', '<code>np.mean(...)</code>'],
      ['$i$', 'which example — an index, never a quantity', 'the loop variable'],
      ['$n$', 'how many examples there are', '<code>len(y)</code>'],
      ['$y_i$', 'the true label of example $i$', '<code>y[i]</code>'],
      ['$\\hat{y}_i$', 'the <i>estimate</i> — the hat always means "our guess at"', '<code>yhat[i]</code>'],
      ['$(\\cdot)^2$', 'punish both directions, punish big errors more', '<code>** 2</code>']
    ])}

<p>Three of those rows are doing the heavy lifting, and between them they decode most of the machine learning literature.</p>

${H.key('The hat means estimate, the subscript means index, and $\\sum$ with $1/n$ means average. Those three conventions decode most of machine learning.')}

<p>Work the formula on four actual numbers so that the marks and the arithmetic sit side by side. Suppose the true labels are $y = (3,\\,1,\\,4,\\,2)$ and the model predicted $\\hat y = (2.5,\\,1.4,\\,3.2,\\,2.9)$, so $n = 4$. The gaps $y_i - \\hat y_i$ are $0.5$, $-0.4$, $0.8$ and $-0.9$. Squaring removes the signs and gives $0.25$, $0.16$, $0.64$ and $0.81$. The $\\sum$ adds those four numbers to make $1.86$, and the $\\tfrac{1}{4}$ in front divides by four to give $\\mathcal{L} = 0.465$. That is the whole formula, executed by hand, and it took one line of arithmetic per mark.</p>

<p>One feature of that formula deserves a paragraph of its own, because it is the first genuine piece of compression and it confuses nearly everyone the first time. Look at the left-hand side: $\\mathcal{L}(w)$ announces that the loss depends on the parameters $w$. Now look at the right-hand side. <b>There is no $w$ anywhere in it.</b></p>

<p>The $w$ is hiding inside $\\hat y_i$. The prediction is really $f_w(x_i)$, the model applied to the $i$-th input, and it changes the moment you change a parameter. Writing $\\hat y_i$ instead of $f_w(x_i)$ is a deliberate abbreviation: it keeps the formula about the <i>shape</i> of the loss rather than about the internals of the model, which is exactly what you want when the model is a hundred-layer network. But the dependence is completely real, and it is the only reason training is possible at all. Whenever you meet a hat, ask what it is a function of, because that is usually the thing being optimised.</p>

${H.history(`<p>The marks in that formula did not arrive together, and knowing where they came from explains why they do not fit together perfectly.</p>
<p>The summation sign $\\sum$ is Euler's, introduced in the middle of the eighteenth century; before it, sums were written out with dots and an "and so on". The $d$ of $dy/dx$ is Leibniz's, from the 1670s. Newton had a rival notation, a dot over the variable, and it lost — not because it was wrong but because Leibniz's version composes: you can write $dy/dx \\cdot dx/dt$ and watch the symbols cancel, which is the chain rule doing your bookkeeping for you. Notation that helps you calculate beats notation that merely records.</p>
<p>Vectors, the dot product and the modern use of bold type were carved out by Gibbs and Heaviside in the 1880s, extracted from Hamilton's quaternions over the loud objections of people who thought quaternions were the more beautiful object. The hat for an estimate comes later still, out of twentieth-century statistics, where the founding distinction of the entire field is between a parameter you will never observe and your guess at it.</p>
<p>Machine learning inherited from all three traditions at once — statistics, linear algebra and numerical computing — and reconciled none of them. That is not a historical curiosity. It is the direct cause of every clash in the table in §0.6.5, and it is why you will meet the same letter meaning three different things in one paper.</p>`)}

<h2><span class="sn">0.6.2</span> Meet the dissector</h2>

<p><b>What you are looking at.</b> The lab below shows one formula, typeset in full at the top, and underneath it a panel that isolates a single piece of that formula at a time. The panel has three lines: the fragment itself in mathematics, one sentence of plain English saying what that fragment is <i>for</i>, and — in monospaced blue — the actual value that fragment takes on a small set of toy numbers. For the mean squared error those numbers are the $y = (3,1,4,2)$ and $\\hat y = (2.5,1.4,3.2,2.9)$ worked through above, so the values you will see are exactly the ones you just computed by hand.</p>

<p><b>What to do with it.</b> The dropdown chooses between five formulas: mean squared error, softmax, the gradient descent update, Bayes' theorem and scaled dot-product attention. Press <b>Play</b> and the panel advances through the pieces on its own, or use <b>Step</b> and <b>Back</b> to move one piece at a time, which is the better way to read it the first time. Start with mean squared error, because you already know the answer, and check the blue line at each step against your own arithmetic. Then choose attention, which is the formula from §4.3 that people find most intimidating, and watch it decompose into five ordinary steps.</p>

<p><b>The thing genuinely worth noticing.</b> Every one of these five formulas, including attention, is a short sequence of operations you could carry out with a calculator. Nothing appears in a step that was not present in the one before it. What makes a formula feel impenetrable is never the individual operations; it is that they are all printed at once, on a single line, with no indication of what to do first. Reading one piece at a time is not a beginner's crutch that you will later outgrow — it is what everyone does, silently, and the only difference with practice is how fast the pieces go by.</p>

${H.lab('dissect', 'The formula dissector', 'Step through a formula one symbol at a time. Each step highlights a piece, says what it is for in plain English, and evaluates that piece on the toy numbers shown — so the abstract mark and the concrete number are on the screen together.')}

<h2><span class="sn">0.6.3</span> Every convention is the answer to a confusion</h2>

<p>It is tempting to treat notation as a list of arbitrary decorations to be memorised. It is much easier than that, because each convention was invented to prevent a specific ambiguity, and once you know which ambiguity, the convention stops needing to be remembered. Here are the ones that carry real weight.</p>

<h3>Subscripts: five different jobs, one mark</h3>

<p>A subscript is a label attached to the bottom right of a symbol, and its job is to say <i>which one</i>. The confusion it prevents is having to invent a new letter for every item in a list. Without subscripts, a dataset of a thousand examples would need a thousand names.</p>

<p>The complication is that the same mark is used for five distinguishable purposes, and telling them apart is entirely a matter of context:</p>

${H.table(['Subscript', 'What the subscript is', 'Example'], [
      ['$x_i$', 'which example, in a dataset of $n$', '$i$ runs from 1 to $n$'],
      ['$x_j$', 'which feature, within one example', '$j$ runs from 1 to $d$'],
      ['$x_t$', 'which time step, in a sequence', '$t$ runs along a sentence or a trajectory'],
      ['$d_k$', 'a <b>name</b>, not an index — "the $k$ dimension"', '$d_k$ in attention is one fixed number (§4.3)'],
      ['$\\nabla_w,\\ \\mathbb{E}_{x\\sim p}$', '<b>with respect to</b> — not an index at all', 'the subscript says what varies, not which item']
    ])}

${H.pitfall(`<p>The last two rows are where people come unstuck. When you see $\\nabla_w \\mathcal{L}$, there is a strong pull to read the $w$ as "the $w$-th something", by analogy with $x_i$. It is not. The subscript on a gradient names the variable you are differentiating with respect to, and $\\nabla_w \\mathcal{L}$ is a single object with the same shape as the whole of $w$ (§0.7).</p>
<p>The same trap sits under $\\mathbb{E}_{x \\sim p}[f(x)]$: the subscript is announcing which distribution the average is taken over, not selecting an item. And under $\\|x\\|_2$, where the 2 names <i>which</i> norm — the Euclidean one — rather than indexing anything. A useful test: if replacing the subscript with the number 7 produces a sentence that makes sense ("the seventh example"), it is an index. If it does not, it is a name.</p>`)}

<h3>Superscripts, and why some of them wear parentheses</h3>

<p>A superscript sits at the top right, and by default it means exponentiation: $x^2$ is $x$ times $x$. That creates an immediate problem the moment you also want to number things at the top right, which is exactly what deep learning wants to do — one index for which example and another for which layer.</p>

<p>The solution is brackets. <b>A superscript in round or square brackets is a label, never a power.</b> So $x^{(i)}$ is the $i$-th training example, $W^{[3]}$ is the weight matrix of layer 3, and $x^2$ remains an honest square. The brackets exist purely to disarm the exponent reading, which is why removing them changes the meaning completely.</p>

<p>The remaining superscripts are a small fixed vocabulary: $A^{\\top}$ is the transpose (§0.2), $A^{-1}$ is the inverse, $w^{\\star}$ is the optimal value of $w$, and $x^{\\prime}$ is either a derivative in scalar calculus or simply "a different one of the same kind" depending on which subject you are in.</p>

<h3>Capitals for matrices, lowercase for vectors and scalars</h3>

<p>This one is a shape convention and it is doing more work than it appears to. When you see $Xw$ you already know, before reading anything else, that a grid is being applied to a list. When you see $x^{\\top}w$ you know two lists are being collapsed into one number. The case of the letter is carrying the shape, so you can check whether an expression is even meaningful without knowing what any of it stands for.</p>

<p>Bold type is an older convention doing the same job — $\\mathbf{x}$ for a vector, $x$ for a scalar — and it is quietly disappearing from machine learning papers, partly because it is invisible in code and unreliable on a whiteboard. This site does not use bold for vectors. Its absence is not a claim that everything is a scalar.</p>

<h3>Hats, bars, tildes and stars: the family of marks that mean "not the real thing"</h3>

<p>These four are worth learning as a group, because they all answer the same question — <i>how does this quantity relate to the true one?</i> — and their differences are precise:</p>

${H.table(['Mark', 'Name', 'Meaning', 'Concrete example'], [
      ['$\\hat{y}$', 'hat', 'an <b>estimate</b> produced from data', '$\\hat y$ is the prediction; $y$ is what actually happened'],
      ['$\\bar{x}$', 'bar', 'an <b>average</b> over a sample', '$\\bar x = \\frac{1}{n}\\sum_i x_i$, the sample mean'],
      ['$\\tilde{x}$', 'tilde', 'a <b>modified</b> version — noisy, approximate or augmented', '$\\tilde x$ is the corrupted input in a denoising model'],
      ['$w^{\\star}$', 'star', 'the <b>optimal</b> value, usually unattainable', '$w^{\\star} = \\arg\\min_w \\mathcal{L}(w)$']
    ])}

<p>The distinction between $\\hat y$ and $y$ is the one that matters most, because it is the whole of supervised learning written in two marks. The bar is a special case of the hat: a sample mean is an estimate of a population mean, and some authors will write $\\hat\\mu$ where others write $\\bar x$ for exactly the same number.</p>

<h3>Random variables and their realisations</h3>

<p>This is the convention that most often separates a reader who follows a probability argument from one who does not, and it is almost never stated explicitly.</p>

<p>Roll a die. There are two quite different objects in that sentence. One is the <i>mechanism</i>: a process with six possible outcomes, each with probability one sixth, which has an average of 3.5 even though no die ever shows 3.5. The other is the <i>outcome</i>: the 4 that came up this time, an ordinary number sitting on the table. Statistics writes the mechanism with a capital, $X$, and the outcome with a lowercase, $x$.</p>

<p>Once you see it, the notation $P(X = x)$ stops being redundant. It reads "the probability that the mechanism $X$ produces the particular value $x$", and the two letters are doing genuinely different jobs. So does the distinction between $\\mathbb{E}[X]$, which is a fixed number, and $\\mathbb{E}[X \\mid Y]$, which is <i>itself a random variable</i> because it still depends on which value $Y$ takes.</p>

${H.analogy(`<p>Think of $X$ as a slot machine and $x$ as a payout. The machine has properties that no individual payout has: an average return, a spread, a shape. The payout has properties the machine does not: it is a single number, it already happened, and it is either large or small. Confusing them is the same mistake as saying "the slot machine is £40".</p>
<p>Nearly every probability confusion in machine learning is a version of this mix-up. A model outputs $\\hat p = 0.7$ — is that a number or a random variable? It is a number, given this input. But across the distribution of inputs it is a random variable with a mean and a spread, and that is why calibration (§2.12) is a question about the whole distribution of outputs rather than about any single prediction.</p>`)}

${H.flag('Be warned that machine learning papers break this convention constantly, using lowercase for both, and reserving capitals for matrices instead. The rule that still works is contextual: <b>if a symbol appears under an $\\mathbb{E}$, after a $\\sim$, or inside a $P(\\cdot)$, it is being treated as random, whatever case it is written in.</b>')}

<h2><span class="sn">0.6.4</span> The decoder ring</h2>

<p>What follows is the full working vocabulary. Skim it now to know what is in it; come back whenever a line stops you. Reading it end to end is not a good use of your time, because these marks stick when you meet them doing a job, not when you meet them in a list.</p>

${H.table(['Symbol', 'Name', 'What it does', 'Read it as'], [
      ['$\\sum_{i=1}^{n} x_i$', 'sum', 'adds a list up', '"add up all the $x$s"'],
      ['$\\prod_{i} p_i$', 'product', 'multiplies a list — usually independent probabilities', '"multiply all the $p$s"'],
      ['$\\mathbb{E}[X]$', 'expectation', 'the long-run average of a random quantity', '"on average, $X$ is…"'],
      ['$\\mathbb{E}_{x\\sim p}[f(x)]$', 'expectation under $p$', 'average of $f$ weighting each $x$ by how likely $p$ says it is', '"average $f$, as $p$ sees the world"'],
      ['$\\mathrm{Var}(X)$', 'variance', 'average squared distance from the mean', '"how spread out"'],
      ['$P(A\\mid B)$', 'conditional', 'probability of $A$ <i>in the world where $B$ happened</i>', '"$A$ given $B$"'],
      ['$\\arg\\min_{w} f(w)$', 'argmin', 'the $w$ that makes $f$ smallest — <b>not</b> the smallest value', '"whichever $w$ wins"'],
      ['$\\nabla_w f$', 'gradient', 'vector of partial derivatives — the uphill direction', '"which way is up, in every parameter at once"'],
      ['$\\partial f/\\partial x$', 'partial derivative', 'slope in $x$ holding everything else still', '"if I nudge only $x$…"'],
      ['$x^\\top y$', 'dot product', 'sum of elementwise products; one number', '"how aligned are these"'],
      ['$A \\odot B$', 'Hadamard', 'elementwise multiply, same shape out', '"multiply cell by cell"'],
      ['$\\|x\\|_2$', 'L2 norm', 'length: $\\sqrt{\\sum x_i^2}$', '"how big"'],
      ['$\\|x\\|_1$', 'L1 norm', '$\\sum|x_i|$ — the one that makes things sparse (§2.3)', '"total absolute size"'],
      ['$\\propto$', 'proportional to', 'equal up to a constant we do not care about', '"same shape, ignore the scale"'],
      ['$x \\sim \\mathcal{N}(\\mu,\\sigma^2)$', 'distributed as', '$x$ is drawn from that distribution', '"$x$ comes from a bell curve"'],
      ['$\\mathbb{1}[\\cdot]$', 'indicator', '1 when the condition holds, 0 otherwise', '"count it if…"'],
      ['$\\theta$', 'theta', 'the generic name for "all the parameters"', '"the knobs"'],
      ['$:=$', 'defined as', 'this is a definition, not a claim', '"let this mean"']
    ], 'plain')}

<p>Four of those entries repay a second look, because each hides an idea rather than an abbreviation.</p>

<p><b>$\\arg\\min$ returns an input, not a value.</b> If $f(w) = (w-3)^2$, then $\\min_w f(w) = 0$, because zero is the smallest the function gets. But $\\arg\\min_w f(w) = 3$, because 3 is the input that achieves it. Training a model returns the second kind of answer: you want the parameters, not the loss they produce. Confusing them is the most common notation slip in interviews, and it is easy to avoid once you have said the two sentences aloud a single time.</p>

<p><b>$\\mathbb{E}$ is a weighted average, and the weights are probabilities.</b> For a fair die, $\\mathbb{E}[X] = \\tfrac{1}{6}(1+2+3+4+5+6) = 3.5$: every outcome carries weight one sixth. Load the die so that six comes up half the time and the other five outcomes share the rest equally, and the weights become $\\tfrac{1}{10}$ apiece for one through five and $\\tfrac{1}{2}$ for six, giving $\\mathbb{E}[X] = \\tfrac{1}{10}(1+2+3+4+5) + \\tfrac{1}{2}\\cdot 6 = 1.5 + 3 = 4.5$. The formula never changed; only the weights did. That is all the subscript in $\\mathbb{E}_{x \\sim p}$ is telling you — which set of weights to use.</p>

<p><b>The indicator $\\mathbb{1}[\\cdot]$ turns a condition into arithmetic.</b> It is 1 when the statement inside is true and 0 otherwise, which lets you count things inside a sum instead of writing an <code>if</code>. Accuracy, for instance, is $\\frac{1}{n}\\sum_i \\mathbb{1}[\\hat y_i = y_i]$ — the average of a list of ones and zeros, which is exactly the fraction that were right. It is also the reason accuracy has no useful gradient: nudging a parameter usually leaves every indicator unchanged, so the derivative is zero almost everywhere (§0.1).</p>

<p><b>$\\propto$ is a licence to be lazy, and a well-defined one.</b> Writing $a \\propto b$ says the two differ by a multiplicative constant that does not depend on anything you care about. It shows up all over Bayes' theorem because the denominator $P(E)$ is the same number for every hypothesis under comparison, so dropping it cannot change which hypothesis wins. You restore it only at the end, when you need a calibrated probability rather than a ranking (§1.1).</p>

<h2><span class="sn">0.6.5</span> Where the literature disagrees with itself</h2>

<p>Everything so far has been presented as though the conventions were settled. Several of them are not, and being told so plainly is more useful than discovering it halfway through a paper that appears to contradict the one you read yesterday.</p>

<p>These are the live disagreements. None of them is a mistake by either side; they are different fields' habits colliding in a field young enough not to have picked a winner.</p>

${H.table(['The clash', 'One camp writes', 'The other writes', 'How to tell which you are reading'], [
      ['<b>Row or column examples</b>', '$x$ is a column, model is $w^\\top x$ or $Wx$', '$x$ is a row, model is $xW$', 'Look at the data matrix: if $X$ is $n \\times d$ and they write $Xw$, examples are rows'],
      ['<b>Feature count</b>', '$d$ (machine learning)', '$p$ (statistics)', 'If the paper also says "regression" and "covariates", expect $p$'],
      ['<b>Sample count</b>', '$n$', '$N$ or $m$', 'Harmless — but check it is not the number of <i>classes</i>'],
      ['<b>Learning rate</b>', '$\\eta$ (eta)', '$\\alpha$ (alpha)', 'Both appear in the same conference; $\\alpha$ also means a Dirichlet parameter'],
      ['<b>$\\sigma$</b>', 'the sigmoid function', 'a standard deviation, or a singular value', 'Sigmoid takes an argument: $\\sigma(z)$. A standard deviation does not'],
      ['<b>$\\mathcal{L}$</b>', 'the loss to minimise', 'the <i>likelihood</i> to maximise', 'Opposite directions. Check the sign in front and whether they minimise or maximise'],
      ['<b>$\\lambda$</b>', 'regularisation strength', 'an eigenvalue, or a Poisson rate', 'Regularisation $\\lambda$ multiplies a norm; an eigenvalue sits next to a vector'],
      ['<b>Derivative layout</b>', 'gradient has the shape of $w$', 'gradient is a row vector', 'Almost every framework uses the first; some textbooks use the second (§0.7)'],
      ['<b>$\\log$</b>', 'natural log, always, in ML', 'base 10 in engineering, base 2 in information theory', 'If it appears in a loss, it is natural. If entropy is measured in bits, it is base 2 (§1.10)']
    ])}

<p>The $\\mathcal{L}$ row is the one that causes real damage. In statistics, $\\mathcal{L}(\\theta)$ is the likelihood — how probable your data is under parameters $\\theta$ — and you <i>maximise</i> it. In machine learning, $\\mathcal{L}(w)$ is the loss and you <i>minimise</i> it. These are close cousins, since minimising negative log-likelihood is maximising likelihood (§1.5), but a sign error here inverts your entire reading of a paper. The tell is the word next to it: look for "maximise" or an $\\arg\\max$.</p>

${H.key('When two papers disagree about notation, they almost never disagree about the mathematics. Write the shapes down and the disagreement usually evaporates.')}

<h2><span class="sn">0.6.6</span> Shapes come first</h2>

<p>Before you read what a formula <i>says</i>, work out what shape everything in it is. This sounds like a preliminary chore and it is actually the main event: something like nine tenths of the confusion people report when reading a formula turns out to be a shape confusion, and shape confusions are cheap to resolve because you need not understand the meaning of anything to resolve them.</p>

<p>The convention used on this site, which is the common one in machine learning, is:</p>

${H.table(['Object', 'Shape', 'Convention here'], [
      ['$x$ (one example)', '$d$', 'a column vector of $d$ features'],
      ['$X$ (the data)', '$n \\times d$', '<b>rows are examples</b>, columns are features'],
      ['$y$ (the labels)', '$n$', 'one number per row of $X$'],
      ['$w$ (the weights)', '$d$', 'one weight per feature'],
      ['$Xw$', '$n$', 'one prediction per example'],
      ['$W$ (a layer, §3.1)', '$d_{\\text{out}} \\times d_{\\text{in}}$', 'so $Wx$ maps in to out']
    ])}

<p>The rule that makes all of this mechanical was set out in §0.2: an $[n \\times d]$ object times a $[d \\times k]$ object gives an $[n \\times k]$ object. The inner dimensions must match, and when they do they vanish into a sum; the outer two survive and become the shape of the answer. Say it aloud as "n by d, times d by k, gives n by k", and check every product you meet against it.</p>

${H.note('Papers differ. Some make $x$ a row and write $xW$; deep learning libraries usually store a batch as $(\\text{batch}, \\text{features})$ and compute $XW^\\top$. Neither is more correct. <b>Write the shapes in the margin and every product either fits or it does not.</b>')}

${H.svg('reading a matrix product', '0 0 640 150', `
<defs><style>
.lbl{font:11px ui-monospace,monospace;fill:var(--muted)}
.big{font:600 13px ui-sans-serif,system-ui;fill:var(--text)}
.bx{fill:var(--panel);stroke:var(--line)}
.hl{fill:color-mix(in oklab,var(--blue) 22%,transparent);stroke:var(--blue)}
.hl2{fill:color-mix(in oklab,var(--c2) 22%,transparent);stroke:var(--c2)}
</style></defs>
<rect class="bx" x="20" y="35" width="150" height="90" rx="6"/>
<rect class="hl" x="20" y="60" width="150" height="18"/>
<text class="big" x="95" y="25" text-anchor="middle">X</text>
<text class="lbl" x="95" y="140" text-anchor="middle">n × d  (rows = examples)</text>
<text class="big" x="196" y="85">×</text>
<rect class="bx" x="222" y="35" width="46" height="90" rx="6"/>
<rect class="hl2" x="222" y="35" width="46" height="90"/>
<text class="big" x="245" y="25" text-anchor="middle">w</text>
<text class="lbl" x="245" y="140" text-anchor="middle">d × 1</text>
<text class="big" x="292" y="85">=</text>
<rect class="bx" x="320" y="35" width="46" height="90" rx="6"/>
<rect class="hl" x="320" y="60" width="46" height="18"/>
<text class="big" x="343" y="25" text-anchor="middle">Xw</text>
<text class="lbl" x="343" y="140" text-anchor="middle">n × 1</text>
<text class="lbl" x="400" y="60">one highlighted row of X,</text>
<text class="lbl" x="400" y="76">dotted with all of w,</text>
<text class="lbl" x="400" y="92">gives one prediction.</text>
<text class="lbl" x="400" y="116">The inner dimensions (d) must match;</text>
<text class="lbl" x="400" y="132">the outer ones (n, 1) survive.</text>
`, 'Every matrix product is this picture. The inner dimensions vanish into a sum; the outer dimensions are the shape of the answer.')}

${H.practice(`<p>The habit that makes this real is to instantiate. When a formula will not resolve, stop trying to think about it in general and pick the smallest numbers that are not degenerate: $n = 3$ examples, $d = 2$ features, $k = 2$ outputs. Write the matrices out with actual entries. Nearly every formula becomes obvious at this size, and the ones that do not are the ones genuinely worth spending an hour on.</p>
<p>Avoid $n = d$ when you instantiate. If your batch size equals your feature count, every wrong product still conforms and the error hides. The same goes for square weight matrices and for a batch of size 1, which lets a missing axis broadcast silently. Choosing 3 examples and 2 features costs nothing and turns a whole class of bugs into an immediate shape error.</p>
<p>The second habit is to write the shape in a comment after every line of tensor code, using a fixed letter per axis — $n$ batch, $d$ input features, $k$ outputs, $L$ sequence length, $h$ heads. A mismatch then shows up as a mismatch of <i>letters</i>, which you notice, rather than of digits, which you do not.</p>`)}

<h2><span class="sn">0.6.7</span> Three passes, every time</h2>

<p>Here is the procedure. It is short, it is the same every time, and it works on formulas you have never seen.</p>

${H.steps([
      '<b>Shapes.</b> Label every symbol scalar / vector / matrix and write its dimensions. If a product does not conform, you have misread something — stop and fix it before continuing.',
      '<b>The loop.</b> Find the $\\sum$, $\\prod$ or $\\mathbb{E}$. What is the index? What is inside? Anything not indexed is constant with respect to that loop and can be pulled out — this is the single most useful algebraic move in the subject.',
      '<b>The story.</b> Ask: what makes this quantity large? What makes it zero? What happens at the extremes? A formula you can interrogate this way is a formula you will not forget.'
    ])}

<p>Pass two contains a move worth naming, because you will use it constantly. If a factor does not depend on the summation index, it is a constant as far as that sum is concerned, and constants come out of sums: $\\sum_i c\\,x_i = c\\sum_i x_i$. That is why the $\\tfrac{1}{n}$ can sit outside the sum in the mean squared error, and it is the step that turns most intimidating expressions into short ones. Conversely, if you find yourself pulling something out that <i>does</i> carry the index, you have made an error, and it will usually be visible as a shape that no longer conforms.</p>

<p>Pass three is the one people skip and the one that produces understanding. A formula you have merely parsed is a formula you will forget by next week. A formula whose extremes you have probed — what makes it blow up, what makes it vanish, what happens if one input doubles — has been attached to something, and it stays.</p>

${H.worked('pass three on cross-entropy', `
$$\\mathcal{L} = -\\frac{1}{n}\\sum_i \\big[y_i\\log \\hat p_i + (1-y_i)\\log(1-\\hat p_i)\\big]$$
<p><b>Zero when?</b> When $\\hat p_i = y_i$ exactly for every $i$ — because $\\log 1 = 0$. <b>Infinite when?</b> When you predict $\\hat p = 0$ for something that happens: $\\log 0 = -\\infty$. That is the whole reason confident wrong answers are punished so brutally, and the reason production code clips probabilities away from 0 and 1 (§1.15). <b>Only one term survives</b> per example, because $y_i$ is 0 or 1 — the bracket is a switch, not a sum.</p>
<p><b>Put numbers on it.</b> Suppose the true label is $y = 1$. Predict $\\hat p = 0.9$ and the loss for that example is $-\\log 0.9 \\approx 0.105$. Predict $\\hat p = 0.5$, which is a shrug, and it rises to $-\\log 0.5 \\approx 0.693$. Predict $\\hat p = 0.01$, a confident mistake, and it is $-\\log 0.01 \\approx 4.605$ — forty-four times the penalty of the good prediction, from a model that was only slightly more certain than the shrug. That steepness is the point of the loss, not a side effect of it.</p>`)}

${H.more('index notation, and the sum that is not written down', `<p>Once you are comfortable with $\\sum$, you will meet expressions where it has been removed. In the <b>Einstein summation convention</b>, any index that appears twice in a term is silently summed over, so $A_{ij}x_j$ means $\\sum_j A_{ij}x_j$ — the $i$-th entry of $Ax$. The convention exists because in physics almost every repeated index is summed, so writing the sign every time is noise.</p>
<p>You do not need to adopt it, but you should be able to read it, because it survives into code: NumPy and PyTorch both offer <code>einsum</code>, where <code>"ij,j->i"</code> is exactly the expression above and <code>"bhqd,bhkd->bhqk"</code> is a batched multi-head attention score. The letters before the arrow name the axes of each input; the letters after it name the axes of the output; and <b>any letter that appears on the left but not on the right is summed away</b>. That single rule makes <code>einsum</code> readable, and it makes shape bugs in attention much harder to write.</p>`)}

${H.probe([
      ['What is the difference between $\\min f$ and $\\arg\\min f$?', '$\\min$ is the smallest <i>value</i> of $f$; $\\arg\\min$ is the <i>input</i> that achieves it. Training returns an $\\arg\\min$.'],
      ['What does $\\mathbb{E}_{x\\sim p}[f(x)]$ mean when $p$ is the data distribution?', 'The average of $f$ over the data-generating process — which we approximate by the sample mean over a finite dataset. That gap is the whole of generalization (§1.4).'],
      ['Why does $\\propto$ show up all over Bayes?', 'Because the denominator $P(B)$ does not depend on the hypothesis, so it cannot change which hypothesis wins. Dropping it saves work and never changes the argmax (§1.1).'],
      ['A paper writes $x^{(i)}$ and $W^{[2]}$ in the same line. What are the brackets doing?', 'Marking both superscripts as labels rather than powers: example $i$ and layer 2. Without the brackets they would read as an exponent, which is why the convention exists.'],
      ['In $\\nabla_w \\mathcal{L}$, what is the subscript?', 'The variable being differentiated with respect to — not an index. The result is one object with the same shape as $w$ (§0.7).']
    ], 'Saying "sigma" instead of reading the sum. If you cannot say what the index ranges over, you have not read the formula.')}`,
    labs: {
      dissect: function (host) {
        const el = ML.el;
        const FORMULAS = {
          mse: {
            name: 'Mean squared error',
            tex: '\\mathcal{L} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\hat y_i)^2',
            data: { y: [3, 1, 4, 2], yh: [2.5, 1.4, 3.2, 2.9] },
            steps: [
              ['\\hat y_i', 'The predictions — one per example. The hat says "estimate".', d => 'ŷ = [' + d.yh.join(', ') + ']'],
              ['y_i - \\hat y_i', 'The residual: how wrong we were, with a sign.', d => 'r = [' + d.y.map((v, i) => (v - d.yh[i]).toFixed(1)).join(', ') + ']'],
              ['(y_i - \\hat y_i)^2', 'Square it: sign gone, big errors punished quadratically.', d => 'r² = [' + d.y.map((v, i) => ((v - d.yh[i]) ** 2).toFixed(2)).join(', ') + ']'],
              ['\\sum_{i=1}^{n}', 'Add the four squared errors into one number.', d => 'Σ = ' + d.y.reduce((a, v, i) => a + (v - d.yh[i]) ** 2, 0).toFixed(3)],
              ['\\frac{1}{n}\\sum', 'Divide by n so the loss does not grow just because the dataset did.', d => 'MSE = ' + (d.y.reduce((a, v, i) => a + (v - d.yh[i]) ** 2, 0) / d.y.length).toFixed(4)]
            ]
          },
          softmax: {
            name: 'Softmax',
            tex: '\\sigma(z)_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}',
            data: { z: [2.0, 1.0, 0.1] },
            steps: [
              ['z_k', 'The logits — raw scores. Any real number, no constraints.', d => 'z = [' + d.z.join(', ') + ']'],
              ['e^{z_k}', 'Exponentiate: everything becomes positive, and gaps become ratios.', d => 'e^z = [' + d.z.map(v => Math.exp(v).toFixed(3)).join(', ') + ']'],
              ['\\sum_j e^{z_j}', 'The normalising constant — the same number for every class.', d => 'Σ = ' + d.z.reduce((a, v) => a + Math.exp(v), 0).toFixed(4)],
              ['\\frac{e^{z_k}}{\\sum_j e^{z_j}}', 'Divide: now they are positive and sum to 1, so they can be read as probabilities.', d => { const s = d.z.reduce((a, v) => a + Math.exp(v), 0); return 'p = [' + d.z.map(v => (Math.exp(v) / s).toFixed(3)).join(', ') + ']'; }]
            ]
          },
          gd: {
            name: 'The gradient-descent update',
            tex: 'w_{t+1} = w_t - \\eta\\,\\nabla_w \\mathcal{L}(w_t)',
            data: { w: 3.0, eta: 0.3 },
            steps: [
              ['w_t', 'Where the parameters are right now.', d => 'w = ' + d.w.toFixed(3)],
              ['\\nabla_w \\mathcal{L}(w_t)', 'The uphill direction of the loss at that point. Here L = w², so ∇ = 2w.', d => '∇ = ' + (2 * d.w).toFixed(3)],
              ['-\\,\\nabla_w \\mathcal{L}', 'Negate: we want downhill, not uphill. This is the only reason for the minus sign.', d => '−∇ = ' + (-2 * d.w).toFixed(3)],
              ['\\eta\\,\\nabla_w \\mathcal{L}', 'Scale by the learning rate — how far to trust a local slope.', d => 'η∇ = ' + (d.eta * 2 * d.w).toFixed(3)],
              ['w_t - \\eta\\nabla', 'Step. Repeat until the gradient is small (§3.5).', d => "w' = " + (d.w - d.eta * 2 * d.w).toFixed(3)]
            ]
          },
          bayes: {
            name: 'Bayes’ theorem',
            tex: 'P(H\\mid E) = \\frac{P(E\\mid H)\\,P(H)}{P(E)}',
            data: { pH: 0.001, sens: 0.99, fpr: 0.05 },
            steps: [
              ['P(H)', 'The prior: how common the hypothesis is before any evidence.', d => 'P(H) = ' + d.pH],
              ['P(E\\mid H)', 'The likelihood: how well the hypothesis explains what we saw.', d => 'P(E|H) = ' + d.sens],
              ['P(E\\mid H)P(H)', 'Multiply: the weight of the "H is true and we saw E" world.', d => (d.sens * d.pH).toExponential(3)],
              ['P(E)', 'The evidence — every way E could have happened, true positives plus false positives.', d => (d.sens * d.pH + d.fpr * (1 - d.pH)).toFixed(5)],
              ['\\frac{P(E\\mid H)P(H)}{P(E)}', 'Divide: what share of the E-worlds are H-worlds. That share is the posterior.', d => (d.sens * d.pH / (d.sens * d.pH + d.fpr * (1 - d.pH)) * 100).toFixed(2) + '%']
            ]
          },
          attn: {
            name: 'Scaled dot-product attention',
            tex: '\\mathrm{Attn}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V',
            data: {},
            steps: [
              ['Q, K, V', 'Three projections of the same tokens: what I am looking for, what I advertise, what I hand over.', () => 'shapes (n × d_k), (n × d_k), (n × d_v)'],
              ['QK^\\top', 'Every query dotted with every key — an n × n table of "how relevant is j to i".', () => 'shape (n × n)'],
              ['\\frac{QK^\\top}{\\sqrt{d_k}}', 'Divide by √d_k so the scores do not grow with dimension and saturate the softmax (§4.3).', () => 'variance held at ≈1'],
              ['\\mathrm{softmax}(\\cdot)', 'Turn each row into weights that sum to 1 — a distribution over the other tokens.', () => 'each row sums to 1'],
              ['(\\cdot)V', 'Weighted average of the values. That is the entire operation.', () => 'shape (n × d_v)']
            ]
          }
        };

        const st = Viz.controls(host, [{
          k: 'f', label: 'formula', type: 'select', value: 'mse',
          options: Object.keys(FORMULAS).map(k => ({ v: k, t: FORMULAS[k].name }))
        }], () => build());

        const stage = el('div');
        host.appendChild(stage);

        function build() {
          stage.innerHTML = '';
          const F = FORMULAS[st.f];
          const eq = el('div', { style: 'text-align:center;padding:14px 6px;font-size:1.05em' });
          stage.appendChild(eq);
          const piece = el('div', {
            style: 'border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:13px 15px;min-height:96px'
          });
          stage.appendChild(piece);
          const player = el('div');
          stage.appendChild(player);

          function show(i) {
            const s = F.steps[i];
            eq.innerHTML = '$$' + F.tex + '$$';
            ML.typeset(eq);
            piece.innerHTML =
              '<p class="boxtitle" style="margin-bottom:8px">piece ' + (i + 1) + ' of ' + F.steps.length + '</p>' +
              '<p style="margin:0 0 8px;font-size:1.15em">$' + s[0] + '$</p>' +
              '<p style="margin:0 0 8px;font-family:var(--sans);font-size:14px">' + s[1] + '</p>' +
              '<p style="margin:0;font-family:var(--mono);font-size:12.5px;color:var(--blue)">' + s[2](F.data) + '</p>';
            ML.typeset(piece);
          }
          Viz.player(player, {
            frames: F.steps.length, fps: 0.7, repeat: true,
            label: i => 'piece ' + (i + 1) + ' / ' + F.steps.length,
            onFrame: show
          });
        }
        build();
        Viz.note(host, 'Press play and let it walk. The point is the last line of each step: an abstract mark, and the actual number it produced on these four examples, side by side.');
      }
    },
    quiz: [
      {
        q: '$\\arg\\min_w \\mathcal{L}(w)$ returns…',
        options: ['the smallest value the loss reaches', 'the parameters that achieve the smallest loss', 'the gradient at the minimum', 'the learning rate'],
        answer: 1,
        why: 'The two marks answer different questions. $\\min_w \\mathcal{L}(w)$ asks "how small does the loss get?" and returns a number on the loss axis; $\\arg\\min_w \\mathcal{L}(w)$ asks "which $w$ got it there?" and returns a point in parameter space. Take $\\mathcal{L}(w) = (w-3)^2$: the minimum value is 0, and the argmin is 3. Option A is tempting precisely because the loss value is the number you watch during training and the one printed in every log line — but it is not what training <i>returns</i>. What you save to disk at the end is the parameters, which is an argmin. The distinction matters beyond pedantry: two models can reach the same minimum value through entirely different parameter settings, and which setting you landed on is what determines how the model behaves on data it has not seen.'
      },
      {
        q: 'In $X w$ with $X$ of shape $n\\times d$, the vector $w$ must have length…',
        options: ['$n$', '$d$', '$n \\times d$', 'either'],
        answer: 1,
        why: 'Apply the shape rule from §0.2: $[n \\times d]$ times $[d \\times 1]$ gives $[n \\times 1]$. The inner dimensions have to match, because each row of $X$ is dotted with the whole of $w$ and every entry needs a partner; when they match they cancel, and the surviving outer dimensions are the shape of the answer. Option A is the tempting one, and the reason it tempts is worth noticing: $n$ is the number that feels like "the size of the data", so it feels like the natural length for anything data-shaped. But $w$ is not data-shaped. It holds one weight per <i>feature</i>, so its length is fixed by the columns of $X$ and does not change when you collect more examples. That is exactly the property that lets a trained model score a single new row.'
      },
      {
        q: 'Why can Bayes be written with $\\propto$ instead of dividing by $P(E)$?',
        options: ['because $P(E)$ is always 1', 'because $P(E)$ is the same for every hypothesis, so it cannot change which one wins', 'because the prior cancels', 'it cannot — that is an error'],
        answer: 1,
        why: 'The denominator $P(E)$ is the probability of the evidence, computed by summing over every hypothesis. It therefore has no hypothesis attached to it: it is one fixed number that divides every posterior equally. Dividing a whole list of numbers by the same constant cannot reorder them, so if you only want to know which hypothesis is most probable, you may drop it. Option A is the seductive misreading — $P(E)$ is a probability, so surely it is 1? — but $P(E)$ is generally well below 1, and in the classic medical-test example it is around 0.05. What is true is that it is <i>constant across hypotheses</i>, which is a different and weaker claim doing all the work. Restore the denominator whenever you need a calibrated number rather than a ranking (§1.1).'
      },
      {
        q: 'A paper writes $x^{(i)}$ for the $i$-th training example and $W^{[2]}$ for the second layer. What are the brackets doing?',
        options: ['indicating repeated exponentiation', 'marking the superscript as a label rather than a power', 'denoting a transpose', 'nothing — they are decorative'],
        answer: 1,
        why: 'The top-right position already has a job: $x^2$ means $x$ times $x$. Deep learning wants to number two other things up there — which example, and which layer — so it needs a way to say "this superscript is a name, not an exponent". Brackets are that signal, and the convention is consistent enough that you can rely on it: round brackets conventionally index examples, square brackets index layers. Option A is the reading the notation exists to prevent, which is exactly why it is the tempting answer. The practical consequence is that dropping the brackets changes the meaning entirely, and that $x^{(2)}$ and $x^2$ are different objects that happen to look similar.'
      },
      {
        q: 'You meet $\\sigma$ three times in one paper: in $\\sigma(z)$, in $\\mathcal{N}(\\mu, \\sigma^2)$, and in a list $\\sigma_1 \\ge \\sigma_2 \\ge \\sigma_3$. How many different things does it mean?',
        options: ['one — it is the same function throughout', 'two', 'three', 'it is a typo in at least one place'],
        answer: 2,
        why: 'Three. In $\\sigma(z)$ it is the sigmoid function, which you can tell because it takes an argument. In $\\mathcal{N}(\\mu, \\sigma^2)$ it is a standard deviation, a plain number describing the spread of a distribution. In the ordered list it is a singular value from the decomposition in §0.2, which you can tell because singular values come sorted and indexed. Option A is tempting if you have learned $\\sigma$ as "the sigmoid" from a deep learning course and never met the other two, and this is the single most reliable way to lose your footing in a paper. The general lesson is that a Greek letter is not a name, it is a slot: work out what a symbol is from the shape of the expression around it and from whether it takes an argument, not from what the letter meant in the last thing you read.'
      }
    ],
    cards: [
      { q: 'What does a hat mean?', a: 'An estimate: $\\hat y$ is our prediction of $y$, $\\hat\\theta$ our estimate of $\\theta$.' },
      { q: 'The three passes for reading a formula', a: 'Shapes → the loop (what index, what is constant inside it) → the story (what makes it big, small, zero, infinite).' },
      { q: '$\\mathbb{E}_{x\\sim p}[f(x)]$', a: 'The average of $f$ weighting each $x$ by $p(x)$ — "$f$ on average, as $p$ sees the world".' },
      { q: 'Index subscript or "with respect to"?', a: 'If replacing it with 7 makes sense ("the seventh example") it is an index; otherwise it names a variable or a norm, as in $\\nabla_w$ and $\\|x\\|_2$.' },
      { q: 'Why do superscripts wear brackets?', a: 'To say "label, not power": $x^{(i)}$ is example $i$ and $W^{[l]}$ is layer $l$, while $x^2$ stays a square.' },
      { q: 'Random variable vs realisation', a: 'Capital $X$ is the mechanism, lowercase $x$ is the value it produced. Anything under $\\mathbb{E}$, after $\\sim$, or inside $P(\\cdot)$ is being treated as random.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.7 */
  ML.section({
    id: 'matrix-calculus', track: 'start', num: '0.7', level: 2,
    title: 'Matrix calculus without tears',
    lede: 'Backpropagation, the normal equations, ridge regression and the attention gradient are all one skill: differentiating an expression whose parts are vectors and matrices. There are about six results to know, and one rule for checking every one of them.',
    prereq: ['calculus-basics'],
    related: ['backprop', 'linear-algebra', 'optimization'],
    html: `
${H.tldr([
      'Pick the <b>denominator layout</b> and never change it: $\\partial f/\\partial w$ has <i>the same shape as $w$</i>. Then a gradient can always be subtracted from its parameter.',
      'Six results carry almost everything: $\\nabla_w a^\\top w = a$; $\\nabla_w w^\\top A w = (A+A^\\top)w$; $\\nabla_w \\|w\\|^2 = 2w$; $\\nabla_w \\|Xw-y\\|^2 = 2X^\\top(Xw-y)$; $\\partial(\\text{softmax+CE})/\\partial z = \\hat p - y$; and the chain rule as a product of Jacobians.',
      'The chain rule is associative, so you may multiply the Jacobians in any order — and one order is hundreds of times cheaper than the other. That asymmetry is why training large models is affordable.',
      'Never trust a derivation you have not gradient-checked. The check is four lines and it is in the lab below.'
    ])}

<p>You have a small linear model with two features, and you want to train it. From §0.1 you know the shape of the task: write a loss, find the direction that lowers it, take a step. From §0.3 you know how to differentiate. So do it.</p>

<p>The model predicts $\\hat y_i = w_1 x_{i1} + w_2 x_{i2}$, where $x_{i1}$ is the first feature of the $i$-th example. The loss is the sum of squared errors over $n$ examples. Differentiating with respect to $w_1$ is ordinary scalar calculus — hold $w_2$ still, apply the chain rule to the square, and you get</p>

$$\\frac{\\partial \\mathcal{L}}{\\partial w_1} = \\sum_{i=1}^{n} 2\\,(\\hat y_i - y_i)\\,x_{i1}.$$

<p>Now do $w_2$. You get exactly the same expression with $x_{i2}$ in place of $x_{i1}$. Nothing about the derivation changed; only one subscript did. And that is the moment the approach stops scaling, because a sentence embedding has 1536 features and a single transformer weight matrix has millions of entries. Writing out one scalar derivative per parameter is not merely tedious, it is impossible.</p>

<p>What you want is to differentiate <b>with respect to the whole vector at once</b> and get a single object back. The moment you try to write that down, though, you hit a question that scalar calculus never had to answer: <i>what shape is the answer?</i> The input $w$ was a $d$-vector, so the derivative has $d$ numbers in it, but is it a column or a row? For a matrix parameter it is worse still — the derivative has $mn$ numbers, and there are several defensible ways to arrange them.</p>

<p>That question has two standard answers in circulation, textbooks routinely fail to say which one they are using, and essentially all of the difficulty people report with matrix calculus comes from the collision. Settling it is the first job of this section, and once it is settled the rest is six results and a rule for checking them.</p>

<h2><span class="sn">0.7.1</span> Do it component-wise, once</h2>

<p>Before adopting any convention, it is worth watching the vector result assemble itself out of the scalar ones you already trust. Nothing below uses a rule you have not already met in §0.3.</p>

${H.deriv('the least-squares gradient, built one component at a time', [
      ['$\\mathcal{L}(w) = \\sum_{i=1}^{n}\\big(\\hat y_i - y_i\\big)^2, \\quad \\hat y_i = \\sum_{j=1}^{d} x_{ij}w_j$', 'Write everything with explicit indices and no vector notation at all. $x_{ij}$ is feature $j$ of example $i$, so the inner sum is one prediction.'],
      ['$\\dfrac{\\partial \\mathcal{L}}{\\partial w_m} = \\sum_{i} 2\\big(\\hat y_i - y_i\\big)\\dfrac{\\partial \\hat y_i}{\\partial w_m}$', 'Differentiate the outer square by the chain rule (§0.3), once per example. The sum passes straight through, because the derivative of a sum is the sum of the derivatives.'],
      ['$\\dfrac{\\partial \\hat y_i}{\\partial w_m} = x_{im}$', 'In the inner sum $\\sum_j x_{ij}w_j$, only the $j = m$ term contains $w_m$; every other term is a constant with respect to it and differentiates to zero.'],
      ['$\\dfrac{\\partial \\mathcal{L}}{\\partial w_m} = 2\\sum_{i} r_i\\,x_{im}, \\quad r_i := \\hat y_i - y_i$', 'Substitute, and give the residual a name. This holds for every $m$ from 1 to $d$, and the only thing that changes between them is which feature column is used.'],
      ['$\\nabla_w \\mathcal{L} = 2\\,X^\\top r$', 'Stack those $d$ numbers into one vector. The $m$-th entry is column $m$ of $X$ dotted with $r$, and "every column of $X$ dotted with $r$" is precisely what $X^\\top r$ means (§0.2).'],
      ['$= 2\\,X^\\top(Xw - y)$', 'Write the residual back out as $r = Xw - y$. Shapes: $X^\\top$ is $[d \\times n]$, $r$ is $[n]$, so the gradient is $[d]$ — the same shape as $w$, which is the only shape it could usefully have.']
    ], 'Read the last line as a sentence: <i>the design matrix transposed, applied to the residuals.</i> Each feature is credited with its own share of the error, weighted by how strongly that feature was present in the examples we got wrong. Nothing in this ladder needed a new rule of calculus. All the vector notation did was stop you writing the same line $d$ times.')}

<p>Put numbers on it once and it stops being abstract. Take three examples with an intercept and one feature, so $X = \\begin{bmatrix} 1 & 0 \\\\ 1 & 1 \\\\ 1 & 2\\end{bmatrix}$ and $y = (1,\\,2,\\,2)$, and try the parameters $w = (0.5,\\,0.5)$. The predictions are $Xw = (0.5,\\,1.0,\\,1.5)$, so the residuals are $r = Xw - y = (-0.5,\\,-1.0,\\,-0.5)$. Dotting $r$ with the first column of $X$, which is all ones, gives $-2$; dotting it with the second column $(0,1,2)$ gives $0 - 1 - 1 = -2$. The gradient is therefore $2 \\times (-2, -2) = (-4, -4)$: both parameters should increase, which is exactly what you would expect given that every single prediction is too low.</p>

<h2><span class="sn">0.7.2</span> Layout: the convention that removes the confusion</h2>

<p>The stacking step in that ladder quietly made a decision. It put the $d$ partial derivatives into a column, matching the shape of $w$. That choice has a name, and it is the one to adopt permanently:</p>

${H.key('The derivative of a scalar with respect to $w$ has the same shape as $w$. Full stop.')}

<p>This is <b>denominator layout</b>, so called because the shape of the answer is taken from the thing in the denominator of $\\partial \\mathcal{L}/\\partial w$. The alternative, <b>numerator layout</b>, takes the shape from the numerator instead and produces a row vector — the transpose of what we just derived. Neither is mathematically wrong. They contain the same $d$ numbers. But one of them makes the rest of your life easy and the other one does not.</p>

<p>The reason is the update rule. Gradient descent says $w \\leftarrow w - \\eta\\,\\nabla_w\\mathcal{L}$, and subtraction requires both operands to have the same shape. Under denominator layout that is automatic. Under numerator layout you must transpose the gradient before every single step, and if you forget, one of two things happens.</p>

<p>In strict linear algebra, nothing happens: the expression is undefined and you have written a type error. In NumPy or PyTorch, something worse happens, because array libraries broadcast rather than complain. Suppose $d = 3$, the parameters are the column $w = (1,\\,2,\\,3)$ with shape $[3 \\times 1]$, and a numerator-layout gradient arrives as the row $g = (0.1,\\,0.2,\\,0.3)$ with shape $[1 \\times 3]$. Subtracting them does not raise an error. It broadcasts, and you get</p>

$$w - g = \\begin{bmatrix} 0.9 & 0.8 & 0.7 \\\\ 1.9 & 1.8 & 1.7 \\\\ 2.9 & 2.8 & 2.7 \\end{bmatrix},$$

<p>a $[3 \\times 3]$ matrix where a $[3 \\times 1]$ parameter vector used to be. Your model still runs. It has simply stopped being the model you wrote. At $d = 768$ this produces a 590,000-entry array in place of a 768-entry one, the memory usage jumps, the loss behaves strangely, and nothing anywhere prints a warning. This is the single most expensive bug in the subject to track down, and choosing a layout and holding it is what prevents it.</p>

${H.pitfall(`<p>The trap is not choosing the wrong layout. It is <b>changing layout halfway through a derivation</b> because one line looked neater the other way round. A derivation that switches conventions produces a result that is right up to a transpose, which means it passes every dimension check you make on square matrices and fails silently on rectangular ones.</p>
<p>If you catch yourself transposing something "to make it fit", stop and write out the shapes of every factor. Nine times out of ten you will find you have mixed layouts, and the fix is to redo the line rather than to patch the shape.</p>`)}

<p>Under denominator layout, every derivative in machine learning has a shape you can predict before computing anything:</p>

${H.table(['Function', 'Shapes', 'Derivative shape'], [
      ['scalar → scalar', '$f:\\mathbb{R}\\to\\mathbb{R}$', 'scalar'],
      ['vector → scalar', '$f:\\mathbb{R}^d\\to\\mathbb{R}$', '$d$ — <b>the gradient</b>'],
      ['vector → vector', '$f:\\mathbb{R}^n\\to\\mathbb{R}^m$', '$m\\times n$ — <b>the Jacobian</b>'],
      ['matrix → scalar', '$f:\\mathbb{R}^{m\\times n}\\to\\mathbb{R}$', '$m\\times n$'],
      ['vector → scalar, twice', '$f:\\mathbb{R}^d\\to\\mathbb{R}$', '$d\\times d$ — <b>the Hessian</b> (§1.12)']
    ])}

<p>The middle row is the one that will matter most later, so it deserves naming properly. When a function takes $n$ numbers in and produces $m$ numbers out, there is no single slope; there are $m \\times n$ of them, one for every pairing of an output with an input. Arranged as a grid with outputs down the rows and inputs across the columns, that grid is the <b>Jacobian</b>, written $J$ or $\\partial u/\\partial v$. Entry $(k, j)$ answers "if I nudge input $j$, how much does output $k$ move?". A gradient is just the special case where $m = 1$, which is why a gradient behaves like a single row or column rather than a grid.</p>

${H.intuition(`<p>It is worth asking why a loss is always a single number. Nothing forces it: you could imagine reporting a whole vector of losses, one per example, and never collapsing them.</p>
<p>The reason you cannot is that optimisation needs an unambiguous direction to move in. With a scalar loss, "downhill" is one direction and every parameter gets one number telling it what to do. With a vector of losses there is no single downhill, because lowering one component can raise another and nothing in the mathematics says which trade you prefer. That preference has to come from you, and expressing it means choosing weights and summing — which produces a scalar again.</p>
<p>So the scalar loss is not a simplification made for convenience. It is what makes the gradient a single object with the same shape as the parameters, and it is why multi-objective training always ends in a weighted sum, whether that is the $\\lambda$ balancing a regulariser against a fit (§2.3) or the coefficients balancing several terms in a modern training recipe.</p>`)}

<h2><span class="sn">0.7.3</span> The six results, and where each one comes from</h2>

<p>Almost every gradient you will ever need is one of these six, or a chain-rule composition of them. Each can be verified by writing out a single component, exactly as in §0.7.1, and it is worth doing that once for each before using them freely for the rest of your life.</p>

${H.table(['Expression', 'Derivative w.r.t. $w$', 'Where it shows up'], [
      ['$a^\\top w$', '$a$', 'any linear score; the logit of a linear model'],
      ['$w^\\top A w$', '$(A + A^\\top)w$, and $2Aw$ if $A$ symmetric', 'quadratic forms, Newton steps, PCA'],
      ['$\\|w\\|_2^2 = w^\\top w$', '$2w$', 'L2 regularization → weight decay (§2.3)'],
      ['$\\|Xw - y\\|_2^2$', '$2X^\\top(Xw-y)$', 'linear regression, ridge, the normal equations'],
      ['$\\log\\big(\\sum_j e^{z_j}\\big)$', '$\\mathrm{softmax}(z)$', 'the log-partition function; why softmax appears at all'],
      ['softmax + cross-entropy', '$\\hat p - y$', 'the cleanest gradient in machine learning (§3.2)']
    ])}

<p>The first is the one to internalise, because it is the vector version of the fact that the derivative of $5x$ is 5.</p>

${H.deriv('why $\\nabla_w\\, a^\\top w = a$', [
      ['$a^\\top w = \\sum_{j=1}^{d} a_j w_j$', 'Expand the dot product into a sum (§0.2). This is now an ordinary function of $d$ ordinary numbers.'],
      ['$\\dfrac{\\partial}{\\partial w_m}\\sum_j a_j w_j = a_m$', 'Only the $j = m$ term contains $w_m$, and its derivative is the coefficient $a_m$. Every other term is constant and vanishes.'],
      ['$\\nabla_w\\,a^\\top w = (a_1, \\ldots, a_d) = a$', 'Stack the $d$ answers. Each slot of the gradient holds the corresponding slot of $a$, so the gradient <i>is</i> $a$.']
    ], 'Say it in words: in a linear score, the gradient with respect to the weights is whatever the weights were multiplying. This is why the gradient of a linear model is the input itself, and it is the reason a feature that is always large dominates the update unless the features have been standardised (§0.2).')}

<p>The second result is the one interviewers use to find out whether you have memorised a formula or understood it, because the widely-quoted version $2Aw$ is only true for symmetric $A$.</p>

${H.deriv('why $\\nabla_w\\, w^\\top A w = (A + A^\\top)w$', [
      ['$w^\\top A w = \\sum_{i}\\sum_{j} A_{ij}\\,w_i w_j$', 'Expand both matrix products into explicit sums. Every term is a coefficient times a product of two components of $w$.'],
      ['$\\dfrac{\\partial}{\\partial w_m}\\big(w_i w_j\\big) = \\delta_{im}w_j + w_i\\delta_{jm}$', 'Product rule. The symbol $\\delta_{im}$, the Kronecker delta, is 1 when $i = m$ and 0 otherwise — it is the indicator from §0.6 doing the job of an <code>if</code> inside a sum.'],
      ['$\\dfrac{\\partial}{\\partial w_m}\\,w^\\top A w = \\sum_j A_{mj}w_j + \\sum_i A_{im}w_i$', 'Apply that inside the double sum. The first delta collapses the $i$ index to $m$ and the second collapses $j$ to $m$, leaving two single sums.'],
      ['$= (Aw)_m + (A^\\top w)_m$', 'The first sum is row $m$ of $A$ dotted with $w$; the second is column $m$ of $A$ dotted with $w$, which is row $m$ of $A^\\top$ dotted with $w$.'],
      ['$\\nabla_w\\, w^\\top A w = (A + A^\\top)w$', 'Stack over $m$. When $A$ is symmetric the two terms coincide and this collapses to the familiar $2Aw$ — but only then.']
    ], 'Setting $A = I$ recovers the third result immediately: $\\nabla_w \\|w\\|^2 = (I + I)w = 2w$. That is the gradient behind weight decay, and it says something you can picture — the penalty always pushes straight back towards the origin, along the line joining $w$ to zero, with a strength proportional to how far out $w$ has drifted (§2.3).')}

${H.worked('the non-symmetric trap, in four numbers', `<p>Take $A = \\begin{bmatrix} 1 & 2 \\\\ 0 & 3\\end{bmatrix}$, which is not symmetric, and $w = (1, 1)$.</p>
<p><b>Expand by hand.</b> $w^\\top A w = 1\\cdot w_1^2 + 2\\,w_1w_2 + 0\\,w_2w_1 + 3\\,w_2^2 = w_1^2 + 2w_1w_2 + 3w_2^2$. Differentiate that ordinary two-variable function: $\\partial/\\partial w_1 = 2w_1 + 2w_2 = 4$ and $\\partial/\\partial w_2 = 2w_1 + 6w_2 = 8$. So the true gradient at $(1,1)$ is $(4, 8)$.</p>
<p><b>The correct rule.</b> $A + A^\\top = \\begin{bmatrix} 2 & 2 \\\\ 2 & 6\\end{bmatrix}$, and $(A+A^\\top)w = (4, 8)$. It agrees.</p>
<p><b>The remembered rule.</b> $2Aw = 2\\,(3, 3) = (6, 6)$. It does not agree, and it is wrong in <i>both</i> coordinates — not off by a constant, not off by a sign, just wrong. You can reproduce this failure yourself in the lab below by selecting the non-symmetric quadratic and switching the bug on.</p>`)}

<h2><span class="sn">0.7.4</span> Least squares, and the normal equations that fall out of it</h2>

<p>§0.7.1 derived the least-squares gradient the honest way, one component at a time. Here is the same result derived in vector notation, which is how you will actually do it once you trust the six results — and which yields, for free, the closed-form solution behind every linear regression you will ever fit.</p>

${H.deriv('the least-squares gradient, and the normal equations that fall out of it', [
      ['$\\mathcal{L}(w) = \\|Xw-y\\|_2^2$', 'Start. $X$ is $n\\times d$, $w$ is $d$, so $Xw-y$ is $n$ and $\\mathcal{L}$ is a scalar — as a loss must be.'],
      ['$= (Xw-y)^\\top(Xw-y)$', 'Squared L2 norm <i>is</i> a dot product with itself. This is the move that turns a norm into algebra.'],
      ['$= w^\\top X^\\top X w - 2y^\\top Xw + y^\\top y$', 'Expand. The two cross terms are equal because each is a scalar, and a scalar equals its own transpose.'],
      ['$\\nabla_w = 2X^\\top X w - 2X^\\top y$', 'Term by term: rule 2 with $A = X^\\top X$ (symmetric), then rule 1 with $a = X^\\top y$. The constant $y^\\top y$ dies.'],
      ['$= 2X^\\top(Xw - y)$', 'Factor. Read it: <b>the design matrix transposed, applied to the residuals</b> — every feature is credited with its share of the error.'],
      ['$X^\\top X\\,\\hat w = X^\\top y$', 'Set the gradient to zero. These are the normal equations; adding $\\lambda\\|w\\|^2$ to the loss adds $\\lambda I$ here, which is ridge regression (§2.3) and is exactly why ridge fixes a singular $X^\\top X$.']
    ], 'The last line is worth memorising as a sentence: <i>ridge adds a positive constant to the diagonal, so the matrix is always invertible.</i> That single fact answers three separate interview questions — conditioning, collinearity, and why ridge has a closed form while lasso does not.')}

<p>Two things about that ladder deserve unpacking, because they are the places a reader usually nods along without checking.</p>

<p>The third line claims the two cross terms are equal. They look different: one is $w^\\top X^\\top y$ and the other is $y^\\top X w$. But both are scalars, a scalar is a $[1\\times 1]$ matrix, and a $[1\\times 1]$ matrix is equal to its own transpose. Transposing $y^\\top X w$ reverses the order and transposes each factor, giving $w^\\top X^\\top y$ — so the two expressions are the same number and may be combined into one term with a coefficient of 2.</p>

<p>The last line sets the gradient to zero. That is legitimate here for a reason worth stating rather than assuming: the loss is a quadratic in $w$ with $X^\\top X$ in the quadratic slot, and $X^\\top X$ is positive semi-definite for any $X$ at all, because $v^\\top X^\\top X v = \\|Xv\\|^2 \\ge 0$. The surface is therefore a bowl rather than a saddle, so a stationary point is a minimum and not merely a point where the slope happens to vanish (§1.12).</p>

${H.worked('solving the normal equations on three data points', `<p>Reuse the tiny dataset from §0.7.1: $X = \\begin{bmatrix} 1 & 0 \\\\ 1 & 1 \\\\ 1 & 2\\end{bmatrix}$ and $y = (1, 2, 2)$, so the first column is an intercept and the second is one feature taking the values 0, 1 and 2.</p>
<p>$X^\\top X = \\begin{bmatrix} 3 & 3 \\\\ 3 & 5\\end{bmatrix}$ and $X^\\top y = (5,\\, 6)$. The determinant is $3\\times5 - 3\\times3 = 6$, which is not zero, so the inverse exists and $\\hat w = \\frac{1}{6}\\begin{bmatrix} 5 & -3 \\\\ -3 & 3\\end{bmatrix}(5, 6) = \\frac{1}{6}(25-18,\\; -15+18) = \\big(\\tfrac{7}{6},\\, \\tfrac{1}{2}\\big).$</p>
<p><b>Check it against the gradient condition.</b> The fitted values are $X\\hat w = (1.167,\\, 1.667,\\, 2.167)$ and the labels were $(1, 2, 2)$, so the residuals are $r = X\\hat w - y = (0.167,\\, -0.333,\\, 0.167)$. Dot $r$ with the first column of $X$, which is all ones: $0.167 - 0.333 + 0.167 = 0$. Dot it with the second column: $0 - 0.333 + 2(0.167) = 0$. Both come out zero, which is precisely the statement $X^\\top(X\\hat w - y) = 0$, and it has a geometric reading: <b>at the optimum the residual vector is orthogonal to every feature column.</b> There is no linear signal left in the data that the model has failed to use.</p>`)}

<p>Now the ridge claim, with numbers, because "always invertible" is the kind of sentence that is easy to repeat and easy to have never checked.</p>

<p>Suppose two of your features are perfectly correlated — someone recorded a length in metres and again in centimetres, say — so the design matrix is $X = \\begin{bmatrix} 1 & 1 \\\\ 2 & 2 \\\\ 3 & 3\\end{bmatrix}$. Then $X^\\top X = \\begin{bmatrix} 14 & 14 \\\\ 14 & 14\\end{bmatrix}$, whose determinant is $14 \\times 14 - 14 \\times 14 = 0$. The matrix is singular, the normal equations have no unique solution, and a naive least-squares solver either raises an error or returns whatever the floating-point noise happens to favour that day.</p>

<p>Add $\\lambda I$ with $\\lambda = 0.1$. The matrix becomes $\\begin{bmatrix} 14.1 & 14 \\\\ 14 & 14.1\\end{bmatrix}$, whose determinant is $14.1^2 - 14^2 = 0.1 \\times 28.1 = 2.81$. Not zero, so an inverse exists and the solution is unique. In terms of eigenvalues the story is cleaner still: the eigenvalues of $X^\\top X$ are 28 and 0, and adding $\\lambda I$ adds $\\lambda$ to every one of them, giving 28.1 and 0.1. A zero eigenvalue is exactly the flattened direction from §0.2, and ridge lifts it off the floor.</p>

${H.flag('Note what this argument does <i>not</i> claim. With $\\lambda = 0.1$ the condition number is $28.1 / 0.1 = 281$, so the problem is invertible but still badly conditioned, and the solution along the collapsed direction is essentially determined by $\\lambda$ rather than by your data. Taking $\\lambda = 1$ gives eigenvalues 29 and 1 and a condition number of 29, which is comfortable. Ridge guarantees a unique answer; it does not by itself guarantee a trustworthy one, and how much regularisation you need is an empirical question settled by validation (§2.3).')}

<h2><span class="sn">0.7.5</span> Checking a derivation against reality</h2>

<p><b>What you are looking at.</b> The lab below evaluates a gradient two completely independent ways and draws them side by side. Each pair of bars is one coordinate of $w$, so there are five pairs, and the vertical axis is the value of $\\partial\\mathcal{L}/\\partial w_j$ at one fixed point in parameter space. The first bar in each pair is the <b>analytic</b> gradient — a formula from this section, implemented directly. The second is the <b>numerical</b> gradient, obtained by nudging that one coordinate up by $\\varepsilon$, nudging it down by $\\varepsilon$, and dividing the change in the loss by $2\\varepsilon$. The numerical version knows nothing about calculus; it just measures. The readout gives the largest relative disagreement across the five coordinates and turns it into a verdict.</p>

<p><b>What to do with it.</b> Start on the least-squares expression with the bug toggle off, and confirm the bars are indistinguishable and the verdict reads a tick. Now switch on <i>introduce the classic bug</i>. The bug for this expression is dropping the factor of 2, and the relative error jumps to exactly 0.333 — every bar is half the height it should be. Try the other three expressions with the bug on as well: the ridge version forgets the $\\lambda$ term, so only the coordinates where $w_j$ is large go wrong; the quadratic form uses $2Aw$ on the non-symmetric $A$ from the worked example above; and the logistic loss forgets to divide by $n = 40$, which sends the relative error to about 0.95.</p>

<p><b>The thing genuinely worth noticing.</b> The size of the relative error tells you <i>what kind</i> of mistake you made, not merely that you made one. If your analytic gradient is the true one multiplied by some factor $k$, the relative error is $|1-k|/(1+k)$ regardless of the data. A factor of one half gives exactly $1/3$. A sign flip gives exactly 1. A missing term gives something ragged that differs from coordinate to coordinate. Learning to read that number saves hours: a clean 0.333 sends you looking for a dropped constant, while a value that varies across coordinates sends you looking for a missing term or a transposed index.</p>

${H.lab('gradcheck', 'Gradient checker — analytic vs numerical', 'The analytic gradient is the formula above, implemented directly. The numerical gradient perturbs one coordinate at a time and takes a central difference. If your derivation is right, they agree to about 1e-8 in double precision. Break the formula with the toggle and watch the agreement collapse.')}

<p>One control in that lab rewards a separate experiment. Drag $\\varepsilon$ while watching the relative error, first on the logistic loss and then on least squares, and you will see two different behaviours. The logistic loss is not a quadratic, so shrinking $\\varepsilon$ first improves the estimate — less truncation error — and then makes it worse again as floating-point cancellation takes over, giving the U-shape described in §1.15. The least-squares loss <i>is</i> exactly quadratic, so the central difference has no truncation error at all in exact arithmetic, and shrinking $\\varepsilon$ can only ever hurt. The moral is the same either way: a numerical gradient is a diagnostic tool with its own error budget, never a substitute for the real thing.</p>

<h2><span class="sn">0.7.6</span> The chain rule is a product of Jacobians</h2>

<p>Everything so far has differentiated a single expression. A neural network is not a single expression; it is a stack of them, each feeding the next. To train it you need the derivative of the loss with respect to parameters buried several layers deep, and the tool for that is the chain rule you met in §0.3, promoted from scalars to shapes.</p>

<p>In the scalar case the chain rule multiplies slopes: if $\\mathcal{L}$ depends on $u$, and $u$ depends on $v$, and $v$ depends on $x$, then $d\\mathcal{L}/dx$ is the product of the three intermediate slopes. In the vector case it is the same statement with matrices in place of numbers:</p>

$$\\frac{\\partial \\mathcal{L}}{\\partial x} = \\frac{\\partial \\mathcal{L}}{\\partial u}\\cdot\\frac{\\partial u}{\\partial v}\\cdot\\frac{\\partial v}{\\partial x}$$

<p>Each factor is a Jacobian, and each is a rectangle whose rows are outputs and whose columns are inputs. Because the outputs of one stage are the inputs of the next, the inner dimensions of neighbouring factors always match, and the product conforms automatically. That is not a coincidence; it is the shape rule from §0.2 restating the fact that the stages were composed in the first place.</p>

<p>Now make it concrete, because the abstraction hides the whole point. Take a small network on image inputs: $x$ has 784 entries, the first layer produces $h_1$ with 512, the second produces $h_2$ with 256, the third produces 10 class scores, and the loss collapses those to one number. The chain from the loss back to the input is a product of four Jacobians with these shapes:</p>

$$\\underbrace{[1 \\times 10]}_{\\partial\\mathcal{L}/\\partial h_3} \\cdot \\underbrace{[10 \\times 256]}_{\\partial h_3/\\partial h_2} \\cdot \\underbrace{[256 \\times 512]}_{\\partial h_2/\\partial h_1} \\cdot \\underbrace{[512 \\times 784]}_{\\partial h_1/\\partial x}$$

<p>Check the inner dimensions: 10 meets 10, 256 meets 256, 512 meets 512. Every one cancels, and the answer is $[1 \\times 784]$ — one number per input pixel, which is exactly what a gradient with respect to $x$ should be.</p>

<h3>The same product, two orders, two very different bills</h3>

<p>Matrix multiplication is associative, so you may bracket that product any way you like and get the same answer. What is not the same is the cost. Multiplying an $[a \\times b]$ matrix by a $[b \\times c]$ matrix costs about $abc$ multiply-and-add operations, so the bracketing determines how much arithmetic you do.</p>

${H.table(['Order', 'Step', 'Shape produced', 'Multiply-adds'], [
      ['<b>Left to right</b><br>(reverse mode)', '$[1\\times10]\\cdot[10\\times256]$', '$[1\\times256]$', '2,560'],
      ['', '$[1\\times256]\\cdot[256\\times512]$', '$[1\\times512]$', '131,072'],
      ['', '$[1\\times512]\\cdot[512\\times784]$', '$[1\\times784]$', '401,408'],
      ['', '<b>total</b>', '', '<b>535,040</b>'],
      ['<b>Right to left</b><br>(forward mode)', '$[256\\times512]\\cdot[512\\times784]$', '$[256\\times784]$', '102,760,448'],
      ['', '$[10\\times256]\\cdot[256\\times784]$', '$[10\\times784]$', '2,007,040'],
      ['', '$[1\\times10]\\cdot[10\\times784]$', '$[1\\times784]$', '7,840'],
      ['', '<b>total</b>', '', '<b>104,775,328</b>']
    ])}

<p>The same answer, computed two ways, and one of them costs 196 times more than the other. The memory story is just as stark: going left to right, the largest object ever held is a 784-entry row, while going right to left you must build and store a $[256 \\times 784]$ matrix with 200,704 entries in it.</p>

<p>The reason for the asymmetry is visible in the shapes, and it is completely general. Starting from the left means starting at the <b>scalar</b> end. Because the leftmost factor has a single row, every partial product also has a single row, so you are only ever multiplying a vector by a matrix — never a matrix by a matrix. Starting from the right means starting at the wide end, where nothing is small, and you pay for full matrix products until the very last step.</p>

${H.key('Reverse mode costs one sweep per <i>output</i>; forward mode costs one per <i>input</i>. A loss has exactly one output and a network has millions of inputs, so the choice is not close.')}

${H.analogy(`<p>Picture a company with a million employees and one number at the end of the year: profit. You want to know, for each employee, how much profit would change if that person worked slightly differently.</p>
<p><b>Forward mode</b> answers this by simulation. Perturb employee one, run the whole company forward, see what profit does. Then reset, perturb employee two, run the whole company again. You need a million full simulations, because each one propagates the influence of a single starting change all the way to the end.</p>
<p><b>Reverse mode</b> answers it by assigning blame. Start from the profit figure and ask which departments moved it, then ask which teams moved those departments, then which people moved those teams. One pass down the organisation chart gives every employee their number, because you started from the single thing you cared about and worked backwards. That is the whole of backpropagation, and the reason it costs about as much as one forward pass rather than a million.</p>`)}

${H.svg('why reverse mode is the cheap direction', '0 0 640 190', `
<defs><style>
.n{fill:var(--panel);stroke:var(--line)}
.t{font:11px ui-sans-serif,system-ui;fill:var(--text);text-anchor:middle}
.s{font:10px ui-monospace,monospace;fill:var(--faint);text-anchor:middle}
.fw{stroke:var(--c1);fill:none;stroke-width:1.6;marker-end:url(#a1)}
.bw{stroke:var(--c2);fill:none;stroke-width:1.6;stroke-dasharray:4 3;marker-end:url(#a2)}
.cap{font:11px ui-sans-serif,system-ui}
</style>
<marker id="a1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--c1)"/></marker>
<marker id="a2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--c2)"/></marker>
</defs>
<g>
<rect class="n" x="18" y="52" width="86" height="40" rx="7"/><text class="t" x="61" y="70">x</text><text class="s" x="61" y="84">10⁶ params</text>
<rect class="n" x="158" y="52" width="86" height="40" rx="7"/><text class="t" x="201" y="70">h₁</text><text class="s" x="201" y="84">1024</text>
<rect class="n" x="298" y="52" width="86" height="40" rx="7"/><text class="t" x="341" y="70">h₂</text><text class="s" x="341" y="84">1024</text>
<rect class="n" x="438" y="52" width="86" height="40" rx="7"/><text class="t" x="481" y="70">ℒ</text><text class="s" x="481" y="84">1 scalar</text>
<path class="fw" d="M104 66 H154"/><path class="fw" d="M244 66 H294"/><path class="fw" d="M384 66 H434"/>
<path class="bw" d="M434 82 H388"/><path class="bw" d="M294 82 H248"/><path class="bw" d="M154 82 H108"/>
<text class="cap" x="18" y="128" fill="var(--c1)">forward →  compute activations, cache them</text>
<text class="cap" x="18" y="148" fill="var(--c2)">backward ←  one row vector per layer, never a full Jacobian</text>
<text class="cap" x="18" y="172" fill="var(--muted)">Many inputs, one output ⇒ start from the scalar end. Reverse mode costs about the same as one forward pass.</text>
</g>
`, 'The asymmetry is the whole reason training a billion-parameter model is affordable: the cost of the gradient is a constant multiple of the cost of the prediction, no matter how many parameters there are.')}

<h3>No layer ever builds its Jacobian</h3>

<p>There is a second saving hiding inside the first, and it is the one that makes an autodiff framework possible to write. Going left to right, a layer never has to produce its Jacobian at all. It only has to answer one question: <i>given the gradient of the loss with respect to my output, what is the gradient with respect to my input, and with respect to my parameters?</i> That is a local contract, and it is why you can write a new layer type without knowing anything about the rest of the network.</p>

<p>Work it out for the layer that matters most, a plain matrix multiply $h = Wx$.</p>

${H.deriv('the backward pass of a linear layer', [
      ['$h_k = \\sum_{j} W_{kj}\\,x_j$', 'One output entry is one row of $W$ dotted with $x$ (§0.2). Shapes: $W$ is $[m\\times n]$, $x$ is $[n]$, $h$ is $[m]$.'],
      ['$g := \\partial\\mathcal{L}/\\partial h \\in \\mathbb{R}^{m}$', 'Name the thing arriving from above. Under denominator layout it has the same shape as $h$: one number per output, saying how much the loss would move if that output moved.'],
      ['$\\dfrac{\\partial\\mathcal{L}}{\\partial W_{ij}} = \\sum_k g_k\\,\\dfrac{\\partial h_k}{\\partial W_{ij}} = \\sum_k g_k\\,\\delta_{ki}x_j = g_i x_j$', 'Chain rule through every output. $W_{ij}$ appears in exactly one output, namely $h_i$, so the delta collapses the sum to a single term.'],
      ['$\\dfrac{\\partial\\mathcal{L}}{\\partial W} = g\\,x^\\top$', 'Every entry is an entry of $g$ times an entry of $x$, which is the definition of an outer product. Shapes: $[m\\times1]$ times $[1\\times n]$ gives $[m\\times n]$ — the shape of $W$, as denominator layout demands.'],
      ['$\\dfrac{\\partial\\mathcal{L}}{\\partial x_j} = \\sum_k g_k W_{kj} \\;\\Rightarrow\\; \\dfrac{\\partial\\mathcal{L}}{\\partial x} = W^\\top g$', 'Now differentiate with respect to the input instead. Column $j$ of $W$ dotted with $g$ is row $j$ of $W^\\top$ dotted with $g$. Shapes: $[n\\times m]$ times $[m]$ gives $[n]$ — the shape of $x$.']
    ], 'These two lines are the entire backward pass of a dense layer, and §3.2 is largely this result applied repeatedly. Notice what never appeared: the Jacobian $\\partial h/\\partial x$, which is $W$ itself and would be $[m \\times n]$, is never formed as a separate object — the code computes $W^\\top g$ directly. That substitution is the difference between a framework that fits on a GPU and one that does not.')}

${H.history(`<p>Reverse-mode differentiation has been rediscovered an unusual number of times, which is a good sign that it is the natural answer rather than a clever trick.</p>
<p>Forward accumulation was described by Wengert in 1964. Reverse accumulation appeared in Linnainmaa's master's thesis in 1970, in the context of rounding-error analysis rather than learning, and was applied to error minimisation in networks by Werbos in the mid-1970s. It reached the wider community through the 1986 paper of Rumelhart, Hinton and Williams, which is where the name "backpropagation" stuck.</p>
<p>The lesson worth carrying is that backpropagation is not a neural network algorithm. It is the chain rule with a sensible bracketing, and it applies to any computation you can write as a graph of differentiable steps — which is why the same machinery now differentiates physics simulators, renderers and optimisation solvers (§3.11).</p>`)}

${H.more('Deriving the softmax + cross-entropy gradient (the one worth doing by hand once)', `
<p>Let $z$ be the logits, $\\hat p = \\mathrm{softmax}(z)$, and $\\mathcal{L} = -\\sum_k y_k \\log \\hat p_k$ with $y$ one-hot.</p>
${H.deriv('show that $\\partial\\mathcal{L}/\\partial z = \\hat p - y$', [
        ['$\\mathcal{L} = -\\sum_k y_k\\big(z_k - \\log\\sum_j e^{z_j}\\big)$', 'Substitute $\\log\\hat p_k = z_k - \\log\\sum_j e^{z_j}$. Writing softmax in log space first is the trick — it is also the numerically stable implementation (§1.15).'],
        ['$= -\\sum_k y_k z_k + \\log\\sum_j e^{z_j}$', 'The second term is constant in $k$, and $\\sum_k y_k = 1$ because $y$ is one-hot.'],
        ['$\\dfrac{\\partial}{\\partial z_m}\\Big(-\\sum_k y_k z_k\\Big) = -y_m$', 'Only the $k=m$ term survives.'],
        ['$\\dfrac{\\partial}{\\partial z_m}\\log\\sum_j e^{z_j} = \\dfrac{e^{z_m}}{\\sum_j e^{z_j}} = \\hat p_m$', 'The derivative of the log-sum-exp <i>is</i> the softmax. This is result five in the table, and it is why softmax and cross-entropy are married.'],
        ['$\\dfrac{\\partial\\mathcal{L}}{\\partial z_m} = \\hat p_m - y_m$', 'Add. The gradient is literally "how wrong the probability was" — no Jacobian of the softmax ever needs to be formed.']
      ])}
<p>Two consequences that get asked about. First, the softmax Jacobian $\\mathrm{diag}(\\hat p) - \\hat p\\hat p^\\top$ is <i>never materialised</i> in practice, because it cancels against the cross-entropy derivative — which is why frameworks fuse the two into one op. Second, the gradient is bounded in $[-1,1]$, so cross-entropy on logits cannot explode the way a naive squared loss on probabilities can.</p>
<p>Put a number on the first consequence. With ten classes the softmax Jacobian is a $[10\\times10]$ matrix, which sounds harmless — until you remember it is per token and per example. A batch of 32 sequences of length 1024 over a 50,000-token vocabulary would need 32 × 1024 × 50,000 × 50,000 entries if the Jacobian were formed honestly. The fused form needs 32 × 1024 × 50,000, which is the size of the probabilities you already had. The cancellation is not a micro-optimisation; it is the difference between possible and impossible.</p>
`)}

<h2><span class="sn">0.7.7</span> Always gradient-check</h2>

<p>Every derivation above can be got wrong, and the failures are quiet. A gradient that is off by a constant factor still points downhill, so training still appears to work — it simply behaves as though the learning rate were different from the one you set. A gradient with a transpose in the wrong place can still reduce the loss for a while. You will not find these by staring at the loss curve. You find them by comparing your formula against a measurement.</p>

<p>The measurement is the definition of a derivative, evaluated on a computer. Nudge one coordinate up by $\\varepsilon$, nudge it down by $\\varepsilon$, and divide the change in the function by $2\\varepsilon$:</p>

${H.code(`def grad_check(f, w, analytic, eps=1e-5):
    """Central differences. Agreement to ~1e-8 in float64 means the
    derivation is right; ~1e-3 usually means a transpose is missing."""
    num = np.zeros_like(w)
    for i in range(w.size):
        p = w.copy(); p.flat[i] += eps
        m = w.copy(); m.flat[i] -= eps
        num.flat[i] = (f(p) - f(m)) / (2 * eps)
    rel = np.abs(num - analytic) / (np.abs(num) + np.abs(analytic) + 1e-12)
    return rel.max()`)}

<p>Two details in that function are load-bearing. It compares <b>relative</b> rather than absolute error, because a gradient entry of size $10^{6}$ and one of size $10^{-6}$ cannot be judged against the same threshold. And it adds a tiny constant to the denominator so that a coordinate where both gradients are genuinely zero does not divide by zero and report a spurious failure.</p>

${H.pitfall('Use <b>central</b> differences, not forward. The forward difference has error $O(\\varepsilon)$; the central difference has error $O(\\varepsilon^2)$, which buys you four extra digits for one extra function evaluation. And check in float64 — in float32 the rounding noise swamps the signal and every correct gradient looks broken.')}

${H.practice(`<p>Gradient checking is $O(\\text{number of parameters})$ function evaluations, so you never run it on a real model. You run it on a toy: one layer, five inputs, three outputs, random data, float64. If the layer is right at that size it is right at every size, because the formula does not know how big the tensors are.</p>
<p>Three things reliably produce a failed check on correct code, and recognising them saves an afternoon. <b>Non-differentiable points:</b> ReLU has no derivative at exactly zero, so if a pre-activation lands there the numerical estimate straddles the kink and disagrees. Perturb the test point slightly and re-run. <b>Randomness:</b> dropout and any stochastic augmentation must be frozen, or $f(w+\\varepsilon)$ and $f(w-\\varepsilon)$ are evaluations of two different functions. <b>Batch-dependent layers:</b> batch normalisation makes each example's output depend on its neighbours, so check it in evaluation mode or with a batch size large enough that the statistics barely move.</p>
<p>When a check does fail, bisect rather than stare. Check the last layer alone, then the last two, and so on. The first composition that fails contains the bug, and in practice the bug is a transpose, a sum over the wrong axis, or a factor of $n$ that was applied in the loss but not in its gradient.</p>`)}

${H.probe([
      ['What shape is $\\nabla_W \\mathcal{L}$ for a weight matrix $W$?', 'The same shape as $W$. Otherwise you could not subtract it during the update.'],
      ['Why is reverse-mode differentiation the right choice for neural networks?', 'Many parameters in, one scalar out. Reverse mode costs one gradient per <i>output</i>; forward mode costs one per <i>input</i>. With 10⁹ inputs and 1 output the choice is not close (§3.11).'],
      ['Derive $\\nabla_w\\|Xw-y\\|^2$.', '$2X^\\top(Xw-y)$ — expand into a quadratic form, apply $\\nabla w^\\top Aw = 2Aw$ for symmetric $A=X^\\top X$ and $\\nabla a^\\top w = a$, then factor.'],
      ['What are the two gradients a dense layer has to return?', '$\\partial\\mathcal{L}/\\partial W = g\\,x^\\top$, an outer product with the shape of $W$, and $\\partial\\mathcal{L}/\\partial x = W^\\top g$, which is what gets passed further back. The Jacobian is never formed.'],
      ['Your gradient check gives a relative error of 0.3. What is the most likely bug?', 'A missing or extra transpose, or a sum over the wrong axis. Errors around 1e-2 to 1e-1 are structural; errors around 1e-6 in float32 are just precision. A value very close to 1/3 specifically suggests a factor-of-two error, and exactly 1.0 suggests a sign flip.']
    ], 'Reciting "the derivative of $w^\\top A w$ is $2Aw$" without the symmetry condition. For general $A$ it is $(A+A^\\top)w$, and interviewers who know the difference are checking whether you do.')}`,
    labs: {
      gradcheck: function (host) {
        const R = Num.rng(17);
        const n = 40, d = 5;
        const X = Array.from({ length: n }, () => Array.from({ length: d }, () => R.normal(0, 1)));
        const wTrue = [1.5, -0.8, 0.3, 2.0, -1.2];
        const y = X.map(r => Num.dot(r, wTrue) + R.normal(0, 0.3));

        const st = Viz.controls(host, [
          { k: 'expr', label: 'expression', type: 'select', value: 'ls', options: [
            { v: 'ls', t: '‖Xw − y‖²  (least squares)' },
            { v: 'ridge', t: '‖Xw − y‖² + λ‖w‖²  (ridge)' },
            { v: 'quad', t: 'wᵀAw  with A non-symmetric' },
            { v: 'logreg', t: 'logistic cross-entropy' }
          ] },
          { k: 'lam', label: 'λ (ridge only)', min: 0, max: 5, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'bug', label: 'introduce the classic bug', type: 'toggle', value: false },
          { k: 'eps', label: 'ε for the difference', min: -9, max: -2, step: 1, value: -5, fmt: v => '1e' + v }
        ], () => S.redraw());

        const out = Viz.readout(host, [
          { k: 'rel', label: 'max relative error', cls: 'key' },
          { k: 'verdict', label: 'verdict' },
          { k: 'gn', label: '‖analytic‖' },
          { k: 'note', label: 'the bug' }
        ]);

        const A = [[2, 1, 0, 0, 0], [0, 3, 1, 0, 0], [0, 0, 1, 2, 0], [0, 0, 0, 4, 1], [1, 0, 0, 0, 2]];

        function value(w) {
          if (st.expr === 'quad') { const Aw = Num.matvec(A, w); return Num.dot(w, Aw); }
          if (st.expr === 'logreg') {
            let L = 0;
            for (let i = 0; i < n; i++) {
              const z = Num.dot(X[i], w), p = Num.sigmoid(z), t = y[i] > 0 ? 1 : 0;
              L += -(t * Math.log(Math.max(1e-12, p)) + (1 - t) * Math.log(Math.max(1e-12, 1 - p)));
            }
            return L / n;
          }
          let s = 0;
          for (let i = 0; i < n; i++) { const r = Num.dot(X[i], w) - y[i]; s += r * r; }
          if (st.expr === 'ridge') s += st.lam * Num.dot(w, w);
          return s;
        }

        function analytic(w) {
          const g = new Array(d).fill(0);
          if (st.expr === 'quad') {
            // correct: (A + Aᵀ)w. bug: 2Aw, which is only right for symmetric A
            const Aw = Num.matvec(A, w), Atw = Num.matvec(Num.transpose(A), w);
            for (let j = 0; j < d; j++) g[j] = st.bug ? 2 * Aw[j] : Aw[j] + Atw[j];
            return g;
          }
          if (st.expr === 'logreg') {
            for (let i = 0; i < n; i++) {
              const p = Num.sigmoid(Num.dot(X[i], w)), t = y[i] > 0 ? 1 : 0;
              // bug: forget to divide by n
              for (let j = 0; j < d; j++) g[j] += (p - t) * X[i][j] / (st.bug ? 1 : n);
            }
            return g;
          }
          for (let i = 0; i < n; i++) {
            const r = Num.dot(X[i], w) - y[i];
            for (let j = 0; j < d; j++) g[j] += 2 * r * X[i][j];
          }
          // bug: forget the ridge term's derivative
          if (st.expr === 'ridge' && !st.bug) for (let j = 0; j < d; j++) g[j] += 2 * st.lam * w[j];
          if (st.expr === 'ls' && st.bug) for (let j = 0; j < d; j++) g[j] *= 0.5;   // dropped the 2
          return g;
        }

        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w2, h, T) {
            const w = [0.6, -0.4, 1.1, 0.2, -0.9];
            const eps = Math.pow(10, st.eps);
            const ana = analytic(w);
            const num = w.map((_, j) => {
              const a = w.slice(), b = w.slice();
              a[j] += eps; b[j] -= eps;
              return (value(a) - value(b)) / (2 * eps);
            });
            let rel = 0;
            for (let j = 0; j < d; j++) {
              const r = Math.abs(num[j] - ana[j]) / (Math.abs(num[j]) + Math.abs(ana[j]) + 1e-12);
              if (r > rel) rel = r;
            }
            const P = Viz.plot(ctx, w2, h, {
              xd: [-0.5, d - 0.5],
              yd: [Math.min(0, Math.min.apply(null, ana.concat(num))) * 1.25, Math.max(0.1, Math.max.apply(null, ana.concat(num))) * 1.25],
              pad: { l: 52, r: 14, t: 16, b: 40 }
            }).frame({ xticks: ana.map((_, j) => j), xlabel: 'coordinate of w', ylabel: '∂L / ∂wⱼ' });
            P.clip(() => {
              ana.forEach((v, j) => {
                const bw = P.pw / d * 0.3;
                ctx.fillStyle = T.c1; ctx.globalAlpha = .85;
                ctx.fillRect(P.x(j) - bw, Math.min(P.y(0), P.y(v)), bw, Math.abs(P.y(v) - P.y(0)));
                ctx.fillStyle = T.c2;
                ctx.fillRect(P.x(j), Math.min(P.y(0), P.y(num[j])), bw, Math.abs(P.y(num[j]) - P.y(0)));
                ctx.globalAlpha = 1;
              });
              P.hline(0, { color: T.faint, dash: false, width: 1 });
            });
            const bugs = {
              ls: 'dropped the factor of 2', ridge: 'forgot the λ term', quad: 'used 2Aw for non-symmetric A', logreg: 'forgot to divide by n'
            };
            out({
              rel: rel < 1e-4 ? rel.toExponential(1) : rel.toFixed(4),
              verdict: rel < 1e-6 ? '✓ matches' : rel < 1e-3 ? '~ precision only' : '✗ wrong',
              gn: Num.norm(ana).toFixed(3),
              note: st.bug ? bugs[st.expr] : 'none'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: 'analytic (your derivation)' }, { c: 'var(--c2)', t: 'numerical (central difference)' }]);
        Viz.note(host, 'Now drag ε. Too large and the truncation error dominates; too small and floating-point cancellation does — the relative error is a U-shape with its floor near 1e-5 for a well-scaled problem. This U is the same phenomenon as §1.15’s catastrophic cancellation, and it is why nobody trains with numerical gradients.');
      }
    },
    quiz: [
      {
        q: 'In denominator layout, $\\partial \\mathcal{L}/\\partial W$ for $W\\in\\mathbb{R}^{m\\times n}$ has shape…',
        options: ['$n\\times m$', '$m\\times n$', '$mn \\times 1$', 'scalar'],
        answer: 1,
        why: 'Denominator layout takes the shape of the answer from the thing in the denominator, which here is $W$ itself, so the gradient is $[m \\times n]$. The reason to insist on this is entirely practical: the update $W \\leftarrow W - \\eta\\,\\partial\\mathcal{L}/\\partial W$ has to be a legal subtraction, and it only is when both sides have the same shape. Option A is the tempting one, because the transposed arrangement is exactly what numerator layout gives you and several respectable textbooks use it. Both conventions hold the same $mn$ numbers and neither is wrong — but mixing them within one derivation produces a result that is correct up to a transpose, which passes every check you make on square matrices and fails silently on rectangular ones. Every mainstream framework returns the denominator-layout version, so that is the one to hold.'
      },
      {
        q: '$\\nabla_w\\, w^\\top A w$ equals $2Aw$ only when…',
        options: ['$A$ is invertible', '$A$ is symmetric', '$w$ is a unit vector', 'always'],
        answer: 1,
        why: 'The general result is $(A + A^\\top)w$, which you get by expanding $w^\\top A w$ into $\\sum_{i,j}A_{ij}w_iw_j$ and applying the product rule: the index $m$ you are differentiating with respect to can appear in either the $i$ slot or the $j$ slot, so two terms survive rather than one. When $A$ is symmetric those two terms are identical and add to $2Aw$. Option D is the answer most people give, because $2Aw$ is the version quoted in every cheat sheet and the symmetric case is the one that actually arises in practice — $X^\\top X$, covariance matrices and Hessians are all symmetric, so the shortcut almost never bites. It bites in exactly the situations where nobody is checking. With $A = \\begin{bmatrix} 1 & 2 \\\\ 0 & 3 \\end{bmatrix}$ and $w = (1,1)$, the true gradient is $(4, 8)$ while $2Aw$ gives $(6, 6)$: wrong in both coordinates.'
      },
      {
        q: 'The gradient of softmax + cross-entropy with respect to the logits is…',
        options: ['$\\hat p(1-\\hat p)$', '$\\hat p - y$', '$-y/\\hat p$', 'the softmax Jacobian times $y$'],
        answer: 1,
        why: 'Prediction minus target, which is about as clean as a gradient gets: it is literally "how wrong the probability was", one number per class, bounded between $-1$ and $1$. The derivation goes through log space, where cross-entropy becomes $-\\sum_k y_kz_k + \\log\\sum_j e^{z_j}$, and the derivative of that log-sum-exp term is exactly the softmax. Options A and D are both tempting for the same reason: they are things that genuinely appear in the middle of the derivation. $\\hat p(1-\\hat p)$ is the diagonal of the softmax Jacobian, and the full Jacobian $\\mathrm{diag}(\\hat p) - \\hat p\\hat p^\\top$ really is what you would form if you differentiated softmax and cross-entropy separately. The point is that it cancels. Frameworks fuse the two operations into one precisely so that the Jacobian is never built, which saves both memory and numerical stability (§3.2).'
      },
      {
        q: 'Reverse-mode autodiff is preferred for training because…',
        options: ['it is more accurate', 'the cost scales with the number of outputs, and a loss has exactly one', 'it uses less memory than forward mode', 'it works for non-differentiable functions'],
        answer: 1,
        why: 'The chain rule is a product of Jacobians, and matrix multiplication is associative, so you may bracket the product either way and get the same answer. Starting from the scalar end keeps a single row at every stage, so you only ever multiply a vector by a matrix; starting from the parameter end forces full matrix-by-matrix products until the last step. For a modest network this is a factor of a couple of hundred, and for a real one it is a factor of the parameter count. Option C is the answer that sounds most plausible and is actually backwards: reverse mode uses considerably <i>more</i> memory, because every activation computed on the forward pass has to be cached until the backward pass consumes it, which is why activation checkpointing exists (§3.11). Both modes are exact, so option A is a non-difference, and neither rescues a genuinely non-differentiable function.'
      },
      {
        q: 'Your analytic gradient is correct except that you forgot to divide by the batch size $n = 40$. Your gradient checker reports relative error $|g_{num} - g_{ana}|/(|g_{num}| + |g_{ana}|)$. Roughly what does it read?',
        options: ['about 0.02', 'about 0.33', 'about 0.95', 'about 40'],
        answer: 2,
        why: 'If the analytic gradient is the true one multiplied by a factor $k$, the relative error works out to $|1-k|/(1+k)$ no matter what the data is. Here $k = 40$, so the reading is $39/41 \\approx 0.95$. This is worth knowing because the number tells you what kind of mistake you made, not merely that you made one: a reading near $1/3$ means a factor of two, a reading of exactly 1 means a sign flip, and a reading that varies across coordinates means a missing term rather than a missing constant. Option D is the trap — the gradient is 40 times too large, so it feels as though the error should be 40. The measure is relative and bounded, which is exactly why it is used: it saturates near 1 for grossly wrong answers instead of producing a number whose size depends on your data.'
      }
    ],
    cards: [
      { q: 'Denominator layout, in one line', a: 'The derivative of a scalar w.r.t. $w$ has the same shape as $w$ — so it can be subtracted from it.' },
      { q: '$\\nabla_w \\|Xw-y\\|^2$', a: '$2X^\\top(Xw-y)$: the design matrix transposed, applied to the residuals.' },
      { q: 'Normal equations, and what ridge does to them', a: '$X^\\top X\\hat w = X^\\top y$; ridge makes it $(X^\\top X+\\lambda I)\\hat w = X^\\top y$, always invertible.' },
      { q: 'The backward pass of a dense layer $h = Wx$', a: 'With $g = \\partial\\mathcal{L}/\\partial h$: $\\partial\\mathcal{L}/\\partial W = g x^\\top$ and $\\partial\\mathcal{L}/\\partial x = W^\\top g$. No Jacobian is ever formed.' },
      { q: 'Why is reverse mode cheap?', a: 'The Jacobian product is associative; starting from the one-dimensional loss end keeps a vector at every stage instead of a matrix.' },
      { q: 'Why fuse softmax and cross-entropy?', a: 'The gradient collapses to $\\hat p - y$; computing them separately materialises a Jacobian that immediately cancels, and loses numerical stability.' },
      { q: 'Gradient check, correctly done', a: 'Central differences, float64, ε ≈ 1e-5, compare relative error. Below 1e-6 is right; above 1e-3 is a bug, usually a transpose.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.8 */
  ML.section({
    id: 'first-model', track: 'start', num: '0.8', level: 1,
    title: 'Your first model, end to end',
    lede: 'Every idea in Part 2 is an answer to a problem that shows up in this one workflow. Build it once here, badly and then properly, and the rest of the site has somewhere to attach.',
    prereq: ['what-is-ml'],
    related: ['supervised-setup', 'metrics', 'features'],
    html: `
${H.tldr([
      'The workflow is: <b>frame → split → baseline → features → fit → evaluate → threshold → ship → watch</b>. Skipping "baseline" and "threshold" is the most common way to waste a month.',
      'The split comes <b>before</b> anything that looks at the data. Scaling, imputation and encoding are all fitted on train and only applied to test — otherwise you are measuring a model that has already seen the answers.',
      'A model is not finished when the loss stops falling. It is finished when a <i>decision threshold</i> has been chosen against a cost, and someone has agreed what "worse" would look like in production.',
      'Two numbers are worth almost nothing on their own: a metric with no baseline, and a metric with no threshold. Report three — the baseline, the metric, and the decision it implies.'
    ])}

<p>Someone hands you a file. It has 40,000 rows, one per loan application from the past two years, and about thirty columns: income, requested amount, credit utilisation, months of history, number of recent enquiries, employment type, postcode district. The last column is called <code>defaulted</code>, and it is 1 for the 1,280 applications that went bad and 0 for the rest. The question is whether the bank should be using a model here rather than the rules it uses now.</p>

<p>You have everything you need. §0.1 gave you the three objects, §0.2 gave you the shapes, §0.7 gave you the gradient. So you split the file, fit a logistic regression, compute a score, and report back: <b>AUC 0.91</b>.</p>

<p>Now watch the meeting go wrong. The credit risk manager asks three questions, and none of them is unreasonable:</p>

${H.steps([
      '<b>Compared with what?</b> The bank already declines applications using a rule written in 2016. Is 0.91 better than that rule, or worse? You do not know, because you never scored the rule.',
      '<b>So how many applications do we decline?</b> A score is not a decision. Somewhere between "decline nobody" and "decline everybody" there is a cut-off, and 0.91 is silent about where it should go.',
      '<b>Will it still be 0.91 in March?</b> Your test set was a random sample of the same two years. It contains applications from after the ones you trained on and from before them, mixed together, which is not the situation the model will be in.'
    ])}

<p>Every one of those is a question about the workflow rather than about the model, and none of them is answered by making the model better. That is the point of this section. <mark>The model is the easy part, and it is a small part.</mark> What follows is the whole path from a file to a decision somebody is willing to act on, with each step justified by the specific failure it prevents.</p>

<h2><span class="sn">0.8.1</span> The nine steps, and what each one is defending against</h2>

<p>Here is the map. Read it once now for the shape of the thing; the sections after it walk the steps that carry the most weight.</p>

${H.table(['Step', 'What you do', 'What goes wrong if you skip it'], [
      ['1. Frame', 'Write the decision the model informs, and the cost of each kind of mistake, in one sentence each.', 'You optimise accuracy on a problem where a false negative costs 40× a false positive (§2.13).'],
      ['2. Split', 'Hold out test data <i>first</i>, by time if the future is what you predict (§2.14).', 'Every later number is optimistic and you find out in production.'],
      ['3. Baseline', 'Predict the majority class, or last week’s value. Record the score.', 'You have no idea whether your 0.84 AUC is good or embarrassing.'],
      ['4. Features', 'Encode, impute, scale — all <b>fitted on train only</b> (§2.11).', 'Leakage. The most common cause of a model that is brilliant offline and useless live.'],
      ['5. Fit', 'Start with logistic regression or gradient boosting. Not a neural network.', 'You spend three weeks on the wrong axis of the problem.'],
      ['6. Evaluate', 'On the metric from step 1, with an interval (§1.6), against the baseline from step 3.', 'You ship noise.'],
      ['7. Threshold', 'Pick the operating point that minimises expected cost, not 0.5 (§2.12).', 'A well-calibrated model making bad decisions.'],
      ['8. Ship', 'Shadow, then canary, then full (§5.12).', 'Your first real user is your test suite.'],
      ['9. Watch', 'Monitor input drift and outcome drift separately (§2.18).', 'Silent decay — the model keeps answering confidently while the world moves.']
    ])}

<h2><span class="sn">0.8.2</span> Framing: two sentences that determine everything downstream</h2>

<p>Framing sounds like the soft step and it is the one that decides most of the rest. It has two parts, and each is a single sentence you should be able to write before touching the data.</p>

<p><b>The decision.</b> Not "predict defaults" — that is a prediction, not a decision. Write down what changes as a result: <i>"For each incoming application, decide whether to send it to a human underwriter rather than approving it automatically."</i> Now the model has a job with an owner, a volume constraint and a downstream process, all of which turn out to matter.</p>

<p><b>The costs.</b> There are two ways to be wrong and they are almost never equally expensive. Missing a default means the bank lends money it does not get back; suppose the average loss on a bad loan is £4,000. Wrongly flagging a good applicant means an underwriter spends time on a file that did not need it, and some of those applicants go elsewhere; suppose that costs £120 in forgone margin and handling. The ratio is about 33 to 1.</p>

<p>Write that ratio down, because it will come back in §0.8.8 and set your decision threshold exactly. A team that never writes it down will default to a threshold of 0.5, which silently asserts that the two mistakes cost the same — a claim nobody in the room believes and nobody has noticed they are making.</p>

${H.analogy(`<p>A smoke alarm is a classifier with a threshold, and everyone already has strong intuitions about where it should sit. Set it too high and the house burns down. Set it too low and it screams during every dinner party, at which point somebody removes the battery and you now have no alarm at all.</p>
<p>Notice that the second failure is not merely annoying, it is a <i>model failure</i>: an alarm nobody listens to has an effective recall of zero regardless of what its sensor can detect. The same thing happens to fraud queues and clinical alerts. If you flag more cases than the team can work, the surplus is not triaged, it is ignored, and your measured recall stops describing what actually happens.</p>
<p>This is why "recall at a fixed alert budget" is often the honest metric. The budget is not a statistical quantity. It is how many files an underwriter can read in a day.</p>`)}

<h2><span class="sn">0.8.3</span> Split first, and split the way the future will</h2>

<p>The test set exists to answer one question: how will this model behave on data it has never seen? The moment anything you do is informed by the test data, it stops being able to answer that question, and there is no way to un-see it afterwards. So the split happens first, before plotting, before choosing features, before deciding which columns look interesting.</p>

<p>How you split matters as much as when. A random split is right when the rows are exchangeable — when there is no structure making one row predictive of another. Two structures break that, and both are common:</p>

<p><b>Time.</b> If the model will be used on future applications, a random split lets it train on March and test on February, which is a strictly easier problem than the real one. Economic conditions move, product mixes change, and a model can learn a pattern that only existed in one quarter. Splitting by date — train on the first eighteen months, test on the last six — measures the thing you actually care about.</p>

<p><b>Groups.</b> If the same customer appears in twelve rows and you split randomly, some of that customer's rows land in train and some in test. The model can then recognise the customer rather than the pattern, and your test score is partly a memory test. The fix is to split on the group key, so every row belonging to one customer goes to the same side.</p>

${H.pitfall(`<p>Group leakage is the version of this that most often survives review, because the split code looks correct. It is worth checking directly rather than assuming: count how many identifiers appear on both sides of your split, and if that number is not zero, your test score is inflated by an amount nobody can estimate.</p>
<p>The signature is a test score that is excellent and a production score that is mediocre, with no obvious bug anywhere. On problems with heavy repeat customers this can be worth ten points of AUC, which is more than the difference between a good model and a bad one.</p>`)}

<h2><span class="sn">0.8.4</span> The baseline, which takes ten minutes and reframes everything</h2>

<p>Before fitting anything, score the dumbest thing that could work. On this dataset the base rate is 3.2 per cent, so a model that predicts "will not default" for every single application is <b>96.8 per cent accurate</b>. If your first instinct was to report accuracy, that number should end the instinct permanently.</p>

<p>The right baselines depend on the problem, and it is worth having two or three:</p>

${H.table(['Baseline', 'What it is', 'What it tells you'], [
      ['Majority class', 'Always predict the common outcome', 'The accuracy floor. Usually embarrassingly high on imbalanced data'],
      ['Base rate', 'Predict the overall positive rate for everyone', 'The average-precision floor — here 0.032'],
      ['The incumbent', 'The rule or process already in production', 'The only baseline your stakeholder actually cares about'],
      ['One feature', 'Logistic regression on the single most obvious column', 'How much of the signal is already in the obvious place']
    ])}

<p>The incumbent baseline is the one teams skip and the one that changes conversations. A model that scores 0.91 against a 2016 rule scoring 0.88 is a very different proposal from one scoring 0.91 against a rule scoring 0.62, and the difference is invisible in the model's own metrics. Score the existing process on the same test set, with the same metric, on the same day.</p>

${H.key('A metric without a baseline is not a result. It is a number.')}

<h2><span class="sn">0.8.5</span> Features, and the two ways they leak</h2>

<p>Leakage means the model saw information at training time that will not be available at prediction time. It is the single most common reason a model is brilliant offline and useless live, and it comes in two quite different flavours with completely different signatures.</p>

<p><b>Target leakage</b> is a column that is a consequence of the label rather than a cause of it. In the loan file, imagine a column called <code>days_since_default_letter</code>. For applications that defaulted it holds a small number; for the rest it is large or missing. It is enormously predictive, and it is worthless, because at the moment you have to make a decision no letter has been posted. The signature is unmistakable: a suspiciously excellent score, often above 0.99, from a feature that turns out to be the answer in disguise.</p>

<p>The defence is a question asked of every column: <i>at the instant the decision is made, does this value exist yet?</i> Anything derived from an event that happens after the decision fails, and so does anything computed from an aggregate over a window that extends past it.</p>

<p><b>Preprocessing leakage</b> is subtler and far more common. If you fit a scaler, an imputer or an encoder on the whole dataset before splitting, the parameters of those transforms — a mean, a standard deviation, a set of category levels, a median used to fill gaps — have been computed partly from the test rows.</p>

${H.intuition(`<p>Why does fitting the scaler on all the data leak? Because the mean and standard deviation of the test set are facts about the test set. A model whose inputs were centred using the test mean has been told something about the test distribution that it will not know at prediction time. The effect is usually small — a fraction of a point of AUC — which is exactly what makes it dangerous: it is big enough to change which model you pick and small enough to never look suspicious.</p>
<p>Compare the two signatures and the asymmetry becomes clear. Target leakage announces itself with an implausible score, and somebody will investigate. Preprocessing leakage moves the third decimal place, survives every review, and quietly biases every model comparison you make in the same direction — so the model that wins your bake-off may be the one that benefited most from it rather than the one that is best.</p>`)}

<h2><span class="sn">0.8.6</span> Fitting, and the pipeline that makes leakage structurally impossible</h2>

<p>The fix for preprocessing leakage is not vigilance. Vigilance fails, because the transformation has to be repeated identically at training time, at cross-validation time, at evaluation time and at serving time, and a human repeating four things by hand will eventually repeat three. The fix is to make the whole chain a single object that can only be fitted on what it is given.</p>

${H.code(`import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)

num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))])

# The Pipeline is not a style preference. It is what makes "fit on train only"
# structurally impossible to get wrong.
pipe = Pipeline([
    ("prep", ColumnTransformer([("n", num, num_cols), ("c", cat, cat_cols)])),
    ("clf",  LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(X_tr, y_tr)

p = pipe.predict_proba(X_te)[:, 1]
print("AUC", roc_auc_score(y_te, p), "AP", average_precision_score(y_te, p))
print("baseline AP", y_te.mean())          # always print the baseline`)}

<p>Three lines in that block are doing more than they appear to. <code>stratify=y</code> forces the split to preserve the 3.2 per cent positive rate on both sides, which matters when positives are rare enough that a careless random split could hand you noticeably different base rates. <code>handle_unknown="ignore"</code> tells the encoder what to do when a category appears at serving time that was absent from training, which will happen, and which otherwise raises an exception in production rather than during your tests. And <code>class_weight="balanced"</code> reweights the loss so that the 1,280 defaults are not simply drowned by the 38,720 repayments.</p>

${H.key('Use a Pipeline object, not a sequence of transformations you remember to repeat. Leakage is not usually a conceptual error — it is a bookkeeping error, and the fix is structural.')}

${H.flag('The advice to start with logistic regression or gradient boosting rather than a neural network is deliberate and it is also contested. Boosted trees have been the reliable default on tabular data for a decade, and several careful comparisons still favour them; more recent work on tabular foundation models and specialised deep architectures has closed the gap on some benchmarks. The advice survives anyway, for a reason that is not really about accuracy: a linear model or a boosted tree fits in seconds, exposes which features carry the signal, and fails in ways you can read. That is worth more in week one than a point of AUC, and you can always revisit the choice once the workflow around it is sound (§2.8).')}

<h2><span class="sn">0.8.7</span> Evaluating: which number, and what it means</h2>

<p>Accuracy is out, for the reason established in §0.8.4. The two metrics that replace it on a problem like this are worth understanding rather than merely selecting.</p>

<p><b>AUC</b> — the area under the receiver operating characteristic curve — has an interpretation that makes it much less mysterious than the name suggests. <mark>Take one random defaulter and one random non-defaulter. AUC is the probability that the model gives the defaulter the higher score.</mark> So 0.5 is a coin flip, 1.0 is perfect ranking, and 0.91 means that in 91 such pairings out of 100 the model puts them in the right order. Notice what this definition does not mention: any threshold. AUC is a property of the ranking alone.</p>

<p><b>Average precision</b> summarises the precision–recall curve instead, and it is the one to prefer when positives are rare. Its floor is the base rate, which here is 0.032, so an AP of 0.30 represents roughly a ninefold improvement on guessing — a statement AUC cannot make, because AUC's floor is 0.5 regardless of how imbalanced the problem is.</p>

<p>Put those together with a plausible operating point to see what the model is actually offering. Suppose you decline the top 5 per cent of applications by score, which is 2,000 files, and that this captures 60 per cent of the eventual defaults, which is 768 of the 1,280. Precision is $768/2000 = 38.4$ per cent, recall is 60 per cent, and the lift over the base rate is $0.384/0.032 = 12$. One in every 2.6 declined applications would genuinely have defaulted, against one in 31 if you had declined at random. That is a sentence a credit risk manager can act on, and it took no new modelling at all — only a threshold and the base rate.</p>

${H.pitfall('Reporting accuracy on an imbalanced problem. At 1% positives, a model that predicts "no" forever is 99% accurate and worth nothing. If you catch yourself reporting accuracy, ask what the base rate is — and then report average precision or recall at a fixed budget instead.')}

<h2><span class="sn">0.8.8</span> The threshold, which is where the cost finally arrives</h2>

<p>The model produces a probability. The business needs a decision. Converting one into the other requires a cut-off, and the cut-off is not a modelling choice — it is an economic one, and it follows from the two costs you wrote down in §0.8.2.</p>

<p>The reasoning is short enough to do in full, and doing it once means never guessing again.</p>

${H.deriv('where the cost-optimal threshold comes from', [
      ['$p$ — the predicted probability that this application defaults', 'This is the model output, and by assumption it is calibrated — $p = 0.03$ really does mean about three in a hundred such applicants default. Calibration is doing real work here and §2.12 is about when you have it.'],
      ['Cost of declining $= (1-p)\\,c_{FP}$', 'If you decline and they would have repaid, which happens with probability $1-p$, you lose the margin $c_{FP} = 120$. If they would have defaulted, declining costs you nothing.'],
      ['Cost of approving $= p\\,c_{FN}$', 'If you approve and they default, which happens with probability $p$, you lose $c_{FN} = 4000$. If they repay, approving was the right call and costs nothing.'],
      ['Decline when $(1-p)\\,c_{FP} < p\\,c_{FN}$', 'Choose whichever action has the lower expected cost. This is the entire decision rule, and it is per application rather than global.'],
      ['$c_{FP} < p\\,(c_{FN} + c_{FP})$', 'Expand the left side to $c_{FP} - p\\,c_{FP}$ and move that term across. Both costs are positive, so no inequality flips.'],
      ['$p > \\dfrac{c_{FP}}{c_{FP} + c_{FN}} = t^{\\star}$', 'Divide through by the positive quantity $c_{FN} + c_{FP}$. The threshold depends only on the <i>ratio</i> of the two costs, not on their absolute size — doubling both currencies changes nothing.']
    ], 'For this problem $t^{\\star} = 120/(120+4000) = 0.029$, not 0.5. A threshold of 0.5 is optimal in exactly one situation: when the two mistakes cost the same. Whenever you see 0.5 used without comment, the team has asserted that, usually without realising it.')}

<p>Two things are worth extracting from that derivation. The first is the direction: <b>expensive misses push the threshold down</b>, so you alert more often. That is the opposite of the instinct many people have, which is that a costly error calls for more caution before acting — but here the costly error is the failure to act.</p>

<p>The second is that the threshold is now a number somebody can argue with. If the credit team believes a bad loan costs £6,000 rather than £4,000, the threshold moves to $120/6120 = 0.020$ and the alert volume rises. That is a productive argument to be having, and it is only possible because the cost assumption is written down instead of buried in a default parameter.</p>

<h2><span class="sn">0.8.9</span> Breaking the pipeline on purpose</h2>

<p><b>What you are looking at.</b> The lab below fits a real logistic regression, by gradient descent, on generated data shaped like a credit-risk problem: four honest features — income, credit utilisation, age and recent enquiries — plus one poisoned one, held back unless you ask for it. The plot is an ROC curve, with false positive rate across and true positive rate up. The solid, shaded curve is the model's performance on <b>held-out</b> data; the faint dashed curve is its performance on the data it trained on; the straight diagonal is what random guessing looks like. The single dot is your current operating point, and it moves along the curve as you change the threshold. The readout gives test AUC and AP, the baseline AP, precision and recall at your operating point, alerts per thousand applications, and the expected cost.</p>

<p><b>What to do with it.</b> Start by switching on <i>keep a feature built from the label</i>. That adds the poisoned column, which is "days since the default letter was posted" — roughly 9 days for accounts that defaulted and roughly 60 for those that did not. The AUC leaps towards 1.0 and the model becomes worthless in the same instant, because at the moment of decision no letter exists. Switch it off again and drag the <i>decision threshold</i> slider from end to end. Watch two readouts as you do: the AUC does not move at all, while the expected cost changes by a factor of several. Then set the cost ratio to 10× and hunt for the threshold that minimises expected cost — it should land near 0.09, which is $1/(1+10)$ as the derivation above predicts.</p>

<p><b>The thing genuinely worth noticing.</b> Now try <i>scale using ALL the data</i>, the preprocessing-leakage toggle, and watch how little happens. The curve barely moves; the AUC changes in the third decimal if at all. That is not a failure of the demonstration, it is the demonstration. Target leakage is loud and somebody will catch it. Preprocessing leakage is inaudible, and it biases every comparison you make in the same direction, which is precisely why the defence has to be structural rather than watchful. The readout labelled <i>AUC you would report</i> switches to the training-set number whenever either mistake is switched on, standing in for the figure a leaky workflow hands you — and the gap between that and the honest test number is the thing your stakeholder eventually experiences.</p>

${H.lab('pipeline', 'The pipeline sandbox — break it on purpose', 'A real logistic regression on generated customer data, retrained every time you flip a switch. Each toggle is a specific mistake from the table above. Watch the gap between what you would have reported and what the model actually does on held-out data.')}

<h2><span class="sn">0.8.10</span> Shipping, watching, and the three numbers to report</h2>

<p>A model that has never met live traffic has an unknown failure mode, so the last two steps are about finding it cheaply. <b>Shadow mode</b> runs the model on real requests and records what it would have done without acting on it, which catches the entire class of problems where a feature is computable offline and unavailable at serving time. A <b>canary</b> then routes a small fraction of real decisions through it. Only after both does the model take full traffic (§5.12).</p>

<p>Then you watch, and it is worth watching two different things because they fail on different timescales. <b>Input drift</b> is a change in the distribution of features arriving, and you see it immediately. <b>Outcome drift</b> is a change in the relationship between features and the label, and you cannot see it until the labels arrive — which for a loan book might be nine months. A model can therefore be quietly wrong for two quarters while every input monitor stays green (§2.18).</p>

<p>Finally, the report. Whatever the project, three numbers make it interpretable and any two of them do not:</p>

${H.steps([
      '<b>The baseline.</b> Majority class, or the current rule-based system. Without it, no number means anything.',
      '<b>The metric that matches the cost</b>, with an interval. Average precision if positives are rare; recall at a fixed alert budget if a team has to work the alerts; calibrated probability if a downstream system multiplies by a monetary amount (§2.13).',
      '<b>The decision.</b> At the chosen threshold: how many alerts per day, what fraction are right, and what the expected cost is versus the baseline. This is the only one your stakeholder actually needed.'
    ])}

<p>For the loan problem, the whole result fits in one sentence: <i>"At a threshold of 0.029 the model sends 2,000 of 40,000 applications to an underwriter, catching 768 of the 1,280 defaults; 38 per cent of the files it flags are genuine against a base rate of 3.2 per cent, and the 2016 rule, given the same 2,000 files to work with, catches 512."</i> That sentence contains a baseline, a metric, a volume and a threshold, and every number in it came from the two costs written down at the start. The AUC of 0.91 does not appear in it at all, which is a fair reflection of how much it was ever going to decide.</p>

${H.practice(`<p>Three things go wrong on real projects far more often than anything in the modelling, and none of them is visible in a notebook.</p>
<p><b>The label is ambiguous.</b> "Defaulted" turns out to mean ninety days past due in one system and written off in another, and the two populations differ by a factor of three. Before any modelling, find out exactly how the label column was produced and by whom. Half of all confusing results are a label definition nobody checked.</p>
<p><b>The rows are not independent.</b> Duplicated records from a botched join, the same customer under two identifiers, resubmitted applications — all of these inflate your test score and none of them is visible in a summary statistic. Count distinct identifiers, compare with the row count, and look at the difference.</p>
<p><b>The base rate is not stable.</b> If the positive rate in your training window is 3.2 per cent and in production it is 5 per cent, a model calibrated on the first will systematically under-predict on the second. Plot the label rate by month before you split; a visible trend is a reason to split by time and to re-calibrate periodically rather than once (§2.12).</p>`)}

${H.history(`<p>The clearest demonstration that the model is the small part came from the Netflix Prize, which ran from 2006 to 2009 and offered a million dollars for a ten per cent improvement in rating prediction.</p>
<p>The prize was won, by a blend of over a hundred models assembled by several merged teams. Netflix then declined to deploy the winning ensemble, saying publicly that the accuracy gain did not justify the engineering effort required to run it — and that in the meantime their business had moved from mailing DVDs to streaming, which changed what a good recommendation even was.</p>
<p>Nothing about the modelling was wrong. What the competition had removed, in order to be a competition at all, was the framing, the deployment constraint and the possibility that the objective might change. Those are precisely steps 1, 8 and 9 of the workflow above, and they are the steps that decided the outcome.</p>`)}

${H.probe([
      ['A colleague scaled the features before splitting. How bad is it?', 'Usually a small optimistic bias, occasionally a large one — but the real problem is that you can no longer trust the comparison between models. Refit inside a Pipeline and re-run.'],
      ['What is your first model on a new tabular problem?', 'A baseline, then logistic regression with sensible encoding, then gradient boosting. Neural networks rarely win on tabular data and cost far more to debug (§2.8).'],
      ['You have 0.91 AUC. Ship it?', 'Not yet. What is the baseline, what is the operating threshold, what does a false positive cost, and is the probability calibrated if anything downstream multiplies by it?'],
      ['How would you split a dataset where each customer appears many times?', 'By customer, not by row. Otherwise the same customer sits on both sides of the split and the test score partly measures memorisation. Check afterwards that no identifier appears in both halves.'],
      ['A miss costs 33× a false alarm. Where does the threshold go, and why that direction?', 'To about $1/34 \\approx 0.03$. Expensive misses push the threshold <i>down</i>, because the costly error here is failing to act, so you act more readily.']
    ], 'Answering "I would try a few models and pick the best AUC". It skips framing, baseline and threshold — the three steps that decide whether the work is worth anything — and it tells the interviewer you have not shipped one.')}`,
    labs: {
      pipeline: function (host) {
        const st = Viz.controls(host, [
          { k: 'leak', label: 'scale using ALL the data (leak)', type: 'toggle', value: false },
          { k: 'target', label: 'keep a feature built from the label', type: 'toggle', value: false },
          { k: 'thr', label: 'decision threshold', min: .05, max: .95, step: .01, value: .5, fmt: v => v.toFixed(2) },
          { k: 'cost', label: 'cost of a miss ÷ cost of a false alarm', min: 1, max: 40, step: 1, value: 10, fmt: v => v + '×' }
        ], () => S.redraw());

        const out = Viz.readout(host, [
          { k: 'auc', label: 'test AUC', cls: 'key' },
          { k: 'ap', label: 'test AP' },
          { k: 'base', label: 'baseline AP' },
          { k: 'rep', label: 'AUC you’d report', cls: 'warn' },
          { k: 'alerts', label: 'alerts / 1000' },
          { k: 'prec', label: 'precision there' },
          { k: 'rec', label: 'recall there' },
          { k: 'exp', label: 'expected cost', cls: 'bad' }
        ]);

        /* a small credit-risk-shaped dataset: 4 honest features, one poisoned */
        function makeData(n, seed) {
          const R = Num.rng(seed);
          const X = [], y = [];
          for (let i = 0; i < n; i++) {
            const income = R.normal(60, 22);
            const util = Math.min(1.4, Math.max(0, R.beta(2, 3) * 1.2));
            const age = 22 + R.gamma(4) * 6;
            const inq = R.poisson(1.2);
            const z = -2.4 - 0.028 * (income - 60) + 2.6 * util + 0.30 * inq - 0.02 * (age - 40);
            const p = Num.sigmoid(z);
            const label = R() < p ? 1 : 0;
            // the poisoned feature: "days since the default letter was posted"
            const leaky = label ? R.normal(9, 3) : R.normal(60, 20);
            X.push([income, util, age, inq, leaky]);
            y.push(label);
          }
          return { X: X, y: y };
        }
        const tr = makeData(700, 3), te = makeData(700, 99);

        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const cols = st.target ? 5 : 4;
            const cut = arr => arr.map(r => r.slice(0, cols));
            const Xtr = cut(tr.X), Xte = cut(te.X);

            // scaling statistics: honest (train only) or leaked (train+test)
            const pool = st.leak ? Xtr.concat(Xte) : Xtr;
            const mu = [], sd = [];
            for (let j = 0; j < cols; j++) {
              const col = pool.map(r => r[j]);
              mu.push(Num.mean(col)); sd.push(Num.sd(col) || 1);
            }
            const scale = M => M.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
            const Ztr = scale(Xtr), Zte = scale(Xte);

            const model = Num.logistic(Ztr, tr.y, { lr: .35, l2: 1e-3 });
            model.step(600);

            const ptr = Ztr.map(r => model.predict(r));
            const pte = Zte.map(r => model.predict(r));
            const rocTe = Num.rocCurve(pte, te.y);
            const rocTr = Num.rocCurve(ptr, tr.y);
            const baseAP = Num.mean(te.y);

            const cm = Num.confusion(pte, te.y, st.thr);
            const missCost = st.cost, faCost = 1;
            const expected = (cm.fn * missCost + cm.fp * faCost) / te.y.length;

            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 48, r: 16, t: 16, b: 40 } })
              .frame({ xlabel: 'false positive rate', ylabel: 'true positive rate' });
            P.clip(() => {
              P.line([[0, 0], [1, 1]], { color: T.faint, dash: [4, 4], width: 1 });
              P.line(rocTr.roc, { color: T.faint, width: 1.6, dash: [5, 3] });
              P.area(rocTe.roc, { color: T.c1, alpha: .12 });
              P.line(rocTe.roc, { color: T.c1, width: 2.6 });
              const fpr = cm.fp / (cm.fp + cm.tn || 1), tpr = cm.tp / (cm.tp + cm.fn || 1);
              P.dots([[fpr, tpr]], { r: 5.5, color: T.c2, stroke: true, strokeWidth: 2 });
              P.text(fpr, tpr, ' operating point', { dx: 8, color: T.c2, font: '11px ui-sans-serif' });
            });

            out({
              auc: rocTe.auc.toFixed(3),
              ap: rocTe.ap.toFixed(3),
              base: baseAP.toFixed(3),
              rep: (st.leak || st.target ? rocTr.auc : rocTe.auc).toFixed(3),
              alerts: Math.round(1000 * (cm.tp + cm.fp) / te.y.length),
              prec: (cm.precision * 100).toFixed(1) + '%',
              rec: (cm.recall * 100).toFixed(1) + '%',
              exp: expected.toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c1)', t: 'held-out test ROC' },
          { c: 'var(--faint)', t: 'training ROC (what leakage lets you believe)' },
          { c: 'var(--c2)', t: 'your chosen operating point' }
        ]);
        Viz.note(host, 'Turn on <b>keep a feature built from the label</b>: AUC jumps toward 1.0 and the model is worthless, because at prediction time nobody has posted the default letter yet. Then leave it off and drag the threshold — notice that the AUC never moves, because AUC is threshold-free, while expected cost changes by a factor of three. <b>AUC ranks; thresholds decide.</b> The cost-optimal threshold sits near $1/(1+\\text{cost ratio})$, so at 10× it is about 0.09, not 0.5.');
      }
    },
    quiz: [
      {
        q: 'You fit a StandardScaler on the full dataset before splitting. This is…',
        options: ['fine, scaling is not learning', 'leakage — the test distribution has informed the transform', 'only a problem for neural networks', 'good practice for stability'],
        answer: 1,
        why: 'A scaler has parameters — a mean and a standard deviation per column — and fitting it on the full dataset estimates those parameters partly from rows you are about to be scored on. The model has therefore been told something about the test distribution that it will not know at prediction time, when a single new application arrives with no distribution attached. Option A is the tempting answer and the reasoning behind it is half right: scaling is not learning a decision rule. But it is still <i>fitting</i> something, and anything fitted must be fitted on training data alone. What makes this failure worth a whole convention is its size. The bias is usually a fraction of a point of AUC, which is small enough to survive every review and large enough to change which model wins your comparison — so the model you ship may be the one that leaked most rather than the one that is best.'
      },
      {
        q: 'A fraud model has AUC 0.94 and you must choose a threshold. A miss costs 20× a false alarm. Roughly where does the cost-optimal threshold sit?',
        options: ['0.5, always', 'near 0.05', 'near 0.95', 'thresholds do not affect cost'],
        answer: 1,
        why: 'Compare the two expected costs at a case with predicted probability $p$. Acting costs $(1-p)\\,c_{FP}$, because you only lose anything if the case was innocent; not acting costs $p\\,c_{FN}$. Acting is cheaper when $p > c_{FP}/(c_{FP}+c_{FN})$, which with a 20-to-1 ratio is $1/21 \\approx 0.048$. Option A is the answer that gets given by default, and it is worth seeing why it is a claim rather than a neutral choice: a threshold of 0.5 is cost-optimal in exactly one situation, when the two mistakes cost the same amount. The direction also catches people out. Expensive misses push the threshold <i>down</i>, so you alert more often, because here the costly error is the failure to act. Note that none of this touched the AUC: 0.94 describes the ranking, and the threshold decides what to do with it (§2.12).'
      },
      {
        q: 'Which number should always accompany a reported metric?',
        options: ['the training loss', 'the baseline that metric is being compared against', 'the number of epochs', 'the random seed'],
        answer: 1,
        why: 'Without a baseline, 0.84 is a number rather than a result, because nothing in it says whether the problem was easy or hard. On a dataset with 3 per cent positives, 96.8 per cent accuracy is what you get by predicting "no" forever, and an average precision of 0.03 is what you get from guessing — so a metric can look impressive purely because of the base rate. The baseline that matters most is rarely the statistical one: it is the rule or process already running in production, scored on the same test set with the same metric on the same day. Option D is the strongest distractor because seeds genuinely do matter for reproducibility and for knowing how much of a difference is noise. But a seed makes your number repeatable, whereas a baseline makes it <i>interpretable</i>, and only one of those turns a metric into a claim somebody can act on.'
      },
      {
        q: 'A loan model is trained on rows where the same customer can appear a dozen times, and the data is split at random. What is the most likely consequence?',
        options: ['nothing — random splits are always safe', 'the test score is inflated, because the model can recognise customers it has already seen', 'the model will underfit', 'the base rate will differ between the two halves'],
        answer: 1,
        why: 'With a random split, some of a given customer’s rows land in training and some in test. The model can then key on whatever makes that customer identifiable rather than on the pattern you wanted, and your test set has quietly become partly a memory test. The right fix is to split on the group key so every row belonging to one customer goes to the same side, and then to verify it by checking that no identifier appears in both halves. The signature in production is distinctive: an excellent offline score, a mediocre live one, and no bug anywhere in the code, because the split itself looked correct. On problems with heavy repeat customers this is worth far more than the difference between a good model and a bad one, which is why it belongs to step 2 of the workflow rather than to debugging.'
      },
      {
        q: 'Your model scores AUC 0.91 on data it has never seen, and the leakage checks are clean. What is still missing before this is a result?',
        options: ['a larger model', 'a baseline, a threshold, and the cost of each kind of mistake', 'more training epochs', 'a confusion matrix at 0.5'],
        answer: 1,
        why: 'AUC is a property of the ranking alone: it is the probability that a randomly chosen positive is scored above a randomly chosen negative, and it mentions no threshold at all. So a good AUC tells you the model can sort cases, and says nothing about whether sorting them is worth doing or what to do with the sorted list. Three things convert it into a result. A baseline says whether 0.91 beats what already exists. A cost ratio says which of the two mistakes you would rather make. A threshold, derived from that ratio, turns scores into decisions and finally produces the numbers a stakeholder can act on — volume, precision and expected cost. Option D is the near miss: a confusion matrix is exactly the right kind of object, but computing it at 0.5 rather than at the cost-optimal point simply picks an arbitrary operating point and reports it with confidence.'
      }
    ],
    cards: [
      { q: 'The nine steps', a: 'Frame → split → baseline → features → fit → evaluate → threshold → ship → watch.' },
      { q: 'Why use a Pipeline object?', a: 'Leakage is a bookkeeping error; a Pipeline makes "fit on train only" structural rather than remembered.' },
      { q: 'Cost-optimal threshold', a: '$t^\\star \\approx c_{FP}/(c_{FP}+c_{FN})$ — expensive misses drive the threshold down.' },
      { q: 'Why is accuracy the wrong metric at 1% positives?', a: 'Predicting "no" forever scores 99%. Use average precision, or recall at a fixed alert budget.' },
      { q: 'What does AUC actually mean?', a: 'The probability that a random positive is scored above a random negative. It measures ranking, and mentions no threshold.' },
      { q: 'The two kinds of leakage', a: 'Target leakage is a column caused by the label — loud and obvious. Preprocessing leakage is a transform fitted on all the data — quiet and biases every comparison.' },
      { q: 'The three numbers to report', a: 'The baseline, the metric that matches the cost, and the decision the chosen threshold implies.' }
    ]
  });
})();
