/* ============================================================
   PART 0 — Start here
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 0.1 */
  ML.section({
    id: 'what-is-ml', track: 'start', num: '0.1',
    title: 'What machine learning actually is',
    lede: 'Almost every machine learning system ever built — from a two-parameter straight line to a model with hundreds of billions of parameters — is assembled from the same three pieces: a model that makes guesses, a loss that scores how wrong those guesses were, and an optimiser that adjusts the model to improve the score. This section builds all three from nothing, and everything that follows on this site is a variation on them.',
    html: `
<p>Let us begin with a problem you already know how to think about, and then walk to the exact point where ordinary programming runs out of road.</p>

<p>Suppose you have to write a program that decides whether an email is spam. You sit down and start writing rules. Messages containing the phrase "free money" are probably spam, so that becomes your first rule. Then you notice that legitimate emails from your bank occasionally mention money, so you add an exception. Then the spammers start writing "fr€€ m0ney", so you add another rule to catch that. Then they find a third spelling. Every rule you add fixes one case and quietly breaks another, the program swells to thousands of lines, and it is never quite right.</p>

<p>Now notice something odd about the situation you are in. <b>You can look at any individual email and tell almost instantly whether it is spam.</b> You are not confused about the answer. What you cannot do is write down the rule you are using to decide, because you are not really following a rule at all. You are drawing on something absorbed from having seen thousands of examples.</p>

<p>That gap between being able to recognise an answer and being able to state the rule is exactly where machine learning lives.</p>

${H.key('Machine learning is for problems where you can recognise the correct answer but cannot write down the rule that produces it.')}

<p>The consequence is a genuine reversal of who supplies what. It is worth laying the two approaches side by side, because the difference changes not only how you build the system but how you debug it when it goes wrong:</p>

${H.table(['', 'Ordinary programming', 'Machine learning'], [
      ['<b>You supply</b>', 'the rule', 'examples of correct answers'],
      ['<b>The computer supplies</b>', 'the answers, by applying your rule', 'the rule, by searching for one that reproduces your examples'],
      ['<b>You debug by</b>', 'reading the code', 'examining the data and the errors it makes'],
      ['<b>It fails when</b>', 'you wrote the rule wrongly', 'your examples did not represent what the system meets later']
    ])}

<p>The phrase "searching for a rule" is carrying a great deal of weight in that table, so let us make it concrete straight away. The computer does not conjure rules out of nothing. You hand it an entire <i>family</i> of possible rules — every straight line, say, or every decision tree of depth five, or every neural network of a particular shape — and its job is to find the member of that family which best reproduces your examples. Choosing the family is your decision as the engineer. Finding the best member of it is the machine's work.</p>

${H.analogy(`<p>Think of tuning an old analogue radio. The family of possible rules is every position the dial can occupy. Your examples are the station you are trying to hear. You did not build the radio and you did not invent the frequency; you simply turn the dial until the static clears and music comes through. Machine learning is that same act, with millions of dials instead of one, and with a precise mathematical definition of what "the static cleared" means.</p>`)}

<h2><span class="sn">0.1.1</span> The three objects, taken one at a time</h2>

<p>Every system in this course is assembled from exactly three parts. That is true of the straight line you will be dragging around in a few minutes and equally true of the models behind the assistants you use daily. It is worth meeting the three slowly now, because they recur in every section that follows, and before long you will find yourself identifying them in a research paper before you have understood anything else about it.</p>

<h3>1. The model is a function with adjustable numbers inside it</h3>

<p>A <b>model</b> is simply a function. An input goes in, a prediction comes out. What makes it a <i>learning</i> model rather than an ordinary function is that some of the numbers inside it are not fixed in advance. Those adjustable numbers are called <b>parameters</b>, and by long convention we gather them all into a single symbol, $\\theta$ (the Greek letter theta). The model is then written $f_\\theta(x)$, which you can read aloud as "f, tuned by theta, applied to x".</p>

<p>Here is the smallest example that is still genuinely a model. A straight line:</p>

$$f_\\theta(x) = wx + b, \\qquad \\theta = (w, b)$$

<p>It has two parameters, the slope $w$ and the intercept $b$. Choose different values for those two numbers and you get a different line, which makes a different prediction for the very same input. In this case the family of candidate rules is "all straight lines", and selecting one member of the family amounts to choosing two numbers.</p>

<p>Now scale that idea up, and notice that conceptually nothing changes. A large language model is also a function with adjustable numbers inside it. It simply has hundreds of billions of them rather than two, and its input is a sequence of text rather than a single value. <mark>The distance from two parameters to two hundred billion is a difference of scale, not of kind.</mark></p>

<h3>2. The loss says how wrong a prediction was</h3>

<p>If the computer is going to search for a good rule, it needs a definition of "good" precise enough to be computed. That definition is the <b>loss function</b>, written $\\ell(f_\\theta(x), y)$. It takes the model's prediction alongside the correct answer and returns a single number describing how bad that prediction was. Small is good, and zero means perfect.</p>

<p>When the thing being predicted is a number, the usual choice is squared error, $(\\hat{y} - y)^2$, where $\\hat{y}$ is shorthand for whatever the model predicted. If you predict 7 and the true answer was 10, the loss is $(7-10)^2 = 9$. Predict 9 instead and the loss falls to 1. The squaring accomplishes two separate things worth noticing. It makes the direction of the error irrelevant, so missing by 3 too high and 3 too low count equally. And it punishes large errors out of proportion to small ones, so a miss of 10 is a hundred times worse than a miss of 1 rather than merely ten times worse.</p>

<p>The loss on a single example is not yet the thing we minimise. We average it across every example we have, and that average is the objective:</p>

$$L(\\theta) = \\frac{1}{n}\\sum_{i=1}^{n} \\ell\\big(f_\\theta(x_i),\\, y_i\\big)$$

<p>It is worth reading that formula slowly, because its shape recurs throughout the course. The sum runs over your $n$ examples. For each one you compare the model's prediction against the truth and score it. You average those scores. The result, $L(\\theta)$, is a single number that depends on the parameters: change $\\theta$ and the number changes. Training is the act of hunting for the $\\theta$ that makes it small.</p>

${H.note('Notice that the loss is chosen rather than discovered. Squared error is not the only reasonable way to score a numerical prediction, and the choice carries real consequences — it is precisely what makes a model sensitive to outliers, for instance. §1.5 shows that the standard losses are not arbitrary conveniences at all: each one falls out of a specific assumption about how the noise in your data behaves.')}

<h3>3. The optimiser is the procedure that improves the parameters</h3>

<p>We now have a family of candidate rules and a way of scoring any member of it. What remains is a method for actually locating a good member. For anything beyond the smallest problems, simply checking every candidate is hopeless, because the space of possible parameter settings is infinite.</p>

<p>The idea that rescues us is this: for whatever parameters we currently hold, we can usually work out <i>which direction to nudge them so that the loss goes down</i>. That direction is called the <b>gradient</b>, written $\\nabla_\\theta L$. The optimiser takes a small step in the opposite direction of the gradient, then recomputes and repeats:</p>

$$\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta L$$

<p>The arrow means "replace what is on the left with what is on the right". The symbol $\\eta$ (eta) is the <b>learning rate</b>, which controls how large a step to take. Steps that are too small make training take forever. Steps that are too large overshoot the bottom, so the parameters bounce around the valley or, in the worst case, fly off entirely.</p>

${H.analogy(`<p>The standard picture is walking down a hill in thick fog. You cannot see the valley floor, so you cannot simply head straight for it. But you can feel which way the ground slopes beneath your feet, so you take a step in the downhill direction, feel again, and repeat. The gradient is the slope you feel underfoot. The learning rate is the length of your stride. It is a decidedly unglamorous procedure, and it is how very nearly every model in this course is trained.</p>`)}

<p>Those are the three objects. Here they are collected together, which is the form worth committing to memory:</p>

${H.table(['Object', 'Symbol', 'What it does', 'Developed in'], [
      ['<b>Model</b>', '$f_\\theta(x)$', 'Maps an input to a prediction using adjustable parameters $\\theta$', 'Parts 2, 3 and 4'],
      ['<b>Loss</b>', '$\\ell(f_\\theta(x), y)$', 'Scores how wrong one prediction was; averaged over the data to give the objective $L(\\theta)$', '§1.5 and §2.1'],
      ['<b>Optimiser</b>', '$\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta L$', 'Repeatedly nudges the parameters in whichever direction lowers the loss', '§1.9 and §3.4']
    ])}

<h2><span class="sn">0.1.2</span> The three kinds of learning problem</h2>

<p>Machine learning problems are conventionally sorted into three families. The thing that separates them is not the algorithms involved but something more basic: <i>what information you are given in the first place</i>.</p>

<p><b>Supervised learning</b> is the case where you hold matched pairs — an input $x$ together with the correct answer $y$ that belongs with it. A thousand emails, each already marked spam or not. A hundred thousand houses, each with the price it sold for. The task is to learn the mapping from $x$ to $y$ well enough to apply it to inputs nobody has seen yet. This is the setting for most of this course, and for very nearly all of the machine learning that currently earns money.</p>

<p><b>Unsupervised learning</b> is the case where you have inputs but no answers attached to them. You hold a million customer records and nobody has labelled anything. The task therefore cannot be "predict $y$", because there is no $y$. Instead the task is to find structure: which records naturally group together (clustering, §2.10), which directions in the data carry most of the variation (PCA, §2.11), which records look unlike all the rest (anomaly detection).</p>

<p><b>Reinforcement learning</b> is the case where you have neither labels nor even a fixed dataset. What you have instead is an environment you can act inside, which occasionally reports how well you are doing by way of a reward. A program learning to play a game is never told the correct move for a given position; it is told the final score at the end. The task is to learn a <b>policy</b>, meaning a rule for choosing actions, that collects the most reward over time (§6.1).</p>

${H.intuition(`<p>The most consequential thing to understand about this taxonomy is that the largest models in the world are trained using the <i>first</i> category, wearing a disguise.</p>
<p>Training a language model looks unsupervised at first glance. You point it at an enormous quantity of text and nobody has labelled any of it. But look at what the objective actually is: predict the next word, given the words that came before. The next word is already sitting right there in the text. The label comes free. Every sentence ever written is therefore a fully labelled training example at zero annotation cost, which is exactly why this objective scales to trillions of words when paying humans to label things would have run out of money in the millions.</p>
<p>This arrangement has a name: <b>self-supervised</b> learning, meaning supervised learning whose labels are extracted from the structure of the data itself rather than supplied by a person.</p>`)}

<h2><span class="sn">0.1.3</span> The goal is not to fit the data you have</h2>

<p>Here is the point at which most people's intuition initially goes astray, so it is worth being emphatic about it.</p>

<p>We have just said that training means finding parameters that make the average loss small. It would be entirely natural to conclude that smaller is always better, and that a model driving its loss to zero has therefore succeeded completely. That conclusion is wrong, and understanding precisely why is a large part of what the first half of this course exists to explain.</p>

<p>The reason is that you do not actually care about the examples you already hold. You know the answers to those; they are written down. What you want is a model that performs on examples you have <i>not</i> seen — tomorrow's emails, next quarter's applicants, the sentence a user is about to type. The loss you are able to measure is the average over your training data, which is called the <b>empirical risk</b>. The quantity you genuinely want to be small is the average over everything the model will ever encounter, which is called the <b>expected risk</b>. These are two different numbers, and only the first one is available to you.</p>

<p>Worse, a model can drive the first to zero in a way that does nothing whatsoever for the second: by memorising. A rule which says "this exact email is spam, and this exact one is not, and here are the remaining 998 individually" achieves a perfect training loss and is completely worthless in practice, because it has learned the examples instead of learning the pattern.</p>

${H.key('The goal is never to fit the data you have. It is to fit the data you do not have.')}

<p>That this is achievable at all is neither obvious nor guaranteed. It works when your training examples genuinely represent what arrives later, and when the family of rules you searched was not so flexible that it could fit random noise as readily as real signal. Making that statement precise is the work of §1.4 on concentration. Controlling it in practice is the work of §2.2 on bias and variance, §2.3 on regularisation, and §2.14 on validation.</p>

<h2><span class="sn">0.1.4</span> Watching all three objects at once</h2>

<p>Everything above stays abstract until you see it move, so here is the entire loop running for real, using the simplest model there is.</p>

<p><b>What you are looking at.</b> Each dot is a single training example. Its horizontal position is the input $x$ and its vertical position is the correct answer $y$. The straight line is the model, $f_\\theta(x) = wx + b$. The dashed vertical segments are the <b>residuals</b>: the gap, for each point, between what the model predicted and what was actually true. The loss is the average of the squares of those dashed lengths, which is precisely why it is called squared error. The line drawn is the exact one that makes that average as small as it can possibly be.</p>

<p><b>What to do with it.</b> Drag any point and watch the line chase after it. You are changing the data, and the model is re-solving for its best parameters on every single frame. Notice while you do this that dragging a point near the middle of the cloud barely disturbs the line, whereas dragging one out at the far edge swings it noticeably. That is your first encounter with the fact that data points do not carry equal influence.</p>

<p><b>The thing genuinely worth noticing.</b> Switch on the <i>train / test split</i> toggle. Some points turn red. The line is now fitted using the blue points alone but scored against both groups, and two separate numbers appear in the readout: train MSE and test MSE. Now press <i>Add an outlier</i> a few times, or drag a point far away from its neighbours, and watch those two numbers come apart from each other. The gap between how a model scores on data it learned from and how it scores on data it did not is the single most important diagnostic in applied machine learning, and you have just manufactured it on purpose.</p>

${H.lab('fit', 'The whole loop, in one picture', 'Click empty space to add a point, drag any point to move it, and shift-click a point to delete it. The line is the exact least-squares solution, recomputed on every frame — there is no animation trickery here.')}

<h2><span class="sn">0.1.5</span> Why "learning" is a fair word for this</h2>

<p>It is quite reasonable to feel sceptical about the word "learning" here. Nothing in that demonstration understood anything at all. So it is worth being clear about what was achieved and what was not.</p>

<p>Nobody told the program what a line is <i>for</i>, which points were important, or what the relationship between $x$ and $y$ ought to look like. It received three things and nothing more: a family of candidate rules (all straight lines), a way to score any rule (squared error), and a procedure for finding a good one. What emerged was a rule that no human being wrote down. The slope and intercept were derived from the data, and they change whenever the data changes.</p>

<p>That is a modest but perfectly genuine form of learning. The important part, though, is that <b>the loop itself does not change as the models grow.</b> Swap the family of straight lines for a hundred-layer neural network. Swap squared error for cross-entropy. Swap the direct solve for millions of gradient steps spread across thousands of GPUs. You now have modern deep learning, and its structure is identical to the thing you were just dragging around: a parameterised family, a way of scoring it, and a procedure that improves the score.</p>

${H.more('a caveat worth carrying forward', `<p>Saying "the loop does not change" is true of the <i>structure</i> and misleading about the <i>difficulty</i>. In the line-fitting demonstration above, the best parameters can be computed in closed form. There is a formula, derived in §2.4, which simply hands you the answer with no searching whatsoever. That is a rare luxury. For almost every interesting model no such formula exists, the loss surface has many valleys rather than a single one, and the procedure for finding a good setting becomes a genuinely delicate piece of engineering. Parts 3 and 4 are largely concerned with that difficulty. The claim that the three objects are always the same still holds — it is simply that the third one becomes hard.</p>`)}

${H.probe([
      ['What is machine learning, in one sentence?', 'Searching a parameterised family of functions for the member that minimises expected loss on data drawn from the distribution you will be scored against.'],
      ['Why can a model be excellent on training data and useless in production?', 'Because it minimised empirical risk, the average loss over the examples it was shown, whereas what matters is expected risk, the average over everything it will actually meet. A sufficiently flexible family can drive the first to zero by memorising noise, which does nothing at all for the second.'],
      ['Someone hands you a model with a perfect training score. What do you ask next?', 'What the held-out score is, and how the split was constructed. A perfect training score carries essentially no information on its own about whether the model works.']
    ], 'Saying "the model learns patterns from the data". Every interviewer hears "I have not thought about what is being optimised". Name the family, the objective and the procedure instead.')}

<h2><span class="sn">0.1.6</span> The map of the rest</h2>
${H.table(['If you want to…', 'Go to'], [
      ['Understand why any of this generalises', '<a href="#/concentration">§1.4 concentration</a>'],
      ['Know where losses come from', '<a href="#/mle-map">§1.5 MLE and MAP</a>'],
      ['Fit and defend a tabular model', '<a href="#/boosting">§2.8 gradient boosting</a> and <a href="#/metrics">§2.13 metrics</a>'],
      ['Understand a transformer', '<a href="#/attention">§4.3 self-attention</a>'],
      ['Build something with an LLM', '<a href="#/rag">§5.1 RAG</a> and <a href="#/decision-ladder">§5.13 the ladder</a>']
    ])}`,
    labs: {
      fit: function (host) {
        const R = Num.rng(4);
        let pts = Array.from({ length: 14 }, (_, i) => {
          const x = -2.6 + 5.2 * (i + R() * .5) / 14;
          return { x: x, y: .85 * x + .4 + R.normal(0, .55), test: i % 4 === 3 };
        });
        const st = Viz.controls(host, [
          { k: 'split', label: 'train / test split', type: 'toggle', value: false },
          { k: 'noise', label: 'add noisy point', type: 'buttons', value: '', options: [{ v: '', t: '—' }] }
        ], () => S.redraw());
        host.querySelector('.controls').lastChild.remove();

        const out = Viz.readout(host, [
          { k: 'w', label: 'slope w' }, { k: 'b', label: 'intercept b' },
          { k: 'tr', label: 'train MSE', cls: 'key' }, { k: 'te', label: 'test MSE' }
        ]);

        let drag = -1;
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-3.4, 3.4] }).frame({ xlabel: 'x — the feature', ylabel: 'y — the label' });
            const tr = pts.filter(p => !(st.split && p.test));
            const te = pts.filter(p => st.split && p.test);
            const X = Num.polyDesign(tr.map(p => p.x), 1);
            const beta = tr.length > 1 ? Num.ridgeFit(X, tr.map(p => p.y), 1e-9) : [0, 0];
            P.clip(() => P.fn(x => beta[0] + beta[1] * x, { color: T.text, width: 2.4 }));
            // residuals
            tr.forEach(p => {
              const yh = beta[0] + beta[1] * p.x;
              P.line([[p.x, p.y], [p.x, yh]], { color: T.faint, width: 1, dash: [3, 3] });
            });
            P.dots(tr.map(p => [p.x, p.y]), { r: 5, color: T.blue, stroke: true });
            P.dots(te.map(p => [p.x, p.y]), { r: 5, color: T.red, stroke: true });
            const mse = arr => arr.length ? Num.mean(arr.map(p => (p.y - (beta[0] + beta[1] * p.x)) ** 2)) : 0;
            out({
              w: beta[1].toFixed(3), b: beta[0].toFixed(3),
              tr: mse(tr).toFixed(3), te: te.length ? mse(te).toFixed(3) : '—'
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          const dx = P.ix(e.x), dy = P.iy(e.y);
          if (e.type === 'down') {
            let best = -1, bd = 1e9;
            pts.forEach((p, i) => { const d = (p.x - dx) ** 2 + (p.y - dy) ** 2; if (d < bd) { bd = d; best = i; } });
            if (bd < .06) {
              if (e.shiftKey) { pts.splice(best, 1); drag = -1; }
              else drag = best;
            } else { pts.push({ x: dx, y: dy, test: false }); drag = pts.length - 1; }
            S.redraw();
          } else if (e.type === 'move' && e.down && drag >= 0) {
            pts[drag].x = dx; pts[drag].y = dy; S.redraw();
          } else if (e.type === 'up') drag = -1;
        });
        Viz.buttons(host, [
          { label: 'Reseed data', on: () => { const R2 = Num.rng(Math.floor(Math.random() * 1e6)); pts = Array.from({ length: 14 }, (_, i) => { const x = -2.6 + 5.2 * (i + R2() * .5) / 14; return { x: x, y: .85 * x + .4 + R2.normal(0, .55), test: i % 4 === 3 }; }); S.redraw(); } },
          { label: 'Add an outlier', on: () => { pts.push({ x: 2.6, y: -2.6, test: false }); S.redraw(); } }
        ]);
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'training points (fit uses these)' }, { c: Viz.theme().red, t: 'held-out points (judged on these)' }, { c: Viz.theme().text, t: 'least-squares fit' }]);
      }
    },
    quiz: [
      {
        q: 'A model scores 0.99 on the data it was fitted to and 0.61 on data it has never seen. Which of the three objects is most likely the problem?',
        options: ['The optimiser failed to converge', 'The model family is too flexible for the amount of data available', 'The loss was computed incorrectly', 'The data was not shuffled'],
        answer: 1,
        why: 'The pattern to recognise here is the <i>gap</i> between the two numbers, rather than either number taken on its own. Start by noticing what the 0.99 tells you: the optimiser worked beautifully, because it found parameters that fit the training examples almost exactly. So the very number that looks alarming actually rules out option A. What the gap tells you instead is that the family of rules you searched was flexible enough to fit the random noise in your particular sample, and given the chance, it did exactly that. Memorising the training set is always available to a sufficiently flexible model, and it always produces this signature. The fix is to reduce that flexibility, either by choosing a simpler family or by adding regularisation (§2.3), or to supply more data so that the noise averages out and only real structure survives. This gap has a name — variance — and §2.2 is devoted entirely to it.'
      },
      {
        q: 'Next-token prediction on a large corpus of text is best described as…',
        options: ['unsupervised learning, because nobody labelled the corpus', 'supervised learning where the label comes free with the data', 'reinforcement learning, because the model is rewarded for producing good text', 'semi-supervised learning'],
        answer: 1,
        why: 'It has precisely the structure of supervised learning. There is an input, namely the text so far, and there is a correct answer, namely the word that actually came next, and the loss compares the prediction against that answer. What makes it <i>feel</i> unsupervised is that no human sat down and wrote the labels, but the labels were already present in the text all along; they simply had to be revealed by hiding the next word and asking the model to guess it. That is the entire trick, and it explains why the approach scales so far: annotation cost, which is normally the binding constraint on supervised learning, drops to zero. The accepted name for this arrangement is self-supervised learning. Option C is a genuinely understandable confusion, because reinforcement learning does appear in the training of modern chat models — but it arrives later, as a separate fine-tuning stage (§4.10), and it is not the pretraining objective.'
      },
      {
        q: 'You replace squared error with a loss that returns 1 for any wrong prediction and 0 for a correct one. Gradient descent now makes no progress at all. Why?',
        options: ['The loss is not a valid measure of error', 'The loss is flat almost everywhere, so its gradient carries no information about which way to move', 'The learning rate is too small', 'The model family is too simple'],
        answer: 1,
        why: 'This question is really about what the optimiser needs <i>from</i> the loss, which is easy to overlook when first meeting the three objects. Gradient descent works by asking a very specific question: if I nudge this parameter a little, does the loss improve? A 0/1 loss answers "nothing changed" for almost every small nudge, because the prediction is still wrong and so the loss is still exactly 1 — right up until the prediction abruptly flips and the loss jumps to 0. A function that is flat almost everywhere and jumps in between has a gradient of zero almost everywhere, and a gradient of zero means there is no direction to step in. The optimiser is not stuck because it is badly configured; it is stuck because it has been handed no information. This is exactly why the losses you meet in practice are smooth stand-ins for the thing you actually care about, with cross-entropy standing in for accuracy. The gap between the loss you optimise and the metric you report is important enough that §2.1 is built around it.'
      }
    ],
    cards: [
      { q: 'What machine learning is for', a: 'Problems where you can recognise the correct answer but cannot write down the rule that produces it.' },
      { q: 'The three objects of every ML system', a: 'A parameterised model $f_\\theta$, a loss $\\ell$ that scores its predictions, and an optimiser that adjusts $\\theta$ to reduce the loss.' },
      { q: 'The goal of learning, stated precisely', a: 'Minimise <i>expected</i> loss on unseen data drawn from the same distribution — not empirical loss on the training set.' },
      { q: 'Empirical risk vs expected risk', a: 'Empirical risk is the average loss on the data you hold; expected risk is the average over everything the model will meet. Training minimises the first, but you care about the second.' },
      { q: 'Why is next-token prediction called self-supervised?', a: 'It is supervised learning whose labels are extracted from the structure of the data itself — the next word is already in the text — so annotation cost is zero.' },
      { q: 'What the learning rate $\\eta$ controls', a: 'The size of each optimiser step. Too small and training crawls; too large and the parameters overshoot the minimum and may diverge.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.2 */
  ML.section({
    id: 'linear-algebra-basics', track: 'start', num: '0.2',
    title: 'Vectors and matrices, from zero',
    lede: 'A vector is a list of numbers, and it is also an arrow. A matrix is a grid of numbers, and it is also a machine that moves arrows about. This section builds both from ordinary arithmetic and assumes nothing else. It ends with one small operation, the dot product, which turns out to be the answer to "how alike are these two things?", the inside of every neural network layer, and the entire mechanism of attention.',
    html: `
<p>Open a spreadsheet of loan applications. Each row is one applicant, and the columns hold what you know about them: credit utilisation 0.94, tenure with the bank 9 years, enquiries in the past six months 5. Now ask a question that sounds trivial. <i>Which two applicants are most alike?</i></p>

<p>Try to answer it with the arithmetic you already have and you stall almost immediately. You can compare utilisation with utilisation, and tenure with tenure, but that leaves you holding three separate comparisons and no way to fuse them into one verdict. The three columns are not even measured in compatible things: a ratio, a number of years, a count of events. Nothing tells you whether "0.02 more utilisation" is a bigger or a smaller difference than "one more year of tenure".</p>

<p>What is missing is a language in which <b>a whole row is one object</b> with its own arithmetic — objects you can add, stretch, measure the length of, and compare in a single stroke. That language is linear algebra. This section builds it from nothing, and by the end you will have met every piece of it that the rest of this course actually uses.</p>

<h2><span class="sn">0.2.1</span> A vector is a row of your data, and also an arrow</h2>

<p>A <b>vector</b> is an ordered list of numbers. That really is the whole definition; the applicant above <i>is</i> the vector</p>

$$x = (0.94,\\; 9,\\; 5).$$

<p>We name the whole list with a single letter, $x$, and we get at the individual numbers with a <b>subscript</b>: $x_1 = 0.94$, $x_2 = 9$, $x_3 = 5$. Read $x_2$ aloud as "x sub two", meaning "the second entry of x". Those entries are called the <b>components</b> or <b>coordinates</b> of the vector. How many of them there are is the <b>dimension</b>, written $d$; here $d = 3$. In machine learning $d$ is nearly always the number of features you measured, so a model working on 64 features is working in 64 dimensions, and a sentence embedding with 1536 numbers in it is a vector with $d = 1536$.</p>

<p>Two things are worth noticing about that definition straight away. The order matters, because the second slot always means tenure and never means anything else. And a vector is a single object: the notation $x$ refers to the applicant, not to any one fact about them.</p>

<p>So far this is just a list, and a list hardly needs a new name. The reason vectors earn one is a second picture that sits on top of the first.</p>

<p>Take a vector with only two components, say $a = (3, 1)$, and draw it on ordinary graph paper. Start at the origin, go 3 units right and 1 unit up, and draw an arrow from the origin to where you land. That arrow <i>is</i> the vector, drawn. Everything about the list is now visible as geometry: how far the arrow reaches is a fact about the size of the numbers, and which way it points is a fact about their ratio.</p>

${H.analogy(`<p>A vector is a set of walking directions given all at once. "Three streets east, one street north" is not two instructions performed in sequence; it is a single displacement, and it takes you to the same corner however you walk it. That is exactly the sense in which $(3, 1)$ is one object rather than two numbers standing next to each other.</p>
<p>The directions picture also tells you what adding vectors ought to mean before anyone defines it. Follow one set of directions, then follow another from wherever you ended up. Where do you finish?</p>`)}

<h3>The two operations that generate everything</h3>

<p>There are exactly two things you can do to vectors, and every other idea in this section is built out of them.</p>

<p><b>Addition</b> works component by component:</p>

$$a + b = (a_1 + b_1,\\; a_2 + b_2,\\; \\ldots,\\; a_d + b_d)$$

<p>Read that as: to add two vectors, add their first entries to get the new first entry, add their second entries to get the new second entry, and so on down the list. With $a = (3, 1)$ and $b = (1, 2)$ you get $a + b = (4, 3)$. In the arrow picture this is the walking-directions answer: place the tail of $b$ at the tip of $a$, and the sum is the arrow from the origin to where $b$ now ends. Vectors of different dimensions cannot be added at all, because some entries would have no partner.</p>

<p><b>Scaling</b> multiplies every component by the same single number:</p>

$$\\alpha a = (\\alpha a_1,\\; \\alpha a_2,\\; \\ldots,\\; \\alpha a_d)$$

<p>Here $\\alpha$ is the Greek letter alpha, and a plain number used this way is called a <b>scalar</b>, precisely because its job is to scale. So $2a = (6, 2)$, which is the same arrow pointing the same way but twice as long; $0.5a = (1.5, 0.5)$ is half as long; and $-a = (-3, -1)$ is the same length pointing backwards. <mark>Scaling changes length and possibly flips direction, but it never rotates.</mark></p>

${H.worked('adding and scaling, with the numbers written out', `<p>Let $a = (3, 1)$ and $b = (1, 2)$.</p>
<p>$a + b = (3+1,\\; 1+2) = (4, 3)$.</p>
<p>$2a = (6, 2)$, and $-b = (-1, -2)$, so $2a - b = (6-1,\\; 2-2) = (5, 0)$ — an arrow lying flat along the horizontal axis, because the vertical parts happened to cancel exactly.</p>
<p>$0a = (0, 0)$, the <b>zero vector</b>: the arrow of no length at all, which is the origin itself.</p>`)}

<p>Combine the two operations and you get the central construction of the whole subject. An expression like $\\alpha a + \\beta b$ — scale $a$ by some amount, scale $b$ by some amount, add the results — is called a <b>linear combination</b> of $a$ and $b$. If you let $\\alpha$ and $\\beta$ range over every possible number, the set of points you can reach is called the <b>span</b> of $a$ and $b$.</p>

<p>That set is always flat and always passes through the origin, which is worth pausing on because it explains the word "linear" everywhere else in this course. Span one non-zero vector in two dimensions and you can reach every point on a line through the origin. Span two vectors pointing in genuinely different directions and you can reach every point in the plane. But span two vectors that happen to point along the same line — say $(1, 2)$ and $(2, 4)$, where the second is just twice the first — and you are stuck on that one line no matter what coefficients you choose. The second vector added nothing. Keep that failure in mind; when it happens to the columns of a matrix it has a name, and it is the reason a regression can refuse to fit.</p>

<h2><span class="sn">0.2.2</span> Length, and the first real question you can now ask</h2>

<p>Once a row of data is an arrow, "how long is it?" becomes a sensible question with an answer you already know. In two dimensions an arrow to $(3, 4)$ is the hypotenuse of a right-angled triangle with sides 3 and 4, so by Pythagoras its length is $\\sqrt{3^2 + 4^2} = \\sqrt{25} = 5$. The general rule is the same rule, extended to as many terms as you have components:</p>

$$\\|x\\| = \\sqrt{x_1^2 + x_2^2 + \\cdots + x_d^2}$$

<p>The double bars are read "the norm of x", and <b>norm</b> is simply the technical word for length. Say the formula in words: square every component, add the squares together, take the square root of the total. Every component contributes something non-negative, so a norm is never negative, and it is zero only for the zero vector. Try it in three dimensions with $x = (2, 3, 6)$: the squares are 4, 9 and 36, the total is 49, and the norm is exactly 7. Nothing about the calculation cared that you cannot easily draw it.</p>

<p>With length in hand, distance follows for free. The vector $a - b$ is the arrow that takes you from the tip of $b$ to the tip of $a$, so the <b>distance</b> between two data points is</p>

$$\\mathrm{dist}(a, b) = \\|a - b\\| = \\sqrt{(a_1-b_1)^2 + \\cdots + (a_d-b_d)^2}$$

<p>which is the ordinary straight-line distance you would measure with a ruler, generalised to $d$ dimensions. This single formula is the whole of $k$-nearest-neighbours (§2.5), it is what $k$-means uses to decide which cluster a point belongs to (§2.9), and it is the quantity being minimised, in disguise, every time you fit a model by least squares (§2.4).</p>

${H.pitfall(`<p>Distance takes your units at face value, and this quietly ruins more models than it has any right to. Suppose one applicant differs from another by 0.3 in utilisation and by 4 years in tenure. The squared distance is $0.3^2 + 4^2 = 0.09 + 16$, so tenure supplies 99.4 per cent of the answer and utilisation is effectively ignored — not because tenure matters more, but because years happen to be numbered in larger units than ratios.</p>
<p>Re-express tenure in months and the same two applicants are now 48 apart on that axis, and the imbalance gets 144 times worse. The fix is <b>standardisation</b>: subtract each column's mean and divide by its standard deviation, so every feature contributes on comparable terms. This is why §0.5 insists on fitting the scaler inside the cross-validation fold, and why §2.5 warns that $k$-NN without scaling is barely a model at all.</p>`)}

<h2><span class="sn">0.2.3</span> The dot product, which is nearly the whole of similarity</h2>

<p>Distance answers "how far apart are these two rows?". It is not always the question you want. Consider two customers with identical shopping habits, one of whom simply buys ten times as much of everything. Their vectors point in exactly the same direction, but one is far longer than the other, so the distance between them is large while their <i>taste</i> is identical. What you want is a measure of <b>alignment</b>: how much do these two arrows point the same way, setting aside how long they are?</p>

<p>Here is the operation that answers it. Multiply the two vectors component by component, then add up all the products:</p>

$$a \\cdot b = a_1 b_1 + a_2 b_2 + \\cdots + a_d b_d = \\sum_{i=1}^{d} a_i b_i$$

<p>This is the <b>dot product</b>, read aloud as "a dot b". The last form uses $\\sum$, the Greek capital sigma, which means "add up what follows"; the $i = 1$ underneath and the $d$ on top say that $i$ should run through every whole number from 1 to $d$. So the notation says exactly what the longer form says: form the product $a_i b_i$ for each slot in turn, and total them. Note the shape of the answer — two vectors go in, and a <i>single number</i> comes out.</p>

${H.worked('three dot products that show the whole range of behaviour', `<p>With $a = (3, 1)$ and $b = (1, 2)$: &nbsp; $a \\cdot b = 3 \\times 1 + 1 \\times 2 = 5$. Positive, so the arrows broadly agree.</p>
<p>With $a = (1, 2)$ and $c = (2, -1)$: &nbsp; $a \\cdot c = 1 \\times 2 + 2 \\times (-1) = 0$. Draw those two and you will find they meet at a right angle. A dot product of zero means <b>orthogonal</b>, which is the technical word for perpendicular.</p>
<p>With $a = (3, 1)$ and $-a = (-3, -1)$: &nbsp; $a \\cdot (-a) = -9 - 1 = -10$. Negative, and as negative as it can get for arrows of these lengths, because they point in exactly opposite directions.</p>`)}

<p>So the sign of a dot product already reports something geometric: positive for broadly-agreeing arrows, zero for perpendicular ones, negative for opposing ones. That is not a coincidence, and the exact statement is one of the most useful identities in the subject:</p>

$$a \\cdot b = \\|a\\|\\,\\|b\\|\\cos\\theta$$

<p>Here $\\theta$ is the Greek letter theta and denotes the angle between the two arrows, measured at the origin. The cosine of an angle runs from 1 when the angle is zero, through 0 at ninety degrees, to $-1$ at a hundred and eighty. So the identity says, in words: <b>the dot product is length, times length, times how well aligned the two directions are.</b> Both of the two very different-looking descriptions — "sum of pairwise products" and "lengths multiplied by alignment" — describe the same number, and the reason is a piece of school geometry.</p>

${H.deriv('why the sum of products equals $\\|a\\|\\,\\|b\\|\\cos\\theta$', [
      ['$\\|a-b\\|^2 = \\|a\\|^2 + \\|b\\|^2 - 2\\|a\\|\\,\\|b\\|\\cos\\theta$', 'The law of cosines, applied to the triangle whose two sides are the arrows $a$ and $b$ and whose third side joins their tips. $\\theta$ is the angle between them at the origin. This is pure geometry and contains no dot products at all.'],
      ['$\\|a-b\\|^2 = (a-b)\\cdot(a-b)$', 'Now compute the same length a second way, algebraically. Squared length is a vector dotted with itself, straight from the definitions: $v \\cdot v = \\sum_i v_i^2 = \\|v\\|^2$.'],
      ['$= a\\cdot a - 2\\,(a\\cdot b) + b\\cdot b$', 'Expand the bracket exactly as you would expand $(p-q)^2$ in ordinary algebra. This step is legitimate because the dot product distributes over addition and is symmetric — both facts are immediate from the sum-of-products definition, since they are just sums of ordinary numbers being rearranged.'],
      ['$= \\|a\\|^2 - 2\\,(a\\cdot b) + \\|b\\|^2$', 'Replace $a \\cdot a$ by $\\|a\\|^2$ and $b \\cdot b$ by $\\|b\\|^2$, using the definition invoked in line 2.'],
      ['$\\|a\\|^2 + \\|b\\|^2 - 2\\|a\\|\\,\\|b\\|\\cos\\theta = \\|a\\|^2 - 2\\,(a\\cdot b) + \\|b\\|^2$', 'Lines 1 and 4 are two expressions for the same single number, $\\|a-b\\|^2$, so they may be set equal to each other.'],
      ['$a\\cdot b = \\|a\\|\\,\\|b\\|\\cos\\theta$', 'Cancel $\\|a\\|^2$ and $\\|b\\|^2$ from both sides, then divide through by $-2$. The algebraic definition and the geometric one were never two different things.']
    ], 'Notice that nothing in this argument mentions how many dimensions you are in. The law of cosines holds inside the flat plane containing the two arrows, and any two arrows lie in some plane, whether they live in 2 dimensions or 4096. That is the standard escape route in this subject: reduce a high-dimensional question to a two-dimensional one and then draw it.')}

<h3>Cosine similarity: alignment with the lengths divided out</h3>

<p>The identity hands you the similarity measure asked for at the start of this subsection. If you want alignment <i>only</i>, without the arrow lengths contaminating it, simply divide them out:</p>

$$\\cos\\theta = \\frac{a \\cdot b}{\\|a\\|\\,\\|b\\|}$$

<p>This is <b>cosine similarity</b>. Because it is a cosine it always lands between $-1$ and $1$: the value 1 means the arrows point in precisely the same direction, 0 means perpendicular, and $-1$ means precisely opposite. Our two customers with identical taste and very different budgets now score 1, which is the answer we wanted, whereas their distance was large. Distance asks "how far apart?" and cosine asks "how similarly oriented?", and the two answers can disagree completely.</p>

<p>Working through the earlier example: $a = (3,1)$ and $b = (1,2)$ give $a \\cdot b = 5$, $\\|a\\| = \\sqrt{10} \\approx 3.162$ and $\\|b\\| = \\sqrt{5} \\approx 2.236$, so the cosine similarity is $5 / (3.162 \\times 2.236) \\approx 0.707$. That is the cosine of forty-five degrees, which is exactly the angle you will measure if you draw the two arrows.</p>

<h3>Projection: the shadow one arrow casts on another</h3>

<p>There is one more reading of the dot product, and it is the one the lab below draws. Shine a light straight down onto the direction of $b$ and ask where the tip of $a$ throws its shadow. That shadow is the <b>projection</b> of $a$ onto $b$, and it is the point on $b$'s line closest to $a$:</p>

$$\\mathrm{proj}_b(a) = \\frac{a \\cdot b}{b \\cdot b}\\, b$$

<p>Read the formula from the outside in. The final $b$ says the answer points along $b$, as a shadow on a line must. The fraction in front is a scalar saying how far along. Its length works out to $a \\cdot b / \\|b\\|$, so it is the dot product with just <i>one</i> of the two lengths removed: how much of $a$ lies in the direction of $b$, measured in the original units of $a$. If $a$ is perpendicular to $b$, the numerator is zero and the shadow collapses to the origin, which is precisely what a shadow does when the object is edge-on to the light.</p>

${H.intuition(`<p>Almost every "score" in machine learning is a dot product, and once you see this the notation across several later sections collapses into one idea.</p>
<p>A linear model holds a weight vector $w$, one weight per feature, and scores an example by computing $w \\cdot x$ (§2.4). Read geometrically, the weight vector defines a direction in feature space, and the score is how far along that direction your example lies. Training the model means rotating and stretching that arrow until the scores line up with the labels.</p>
<p>A retrieval system stores each document as a vector and answers a query by ranking documents on cosine similarity with the query vector (§5.10). Attention, the operation inside every transformer, computes a dot product between a query vector and a key vector for every pair of positions and uses the results as weights (§4.3). The famous $\\sqrt{d_k}$ in the attention formula exists purely because dot products of long random vectors grow with $d$, and that growth has to be undone.</p>
<p>Three chapters, three vocabularies, one operation. If you can say <i>a score is a dot product, and a dot product is alignment scaled by magnitude</i>, none of them will surprise you.</p>`)}

<p><b>What you are looking at.</b> The blue arrow is $a$ and the red arrow is $b$, both drawn from the origin on ordinary axes. The thick grey arrow lying along $b$ is the projection of $a$ onto $b$ — the shadow described above — and the faint dashed line joining it to the tip of $a$ is the light ray casting it, always meeting $b$ at a right angle. The thin green arrow is the sum $a + b$, with dashed lines completing the parallelogram so you can see the tip-to-tail construction. The readout gives $a \\cdot b$, the two lengths, the angle between the arrows in degrees, and the cosine similarity.</p>

<p><b>What to do with it.</b> Drag the head of either arrow. Start by holding $b$ still and swinging $a$ around it, and watch the dot product fall as the angle opens up. Bring the angle to exactly ninety degrees and confirm that the dot product reads zero and the grey shadow shrinks to nothing. Push past ninety and both the dot product and the cosine go negative, while the shadow reappears on the opposite side of the origin.</p>

<p><b>The thing genuinely worth noticing.</b> Now leave the angle alone and drag $a$ straight outwards along its own direction, making it longer without turning it. The dot product climbs steadily while the cosine similarity does not move at all. That is the entire difference between the two measures displayed in one gesture: the dot product mixes alignment with magnitude, and cosine similarity is the same quantity with magnitude divided out. Choosing between them is a real modelling decision, and it is exactly the decision a vector database makes when it normalises its embeddings before indexing them.</p>

${H.lab('vec', 'Vector playground — drag the two arrows', 'Drag either arrow head. The projection of <b>a</b> onto <b>b</b> is drawn in grey: it is the shadow <b>a</b> casts on <b>b</b>, and its length is exactly $a\\cdot b / \\|b\\|$. The green parallelogram is the tip-to-tail construction of $a+b$.')}

<h2><span class="sn">0.2.4</span> A matrix is a function on space</h2>

<p>One score is rarely enough. A layer of a neural network does not produce a single number from your features; it produces a whole new list of numbers, each one a differently-weighted view of the same input. If you want four such scores from three features, you need four weight vectors, one per score. Stack those four rows on top of each other into a rectangular block of numbers, and you have invented the <b>matrix</b>.</p>

<p>A matrix is a grid of numbers with a <b>shape</b>, written $[m \\times n]$ and read "m by n", meaning $m$ rows and $n$ columns. The entry in row $i$ and column $j$ is written $A_{ij}$, read "A sub i j" — rows first, always. Multiplying a matrix by a vector is defined so that it does precisely the stacking job just described:</p>

$$(Ax)_i = \\sum_{j=1}^{n} A_{ij}\\,x_j$$

<p>In words: the $i$-th entry of the answer is the dot product of the $i$-th <i>row</i> of $A$ with the vector $x$. A matrix times a vector is therefore a stack of dot products, computed all at once. The input $x$ must have exactly as many entries as $A$ has columns, or the dot products would run out of partners, and the output has exactly as many entries as $A$ has rows.</p>

${H.worked('a two-by-two matrix applied to a vector, both ways', `<p>Let $A = \\begin{bmatrix} 2 & 1 \\\\ 0 & 3 \\end{bmatrix}$ and $x = (4, 1)$.</p>
<p><b>Reading by rows.</b> The first row is $(2, 1)$, and $(2,1) \\cdot (4,1) = 8 + 1 = 9$. The second row is $(0, 3)$, and $(0,3) \\cdot (4,1) = 0 + 3 = 3$. So $Ax = (9, 3)$.</p>
<p><b>Reading by columns.</b> The columns of $A$ are $(2, 0)$ and $(1, 3)$. Now compute $4 \\times (2,0) + 1 \\times (1,3) = (8, 0) + (1, 3) = (9, 3)$ — the same answer, obtained by treating the entries of $x$ as instructions for how much of each column to take.</p>`)}

<p>Both readings are correct and both are useful, but the second is the one that turns a matrix into geometry. It says that $Ax$ is a linear combination of the columns of $A$, with the components of $x$ supplying the coefficients. Feed in $x = (1, 0)$ and you take one lot of the first column and none of the second, so you get exactly the first column back. Feed in $(0, 1)$ and you get the second column. In other words:</p>

${H.key('The columns of a matrix are the images of the basis vectors. A matrix is completely specified by where it sends $(1,0)$ and $(0,1)$.')}

<p>That is a stronger statement than it looks. A matrix acting on space obeys $A(\\alpha x + \\beta y) = \\alpha Ax + \\beta Ay$, which says the map respects addition and scaling — the property that gives <i>linear</i> algebra its name. Because every vector is a linear combination of the basis vectors, and because the map respects linear combinations, knowing what happens to just two arrows determines what happens to all of them. A grid of four numbers pins down the fate of every point in the plane.</p>

${H.analogy(`<p>Picture the plane as an infinite sheet of stretchy rubber with a grid drawn on it, pinned down with a single tack at the origin. A matrix is a way of pulling the sheet: you may stretch it, squash it, shear it sideways, spin it, or flip it over, but you may not tear it, curve the grid lines, or move the tack. Straight lines stay straight, parallel lines stay parallel, and the origin stays put.</p>
<p>That is the complete list of what "linear" permits, and it is also the complete list of what a single neural network layer can do before its activation function is applied. It is precisely why activation functions are not optional: without something non-linear between the layers, a stack of matrices only ever amounts to one matrix, and the whole tower can do no more than a single pull of the rubber sheet (§3.3).</p>`)}

<h3>Determinant, rank, and what it means for a matrix to destroy information</h3>

<p>Take the unit square whose corners are the origin, $(1,0)$, $(0,1)$ and $(1,1)$, and apply the matrix to it. It comes out as a parallelogram. The factor by which its area changed is the <b>determinant</b>, written $\\det A$, and for a two-by-two matrix it has a short formula:</p>

$$\\det \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix} = ad - bc$$

<p>For $\\begin{bmatrix} 2 & 0 \\\\ 0 & 3 \\end{bmatrix}$ this gives $6$: the map stretches by 2 horizontally and 3 vertically, so areas grow six-fold, exactly as the formula says. A negative determinant means the sheet was flipped over as well as stretched. And a determinant of zero is the interesting case. Try $\\begin{bmatrix} 1 & 2 \\\\ 2 & 4 \\end{bmatrix}$: the formula gives $1 \\times 4 - 2 \\times 2 = 0$, because the second column is exactly twice the first. Both columns lie along the same line, so every possible output is a combination of two arrows pointing the same way, and the whole plane is squashed flat onto that single line.</p>

<p>The number of genuinely independent directions surviving in the output is the <b>rank</b>. A two-by-two matrix with rank 2 spreads the plane over the plane; one with rank 1 collapses it onto a line; rank 0 sends everything to the origin. Once a dimension has been flattened away, no matrix can bring it back, because every point on that line came from a whole line's worth of inputs and there is no way to know which. That is what it means for a matrix to be <b>singular</b>, or non-invertible, and it is not an exotic edge case: it is what two perfectly correlated columns in your data produce, and it is exactly why $X^\\mathsf{T}X$ can fail to invert in linear regression and why adding $\\lambda I$ — ridge regression — repairs it (§2.3, §2.4).</p>

<p>Finally, a fact worth carrying even though its proof waits for §1.8. However complicated a matrix looks, everything it does can be decomposed into exactly three steps in sequence: <b>rotate, stretch along the resulting axes, then rotate again</b>. That statement is the singular value decomposition, the stretch factors are the singular values $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots$ (sigma), and their ratio $\\sigma_1/\\sigma_2$ is the <b>condition number</b>, which measures how badly the map distorts. A large condition number means the matrix stretches enormously in one direction and barely at all in another, which is the geometric shape of an optimisation problem that trains slowly (§1.9) and of a numerical calculation you should not trust (§1.15).</p>

<p><b>What you are looking at.</b> You are watching a single matrix act on the plane. The faint grey grid is the <i>input</i> space, undisturbed, drawn for reference. The blue grid is where that grid <i>lands</i> after the matrix is applied, so every blue line is the image of a grey line. The dashed pale outline is the test shape before the map — a unit circle, or a letter F if you switch shapes — and the solid red outline is where it ends up. The two thick arrows are the columns of the matrix themselves: blue for column 1, green for column 2, which by the boxed rule above are the images of $(1,0)$ and $(0,1)$. The readout reports the determinant, the two singular values, and the condition number.</p>

<p><b>What to do with it.</b> Drag either column arrow; you are editing the matrix by hand, one column at a time. Press <i>Rotation 30°</i> and note that the shape turns without changing size and the determinant stays at 1. Press <i>Shear</i> and watch the F lean over while the determinant remains 1, which tells you a shear moves every point yet preserves area exactly. Switch the shape control to the letter F rather than the circle, because a circle is too symmetric to reveal reflections: only an asymmetric shape shows you when the sheet has been flipped.</p>

<p><b>The thing genuinely worth noticing.</b> Drag the two column arrows until they point along the same line. The determinant slides to zero, the smaller singular value collapses towards zero, the condition number runs away towards infinity, and the red shape flattens onto a line — the circle becomes a segment. You have just built a singular matrix with your hands. This is precisely the situation two collinear features create in a regression, and seeing the geometry once is worth more than the algebraic statement about determinants. Notice too how the condition number becomes enormous <i>before</i> the collapse is complete: numerical trouble arrives well before the matrix is exactly singular, which is why "nearly collinear" is already a problem.</p>

${H.lab('mat', 'What a matrix does to space', 'Drag the two column arrows — you are editing the matrix directly. Faint grey is the original grid; blue is where that grid lands. The dashed outline is the shape before the map and the solid red one is its image. The amber dashed lines show the eigen-directions of the symmetric part of the matrix, which are the axes it stretches along.')}

<h3>The vocabulary you actually need</h3>

<p>Everything below appears somewhere later in the course. Each row gives the plain-words meaning first and the place it bites second, so the table can be used as a lookup while reading other sections.</p>

${H.table(['Term', 'What it means, in words', 'Where it bites'], [
      ['Transpose $A^\\mathsf{T}$', 'Flip the grid across its diagonal, so rows become columns. Read "A transpose"', '$X^\\mathsf{T}X$ in the normal equations (§2.4); $QK^\\mathsf{T}$ in attention (§4.3)'],
      ['Matrix product $AB$', 'Apply $B$ first, then $A$. Entry $(i,j)$ is row $i$ of $A$ dotted with column $j$ of $B$', 'Every layer of every network; composing two transformations'],
      ['Identity $I$', 'The matrix that leaves every vector alone — ones down the diagonal, zeros elsewhere', 'Ridge adds $\\lambda I$ to force a matrix to be invertible (§2.3)'],
      ['Inverse $A^{-1}$', 'The map that undoes $A$. Exists only when $\\det A \\ne 0$, i.e. when nothing was flattened', 'Closed-form least squares (§2.4); Newton steps (§1.12)'],
      ['Rank', 'How many genuinely independent directions survive in the output', 'LoRA (§4.13) is the claim that a weight update only needs rank 16'],
      ['Norm $\\|x\\|_2$', 'Length: $\\sqrt{\\sum_i x_i^2}$', 'L2 regularisation is a penalty on this length (§2.3)'],
      ['Orthogonal matrix', 'Columns are unit length and mutually perpendicular; the map is a pure rotation or reflection', 'The $U$ and $V$ of the SVD; why rotations do not amplify error (§1.15)'],
      ['Symmetric, PSD', '$A = A^\\mathsf{T}$, and $x^\\mathsf{T}Ax \\ge 0$ for every $x$ — the map never reverses a direction back on itself', 'Covariance and kernel matrices; guarantees a bowl-shaped loss (§1.8)']
    ])}

<h2><span class="sn">0.2.5</span> Shapes are the debugging tool</h2>

<p>Most of the errors you will meet in deep learning are not subtle mathematical errors. They are shape errors: a matrix of the wrong size meeting a vector of the wrong size, or two tensors combining along the wrong axis and silently producing something meaningless instead of raising an exception. Almost all of that confusion dissolves with one habit — <b>write the shape in a comment after every line</b>.</p>

<p>The rule you are enforcing is simple. If $X$ is $[n \\times d]$ and $W$ is $[d \\times k]$, then $XW$ is $[n \\times k]$: the inner dimensions must match, and when they do they cancel, leaving the outer two. Say it aloud as "n by d, times d by k, gives n by k". Since $X$ is normally a batch of $n$ examples each holding $d$ features, and $W$ is a layer mapping $d$ features to $k$ outputs, the shape arithmetic is also the sentence "a batch of n examples goes in, a batch of n outputs comes out".</p>

${H.code(`import numpy as np
X = np.random.randn(256, 64)      # [n=256, d=64]  a batch of examples
W = np.random.randn(64, 16)       # [d=64,  k=16]  a linear layer
b = np.zeros(16)                  # [k=16]         broadcast over rows
H = X @ W + b                     # [256, 16]      inner 64 cancels
print(H.shape, (X @ W).T.shape)   # (256, 16) (16, 256)`)}

<p>The addition of <code>b</code> on the fourth line deserves a word, because it looks like it should fail. <code>X @ W</code> has shape $[256 \\times 16]$ and <code>b</code> has shape $[16]$, which are not the same thing at all. NumPy applies <b>broadcasting</b>: when one operand is missing a dimension, it is stretched, conceptually and for free, to match. Here the length-16 bias is reused for all 256 rows, which is exactly what a bias term is meant to do. Broadcasting is a convenience most of the time and a trap occasionally, because a $[256 \\times 1]$ array and a $[1 \\times 256]$ array will happily broadcast against each other into a $[256 \\times 256]$ array that you never wanted.</p>

${H.practice(`<p>When a network misbehaves and you cannot see why, the first move is not to reason about the mathematics. It is to print every shape in the forward pass and read them in order. Roughly half of all "the model trains but the loss will not go down" reports turn out to be a silent broadcast that averaged something across the wrong axis.</p>
<p>Two specific habits are worth adopting early. Keep a fixed letter for each axis — $n$ for batch, $d$ for input features, $k$ for outputs, $L$ for sequence length, $h$ for heads — and use those letters in your comments, so a mismatch becomes visible as a mismatch of letters rather than of digits. And when reshaping, prefer being explicit about which axes you are combining, because a transpose that you needed but forgot produces a perfectly valid shape and completely wrong numbers. §4.5 walks the shapes of an entire transformer block, and that page is nothing but this habit applied a dozen times.</p>`)}

<h2><span class="sn">0.2.6</span> What this buys you later</h2>

<p>It is worth being concrete about the payoff, because the abstraction only feels justified once you see the same three objects reappear wearing different names.</p>

<p><b>A neural network layer</b> is $h = Wx + b$ (§3.1). Read with the tools you now have: $W$ is a stack of weight vectors, so $Wx$ is a stack of dot products, one per output unit, each measuring how strongly the input aligns with that unit's preferred direction. The bias $b$ shifts the result, which is the one thing a pure matrix cannot do because a matrix must fix the origin.</p>

<p><b>Attention</b> (§4.3) turns every position in a sentence into three vectors — a query, a key and a value. The relevance of position $j$ to position $i$ is the dot product of query $i$ with key $j$: alignment, exactly as in this section. Doing that for all pairs at once is a single matrix product $QK^\\mathsf{T}$, and the resulting grid of scores is turned into weights and used to mix the value vectors. If you understood the dot product above, you understand the core of the transformer, and §4.3 is mostly bookkeeping on top of it.</p>

<p><b>PCA</b> (§2.10) asks which direction in your data carries the most variation. Written in this language, it is a question about a matrix: build the covariance matrix, find the direction along which it stretches the most, and keep that. The stretch directions of a symmetric matrix are its eigenvectors, which is why PCA and the SVD are the same result told twice.</p>

${H.more('why high dimensions behave strangely, and why you should not fear them', `<p>Every operation defined in this section is defined identically in 4096 dimensions. Nothing in the formula for a norm, a dot product or a matrix product cares about $d$. What fails in high dimensions is your ability to <i>draw</i> the situation, which is a limitation of pencils rather than of mathematics.</p>
<p>Some genuine surprises do lurk there, and they are worth knowing about before they bite. Distances concentrate: pick many random points in a high-dimensional cube and the nearest pair and the farthest pair end up almost equally far apart, which quietly destroys the meaning of "nearest neighbour" (§2.5 measures this directly). Random vectors are very nearly orthogonal to one another, so a space of dimension 4096 can hold a huge number of nearly-non-interfering directions, which is the reason embeddings work at all (§3.9). And volume migrates to the surface, so almost all of a high-dimensional ball is a thin shell near the edge.</p>
<p>The productive attitude is neither mysticism nor complacency. Compute in $d$ dimensions using the formulas, which never change, and picture the geometry in two, which is usually right — except for the specific list of effects above, each of which has its own section.</p>`)}

${H.probe([
      ['What does the dot product mean geometrically?', 'Length times length times the cosine of the angle between the two vectors — alignment scaled by magnitude. It is zero exactly when they are orthogonal, positive when they broadly agree, negative when they oppose.'],
      ['When would you use cosine similarity rather than Euclidean distance?', 'When the magnitude of the vector is an artefact rather than a signal — document length, spending volume, embedding norm. Cosine divides both lengths out and compares direction alone.'],
      ['When is a square matrix not invertible, and why do you care?', 'When its determinant is zero, meaning its columns are linearly dependent — collinear features, in data terms. The map has flattened a dimension and no inverse can restore it. It is why $X^\\mathsf{T}X$ can fail to invert and why ridge\'s $\\lambda I$ fixes it (§2.4).'],
      ['What does a large condition number tell you?', 'That the matrix stretches far more along one direction than another. It predicts slow gradient descent (§1.9) and unreliable numerical solves (§1.15), and it shows up long before a matrix is exactly singular.']
    ], 'Treating "high-dimensional" as mystical. Every operation you know in 2-D is defined identically in 4096-D; only your ability to draw it fails. Say what the formula does, then name the specific high-dimensional effects that genuinely differ.')}`,
    labs: {
      vec: function (host) {
        let a = [2.2, 1.4], b = [3.0, -0.8], drag = null;
        const out = Viz.readout(host, [
          { k: 'dot', label: 'a · b', cls: 'key' }, { k: 'na', label: '‖a‖' }, { k: 'nb', label: '‖b‖' },
          { k: 'ang', label: 'angle' }, { k: 'cos', label: 'cosine sim' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3, 3] }).frame({ grid: true });
            // projection of a onto b
            const bb = Num.dot(b, b) || 1e-9, k = Num.dot(a, b) / bb;
            const proj = [k * b[0], k * b[1]];
            P.line([[proj[0], proj[1]], [a[0], a[1]]], { color: T.faint, width: 1.2, dash: [4, 4] });
            P.arrow(0, 0, proj[0], proj[1], { color: T.faint, width: 5 });
            P.arrow(0, 0, a[0], a[1], { color: T.blue, width: 2.6 });
            P.arrow(0, 0, b[0], b[1], { color: T.red, width: 2.6 });
            P.arrow(0, 0, a[0] + b[0], a[1] + b[1], { color: T.green, width: 1.6 });
            P.line([[a[0], a[1]], [a[0] + b[0], a[1] + b[1]]], { color: T.green, width: 1, dash: [3, 3] });
            P.line([[b[0], b[1]], [a[0] + b[0], a[1] + b[1]]], { color: T.green, width: 1, dash: [3, 3] });
            P.text(a[0], a[1], ' a', { color: T.blue, font: 'bold 13px ui-sans-serif' });
            P.text(b[0], b[1], ' b', { color: T.red, font: 'bold 13px ui-sans-serif' });
            P.text(a[0] + b[0], a[1] + b[1], ' a+b', { color: T.green, font: '12px ui-sans-serif' });
            const d = Num.dot(a, b), na = Num.norm(a), nb = Num.norm(b);
            out({
              dot: d.toFixed(2), na: na.toFixed(2), nb: nb.toFixed(2),
              ang: (Math.acos(Math.max(-1, Math.min(1, d / (na * nb || 1)))) * 180 / Math.PI).toFixed(0) + '°',
              cos: (d / (na * nb || 1)).toFixed(3)
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          const x = P.ix(e.x), y = P.iy(e.y);
          if (e.type === 'down') {
            const da = (a[0] - x) ** 2 + (a[1] - y) ** 2, db = (b[0] - x) ** 2 + (b[1] - y) ** 2;
            drag = Math.min(da, db) < .35 ? (da < db ? 'a' : 'b') : null;
          } else if (e.type === 'move' && e.down && drag) {
            if (drag === 'a') a = [x, y]; else b = [x, y];
            S.redraw();
          } else if (e.type === 'up') drag = null;
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'a' }, { c: Viz.theme().red, t: 'b' }, { c: Viz.theme().green, t: 'a + b' }, { c: Viz.theme().faint, t: 'projection of a onto b' }]);
      },

      mat: function (host) {
        let A = [[1.2, 0.6], [0.3, 1.1]]; // columns are A[:,0], A[:,1]
        let drag = null;
        const st = Viz.controls(host, [
          { k: 'shape', label: 'shape', type: 'buttons', value: 'circle', options: [{ v: 'circle', t: 'circle + grid' }, { v: 'f', t: 'letter F' }] },
          { k: 'eig', label: 'show eigenvectors (symmetric part)', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'det', label: 'determinant', cls: 'key' }, { k: 's1', label: 'σ₁ (max stretch)' }, { k: 's2', label: 'σ₂ (min stretch)' }, { k: 'cond', label: 'condition κ' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-2.4, 2.4] }).frame({});
            const ap = (p) => [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
            // faint input grid
            for (let g = -2; g <= 2; g++) {
              P.line([[-2.4, g], [2.4, g]], { color: T.line, width: 1, alpha: .5 });
              P.line([[g, -2.4], [g, 2.4]], { color: T.line, width: 1, alpha: .5 });
            }
            // transformed grid
            for (let g = -2; g <= 2; g++) {
              const l1 = [], l2 = [];
              for (let t = -2.4; t <= 2.41; t += .3) { l1.push(ap([t, g])); l2.push(ap([g, t])); }
              P.line(l1, { color: T.blue, width: 1, alpha: .35 });
              P.line(l2, { color: T.blue, width: 1, alpha: .35 });
            }
            if (st.shape === 'circle') {
              const circ = [], img = [];
              for (let t = 0; t <= 6.3; t += .05) { circ.push([Math.cos(t), Math.sin(t)]); img.push(ap([Math.cos(t), Math.sin(t)])); }
              P.line(circ, { color: T.faint, width: 1.4, dash: [4, 3] });
              P.line(img, { color: T.red, width: 2.4 });
            } else {
              const F = [[0, 0], [0, 1.4], [.9, 1.4], [.9, 1.1], [.3, 1.1], [.3, .8], [.75, .8], [.75, .5], [.3, .5], [.3, 0], [0, 0]];
              P.line(F, { color: T.faint, width: 1.4, dash: [4, 3] });
              P.line(F.map(ap), { color: T.red, width: 2.4 });
            }
            // columns
            P.arrow(0, 0, A[0][0], A[1][0], { color: T.blue, width: 2.6 });
            P.arrow(0, 0, A[0][1], A[1][1], { color: T.green, width: 2.6 });
            P.text(A[0][0], A[1][0], ' col 1', { color: T.blue, font: '11px ui-monospace' });
            P.text(A[0][1], A[1][1], ' col 2', { color: T.green, font: '11px ui-monospace' });
            const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
            const sv = Num.svd([[A[0][0], A[0][1]], [A[1][0], A[1][1]]]);
            if (st.eig) {
              const Sym = [[A[0][0], (A[0][1] + A[1][0]) / 2], [(A[0][1] + A[1][0]) / 2, A[1][1]]];
              const e = Num.eigSym(Sym);
              e.vectors.forEach((v, i) => {
                const s = Math.sign(e.values[i]) * Math.min(2.2, Math.abs(e.values[i]));
                P.line([[-v[0] * 2.6, -v[1] * 2.6], [v[0] * 2.6, v[1] * 2.6]], { color: T.amber, width: 1, dash: [6, 4], alpha: .8 });
              });
            }
            out({
              det: det.toFixed(2), s1: sv.s[0].toFixed(2), s2: sv.s[1].toFixed(2),
              cond: (sv.s[0] / (sv.s[1] || 1e-9)).toFixed(1)
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
          } else if (e.type === 'move' && e.down && drag !== null) {
            A[0][drag] = x; A[1][drag] = y; S.redraw();
          } else if (e.type === 'up') drag = null;
        });
        Viz.buttons(host, [
          { label: 'Identity', on: () => { A = [[1, 0], [0, 1]]; S.redraw(); } },
          { label: 'Rotation 30°', on: () => { const c = Math.cos(.52), s = Math.sin(.52); A = [[c, -s], [s, c]]; S.redraw(); } },
          { label: 'Shear', on: () => { A = [[1, 1], [0, 1]]; S.redraw(); } },
          { label: 'Rank-1 (collapse)', on: () => { A = [[1.4, .7], [1.2, .6]]; S.redraw(); } }
        ]);
      }
    },
    quiz: [
      {
        q: 'Two feature vectors have cosine similarity 0. What does that tell you?',
        options: ['They are identical', 'They are orthogonal — no linear alignment', 'One is the negative of the other', 'Both have zero length'],
        answer: 1,
        why: 'Cosine similarity is the cosine of the angle between the two arrows, so a value of 0 means the angle is exactly ninety degrees and the dot product vanishes: the sum of their pairwise products cancels out to nothing. Option C is the tempting one, because "no similarity" sounds like "opposite", but pointing the other way is the <i>most</i> dissimilar case a cosine can describe and it scores $-1$, not 0. Zero is the neutral middle of the range, not its bottom. One caution worth carrying: orthogonality only rules out <i>linear</i> alignment. Two features can have a cosine of zero and still be perfectly predictable from each other through a curved relationship, which is exactly the gap between correlation and dependence that §1.3 makes precise.'
      },
      {
        q: 'A 2×2 matrix has determinant 0. Which statement follows?',
        options: ['It is symmetric', 'Its columns are linearly dependent and it is not invertible', 'It has orthogonal columns', 'Its condition number is 1'],
        answer: 1,
        why: 'The determinant is the factor by which the map multiplies area, so a determinant of zero says the unit square is squashed to zero area — the whole plane has been flattened onto a line, or onto the origin. That can only happen when the two columns lie along the same line, which is what linear dependence means. Because a whole line of inputs now maps to each single output point, there is no way to run the map backwards, so no inverse exists. Option D is the attractive wrong answer if you have half-remembered that the condition number measures degeneracy: it does, but in the opposite direction. A condition number of 1 describes the <i>best</i>-behaved matrices, the pure rotations and uniform scalings; a collapsing matrix sends the condition number to infinity, because the smallest singular value has gone to zero. In data this situation is collinearity, and it is precisely the failure that adding $\\lambda I$ in ridge regression repairs (§2.3).'
      },
      {
        q: 'You are matching customers by their basket of purchases. One customer buys the same mix as another but ten times as much of everything. Which measure calls them similar?',
        options: ['Euclidean distance $\\|a-b\\|$', 'Cosine similarity', 'The raw dot product $a \\cdot b$', 'Neither — they are genuinely different customers'],
        answer: 1,
        why: 'Scaling a vector by ten leaves its direction untouched and multiplies its length by ten. Cosine similarity divides both lengths out before comparing, so it sees only the direction and returns exactly 1. Euclidean distance sees a very large gap, because the two arrow tips are now far apart in absolute terms. The raw dot product is the trap here: it does respond to alignment, which makes it feel like the right tool, but it also multiplies by both lengths, so simply spending more inflates the score against everybody — a big customer would come out as the closest match to every other customer in the database. This is exactly why retrieval systems normalise their embedding vectors to unit length before indexing them: once every vector has length 1, the dot product and the cosine become the same number, and magnitude can no longer contaminate the ranking (§5.10).'
      }
    ],
    cards: [
      { q: 'Dot product, in words', a: 'Length × length × cosine of the angle. Alignment scaled by magnitude — the score in linear models and in attention.' },
      { q: 'Cosine similarity vs Euclidean distance', a: 'Cosine compares direction with both lengths divided out; distance compares position. Use cosine when magnitude is an artefact.' },
      { q: 'What are the columns of a matrix?', a: 'The images of the basis vectors. A matrix is fully specified by where it sends $(1,0)$ and $(0,1)$.' },
      { q: 'What does $\\det A = 0$ mean?', a: 'The map flattens a dimension, so the columns are linearly dependent and no inverse exists. In data: collinear features.' },
      { q: 'The matrix shape rule', a: '$[n \\times d]$ times $[d \\times k]$ gives $[n \\times k]$ — inner dimensions must match, and they cancel.' },
      { q: 'Every linear map is which three operations?', a: 'Rotate, scale along axes, rotate — the SVD $A = U\\Sigma V^\\mathsf{T}$.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.3 */
  ML.section({
    id: 'calculus-basics', track: 'start', num: '0.3',
    title: 'Derivatives, gradients, and the chain rule',
    lede: 'Training is one idea repeated many times: work out which way the loss goes up, then step the other way. The derivative is how you work it out, the gradient is the same measurement taken along every parameter at once, and the chain rule is what makes it computable through a hundred layers of a network. This section builds all three from the slope of a straight line and assumes nothing else.',
    html: `
<p>Suppose you have a model with one adjustable number in it — call it $w$ — and, thanks to §0.1, a loss $L(w)$ that reports how badly the model is doing for any value of $w$ you care to try. Your task is to find the $w$ that makes $L$ as small as possible. How would you actually go about it?</p>

<p>The first plan anyone proposes is to try things. Compute $L(0)$, then $L(1)$, then $L(2)$, sweep across a sensible range, and keep whichever value came out best. For a single parameter that is a perfectly reasonable plan, and people really do it.</p>

<p>Now watch it collapse. Suppose you allow ten candidate values per parameter. One parameter needs ten evaluations of the loss. Two parameters need every pairing of the two lists, so a hundred. Ten parameters need $10^{10}$, which is ten billion evaluations; at a millisecond each that is about four months of computing. Twenty parameters need $10^{20}$ evaluations, which at the same rate is roughly three billion years. And a small neural network has more like a million parameters.</p>

<p>The plan does not fail because computers are slow. It fails because of what it asks. Evaluating $L$ at a point answers only one question — <i>how bad is it here?</i> — and that answer tells you nothing whatsoever about anywhere else, so you are obliged to visit everywhere. What you want instead is a question whose answer <i>points</i>: stand at your current $w$ and ask which way the loss goes down. One such answer replaces the entire sweep, because it tells you where to go next.</p>

<p>That question has an answer, it is computable, and it is called the derivative.</p>

<h2><span class="sn">0.3.1</span> From the slope of a line to the slope of a curve</h2>

<p>Start with something you already know. A straight line has a slope, and you find it by taking any two points on the line and dividing the rise by the run. If the line passes through $(1, 4)$ and $(3, 10)$, the rise is $10 - 4 = 6$, the run is $3 - 1 = 2$, and the slope is $6/2 = 3$. Move one unit to the right anywhere on that line and you go up by exactly 3. The slope is a single number that describes the whole line, because a line does not change its mind.</p>

<p>A curve does change its mind. Take $f(x) = x^2$. Between $x = 0$ and $x = 1$ it rises by 1. Between $x = 3$ and $x = 4$ it rises by $16 - 9 = 7$. There is no single slope, so the honest question becomes: what is the slope <i>at one particular place</i>, say at $x = 3$?</p>

<p>Here is the trick that answers it, and it is genuinely the whole of differential calculus. Pick a small step $h$, take the rise over the run between $x$ and $x + h$, and then make $h$ smaller and see whether the answer settles down. For $f(x) = x^2$ at $x = 3$:</p>

${H.table(['step $h$', 'rise: $f(3+h) - f(3)$', 'rise ÷ run'], [
      ['$1$', '$16 - 9 = 7$', '$7$'],
      ['$0.1$', '$9.61 - 9 = 0.61$', '$6.1$'],
      ['$0.01$', '$9.0601 - 9 = 0.0601$', '$6.01$'],
      ['$0.001$', '$9.006001 - 9 = 0.006001$', '$6.001$']
    ])}

<p>The numbers are marching towards 6 and are not going anywhere else. That settled value is the <b>derivative</b> of $f$ at $x = 3$, and we write it $f'(3) = 6$, read aloud as "f prime of three". You will also see it written $\\frac{df}{dx}$, read "d f by d x", which is a deliberate reminder that it began life as a rise divided by a run. The formal statement of the shrinking process is</p>

$$f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$$

<p>where $\\lim_{h \\to 0}$ is read "the limit as h goes to zero" and means precisely what the table shows: the value the fraction approaches as $h$ is made small, not the value at $h = 0$ itself, which would be $0/0$ and meaningless. If that distinction feels like a dodge, notice that the table never divides by zero anywhere — it only ever divides by $1$, $0.1$, $0.01$ and so on, and simply observes where the results are heading.</p>

${H.analogy(`<p>Your car has an odometer and a speedometer. The odometer reads a function: total distance travelled as of now. The speedometer reads that function's derivative: how fast the distance is currently growing, in miles per hour, at this instant.</p>
<p>Two things about the speedometer are exactly true of derivatives. First, the reading is meaningful even though "distance travelled in an instant" is zero — the speedometer reports a <i>rate</i>, not a distance, and a rate survives the shrinking of the interval when a distance does not. Second, the reading is local: 70 mph tells you what happens over the next few seconds and says nothing at all about whether there is a junction two miles ahead. Derivatives are speedometers, and every optimiser in this course is driving on a speedometer alone.</p>`)}

<p>Doing that table by hand for every function would be intolerable, so instead we do the limit once, symbolically, and get a rule that works everywhere.</p>

${H.deriv('the derivative of $f(x) = x^2$, from the definition', [
      ['$\\dfrac{f(x+h) - f(x)}{h} = \\dfrac{(x+h)^2 - x^2}{h}$', 'Substitute the definition of $f$ into the difference quotient. Nothing has happened yet except writing $f$ out.'],
      ['$= \\dfrac{x^2 + 2xh + h^2 - x^2}{h}$', 'Expand $(x+h)^2$ using ordinary algebra: $(x+h)^2 = x^2 + 2xh + h^2$.'],
      ['$= \\dfrac{2xh + h^2}{h}$', 'The $x^2$ and the $-x^2$ cancel. This is the step that makes the whole thing work: the parts that do not depend on $h$ have gone, leaving every surviving term with a factor of $h$ in it.'],
      ['$= 2x + h$', 'Divide top and bottom by $h$. This is legitimate because $h$ is small but never actually zero — we are studying where the fraction heads, not evaluating it at zero.'],
      ['$f\'(x) = 2x$', 'Now let $h$ shrink. The expression $2x + h$ visibly approaches $2x$, so that is the limit. At $x = 3$ this gives 6, exactly the number the table was marching towards.']
    ], 'The same argument run on $x^n$ gives $nx^{n-1}$, which is the power rule in the table below. Notice that the derivative is itself a function of $x$ rather than a single number: $f\'$ tells you the slope at every point, and you get the slope at a particular place by evaluating it there.')}

<h3>The two readings that matter</h3>

<p><b>Reading one: the derivative predicts what a small step will do.</b> Rearranging the definition and dropping the limit gives</p>

$$f(x + \\epsilon) \\approx f(x) + \\epsilon\\, f'(x)$$

<p>where $\\epsilon$ is the Greek letter epsilon and denotes a small step. In words: to guess the function's value a little way along, take where you are and add the step multiplied by the slope. Try it. At $x = 3$ with $f(x) = x^2$ and $f'(3) = 6$, stepping $\\epsilon = 0.1$ predicts $9 + 0.1 \\times 6 = 9.6$. The true value is $3.1^2 = 9.61$, so the prediction is wrong by 0.01. Halve the step to $0.05$ and the prediction is $9.3$ against a true $9.3025$, wrong by $0.0025$. Halving the step quartered the error, which is the signature of an approximation that is right to first order: the leftover shrinks like $\\epsilon^2$, and $\\epsilon^2$ becomes negligible against $\\epsilon$ long before $\\epsilon$ itself becomes negligible.</p>

<p>This reading is the one optimisation lives on. It says the derivative is the best straight-line stand-in for the curve near where you are standing, and a straight line is something you can reason about immediately.</p>

<p><b>Reading two: where the derivative is zero, the function is flat.</b> A zero slope means the tangent line is horizontal, which happens at the bottom of a valley, at the top of a hill, and at a saddle where the surface turns up in one direction and down in another. Every candidate for "the minimum" satisfies $f'(x) = 0$, which is why so much of optimisation amounts to hunting for the places where a derivative vanishes.</p>

<h2><span class="sn">0.3.2</span> The rules, and one gradient step by hand</h2>

<p>You will almost never compute a limit again. In practice you memorise a handful of rules, and everything else is assembled from them. Each of these has a place in this course where it earns its keep.</p>

${H.table(['Rule', 'Statement', 'Where you meet it here'], [
      ['Power', '$\\frac{d}{dx}x^n = nx^{n-1}$', 'Squared error $(\\hat y - y)^2$ differentiates to $2(\\hat y - y)$ — the residual itself'],
      ['Constant multiple', '$\\frac{d}{dx}\\,c\\,f(x) = c\\,f\'(x)$', 'Why the $\\tfrac{1}{2}$ in $\\tfrac{1}{2}(\\hat y - y)^2$ is free: it cancels the 2'],
      ['Sum', '$(f+g)\' = f\' + g\'$', 'The loss is a sum over examples, so its gradient is the sum of per-example gradients'],
      ['Exponential / log', '$\\frac{d}{dx}e^x = e^x$, $\\frac{d}{dx}\\ln x = 1/x$', 'Every log-likelihood, and therefore every cross-entropy loss (§1.5)'],
      ['Product', '$(uv)\' = u\'v + uv\'$', 'Gated units, attention scores, anything where two learned quantities multiply'],
      ['Chain', '$\\frac{d}{dx}f(g(x)) = f\'(g(x))\\,g\'(x)$', '<b>Backpropagation, in its entirety</b> — §0.3.5 below and §3.2'],
      ['Sigmoid', '$\\sigma\' = \\sigma(1-\\sigma)$', 'The two-line logistic gradient (§2.4), and the cause of vanishing gradients']
    ])}

<p>Now put the first reading to work. If $f'(x)$ is positive then stepping to the right increases $f$, so to <i>decrease</i> $f$ you step left. If $f'(x)$ is negative then stepping right decreases $f$. In both cases the safe move is to step in the direction opposite to the sign of the derivative, and the natural way to write that in one expression is</p>

$$x \\leftarrow x - \\eta\\, f'(x)$$

<p>which is <b>gradient descent</b>, already met in §0.1 and now earned. The arrow means "replace the left with the right", and $\\eta$ (eta) is the learning rate. Notice that the update automatically takes big steps where the slope is steep and small ones where it flattens out, which is exactly the behaviour you would want and nobody had to program in.</p>

${H.worked('four steps of gradient descent, by hand', `<p>Take $f(x) = x^2$, so $f'(x) = 2x$, and start at $x = 3$ with $\\eta = 0.1$.</p>
<p>The update is $x \\leftarrow x - 0.1 \\times 2x = 0.8x$, so each step multiplies $x$ by $0.8$:</p>
<p>$3 \\to 2.4 \\to 1.92 \\to 1.536 \\to 1.2288$, and the loss falls $9 \\to 5.76 \\to 3.69 \\to 2.36 \\to 1.51$.</p>
<p>It is heading to zero, which is indeed the minimum, and it never quite arrives — each step covers 20 per cent of the remaining distance. That "approaches but does not reach" behaviour is completely normal and is why training runs stop on a budget rather than on arrival.</p>`)}

${H.pitfall(`<p>Now repeat the same calculation with $\\eta = 1.1$ instead. The update becomes $x \\leftarrow x - 1.1 \\times 2x = -1.2x$, so from $x = 3$ you get</p>
<p>$3 \\to -3.6 \\to 4.32 \\to -5.184 \\to 6.2208$, with the loss climbing $9 \\to 12.96 \\to 18.66 \\to 26.9 \\to 38.7$.</p>
<p>Every step is in the correct direction and every step makes things worse, because each one overshoots the bottom and lands further up the far side than it started. This is divergence, and it is the most common way a training run fails outright: the loss becomes NaN within a few dozen steps and everyone blames the model.</p>
<p>For this function you can state the boundary exactly. The update multiplies $x$ by $(1 - 2\\eta)$, which shrinks $x$ only when $|1 - 2\\eta| < 1$, that is when $\\eta$ lies strictly between 0 and 1. The 2 in that condition is the curvature of $f$, and the general statement is that the largest stable step is $2$ divided by the curvature. Steeper bowls demand smaller steps. Keep that sentence; §0.3.4 turns it into the single most important fact about how hard a model is to train.</p>`)}

<h3>Watching it happen</h3>

<p><b>What you are looking at.</b> The blue curve is the function $f$ you are minimising, plotted against $x$ along the horizontal axis. The red dot is your current position $x$, and the red dashed line through it is the <b>tangent</b> — the straight-line approximation $f(x) + \\epsilon f'(x)$ from reading one, drawn out across the whole panel so you can see how far it stays close to the curve. The green arrow runs from where you are to where one gradient step would put you, and the green dot marks that landing point. The readout gives $x$, $f(x)$, the slope $f'(x)$, and the next $x$ the update rule would produce.</p>

<p><b>What to do with it.</b> Click or drag anywhere inside the panel; only the horizontal position is used, so the point snaps onto the curve at whatever $x$ you clicked. Move it around on the $x^2/2$ curve and watch the dashed tangent tilt: steep and negative on the left, flat at the bottom, steep and positive on the right. Then press <i>Take one step</i> repeatedly and watch the green landing point creep towards the bottom in ever-smaller hops — the same shrinking-step behaviour the worked example above computed by hand. Now raise the step size $\\eta$ past 1 and step again: the point overshoots and lands on the <i>far</i> side of the valley, then flips back, closing in from alternating directions. For this particular curve the curvature is 1, so by the rule above it tolerates anything below $\\eta = 2$ and the slider cannot break it. Switch to $x^2/6 + \\sin 2x$, whose curvature reaches about 4.3, and press <i>Run 30 steps</i> at a large $\\eta$: now the point genuinely refuses to settle.</p>

<p><b>The thing genuinely worth noticing.</b> Switch the function to $x^4/8 - x^2$, which has two separate valleys with a bump between them. Put the point on the left of the bump and run it: it settles into the left valley. Reset, put it on the right, and run it again: it settles into the right one. The optimiser did not choose the better valley, and it did not even know the other one existed, because a derivative is a purely local measurement — the speedometer from the analogy above cannot see a junction two miles ahead. Where you start decides where you finish. That single observation is why deep learning cares about initialisation (§3.4), why people run the same training job with several random seeds, and why "the loss stopped improving" is not the same claim as "this is the best the model can do".</p>

${H.lab('deriv', 'Slope, tangent, and one gradient step', 'Click or drag anywhere in the panel to move the point — only the horizontal position matters. The red dashed line is the tangent, which is the straight-line model the derivative gives you. The green arrow is one gradient step at the current $\\eta$.')}

<h2><span class="sn">0.3.3</span> Many knobs at once: partial derivatives and the gradient</h2>

<p>Real models have more than one parameter, so the picture has to grow. The awkwardness is that "the slope" is no longer a single question. If you are standing on a hillside, the ground slopes differently depending on which way you face, so asking how steep it is has no answer until you say in which direction.</p>

<p>The resolution is to ask a series of very restricted questions and then assemble the answers. Pick one parameter, freeze every other parameter at its current value, and ask for the ordinary one-variable derivative in that single direction. That is a <b>partial derivative</b>, written $\\partial f/\\partial w$. The symbol $\\partial$ is a stylised letter d and is read "partial", so the whole expression is read "partial f by partial w". It signals nothing more than "there are other variables here and I am holding them still".</p>

${H.worked('both partial derivatives of a tiny loss, with numbers', `<p>Take a model with one feature whose value happens to be $x = 2$, a weight $w$ and a bias $b$, so the prediction is $2w + b$. The true label is $y = 5$, and the loss is the squared error</p>
<p>$L(w, b) = (2w + b - 5)^2.$</p>
<p>Evaluate everything at $w = 1$, $b = 0$. The prediction is $2(1) + 0 = 2$, the error is $2 - 5 = -3$, and the loss is $(-3)^2 = 9$.</p>
<p>For $\\partial L/\\partial w$, freeze $b$ and differentiate. The outside is a square, contributing $2 \\times (2w + b - 5)$ by the power rule, and the inside contributes the rate at which $2w + b - 5$ changes as $w$ changes, which is 2. Multiplying, $\\partial L/\\partial w = 2(-3)(2) = -12$.</p>
<p>For $\\partial L/\\partial b$, freeze $w$. The outside contributes the same $2(-3)$, and the inside changes at rate 1 as $b$ changes, so $\\partial L/\\partial b = 2(-3)(1) = -6$.</p>
<p>Both are negative, meaning increasing either parameter reduces the loss, which is right — the model is predicting 2 when the answer is 5, so it needs to predict higher. And $\\partial L/\\partial w$ is exactly twice $\\partial L/\\partial b$, because $w$ is multiplied by the feature value 2 before it reaches the prediction, so a nudge to $w$ has twice the leverage of the same nudge to $b$.</p>`)}

<p>Collect one partial derivative per parameter into a list and you have the <b>gradient</b>:</p>

$$\\nabla f(\\theta) = \\left(\\frac{\\partial f}{\\partial\\theta_1},\\; \\frac{\\partial f}{\\partial\\theta_2},\\; \\ldots,\\; \\frac{\\partial f}{\\partial\\theta_p}\\right)$$

<p>The symbol $\\nabla$ is called <b>nabla</b> or "del", and $\\nabla f$ is read "the gradient of f". It is a list of numbers, one per parameter, and by §0.2 a list of numbers is a vector — so the gradient is an arrow, and it lives in the space of parameters rather than the space of data. In the worked example above the gradient is $\\nabla L = (-12, -6)$, an arrow in the $(w, b)$ plane.</p>

<p>Continue that example one step. Gradient descent updates every parameter at once by $\\theta \\leftarrow \\theta - \\eta \\nabla L$. With $\\eta = 0.05$ that is $w \\leftarrow 1 - 0.05(-12) = 1.6$ and $b \\leftarrow 0 - 0.05(-6) = 0.3$. The new prediction is $2(1.6) + 0.3 = 3.5$, the new error is $-1.5$, and the new loss is $2.25$ where it was 9. One step, and three quarters of the loss is gone.</p>

<h3>Why the gradient is the steepest direction</h3>

<p>The gradient is usually introduced with the claim that it "points in the direction of steepest increase", which sounds like a fact you have to take on trust. It is not; it follows in two lines from the dot product of §0.2.</p>

${H.deriv('why $\\nabla f$ is the direction of fastest increase', [
      ['$f(\\theta + \\epsilon u) \\approx f(\\theta) + \\epsilon\\,(\\nabla f \\cdot u)$', 'The multi-variable version of reading one. Step a small distance $\\epsilon$ in the direction of a unit vector $u$; the change in $f$ is the step size times the dot product of the gradient with the direction you chose. This is exactly the one-variable statement $f(x+\\epsilon) \\approx f(x) + \\epsilon f\'(x)$ with the single slope replaced by a whole list of them.'],
      ['change in $f$ $\\approx \\epsilon\\,\\|\\nabla f\\|\\,\\|u\\|\\cos\\theta$', 'Rewrite the dot product geometrically, using the identity derived in §0.2: a dot product is length times length times the cosine of the angle between the two arrows. Here $\\theta$ is the angle between the gradient and your chosen direction.'],
      ['$= \\epsilon\\,\\|\\nabla f\\|\\cos\\theta$', 'The direction $u$ is a unit vector, so $\\|u\\| = 1$ and that factor disappears. We are comparing directions on equal terms, which is precisely why $u$ was required to have length 1.'],
      ['maximised when $\\cos\\theta = 1$, i.e. $u$ parallel to $\\nabla f$', 'Everything else in the expression is fixed once you are standing at $\\theta$, so the only thing you control is the cosine. Cosine is largest at $\\theta = 0$, meaning you should walk in exactly the gradient\'s own direction; it is smallest at $180°$, meaning the fastest <i>decrease</i> is straight against it.']
    ], 'So $-\\nabla f$ is the steepest descent direction, and the minus sign in every update rule on this site is the consequence. Notice the bonus in line 3: any direction at right angles to the gradient has $\\cos\\theta = 0$, so moving along it changes $f$ by nothing to first order. Those are the contour lines of the loss surface, and it is why gradients always cross contours perpendicularly.')}

<h2><span class="sn">0.3.4</span> Curvature, and why some models train easily</h2>

<p>The derivative of the derivative is the <b>second derivative</b>, written $f''(x)$, and it measures curvature: how fast the slope itself is changing. For $f(x) = x^2$ we have $f' = 2x$ and $f'' = 2$, a constant — the slope increases at a steady rate, which is what makes a parabola a parabola.</p>

<p>With many parameters, every partial derivative can be differentiated with respect to every parameter, so the second derivative becomes a whole grid of numbers. That grid is the <b>Hessian</b>, written $H$, an $[p \\times p]$ matrix whose entry in row $i$ and column $j$ is $\\partial^2 f / \\partial\\theta_i \\partial\\theta_j$. You will rarely build one — for a model with a million parameters it would have $10^{12}$ entries — but you need what it means, because it explains the single biggest difference between a model that trains in an hour and one that crawls.</p>

<p>Recall the boundary from the pitfall above: the largest stable learning rate is 2 divided by the curvature. With many parameters there is a curvature in every direction, and the largest one sets the speed limit for the whole run, because a step that is stable along the gentle directions but unstable along the steep one will still blow up. So you are forced to choose $\\eta$ small enough for the steepest direction, and then every gentle direction is crawled along at that same tiny rate.</p>

${H.worked('the cost of an elongated bowl, in steps', `<p>Take $f(w_1, w_2) = \\tfrac{1}{2}(w_1^2 + 100\\,w_2^2)$. The curvature is 1 along $w_1$ and 100 along $w_2$ — the same bowl you would get from two features measured on scales that differ by a factor of ten.</p>
<p>The steep direction forces $\\eta < 2/100 = 0.02$, so take $\\eta = 0.019$. Along $w_2$ the update multiplies by $1 - 0.019 \\times 100 = -0.9$, which is fine. Along $w_1$ it multiplies by $1 - 0.019 = 0.981$.</p>
<p>To shrink $w_1$ by a factor of 100 you therefore need $n$ steps with $0.981^n = 0.01$, which gives $n = \\ln(0.01)/\\ln(0.981) \\approx 240$ steps. Had both curvatures been 1, a single well-chosen step would have done it.</p>
<p>Two hundred and forty times slower, and the model, the data and the loss are all unchanged. The only thing wrong is the <i>ratio</i> of the curvatures.</p>`)}

<p>That ratio — largest curvature divided by smallest — is the <b>condition number</b> of the Hessian, the same quantity §0.2 met as the ratio of singular values of a matrix. A condition number near 1 is a round bowl and trains beautifully. A large condition number is a long thin valley, and gradient descent zig-zags across it making painfully slow progress along its length.</p>

${H.intuition(`<p>This is the second time features on mismatched scales have caused trouble, and it is worth connecting the two.</p>
<p>In §0.2 the complaint was about distance: a feature measured in years and a feature measured as a ratio contribute wildly unequal amounts to $\\|a - b\\|$, so nearest-neighbour methods effectively ignore the small-numbered column. Here the complaint is about optimisation: a feature with large values produces large gradients along its weight, so the loss surface is steep in that direction and shallow in the others, the condition number blows up, and training crawls.</p>
<p>Both complaints have the same cause and the same one-line fix — standardise every column to mean 0 and standard deviation 1 before you do anything else. That is why <i>scale your features</i> appears in every practical checklist without much explanation attached. The explanation is this paragraph, and §0.5 is about doing it without leaking information between folds.</p>`)}

${H.more('what the second derivative buys you if you can afford it', `<p>If you know the curvature, you no longer have to guess a step size — you can compute the step that jumps straight to the bottom of the local parabola. In one dimension that is <b>Newton's method</b>, $x \\leftarrow x - f'(x)/f''(x)$, and on a genuine parabola it lands on the exact minimum in a single step from anywhere.</p>
<p>The multi-parameter version replaces the division by multiplication with the inverse Hessian, $\\theta \\leftarrow \\theta - H^{-1}\\nabla f$, and inherits both the speed and the price. Building $H$ costs $O(p^2)$ memory and inverting it costs $O(p^3)$ time, which for a million parameters is out of the question by many orders of magnitude. It is also only trustworthy when the surface really is bowl-shaped nearby, which for a deep network it frequently is not.</p>
<p>The practical middle ground is to approximate curvature cheaply, one parameter at a time, from the gradients you are already computing — which is exactly what Adam and its relatives do when they divide each parameter's step by a running estimate of the size of its own recent gradients (§3.5). Seen this way, adaptive optimisers are not a bag of tricks; they are an attempt to buy some of Newton's benefit at first-order prices. §1.12 develops the full picture.</p>`)}

<h2><span class="sn">0.3.5</span> The chain rule, which is the entire mechanism of backpropagation</h2>

<p>Everything so far assumed you could differentiate the function in front of you. A network is not a function in front of you; it is a function inside a function inside a function, twenty or a hundred deep. The chain rule is what lets you differentiate such a thing without ever writing it out.</p>

${H.analogy(`<p>You are converting money. One pound buys 1.25 dollars, and one dollar buys 150 yen. How many yen does a pound buy? You multiply: $1.25 \\times 150 = 187.5$.</p>
<p>Every one of those numbers is a rate — output per unit of input — and the rule for chaining rates is that they multiply. That is the chain rule in full, and the only reason the calculus version looks harder is that the rates change depending on where you are standing, so each one has to be evaluated at the right place.</p>`)}

<p>Written out, for a composition $f(g(x))$ — read "f of g of x", meaning apply $g$ first and then feed the result to $f$ — the rule is</p>

$$\\frac{d}{dx} f(g(x)) = f'(g(x)) \\cdot g'(x)$$

<p>and in words: differentiate the outer function, evaluate that derivative at the value the inner function actually produced, and multiply by the derivative of the inner function. The phrase "evaluate at the value the inner function produced" is the part people drop, and it is the part the analogy makes obvious — the dollar-to-yen rate has to be the rate that applies at the number of dollars you actually have.</p>

${H.worked('the chain rule against a direct calculation', `<p>Let $g(x) = 2x + 1$ and $f(u) = u^2$, so the composition is $f(g(x)) = (2x+1)^2$. Differentiate at $x = 3$.</p>
<p><b>By the chain rule.</b> The inner function gives $g(3) = 7$. The outer derivative is $f'(u) = 2u$, evaluated at $u = 7$, giving 14. The inner derivative is $g'(x) = 2$. Multiply: $14 \\times 2 = 28$.</p>
<p><b>Directly.</b> Expand first: $(2x+1)^2 = 4x^2 + 4x + 1$, whose derivative is $8x + 4$. At $x = 3$ that is $24 + 4 = 28$. The same.</p>
<p><b>Numerically, as a check.</b> $(2 \\times 3.001 + 1)^2 = 7.002^2 = 49.028004$, and $(49.028004 - 49)/0.001 = 28.004$. Marching towards 28, exactly as the table in §0.3.1 marched towards 6.</p>`)}

<p>Now the point of all this. A neural network computes something of the shape</p>

$$L = \\ell\\big(f_3(f_2(f_1(x)))\\big)$$

<p>where each $f$ is a layer and $\\ell$ is the loss. Applying the chain rule to that composition gives the derivative as a <i>product</i> of the local derivatives of each stage. <b>Backpropagation is nothing more than evaluating that product, starting from the loss end and working backwards, storing each partial result so that no factor is ever computed twice.</b> There is no additional idea in it. §3.2 is this paragraph, done carefully, with matrices in place of single numbers.</p>

${H.deriv('the gradient of a one-unit network, which is exactly what the lab below computes', [
      ['$z = wx + b$', 'The linear part: multiply the input by the weight and add the bias.'],
      ['$a = \\mathrm{ReLU}(z) = \\max(0, z)$', 'The non-linearity. ReLU passes positive numbers through unchanged and flattens everything negative to zero.'],
      ['$L = \\tfrac{1}{2}(a - y)^2$', 'The loss. The $\\tfrac{1}{2}$ is there purely so that the 2 from the power rule cancels and the answer comes out clean.'],
      ['$\\dfrac{\\partial L}{\\partial a} = a - y$', 'Differentiate the loss with respect to its own input. Power rule on the square gives $2 \\times \\tfrac{1}{2}(a-y) = (a-y)$, and the inner derivative of $(a - y)$ with respect to $a$ is 1. The prediction error <i>is</i> the first gradient.'],
      ['$\\dfrac{\\partial a}{\\partial z} = 1$ if $z > 0$, else $0$', 'The local derivative of ReLU. On the active side the function is the line $a = z$, whose slope is 1; on the flat side it is the constant 0, whose slope is 0. (At exactly $z = 0$ there is no derivative; every framework simply picks a value, usually 0, and no harm comes of it.)'],
      ['$\\dfrac{\\partial L}{\\partial z} = (a-y)\\cdot\\mathbf{1}[z>0]$', 'Chain the two previous lines together: rates multiply. The notation $\\mathbf{1}[z>0]$ is an indicator, worth 1 when the condition holds and 0 when it does not.'],
      ['$\\dfrac{\\partial L}{\\partial w} = \\dfrac{\\partial L}{\\partial z}\\cdot x, \\quad \\dfrac{\\partial L}{\\partial b} = \\dfrac{\\partial L}{\\partial z}$', 'One more link in the chain. Since $z = wx + b$, the local derivative with respect to $w$ is $x$ and with respect to $b$ is 1. So the weight\'s gradient is the incoming signal scaled by the input that arrived on it — which is why a feature that is always large produces large gradients, connecting straight back to §0.3.4.']
    ], 'Read the ladder from the bottom up and you are doing the forward pass; read it from line 4 downwards and you are doing the backward pass. Every quantity on the right-hand side of the backward lines was already computed on the way forward, which is precisely why backpropagation caches activations, and precisely why training a network needs several times the memory of merely running one.')}

<h3>Watching numbers flow both ways</h3>

<p><b>What you are looking at.</b> Four boxes laid left to right are the four stages of the derivation you have just read: the input $x$, the linear combination $z = wx + b$, the activation $a = \\mathrm{ReLU}(z)$, and the loss $L = \\tfrac{1}{2}(a-y)^2$. The bold blue number inside each box is that stage's <i>value</i>, computed left to right, which is the forward pass. The red number printed beneath each box is $\\partial L/\\partial\\,\\cdot$, the derivative of the final loss with respect to that stage, computed right to left, which is the backward pass. Blue arrows along the top carry values forwards; red arrows along the bottom carry gradients backwards. The line at the foot of the panel shows the two numbers you actually want, $\\partial L/\\partial w$ and $\\partial L/\\partial b$.</p>

<p><b>What to do with it.</b> The four sliders set the input $x$, the weight $w$, the bias $b$ and the target $y$. Start by moving the target $y$ and watching the red number under the loss box, which is $a - y$: it is the prediction error, and it flips sign exactly when the prediction crosses the target. Then check the arithmetic yourself on one setting — read off $\\partial L/\\partial z$ and $x$, multiply them, and confirm the product is the $\\partial L/\\partial w$ printed at the bottom. The lab is doing no more than the multiplication you just did.</p>

<p><b>The thing genuinely worth noticing.</b> Now drive $z$ negative, by dragging $w$ or $b$ down until the value in the second box goes below zero. The ReLU output collapses to 0, and — this is the part to watch — every red number to the <i>left</i> of the activation collapses to exactly 0 as well, including $\\partial L/\\partial w$ and $\\partial L/\\partial b$. The loss is still large and the gradient with respect to the activation is still non-zero, but nothing gets past the ReLU, because the local derivative there is 0 and anything multiplied by 0 is 0. That unit is now unable to learn from any example that leaves it switched off: it is a <b>dead ReLU</b>, and you have just built one with two sliders. Everything about activation function design (§3.3) and initialisation (§3.4) is an argument about how to stop this happening across millions of units at once.</p>

${H.lab('chain', 'A computation graph, with numbers flowing both ways', 'Move the sliders and watch the forward values (blue, left to right) and the gradients (red, right to left). Every red number is the product of the local derivatives on the path back from the loss — that is all backpropagation is.')}

${H.key('Every gradient in a network is a product of local derivatives along a path from the loss back to the parameter.')}

<p>That sentence has an immediate and brutal consequence. Multiply forty numbers together and the result is exquisitely sensitive to their typical size. The sigmoid's derivative $\\sigma(1-\\sigma)$ is at most $0.25$, attained when $\\sigma = 0.5$, and usually a good deal less. A forty-layer stack of sigmoids therefore attenuates the gradient by at most $0.25^{40} \\approx 8 \\times 10^{-25}$ before it reaches the first layer, which is indistinguishable from zero in any arithmetic a computer performs. The early layers receive no instruction at all and simply never move: <b>vanishing gradients</b>.</p>

<p>The same argument run the other way is just as unforgiving. If the typical local factor is 1.5 rather than 0.25, forty layers multiply the gradient by roughly $1.5^{40} \\approx 1.1 \\times 10^7$, and the first parameter update is ten million times too large. That is <b>exploding gradients</b>, and it usually announces itself as a loss that becomes NaN in the first few hundred steps.</p>

<p>Now read the standard deep-learning toolkit as answers to that one problem. ReLU has a local derivative of exactly 1 wherever it is active, so it neither shrinks nor grows the product. A residual connection adds a path whose local derivative is exactly 1, giving the gradient a route to the early layers that bypasses the multiplication entirely (§3.7). Normalisation layers keep the activations in a range where the local derivatives stay near 1 (§3.6). Gradient clipping simply refuses to apply an update larger than a set size (§4.11). Four apparently unrelated techniques, one shared cause.</p>

${H.history(`<p>The chain rule itself is old — Leibniz was using it in the 1670s. What took much longer was noticing that it could be applied <i>mechanically</i> to an arbitrarily large computation, and that the order in which you accumulate the product matters enormously.</p>
<p>Reverse-mode accumulation, which is what backpropagation is, was published in general form by Seppo Linnainmaa in 1970 in the context of estimating rounding errors in long numerical programs; Paul Werbos proposed applying it to neural networks in his 1974 doctoral thesis; and the 1986 paper of Rumelhart, Hinton and Williams is what actually put it into general circulation. It was reinvented independently more than once, which tells you it was not obvious at the time even though it now looks inevitable.</p>
<p>For years afterwards a good deal of effort went into optimisation methods that avoided derivatives altogether — genetic algorithms, simulated annealing, and the like — partly because differentiating a large hand-written program was genuinely miserable work. Automatic differentiation removed that objection completely, and one way to read the last decade is that it became worthwhile to design models purely for differentiability, because anything differentiable trains itself (§3.11).</p>`)}

${H.practice(`<p>Three things go wrong with derivatives in real code, and none of them is conceptual.</p>
<p><b>Your hand-derived gradient is wrong.</b> Check it against a finite difference, exactly as the table in §0.3.1 did: compute $(f(\\theta + h) - f(\\theta - h))/2h$ for each parameter and compare against what your code returned. Use $h \\approx 10^{-5}$ in double precision. Much larger and the approximation error dominates; much smaller and subtracting two nearly equal numbers destroys the answer through cancellation, so the error curve is U-shaped in $h$ and there is a sweet spot in the middle (§1.15).</p>
<p><b>The gradient is fine but the function is not differentiable where you asked.</b> ReLU at zero, absolute value at zero, and any hard threshold. Frameworks pick a convention and move on, and in practice this causes far less trouble than it sounds like it should, because the probability of landing exactly on the kink is negligible.</p>
<p><b>The gradient is fine and the numbers overflow anyway.</b> Computing a softmax or a log-likelihood by exponentiating first will overflow for logits above about 710 in double precision, and it is entirely avoidable by rearranging the algebra so the exponentials never exceed 1. This is why every framework has a fused <code>log_softmax</code> and a <code>BCEWithLogitsLoss</code>, and why §0.5 tells you to use them.</p>`)}

${H.probe([
      ['What is a derivative, in one sentence, without the word limit?', 'The rate at which a function changes per unit change in its input, at one specific point — equivalently, the slope of the best straight-line approximation to the function there.'],
      ['Why does the gradient point in the direction of steepest increase?', 'The change in $f$ from a small unit step $u$ is $\\nabla f \\cdot u$, a dot product, which equals $\\|\\nabla f\\|\\cos\\theta$. That is maximised when $\\theta = 0$, meaning $u$ points along the gradient itself.'],
      ['Why reverse-mode and not forward-mode differentiation?', 'The loss is one number and the parameters are many. Reverse mode costs about one forward pass per <i>output</i>; forward mode costs one per <i>input</i>. One output and a billion inputs means reverse mode wins by a factor of a billion.'],
      ['What does the Hessian tell you, and why do you rarely build one?', 'Curvature in every direction. Its condition number sets both the largest stable learning rate and the convergence rate (§1.9). You rarely build it because for $p$ parameters it has $p^2$ entries and costs $p^3$ to invert.'],
      ['Where do vanishing gradients come from?', 'The chain rule makes every gradient a product of local derivatives. Sigmoid contributes at most 0.25 per layer, so across depth the product collapses geometrically. ReLU, residual connections and normalisation all exist to keep those factors near 1.']
    ], 'Describing backpropagation as "the algorithm that trains neural networks". It is not an algorithm for training; it is a method for computing derivatives cheaply. The training is gradient descent, which then consumes those derivatives. Interviewers separate the two on purpose.')}`,
    labs: {
      deriv: function (host) {
        let x = -1.6;
        const st = Viz.controls(host, [
          { k: 'f', label: 'function', type: 'buttons', value: 'quad', options: [{ v: 'quad', t: 'x²/2' }, { v: 'wave', t: 'x²/6 + sin 2x' }, { v: 'quart', t: 'x⁴/8 − x²' }] },
          { k: 'lr', label: 'step size η', min: .02, max: 1.2, step: .02, value: .35, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const F = {
          quad: [x => x * x / 2, x => x],
          wave: [x => x * x / 6 + Math.sin(2 * x), x => x / 3 + 2 * Math.cos(2 * x)],
          quart: [x => x * x * x * x / 8 - x * x, x => x * x * x / 2 - 2 * x]
        };
        const out = Viz.readout(host, [{ k: 'x', label: 'x' }, { k: 'fx', label: 'f(x)' }, { k: 'd', label: "f '(x)", cls: 'key' }, { k: 'nx', label: 'next x' }]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const [f, df] = F[st.f];
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-3, 5] }).frame({ xlabel: 'x' });
            P.clip(() => {
              P.fn(f, { color: T.blue, width: 2.4 });
              const s = df(x), y0 = f(x);
              P.fn(t => y0 + s * (t - x), { color: T.red, width: 1.6, dash: [5, 4] });
              const nx = x - st.lr * s;
              P.arrow(x, y0, nx, f(nx), { color: T.green, width: 2 });
              P.dots([[x, y0]], { r: 6, color: T.red, stroke: true });
              P.dots([[nx, f(nx)]], { r: 5, color: T.green, stroke: true });
            });
            const s = df(x);
            out({ x: x.toFixed(2), fx: f(x).toFixed(3), d: s.toFixed(3), nx: (x - st.lr * s).toFixed(3) });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          if (e.down) { x = Math.max(-3.1, Math.min(3.1, P.ix(e.x))); S.redraw(); }
        });
        Viz.buttons(host, [
          { label: 'Take one step', primary: true, on: () => { const [f, df] = F[st.f]; x = x - st.lr * df(x); S.redraw(); } },
          { label: 'Run 30 steps', on: () => { const [f, df] = F[st.f]; let i = 0; const t = setInterval(() => { x = x - st.lr * df(x); S.redraw(); if (++i > 30) clearInterval(t); }, 60); ML.onCleanup(() => clearInterval(t)); } },
          { label: 'Reset', on: () => { x = -1.6; S.redraw(); } }
        ]);
      },

      chain: function (host) {
        const st = Viz.controls(host, [
          { k: 'x', label: 'input x', min: -2, max: 2, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'w', label: 'weight w', min: -2, max: 2, step: .1, value: .5, fmt: v => v.toFixed(1) },
          { k: 'b', label: 'bias b', min: -2, max: 2, step: .1, value: 0, fmt: v => v.toFixed(1) },
          { k: 'y', label: 'target y', min: -2, max: 3, step: .1, value: 1, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const z = st.w * st.x + st.b;
            const a = Math.max(0, z);                 // ReLU
            const L = .5 * (a - st.y) * (a - st.y);
            const dLda = a - st.y, daz = z > 0 ? 1 : 0;
            const dLdz = dLda * daz, dLdw = dLdz * st.x, dLdb = dLdz;
            const nodes = [
              { x: .07, l: 'x', v: st.x, g: null },
              { x: .30, l: 'z = wx + b', v: z, g: dLdz },
              { x: .58, l: 'a = ReLU(z)', v: a, g: dLda },
              { x: .87, l: 'L = ½(a−y)²', v: L, g: 1 }
            ];
            const cy = h * .42, bw = Math.min(126, w * .21), bh = 46;
            ctx.font = '11px ui-sans-serif, system-ui';
            nodes.forEach((n, i) => {
              const cx = n.x * w + bw / 2;
              Labs.roundRect(ctx, cx - bw / 2, cy - bh / 2, bw, bh, 9);
              ctx.fillStyle = T.panel; ctx.fill(); ctx.strokeStyle = T.line; ctx.stroke();
              ctx.fillStyle = T.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.font = '11px ui-monospace, monospace';
              ctx.fillText(n.l, cx, cy - 9);
              ctx.fillStyle = T.blue; ctx.font = 'bold 13px ui-monospace, monospace';
              ctx.fillText(n.v.toFixed(3), cx, cy + 11);
              if (i < nodes.length - 1) {
                const nx = nodes[i + 1].x * w + bw / 2;
                ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6;
                ctx.beginPath(); ctx.moveTo(cx + bw / 2, cy - 8); ctx.lineTo(nx - bw / 2 - 4, cy - 8); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(nx - bw / 2 - 4, cy - 8); ctx.lineTo(nx - bw / 2 - 10, cy - 12); ctx.lineTo(nx - bw / 2 - 10, cy - 4); ctx.closePath(); ctx.fillStyle = T.blue; ctx.fill();
                ctx.strokeStyle = T.red; ctx.lineWidth = 1.6;
                ctx.beginPath(); ctx.moveTo(nx - bw / 2 - 4, cy + 10); ctx.lineTo(cx + bw / 2, cy + 10); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(cx + bw / 2, cy + 10); ctx.lineTo(cx + bw / 2 + 6, cy + 6); ctx.lineTo(cx + bw / 2 + 6, cy + 14); ctx.closePath(); ctx.fillStyle = T.red; ctx.fill();
              }
              if (n.g !== null) {
                ctx.fillStyle = T.red; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
                ctx.fillText('∂L/∂· = ' + n.g.toFixed(3), cx, cy + bh / 2 + 16);
              }
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('forward →', 10, 10);
            ctx.fillStyle = T.red; ctx.fillText('← backward', 10, 28);
            ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('∂L/∂w = ∂L/∂z · x = ' + dLdw.toFixed(3) + '     ∂L/∂b = ' + dLdb.toFixed(3), w / 2, h - 26);
          }
        });
      }
    },
    quiz: [
      {
        q: 'Gradient descent updates $\\theta \\leftarrow \\theta - \\eta\\nabla L$. Why the minus sign?',
        options: ['To keep parameters positive', 'The gradient points uphill; we want to go downhill', 'It cancels the learning rate', 'Convention only — plus works too'],
        answer: 1,
        why: 'The gradient is the direction of steepest <i>increase</i>, which §0.3.3 derives from the dot product rather than asserting: a small step $u$ changes the loss by $\\nabla L \\cdot u = \\|\\nabla L\\|\\cos\\theta$, and that is largest when $u$ points along the gradient itself. Since you want the loss to fall, you walk the other way, and the minus sign is how that gets written. Option D is the tempting one because signs often are conventional — but here it is not. Flip it to a plus and you have gradient <i>ascent</i>, which climbs the loss surface and diverges, and that mistake is easy to make and unpleasant to diagnose because the code runs perfectly and the loss simply goes up for ever.'
      },
      {
        q: 'A 40-layer network trains with sigmoid activations and the early layers barely move. The most direct explanation is…',
        options: ['The learning rate is too large', 'Each backward step multiplies by σ′ ≤ 0.25, so the product over depth collapses', 'The loss is non-convex', 'Batch size is too small'],
        answer: 1,
        why: 'This is the vanishing gradient, and it follows directly from the boxed sentence in §0.3.5: every gradient is a <i>product</i> of local derivatives along the path back from the loss. The sigmoid\'s derivative $\\sigma(1-\\sigma)$ peaks at 0.25 and is usually smaller, so forty layers attenuate by at most $0.25^{40} \\approx 8\\times10^{-25}$ — the early layers receive an instruction indistinguishable from zero and never move. Option A is the tempting answer because a bad learning rate is the usual first suspect, but too large a rate makes the loss <i>diverge</i> rather than making one end of the network freeze while the other trains normally; the depth-dependent pattern is the clue. The fixes all attack the same product: ReLU contributes a factor of exactly 1 on its active side, residual connections add a path whose local derivative is exactly 1 (§3.7), and normalisation keeps activations where the local derivatives stay near 1 (§3.6).'
      },
      {
        q: 'You are minimising $f(x) = x^2$ with gradient descent. For which learning rates $\\eta$ does the procedure converge?',
        options: ['Any $\\eta > 0$', 'Only $\\eta < 1$', 'Only $\\eta < 0.5$', 'Any $\\eta$, provided you take enough steps'],
        answer: 1,
        why: 'Work the update out rather than guessing. With $f\' = 2x$ the rule $x \\leftarrow x - \\eta\\,2x$ multiplies $x$ by $(1 - 2\\eta)$ every step, so the iterates shrink exactly when $|1-2\\eta| < 1$, which means $0 < \\eta < 1$. At $\\eta = 1$ the factor is $-1$ and the point oscillates for ever between $+x$ and $-x$; above 1 it grows without bound. Options A and D are tempting because more steps usually help — but no number of steps rescues a procedure whose every step makes things worse, and the pitfall in §0.3.2 traces $3 \\to -3.6 \\to 4.32 \\to -5.184$ to show it happening. The general statement is the one to carry away: the largest stable step is 2 divided by the curvature, which for $x^2$ is 2, giving the bound 1. In a real model the curvature differs by direction, the steepest one sets the limit for everything, and that is exactly what the condition number in §0.3.4 measures.'
      }
    ],
    cards: [
      { q: 'What a derivative is', a: 'The rate of change of a function at one point — the slope of the best straight-line approximation there, and the thing that tells an optimiser which way is downhill.' },
      { q: 'Chain rule, and why it matters here', a: '$\\frac{d}{dx}f(g(x)) = f\'(g(x))g\'(x)$ — rates multiply. Backpropagation is this product accumulated right-to-left with cached activations.' },
      { q: 'Why reverse-mode autodiff?', a: 'One scalar output, many parameters. Reverse mode costs ~one forward pass per output; forward mode costs one per input.' },
      { q: 'Gradient vs Hessian', a: 'Gradient = direction of steepest ascent (first derivatives). Hessian = curvature (second derivatives); its condition number governs how hard optimisation is.' },
      { q: 'Largest stable learning rate', a: 'About $2 \\div$ curvature. The steepest direction sets the limit, so an ill-conditioned loss forces a tiny step on every direction.' },
      { q: 'Why gradients vanish or explode', a: 'They are products of per-layer local derivatives. Factors below 1 collapse geometrically with depth; factors above 1 blow up.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.4 */
  ML.section({
    id: 'probability-basics', track: 'start', num: '0.4',
    title: 'Probability, from the ground',
    lede: 'Joint, marginal and conditional are not three formulas. They are three ways of reading one table, and every one of them is arithmetic you can do by counting rows. Get them straight here and Bayes in §1.1 becomes a one-line consequence rather than something to memorise, while expectation and variance turn into the vocabulary that §1.5 uses to explain where losses come from.',
    html: `
<p>You are asked whether a particular loan applicant will default. You cannot answer yes and you cannot answer no, because you genuinely do not know. But it would be quite wrong to say you know nothing: you have a filing cabinet holding last year's thousand applications with the outcome written on each one, and this applicant looks like some of them more than others.</p>

<p>Ordinary arithmetic has no way to write down that state of knowledge. It can hold a number, and it can hold a yes or a no, but it has no notation for "probably not, and here is how strongly". Probability is the arithmetic that does, and — this is the part worth believing early — almost all of it is counting rows in a table and then dividing.</p>

<h2><span class="sn">0.4.1</span> One table, and everything that comes out of it</h2>

<p>Open the filing cabinet. For each of the 1,000 applications, record just two facts: whether the applicant had a previous default on record, and whether they defaulted this time. Two yes-or-no facts give four possible combinations, so the whole cabinet collapses into four numbers:</p>

${H.table(['out of 1,000 applicants', 'defaulted this time', 'repaid', 'row total'], [
      ['<b>had a prior default</b>', '90', '110', '<b>200</b>'],
      ['<b>no prior default</b>', '60', '740', '<b>800</b>'],
      ['<b>column total</b>', '<b>150</b>', '<b>850</b>', '<b>1,000</b>']
    ])}

<p>Every probability in this section is one of those numbers divided by another. Write $D$ for the event "defaulted this time" and $R$ for the event "had a prior default on record"; the letter $P$ in front of an event is read "the probability of", so $P(D)$ is read "the probability of D".</p>

<p><b>The joint.</b> $P(R, D)$, read "the probability of R and D", is the probability that both things are true of the same applicant. That is the top-left cell over the grand total: $90/1000 = 0.09$. The four joint probabilities are $0.09$, $0.11$, $0.06$ and $0.74$, and they sum to exactly 1 because every applicant falls into precisely one cell. The joint table is the complete description of the situation; nothing else in this section adds information to it, they only summarise it differently.</p>

<p><b>The marginal.</b> Suppose you no longer care about prior defaults and only want the overall default rate. Add up the column: $90 + 60 = 150$, so $P(D) = 150/1000 = 0.15$. In symbols,</p>

$$P(D) = \\sum_{r} P(r, D)$$

<p>where $\\sum$ is the Greek capital sigma met in §0.2 and means "add up what follows", and the $r$ underneath says to run through every possible value of the prior-default variable — here just the two, yes and no. Summing a variable away like this is called <b>marginalising it out</b>, and the name is not a metaphor: these totals were traditionally written in the margins of the table, which is exactly where they appear in the table above.</p>

<p><b>The conditional.</b> Now the question that actually matters. You have learned that this particular applicant <i>does</i> have a prior default. What is the probability they default now?</p>

<p>Do it by hand and the formula will never need memorising. The information rules out 800 of the 1,000 rows, so throw them away; you are left with the top row and its 200 applicants. Of those, 90 defaulted. So the answer is $90/200 = 0.45$. Written as a formula, dividing top and bottom by 1,000 to turn the counts into probabilities:</p>

$$P(D \\mid R) = \\frac{P(R, D)}{P(R)} = \\frac{0.09}{0.20} = 0.45$$

<p>The vertical bar is read "given", so $P(D \\mid R)$ is read aloud as "the probability of D given R". The denominator's whole job is to restore the total to 1 after you have thrown rows away: the surviving cells were $0.09$ and $0.11$, which add to $0.20$ rather than 1, and dividing both by $0.20$ turns them into $0.45$ and $0.55$, a proper distribution over a smaller world.</p>

${H.key('A conditional probability is not a new quantity. It is the same table with the impossible rows deleted, rescaled so what remains sums to one again.')}

${H.analogy(`<p>Conditioning is cropping a photograph. You cut away the part of the frame that the evidence has ruled out, and then you enlarge what is left so that it fills the frame again.</p>
<p>Both halves matter. The cropping is what makes the answer specific to your evidence. The enlarging is what keeps it a probability — a photograph that filled only a fifth of the frame would not be a photograph, and a set of weights adding to 0.20 is not a distribution. When you see the division by $P(B)$ in the formula, that is the enlargement, nothing more.</p>`)}

<h3>The two conditionals are different numbers, and confusing them is the classic error</h3>

<p>Turn the same table round. Among the 150 people who defaulted, how many had a prior default on record? That is $90/150 = 0.60$, so $P(R \\mid D) = 0.60$.</p>

<p>So $P(D \\mid R) = 0.45$ and $P(R \\mid D) = 0.60$. Same table, same 90 in the numerator, entirely different denominators, and entirely different meanings. The first says "given a prior default, defaulting is slightly less likely than not". The second says "most people who default had form". Read the second and conclude the first and you have just made the most expensive mistake in applied probability.</p>

<p>It gets worse when the two base rates are far apart, which is the usual case for rare events. §1.1 is devoted to that gap and to the formula that converts between the two conditionals safely. For now, the habit to build is simply this: whenever you see a conditional probability, say out loud which quantity is being held fixed, because that is the denominator and the denominator is the whole answer.</p>

<h2><span class="sn">0.4.2</span> Independence: when the evidence changes nothing</h2>

<p>In the table above, learning about a prior default moved the default probability from $0.15$ to $0.45$ — a threefold change, which is why the feature is worth collecting. Sometimes learning something moves nothing at all, and that special case has a name.</p>

<p>Two events are <b>independent</b> when</p>

$$P(A, B) = P(A)\\,P(B), \\qquad \\text{equivalently} \\qquad P(A \\mid B) = P(A)$$

<p>The two statements say the same thing; substitute the first into the definition of the conditional and the $P(B)$ cancels. The second form is the one to think in: knowing $B$ leaves your belief about $A$ exactly where it was.</p>

<p>Check the loan table. If prior default and current default were independent, the top-left cell would have to be $P(R) \\times P(D) = 0.20 \\times 0.15 = 0.03$, so 30 applicants. The actual number is 90. They are three times as likely to co-occur as independence would allow, so they are strongly dependent — which is the whole reason the feature earns a place in the model.</p>

${H.intuition(`<p>Independence is what makes large probability calculations possible at all, and it is worth seeing the size of the effect.</p>
<p>A joint distribution over 20 yes-or-no variables has $2^{20}$ cells, which is 1,048,576 numbers you would have to estimate from data. If the variables are independent, the whole thing is determined by 20 numbers, one per variable, because every cell is just a product of them. That is a compression of fifty thousand to one, bought entirely with an assumption.</p>
<p>This is why independence assumptions are everywhere in machine learning even though they are almost always false. Naive Bayes (§2.5) assumes every feature is independent of every other given the label, which is plainly untrue of real features, and it remains a serviceable classifier anyway — for reasons that section makes precise. The pattern to internalise is that an independence assumption is a deliberate trade: you knowingly accept a wrong model in exchange for one you can actually estimate.</p>`)}

${H.pitfall(`<p>Independence is a much stronger claim than "these look unrelated", and two specific confusions cause real damage.</p>
<p><b>Uncorrelated is not independent.</b> Let $X$ take the values $-2, -1, 0, 1, 2$ with equal probability and let $Y = X^2$. Then $Y$ is completely determined by $X$, so they could hardly be more dependent — yet their correlation is exactly zero. Check it: the covariance is $\\mathbb{E}[XY] - \\mathbb{E}[X]\\mathbb{E}[Y]$, and here $\\mathbb{E}[X] = 0$ by symmetry while $\\mathbb{E}[XY] = \\mathbb{E}[X^3] = (-8 - 1 + 0 + 1 + 8)/5 = 0$, so both terms vanish. Correlation detects straight-line relationships only, and a symmetric parabola is invisible to it (§1.3).</p>
<p><b>Assumed independence inflates your confidence.</b> Suppose three weak models each get 55 per cent of cases right. If their errors are genuinely independent, a majority vote is right with probability $3(0.55)^2(0.45) + (0.55)^3 = 0.575$ — a useful gain. If instead they are all wrong on the same hard cases, the vote is right 55 per cent of the time and you have gained nothing while paying for three models. Real ensembles land somewhere between, and the whole art of §2.16 is engineering the errors to be as independent as possible.</p>`)}

<h3>Seeing all three views move together</h3>

<p><b>What you are looking at.</b> The four coloured squares are the joint distribution over two yes-or-no variables, $A$ and $B$: the top row is $A=1$ in blue, the bottom row is $A=2$ in red, the left column is $B=1$ and the right column is $B=2$. The number printed in each square is that cell's joint probability, and the colour deepens as the probability grows. Down the right-hand side are the row totals, which are the marginals $P(A=1)$ and $P(A=2)$; underneath the grid are the column totals, the marginals of $B$. The horizontal bar below everything is the conditional distribution of $A$ given whichever value of $B$ you have selected — it is the corresponding column of the grid, stretched to fill the full width, which is the cropping-and-enlarging of the analogy above drawn literally.</p>

<p><b>What to do with it.</b> The four sliders set the four cell values, and they are renormalised to sum to 1 before anything is drawn, so moving one slider changes all four displayed probabilities. Start by pushing the top-left cell up and watch two things at once: the row marginal on the right grows, and the conditional bar shifts blue-wards. Then use the <i>condition on</i> buttons to switch the evidence between $B=1$ and $B=2$ and watch the bar jump — that jump is the entire content of the phrase "the evidence is informative".</p>

<p><b>The thing genuinely worth noticing.</b> Press <i>Strong dependence</i>. The marginal $P(A=1)$ sits at $0.500$, but the conditional $P(A=1 \\mid B=1)$ reads $0.900$: before the evidence it was a coin flip, after the evidence it is nearly certain. Now press <i>Make them independent</i>. The marginal is $0.450$ and the conditional reads $0.455$, which is the same number up to the two decimal places the sliders can hold, and the <i>independent?</i> readout flips to <i>yes</i>. That is the definition made visible — under independence the conditional and the marginal coincide, so conditioning on $B$ is a null operation and the feature carries no information about $A$ whatsoever. A feature that does this to your label is a feature you can delete.</p>

${H.lab('joint', 'One table, three views', 'The four sliders set the joint probabilities; they are rescaled to sum to 1 before drawing. Marginals appear on the edges. The bar below is the conditional — the selected column of the table, stretched back out to full width. Independence is the case where that bar matches the marginal exactly.')}

<h2><span class="sn">0.4.3</span> From events to quantities: random variables and distributions</h2>

<p>So far everything has been a yes-or-no event. Most of the time you want a number instead: how many of the next three loans default, how much a customer spends, what the next token is. A <b>random variable</b> is a quantity whose value is not yet determined, written with a capital letter, usually $X$. Its <b>distribution</b> is the complete list of values it might take together with the weight attached to each.</p>

<p>The entire rulebook is two lines. Every weight must be zero or greater, because a negative probability means nothing. And the weights must total exactly 1, because the variable is certain to take one of its values. That is all the probability axiomatics this course needs from you.</p>

${H.worked('a distribution built from scratch', `<p>Three loans are about to be decided. Suppose each defaults with probability $0.15$, independently of the others, and let $X$ be the number that default. $X$ can be 0, 1, 2 or 3, so its distribution has four numbers in it.</p>
<p>$P(X=0) = 0.85^3 = 0.614125$ — all three repay.</p>
<p>$P(X=1) = 3 \\times 0.15 \\times 0.85^2 = 0.325125$ — the 3 is there because the defaulter could be any one of the three.</p>
<p>$P(X=2) = 3 \\times 0.15^2 \\times 0.85 = 0.057375$, and $P(X=3) = 0.15^3 = 0.003375$.</p>
<p>Add them: $0.614125 + 0.325125 + 0.057375 + 0.003375 = 1.000000$ exactly. That check is worth doing every time you build a distribution by hand, because it catches almost every arithmetic slip.</p>
<p>Notice where independence was used: it is what allowed the probabilities to be multiplied together in each line. Without it, none of these products would be legitimate.</p>`)}

<p>That particular shape has a name, the <b>binomial distribution</b>, and §1.2 catalogues it alongside the other distributions worth recognising on sight. What matters here is the shape of the object: a list of values and a list of weights, adding to one.</p>

<h2><span class="sn">0.4.4</span> Expectation: the one number that summarises a distribution</h2>

<p>Four numbers is already more than you want to carry around, so the natural next question is how to compress a distribution into a single representative value. The obvious idea — average the possible values — is wrong, because it treats a wildly unlikely outcome as equal to a near-certain one. The fix is to weight each value by how likely it is:</p>

$$\\mathbb{E}[X] = \\sum_x x \\, p(x)$$

<p>The symbol $\\mathbb{E}$ is a decorated capital E, and $\\mathbb{E}[X]$ is read "the expectation of X" or "the expected value of X". The formula says: go through every value $X$ might take, multiply it by its probability, and total the results. It is a weighted average in which the weights are the probabilities.</p>

<p>On the three-loan distribution:</p>

$$\\mathbb{E}[X] = 0(0.614125) + 1(0.325125) + 2(0.057375) + 3(0.003375) = 0.45$$

<p>which is exactly $3 \\times 0.15$. That is not a coincidence and it is worth noticing early: the expected number of defaults among three loans is three times the expected number among one, even though the loans interact in the formula in complicated ways. Expectation simply adds, always, whether or not the variables are independent, and §1.3 shows how much mileage that one fact gives you.</p>

${H.pitfall(`<p>The expectation need not be a value the variable can ever take. You will never observe 0.45 defaults; $X$ is a count, so it is always 0, 1, 2 or 3. "Expected" is a technical term meaning the long-run average, not a prediction of what will happen next.</p>
<p>This bites in practice whenever a model trained on squared error is asked for a decision rather than a number. Squared error drives the model towards the conditional expectation, and the expectation of a lumpy or skewed target can sit in a valley where no real outcome lives — the average of a distribution that is mostly zero with an occasional 1,000 is neither zero nor 1,000. §1.5 explains why squared error targets the mean specifically, and §2.13 explains how to pick a metric that reflects the decision you actually face.</p>`)}

<h2><span class="sn">0.4.5</span> Variance: how much the outcome moves about</h2>

<p>Two distributions can share an expectation and be nothing alike. A payout that is always £5 and a payout that is £0 or £10 on a coin flip both have expectation £5, and no one would call them equivalent. What separates them is spread, and the standard measure of spread is the average squared distance from the expectation:</p>

$$\\mathrm{Var}(X) = \\mathbb{E}\\big[(X - \\mathbb{E}[X])^2\\big]$$

<p>In words: take how far each outcome falls from the centre, square it so that overshoots and undershoots both count as spread rather than cancelling, and average those squares with the probabilities as weights. The squaring is the same device that appeared in the squared-error loss of §0.1 and for the same two reasons — it removes the sign, and it penalises large departures disproportionately.</p>

<p>Computing it straight from that definition means finding the mean first and then sweeping the distribution a second time. There is a rearrangement that lets you do it in one pass, and it is used constantly:</p>

${H.deriv('the computational form $\\mathrm{Var}(X) = \\mathbb{E}[X^2] - \\mathbb{E}[X]^2$', [
      ['$\\mathrm{Var}(X) = \\mathbb{E}[(X - \\mu)^2]$', 'The definition, writing $\\mu$ (the Greek letter mu) for $\\mathbb{E}[X]$ to keep the algebra readable. Note that $\\mu$ is an ordinary fixed number, not a random one.'],
      ['$= \\mathbb{E}[X^2 - 2\\mu X + \\mu^2]$', 'Expand the square inside, exactly as you would expand $(a-b)^2$ in school algebra.'],
      ['$= \\mathbb{E}[X^2] - 2\\mu\\,\\mathbb{E}[X] + \\mu^2$', 'Split the expectation across the sum. This is allowed because expectation is a weighted sum and sums can be regrouped; constants such as $2\\mu$ and $\\mu^2$ pass straight through because the weights already total 1.'],
      ['$= \\mathbb{E}[X^2] - 2\\mu^2 + \\mu^2$', 'Substitute $\\mathbb{E}[X] = \\mu$ into the middle term.'],
      ['$= \\mathbb{E}[X^2] - \\mu^2$', 'Combine the last two terms: $-2\\mu^2 + \\mu^2 = -\\mu^2$. So the variance is the mean of the squares minus the square of the mean.']
    ], 'Check it on the three-loan distribution. $\\mathbb{E}[X^2] = 0 + 1(0.325125) + 4(0.057375) + 9(0.003375) = 0.585$, and $\\mu^2 = 0.45^2 = 0.2025$, so the variance is $0.585 - 0.2025 = 0.3825$. The textbook formula for a binomial gives $np(1-p) = 3 \\times 0.15 \\times 0.85 = 0.3825$ — the same number. One warning: this form is elegant on paper and treacherous in floating-point, because when the mean is large the two terms are nearly equal and subtracting them destroys most of the significant digits (§1.15).')}

<p>Variance is measured in the square of whatever units $X$ has, which makes it awkward to talk about — the variance of a height in metres is in square metres. Its square root, the <b>standard deviation</b>, written $\\sigma$ (the Greek letter sigma), is back in the original units and is the number to quote. Here $\\sigma = \\sqrt{0.3825} \\approx 0.62$ defaults.</p>

${H.intuition(`<p>The single most useful consequence of variance in all of applied work is what happens when you average.</p>
<p>If you take $n$ independent measurements each with standard deviation $\\sigma$ and average them, the average has standard deviation $\\sigma/\\sqrt{n}$. The square root is the entire story. Averaging 4 measurements halves the noise. Getting another factor of two costs 16 measurements, then 64, then 256.</p>
<p>That one expression sets the price of nearly everything you will want to be sure about. It is why an A/B test needs four times the traffic to detect an effect half the size (§2.25), why a validation set of 200 rows gives an accuracy estimate you should not trust to the second decimal place (§1.6), and why increasing the batch size gives steadily diminishing returns on gradient quality (§3.5). The <i>reason</i> the square root appears is that variances add when you sum independent quantities while standard deviations do not — and that asymmetry, expectation adding always but variance adding only under independence, is the subject of §1.3.</p>`)}

<h2><span class="sn">0.4.6</span> Mass and density: the distinction that causes real trouble</h2>

<p>Everything so far assumed a variable with a countable list of possible values. Now ask a question about a continuous one. What is the probability that an adult chosen at random is exactly 1.8 metres tall — not 1.80001, not 1.79999, but exactly 1.800000... with zeros for ever?</p>

<p>The answer is zero. There are infinitely many possible heights, and if any single one carried a positive probability then adding up enough of them would take the total past 1. So every individual height has probability exactly zero, and yet heights near 1.8 metres are obviously more common than heights near 2.4 metres. Both statements are true, and they need reconciling.</p>

<p>The reconciliation is to stop asking about points and start asking about intervals. The probability that a height falls between 1.79 and 1.81 metres is a perfectly sensible positive number. What describes how that probability is spread out is a <b>density</b>, written $p(x)$, and the rule connecting it to probabilities is</p>

$$P(a \\le X \\le b) = \\int_a^b p(x)\\,dx$$

<p>The elongated S is an <b>integral</b> sign, and the expression is read "the integral of p of x, dx, from a to b". It means the area under the curve $p$ between $a$ and $b$. If that sounds like an entirely new piece of machinery, notice it is built the same way the derivative was in §0.3: chop the interval into thin strips, add up the little rectangles, and let the strips get thinner. Area is what a sum of shrinking slices converges to, just as a slope was what a shrinking rise-over-run converged to.</p>

${H.key('A density is not a probability. It is probability per unit of $x$, and only its area over an interval is a probability.')}

<p>The clearest way to feel the difference is to change the units and watch what happens to each.</p>

${H.worked('the same bus, two sets of units', `<p>A bus arrives at a uniformly random moment within a 10-minute window. Its density, measured in minutes, is flat at $1/10 = 0.1$ per minute across the window.</p>
<p>The probability it arrives during a particular 2-minute stretch is the density times the width: $0.1 \\times 2 = 0.2$. The probability it arrives at exactly 4 minutes past is $0.1 \\times 0 = 0$, as it must be.</p>
<p>Now measure the same bus in hours. The window is $1/6$ of an hour, so the density is $1/(1/6) = 6$ per hour. The density value has gone from $0.1$ to $6$ while absolutely nothing about the bus has changed.</p>
<p>The probability, however, has not moved. The 2-minute stretch is $1/30$ of an hour, and $6 \\times 1/30 = 0.2$, exactly as before. Probabilities are pure numbers; densities carry units of "per something", and their numerical value depends entirely on what that something is.</p>`)}

<p>Two consequences follow immediately, and both routinely alarm people who have not seen this.</p>

<p><b>A density can exceed 1.</b> A uniform distribution on the interval $[0, 0.5]$ must have density 2 everywhere on that interval, because the area of a rectangle of width $0.5$ and height 2 is exactly 1. Nothing is wrong. Similarly a Gaussian bell curve of standard deviation $\\sigma$ has peak height $1/(\\sigma\\sqrt{2\\pi})$, which for $\\sigma = 0.1$ is about $3.99$ and for $\\sigma = 0.01$ is about $39.9$. Squeeze a distribution narrower and its density must rise to keep the area at 1.</p>

<p><b>A log-likelihood of continuous data can be positive.</b> If the density at your observed point is 4, its logarithm is $+1.39$, and a model reporting a positive log-likelihood — equivalently a negative loss — has not gone wrong. This surprises people every time it appears in a density model such as a normalising flow or a variational autoencoder (§6.3), and it is entirely explained by the paragraph above.</p>

${H.table(['', 'Discrete: probability <b>mass</b> $p(x)$', 'Continuous: probability <b>density</b> $p(x)$'], [
      ['What $p(x)$ is', 'The probability that $X$ equals $x$', 'Probability per unit of $x$ near $x$'],
      ['Can it exceed 1?', 'No, never', 'Yes, freely'],
      ['$P(X = x)$', 'Equal to $p(x)$', 'Exactly 0, for every single $x$'],
      ['Total is 1 by', '$\\sum_x p(x) = 1$', '$\\int p(x)\\,dx = 1$'],
      ['Changing units', 'Nothing changes', 'Every value rescales'],
      ['Typical use here', 'Class probabilities, token distributions', 'Gaussian noise models, latent variables']
    ])}

${H.more('what a probability actually means, and why anyone argues about it', `<p>Nothing above says what a probability <i>is</i>, only how the numbers behave. That is deliberate: the rules of arithmetic are agreed, and the interpretation is not.</p>
<p>One reading is <b>frequentist</b>: a probability is the long-run fraction of times an event occurs if the situation is repeated indefinitely. It is concrete and it fits coin flips and loan books well. It struggles with one-off questions, because "the probability that this particular bridge fails next year" has no repetitions to count.</p>
<p>The other reading is <b>Bayesian</b>: a probability is a degree of belief, which may be updated as evidence arrives. It handles one-off questions comfortably and pays for it by requiring you to state a prior belief before seeing any data, which is a genuine commitment people reasonably disagree about.</p>
<p>The formal framework beneath both was settled by Kolmogorov in 1933, who took non-negativity, total mass 1 and additivity as axioms and derived everything else, which is why the two camps never disagree about a calculation — only about what the answer means. §1.14 develops the Bayesian machinery properly, and §1.6 shows the two readings giving genuinely different answers to what looks like the same question about a confidence interval.</p>`)}

<h2><span class="sn">0.4.7</span> Where this is going</h2>

<p>Two payments come due almost immediately, and both are worth previewing so you can see that this section was not bookkeeping.</p>

<p><b>Bayes' rule is four lines of algebra from the definition of a conditional.</b> Nothing new is required — only the observation that a joint probability can be factored in either order.</p>

${H.deriv("Bayes' rule, straight from §0.4.1", [
      ['$P(A \\mid B) = \\dfrac{P(A, B)}{P(B)}$', 'The definition of conditional probability, exactly as derived by deleting rows and rescaling.'],
      ['$P(A, B) = P(A \\mid B)\\,P(B)$', 'Multiply both sides by $P(B)$. This is the same statement rearranged, and it says a joint probability is a conditional times the thing conditioned on.'],
      ['$P(A, B) = P(B \\mid A)\\,P(A)$', 'Apply the identical rearrangement with the roles of $A$ and $B$ swapped. The joint does not care about the order you write it in, so both factorisations describe the same number.'],
      ['$P(A \\mid B)\\,P(B) = P(B \\mid A)\\,P(A)$', 'Lines 2 and 3 are two expressions for the same joint probability, so they may be set equal.'],
      ['$P(A \\mid B) = \\dfrac{P(B \\mid A)\\,P(A)}{P(B)}$', 'Divide through by $P(B)$, which is legitimate whenever $P(B) > 0$ — and if $P(B)$ were zero you would not be conditioning on it.']
    ], 'That is the whole derivation, and it is the answer to the question raised in §0.4.1 about the two different conditionals. Check it on the loan table: $P(D \\mid R) = P(R \\mid D)P(D)/P(R) = (0.60 \\times 0.15)/0.20 = 0.45$, which is the number we counted directly. §1.1 does not derive anything new; it explains why the formula is so persistently counter-intuitive when $P(A)$ is small.')}

<p><b>Losses come from probability.</b> §0.1 introduced squared error as a reasonable-looking way to score a numerical prediction and flagged that the choice was not obvious. The resolution is that a loss is a statement about how you believe the noise in your data behaves: assume Gaussian noise and maximising the likelihood of your data is <i>identical</i> to minimising squared error, while assuming a yes-or-no outcome gives cross-entropy instead. §1.5 makes both of those exact. Until then, hold the idea that every loss you meet is a probabilistic assumption in disguise.</p>

${H.practice(`<p>Three things about probability that only bite once you are writing code.</p>
<p><b>Work in logs.</b> A joint probability over 500 tokens is a product of 500 numbers below 1, which underflows to exactly zero in floating point long before you reach the end. Adding logarithms instead keeps everything in a comfortable range and turns the products into sums. Every library therefore hands you <code>log_softmax</code> and <code>logsumexp</code> rather than the raw quantities (§1.10, §1.15).</p>
<p><b>A number between 0 and 1 is not automatically a probability.</b> A classifier's output is a probability only if it is calibrated — if, among all the cases it scored 0.7, roughly 70 per cent really were positive. Modern networks are frequently and badly overconfident, and the fix is a separate step rather than a better model (§2.12).</p>
<p><b>Renormalise after masking.</b> The moment you zero out some options — banned tokens, unavailable products, invalid actions — the remaining weights no longer sum to 1, and every subsequent expectation is silently wrong. Conditioning is deletion <i>and</i> rescaling, and code that forgets the second half of that sentence produces plausible numbers that are quietly incorrect.</p>`)}

${H.probe([
      ['Define conditional probability without writing the formula.', 'Restrict attention to the world in which the evidence is true, discard everything incompatible with it, then rescale what is left so it sums to one again.'],
      ['Can a probability density be greater than 1?', 'Yes. A density is probability per unit of $x$, not a probability. A uniform on $[0, 0.5]$ has density 2 everywhere on its support, and a Gaussian with $\\sigma = 0.1$ peaks near 4.'],
      ['What is $P(X = x)$ for a continuous variable?', 'Exactly zero, for every $x$. Only intervals carry probability, and their probability is the area under the density.'],
      ['State independence in two equivalent ways, and say which is more useful.', '$P(A,B) = P(A)P(B)$, and $P(A \\mid B) = P(A)$. The second is more useful because it says what independence <i>means</i>: the evidence leaves your belief unchanged.'],
      ['Why does averaging $n$ measurements reduce noise by $\\sqrt{n}$ rather than $n$?', 'Variances of independent quantities add, so the variance of a mean falls as $1/n$; the standard deviation is its square root and therefore falls as $1/\\sqrt{n}$.']
    ], 'Confusing $P(A\\mid B)$ with $P(B\\mid A)$. In the loan table above they are 0.45 and 0.60 — same numerator, different denominators, different meanings. That single confusion is the base-rate error of §1.1 and the most common probability mistake in interviews.')}`,
    labs: {
      joint: function (host) {
        // joint over A in {a1,a2}, B in {b1,b2}
        let J = [[.30, .12], [.18, .40]];
        const st = Viz.controls(host, [
          { k: 'p11', label: 'P(A=1, B=1)', min: .01, max: .8, step: .01, value: .30, fmt: v => v.toFixed(2) },
          { k: 'p12', label: 'P(A=1, B=2)', min: .01, max: .8, step: .01, value: .12, fmt: v => v.toFixed(2) },
          { k: 'p21', label: 'P(A=2, B=1)', min: .01, max: .8, step: .01, value: .18, fmt: v => v.toFixed(2) },
          { k: 'p22', label: 'P(A=2, B=2)', min: .01, max: .8, step: .01, value: .40, fmt: v => v.toFixed(2) },
          { k: 'cond', label: 'condition on', type: 'buttons', value: 'B1', options: [{ v: 'B1', t: 'B = 1' }, { v: 'B2', t: 'B = 2' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'pa', label: 'P(A=1)' }, { k: 'pb', label: 'P(B=1)' },
          { k: 'cond', label: 'P(A=1 | evidence)', cls: 'key' }, { k: 'ind', label: 'independent?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const raw = [[st.p11, st.p12], [st.p21, st.p22]];
            const tot = raw[0][0] + raw[0][1] + raw[1][0] + raw[1][1];
            J = raw.map(r => r.map(v => v / tot));
            const gx = 60, gy = 46, cell = Math.min(96, (w - 260) / 2);
            const cols = [T.blue, T.red];
            ctx.font = '11px ui-monospace, monospace';
            for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
              const x = gx + j * cell, y = gy + i * cell;
              const p = J[i][j];
              ctx.fillStyle = i === 0 ? 'rgba(90,120,230,' + (0.15 + p) + ')' : 'rgba(220,90,80,' + (0.15 + p) + ')';
              ctx.fillRect(x, y, cell - 4, cell - 4);
              ctx.strokeStyle = T.line; ctx.strokeRect(x, y, cell - 4, cell - 4);
              ctx.fillStyle = T.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.font = 'bold 14px ui-monospace, monospace';
              ctx.fillText(p.toFixed(3), x + cell / 2 - 2, y + cell / 2 - 2);
            }
            ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = T.muted;
            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            ctx.fillText('A = 1', gx - 8, gy + cell / 2); ctx.fillText('A = 2', gx - 8, gy + cell * 1.5);
            ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText('B = 1', gx + cell / 2, gy - 8); ctx.fillText('B = 2', gx + cell * 1.5, gy - 8);
            // marginals
            const pa = [J[0][0] + J[0][1], J[1][0] + J[1][1]];
            const pb = [J[0][0] + J[1][0], J[0][1] + J[1][1]];
            ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = T.text;
            pa.forEach((v, i) => ctx.fillText('P(A=' + (i + 1) + ') = ' + v.toFixed(3), gx + 2 * cell + 6, gy + cell * (i + .5)));
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            pb.forEach((v, j) => ctx.fillText(v.toFixed(3), gx + cell * (j + .5), gy + 2 * cell + 4));
            // conditional strip
            const jb = st.cond === 'B1' ? 0 : 1;
            const denom = pb[jb];
            const cx = gx, cy = gy + 2 * cell + 34;
            const stripW = Math.min(2 * cell - 4, w - gx - 20);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('conditional on ' + (jb ? 'B = 2' : 'B = 1') + ' — that column, rescaled to sum to 1', cx, cy - 5);
            const c1 = J[0][jb] / denom;
            ctx.fillStyle = T.blue; ctx.fillRect(cx, cy, stripW * c1, 26);
            ctx.fillStyle = T.red; ctx.fillRect(cx + stripW * c1, cy, stripW * (1 - c1), 26);
            ctx.fillStyle = '#fff'; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            if (c1 > .12) ctx.fillText('A=1 · ' + c1.toFixed(3), cx + stripW * c1 / 2, cy + 13);
            if (1 - c1 > .12) ctx.fillText('A=2 · ' + (1 - c1).toFixed(3), cx + stripW * (c1 + (1 - c1) / 2), cy + 13);
            const indep = Math.abs(J[0][0] - pa[0] * pb[0]) < .004;
            out({ pa: pa[0].toFixed(3), pb: pb[0].toFixed(3), cond: c1.toFixed(3), ind: indep ? 'yes' : 'no' });
          }
        });
        Viz.buttons(host, [
          { label: 'Make them independent', on: () => { const pa = .45, pb = .55; st.$set('p11', +(pa * pb).toFixed(2)); st.$set('p12', +(pa * (1 - pb)).toFixed(2)); st.$set('p21', +((1 - pa) * pb).toFixed(2)); st.$set('p22', +((1 - pa) * (1 - pb)).toFixed(2)); S.redraw(); } },
          { label: 'Strong dependence', on: () => { st.$set('p11', .45); st.$set('p12', .05); st.$set('p21', .05); st.$set('p22', .45); S.redraw(); } }
        ]);
      }
    },
    quiz: [
      {
        q: 'Given the joint table, how do you get the marginal $P(A)$?',
        options: ['Divide by $P(B)$', 'Sum the joint over all values of $B$', 'Multiply by $P(B\\mid A)$', 'Take the maximum over $B$'],
        answer: 1,
        why: 'Marginalising is summing away the variable you no longer care about, which in the table of §0.4.1 was literally adding up a column: $90 + 60 = 150$ defaults out of 1,000. Option A is the tempting distractor because dividing by $P(B)$ is also a legitimate operation on the joint — but that is <i>conditioning</i>, not marginalising, and it answers a different question. Marginalising asks "forget B entirely"; conditioning asks "assume a particular B". One sums the table, the other selects a slice of it and rescales. The sum in this question also has a formal name, the law of total probability, and it is exactly the denominator that appears in Bayes\' rule (§1.1) — which is why a badly estimated marginal quietly corrupts every posterior you compute from it.'
      },
      {
        q: 'Which is true of a continuous probability density $p(x)$?',
        options: ['It is always ≤ 1', 'It integrates to 1 and may exceed 1 pointwise', 'It equals $P(X = x)$', 'It must be symmetric'],
        answer: 1,
        why: 'A density is probability <i>per unit</i> of $x$, so its numerical value depends on the units you chose and can be arbitrarily large: the bus in §0.4.6 has density 0.1 per minute or 6 per hour for exactly the same journey. What is fixed is the area, which must be 1 overall and gives the probability of any interval you pick. Option A is the trap, and it is tempting precisely because it is true for discrete <i>mass</i> functions, where $p(x)$ really is a probability and really is capped at 1 — the whole point of this subsection is that the continuous case is not the same object wearing a different hat. Option C is the same confusion stated more sharply: for continuous $X$, $P(X=x)$ is exactly 0 for every $x$, since infinitely many positive point-probabilities could not total 1. This matters in practice because a log-likelihood computed from a density can come out positive, and people report that as a bug when it is not (§6.3).'
      },
      {
        q: 'A feature $B$ turns out to be independent of the label $A$. What is $P(A \\mid B)$?',
        options: ['Zero', 'Equal to $P(A)$ — unchanged by the evidence', 'Equal to $P(B)$', 'Impossible to say without the joint table'],
        answer: 1,
        why: 'Independence is defined by $P(A,B) = P(A)P(B)$, and substituting that into the definition of the conditional makes the $P(B)$ cancel and leaves $P(A \\mid B) = P(A)$ exactly. That second form is the one to think in: the evidence arrived and moved your belief nowhere. Option D is the plausible-sounding wrong answer, because normally you cannot compute a conditional without the joint — but independence <i>is</i> a complete statement about the joint, so nothing further is needed. The practical reading is the one to keep: a feature independent of the label carries zero information about it and can be deleted without any loss, which you can watch happen in the lab by pressing <i>Make them independent</i> and seeing the conditional bar settle onto the marginal. Be careful about the converse, though: zero <i>correlation</i> does not imply independence, as the parabola example in §0.4.2 shows.'
      }
    ],
    cards: [
      { q: 'Conditional probability in one sentence', a: 'The same table with the impossible rows deleted, renormalised so what remains sums to one.' },
      { q: 'Definition of independence', a: '$P(A,B)=P(A)P(B)$, equivalently $P(A\\mid B)=P(A)$ — the evidence changes nothing.' },
      { q: 'Expectation, in words', a: 'The weighted average of the possible values, weighted by their probabilities. It need not be a value the variable can take.' },
      { q: 'Variance, two forms', a: '$\\mathrm{Var}(X)=\\mathbb{E}[(X-\\mu)^2]=\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$. Its square root, $\\sigma$, is in the original units.' },
      { q: 'Mass versus density', a: 'Mass $p(x)$ <i>is</i> a probability and is at most 1. Density is probability per unit of $x$, may exceed 1, and only its area over an interval is a probability.' },
      { q: 'Why averaging helps only as $\\sqrt{n}$', a: 'Independent variances add, so the variance of a mean falls as $1/n$ and the standard deviation as $1/\\sqrt{n}$. Four times the data halves the noise.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.5 */
  ML.section({
    id: 'python-toolkit', track: 'start', num: '0.5',
    title: 'The working toolkit: numpy, pandas, scikit-learn, PyTorch',
    lede: 'Four libraries carry very nearly all of the practical work: numpy for arrays, pandas for tables, scikit-learn for classical models, PyTorch for networks. This section is about the three ideas that make them usable — thinking in whole arrays rather than loops, keeping shapes straight, and the fit-then-transform discipline — and about the one habit that separates a number you can report from a number that is quietly a lie.',
    html: `
<p>You have a million numbers in a Python list and you want to standardise them: subtract the mean, divide by the standard deviation. The obvious code writes itself.</p>

${H.code(`total = 0.0
for v in values:            # a plain Python list of 1,000,000 floats
    total += v
mu = total / len(values)

out = []
for v in values:
    out.append((v - mu) / sd)`)}

<p>It is correct, it is readable, and on a modern laptop it takes something in the region of a tenth of a second. That does not sound like a problem until you notice that this is one operation on one column, and a real preprocessing pass does perhaps fifty such operations across a hundred columns. The tenth of a second becomes several minutes, every time you change anything, and the loop that felt harmless has eaten your afternoon.</p>

<p>The reason it is slow is worth knowing, because it explains the entire design of the library that replaces it. Every element of a Python list is a full object with a type tag and a reference count, scattered somewhere in memory; every trip round the loop the interpreter has to look up what <code>+</code> means for these particular objects, allocate a new object for the result, and update bookkeeping. Almost none of the time is spent adding.</p>

<p>What you want is for the million numbers to sit side by side in one block of memory, all the same type, so that a compiled loop can march through them with no interpretation at all. That is what a numpy array is, and the same operation on one takes a couple of milliseconds — commonly fifty to a hundred times faster, and considerably more when the operation is one the library can hand to specialised linear-algebra code.</p>

${H.key('Describe the operation on the whole array. Never write a Python loop over rows when an array operation exists.')}

<h2><span class="sn">0.5.1</span> numpy: arrays, axes, and shapes that must line up</h2>

<p>An array has three things worth knowing about it: its <b>shape</b>, a tuple saying how large it is along each axis; its <b>dtype</b>, the single machine type every element shares, usually <code>float64</code> or <code>float32</code>; and its contents. §0.2 established the shape rule that governs everything — $[n \\times d]$ times $[d \\times k]$ gives $[n \\times k]$, with the inner dimensions cancelling — so what follows builds on that rather than repeating it.</p>

${H.code(`import numpy as np

X = np.random.randn(1000, 8)              # [n=1000, d=8] a batch of examples
w = np.random.randn(8)                    # [d=8]         one weight per feature
y = X @ w + 0.1 * np.random.randn(1000)   # [n=1000]      @ is matrix multiply

# vectorised standardisation: the whole loop above, in two lines
mu, sd = X.mean(axis=0), X.std(axis=0)    # [8] and [8] — per-column statistics
Z = (X - mu) / sd                         # [1000, 8] — broadcasting does the work

# the least-squares solution, three ways
beta_normal = np.linalg.solve(X.T @ X, X.T @ y)     # fine, and fast
beta_lstsq  = np.linalg.lstsq(X, y, rcond=None)[0]  # numerically safer
beta_ridge  = np.linalg.solve(X.T @ X + 1e-2*np.eye(8), X.T @ y)`)}

<p>The single most confusing thing for a newcomer is the <code>axis</code> argument, and there is a rule that removes the confusion permanently. <code>X.mean(axis=0)</code> on a $[1000 \\times 8]$ array returns 8 numbers, not 1000. That seems backwards until you say the rule aloud: <b>the axis you name is the axis that disappears</b>. Axis 0 is the row axis, so naming it collapses the thousand rows away and leaves one value per column. Naming <code>axis=1</code> collapses the eight columns and leaves one value per row.</p>

<p>The second thing worth learning properly is <b>broadcasting</b>, which §0.2 introduced as the reason a bias vector can be added to a matrix. The full rule is short. Line the two shapes up from the right-hand end. Two dimensions are compatible if they are equal, or if one of them is 1, or if one array has run out of dimensions entirely. Compatible dimensions of size 1 are stretched — conceptually, and without ever copying the data — to match the other. So $[1000 \\times 8]$ against $[8]$ lines up as $8$ against $8$, then $1000$ against nothing, and the result is $[1000 \\times 8]$: the same eight column means are reused for every row, which is exactly what standardisation wants.</p>

${H.pitfall(`<p>Now look at what those rules permit. Take a vector of true labels with shape $[1000]$ and a vector of predictions that came out of a model with shape $[1000 \\times 1]$ — a difference nobody notices, because both print as a column of a thousand numbers.</p>
<p>Line the shapes up from the right: $1$ against $1000$, which is compatible because one of them is 1; then $1000$ against nothing, also compatible. Broadcasting therefore produces a $[1000 \\times 1000]$ array containing every pairwise difference, a million numbers where you wanted a thousand. Taking <code>.mean()</code> of that gives a perfectly plausible-looking float, no exception is raised, and your reported error is meaningless.</p>
<p>This is the most common silent bug in numerical Python and it costs people days. Three habits kill it. Write the expected shape in a comment after every line, as §0.2 insisted. Use <code>.ravel()</code> or <code>.squeeze(-1)</code> to flatten a stray trailing axis the moment it appears. And in any function whose output you will report, put an outright <code>assert pred.shape == y.shape</code> at the top — one line, and it converts a silent wrong answer into a loud stack trace.</p>`)}

${H.practice(`<p>Two further numpy behaviours cause trouble in real code.</p>
<p><b>Slices are views, not copies.</b> <code>B = A[:, :3]</code> does not copy anything; it hands you a window onto the same memory, so writing into <code>B</code> changes <code>A</code>. This is a deliberate performance decision — copying a large array is expensive — and it is fine once you expect it. When you want a genuine copy, say <code>A[:, :3].copy()</code>.</p>
<p><b>Prefer a decomposition to an explicit inverse.</b> The first of the three least-squares lines above forms $X^\\mathsf{T}X$ and solves with it, which is fast and usually fine. But squaring a matrix squares its condition number, so if $X$ was already awkward — collinear features, exactly the case §0.2 warned about — the squared version is far worse, and you can lose most of your significant digits. <code>np.linalg.lstsq</code> works on $X$ directly through a QR or SVD factorisation and is the safer default. §1.15 quantifies how many digits each route costs you.</p>`)}

<h2><span class="sn">0.5.2</span> pandas: tables with names, and the aggregate that sees the future</h2>

<p>numpy arrays are homogeneous and anonymous: every element has the same type and the columns have no names. Real data is neither. A loan application table has dates, strings, integers and floats side by side, and you want to refer to a column as <code>utilisation</code> rather than as column 6. A pandas <b>DataFrame</b> is a collection of named columns, each internally a numpy array of its own type, plus an <b>index</b> that labels the rows.</p>

${H.code(`import pandas as pd

df = pd.read_parquet("applications.parquet")
df["utilisation"] = df.balance / df.limit           # whole-column arithmetic
df["age_days"] = (df.decision_date - df.opened_date).dt.days

# GOOD: an aggregate with an explicit time boundary
hist = (df[df.decision_date < cutoff]
        .groupby("customer_id")["amount"].mean()
        .rename("mean_amount_before_cutoff"))

# WRONG: this aggregate sees the whole history, including the future
df["mean_amount"] = df.groupby("customer_id")["amount"].transform("mean")`)}

<p>Before the wrong line, two things about the index, because it is where pandas surprises people. The index is not decoration: arithmetic between two Series aligns on <i>labels</i>, not on positions. Add a Series indexed $0,1,2$ holding $1,2,3$ to one indexed $1,2,3$ holding $10,20,30$ and you get four rows — $\\mathrm{NaN}$, $12$, $23$, $\\mathrm{NaN}$ — because labels 0 and 3 had no partner. If you were expecting three rows of element-wise sums, you now have a column of nulls arriving from nowhere. The habit that prevents it is <code>.reset_index(drop=True)</code> after any filtering, or using <code>.values</code> when you genuinely want positional arithmetic.</p>

<p>The second is dtypes. A column of strings is stored as <code>object</code>, meaning a numpy array of pointers to Python objects, which puts you straight back in the slow world described at the top of this section. Converting such a column to <code>category</code> can shrink it by an order of magnitude and speed up grouping considerably. Equally, a date column read as a string will compare and sort alphabetically, so <code>"2024-1-9" &lt; "2024-11-02"</code> comes out true and every time-based split you build afterwards is wrong. Check <code>df.dtypes</code> before you check anything else.</p>

<h3>Why that last line is the most expensive mistake in tabular work</h3>

<p>Both <code>groupby</code> lines compute a mean per customer. The difference is which rows go into the mean, and it is worth making completely concrete.</p>

<p>Suppose customer 4471 appears three times: a January application for £100, a February one for £200, and a December one for £900. The <code>transform("mean")</code> call attaches the same number to all three rows, namely $(100 + 200 + 900)/3 = 400$. So the January row now carries a feature whose value is 400 — a number that depended on a transaction that had not happened yet and could not possibly have been known in January.</p>

<p>Your model will find this feature enormously useful, because it is: it is a partial view of the future. Your backtest will look excellent. And on the day the model goes live, the feature is computed from history alone, comes out as 100 rather than 400, and the performance you promised evaporates. The failure is not detectable by any check on the model, because the model is fine. The data was wrong.</p>

${H.flag('This is the single most common leakage bug in tabular machine learning, and it is worth stating as a rule: any aggregate attached to a row must be computed only from information that existed before that row\'s timestamp. §2.11 has the full checklist, and §2.14 covers the out-of-time validation that catches what the checklist misses.')}

<h2><span class="sn">0.5.3</span> scikit-learn: fit, transform, and the discipline that protects your score</h2>

<p>scikit-learn's whole API is one idea repeated. Every object has <code>fit</code>, which looks at data and stores something; and then either <code>transform</code>, which applies what it stored, or <code>predict</code>, which uses what it stored to make predictions. That is it, across two hundred classes.</p>

<p>The important consequence is easy to miss. Look at what <code>StandardScaler.fit</code> actually does: it computes the mean and the standard deviation of each column and stores them, as <code>mean_</code> and <code>scale_</code>. Those stored numbers are then used by <code>transform</code>. In other words <b>a scaler is a model with learned parameters</b>, no different in kind from a regression's coefficients. It just happens to have two per column instead of one.</p>

<p>Once you see the scaler as a model, the rule about validation writes itself. You would never dream of fitting a classifier on your validation rows and then reporting its accuracy on those same rows. Fitting a scaler on them is the same act, only quieter — the validation rows contributed to the mean and the standard deviation, so information from them has been baked into the training data before the model ever saw it.</p>

${H.code(`from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([
    ("scale",  StandardScaler()),                  # fitted INSIDE each fold
    ("select", SelectKBest(f_classif, k=20)),      # fitted INSIDE each fold
    ("clf",    LogisticRegression(C=1.0, max_iter=1000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(auc.mean(), auc.std())`)}

<p>A <code>Pipeline</code> is not a tidiness device. It is a correctness device. When <code>cross_val_score</code> splits the data, it calls <code>fit</code> on the whole pipeline using the training fold only, so the scaler's mean, the selector's chosen columns and the classifier's coefficients are all learned from the same restricted set of rows; then it calls <code>predict</code> on the held-out fold, and every stage merely applies what it stored. Standardise the whole matrix first and cross-validate afterwards and none of that protection exists.</p>

${H.key('Every learned transformation — scaling, imputation, encoding, binning, feature selection — is fitted on the training rows only, inside the fold. Treat them as model parameters, because that is what they are.')}

<p>Now the honest part, because the rule is usually taught with more conviction than evidence. If the only thing you fit outside the fold is a <code>StandardScaler</code> on a few thousand rows, the leak is real but tiny — the validation rows moved the column mean by a fraction of its standard error, and the reported score is inflated by well under a percentage point. Many people notice this, conclude that the rule is pedantry, and drop it.</p>

<p>That conclusion is the trap, because the same mistake applied to a transformation that <i>looks at the labels</i> is not small at all. Feature selection, target encoding and group-mean imputation all read $y$ while they fit. Do any of those on the full dataset and the inflation stops being a rounding error and becomes the entire result.</p>

<h3>Manufacturing a discovery out of pure noise</h3>

<p><b>What you are looking at.</b> The horizontal axis is the number of candidate features you offer to a feature selector, doubling from 8 to 256. The vertical axis is five-fold cross-validated accuracy. The dashed grey line at $0.500$ is the accuracy of guessing. Underneath both curves sits exactly the same procedure: standardise, keep the $k$ features most strongly correlated with the label, fit a small ridge classifier on those, score the held-out fold. The <span style="color:var(--red)">red</span> curve does the selection once on the whole dataset before cross-validating, which is the mistake. The <span style="color:var(--blue)">blue</span> curve does the selection separately inside each training fold, which is correct. Both are averaged over eight independently generated datasets, which steadies the lines without removing all of the wobble that a 60-row sample honestly has.</p>

<p><b>What to do with it.</b> Leave <i>genuinely informative features</i> at 0. This means the labels are coin flips and every single feature is pure noise: there is nothing to find, and any honest procedure must report 0.5. Now read the two curves from left to right. Then raise the number of informative features to 2 or 3 and watch what changes — the blue curve lifts off the chance line because there is now something real to detect, and it <i>sinks</i> again as the pile of noise features grows and the genuine ones get harder to pick out of the crowd. Increase the number of rows and both curves become better behaved, which is the usual relationship between sample size and self-deception.</p>

<p><b>The thing genuinely worth noticing.</b> With zero informative features, the blue curve stays flat along the 0.500 line — wandering a couple of points either side, because 60 rows is a small sample and even a correct estimate is noisy — while the red curve climbs steadily past 0.60, past 0.70, and keeps going as you offer the selector more noise to choose from. There is no signal in this data. None. The red procedure is reporting a discovery it manufactured entirely out of the act of looking — with 256 candidates, a handful will correlate with the labels by luck, and because the selection was made using all the rows, those lucky features are still lucky in the held-out fold. The gap widens with the number of candidates because the more you look, the luckier the best-looking thing gets. Nothing in the code raises a warning, the cross-validation is textbook, and the number is a fabrication.</p>

${H.lab('leak', 'Fitting outside the fold, measured', 'Both curves run the identical modelling procedure. The only difference is whether the feature selection is fitted on all the rows or refitted inside each training fold. Set the informative-feature count to 0 to work with data that contains nothing at all to find.')}

${H.practice(`<p>The pipeline rule covers the transformations scikit-learn knows about. Three real-world leaks slip past it, and all three have to be handled by hand.</p>
<p><b>Leaks upstream of the code.</b> If the feature table was assembled by a SQL job that aggregated over the whole period, no pipeline can help you — the damage was done before Python started. Read the feature definitions, not just the model code.</p>
<p><b>Duplicate or near-duplicate rows.</b> The same customer appearing in both the training and the validation fold is memorisation dressed up as generalisation. Split by entity, using <code>GroupKFold</code>, rather than by row.</p>
<p><b>Time.</b> A random split on time-ordered data trains on the future to predict the past, which no amount of fold discipline repairs, because the folds themselves are wrong. Use an out-of-time split (§2.14) whenever the data has an arrow of time in it, which is nearly always.</p>`)}

<h2><span class="sn">0.5.4</span> PyTorch: the same three objects, differentiated automatically</h2>

<p>A PyTorch <b>tensor</b> is a numpy array with two additions: it can live on a GPU, and it can remember the operations performed on it. That second property is the whole point. As you compute, PyTorch quietly records the computation graph — the same chain of stages you dragged sliders through in §0.3 — and <code>loss.backward()</code> then walks that graph from the loss backwards, applying the chain rule at each node and depositing the resulting gradient in each parameter's <code>.grad</code> attribute. Reverse-mode automatic differentiation is exactly §0.3.5 done by machine, and §3.11 covers how the graph is built and freed.</p>

${H.code(`import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.BCEWithLogitsLoss()      # logits in, not probabilities

for xb, yb in loader:
    opt.zero_grad(set_to_none=True)   # gradients accumulate by default
    logits = model(xb).squeeze(-1)    # [B, 1] -> [B]; shapes must match yb
    loss = loss_fn(logits, yb.float())
    loss.backward()                   # reverse-mode autodiff (§0.3)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()                        # theta <- theta - eta * grad`)}

<p>Read that loop against §0.1 and it is the three objects and nothing else. <code>model</code> is $f_\\theta$. <code>loss_fn</code> is $\\ell$. <code>opt.step()</code> is $\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta L$. Everything in between exists to compute the gradient that the last line consumes.</p>

<p>The one genuinely surprising line is the first. PyTorch <i>adds</i> each new gradient to whatever is already in <code>.grad</code> rather than replacing it, which sounds like a design mistake until you want to simulate a batch larger than your GPU can hold — then you deliberately run several small batches, let the gradients pile up, and step once. Because accumulation is the default, forgetting to clear it is silent: your first step uses one batch's gradient, your second uses the sum of two, your tenth the sum of ten, and the effective learning rate grows without bound while the code runs perfectly.</p>

${H.table(['Idiom', 'What goes wrong without it'], [
      ['<code>BCEWithLogitsLoss</code>, not <code>Sigmoid</code> then <code>BCELoss</code>', 'The fused version rearranges the algebra so the exponentials never overflow. Done separately, a logit of 800 saturates the sigmoid to exactly 1.0, the log of $1-1.0$ is $-\\infty$, and the loss becomes NaN (§1.10)'],
      ['<code>opt.zero_grad()</code> at the top of every step', 'Gradients accumulate by design, so each step applies an ever-growing sum of past batches'],
      ['<code>model.eval()</code> and <code>torch.no_grad()</code> at inference', '<code>eval()</code> switches dropout off and stops batch-norm updating its running statistics; <code>no_grad()</code> stops building the graph, saving a large amount of memory'],
      ['<code>clip_grad_norm_</code>', 'One pathological batch produces a huge gradient, the parameters jump somewhere absurd, and every subsequent loss is NaN (§4.11)'],
      ['<code>AdamW</code>, not <code>Adam(weight_decay=...)</code>', 'In plain Adam the decay term is divided by the same running gradient scale as everything else, so it decays much less than you asked (§3.5)'],
      ['<code>float32</code> by default, not <code>float64</code>', 'Doubles halve your throughput and memory bandwidth for accuracy a network cannot use. Save the extra precision for the places that need it (§1.15)']
    ])}

<h2><span class="sn">0.5.5</span> The three mistakes that invalidate results</h2>

<p>Everything above is craft. These three are the ones that turn a working project into a number nobody should have believed, and every one of them produces code that runs cleanly and reports a confident answer.</p>

${H.checklist([
      '<b>Fitting a transformer outside the fold.</b> Scaling, imputation, encoding and selection all learn parameters from data. Treat them as model parameters, because they are. The lab above shows the size of the damage when the transformation reads the labels: chance dressed up as 0.75.',
      '<b>Shuffling time-structured data.</b> A random k-fold on a time series trains on the future to predict the past. The score will be excellent and unreproducible in production. Use out-of-time validation (§2.14), and split by entity when rows repeat.',
      '<b>Reporting the tuned score.</b> The validation set that chose your hyperparameters has been used for fitting, just at a coarser grain, so it is no longer an unbiased estimate of anything. Nest the tuning inside an outer loop, or hold out a final set you touch exactly once (§2.15).'
    ])}

${H.history(`<p>None of this stack was planned. Numeric, the ancestor of numpy, appeared in the mid-1990s; a rival array package split the community, and numpy exists because Travis Oliphant merged the two in 2006. pandas began in 2008 as one analyst’s tooling at a quantitative fund and was open-sourced two years later. scikit-learn started as a Google Summer of Code project in 2007 and was rebuilt at INRIA into the library that fixed the fit/predict convention now imitated everywhere.</p>
<p>The idea underneath all of them — that you should describe an operation on a whole array rather than on its elements — is much older, and goes back at least to APL in the 1960s and to the array syntax added to Fortran in 1990. What changed was not the idea but the economics: once vectorised numerical code became dramatically faster than interpreted loops on the same hardware, an ergonomic array language stopped being an elegance and became the only practical way to work.</p>`)}

${H.probe([
      ['Why is a StandardScaler fitted inside the cross-validation fold rather than before the split?', 'Because it has learned parameters — the per-column mean and standard deviation — so fitting it on all the rows lets the validation rows influence the training data. It is the same error as fitting a classifier on its own test set, only quieter.'],
      ['You are told a model scores 0.78 AUC in cross-validation. What do you ask?', 'How the folds were built, whether every learned transformation was inside them, whether rows can repeat across folds, and whether the data has a time order that a random split destroyed. Then whether 0.78 is the tuned number or a genuinely held-out one.'],
      ['What does axis=0 mean in numpy?', 'The axis being collapsed. On a $[n \\times d]$ array, <code>mean(axis=0)</code> removes the row axis and returns $d$ column means.'],
      ['Why does PyTorch accumulate gradients instead of replacing them?', 'So that a large batch can be simulated by several small ones on limited memory. The cost is that forgetting <code>zero_grad</code> is silent rather than an error.'],
      ['Name a leak that a scikit-learn Pipeline cannot protect you from.', 'Anything upstream of Python — a feature table built by a SQL aggregate over the whole period — plus duplicate entities spanning folds, and a random split on time-ordered data.']
    ], 'Answering "I use a Pipeline" as though it were a complete answer. The pipeline handles the transformations sklearn can see. It does nothing about how the folds were constructed, about the same customer appearing in two of them, or about a feature that was already contaminated before the file was written.')}`,
    labs: {
      leak: function (host) {
        const PS = [8, 16, 32, 64, 128, 256];
        let seed = 23, cacheKey = '', cache = null;

        const st = Viz.controls(host, [
          { k: 'signal', label: 'genuinely informative features', min: 0, max: 4, step: 1, value: 0, fmt: v => String(v) },
          { k: 'n', label: 'rows of data', min: 40, max: 160, step: 20, value: 60, fmt: v => String(v) },
          { k: 'k', label: 'features the selector keeps', min: 2, max: 10, step: 1, value: 5, fmt: v => String(v) }
        ], () => S.redraw());

        const out = Viz.readout(host, [
          { k: 'lk', label: 'reported CV, fitted outside', cls: 'key' },
          { k: 'hn', label: 'honest CV, fitted inside' },
          { k: 'gap', label: 'invented accuracy' }
        ]);

        /* one dataset: PS[last] candidate columns, `signal` of them real */
        function run(n, k, signal, sd) {
          const R = Num.rng(sd), P = PS[PS.length - 1], folds = 5;
          const cols = [], y = new Array(n);
          for (let j = 0; j < P; j++) {
            const c = new Array(n);
            for (let i = 0; i < n; i++) c[i] = R.normal(0, 1);
            cols.push(c);
          }
          for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < signal; j++) s += cols[j][i];
            y[i] = (0.8 * s + R.normal(0, 1)) > 0 ? 1 : 0;
          }
          const all = []; for (let i = 0; i < n; i++) all.push(i);

          function absCorr(c, rows) {
            const m = rows.length;
            let mx = 0, my = 0;
            for (let q = 0; q < m; q++) { mx += c[rows[q]]; my += y[rows[q]]; }
            mx /= m; my /= m;
            let sxy = 0, sxx = 0, syy = 0;
            for (let q = 0; q < m; q++) {
              const dx = c[rows[q]] - mx, dy = y[rows[q]] - my;
              sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
            }
            const den = Math.sqrt(sxx * syy);
            return den > 1e-12 ? Math.abs(sxy / den) : 0;
          }

          const hit = { leaky: PS.map(() => 0), honest: PS.map(() => 0) };
          const seen = PS.map(() => 0);

          for (let f = 0; f < folds; f++) {
            const tr = all.filter(i => i % folds !== f), te = all.filter(i => i % folds === f);
            if (tr.length < 6 || !te.length) continue;
            const sc = { leaky: cols.map(c => absCorr(c, all)), honest: cols.map(c => absCorr(c, tr)) };
            ['leaky', 'honest'].forEach(mode => {
              const s = sc[mode];
              PS.forEach((p, pi) => {
                const order = []; for (let j = 0; j < p; j++) order.push(j);
                order.sort((a, b) => s[b] - s[a]);
                const use = order.slice(0, Math.min(k, p));
                const mu = use.map(j => { let t = 0; tr.forEach(i => { t += cols[j][i]; }); return t / tr.length; });
                const Phi = tr.map(i => [1].concat(use.map((j, q) => cols[j][i] - mu[q])));
                const wv = Num.ridgeFit(Phi, tr.map(i => y[i]), 1.0);
                te.forEach(i => {
                  let z = wv[0];
                  use.forEach((j, q) => { z += wv[q + 1] * (cols[j][i] - mu[q]); });
                  if (((z > 0.5) ? 1 : 0) === y[i]) hit[mode][pi]++;
                });
                if (mode === 'leaky') seen[pi] += te.length;
              });
            });
          }
          return {
            leaky: hit.leaky.map((v, i) => seen[i] ? v / seen[i] : 0.5),
            honest: hit.honest.map((v, i) => seen[i] ? v / seen[i] : 0.5)
          };
        }

        function curves() {
          const key = [st.n, st.k, st.signal, seed].join('|');
          if (key === cacheKey && cache) return cache;
          const reps = 8, L = PS.map(() => 0), Hn = PS.map(() => 0);
          for (let r2 = 0; r2 < reps; r2++) {
            const o = run(st.n, st.k, st.signal, seed + r2 * 977);
            PS.forEach((p, i) => { L[i] += o.leaky[i] / reps; Hn[i] += o.honest[i] / reps; });
          }
          cacheKey = key; cache = { L: L, Hn: Hn };
          return cache;
        }

        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const c = curves();
            const P = Viz.plot(ctx, w, h, { xd: [2.7, 8.3], yd: [0.3, 1] }).frame({
              xlabel: 'candidate features offered to the selector',
              ylabel: '5-fold CV accuracy',
              xticks: [3, 4, 5, 6, 7, 8],
              xfmt: v => String(Math.round(Math.pow(2, v)))
            });
            P.hline(0.5, { color: T.faint, dash: [4, 4], label: 'coin flip' });
            const xs = PS.map(p => Math.log(p) / Math.LN2);
            P.clip(() => {
              P.line(xs.map((x, i) => [x, c.L[i]]), { color: T.red, width: 2.6 });
              P.line(xs.map((x, i) => [x, c.Hn[i]]), { color: T.blue, width: 2.6 });
              P.dots(xs.map((x, i) => [x, c.L[i]]), { r: 4.5, color: T.red, stroke: true });
              P.dots(xs.map((x, i) => [x, c.Hn[i]]), { r: 4.5, color: T.blue, stroke: true });
            });
            const last = PS.length - 1, d = c.L[last] - c.Hn[last];
            out({
              lk: c.L[last].toFixed(3),
              hn: c.Hn[last].toFixed(3),
              gap: (d >= 0 ? '+' : '') + d.toFixed(3)
            });
          }
        });

        Viz.buttons(host, [
          { label: 'New random datasets', on: () => { seed = 1 + Math.floor(Math.random() * 1e6); S.redraw(); } },
          { label: 'Pure noise (signal = 0)', on: () => { st.$set('signal', 0); S.redraw(); } }
        ]);
        Viz.legend(host, [
          { c: Viz.theme().red, t: 'selection fitted on all rows, then cross-validated' },
          { c: Viz.theme().blue, t: 'selection refitted inside each training fold' },
          { c: Viz.theme().faint, t: 'chance (0.500)' }
        ]);
      }
    },
    quiz: [
      {
        q: 'Why wrap the scaler and the model in a Pipeline before cross-validating?',
        options: ['It is faster', 'So the scaler is re-fitted inside each training fold, preventing the validation rows from influencing it', 'It makes the model more accurate', 'Because sklearn requires it'],
        answer: 1,
        why: 'A scaler has learned parameters — the per-column mean and standard deviation it stores as <code>mean_</code> and <code>scale_</code> — so fitting it is fitting a model. Do that on the whole dataset and the validation rows have already influenced the numbers the training rows are transformed with, which is the same error as training a classifier on its test set, just quieter. Option C is the tempting one: a pipeline does not improve the model at all, and in fact the honest score it produces is usually <i>lower</i> than the leaky one, which is precisely why the practice feels like a step backwards when you first adopt it. For a scaler alone the inflation is small, often under a percentage point, and that is exactly why people talk themselves out of the rule. The lab above shows what the same mistake does to a transformation that reads the labels: an accuracy of 0.75 on data that contains no signal whatsoever.'
      },
      {
        q: 'You forget to call opt.zero_grad() in a PyTorch loop. What happens?',
        options: ['The model does not train at all', 'Gradients accumulate across batches, so each step uses a growing sum of past gradients', 'Learning rate is ignored', 'It raises an exception'],
        answer: 1,
        why: 'PyTorch adds each new gradient into <code>.grad</code> rather than overwriting it, and this is deliberate: it is what lets you simulate a batch larger than your memory allows by running several small batches and stepping once. The consequence of forgetting to clear it is that step one applies one batch\'s gradient, step two the sum of two, step ten the sum of ten — so the effective learning rate grows without bound and the run usually diverges after a few hundred steps. Option D is the answer people expect, because a mistake this damaging surely ought to be an error; it is not, and that is the whole reason it is worth knowing. Nothing in the API can tell the difference between deliberate accumulation and a forgotten line, so the code runs perfectly and produces nonsense.'
      },
      {
        q: 'You subtract predictions of shape (1000, 1) from labels of shape (1000,) and take the mean of the result. What do you get?',
        options: ['A correct mean error', 'A shape error', 'The mean of a 1000×1000 array of all pairwise differences', 'Always zero'],
        answer: 2,
        why: 'Broadcasting lines shapes up from the right: 1 against 1000 is compatible because one side is 1, and then 1000 against nothing is compatible because the second array has run out of dimensions. So numpy obligingly builds a $[1000 \\times 1000]$ array holding every pairwise difference, and <code>.mean()</code> of it is a real number that looks entirely reasonable. Option B is the tempting answer, and it is tempting for a good reason — this <i>should</i> be an error, and the fact that it is not is why the bug survives to production. The defences are cheap: annotate shapes in comments as §0.2 insists, flatten stray trailing axes with <code>.ravel()</code> or <code>.squeeze(-1)</code> as soon as they appear, and put an <code>assert pred.shape == y.shape</code> in any function whose output you intend to report.'
      }
    ],
    cards: [
      { q: 'The vectorisation rule', a: 'Describe the operation on the whole array. A Python loop over rows pays interpreter overhead per element; an array operation runs a compiled loop over contiguous memory.' },
      { q: 'What does axis=0 mean?', a: 'The axis you name is the axis that disappears. On $[n \\times d]$, <code>mean(axis=0)</code> collapses the rows and returns $d$ column means.' },
      { q: 'The broadcasting rule', a: 'Line the shapes up from the right. Dimensions match if equal, if one is 1, or if one array has run out — and size-1 dimensions are stretched.' },
      { q: 'The pipeline rule', a: 'Every learned transformation is fitted inside the training fold only — scaling, imputation, encoding, binning, selection.' },
      { q: 'The groupby-transform leak', a: 'A group aggregate over the whole table attaches each row a number computed from its own future. Bound every aggregate by the row\'s timestamp.' },
      { q: 'Why BCEWithLogitsLoss over Sigmoid+BCELoss', a: 'It fuses the sigmoid and the log using the log-sum-exp trick, so large logits do not saturate to exactly 1 and produce a NaN loss.' }
    ]
  });
})();
