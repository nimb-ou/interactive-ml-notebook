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

<p>That this is achievable at all is neither obvious nor guaranteed. It works when your training examples genuinely represent what arrives later, and when the family of rules you searched was not so flexible that it could fit random noise as readily as real signal. Making that statement precise is the work of §1.4 on concentration. Controlling it in practice is the work of §2.2 on bias and variance, §2.3 on regularization, and §2.14 on validation.</p>

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
        why: 'The pattern to recognise here is the <i>gap</i> between the two numbers, rather than either number taken on its own. Start by noticing what the 0.99 tells you: the optimiser worked beautifully, because it found parameters that fit the training examples almost exactly. So the very number that looks alarming actually rules out option A. What the gap tells you instead is that the family of rules you searched was flexible enough to fit the random noise in your particular sample, and given the chance, it did exactly that. Memorising the training set is always available to a sufficiently flexible model, and it always produces this signature. The fix is to reduce that flexibility, either by choosing a simpler family or by adding regularization (§2.3), or to supply more data so that the noise averages out and only real structure survives. This gap has a name — variance — and §2.2 is devoted entirely to it.'
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
      ['Norm $\\|x\\|_2$', 'Length: $\\sqrt{\\sum_i x_i^2}$', 'L2 regularization is a penalty on this length (§2.3)'],
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
    lede: 'Training is one idea repeated: measure which way the loss goes up, step the other way. The derivative is that measurement, the gradient is its multi-dimensional version, and the chain rule is what makes it computable through a hundred layers.',
    html: `
<h2><span class="sn">0.3.1</span> A derivative is a slope you can act on</h2>
<p>$f'(x)$ is the limit of rise over run as the run shrinks: how much $f$ changes per unit change in $x$, right here. Two readings matter. <b>Locally</b>, $f(x + \\epsilon) \\approx f(x) + \\epsilon f'(x)$ — the derivative is the best linear prediction of what a small step will do. <b>Globally</b>, $f' = 0$ marks the flat points: minima, maxima, saddles.</p>
<p>Optimisation lives entirely in the first reading. If $f'(x) > 0$, moving right increases $f$; to decrease it, move left. Hence <b>gradient descent</b>: $x \\leftarrow x - \\eta f'(x)$, where $\\eta$ is a step size you choose and immediately regret.</p>

${H.lab('deriv', 'Slope, tangent, and one gradient step', 'Drag the point along the curve. The dashed line is the tangent — the linear model the derivative gives you. Press <i>step</i> to take one gradient step and watch it land.')}

<h2><span class="sn">0.3.2</span> Rules you will use daily</h2>
${H.table(['Rule', 'Statement', 'The ML instance'], [
      ['Power', '$\\frac{d}{dx}x^n = nx^{n-1}$', 'Squared error differentiates to $2(\\hat y - y)$'],
      ['Exponential / log', '$\\frac{d}{dx}e^x = e^x$, $\\frac{d}{dx}\\ln x = 1/x$', 'Every log-likelihood'],
      ['Product', '$(uv)\' = u\'v + uv\'$', 'Attention scores, gated units'],
      ['Chain', '$\\frac{d}{dx}f(g(x)) = f\'(g(x))g\'(x)$', '<b>Backpropagation, entirely</b>'],
      ['Sigmoid', '$\\sigma\' = \\sigma(1-\\sigma)$', 'The two-line logistic gradient (§2.4)']
    ])}

<h2><span class="sn">0.3.3</span> Gradients: many knobs at once</h2>
<p>With several parameters, the partial derivative $\\partial f/\\partial \\theta_j$ asks: if I nudge <i>this</i> knob and freeze the others, how does $f$ move? Stack them and you have the gradient</p>
$$\\nabla f(\\theta) = \\left(\\frac{\\partial f}{\\partial\\theta_1}, \\ldots, \\frac{\\partial f}{\\partial\\theta_p}\\right)$$
<p>which points in the direction of <b>steepest increase</b>, so $-\\nabla f$ is the direction of steepest decrease. The second derivative generalises to the <b>Hessian</b> $H$, the matrix of curvatures; it tells you how much the gradient itself changes as you move, which is what §1.9 needs to explain why some problems train easily and others crawl.</p>

<h2><span class="sn">0.3.4</span> The chain rule is the entire mechanism</h2>
<p>A network is a composition: $L(f_3(f_2(f_1(x))))$. The chain rule says the derivative of a composition is the <i>product</i> of the local derivatives. Backpropagation is nothing more than evaluating that product from the outside in, caching each intermediate so no factor is computed twice.</p>

${H.lab('chain', 'A computation graph, with numbers flowing both ways', 'Change the inputs and watch the forward values (blue, left to right) and the gradients (red, right to left). Every red number is a product of the local derivatives on the path back to it — that is all backprop is.')}

${H.key('Every gradient in a network is a product of local derivatives along a path. Multiply enough factors below one and it vanishes; that single sentence explains ReLU, residual connections and normalisation.')}

${H.probe([
      ['Why reverse-mode and not forward-mode differentiation?', 'The loss is one number and the parameters are many. Reverse mode costs about one forward pass per <i>output</i>; forward mode costs one per <i>input</i>. One output, billions of inputs — reverse wins by a factor of billions.'],
      ['What does the Hessian tell you?', 'Curvature. Its eigenvalue ratio (the condition number) sets both the largest stable step and the convergence rate (§1.9).']
    ])}`,
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
        why: '∇L is the direction of steepest <i>increase</i>. Descending means moving against it.'
      },
      {
        q: 'A 40-layer network trains with sigmoid activations and the early layers barely move. The most direct explanation is…',
        options: ['The learning rate is too large', 'Each backward step multiplies by σ′ ≤ 0.25, so the product over depth collapses', 'The loss is non-convex', 'Batch size is too small'],
        answer: 1,
        why: 'Vanishing gradients: the chain rule multiplies local derivatives, and 0.25⁴⁰ is astronomically small. ReLU passes 1 on the active side; residual connections add a path whose local derivative is exactly 1.'
      }
    ],
    cards: [
      { q: 'Chain rule, and why it matters here', a: '$\\frac{d}{dx}f(g(x)) = f\'(g(x))g\'(x)$ — backpropagation is this product accumulated right-to-left with cached activations.' },
      { q: 'Why reverse-mode autodiff?', a: 'One scalar output, many parameters. Reverse mode costs ~one forward pass per output; forward mode costs one per input.' },
      { q: 'Gradient vs Hessian', a: 'Gradient = direction of steepest ascent (first derivatives). Hessian = curvature (second derivatives); its condition number governs how hard optimisation is.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.4 */
  ML.section({
    id: 'probability-basics', track: 'start', num: '0.4',
    title: 'Probability, from the ground',
    lede: 'Joint, marginal, conditional — three views of one table. Get them straight here and Bayes in §1.1 is a one-line consequence rather than a formula to memorise.',
    html: `
<h2><span class="sn">0.4.1</span> The objects</h2>
<p>A <b>random variable</b> is a quantity whose value is uncertain: a coin flip, tomorrow's demand, the next token. A <b>distribution</b> assigns probability across its possible values — non-negative, summing (or integrating) to one. That is the whole of the axioms you need.</p>
<p>With two variables you get one object and two views of it:</p>
<ul>
<li><b>Joint</b> $P(A, B)$ — the probability of both. The full table.</li>
<li><b>Marginal</b> $P(A) = \\sum_b P(A, b)$ — sum the table along a direction and one variable disappears.</li>
<li><b>Conditional</b> $P(A \\mid B) = P(A, B) / P(B)$ — keep only the row where $B$ happened and rescale it so it sums to one.</li>
</ul>
${H.key('A conditional probability is not a new quantity. It is the same table, restricted to a smaller world and renormalised.')}

${H.lab('joint', 'One table, three views', 'Drag the four joint probabilities. The marginals appear on the edges; the conditional strip shows what happens when you delete every outcome incompatible with the evidence and rescale. Watch independence appear exactly when the conditional matches the marginal.')}

<h2><span class="sn">0.4.2</span> Independence, and why it is a strong claim</h2>
<p>$A$ and $B$ are independent iff $P(A, B) = P(A)P(B)$, equivalently $P(A \\mid B) = P(A)$: learning $B$ tells you nothing about $A$. It is rare in real data and extremely convenient in models — Naive Bayes (§2.5) assumes it between features, and is a useful classifier <i>despite the assumption being false</i>, for reasons that section makes precise.</p>

<h2><span class="sn">0.4.3</span> Expectation and variance, in one line each</h2>
$$\\mathbb{E}[X] = \\sum_x x\\,p(x), \\qquad \\mathrm{Var}(X) = \\mathbb{E}[(X - \\mathbb{E}[X])^2] = \\mathbb{E}[X^2] - \\mathbb{E}[X]^2$$
<p>Expectation is the long-run average — the centre of mass of the distribution. Variance is the average squared distance from that centre; its square root, the standard deviation, is in the same units as $X$ and is the one to quote. §1.3 shows the asymmetry that runs through the entire site: expectation is linear <i>always</i>, variance adds only under independence.</p>

<h2><span class="sn">0.4.4</span> Discrete or continuous</h2>
<p>Discrete variables have a probability <b>mass</b> function: $p(x)$ is a probability. Continuous variables have a <b>density</b>: $p(x)$ is not a probability and can exceed 1 — only $\\int_a^b p(x)dx$ is a probability. This is why a Gaussian's peak height changes with its width, and why likelihoods of continuous data can be greater than one without anything being wrong.</p>

${H.probe([
      ['Define conditional probability without the formula.', 'Restrict attention to the world where the evidence is true, then rescale so the remaining outcomes sum to one.'],
      ['Can a density be greater than 1?', 'Yes — a density is not a probability. A uniform on $[0, 0.5]$ has density 2 everywhere on its support.']
    ], 'Confusing $P(A\\mid B)$ with $P(B\\mid A)$. That single confusion is the base-rate error in §1.1 and the most common probability mistake in interviews.')}`,
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
        why: 'Marginalisation is summation over the variable you want to eliminate — the law of total probability, which is exactly the denominator of Bayes.'
      },
      {
        q: 'Which is true of a continuous probability density $p(x)$?',
        options: ['It is always ≤ 1', 'It integrates to 1 and may exceed 1 pointwise', 'It equals $P(X = x)$', 'It must be symmetric'],
        answer: 1,
        why: 'Density is probability per unit length. Only integrals over intervals are probabilities; $P(X=x)=0$ for continuous $X$.'
      }
    ],
    cards: [
      { q: 'Conditional probability in one sentence', a: 'The same measure restricted to the world where the evidence holds, renormalised to sum to one.' },
      { q: 'Definition of independence', a: '$P(A,B)=P(A)P(B)$, equivalently $P(A\\mid B)=P(A)$ — the evidence changes nothing.' },
      { q: 'Variance, two forms', a: '$\\mathrm{Var}(X)=\\mathbb{E}[(X-\\mu)^2]=\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.5 */
  ML.section({
    id: 'python-toolkit', track: 'start', num: '0.5',
    title: 'The working toolkit: numpy, pandas, scikit-learn, PyTorch',
    lede: 'The four libraries that carry ninety per cent of practical work, with the idioms that matter and the three mistakes that quietly invalidate results.',
    html: `
<h2><span class="sn">0.5.1</span> numpy — everything is an array with a shape</h2>
${H.code(`import numpy as np

X = np.random.randn(1000, 8)          # 1000 examples, 8 features
w = np.random.randn(8)
y = X @ w + 0.1 * np.random.randn(1000)   # @ is matrix multiply

# vectorise: never loop over rows if a matrix op exists
mu, sd = X.mean(0), X.std(0)          # per-column statistics
Z = (X - mu) / sd                     # broadcasting: (1000,8) - (8,) works

# the least-squares solution, three ways
beta_normal = np.linalg.solve(X.T @ X, X.T @ y)     # fine, and fast
beta_lstsq  = np.linalg.lstsq(X, y, rcond=None)[0]  # numerically safer
beta_ridge  = np.linalg.solve(X.T @ X + 1e-2*np.eye(8), X.T @ y)`)}
<p>Two habits pay for themselves: annotate shapes in comments, and prefer <code>np.linalg.lstsq</code> or a QR/SVD-based solver to explicitly inverting $X^\\mathsf{T}X$ — the inverse is numerically fragile precisely when your features are collinear, which is the case you were worried about anyway.</p>

<h2><span class="sn">0.5.2</span> pandas — tables, and the leakage trap</h2>
${H.code(`import pandas as pd

df = pd.read_parquet("applications.parquet")
df["utilisation"] = df.balance / df.limit
df["age_days"] = (df.decision_date - df.opened_date).dt.days

# GOOD: aggregate with an explicit time boundary
hist = (df[df.decision_date < cutoff]
        .groupby("customer_id")["amount"].mean()
        .rename("mean_amount_before_cutoff"))

# WRONG: this aggregate sees the whole history, including the future
df["mean_amount"] = df.groupby("customer_id")["amount"].transform("mean")`)}
${H.flag('That last line is the single most common leakage bug in tabular work: a group aggregate computed over the full dataset lets each row see its own future. §2.11 has the full checklist.')}

<h2><span class="sn">0.5.3</span> scikit-learn — fit/transform, and why pipelines are not optional</h2>
${H.code(`from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([
    ("scale", StandardScaler()),          # fitted INSIDE each fold
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(auc.mean(), auc.std())`)}
<p>The rule the pipeline enforces: <b>every learned transformation — scaling, imputation, target encoding, binning — must be fitted on the training fold only</b>. Scale first and cross-validate second and your validation rows have already influenced the mean and standard deviation; the score you report is optimistic and the mechanism is invisible.</p>

<h2><span class="sn">0.5.4</span> PyTorch — the same three objects, on a GPU</h2>
${H.code(`import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.BCEWithLogitsLoss()      # logits in, not probabilities

for xb, yb in loader:
    opt.zero_grad(set_to_none=True)   # gradients accumulate by default
    logits = model(xb).squeeze(-1)
    loss = loss_fn(logits, yb.float())
    loss.backward()                   # reverse-mode autodiff (§0.3)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()`)}
${H.table(['Idiom', 'Why'], [
      ['<code>BCEWithLogitsLoss</code> not <code>Sigmoid</code> + <code>BCELoss</code>', 'Fuses the log-sum-exp; numerically stable at large logits (§1.10)'],
      ['<code>opt.zero_grad()</code> every step', 'PyTorch accumulates gradients; forgetting it silently sums minibatches'],
      ['<code>model.eval()</code> + <code>torch.no_grad()</code> at inference', 'Turns off dropout and batch-norm updates, and stops building the graph'],
      ['<code>clip_grad_norm_</code>', 'The cheapest insurance against a single bad batch producing NaN (§4.11)'],
      ['<code>AdamW</code> not <code>Adam(weight_decay=)</code>', 'Decoupled decay actually decays; see §3.5']
    ])}

<h2><span class="sn">0.5.5</span> The three mistakes</h2>
${H.checklist([
      '<b>Fitting a transformer outside the fold.</b> Scaling, imputation and encoding are learned parameters. Treat them as model parameters, because they are.',
      '<b>Shuffling time-structured data.</b> A random k-fold on a time series trains on the future to predict the past. Use out-of-time validation (§2.14).',
      '<b>Reporting the tuned score.</b> The validation set that chose your hyperparameters is no longer an unbiased estimate. Nest it, or hold out a final untouched set.'
    ])}`,
    quiz: [
      {
        q: 'Why wrap the scaler and the model in a Pipeline before cross-validating?',
        options: ['It is faster', 'So the scaler is re-fitted inside each training fold, preventing the validation rows from influencing it', 'It makes the model more accurate', 'Because sklearn requires it'],
        answer: 1,
        why: 'Any learned transformation fitted on all the data leaks information from the validation fold into training, inflating the score by an amount you cannot see.'
      },
      {
        q: 'You forget to call opt.zero_grad() in a PyTorch loop. What happens?',
        options: ['The model does not train at all', 'Gradients accumulate across batches, so each step uses a stale sum of gradients', 'Learning rate is ignored', 'It raises an exception'],
        answer: 1,
        why: 'PyTorch accumulates into .grad by design (useful for gradient accumulation). Forgetting to clear it makes every step an increasingly large, increasingly wrong sum.'
      }
    ],
    cards: [
      { q: 'The pipeline rule', a: 'Every learned transformation is fitted inside the training fold only — scaling, imputation, encoding, binning.' },
      { q: 'Why BCEWithLogitsLoss over Sigmoid+BCELoss', a: 'It fuses the sigmoid and the log using the log-sum-exp trick, so large logits do not overflow.' }
    ]
  });
})();
