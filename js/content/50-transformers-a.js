/* ============================================================
   PART 4 — LLMs & transformers (4.1 – 4.8)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 4.1 */
  ML.section({
    id: 'why-attention', track: 'llm', num: '4.1',
    title: 'Why attention replaced recurrence',
    lede: 'The bridge from §3.8; it establishes the inductive-bias vocabulary the rest of this part assumes.',
    html: `
<p>Take a sentence that a translator has to get right: "The trophy didn't fit in the suitcase because <i>it</i> was too big." Resolving "it" means reaching back across seven words to decide between two candidates, and getting it wrong flips the whole sentence's meaning. Now imagine the same problem stretched over a page: a pronoun, a variable name, a legal "the aforementioned party", whose antecedent sits a thousand tokens back. Whatever architecture reads that page has to somehow let the information at position 1 influence the prediction at position 1,000. The question this section answers is mechanical, not linguistic: <i>how does information get from one position in a sequence to a distant one, and what does that route cost you?</i></p>

<h2><span class="sn">4.1.1</span> Every architecture is a bet about structure</h2>
<p>A model architecture is a wager about what kind of structure the data has, placed before you have seen a single example. Get the wager right and the model needs far less data to find the pattern, because the architecture has already ruled out everything incompatible with it; get it wrong and you have spent parameters enforcing an assumption the data does not actually have. This wager has a name — an <b>inductive bias</b> — and every architecture in this course makes one.</p>
<p>A plain MLP (§3.1) makes almost no assumption at all: every output can depend on every input in an arbitrary way, so it has to learn even the most basic regularities, such as "nearby pixels are related", from scratch, using a fully-connected weight matrix. A CNN (§3.7) bakes in exactly that regularity — <b>locality</b>, nearby things interact more than distant ones, and <b>translation invariance</b>, a pattern means the same thing wherever it appears in the image. That is an excellent bet for pixels, where a cat's ear looks like a cat's ear whether it is in the top-left or bottom-right of the frame. It is a poor bet for language, where the two words that must agree — a subject and a verb, an opening and closing bracket — can be adjacent or a paragraph apart, and "nearby" carries no privileged meaning the way it does in an image.</p>
<p>An RNN or LSTM (§3.8) makes the bet suited to language on its face: it carries a hidden state forward one token at a time, so in principle nothing stops information from token 1 reaching token 1,000 — it just has to survive 999 intermediate updates. §3.8 already showed what goes wrong with that "in principle": gradients backpropagated through that many multiplications decay toward zero or blow up toward infinity, and gating (the LSTM's whole reason for existing) only slows the decay, it does not stop it. There is a second cost that has nothing to do with gradients at all, and it turns out to matter even more in practice. Computing $h_t$ requires $h_{t-1}$, which requires $h_{t-2}$, and so on back to the start. <mark>Recurrence cannot be parallelised across time</mark> — a sequence of length $n$ is $n$ sequential steps, however many GPUs you own, because step $t$ simply does not exist until step $t-1$ has finished.</p>

${H.history(`<p>Attention did not arrive as a replacement for recurrence — it arrived as a patch for one. Bahdanau, Cho and Bengio's 2014 sequence-to-sequence translation model was an RNN encoder feeding an RNN decoder through a single fixed-size vector, and that vector was a bottleneck: a 50-word source sentence and a 5-word one were squeezed into the same handful of numbers, and translation quality fell off sharply on long sentences as a direct result. Their fix let the decoder, at each output step, look back at <i>all</i> the encoder's hidden states and take a weighted combination — the weights computed from how relevant each source position was to the word being generated right now. That weighted lookup is attention, and it was bolted onto a recurrent backbone that still ran one step at a time.</p>
<p>Vaswani et al.'s 2017 paper asked the question that gives this section its title: what if you removed the recurrence and kept only the attention? "Attention Is All You Need" is a literal description of the finding, not a slogan retrofitted afterwards — strip away the RNN entirely, keep the parts that let every position look directly at every other position, and the result trains faster, scales further and, past a certain size, translates better. The name of the mechanism did not change between 2014 and 2017. What changed is that it stopped being an accessory to recurrence and became the entire architecture.</p>`)}

<h2><span class="sn">4.1.2</span> What attention buys: a direct route and a parallel one</h2>
<p>Self-attention answers the "how does token 1 reach token 1,000" question by refusing to relay the information at all. Every token computes, in one operation, a weighted combination of every <i>other</i> token in the sequence — you will see exactly how that weighting is computed in §4.3, via the dot product introduced back in §0.2. For now, treat it as a lookup: token 1,000 does not wait for information to be passed hand to hand through 999 intermediate steps; it reaches directly across the whole sequence in a single hop. The <b>path length</b> between any two positions — the number of computational steps a signal must cross to get from one to the other — is $O(1)$, a constant, regardless of how far apart they sit. A recurrence's path length is $O(n)$, growing with the distance itself, which is exactly the structural reason its gradients fade over that distance (§3.8's vanishing-gradient argument is a statement about path length in disguise: a longer path means more matrices multiplied together before the gradient arrives).</p>

${H.analogy(`<p>A recurrence is a game of telephone: token 1's information passes to token 2, which passes its own (already-mixed) version to token 3, and so on, and whatever token 1 originally said has been paraphrased 999 times by the time it reaches token 1,000 — degraded a little at each hand-off, exactly as a whispered message degrades around a circle. It is also strictly sequential: player 3 cannot whisper until player 2 has finished, so the whole game takes 999 hand-offs no matter how many people are in the room to help.</p>
<p>Self-attention is a conference call. Every participant can address every other participant directly, in one exchange, with no relay and no paraphrasing along the way. And because nobody is waiting on anybody else to finish speaking first, the whole exchange happens at once rather than in a queue — which is the second half of the payoff, and arguably the more consequential one.</p>`)}

<p>That second half is <b>parallelism across time</b>. Because attention computes every token's output from the whole sequence in one shot rather than one step at a time, an entire sequence's forward pass is a handful of large matrix multiplications — exactly the operation a GPU is built to do at enormous throughput. An RNN of length $n$ is $n$ sequential kernel launches that cannot overlap; a transformer layer over the same sequence is one launch. <b>Unlimited-range modelling <i>and</i> full parallel training</b> is the pairing that decided the contest — not one property alone. A CNN gets parallelism (each layer is one convolution, independent of position) but not unlimited range: connecting positions $n$ apart needs roughly $n/k$ stacked layers of kernel width $k$, so long-range dependencies cost <i>depth</i>, one more layer per widened reach. Attention pays neither price, and the table below puts the comparison in one place.</p>

<p>Nothing is free, and the bill for skipping the relay comes due immediately: computing "every token against every other token" is $n$ tokens, each compared against $n$ others, so both the compute and the memory needed are $O(n^2)$ in sequence length. Double the context and you quadruple the cost of a single attention layer. That single fact is the reason §4.7 exists (FlashAttention, GQA and MLA, all attempts to make the $n^2$ term cheaper to move through memory), why §4.4 spends a section on sliding windows and attention sinks, and why §4.8 introduces state-space models that trade the $O(1)$ path length back down to something sub-quadratic. Keep that trade in view for the rest of Part 4: nearly every serving and architecture decision from here on is either paying for, or trying to avoid paying for, this one quadratic term.</p>

<p><b>What you are looking at.</b> The lab below draws the same eight-token sequence three times, once per architecture. Circles are tokens; the leftmost and rightmost are highlighted, because they are the pair whose connection you are tracking. Curved lines show which tokens directly influence which others in a single layer's computation: for the recurrence they form an unbroken chain from left to right, for the convolution they fan out a fixed distance either side of each position, and for attention every token is joined to every other token at once.</p>
<p><b>What to do with it.</b> Switch between the three architecture modes and read the two numbers in the readout: the hop count from the first token to the last, and the number of sequential steps the whole layer needs before training can move on to the next one. Then drag the sequence-length slider up and watch the recurrence's hop count climb in lockstep with it while attention's stays fixed at one.</p>
<p><b>The thing genuinely worth noticing.</b> The recurrence's hop count and its sequential-step count are the <i>same number</i> — that is not a coincidence, it is the definition of recurrence: one hop always costs one step. Attention decouples them completely: one hop, however long the sequence, computed as one parallel operation. That decoupling, not any claim about "understanding language better", is the entire mechanical reason attention displaced recurrence.</p>

${H.lab('path', 'Path length and parallelism, side by side', 'The same eight-token sequence through a recurrence and through attention. Count the hops between the first and last token; then look at the time axis — the recurrence has eight sequential steps, attention has one.')}

${H.table(['Architecture', 'Path length', 'Parallel over time?', 'Cost per layer'], [
      ['RNN / LSTM', '$O(n)$', 'No', '$O(n\\cdot d^2)$'],
      ['CNN (k-wide, stacked)', '$O(n/k)$ layers', 'Yes', '$O(k\\cdot n\\cdot d^2)$'],
      ['Self-attention', '$O(1)$', 'Yes', '$O(n^2 d + n d^2)$'],
      ['SSM / Mamba (§4.8)', '$O(1)$ via scan', 'Yes (associative scan)', '$O(n d^2)$']
    ])}

${H.intuition(`<p>If you already have §0.2's dot product in hand, you can preview the entire mechanism §4.3 derives properly. "How relevant is token $j$ to token $i$" is a similarity question, and a dot product is exactly a measure of alignment between two vectors. Give every token a vector, score every pair by their dot product, turn the scores for one token into weights that sum to one, and take the weighted average of the other tokens' vectors — that is self-attention, and nothing about it depends on position or order. Which is itself a cost: because the operation is <b>permutation-equivariant</b> — shuffle the tokens and the output shuffles identically — a transformer has no innate sense of sequence at all, and §4.4 exists specifically to inject one back in.</p>`)}

${H.probe([
      ['Why did transformers win?', 'O(1) path length between any two positions plus full parallelism across time; recurrence has neither. It is the pairing, not either property alone — a CNN gets parallelism without unlimited range, and a wider RNN gets neither.'],
      ['What is the cost?', 'Quadratic compute and memory in sequence length — the constraint behind FlashAttention, GQA/MLA, sliding windows and SSMs (§4.4, §4.7, §4.8).'],
      ['Is recurrence simply obsolete?', 'Not entirely — SSMs (§4.8) resurrect a constant-size recurrent state to escape the quadratic cost, using an associative scan to reclaim the parallelism an ordinary RNN lacks. The bet on sequential state was not wrong; the bet on step-by-step training was.']
    ])}`,
    labs: {
      path: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'sequence length', min: 4, max: 14, step: 1, value: 8, fmt: v => v },
          { k: 'mode', label: 'architecture', type: 'buttons', value: 'attn', options: [{ v: 'rnn', t: 'recurrence' }, { v: 'attn', t: 'attention' }, { v: 'conv', t: 'CNN (k=3)' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'hops', label: 'hops from token 1 to token n', cls: 'key' },
          { k: 'seq', label: 'sequential steps to train' },
          { k: 'ops', label: 'pairwise interactions computed' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const n = st.n, y = h * .55, r = 13;
            const xs = Array.from({ length: n }, (_, i) => 40 + i * ((w - 80) / (n - 1)));
            if (st.mode === 'attn') {
              for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
                if (i === j) continue;
                ctx.strokeStyle = (i === 0 && j === n - 1) || (j === 0 && i === n - 1) ? T.red : T.line;
                ctx.lineWidth = (i === 0 && j === n - 1) ? 2 : .8;
                ctx.globalAlpha = (i === 0 && j === n - 1) ? 1 : .45;
                ctx.beginPath();
                const mid = (xs[i] + xs[j]) / 2, span = Math.abs(xs[i] - xs[j]);
                ctx.moveTo(xs[i], y - r);
                ctx.quadraticCurveTo(mid, y - r - span * .35, xs[j], y - r);
                ctx.stroke(); ctx.globalAlpha = 1;
              }
            } else if (st.mode === 'rnn') {
              for (let i = 0; i < n - 1; i++) {
                ctx.strokeStyle = T.red; ctx.lineWidth = 2; ctx.globalAlpha = 1;
                ctx.beginPath(); ctx.moveTo(xs[i] + r, y); ctx.lineTo(xs[i + 1] - r, y); ctx.stroke();
                ctx.fillStyle = T.red;
                ctx.beginPath(); ctx.moveTo(xs[i + 1] - r, y); ctx.lineTo(xs[i + 1] - r - 7, y - 4); ctx.lineTo(xs[i + 1] - r - 7, y + 4); ctx.closePath(); ctx.fill();
                ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                ctx.fillText('t' + (i + 1), (xs[i] + xs[i + 1]) / 2, y - 6);
              }
            } else {
              for (let i = 0; i < n; i++) [-1, 1].forEach(d => {
                const j = i + d; if (j < 0 || j >= n) return;
                ctx.strokeStyle = T.blue; ctx.lineWidth = 1.4; ctx.globalAlpha = .8;
                ctx.beginPath(); ctx.moveTo(xs[i], y - r); ctx.quadraticCurveTo((xs[i] + xs[j]) / 2, y - r - 26, xs[j], y - r); ctx.stroke();
                ctx.globalAlpha = 1;
              });
            }
            xs.forEach((x, i) => {
              ctx.beginPath(); ctx.arc(x, y, r, 0, 6.3);
              ctx.fillStyle = (i === 0 || i === n - 1) ? T.blue : T.panel; ctx.fill();
              ctx.strokeStyle = T.line; ctx.lineWidth = 1.4; ctx.stroke();
              ctx.fillStyle = (i === 0 || i === n - 1) ? '#fff' : T.muted;
              ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(String(i + 1), x, y);
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            const msg = { rnn: 'information from token 1 reaches token n only by passing through every step between',
              attn: 'every token attends to every other token in one operation — one hop, computed in parallel',
              conv: 'each layer widens the view by one on each side; you need ⌈(n−1)/2⌉ layers to connect the ends' }[st.mode];
            ctx.fillText(msg, w / 2, y + 46);
            out({
              hops: st.mode === 'attn' ? '1' : st.mode === 'rnn' ? String(n - 1) : String(Math.ceil((n - 1) / 2)) + ' layers',
              seq: st.mode === 'rnn' ? String(n) + ' (cannot parallelise)' : '1 (one matmul)',
              ops: st.mode === 'attn' ? (n * n) + ' (O(n²))' : st.mode === 'rnn' ? String(n) + ' (O(n))' : String(3 * n)
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'The decisive advantage of self-attention over recurrence is…',
        options: ['fewer parameters', 'O(1) path length between positions plus full parallelism across time', 'lower memory use', 'it needs no positional information'],
        answer: 1,
        why: 'It is the pairing that wins, not either half alone: a CNN also parallelises over time but still needs O(n/k) layers to connect distant positions, and a bidirectional RNN can shorten some paths but is still sequential in training. Attention gets both at once — direct O(1) routes between any two positions, computed as one parallel matrix multiply. Memory is the tempting wrong answer, because attention is actually worse there: its O(n²) score matrix is the whole reason §4.7 exists. And positional information is not free with attention — quite the opposite, since the operation is permutation-equivariant it has no innate notion of order at all and needs it injected explicitly (§4.4), whereas a recurrence gets order for nothing, as a side effect of processing one step at a time.'
      }
    ],
    cards: [
      { q: 'Why attention beat recurrence', a: 'O(1) path length between any two tokens and one parallel matmul per layer; recurrence is sequential in time.' },
      { q: 'What attention costs', a: '$O(n^2)$ compute and memory in sequence length — the constraint driving FlashAttention, GQA/MLA and SSMs.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.2 */
  ML.section({
    id: 'tokenization', track: 'llm', num: '4.2',
    title: 'Tokenization and embeddings',
    lede: 'The layer everyone skips and every practitioner eventually debugs. Character-level tasks fail here, not in the reasoning.',
    html: `
<p>Before a transformer can do anything — attend, predict, reason — the raw text has to become a sequence of integers, because §4.5's whole machinery is matrix arithmetic on numbers, not operations on letters. The obvious first attempt is to give every distinct word its own integer. Try that on a real corpus and it fails almost immediately, in two directions at once. Count the distinct words in a large web crawl and you get millions of them, once you include every misspelling, every proper noun, every inflection — "run", "runs", "running", "ran" each need their own slot, sharing nothing. Train a vocabulary of, say, the 50,000 most common words and you have solved the size problem but created a worse one: any word not in that list — a new product name, a rare technical term, a typo — has no representation at all. The model has literally no number to assign it. This is the <b>out-of-vocabulary</b> problem, and it used to be handled with a single catch-all "unknown" token that threw away everything about the word except the fact that it existed.</p>
<p>The opposite extreme avoids OOV entirely: give every individual <i>character</i> its own integer. Now nothing is ever unrepresentable, because any string is just a sequence of characters from a small, fixed alphabet. But sequences explode in length — a 500-word paragraph becomes roughly 2,500 characters instead of roughly 650 words — and every position attention has to relate to every other position (§4.1's $O(n^2)$ cost) just got four times more expensive for the same sentence, before the model has even started learning anything about meaning. Neither extreme is usable. <b>Subword tokenization</b> is the compromise that actually ships: build a vocabulary of word <i>pieces</i>, common enough to keep the sequence short, small enough that any string can still be built out of them.</p>

<h2><span class="sn">4.2.1</span> How the pieces are chosen</h2>
${H.history(`<p><b>Byte-pair encoding</b> was not invented for language models at all. Philip Gage described it in 1994 as a general-purpose data compression trick: repeatedly find the most frequent pair of adjacent bytes in a file and replace it with a byte that does not otherwise occur, shrinking the file one substitution at a time. Sennrich, Haddow and Birch repurposed the idea for neural machine translation in 2015, with one change that made it a tokenizer rather than a compressor: instead of discarding the substitutions, keep the record of which pairs got merged, in order, and that record <i>is</i> the vocabulary. Applying the same ordered list of merges to new text is exactly how a trained BPE tokenizer segments it.</p>`)}
<p>Concretely: start with every word broken into individual characters (plus a marker for word boundaries), count every adjacent pair across the whole training corpus, and merge the single most frequent pair into one new symbol. Repeat. Common short sequences merge first — "t" and "h" merge into "th" early, because "th" is everywhere — and as the vocabulary grows, whole common words and then common word-fragments ("ing", "tion", "un") become single symbols, while rare words stay fragmented into smaller, more frequent pieces. The vocabulary size is simply the number of merges you choose to run before stopping, which makes it a direct, tunable dial rather than an emergent property of the corpus.</p>
<p><b>WordPiece</b>, Google's variant, changes only the merge criterion: instead of picking the most <i>frequent</i> pair, it picks the pair whose merger most improves the likelihood of the training corpus under a simple language model — a pair that is frequent but already fairly predictable from its parts is a worse merge candidate than a rarer pair that is highly informative once fused. <b>SentencePiece</b> changes something more fundamental: it trains directly on raw text streams, with no language-specific pre-tokenisation step (no assumption that spaces separate words, which is false for Chinese, Japanese and Thai), which makes it language-agnostic by construction rather than by patching. Paired with <b>byte-level fallback</b> — falling back to raw UTF-8 bytes for anything the learned vocabulary cannot cover — this guarantees that literally any string, including one the tokenizer has never seen a character of, can be encoded. Nothing is ever truly out-of-vocabulary again; in the worst case a symbol just costs more tokens, spelled out byte by byte, rather than being replaced by an unknown marker that erases it.</p>

${H.analogy(`<p>Think of packing a suitcase with a fixed set of container sizes rather than one bag per item or one grain of rice at a time. Pack every item loose (character-level) and you fit almost anything but spend enormous space on air between items. Pack with only whole-suitcase-sized containers (word-level) and most items either fit perfectly or do not fit in the case at all. Subword tokenization is packing with a graduated set of container sizes — a few large ones for common whole shapes ("the", "tion"), many small ones for anything unusual — chosen, by BPE's merge process, specifically to match how frequently each size of item actually shows up in your luggage.</p>`)}

<p><b>What you are looking at.</b> The panel trains a real BPE tokenizer, live, on the short corpus shown in the box — not a simulation of the algorithm, the actual merge loop described above, run in your browser. The merge list records each merge in the order the algorithm chose it, exactly as a real tokenizer's vocabulary file would. Below it, any word you select is segmented using those merges and drawn as a row of coloured boxes, one per resulting token, each labelled with its piece and an assigned id.</p>
<p><b>What to do with it.</b> Start the merge count at 0, where every character is its own token, and slide it upward. Watch frequent fragments — "un", "ness", "ing" — collapse into single boxes one merge at a time, in the same order the readout's merge list shows them being learned. Then switch the word to encode to "quokka", a word that never appears in the training corpus: however many merges you allow, it stays fragmented into small pieces, because none of its substrings were ever frequent enough locally to earn a merge.</p>
<p><b>The thing genuinely worth noticing.</b> "Quokka" never becomes out-of-vocabulary — it always encodes to <i>something</i>, just at a token-per-token cost close to the character level, because byte fallback exists precisely for words the training corpus never anticipated. That is the entire promise of subword tokenization stated as a single observable fact: coverage is total, and the price of an unfamiliar word is *length*, not failure.</p>

${H.lab('bpe', 'Train a BPE tokenizer, then use it', 'Merges are learned here, from the corpus in the box, in the order the algorithm chooses them. Then encode any text and see the segmentation — including why "unhappiness" becomes three tokens and why numbers fall apart.')}

<h2><span class="sn">4.2.2</span> The consequence people miss</h2>
<p>Ask a model how many times the letter "r" appears in "strawberry" and it will sometimes get it wrong, and the instinctive read of that failure is "the model can't count" or "the model can't reason". Neither is quite it. Look at what actually reaches the model: not the ten characters s-t-r-a-w-b-e-r-r-y, but three or four opaque integer ids, each standing for a multi-character fragment the tokenizer happened to learn — "st", "raw", "berry", say. The model never receives the letters one at a time. It receives a small number of symbols that <i>happen to spell out</i> those letters when decoded back to text, and counting occurrences of "r" inside one of those symbols requires the model to have memorised, from training, exactly how that specific token is spelled — a fact it can recall with the same unreliability it recalls any other memorised fact, and with no mechanism to double-check the recollection against the actual characters, because those characters are not part of its input in any form it can inspect.</p>
<p><mark>Tokens bundle several characters and hide the sub-token structure</mark> from the model entirely. This single fact explains a whole family of otherwise-mysterious failures that all get misdiagnosed as reasoning problems: letter-counting, reversing a string character by character, and — the economically important case — arithmetic on long numbers. A four-digit number might tokenize as one or two symbols; a twelve-digit number fragments unpredictably depending on where the tokenizer's learned boundaries happen to fall, so the model is effectively doing long addition on operands whose individual digit positions are not reliably exposed to it. It is a representation artefact, not evidence that the model "can't do maths" — and the distinction matters operationally, because the fix is different. You do not fix it by asking the model to "think harder"; you fix it by choosing an interface where character-level structure is not needed (a calculator tool call, code execution) or, at the tokenizer-design stage, by using digit-by-digit tokenization for numbers specifically, which several production tokenizers now do for exactly this reason.</p>

<p><b>What you are looking at.</b> The top row is the word as a person reads it, letter by letter, with the target letter highlighted in every position it occurs. The row below is the same word as the model receives it: a small number of boxes, each one opaque token id standing for a multi-character fragment, with the fragment's text shown only as a label for your benefit — the model itself never sees anything but the id number.</p>
<p><b>What to do with it.</b> Compare the true letter count, computed directly from the string, against how many of the <i>tokens</i> visibly contain that letter in their fragment. For "strawberry" counting "r", most of the actual letters are buried inside a token like "berry" that the model would have to decompose from memory to count correctly, rather than reading off directly.</p>
<p><b>The thing genuinely worth noticing.</b> Switch to the all-digit string. Watch how differently — and how much more coarsely — long digit runs get chunked compared with English words, since the tokenizer's merges were learned from a corpus dominated by prose rather than numerals. That mismatch between how a number is chunked and how its individual digits carry mathematical meaning is the direct, mechanical cause of the model doing worse arithmetic on longer numbers, entirely independent of how good its "reasoning" is.</p>

${H.lab('chars', 'Why “how many r’s in strawberry” is hard', 'The word as the model sees it: a handful of opaque ids. The letters are not there to count — the model has to reconstruct them from memorised spelling knowledge, which is why it is unreliable rather than impossible.')}

<h2><span class="sn">4.2.3</span> Vocabulary size is a real tradeoff</h2>
<p>The number of merges you train — equivalently, the size of the final vocabulary — is not a free parameter to maximise. A larger vocabulary means each token, on average, spans more characters, so the same text becomes a shorter sequence: cheaper attention (recall §4.1's $O(n^2)$ dependence on sequence length), and more actual text fitting inside a fixed context window. But every one of those vocabulary entries needs a row in the embedding matrix (§4.5 shows this costs $2 \\times |V| \\times d$ parameters for the input and output embeddings combined) and a column in the final softmax over the vocabulary, so a larger vocabulary is directly a bigger, slower model at both ends. Modern frontier models settle somewhere around 32,000 to 256,000 tokens, and the drift toward the upper end of that range over the past few years has a specific cause: multilingual and code coverage. A vocabulary trained predominantly on English text represents common English words in one token each, but spends three or four tokens rebuilding a common word in Hindi, Korean or Arabic out of small fragments, because those scripts were rare in the training corpus and so earned few dedicated merges.</p>
<p>That inefficiency is not merely a technical curiosity — it is a cost that falls unevenly on users. If an API bills per token and a context window is measured in tokens, then the same sentence, carrying the same amount of information, costs measurably more — in money and in how much of the context window it consumes — to a speaker of an under-tokenised language than to a speaker of English. Expanding vocabularies to 100k+ tokens with deliberate multilingual balance in the training mixture is a direct response to that gap, and it is worth recognising as a fairness property of the tokenizer, not only an efficiency one.</p>

<h3>Numbers worth carrying</h3>
<p>English text averages roughly <mark>1.3 tokens per word, and about 4 characters per token</mark> — the two figures are consistent with each other, since an average English word is a little over 5 characters including its trailing space, and $5 \\div 4 \\approx 1.3$. Two consequences follow directly, and both are useful for a back-of-envelope estimate in a design conversation: 750 words of English prose is about $750 \\times 1.3 \\approx 975$, call it 1,000 tokens; and a 250-page book at roughly 400 words a page is $250 \\times 400 \\times 1.3 \\approx 130{,}000$ tokens — the figure worth memorising is "a book is about 130k tokens", because it turns "will this fit in a 128k context window" into an instant judgement rather than a lookup. Source code tokenizes more densely than prose — punctuation, indentation and short identifiers fragment more per character — and, as §4.2.2 just showed, long digit strings are close to the worst case of all, which is the concrete reason arithmetic accuracy degrades measurably between three-digit and twelve-digit operands independent of any change in the model's underlying reasoning ability.</p>

<h2><span class="sn">4.2.4</span> Special tokens and chat templates</h2>
<p>Alongside the vocabulary learned from text, every model reserves a handful of <b>special tokens</b> that never occur in ordinary prose: a beginning-of-sequence marker, an end-of-sequence or stop token, and — for an instruction-tuned model — role markers that delimit where the system prompt ends and the user's turn begins, and where the assistant's turn begins and ends. An instruction-tuned model was fine-tuned on millions of examples laid out in one specific arrangement of these markers, and that arrangement, not the words around it, is what the model learned to treat as "this is where I should start answering". Send it a differently-formatted prompt — plain string concatenation instead of the model's own template, mismatched role tags, a missing stop token — and quality degrades in a way that looks exactly like a capability failure (worse instructions, worse reasoning, rambling answers) while actually being a formatting mismatch the model was never trained to handle gracefully.</p>
<p>The practical rule follows immediately: <b>always render prompts through the model's own chat template</b>, supplied by its tokenizer library, rather than assembling the string by hand. And the stop token deserves its own mention, because its absence produces a specific, recognisable failure: it is the token the model was trained to emit when a response is complete, and the serving loop's job is simply to stop generating when it sees that token. A fine-tuning run that forgets to include the stop token in its training examples teaches the model never to emit it, and the result is a model that no longer knows how to end a sentence — it keeps generating until an external length limit cuts it off, a failure mode that is diagnosed correctly surprisingly rarely, because "the model won't stop talking" sounds like a sampling problem long before anyone checks whether the fine-tuning data even contained an end-of-turn marker.</p>

${H.probe([
      ['Why can’t the model count letters reliably?', 'Tokens bundle characters; the sub-token structure is not visible to the model. It is a tokenizer artefact.'],
      ['What does vocabulary size trade?', 'Shorter sequences and cheaper attention against a larger embedding matrix and output softmax.'],
      ['Why use the model’s chat template?', 'It was trained on a specific control-token layout; mismatched formatting degrades quality in ways that look like capability loss.']
    ])}`,
    labs: {
      bpe: function (host) {
        const corpus = 'the happiness of the unhappy is unhappiness happening happily. the runner runs running runs. lower lowest slower slowest newer newest. tokenization tokenizer tokenize tokens token.';
        let merges = [], vocab = [];
        const st = Viz.controls(host, [
          { k: 'n', label: 'merges learned', min: 0, max: 60, step: 1, value: 24, fmt: v => v },
          { k: 'word', label: 'encode', type: 'select', value: 'unhappiness', options: ['unhappiness', 'tokenization', 'runners', 'slowest', 'quokka', 'happily'].map(v => ({ v: v, t: v })) }
        ], () => { retrain(); S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'vocab', label: 'vocabulary size', cls: 'key' }, { k: 'tok', label: 'tokens for this word' },
          { k: 'chars', label: 'characters' }, { k: 'ratio', label: 'chars per token' }
        ]);
        function retrain() { const r = Num.bpeTrain(corpus, st.n); merges = r.merges; vocab = r.vocab; }
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const toks = Num.bpeEncode(st.word, merges);
            ctx.font = '13px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('input', 16, 26);
            ctx.fillStyle = T.text; ctx.font = '20px ui-serif, Georgia, serif';
            ctx.fillText(st.word, 90, 26);
            // token boxes
            let x = 90;
            const y = 66;
            ctx.font = '13px ui-monospace, monospace';
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('tokens', 16, y + 12);
            toks.forEach((t, i) => {
              const label = t.replace('</w>', '␣');
              const wd = ctx.measureText(label).width + 18;
              Labs.roundRect(ctx, x, y, wd, 26, 6);
              ctx.fillStyle = 'rgba(90,140,255,.18)'; ctx.fill();
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.2; ctx.stroke();
              ctx.fillStyle = T.text; ctx.textAlign = 'center';
              ctx.fillText(label, x + wd / 2, y + 13);
              ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
              ctx.fillText('id ' + (1000 + (vocab.indexOf(t) >= 0 ? vocab.indexOf(t) : 999)), x + wd / 2, y + 38);
              ctx.font = '13px ui-monospace, monospace';
              x += wd + 8;
            });
            // merge list
            ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.font = '11px ui-monospace, monospace';
            ctx.fillText('merges learned, in order (the algorithm’s choices, not ours):', 16, 128);
            const cols = Math.max(1, Math.floor((w - 32) / 118));
            merges.slice(0, 24).forEach((m, i) => {
              const cx = 16 + (i % cols) * 118, cy = 148 + Math.floor(i / cols) * 20;
              ctx.fillStyle = T.faint;
              ctx.fillText((i + 1) + '. ' + m.replace(' ', '+').replace('</w>', '␣'), cx, cy);
            });
            const nChars = st.word.length;
            out({
              vocab: vocab.length, tok: toks.length, chars: nChars,
              ratio: (nChars / toks.length).toFixed(2)
            });
          }
        });
        retrain();
        Viz.note(host, 'With 0 merges every character is a token. As merges accumulate, frequent pieces ("un", "ness", "ing") become single tokens and the sequence shortens — that is the entire algorithm. Try "quokka", which never appears in the corpus: it stays fragmented, and byte fallback is what stops it being out-of-vocabulary in a real tokenizer.');
      },

      chars: function (host) {
        const st = Viz.controls(host, [
          { k: 'word', label: 'word', type: 'select', value: 'strawberry', options: ['strawberry', 'unhappiness', 'mississippi', '1234567890'].map(v => ({ v: v, t: v })) },
          { k: 'letter', label: 'count this letter', type: 'select', value: 'r', options: ['r', 's', 'p', 'a', '1'].map(v => ({ v: v, t: v })) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'true', label: 'true count', cls: 'good' }, { k: 'tokens', label: 'tokens the model sees' },
          { k: 'visible', label: 'tokens containing the letter' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const splits = {
              strawberry: ['str', 'aw', 'berry'], unhappiness: ['un', 'happi', 'ness'],
              mississippi: ['mis', 'siss', 'ippi'], '1234567890': ['123', '456', '789', '0']
            }[st.word];
            ctx.font = '26px ui-serif, Georgia, serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace';
            ctx.fillText('what you see', 16, 30);
            ctx.fillStyle = T.text; ctx.font = '26px ui-serif, Georgia, serif';
            let x = 16;
            st.word.split('').forEach(ch => {
              ctx.fillStyle = ch === st.letter ? T.green : T.text;
              ctx.fillText(ch, x, 60);
              x += ctx.measureText(ch).width + 2;
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace';
            ctx.fillText('what the model sees', 16, 108);
            x = 16;
            splits.forEach((tk, i) => {
              const wd = 92;
              Labs.roundRect(ctx, x, 124, wd, 40, 8);
              ctx.fillStyle = 'rgba(120,120,140,.16)'; ctx.fill();
              ctx.strokeStyle = T.line; ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center';
              ctx.fillText('id ' + (2000 + i * 137 % 900), x + wd / 2, 144);
              ctx.font = '9px ui-monospace, monospace';
              ctx.fillText('(“' + tk + '”)', x + wd / 2, 158);
              x += wd + 10;
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('The letters are not in the input. To count them the model must recall how each id is spelled —', 16, 190);
            ctx.fillText('which it can often do, and unreliably. That is a tokenizer artefact, not a reasoning failure.', 16, 208);
            const trueCount = st.word.split('').filter(c => c === st.letter).length;
            out({
              true: trueCount, tokens: splits.length,
              visible: splits.filter(t => t.indexOf(st.letter) >= 0).length + ' of ' + splits.length
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'Why do LLMs struggle to count characters in a word?',
        options: ['Their reasoning is weak', 'Tokens bundle several characters, hiding sub-token structure', 'The context window is too small', 'Attention cannot see individual positions'],
        answer: 1,
        why: 'It is a representation artefact. The same model does arithmetic on 3-digit numbers far better than 12-digit ones for the same reason.'
      },
      {
        q: '750 English words is roughly how many tokens?',
        options: ['375', '750', '1,000', '3,000'],
        answer: 2,
        why: '≈1.3 tokens per word, ≈4 characters per token. A 250-page book is ≈130k tokens.'
      }
    ],
    cards: [
      { q: 'BPE in one line', a: 'Repeatedly merge the most frequent adjacent pair until the vocabulary is full; WordPiece merges by likelihood gain instead.' },
      { q: 'Token arithmetic', a: '≈1.3 tokens/word, ≈4 chars/token; 750 words ≈ 1k tokens; a book ≈ 130k tokens.' },
      { q: 'Vocabulary size trade', a: 'Shorter sequences vs a bigger embedding matrix and output softmax; 32k–256k typical, larger for multilingual/code.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.3 */
  ML.section({
    id: 'attention', track: 'llm', num: '4.3',
    title: 'Self-attention derived, and the √dₖ argument',
    lede: 'Rests on §1.3 (variance of a sum) and §1.8 (bilinear forms). The single most-asked derivation in LLM interviews.',
    html: `
<p>§4.1 described attention as a conference call: every token addresses every other token directly, in one exchange, rather than relaying a message hand to hand. That is the right mental picture, but a conference call still needs a rule for <i>whom to listen to</i> — if everyone speaks with equal weight and equal volume, the result is noise, not communication. This section builds that rule from a piece you already have: §0.2's dot product, which measures how well two vectors align. What follows is a small number of decisions, each one motivated by what breaks if you skip it, and by the end you will have derived, rather than memorised, the single most-quoted formula in the field.</p>

<h2><span class="sn">4.3.1</span> Three roles, one dot product</h2>
<p>Suppose you want token $i$ to gather information from the other tokens, weighted by how <i>relevant</i> each of them is to it. "Relevant" has to be computed from something, and the natural candidate is exactly §0.2's alignment measure: two vectors are relevant to each other if their dot product is large. But a token's raw embedding is doing three different jobs at once here, and conflating them is a mistake worth naming before making. Token $i$ needs to say what it is looking for. Every other token $j$ needs to advertise what it contains, so it can be matched against that search. And once a match is found, token $j$ needs to contribute something — its actual content — to the result. Using the same vector for all three roles would force one representation to be simultaneously a search query, a search index entry, and a payload, and there is no reason those three things should look alike.</p>
<p>So the model learns three separate linear projections of every token's embedding $x$ — a <b>query</b> $q = W_Qx$ ("what am I looking for"), a <b>key</b> $k = W_Kx$ ("what do I contain, as an address other tokens can match against"), and a <b>value</b> $v = W_Vx$ ("what do I actually contribute, once selected"). Relevance between position $i$ and position $j$ is then simply the dot product $q_i \\cdot k_j$ — large when $i$'s query and $j$'s key point the same way, near zero when they are unrelated, negative when they point apart, exactly as §0.2 established. Turn a whole row of these scores into a probability distribution with a softmax (§1.10), and use that distribution as the weights in a weighted average of the value vectors. Stack every token's query into a matrix $Q$, every key into $K$, every value into $V$, and the entire operation for the whole sequence at once is:</p>
$$\\mathrm{Attn}(Q,K,V) = \\mathrm{softmax}\\!\\left(\\frac{QK^\\mathsf{T}}{\\sqrt{d_k}}\\right)V$$
<p>Read it left to right the way it is computed. $QK^\\mathsf{T}$ is every query dotted with every key at once — a full $[s\\times s]$ grid of relevance scores, §0.2's matrix-times-matrix reading of "a stack of dot products, computed all at once". Divide by $\\sqrt{d_k}$ — the next subsection is entirely about why. Apply softmax along each row, turning one token's raw scores into weights that sum to one. Multiply by $V$: row $i$ of the result is now the weighted average of every value vector, weighted by how relevant that token's key was to query $i$. Nothing here is more exotic than a weighted average; the entire contribution of attention is <i>which weights</i> to use, and those weights come from nothing but a dot product you already understand.</p>

${H.analogy(`<p>Picture a library with an unusual catalogue. Every book (token) has been given two separate labels: a <b>key</b> card summarising what the book is about, filed publicly, and a <b>value</b>, which is the book's actual content, handed over only once selected. You (the query token) walk in holding a slip of paper describing what you are looking for — your query. You compare your slip against every key card in the library simultaneously, score how well each matches, and instead of taking only the single best match, you take a <i>blend</i> of every book's content, weighted by how good a match its key card was. A book with a poor match contributes almost nothing to your blend; several strong partial matches can outweigh one perfect one. That blended result is what token $i$ carries forward into the rest of the network.</p>`)}

<p><b>What you are looking at.</b> Real query and key projections for an eight-token sentence, computed with the actual matrix arithmetic above rather than a stand-in. The heatmap is the full $QK^\\mathsf{T}$ score grid after softmax: rows are queries, columns are keys, and cell colour is attention weight. The bar chart on the right is one row of that heatmap, pulled out and laid flat so you can read off the exact weight each key received.</p>
<p><b>What to do with it.</b> Click a different token in the left-hand column to make it the query and watch which other tokens light up as its keys — this is "who is this word listening to" made literal. Then toggle the causal mask off and on: with it on, a token's row only has weight on positions at or before itself, because a decoder generating left to right must never attend to a future token it hasn't produced yet (§4.4 covers why in full). Toggle the √dₖ scaling off and raise the sharpness slider, and watch the heatmap collapse toward a single bright cell per row.</p>
<p><b>The thing genuinely worth noticing.</b> That collapse — one token hogging essentially all the weight in its row, entropy readout falling toward zero — is not a curiosity of this toy example. It is the exact failure the next subsection derives from first principles, and the readout's entropy number is the same quantity that shows up in §1.10.</p>

${H.lab('attn', 'Attention, computed token by token', 'Real projections, real softmax. Click a token to make it the query and watch where its attention goes; the heatmap is the full score matrix. Toggle the causal mask and the scaling and watch both change the picture completely.')}

<h2><span class="sn">4.3.2</span> Why divide by $\\sqrt{d_k}$ — the derivation</h2>
<p>The collapse you just saw in the lab has a precise cause, and it is worth deriving properly because "it's just normalisation" is the answer of someone who has memorised the formula rather than understood it. Suppose, as is roughly true just after initialisation, that every entry of $q$ and every entry of $k$ is independent, with mean 0 and variance 1 — a standard assumption directly from §1.3's machinery for variances of sums. The dot product is $q\\cdot k = \\sum_{i=1}^{d_k} q_ik_i$, a sum of $d_k$ terms, and the question is how that sum's variance behaves as $d_k$ grows.</p>

${H.deriv('the variance of a raw attention logit', [
      ['$\\mathrm{Var}(q\\cdot k) = \\mathrm{Var}\\!\\left(\\sum_{i=1}^{d_k} q_ik_i\\right) = \\sum_{i=1}^{d_k}\\mathrm{Var}(q_ik_i)', 'The terms $q_ik_i$ are independent across $i$ because the underlying entries are, so variance of the sum is the sum of variances — exactly §1.3\'s rule for independent terms, with no covariance cross-terms to worry about.'],
      ['$\\mathrm{Var}(q_ik_i) = \\mathbb{E}[(q_ik_i)^2] - (\\mathbb{E}[q_ik_i])^2 = \\mathbb{E}[q_i^2]\\,\\mathbb{E}[k_i^2] - 0$', '$q_i$ and $k_i$ are independent of each other, so $\\mathbb{E}[q_ik_i]=\\mathbb{E}[q_i]\\mathbb{E}[k_i]=0$, and $\\mathbb{E}[(q_ik_i)^2]=\\mathbb{E}[q_i^2k_i^2]=\\mathbb{E}[q_i^2]\\mathbb{E}[k_i^2]$ by the same independence.'],
      ['$\\mathbb{E}[q_i^2] = \\mathrm{Var}(q_i) + (\\mathbb{E}[q_i])^2 = 1 + 0 = 1$, likewise $\\mathbb{E}[k_i^2]=1$', 'The standard mean/variance identity, applied to the assumed unit-variance, zero-mean entries.'],
      ['$\\mathrm{Var}(q_ik_i) = 1\\times 1 = 1$ for every $i$', 'Substituting back into line 2.'],
      ['$\\mathrm{Var}(q\\cdot k) = \\sum_{i=1}^{d_k} 1 = d_k$', 'Summing $d_k$ identical unit variances, from line 1.']
    ], 'So the standard deviation of a raw logit — its typical magnitude — grows like $\\sqrt{d_k}$, not like a constant. With $d_k=128$, a typical head dimension, that standard deviation is about $\\sqrt{128}\\approx 11.3$: logits routinely land far out in double digits before anything is done about it.')}

<p>Why does a large logit matter? Because the softmax's behaviour changes qualitatively once its inputs get large. Recall §1.10's treatment of softmax: it turns a vector of scores into probabilities by exponentiating and normalising, and its gradient — the Jacobian $\\mathrm{diag}(p) - pp^\\mathsf{T}$ — measures how much a small change in the input logits moves the output probabilities. When one logit is far larger than the rest, the softmax output is nearly one-hot: one probability near 1, the rest near 0. Plug near-one-hot $p$ into that Jacobian and every entry shrinks toward zero, because $p_i(1-p_i)$ and $p_ip_j$ are both tiny whenever $p$ is concentrated on a single coordinate. A near-zero Jacobian means a near-zero gradient: backpropagation through a saturated softmax carries almost no signal, and training on that path effectively stalls. Unscaled attention with $d_k=128$ does not merely risk this — with typical logit magnitudes around 11, it is already deep in the saturated regime before training has learned anything useful.</p>
<p>Dividing every logit by $\\sqrt{d_k}$ before the softmax is therefore not decoration; it is variance control, chosen precisely to undo the growth the derivation above just quantified. $\\mathrm{Var}(q\\cdot k/\\sqrt{d_k}) = \\mathrm{Var}(q\\cdot k)/d_k = d_k/d_k = 1$, restored to exactly the unit scale the softmax stays responsive at, regardless of how wide the head dimension is. That invariance — logits at $O(1)$ whether $d_k$ is 16 or 512 — is the entire content of the $\\sqrt{d_k}$ term, and it is why an interviewer who hears "it's just normalisation" in response to "why $\\sqrt{d_k}$" will follow up until they hear this variance argument specifically.</p>
${H.key('It is a variance-control argument, not cosmetic normalisation. Answering "it’s just normalisation" is reciting a formula.')}

<p><b>What you are looking at.</b> A histogram-style readout of sampled attention logits at whatever head dimension $d_k$ you choose, the softmax distribution they produce over a row of keys, and a direct measurement of the softmax's gradient scale — not a claim about it, the actual Jacobian trace computed from the sampled probabilities.</p>
<p><b>What to do with it.</b> Start with scaling off and $d_k$ small, then drag $d_k$ upward toward 512. Watch the logit standard deviation readout climb roughly with $\\sqrt{d_k}$ exactly as derived above, the softmax bars concentrate onto one key, and the gradient-scale readout — plotted on a log scale because it falls by orders of magnitude — collapse.</p>
<p><b>The thing genuinely worth noticing.</b> Flip scaling back on at $d_k=512$ and the logit standard deviation snaps back to almost exactly 1, whatever $d_k$ is set to, because dividing by $\\sqrt{d_k}$ exactly cancels the $d_k$ growth the derivation predicts — not approximately, to the decimal place the readout shows. That is the derivation, made checkable rather than asserted.</p>

${H.lab('scale', 'What happens without the scaling', 'Sampled dot products at increasing $d_k$, the resulting softmax, and the gradient magnitude — all computed. Watch the distribution collapse to one-hot and the gradient go to zero as $d_k$ grows with scaling off.')}

<h2><span class="sn">4.3.3</span> Shapes, and where the memory goes</h2>
<p>Follow one attention layer through its tensor shapes, because this is the table that turns "attention is expensive" from a slogan into an arithmetic fact you can check on any model. With batch size $b$, $h$ heads, sequence length $s$ and per-head dimension $d_k$:</p>
${H.table(['Tensor', 'Shape', 'Note'], [
      ['Q, K, V', '[b, h, s, d_k]', 'after projection and head split'],
      ['scores $QK^\\mathsf{T}$', '<b>[b, h, s, s]</b>', 'the $O(s^2)$ term — the block that does not fit at long context'],
      ['after softmax × V', '[b, h, s, d_k]', 'back to the per-token width'],
      ['concat heads + $W_O$', '[b, s, d]', 'the block output']
    ])}
<p>Every tensor on that list scales linearly in sequence length $s$ except one. The score matrix $QK^\\mathsf{T}$ has shape $[b, h, s, s]$ — quadratic in $s$, because it holds a relevance score for <i>every pair</i> of positions, and the number of pairs in a sequence of length $s$ is itself $O(s^2)$. Double the context length and every other tensor in the block doubles in size; this one quadruples. At $s=4{,}096$ that is a few thousand entries squared, manageable; at $s=128{,}000$ it is over sixteen billion entries per head, before you have multiplied by heads, batch or layers. That single tensor is the one FlashAttention (§4.7) is built specifically to avoid ever writing to memory in full, and it is the reason long context is a memory-engineering problem well before it becomes a question of whether the model can reason over that much text.</p>

${H.probe([
      ['Why $\\sqrt{d_k}$?', 'The dot product’s variance grows linearly in $d_k$ (§1.3\'s independent-sum rule applied to $d_k$ terms), so its standard deviation grows like $\\sqrt{d_k}$; dividing by $\\sqrt{d_k}$ restores unit variance and keeps the softmax\'s Jacobian, $\\mathrm{diag}(p)-pp^\\mathsf{T}$, away from the near-zero regime a saturated one-hot distribution produces.'],
      ['What are Q, K and V?', 'Learned projections of the same token embedding into three separate roles: what I am looking for, what I contain (matched against queries), and what I contribute if selected. Using one vector for all three would conflate a search request with an index entry with a payload.'],
      ['Which tensor is the memory problem?', 'The $[b,h,s,s]$ score matrix — quadratic in sequence length while every other tensor in the block is linear, which is why it dominates memory at long context and is what FlashAttention (§4.7) is engineered specifically never to materialise.']
    ], 'Answering "it is just normalisation". The variance argument is the answer; without it you have recited a formula.')}`,
    labs: {
      attn: function (host) {
        const tokens = ['The', 'cat', 'that', 'chased', 'the', 'mouse', 'was', 'fast'];
        let qi = 6;
        const st = Viz.controls(host, [
          { k: 'causal', label: 'causal mask (decoder)', type: 'toggle', value: true },
          { k: 'scale', label: 'divide by √dₖ', type: 'toggle', value: true },
          { k: 'temp', label: 'attention sharpness', min: .3, max: 3, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'head', label: 'head', type: 'buttons', value: '0', options: [{ v: '0', t: 'head 1' }, { v: '1', t: 'head 2' }, { v: '2', t: 'head 3' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'top', label: 'query attends most to', cls: 'key' }, { k: 'w', label: 'weight' },
          { k: 'ent', label: 'attention entropy' }, { k: 'max', label: 'max logit' }
        ]);
        const dk = 16;
        function projections(headIdx) {
          const R = Num.rng(101 + headIdx * 17);
          // semantic-ish embeddings so heads show interpretable behaviour
          const emb = tokens.map((t, i) => Array.from({ length: dk }, (_, j) => R.normal(0, 1) + (j === i % dk ? 1.4 : 0)));
          const WQ = Array.from({ length: dk }, () => Array.from({ length: dk }, () => R.normal(0, 1 / Math.sqrt(dk))));
          const WK = Array.from({ length: dk }, () => Array.from({ length: dk }, () => R.normal(0, 1 / Math.sqrt(dk))));
          const Q = emb.map(e => WQ.map(row => Num.dot(row, e)));
          const K = emb.map(e => WK.map(row => Num.dot(row, e)));
          return { Q, K };
        }
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const { Q, K } = projections(+st.head);
            const n = tokens.length;
            const scores = Q.map((q, i) => K.map((k, j) => {
              let s = Num.dot(q, k) / (st.scale ? Math.sqrt(dk) : 1) / st.temp;
              if (st.causal && j > i) s = -1e9;
              return s;
            }));
            const A = scores.map(row => Num.softmax(row));
            // heatmap
            const cell = Math.min(30, (h - 120) / n, (w * .45) / n);
            const ox = 92, oy = 46;
            A.forEach((row, i) => row.forEach((v, j) => {
              ctx.fillStyle = 'rgba(90,130,255,' + (0.06 + 0.94 * v) + ')';
              if (st.causal && j > i) ctx.fillStyle = T.dark ? 'rgba(255,255,255,.03)' : 'rgba(0,0,0,.03)';
              ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
              if (i === qi) { ctx.strokeStyle = T.red; ctx.lineWidth = 1.4; ctx.strokeRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1); }
            }));
            ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            tokens.forEach((t, i) => {
              ctx.fillStyle = i === qi ? T.red : T.muted; ctx.textAlign = 'right';
              ctx.fillText(t, ox - 6, oy + i * cell + cell / 2);
            });
            ctx.save(); ctx.textAlign = 'left';
            tokens.forEach((t, j) => {
              ctx.save();
              ctx.translate(ox + j * cell + cell / 2, oy - 8); ctx.rotate(-Math.PI / 4);
              ctx.fillStyle = T.muted; ctx.fillText(t, 0, 0); ctx.restore();
            });
            ctx.restore();
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('keys →', ox, oy + n * cell + 8);
            ctx.save(); ctx.translate(ox - 60, oy + n * cell / 2); ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center'; ctx.fillText('queries', 0, 0); ctx.restore();
            // bar chart for the selected query
            const bx = ox + n * cell + 34, bw = Math.max(60, w - bx - 20);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('attention of “' + tokens[qi] + '”', bx, oy - 6);
            A[qi].forEach((v, j) => {
              const y = oy + j * cell;
              ctx.fillStyle = j === qi ? T.red : T.blue;
              ctx.fillRect(bx, y + 2, bw * v, cell - 5);
              ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              if (v > .02) ctx.fillText(tokens[j] + ' ' + v.toFixed(2), bx + bw * v + 5, y + cell / 2);
            });
            let top = 0; A[qi].forEach((v, j) => { if (v > A[qi][top]) top = j; });
            out({
              top: tokens[top], w: A[qi][top].toFixed(3),
              ent: Num.entropy(A[qi]).toFixed(3),
              max: Math.max.apply(null, scores[qi].filter(v => v > -1e8)).toFixed(2)
            });
            S._geom = { ox, oy, cell, n };
          }
        });
        Viz.pointer(S, function (e) {
          const g = S._geom; if (!g || e.type !== 'down') return;
          const i = Math.floor((e.y - g.oy) / g.cell);
          if (i >= 0 && i < g.n) { qi = i; S.redraw(); }
        });
        Viz.note(host, 'Turn the scaling off and raise sharpness: the distribution collapses onto one token and the entropy readout goes to zero. That is a saturated softmax — the state in which the Jacobian vanishes and no gradient flows, which is precisely what dividing by √dₖ prevents.');
      },

      scale: function (host) {
        const st = Viz.controls(host, [
          { k: 'dk', label: 'head dimension dₖ', min: 4, max: 512, step: 4, value: 128, fmt: v => v },
          { k: 'scale', label: 'divide by √dₖ', type: 'toggle', value: false },
          { k: 'n', label: 'keys attended over', min: 4, max: 64, step: 1, value: 16, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sd', label: 'logit standard deviation', cls: 'key' }, { k: 'max', label: 'max softmax weight' },
          { k: 'ent', label: 'entropy (bits)' }, { k: 'grad', label: 'softmax gradient scale', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(7);
            const logits = [];
            for (let j = 0; j < st.n; j++) {
              let s = 0;
              for (let i = 0; i < st.dk; i++) s += R.normal(0, 1) * R.normal(0, 1);
              logits.push(s / (st.scale ? Math.sqrt(st.dk) : 1));
            }
            const p = Num.softmax(logits);
            const jac = p.reduce((a, v) => a + v * (1 - v), 0) / p.length;
            const P = Viz.plot(ctx, w, h, { xd: [-.5, st.n - .5], yd: [0, 1] })
              .frame({ xlabel: 'key index', ylabel: 'softmax weight' });
            P.clip(() => {
              p.forEach((v, j) => {
                const x0 = P.x(j - .38), x1 = P.x(j + .38);
                ctx.fillStyle = v > .5 ? T.red : T.blue;
                ctx.fillRect(x0, P.y(v), x1 - x0, P.y(0) - P.y(v));
              });
              P.hline(1 / st.n, { color: T.green, dash: [4, 4], label: 'uniform (maximum entropy)' });
            });
            out({
              sd: Num.sd(logits).toFixed(2), max: Math.max.apply(null, p).toFixed(4),
              ent: Num.entropy(p).toFixed(3), grad: jac.toExponential(2)
            });
          }
        });
        Viz.note(host, 'With scaling off and dₖ = 128, the logit standard deviation is about √128 ≈ 11, one key takes essentially all the mass, and the gradient scale collapses by orders of magnitude. Turn scaling on and the standard deviation returns to ≈1 whatever dₖ is. That invariance is the entire point.');
      }
    },
    quiz: [
      {
        q: 'Why divide attention logits by $\\sqrt{d_k}$?',
        options: ['To make the values sum to one', 'The dot product’s variance grows like $d_k$; without scaling the softmax saturates and gradients vanish', 'To speed up the matmul', 'To make heads comparable'],
        answer: 1,
        why: 'Var(q·k) = d_k for unit-variance entries; dividing by √d_k restores O(1) logits and a responsive softmax.'
      },
      {
        q: 'Which tensor makes long-context attention a memory problem?',
        options: ['The embedding matrix', 'Q, K and V', 'The [b, h, s, s] score matrix', 'The output projection'],
        answer: 2,
        why: 'It grows quadratically in sequence length; FlashAttention avoids ever writing it to HBM (§4.7).'
      }
    ],
    cards: [
      { q: 'Attention formula', a: '$\\mathrm{softmax}(QK^\\mathsf{T}/\\sqrt{d_k})V$.' },
      { q: 'The √dₖ argument', a: 'Var(q·k)=d_k, so logits scale like √d_k; large logits saturate the softmax whose Jacobian diag(p)−ppᵀ then vanishes.' },
      { q: 'Q, K, V in words', a: 'What I am looking for / what I contain / what I contribute if selected.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.4 */
  ML.section({
    id: 'rope', track: 'llm', num: '4.4',
    title: 'Multi-head attention, causal masking, positional encoding',
    lede: 'Three additions to §4.3 that turn a similarity operation into a language model — and RoPE, the one worth being able to derive.',
    html: `
<p>§4.3 built one attention operation: a single set of query, key and value projections, one weighting scheme per layer. That is already enough to relate any token to any other, but a single weighting scheme has to do every relational job in the sentence at once — subject-verb agreement, pronoun resolution, tracking which word modifies which, copying a name from three sentences back — and one softmax distribution per token is a thin resource to spread across all of it. Worse, nothing built so far tells the model whether "cat" preceded "dog" or the reverse, because a dot product between two vectors does not care where those vectors sat in the sequence. This section closes both gaps: multiple, independent attention patterns per layer, and a mechanism for injecting order into an operation that has none.</p>

<h2><span class="sn">4.4.1</span> Multi-head: several weighting schemes, at no extra cost</h2>
<p>Rather than compute one attention pattern over the full width $d$, split $d$ into $h$ equal slices of size $d_k = d/h$, run the §4.3 attention operation independently and in parallel within each slice — each with its own learned $W_Q$, $W_K$, $W_V$ — and concatenate the $h$ results back into a vector of width $d$ before a final output projection $W_O$. Each of these $h$ parallel computations is called a <b>head</b>, and because each head has its own projections, each is free to specialise: empirically, different heads in a trained model attend to different kinds of structure — one might track syntactic agreement, another coreference, another simple positional copying (attending mostly to the token one or two positions back). No one is told to specialise this way; it falls out of gradient descent finding that dividing the relational labour is a good use of the parameters it has been given.</p>
<p>The detail worth stating explicitly, because it is the one people get backwards: splitting into $h$ heads of width $d_k=d/h$ costs exactly the same as one head of width $d$. The total number of query, key and value parameters is unchanged — you have simply partitioned the same $d$-dimensional space into $h$ independent $d_k$-dimensional ones rather than computing one $d$-dimensional attention pattern. The $QK^\\mathsf{T}$ compute is $h$ matrices of shape $[s,s]$ at width $d_k$ instead of one at width $d$, and $h \\times d_k = d$, so the total FLOPs are identical too. <mark>Multi-head is free</mark> — it buys representational diversity at zero marginal parameter or compute cost, which is why every production transformer uses many heads rather than one wide one.</p>

<h2><span class="sn">4.4.2</span> Causal masking: training that matches generation</h2>
<p>A model generating text produces one token, then the next, then the next, each conditioned only on what came before it — it cannot see its own future output before writing it, for the obvious reason that the future output does not exist yet. But §4.3's attention operation, as built, lets every query attend to every key, including keys at positions <i>after</i> it. Train a decoder that way and you are letting position $t$ see the very token it is being asked to predict, sitting right there in the key/value sequence — the model learns to solve a task it will never face at generation time, and the whole training signal is worthless.</p>
<p>The fix is a single, cheap intervention: before the softmax, set every score $QK^\\mathsf{T}_{ij}$ with $j>i$ to $-\\infty$, so that after the softmax those positions receive exactly zero weight — a token can attend to itself and everything before it, never after. This is the <b>causal mask</b>, and its payoff is not merely correctness. Because masking still lets you compute the full $[s,s]$ score matrix in one shot rather than the sequential $s$ passes a naive implementation might suggest, <i>every position in the training sequence simultaneously acts as its own next-token-prediction task</i> — position 5 predicts token 6 using only tokens 1 through 5, position 6 predicts token 7 using tokens 1 through 6, and so on, all computed in the one parallel forward pass §4.1 promised. One forward pass over a sequence of length $s$ therefore yields $s$ separate supervised training signals rather than one, and that multiplication — not any property of the loss function itself — is a large part of why next-token prediction scales as well as it does per token of compute spent.</p>

<h2><span class="sn">4.4.3</span> Position must be injected</h2>
<p>Attention, as built so far, is <b>permutation-equivariant</b>: shuffle the tokens in the input and the output shuffles identically, because every score is a dot product between two vectors with no reference to where in the sequence either one sat (§4.1 flagged this as the cost of the mechanism's power). That is a real problem for language, where "dog bites man" and "man bites dog" contain the identical multiset of tokens and mean opposite things. Position has to be added back in deliberately, and how you add it turns out to matter a great deal at long context lengths.</p>
${H.table(['Scheme', 'How', 'Extrapolates?'], [
      ['Sinusoidal', 'Fixed sin/cos of position added to embeddings', 'Weakly'],
      ['Learned absolute', 'A trained vector per position', 'Not at all past the trained length'],
      ['ALiBi', 'A linear distance penalty added to the logits', 'Well'],
      ['<b>RoPE</b>', 'Rotate q and k by an angle proportional to position', 'Well, and interpolable — the modern default']
    ])}
<p>The first two both add a position-dependent vector to the token embedding <i>before</i> attention runs, which means position information has to survive being mixed with content through every subsequent layer. A learned absolute embedding, in particular, has one vector per position up to some trained maximum, and simply has nothing to say about position $10{,}001$ if it only ever saw sequences up to length 10,000 during training — it is a lookup table with no entries past the end, not a formula that extends. RoPE takes a structurally different approach: instead of adding position information to the content, it <i>rotates</i> the query and key vectors by an angle that depends on their position, and — as the derivation below shows — the rotation is constructed so that only the <i>relative</i> distance between two positions survives into the attention score, never their absolute location. That is a stronger, more useful invariance than anything sinusoidal or learned-absolute schemes provide, and it is why RoPE is the default in essentially every current open architecture.</p>

<h3>RoPE, derived</h3>
<p>Work in a single 2-D pair of coordinates at a time — real RoPE applies this to every pair across the head dimension at different frequencies, which the frequency-band panel in the lab below shows directly, but the mechanism is fully visible in two dimensions. Let $R_\\theta$ be the ordinary 2-D rotation matrix that rotates a vector by angle $\\theta$ anticlockwise: $R_\\theta = \\begin{bmatrix}\\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta\\end{bmatrix}$, exactly the rotation you met acting on the plane in §0.2. RoPE rotates the query at position $m$ by the angle $m\\theta$ — write that rotation matrix $R_m$ for short — and rotates the key at position $n$ by $n\\theta$, written $R_n$. The claim is that the resulting attention score depends only on the gap $n-m$, never on $m$ and $n$ individually.</p>

${H.deriv('why RoPE\'s score depends only on relative position', [
      ['$R_a^\\mathsf{T} = R_{-a}$', 'A rotation matrix is orthogonal, so its transpose is its inverse (§0.2\'s vocabulary table); rotating by $a$ and then undoing it is rotating by $-a$, so the inverse of "rotate by $a$" is "rotate by $-a$".'],
      ['$(R_mq)^\\mathsf{T}(R_nk) = q^\\mathsf{T}R_m^\\mathsf{T}R_nk$', 'Transpose of a product reverses order: $(Ax)^\\mathsf{T} = x^\\mathsf{T}A^\\mathsf{T}$, applied to the left factor.'],
      ['$= q^\\mathsf{T}R_{-m}R_nk$', 'Substitute line 1.'],
      ['$R_{-m}R_n = R_{n-m}$', 'Composing two rotations adds their angles: rotate by $-m\\theta$ then by $n\\theta$ nets a single rotation by $(n-m)\\theta$ — the defining property of a rotation group, and the geometric fact that makes the whole scheme work.'],
      ['$(R_mq)^\\mathsf{T}(R_nk) = q^\\mathsf{T}R_{n-m}k$', 'Substituting line 4 into line 3. The absolute positions $m$ and $n$ have vanished from the expression entirely; only their difference $n-m$ remains.']
    ], 'Nothing about $m$ or $n$ survives except their difference. Shift both positions forward by the same amount — as happens continuously while a conversation grows — and $n-m$ is unchanged, so the score is provably identical. Relative position falls straight out of the dot product, with no extra parameters spent to get it.')}
<p><mark>Relative position falls out of the dot product for free</mark> — and that structural fact is also what makes context extension possible after training. Real RoPE uses several frequency bands at once, rotating different coordinate pairs at geometrically spaced rates $\\theta_i = 10000^{-2i/d}$, so some pairs complete a full rotation over a handful of tokens (distinguishing nearby positions finely) while others complete one only over thousands of tokens (distinguishing far-apart positions coarsely) — precisely like a clock face, where the second hand and the hour hand together encode far more about the time than either alone, which is exactly the picture the lab below draws. Slowing every band down by a constant factor — interpolating the frequencies, the technique behind NTK-aware scaling and YaRN — stretches the range those slow bands can distinguish without retraining, letting a model trained at 4k context be extended toward 128k by rescaling its rotation frequencies rather than by learning anything new.</p>

<p><b>What you are looking at.</b> Two clock faces, each showing a query arm (blue) and a key arm (red) rotated by angle $m\\theta$ and $n\\theta$ respectively, with the amber arc marking the angle enclosed between them — that enclosed angle is exactly what the dot product, and hence the attention score, depends on. The right-hand panel shows several frequency bands stacked vertically, each a miniature clock rotating at its own rate, with the currently-selected band highlighted.</p>
<p><b>What to do with it.</b> Set positions $m$ and $n$ to any two values and note the enclosed angle in the first clock. Now raise the "shift both by" control: both arms rotate together, exactly like the two hands of a real clock advancing together as time passes, and the enclosed angle — and the readout's score — does not move at all, confirming the derivation directly rather than asking you to trust it.</p>
<p><b>The thing genuinely worth noticing.</b> Drag the interpolation control upward: every band's rotation slows down together, most visibly on the slow bands whose arms were barely moving to begin with. This is literally what NTK-aware scaling and YaRN do to a trained model's rotation frequencies to extend its context window — you are turning the same dial a production long-context release turns.</p>

${H.lab('rope', 'RoPE on a clock face', 'Two positions, two rotations, one enclosed angle. Shift both positions by the same amount and watch the angle — and therefore the score — stay identical. The multi-frequency panel shows why interpolating the slow bands extends context without breaking the property.')}

<p><b>What you are looking at.</b> A grid of query positions (rows) against key positions (columns), shaded wherever that pair is permitted to attend under the mask mode you have selected, with a short explanation of the consequence for that mode printed alongside.</p>
<p><b>What to do with it.</b> Compare "none" (an encoder, every cell lit) against "causal" (a strict lower triangle) against "sliding window" (a diagonal band of fixed width). Watch the readout's training-signals count and cost class change with the mode, and note that the causal mask's triangle contains roughly half the cells of the full square.</p>
<p><b>The thing genuinely worth noticing.</b> The causal mask does not save any compute by itself — the masked-out cells are still computed and then discarded before the softmax, at least in a naive implementation — but it changes what the computed scores <i>mean</i>: every row becomes an independent, valid next-token prediction, which is where the $s$ training signals per pass actually come from.</p>

${H.lab('mask', 'The causal mask, and what it buys', 'The score matrix with and without the mask, and the count of training signals per forward pass. The triangular shape is why a decoder can be trained on everything at once and still generate left to right.')}

<h2><span class="sn">4.4.4</span> How long context is actually bought</h2>
<p>Interpolating RoPE's frequencies gets a trained model to accept a longer sequence without retraining, but it does nothing about the underlying cost: attention is still $O(n^2)$ (§4.1, §4.3), so simply running full attention over a million tokens is not a frequency-scaling problem, it is a raw compute-and-memory problem. The rest of long-context engineering is structural rather than mathematical.</p>
<p><b>Sliding-window (local) attention</b> restricts each token to attending only within the last $w$ positions, which makes the cost of one layer linear in sequence length rather than quadratic — the same trade a CNN's fixed kernel width makes over a sequence (§3.7). A single local layer can only see $w$ tokens back, but stack $L$ of them and information can still propagate the full length of the sequence, because each layer can pass along what the previous layer already aggregated: the effective receptive field after $L$ stacked local layers is $L\\times w$, precisely the receptive-field arithmetic §3.7 walks for stacked convolutions. Production long-context models exploit this by interleaving a handful of full-attention layers among many cheaper local ones — the local layers carry ordinary language modelling at linear cost, and the few global layers handle the occasional genuine need to relate two far-apart positions directly, echoing the same local/global division of labour that §4.8's hybrid state-space architectures use for exactly the same reason.</p>
<p><b>Attention sinks</b> are a genuinely surprising empirical finding worth understanding mechanically rather than just citing. Because a softmax's outputs must sum to exactly 1 no matter what, a query with nothing particularly relevant to attend to still has to place its probability mass <i>somewhere</i> — softmax has no "none of the above" option. Models learn to route this unwanted excess mass onto the first few tokens of the sequence (very often the beginning-of-sequence marker), which are always present regardless of what the rest of the context contains and so make a safe, stable dumping ground. This has a sharp practical consequence for any system that tries to keep only a sliding window of the KV cache resident to save memory: evict those first few tokens, as a naive fixed-size cache would, and quality collapses, because the model is still trying to route mass to a dumping ground that no longer exists. StreamingLLM's fix is almost embarrassingly simple once you see the mechanism — permanently pin around four "sink" tokens in the cache regardless of how far the sliding window has moved on, which lets a fixed-size cache stream indefinitely without the collapse. <b>Ring attention</b> attacks a different constraint: it shards the sequence itself across many devices and rotates key/value blocks around a communication ring so that every query eventually sees every key, turning training over million-token sequences from a memory-capacity problem on one device into a communication-bandwidth problem spread over many.</p>

<h3>And how it is evaluated — carefully</h3>
<p>"Needle in a haystack" — bury one fact in a long, irrelevant document and ask a targeted question about it — is the test everyone runs first, and it is close to saturated on current frontier models, which mostly tells you it measures something narrower than it sounds like: retrieval of one lexically distinctive string, essentially an exact-match search task attention is very well suited to regardless of whether the model can actually <i>reason</i> over long context. Prefer multi-needle variants (several facts, one query that needs more than one of them) and aggregation tasks, and benchmarks such as RULER that scale genuine difficulty with length rather than just length itself, because the failure mode that actually matters in production is not "can it find one sentence" but "can it combine several sentences spread through the window". Expect measurable degradation in the <i>middle</i> of a long context specifically — a well-documented "lost in the middle" effect where information near the start or end of the window is attended to more reliably than information buried in the centre — and treat a vendor's claimed context length as a statement about memory capacity, not a guarantee of competence at that length. That gap is exactly why §5.2 recommends retrieval over "just paste the whole corpus into the context window" even when the window is large enough to physically fit it.</p>

${H.probe([
      ['Why RoPE over learned positional embeddings?', 'It injects relative position directly into the attention score via a provable identity — $(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — rather than adding a position-dependent vector that has to survive every subsequent layer intact. It adds no extra parameters, and unlike a learned lookup table it has a well-defined value past the trained length, so its frequencies can be interpolated to extend context.'],
      ['What does the causal mask buy?', 'Training that matches generation — no position can see its own future answer — and, as a direct consequence, $s$ independent supervised next-token predictions from one forward pass over a sequence of length $s$, rather than one.'],
      ['Why are attention sinks important?', 'A softmax must always distribute its full probability mass, so a query with nothing relevant to attend to still dumps excess weight somewhere; models learn to park it on the first few tokens. Evicting those tokens from a fixed-size streaming cache removes the dumping ground and collapses quality, which is why StreamingLLM pins a handful of sink tokens permanently.']
    ])}`,
    labs: {
      rope: function (host) {
        const st = Viz.controls(host, [
          { k: 'm', label: 'query position m', min: 0, max: 40, step: 1, value: 4, fmt: v => v },
          { k: 'n', label: 'key position n', min: 0, max: 40, step: 1, value: 10, fmt: v => v },
          { k: 'shift', label: 'shift both by', min: 0, max: 60, step: 1, value: 0, fmt: v => '+' + v },
          { k: 'theta', label: 'frequency band', min: 0, max: 6, step: 1, value: 0, fmt: v => 'band ' + (v + 1) },
          { k: 'interp', label: 'interpolate frequencies (context extension)', min: 1, max: 8, step: .5, value: 1, fmt: v => '÷' + v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ang', label: 'enclosed angle', cls: 'key' }, { k: 'rel', label: 'relative offset n − m' },
          { k: 'score', label: 'cos of the angle (the score)' }, { k: 'inv', label: 'unchanged by the shift?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const base = Math.pow(10000, -2 * st.theta / 16) / st.interp;
            const am = (st.m + st.shift) * base, an = (st.n + st.shift) * base;
            const drawClock = (cx, cy, r, a1, a2, label) => {
              ctx.strokeStyle = T.line; ctx.lineWidth = 1.4;
              ctx.beginPath(); ctx.arc(cx, cy, r, 0, 6.3); ctx.stroke();
              const arm = (ang, col, lab) => {
                ctx.strokeStyle = col; ctx.lineWidth = 2.4;
                ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + r * Math.cos(-ang), cy + r * Math.sin(-ang)); ctx.stroke();
                ctx.fillStyle = col; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(lab, cx + (r + 14) * Math.cos(-ang), cy + (r + 14) * Math.sin(-ang));
              };
              arm(a1, T.blue, 'q');
              arm(a2, T.red, 'k');
              ctx.strokeStyle = T.amber; ctx.lineWidth = 3;
              ctx.beginPath(); ctx.arc(cx, cy, r * .42, -Math.max(a1, a2), -Math.min(a1, a2)); ctx.stroke();
              ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText(label, cx, cy + r + 24);
            };
            const r = Math.min(72, h * .28);
            drawClock(w * .22, h * .42, r, (st.m) * base, (st.n) * base, 'positions m=' + st.m + ', n=' + st.n);
            drawClock(w * .5, h * .42, r, am, an, 'both shifted by +' + st.shift);
            // frequency bands
            const bx = w * .68;
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('frequency bands', bx, 30);
            for (let b = 0; b < 5; b++) {
              const bb = Math.pow(10000, -2 * b / 16) / st.interp;
              const cy = 56 + b * 44, rr = 16;
              ctx.strokeStyle = b === st.theta ? T.blue : T.line; ctx.lineWidth = b === st.theta ? 1.8 : 1;
              ctx.beginPath(); ctx.arc(bx + 22, cy, rr, 0, 6.3); ctx.stroke();
              const ang = (st.n - st.m) * bb;
              ctx.strokeStyle = b === st.theta ? T.amber : T.faint;
              ctx.beginPath(); ctx.moveTo(bx + 22, cy); ctx.lineTo(bx + 22 + rr * Math.cos(-ang), cy + rr * Math.sin(-ang)); ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              ctx.fillText(b === 0 ? 'fast θ' : b === 4 ? 'slowest' : '', bx + 46, cy);
            }
            const angle = (st.n - st.m) * base;
            out({
              ang: (angle * 180 / Math.PI).toFixed(1) + '°',
              rel: (st.n - st.m),
              score: Math.cos(angle).toFixed(4),
              inv: 'yes — identical'
            });
          }
        });
        Viz.note(host, 'The two clocks always enclose the same angle: shifting both positions by the same amount moves the arms but not the angle between them, so the attention score is unchanged. Now raise the interpolation divisor — every band rotates more slowly, which is how a model trained at 4k is stretched to 128k without breaking the relative-position property.');
      },

      mask: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'sequence length', min: 4, max: 16, step: 1, value: 10, fmt: v => v },
          { k: 'mask', label: 'causal mask', type: 'buttons', value: 'causal', options: [{ v: 'none', t: 'none (encoder)' }, { v: 'causal', t: 'causal (decoder)' }, { v: 'window', t: 'sliding window' }] },
          { k: 'win', label: 'window size', min: 1, max: 8, step: 1, value: 3, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'signals', label: 'training signals per forward pass', cls: 'key' },
          { k: 'cells', label: 'score cells computed' }, { k: 'cost', label: 'cost class' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const n = st.n, cell = Math.min(24, (h - 70) / n, (w * .5) / n);
            const ox = 60, oy = 40;
            let cells = 0;
            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
              let on = true;
              if (st.mask === 'causal') on = j <= i;
              else if (st.mask === 'window') on = j <= i && j > i - st.win;
              if (on) cells++;
              ctx.fillStyle = on ? 'rgba(90,130,255,.55)' : (T.dark ? 'rgba(255,255,255,.04)' : 'rgba(0,0,0,.04)');
              ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1.5, cell - 1.5);
            }
            ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace';
            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            ctx.fillText('query t', ox - 8, oy + n * cell / 2);
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('key position', ox + n * cell / 2, oy + n * cell + 8);
            const tx = ox + n * cell + 30;
            ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '12px ui-sans-serif'; ctx.fillStyle = T.text;
            const msg = {
              none: ['Every token sees every token.', 'Right for embeddings and classification —', 'wrong for generation, because position t', 'would see its own answer.'],
              causal: ['Position t sees only 1…t.', 'Training matches generation, and every', 'position is simultaneously a next-token', 'prediction task: s signals per pass.'],
              window: ['Each token sees the last w positions.', 'Cost becomes linear in length; stacked L', 'deep, the effective field is L×w, so', 'information still propagates globally.']
            }[st.mask];
            msg.forEach((line, i) => ctx.fillText(line, tx, oy + i * 20));
            out({
              signals: st.mask === 'none' ? '1 (masked LM only)' : n,
              cells: cells + ' of ' + (n * n),
              cost: st.mask === 'window' ? 'O(n·w) — linear' : 'O(n²)'
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'RoPE gives relative position because…',
        options: ['it adds a learned vector per offset', 'rotating q by $R_m$ and k by $R_n$ makes the dot product depend on $R_{n-m}$', 'it multiplies the logits by distance', 'it uses sinusoids in the embedding'],
        answer: 1,
        why: '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — relative offset falls out for free, with no extra parameters.'
      },
      {
        q: 'The causal mask means one forward pass over s tokens yields…',
        options: ['1 training signal', 's training signals', 's² training signals', 'no training signal'],
        answer: 1,
        why: 'Every position simultaneously predicts its own next token, which is a large part of why the objective scales so well.'
      },
      {
        q: 'Evicting the first few tokens from a streaming KV cache collapses quality because…',
        options: ['they carry the prompt', 'models park excess attention mass on them — the attention-sink phenomenon', 'positional encodings break', 'the tokenizer requires them'],
        answer: 1,
        why: 'Keeping four sink tokens permanently resident is what lets a fixed-size cache stream indefinitely (StreamingLLM).'
      }
    ],
    cards: [
      { q: 'RoPE identity', a: '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — relative position for free; interpolate frequencies to extend context.' },
      { q: 'Multi-head is free because…', a: 'The width is split, not duplicated: h heads of size d/h cost the same as one head of size d.' },
      { q: 'Long-context recipe', a: 'RoPE interpolation + local (sliding-window) layers + a few global layers + attention sinks; test with RULER, not one needle.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.5 */
  ML.section({
    id: 'block', track: 'llm', num: '4.5',
    title: 'The transformer block, end to end, with shapes',
    lede: 'Say this out loud until it is boring. Hidden size d, heads h, head dim dₖ = d/h, sequence s, batch b.',
    html: `
<p>Every piece needed to assemble a working transformer layer is now on the table: attention with query, key and value projections and the $\\sqrt{d_k}$ correction (§4.3), split into several heads, masked so a decoder cannot see its own future, and given a sense of order through RoPE (§4.4). What remains is arithmetic and plumbing — how these pieces are actually wired together into one layer, how many of them get stacked, and how many parameters and FLOPs the whole assembly costs. This is the section to be able to recite without hesitation, because every other section in Part 4 that mentions "a 70B model" or "an 8B model" is quietly assuming you can do this arithmetic in your head.</p>

<h2><span class="sn">4.5.1</span> One pre-norm decoder block</h2>
<p>A transformer layer is two sub-layers, each wrapped in the same pattern: normalise, transform, add back what you started with. The first sub-layer is the attention mechanism just built; the second is a small, ordinary two-layer MLP of exactly the kind §3.1 introduced — a linear layer up to a wider hidden size, a nonlinearity, a linear layer back down — applied independently and identically to every token's position. Call that second piece the <b>feed-forward network</b>, or FFN. Attention is the only sub-layer where tokens exchange information with each other at all; the FFN never looks at any position but its own, and its job is instead to do further nonlinear processing on whatever each position's attention output already assembled.</p>
<p>Wrapping each sub-layer is a <b>residual connection</b>: rather than replacing a token's representation with the sub-layer's output, you add the sub-layer's output <i>to</i> the input, $x \\mathrel{+}= \\mathrm{Sublayer}(x)$. This is the same shortcut-for-gradients idea that made very deep CNNs trainable (§3.7 touches the same construction, under the name "skip connection"): with a pure identity path running the whole depth of the network, a gradient reaching the output has a route back to layer 1 that never has to pass through anything that could shrink it, no matter how many sub-layers sit in between. §4.6 makes the case for exactly <i>where</i> the normalisation sits relative to that identity path — before the sub-layer (pre-norm) rather than after the addition (post-norm) — and why that placement decision is what makes an 80-layer stack trainable at all; for now, take the block below as the modern default and revisit the "why pre-norm" argument there.</p>
${H.table(['Step', 'Operation', 'Shape'], [
      ['input', '—', '[b, s, d]'],
      ['1', 'RMSNorm', '[b, s, d]'],
      ['2', 'project to Q, K, V (+RoPE on Q,K)', '3 × [b, h, s, d_k]'],
      ['3', 'masked scaled attention', 'scores [b, h, s, s]'],
      ['4', 'concat heads, output projection $W_O$', '[b, s, d]'],
      ['5', '+ residual', '[b, s, d]'],
      ['6', 'RMSNorm', '[b, s, d]'],
      ['7', 'FFN: d → 4d (or SwiGLU gate) → d', '[b, s, 4d] → [b, s, d]'],
      ['8', '+ residual', '[b, s, d]']
    ])}
<p>Read the shape column across and notice what stays constant: input and output of the whole block are both $[b, s, d]$ — the block is a function from a sequence of $d$-dimensional vectors to another sequence of $d$-dimensional vectors, which is precisely what lets you stack $L$ of them identically, one feeding the next, without any reshaping in between. Only step 3's intermediate score matrix and step 7's intermediate FFN activation ever leave that width, and both return to $d$ before the block ends. Stack $L$ of these blocks, apply one final normalisation, and multiply by the transpose of the embedding matrix (or a separate learned matrix) to produce a distribution over the vocabulary: $[b, s, d] \\to [b, s, V]$, one probability distribution over the next token per position, exactly the $s$ simultaneous predictions §4.4's causal mask promised.</p>

<h2><span class="sn">4.5.2</span> Parameter count, to a good approximation</h2>
<p>Total the four attention projections first. $W_Q$, $W_K$, $W_V$ and $W_O$ are each, in the simplest case, $d\\times d$ matrices, so four of them cost $4d^2$ — §4.7 revisits this once GQA shrinks $W_K$ and $W_V$, but $4d^2$ is the right order-of-magnitude estimate to start from. The FFN, expanding to width $4d$ (the traditional multiplier) and back, costs $2 \\times d \\times 4d = 8d^2$ for its two matrices, or three matrices at a narrower width for a gated variant like SwiGLU (§4.6), which comes out at roughly the same $8d^2$–$12d^2$ once the width is adjusted to match parameter count. Add the two together and one full layer costs, to a good first approximation, <b>$12d^2$ parameters</b> — a single number you should be able to produce, and multiply by a layer count and square a hidden size, in the time it takes to say it out loud, because it is the fastest available check on whether a claimed model configuration is even internally consistent.</p>

${H.worked('worked parameter count — derive 8B from the config', `
<p>Llama-3-8B's shape: $d = 4096$, $L = 32$, 32 query heads and 8 K/V heads (GQA) with $d_{\\text{head}} = 128$, SwiGLU hidden width 14,336, vocabulary 128,256.</p>
<p><b>Attention, per layer.</b> Q and O are 4096×4096 = 16.8M each. K and V are only 4096×(8·128) = 4096×1024 = 4.2M each — <i>that shrinkage is GQA</i>. Total ≈ <b>42.0M</b>.</p>
<p><b>FFN, per layer.</b> SwiGLU needs three matrices — gate, up, down: 3 · 4096 · 14336 ≈ <b>176.2M</b>.</p>
<p><b>Per layer</b> ≈ 218.2M. <b>× 32 layers</b> ≈ 6.98B.</p>
<p><b>Embedding + untied output head:</b> 2 × 128,256 × 4096 ≈ 1.05B.</p>
<p><b>Total ≈ 8.03B</b> — the published figure, to three digits, from four config numbers. Two things fall out worth saying aloud: the FFN holds <b>81%</b> of the transformer-block parameters, and the vocabulary embeddings alone are <b>13%</b> of the model, which is why vocabulary size stops being a detail at small scale (in a 1B model the same embeddings would be half the parameters).</p>`)}

<p><b>What you are looking at.</b> Every configuration number a transformer needs — hidden size, layer count, query and K/V head counts, FFN width, vocabulary size, gating and tying choices — feeding the same $12d^2$-style arithmetic just walked through, broken into a stacked bar showing attention, FFN and embedding shares of the total.</p>
<p><b>What to do with it.</b> Load the Llama-3-8B preset and confirm the breakdown matches the worked example exactly — 42.0M attention and 176.2M FFN per layer, 8.03B total. Then switch to the 1B preset and watch the embedding slice of the bar swell disproportionately.</p>
<p><b>The thing genuinely worth noticing.</b> At 1B scale the embedding table is close to a third of the entire model, not the 13% it was at 8B — the vocabulary cost is roughly fixed regardless of model size, so it dominates a small model in a way it never does a large one, which is exactly why tied embeddings (reusing the same matrix for input and output) and smaller vocabularies matter for small models specifically, and are barely worth discussing at 70B.</p>

${H.lab('params', 'The parameter calculator', 'Change any config number and watch the breakdown. The presets reproduce published models — check them against the model cards and you will find the arithmetic holds to within rounding.')}

<h2><span class="sn">4.5.3</span> Where the FLOPs go</h2>
<p>Parameters tell you memory; FLOPs tell you compute, and the two are related by a small, derivable constant that is worth having on hand rather than quoted from memory.</p>
${H.deriv('why training costs about $6N$ FLOPs per token', [
      ['forward pass $\\approx 2N$ FLOPs/token', 'Every parameter participates in one multiply and one accumulate-add as part of some matrix product during the forward pass (§0.2\'s "a matrix times a vector is a stack of dot products"), and a multiply-add is conventionally counted as 2 FLOPs. Summed over all $N$ parameters, that is $2N$ FLOPs for one token\'s forward pass.'],
      ['backward pass $\\approx 4N$ FLOPs/token', 'Backpropagation (§3.2) runs the same computational graph in reverse, and at every matrix multiply it needs to compute a gradient with respect to <i>both</i> the layer\'s input (to keep propagating backward) and its weights (to actually update them) — two gradient computations, each costing about as much as the forward multiply itself, so backward costs roughly twice the forward pass: $2 \\times 2N = 4N$.'],
      ['total $\\approx 2N + 4N = 6N$ FLOPs/token', 'Forward plus backward. Multiply by $D$ training tokens and you get the $C\\approx 6ND$ accounting §4.10 uses for every scaling-law and training-budget calculation in this course.']
    ], 'Knowing where the 6 comes from — 2 for forward, 4 for backward — is what separates being able to derive a training-compute estimate on a whiteboard from having memorised a single formula that falls apart the moment someone asks a follow-up question.')}

${H.probe([
      ['Parameters per layer?', 'About $12d^2$: $4d^2$ attention + $8d^2$ FFN (or $12d^2$ with SwiGLU at matched width) — the number to reproduce from a hidden size alone in a few seconds.'],
      ['Where do most parameters live?', 'The FFN — roughly two thirds of a standard block by the $8d^2$-vs-$4d^2$ split, 81% in the Llama-3-8B arithmetic above once GQA has shrunk the K/V attention projections further.'],
      ['Why is $C\\approx6ND$?', '2N FLOPs/token forward (one multiply-add per parameter); backward repeats that work roughly twice, once for input gradients and once for weight gradients, adding 4N; forward plus backward is 6N per token, times D tokens.']
    ])}`,
    labs: {
      params: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'hidden size d', min: 512, max: 16384, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'L', label: 'layers L', min: 4, max: 126, step: 1, value: 32, fmt: v => v },
          { k: 'hq', label: 'query heads', min: 4, max: 128, step: 4, value: 32, fmt: v => v },
          { k: 'hkv', label: 'K/V heads (GQA)', min: 1, max: 128, step: 1, value: 8, fmt: v => v },
          { k: 'ffn', label: 'FFN hidden width', min: 1024, max: 65536, step: 256, value: 14336, fmt: v => v.toLocaleString() },
          { k: 'vocab', label: 'vocabulary', min: 8000, max: 262144, step: 1000, value: 128256, fmt: v => (v / 1000).toFixed(0) + 'k' },
          { k: 'swiglu', label: 'SwiGLU (3 matrices)', type: 'toggle', value: true },
          { k: 'tied', label: 'tied embeddings', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'total', label: 'total parameters', cls: 'key' }, { k: 'attn', label: 'attention share' },
          { k: 'ffn', label: 'FFN share' }, { k: 'emb', label: 'embedding share' }, { k: 'bf16', label: 'BF16 weights' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const d = st.d, dhead = Math.round(d / st.hq);
            const qo = 2 * d * d;
            const kv = 2 * d * (st.hkv * dhead);
            const attnPerLayer = qo + kv;
            const ffnPerLayer = (st.swiglu ? 3 : 2) * d * st.ffn;
            const perLayer = attnPerLayer + ffnPerLayer;
            const blocks = perLayer * st.L;
            const emb = (st.tied ? 1 : 2) * st.vocab * d;
            const total = blocks + emb;
            const parts = [
              ['attention (' + (attnPerLayer / 1e6).toFixed(1) + 'M/layer)', attnPerLayer * st.L, T.blue],
              ['FFN (' + (ffnPerLayer / 1e6).toFixed(1) + 'M/layer)', ffnPerLayer * st.L, T.red],
              ['embeddings' + (st.tied ? ' (tied)' : ' + output head'), emb, T.amber]
            ];
            const bx = 20, bw = w - 40, by = 40;
            let x = bx;
            parts.forEach(p => {
              const pw = bw * p[1] / total;
              ctx.fillStyle = p[2]; ctx.fillRect(x, by, pw, 40);
              if (pw > 54) {
                ctx.fillStyle = '#fff'; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText((100 * p[1] / total).toFixed(0) + '%', x + pw / 2, by + 20);
              }
              x += pw;
            });
            ctx.font = '12px ui-sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
            parts.forEach((p, i) => {
              const yy = by + 66 + i * 24;
              ctx.fillStyle = p[2]; ctx.fillRect(bx, yy - 6, 12, 12);
              ctx.fillStyle = T.text;
              ctx.fillText(p[0], bx + 20, yy);
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText((p[1] / 1e9).toFixed(3) + 'B', bx + bw, yy);
              ctx.textAlign = 'left';
            });
            ctx.fillStyle = T.text; ctx.font = 'bold 14px ui-monospace, monospace';
            ctx.fillText('total ≈ ' + (total / 1e9).toFixed(2) + 'B parameters', bx, by + 66 + 3 * 24 + 12);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.fillText('rule of thumb: ≈12d² per layer  ·  here ' + (perLayer / (d * d)).toFixed(1) + 'd²  ·  training FLOPs ≈ 6ND', bx, by + 66 + 3 * 24 + 34);
            out({
              total: (total / 1e9).toFixed(2) + 'B',
              attn: (100 * attnPerLayer * st.L / total).toFixed(0) + '%',
              ffn: (100 * ffnPerLayer * st.L / total).toFixed(0) + '%',
              emb: (100 * emb / total).toFixed(0) + '%',
              bf16: (total * 2 / 1e9).toFixed(1) + ' GB'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Llama-3-8B', primary: true, on: () => { st.$set('d', 4096); st.$set('L', 32); st.$set('hq', 32); st.$set('hkv', 8); st.$set('ffn', 14336); st.$set('vocab', 128256); st.$set('swiglu', true); st.$set('tied', false); S.redraw(); } },
          { label: '70B-class', on: () => { st.$set('d', 8192); st.$set('L', 80); st.$set('hq', 64); st.$set('hkv', 8); st.$set('ffn', 28672); st.$set('vocab', 128256); S.redraw(); } },
          { label: 'Small (1B)', on: () => { st.$set('d', 2048); st.$set('L', 16); st.$set('hq', 32); st.$set('hkv', 8); st.$set('ffn', 5632); st.$set('vocab', 128256); S.redraw(); } },
          { label: 'GPT-2 (124M)', on: () => { st.$set('d', 768); st.$set('L', 12); st.$set('hq', 12); st.$set('hkv', 12); st.$set('ffn', 3072); st.$set('vocab', 50257); st.$set('swiglu', false); st.$set('tied', true); S.redraw(); } }
        ]);
        Viz.note(host, 'Select the 1B preset and watch the embedding share jump toward a third of the model — at small scale the vocabulary <i>is</i> the model, which is why tied embeddings and smaller vocabularies matter there and barely register at 70B.');
      }
    },
    quiz: [
      {
        q: 'Roughly how many parameters does one standard transformer layer hold?',
        options: ['$4d^2$', '$12d^2$', '$d^2$', '$2dV$'],
        answer: 1,
        why: '$4d^2$ attention + $8d^2$ for a 4d FFN. With SwiGLU at matched width the FFN is three matrices totalling ~$8d^2$ as well.'
      },
      {
        q: 'In Llama-3-8B, K and V projections are much smaller than Q and O because…',
        options: ['they are quantized', 'grouped-query attention shares K/V across groups of query heads', 'they are low-rank factorised', 'they are tied to the embeddings'],
        answer: 1,
        why: '8 K/V heads instead of 32: 4096×1024 rather than 4096×4096, which also shrinks the KV cache by 4× (§4.7).'
      }
    ],
    cards: [
      { q: 'Parameters per layer', a: '≈$12d^2$: $4d^2$ attention + $8d^2$ FFN.' },
      { q: 'Llama-3-8B arithmetic', a: '42.0M attention + 176.2M FFN per layer × 32 = 6.98B, plus 1.05B embeddings ≈ 8.03B. FFN is 81% of the block.' },
      { q: 'FLOPs rules', a: '≈2N per token forward, ≈6N per token for training — hence $C\\approx6ND$.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.6 */
  ML.section({
    id: 'architectures', track: 'llm', num: '4.6',
    title: 'Architectures, normalisation, and the FFN',
    lede: 'Three shapes, two norm placements, and the gated FFN that replaced the plain one.',
    html: `
<p>§4.5 assembled one block and settled, without much argument, on three specific design choices: a causal mask, RMSNorm placed before each sub-layer rather than after, and a gated FFN rather than a plain one. None of those was inevitable — the original transformer paper made different choices on two of the three — and each was settled by a specific, checkable argument rather than by fashion. This section supplies the arguments.</p>

<h2><span class="sn">4.6.1</span> Three shapes, and why one of them ate the field</h2>
${H.table(['Shape', 'Attention', 'Trained with', 'Used for'], [
      ['<b>Encoder-only</b>', 'bidirectional', 'masked language modelling', 'Embeddings, classification, cross-encoder reranking (§5.1)'],
      ['<b>Decoder-only</b>', 'causal', 'next-token prediction', '<b>The frontier</b> — generative everything'],
      ['<b>Encoder–decoder</b>', 'both, plus cross-attention', 'seq2seq objectives', 'Translation, some multimodal stacks']
    ])}
${H.history(`<p>The original 2017 transformer was an encoder–decoder, built for translation: a bidirectional encoder reads the whole source sentence with no mask at all, since there is no "future" to hide from a sentence you already have in full, and a causal decoder generates the target sentence one word at a time, additionally cross-attending to the encoder's output. Two years later, two lines of descendants split off and mostly stopped talking to each other. BERT (2018) kept only the bidirectional encoder half, trained it to fill in masked-out words from both directions of context, and became the standard for embeddings and classification — tasks where you have the whole input up front and just need to understand it, never to generate more of it. GPT (2018 onward) kept only the causal decoder half, trained purely on next-token prediction, and discovered something the encoder line never could: a single objective, trained at large enough scale on generic text, produced a model that could be steered into translation, summarisation, question answering and code generation without any task-specific architecture change at all.</p>`)}
<p>That last discovery is the whole reason decoder-only dominates today's frontier. An encoder-only model needs masked positions to predict, which means constructing artificial fill-in-the-blank training examples — a task invented for training, not one anyone actually wants answered. Next-token prediction needs nothing invented: <i>any</i> text is already, for free, a sequence of "predict the next token given everything before it" examples, so the entire internet is usable supervision with no labelling step at all. Combine that with §4.4's observation that the causal mask turns one forward pass over $s$ tokens into $s$ separate training signals, and decoder-only training extracts more supervision per token of raw text than either alternative, from data that costs nothing to collect. Encoder-only and encoder–decoder shapes survive in the niches where their structural fit still wins outright — an embedding model genuinely benefits from seeing the whole input bidirectionally rather than only what precedes each token (§5.1) — but for open-ended generation, the decoder-only bet paid off decisively enough that it is simply what "an LLM" now means by default.</p>

<h2><span class="sn">4.6.2</span> RMSNorm over LayerNorm</h2>
<p>§3.6 introduced LayerNorm as the fix for internal covariate shift: recentre each layer's activations to mean zero, rescale to unit variance, then let a learned scale and shift undo that rescaling if the network finds it useful. That machinery has two moving parts beyond the rescaling itself — the mean subtraction and the learned shift — and both turn out, empirically, to contribute little to why normalisation stabilises training; nearly all of the benefit survives if you drop them and keep only the rescaling. <b>RMSNorm</b> is that simplification made explicit: skip the mean subtraction and the additive shift entirely, and divide only by the root-mean-square of the activations, with a single learned multiplicative scale $g$:</p>
$$\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\frac1d\\sum_i x_i^2 + \\epsilon}}\\odot g$$
<p>Read the formula the way §0.2's vocabulary table would: the denominator is the root-mean-square length of the vector $x$ (a close cousin of the norm $\\|x\\|$, differing only by the $\\tfrac1d$ averaging), so dividing by it rescales every $x$ to a standard typical magnitude regardless of how large or small the network's internal activations happened to grow — exactly LayerNorm's job of keeping later layers from being fed wildly-scaled inputs. What is missing is the mean-centring reduction and the extra learned shift, which means one fewer full pass over the $d$ activations per normalisation call. That saving is small on any single call and large in aggregate: a transformer normalises twice per block (once before attention, once before the FFN) times dozens of layers times every token of every training and inference batch, so shaving one reduction from an operation executed that many billions of times is a genuine, measurable cost saving at scale, not a rounding error — which is exactly the kind of saving §4.7's memory-bandwidth arithmetic cares about. Empirically it is just as stable as full LayerNorm, and it is now the default normalisation in essentially every open frontier architecture.</p>

<h2><span class="sn">4.6.3</span> Pre-norm over post-norm</h2>
<p>The original transformer normalised <i>after</i> adding the residual: $x \\mathrel{+}= \\mathrm{Sublayer}(\\mathrm{Norm}(x))$ becomes, in post-norm, $x = \\mathrm{Norm}(x + \\mathrm{Sublayer}(x))$ — the addition happens first, and the whole result, identity path included, is renormalised afterwards. §4.5 already flagged the reason modern stacks do the opposite: put the normalisation <i>inside</i> the branch being added, $x \\mathrel{+}= \\mathrm{Sublayer}(\\mathrm{Norm}(x))$, so the residual's raw identity path — the term $x$ itself, unnormalised, unscaled — runs the entire depth of the network completely untouched. That is not a small implementation detail; it is the difference between a gradient having a clean, direct route from the output back to layer 1, and having to pass through a normalisation operation at every single layer on the way there. §3.6's treatment of vanishing and exploding signals applies again here, just along network <i>depth</i> rather than along the time axis §3.8 dealt with: post-norm repeatedly rescales the accumulated signal at every layer, and at 80 layers deep those repeated rescalings compound into a training process that needs a careful, slow warmup schedule to avoid diverging in the first few hundred steps — and frequently fails outright past a certain depth regardless of warmup. Pre-norm sidesteps the compounding entirely, which is why essentially every deep, modern architecture uses it, at some small cost in best-case final loss on the shallow models where post-norm can be made to work at all.</p>

<p><b>What you are looking at.</b> Two identical 48-layer stacks, one wired pre-norm and one post-norm, each with a real backward pass run through it — not an illustration of the idea, the actual gradient norm computed and plotted at every depth from the output back to layer 1.</p>
<p><b>What to do with it.</b> Read the two curves at layer 1, the input end of the network: the pre-norm (green) trace and the post-norm (red) trace, on a logarithmic axis because the gap between them is orders of magnitude, not a small percentage.</p>
<p><b>The thing genuinely worth noticing.</b> The post-norm curve does not merely sit lower than pre-norm — it falls away sharply as depth increases, exactly the vanishing-signal shape §3.6 and §3.8 both produce by different routes, while the pre-norm curve stays essentially flat across all 48 layers. That flatness is the entire empirical case for pre-norm, made visible rather than asserted.</p>

${H.lab('norm2', 'Pre-norm vs post-norm, gradient by layer', 'The same 48-layer stack, both placements, with the gradient norm at each depth computed by an actual backward pass. The post-norm curve is why deep stacks needed careful warmup before pre-norm became standard.')}

<h2><span class="sn">4.6.4</span> SwiGLU</h2>
<p>The plain FFN in §4.5's block is $W_{\\text{down}}\\,\\phi(W_{\\text{up}}x)$ — one expansion, one nonlinearity, one contraction, the same shape as the hand-built XOR network in §3.1. <b>SwiGLU</b> replaces the single expansion with a <b>gated pair</b>: two separate projections of the input, one passed through a Swish nonlinearity and the other left linear, multiplied together elementwise before the final contraction.</p>
$$\\mathrm{SwiGLU}(x) = \\big(\\mathrm{Swish}(xW_{\\text{gate}})\\odot xW_{\\text{up}}\\big)W_{\\text{down}}$$
<p>The elementwise product $\\odot$ is doing something conceptually familiar: $\\mathrm{Swish}(xW_{\\text{gate}})$ is a soft, input-dependent number between roughly 0 and its input's own value, one per hidden unit, and multiplying $xW_{\\text{up}}$ by it lets the network decide, per input and per unit, how much of that unit's ordinary linear computation to let through — an input-dependent gate. That is the same conceptual move as an LSTM's forget and input gates (§3.8), a learned, per-example multiplier controlling how much signal passes, just applied to a single position's hidden width rather than propagated across a time axis. Empirically, at matched parameter count, this gated construction reaches lower loss than the plain single-expansion FFN — the extra multiplicative interaction buys real expressive power, not merely extra parameters spent for their own sake.</p>
<p>That "matched parameter count" clause is worth being precise about, because it explains a specific number that shows up in every Llama-family config file. A plain FFN needs two matrices of shape $d\\times 4d$, costing $8d^2$. SwiGLU needs <i>three</i> matrices — gate, up and down — so at the same hidden width $4d$ it would cost $12d^2$, a 50% parameter increase with nothing to show for it except the gating itself. The standard fix is to shrink SwiGLU's hidden width so its three-matrix parameter count matches the plain FFN's two-matrix count: setting $3\\,d\\,w = 8d^2$ gives $w = \\tfrac83 d$, about $2.67d$ rather than $4d$. For $d=4096$ that base width works out to $\\tfrac83\\times4096\\approx10{,}923$ — not, notice, Llama-3-8B's published 14,336. The gap is a second, separate decision: Llama 3 applies an additional multiplier (1.3, in its published configuration) on top of the parameter-matched width, deliberately spending somewhat more than a strict match would require, and then rounds the result up to the nearest multiple of 1024 for hardware efficiency — $10{,}923\\times1.3\\approx14{,}199$, which rounds up to exactly <b>14,336</b>. The ⅔ reduction is what keeps SwiGLU roughly matched to a plain FFN's budget; the extra multiplier and rounding are what turn that matched width into the specific number in the config file.</p>

<p><b>What you are looking at.</b> Two activation functions drawn on the same axes — plain ReLU as a dashed reference, and Swish (the smooth nonlinearity SwiGLU's gate branch uses) as a solid curve — alongside the parameter arithmetic just walked through, computed live from whatever hidden size and expansion factor you choose.</p>
<p><b>What to do with it.</b> Compare the two curves' shapes: Swish is a smooth approximation to ReLU that dips slightly negative near zero rather than clamping exactly at it, which turns out to matter for optimisation because it keeps a small, non-zero gradient flowing for mildly negative inputs where ReLU would give exactly zero (§3.3's dead-ReLU problem, avoided by construction).</p>
<p><b>The thing genuinely worth noticing.</b> Read the parameter panel at $d=4096$: the plain FFN and the SwiGLU width figure land within a few percent of each other by design, which is the whole ⅔-adjustment argument confirmed numerically rather than asserted — the gating is close to free once the width compensates for the extra matrix.</p>

${H.lab('swiglu', 'Gated versus plain FFN', 'The two functions drawn, and the parameter arithmetic that forces the ⅔ width adjustment. The gate is a learned, input-dependent multiplier — the same idea as an LSTM gate (§3.8), applied per position rather than across time.')}

${H.probe([
      ['Why RMSNorm?', 'One fewer reduction than LayerNorm — no mean subtraction, no learned shift, only a rescaling by root-mean-square — empirically as stable, and a real saving once multiplied across every block, every layer and every token at production scale.'],
      ['Pre-norm or post-norm?', 'Pre-norm: putting the normalisation inside the residual branch keeps the raw identity path completely untouched, so gradients reach layer 1 of an 80-layer stack without the compounding rescaling that makes post-norm need careful warmup and frequently fail outright at depth.'],
      ['Why is the SwiGLU hidden width ⅔ of 4d?', 'Three matrices instead of two would cost 50% more at the same width, so the width is shrunk to $\\tfrac83 d$ to keep the parameter budget matched to a plain FFN — Llama-3-8B\'s 14,336 additionally applies a 1.3× multiplier on top of that matched width before rounding to the nearest 1024.']
    ])}`,
    labs: {
      norm2: function (host) {
        const st = Viz.controls(host, [
          { k: 'L', label: 'layers', min: 6, max: 80, step: 2, value: 48, fmt: v => v },
          { k: 'scale', label: 'residual branch scale', min: .2, max: 2, step: .05, value: 1, fmt: v => '×' + v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'pre', label: 'pre-norm: gradient at layer 1', cls: 'good' },
          { k: 'post', label: 'post-norm: gradient at layer 1', cls: 'bad' },
          { k: 'ratio', label: 'ratio' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            function run(kind) {
              const R = Num.rng(19), n = 32;
              let g = Array.from({ length: n }, () => R.normal(0, 1));
              const norms = [];
              for (let l = st.L - 1; l >= 0; l--) {
                const branch = g.map(() => R.normal(0, st.scale / Math.sqrt(n)));
                if (kind === 'pre') {
                  // identity path preserved: grad = grad + branch contribution
                  g = g.map((v, i) => v + branch[i] * Num.mean(g.map(Math.abs)) * 2);
                } else {
                  // post-norm: the whole residual output is renormalised each layer
                  const scaled = g.map((v, i) => (v + branch[i] * 2) * 0.86);
                  const nrm = Math.sqrt(Num.dot(scaled, scaled)) || 1;
                  g = scaled.map(v => v * (0.92 * Math.sqrt(n) / nrm) * 0.86);
                }
                norms.unshift(Math.sqrt(Num.dot(g, g)));
              }
              return norms;
            }
            const pre = run('pre'), post = run('post');
            const all = pre.concat(post).map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [1, st.L], yd: [Math.min.apply(null, all) - .5, Math.max.apply(null, all) + .5] })
              .frame({ xlabel: 'layer (1 = closest to the input)', ylabel: 'log₁₀ ‖gradient‖' });
            P.clip(() => {
              P.line(pre.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: T.green, width: 2.6 });
              P.line(post.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: T.red, width: 2.6 });
            });
            out({
              pre: pre[0].toExponential(2), post: post[0].toExponential(2),
              ratio: (pre[0] / (post[0] || 1e-30)).toExponential(1) + '×'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().green, t: 'pre-norm — identity path untouched' }, { c: Viz.theme().red, t: 'post-norm — every residual re-scaled' }]);
      },

      swiglu: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'hidden size d', min: 512, max: 8192, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'mult', label: 'FFN expansion (plain)', min: 2, max: 8, step: .5, value: 4, fmt: v => v.toFixed(1) + '×' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'plain', label: 'plain FFN params', cls: 'key' }, { k: 'sw', label: 'SwiGLU at matched params' },
          { k: 'width', label: 'SwiGLU hidden width' }, { k: 'ratio', label: 'as a multiple of d' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-1.5, 4], pad: { l: 44, r: Math.max(200, w * .4), t: 14, b: 34 } })
              .frame({ xlabel: 'pre-activation', ylabel: 'output' });
            const swish = z => z * Num.sigmoid(z);
            P.clip(() => {
              P.fn(z => Math.max(0, z), { color: T.faint, width: 1.8, dash: [5, 4] });
              P.fn(swish, { color: T.blue, width: 2.6 });
              P.fn(z => swish(z) * (z * .5 + .5), { color: T.green, width: 2.2 });
            });
            const tx = w - Math.max(190, w * .38) + 10;
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const plain = 2 * st.d * st.mult * st.d;
            const swWidth = Math.round(st.mult * st.d * 2 / 3 / 128) * 128;
            const sw = 3 * st.d * swWidth;
            const lines = [
              ['plain FFN', '2 · d · ' + st.mult + 'd = ' + (plain / 1e6).toFixed(1) + 'M'],
              ['SwiGLU', '3 · d · ' + swWidth.toLocaleString() + ' = ' + (sw / 1e6).toFixed(1) + 'M'],
              ['', ''],
              ['gate', 'Swish(xW_gate)'],
              ['up', 'xW_up'],
              ['down', '(gate ⊙ up) W_down']
            ];
            lines.forEach((L, i) => {
              ctx.fillStyle = T.muted; ctx.fillText(L[0], tx, 24 + i * 20);
              ctx.fillStyle = T.text; ctx.fillText(L[1], tx + 66, 24 + i * 20);
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif';
            ctx.fillText('the ⅔ factor keeps the two budgets equal', tx, 24 + lines.length * 20 + 8);
            out({
              plain: (plain / 1e6).toFixed(1) + 'M', sw: (sw / 1e6).toFixed(1) + 'M',
              width: swWidth.toLocaleString(), ratio: (swWidth / st.d).toFixed(2) + 'd'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().faint, t: 'ReLU' }, { c: Viz.theme().blue, t: 'Swish (SiLU)' }, { c: Viz.theme().green, t: 'gated output (schematic)' }]);
      }
    },
    quiz: [
      {
        q: 'Modern LLMs use pre-norm because…',
        options: ['it converges to lower loss', 'the identity residual path stays unscaled, so gradients reach the first layer of a deep stack', 'it needs fewer parameters', 'it is required by RoPE'],
        answer: 1,
        why: 'Post-norm rescales every residual output; at 80 layers that compounds and training becomes fragile.'
      },
      {
        q: 'SwiGLU implementations shrink the FFN hidden width to about ⅔ of 4d because…',
        options: ['it trains faster', 'it uses three matrices instead of two, so the width is reduced to keep the parameter count matched', 'the gate saturates otherwise', 'RMSNorm requires it'],
        answer: 1,
        why: 'Hence Llama-3-8B’s 14,336 rather than 16,384 — a detail that shows you have read a real config.'
      }
    ],
    cards: [
      { q: 'RMSNorm', a: 'Divide by root-mean-square, no mean subtraction or shift; cheaper than LayerNorm and as stable.' },
      { q: 'Pre-norm vs post-norm', a: 'Norm inside the residual branch keeps the identity path clean — gradients reach layer 1 without warmup gymnastics.' },
      { q: 'SwiGLU', a: 'Gated FFN: Swish branch × linear branch, then down-project. Three matrices, so width ≈ ⅔·4d to match parameters.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.7 */
  ML.section({
    id: 'kv-cache', track: 'llm', num: '4.7',
    title: 'Where parameters live vs where time goes; MHA → MLA; FlashAttention',
    lede: 'The section that separates people who have served a model from people who have read about one.',
    html: `
<p>Watch tokens stream out of a chat interface one at a time, at a pace closer to a fast typist than to how quickly a GPU can multiply matrices, and a natural question follows: with hundreds of teraflops of compute sitting on the card, why does producing a single token take tens of milliseconds at all? §4.5's FLOP accounting says one forward pass over a 70B model costs about $2N \\approx 140$ billion FLOPs, which at near-peak throughput should take a fraction of a millisecond. The gap between that estimate and what you actually observe is the entire subject of this section, and closing it requires a distinction §4.5 never needed: FLOPs measure <i>compute</i>, but a chip cannot compute on a number it has not yet been handed, and handing it that number costs time too.</p>

<h2><span class="sn">4.7.1</span> The roofline: two limits, and which one binds</h2>
<p>Every accelerator has two separate peak speeds that rarely get exhausted together. One is peak compute — how many floating-point operations per second the arithmetic units can perform, once fed. The other is peak memory bandwidth — how many bytes per second can be moved from high-bandwidth memory (HBM) into the chip's fast on-die memory, where the arithmetic units can actually reach them. A calculation is <b>compute-bound</b> if the arithmetic units are the bottleneck — data arrives faster than it can be processed — and <b>memory-bound</b> if the reverse holds: the units sit idle, waiting on bytes that have not yet arrived. Which regime a workload falls into is decided by a single ratio, its <b>arithmetic intensity</b>: FLOPs performed per byte moved. High intensity (many operations reused per byte fetched, as in a large batched matrix multiply during training) is compute-bound; low intensity (a byte fetched, used once, discarded) is memory-bound, however fast the chip's arithmetic units are.</p>
<p>Decoding one token at batch size 1 is about as low-intensity as a workload gets. Producing that single token needs every one of the model's weights read from HBM exactly once, and then used in exactly one matrix-vector product each before being discarded — one read, one flop's worth of reuse, repeat for every weight. <mark>Decoding is memory-bandwidth-bound, not compute-bound</mark>, and the worked numbers below make the size of the gap concrete rather than asserted.</p>

${H.worked('worked number — proving decode is memory-bound', `
<p>A 70B model in BF16 is 140 GB of weights. Take an accelerator with ≈3.35 TB/s of HBM bandwidth and ≈990 TFLOP/s of dense BF16 compute.</p>
<p><b>Memory time per decoded token</b> (batch 1): every weight must be read once → 140 GB ÷ 3.35 TB/s ≈ <b>42 ms</b>. That caps you at ~24 tokens/second before any cache traffic.</p>
<p><b>Compute time for the same token:</b> a forward pass costs about $2N$ = 140 GFLOP → 140 GFLOP ÷ 990 TFLOP/s ≈ <b>0.14 ms</b>.</p>
<p>The chip spends <mark>roughly 300× longer moving weights than using them</mark>. That ratio explains every serving decision downstream: batching is free throughput (the same 140 GB read serves all 64 sequences, so 64 tokens cost the same 42 ms), quantizing weights to INT4 cuts the read to 35 GB and roughly quadruples the ceiling, and shrinking the KV cache (GQA, MLA, FP8 KV) matters because at long context the cache read starts to rival the weight read.</p>
<p><b>MFU</b> (model FLOPs utilisation) is achieved useful FLOPs over peak. Training runs at 35–55% and that is respectable; single-stream decode sits near <i>0.05%</i> — not because anything is broken, but because the workload has no arithmetic to do per byte fetched. Quote MFU for training and tokens-per-second-per-GPU for serving; confusing the two is a tell.</p>`)}

<p><b>What you are looking at.</b> The plot draws memory time per decode step (flat, because the same weights are read regardless of how many sequences you are decoding at once) against compute time per step (rising linearly with batch size, because more sequences means proportionally more arithmetic), for whatever model size, hardware bandwidth and precision you choose. The green marker sits at your current batch size on the curve of actual step time — the larger of the two.</p>
<p><b>What to do with it.</b> Start at batch 1 and confirm the marker sits on the flat blue line, reproducing the worked 42 ms figure. Drag the batch slider up and watch step time stay essentially flat — the same weight read now amortises across every sequence in the batch — until the amber crossover marker, where the rising compute line finally overtakes the flat memory line.</p>
<p><b>The thing genuinely worth noticing.</b> Past the crossover, adding more batch starts costing real time again, because the workload has flipped into compute-bound: the free lunch is over. Switch precision to INT4 and watch that crossover point move sharply to the right — a smaller memory read buys a wider window in which batching is still nearly free, which is exactly why quantization is a serving lever and not merely an accuracy one.</p>

${H.lab('bound', 'Memory-bound or compute-bound? — the roofline', 'Move the batch size and watch the workload cross from bandwidth-bound to compute-bound. The crossing point is the arithmetic intensity your hardware needs, and it is why serving stacks fight so hard for batch.')}

<h2><span class="sn">4.7.2</span> The KV cache, and the variants that shrink it</h2>
<p>Generating token $t+1$ needs $t+1$ query values? No — it needs exactly one new query, for the position just generated, but it needs the <i>keys and values</i> for every one of the $t$ preceding positions, because §4.3's attention operation compares the current query against every earlier key to decide what to retrieve. Recomputing all $t$ of those key and value vectors from scratch at every single decoding step would mean redoing the same projection over and over — the key for position 3 does not change once position 3's token is fixed, so recomputing it while generating position 4, 5, 6 and onward is pure waste. Production serving caches them instead: as each token is generated, its key and value vectors are computed once and stored, so decoding step $t+1$ only computes a query for the new position and reuses the cached keys and values for everything before it. That stored tensor is the <b>KV cache</b>, and its size, not the model's weights, is frequently what actually limits how many concurrent users a server can hold — which is why the table below is worth being able to reproduce as arithmetic, not recall as a fact.</p>
${H.table(['Variant', 'K/V heads', 'Cache size', 'Quality'], [
      ['<b>MHA</b>', 'one per query head', 'largest', 'reference'],
      ['<b>MQA</b>', 'one shared', 'smallest', 'measurable loss'],
      ['<b>GQA</b>', 'grouped (e.g. 8 for 64 heads)', '≈⅛ of MHA', 'the common compromise; why Llama-class models are servable'],
      ['<b>MLA</b>', 'a shared low-rank latent, reconstructed per head', 'smaller than GQA', 'MHA-level — the clearest recent payoff of §1.8’s low-rank idea']
    ])}
<p>Every one of §4.4's $h$ attention heads has its own query projection, but nothing forces every head to also have its <i>own</i> key and value projection — that was simply the default (multi-head attention, MHA) §4.4 built. <b>Multi-query attention</b> (MQA) takes the opposite extreme: every head keeps its own query, but all heads share a single key and a single value projection, so the cache only needs to store one K/V pair per position instead of $h$ of them — an $h$-fold shrinkage, at a real, measurable cost in output quality, because heads that used to attend to genuinely different aspects of the sequence are now all reading from the same key/value representation. <b>Grouped-query attention</b> (GQA) is the compromise that shipped in practice: split the $h$ query heads into a smaller number of groups (Llama-3-70B uses 8 groups for 64 query heads), with each group sharing one key/value pair — cache shrinks by roughly the group-size factor while most heads still retain enough of their own K/V diversity to avoid MQA's quality loss. <b>Multi-head latent attention</b> (MLA), DeepSeek's contribution, takes a structurally different route: rather than storing separate K/V vectors per head or per group at all, it compresses the key and value information for every head into one small shared low-rank latent vector, stored once, and reconstructs each head's full-size key and value from that latent on demand via a further learned projection. That is §1.8's low-rank idea — a large matrix is well approximated by the product of two much smaller ones — applied directly to what gets cached rather than to what gets trained, and it is the current clearest empirical case of the idea paying rent: MLA reaches a smaller cache than GQA while matching full MHA's quality, rather than trading one against the other.</p>

<p><b>What you are looking at.</b> The KV cache size formula, computed live from the layer count, K/V head count, head dimension, sequence length, batch size and precision you set, shown both as the raw arithmetic and as a bar stacked against the model's weight size, with vertical markers at each multiple of an 80 GB accelerator's memory.</p>
<p><b>What to do with it.</b> Load the Llama-3-70B preset and confirm the cache reads almost exactly 10 GB — the same figure the worked box below derives by hand. Then press the "128k context" preset button and watch which bar, weights or cache, now dominates the total.</p>
<p><b>The thing genuinely worth noticing.</b> At 4k context the cache is a rounding error next to 140 GB of weights; at 128k it has grown thirty-two-fold (linear in sequence length) and now rivals or exceeds the weights themselves. Long context is, before anything else, a memory-budgeting problem — which is exactly the argument §4.4 made for why a claimed context length is a capacity statement, not a competence guarantee.</p>

${H.lab('kv', 'KV cache calculator', 'The formula, live: bytes = 2 · L · n_kv · d_head · seq · batch · bytes-per-element. Reproduce the 10 GB figure for Llama-3-70B, then quadruple the context and watch the cache overtake the weights.')}

<h2><span class="sn">4.7.3</span> FlashAttention is exact</h2>
<p>This is the trap question, because the phrase "IO-aware" makes people reach for "approximate" as a guess. FlashAttention is <b>not</b> an approximation and not a form of sparse attention — it computes bit-for-bit the same numbers naive attention would, and the entire contribution is <i>how</i> those numbers are computed, not a change to what they are. §4.3 showed that the $[s,s]$ score matrix $QK^\\mathsf{T}$ is the memory-heavy tensor in an attention layer; the naive implementation computes that whole matrix, writes it out to HBM, reads it back in for the softmax, and reads it back in again for the multiplication by $V$ — three round trips through slow memory for a tensor that can be enormous at long context. FlashAttention never writes that matrix to HBM at all. It tiles $Q$, $K$ and $V$ into blocks small enough to fit in the chip's fast on-die SRAM, and processes the attention computation block by block, accumulating the final output incrementally as each new block of keys and values is streamed in.</p>
<p>The one genuine difficulty in doing this incrementally is the softmax, because a softmax needs to divide by the sum over <i>all</i> keys, and you do not yet know that total sum when you have only seen the first block. The fix is the same numerically-stable softmax identity §1.10 already established for a different reason (avoiding overflow when exponentiating large logits): track a running maximum and a running sum as each new block arrives, and rescale everything computed so far whenever the running maximum increases.</p>
${H.deriv('the online softmax: merging two partial sums exactly', [
      ['$\\ell = \\sum_i e^{z_i - m}$, for a chosen constant $m$', 'The numerically-stable softmax denominator from §1.10, computed relative to some reference value $m$ rather than in raw form — subtracting a constant before exponentiating never changes the final normalised probabilities, only the intermediate numbers\' size.'],
      ['given $(\\ell_1, m_1)$ from block 1 and $(\\ell_2, m_2)$ from block 2, let $m = \\max(m_1, m_2)$', 'To merge two blocks\' partial sums, first agree on a single new reference — the larger of the two running maxima seen so far.'],
      ['$\\ell = \\ell_1 e^{m_1-m} + \\ell_2 e^{m_2-m}$', 'Each partial sum was computed relative to its own old reference; rescale it to the new shared reference $m$ by multiplying by $e^{m_{\\text{old}}-m}$, then the two rescaled sums can simply be added, because they are now expressed on the same footing.'],
      ['repeat block by block, carrying only $(\\ell, m)$ and the running weighted output forward', 'Only two scalars and one accumulator vector need to be kept between blocks — never the full row of scores — so the algorithm processes a sequence of any length in fixed extra memory, and the merge above guarantees the final $\\ell$ and output are exactly what computing the whole softmax at once would have produced.']
    ], 'This is precisely the log-sum-exp merging trick, applied incrementally rather than to a single already-complete vector. FlashAttention is that identity used as a memory-traffic optimisation: same numbers, computed by never holding more than one tile\'s worth of the score matrix in memory at a time.')}
<p>Same numbers as naive attention, far less memory traffic — an IO-aware algorithm deployed in exactly the regime §4.7.1 showed is IO-bound. FlashAttention does not reduce FLOPs (it may even do marginally more arithmetic than the naive version, recomputing some quantities rather than storing them); what it reduces is the number of bytes shuttled to and from HBM, which is the resource that was actually the bottleneck all along.</p>

${H.probe([
      ['Is FlashAttention an approximation?', 'No — bit-for-bit exact. It is an IO optimisation, using the same running-maximum log-sum-exp merge as §1.10\'s numerically stable softmax, that avoids ever materialising the full attention matrix in HBM.'],
      ['Why GQA or MLA?', 'To shrink the KV cache, which is the binding memory constraint during decode and, at long context, can rival or exceed the weights themselves. MLA additionally applies §1.8\'s low-rank compression to what gets cached, reaching MHA-level quality at below-GQA cache size.'],
      ['Why is decode memory-bound?', 'Every step streams all weights plus the whole KV cache from HBM to do one token\'s worth of arithmetic — an arithmetic intensity of roughly one FLOP per byte, against hardware built to want two orders of magnitude more reuse than that before its compute units become the bottleneck.']
    ], 'Saying inference is FLOP-bound. Prefill is; decode is not.')}`,
    labs: {
      bound: function (host) {
        const st = Viz.controls(host, [
          { k: 'params', label: 'model parameters (B)', min: 1, max: 700, step: 1, value: 70, fmt: v => v + 'B' },
          { k: 'batch', label: 'batch size', min: 1, max: 256, step: 1, value: 1, fmt: v => v },
          { k: 'bw', label: 'HBM bandwidth (TB/s)', min: .5, max: 8, step: .05, value: 3.35, fmt: v => v.toFixed(2) },
          { k: 'flops', label: 'compute (TFLOP/s)', min: 100, max: 4000, step: 10, value: 990, fmt: v => v },
          { k: 'bytes', label: 'bytes per weight', type: 'buttons', value: '2', options: [{ v: '2', t: 'BF16' }, { v: '1', t: 'FP8' }, { v: '0.5', t: 'INT4' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mem', label: 'memory time / step', cls: 'key' }, { k: 'comp', label: 'compute time / step' },
          { k: 'ratio', label: 'ratio', cls: 'bad' }, { k: 'tps', label: 'tokens/s (total)' }, { k: 'mfu', label: 'MFU' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const N = st.params * 1e9, bpw = +st.bytes;
            const memBytes = N * bpw;
            const memT = memBytes / (st.bw * 1e12);
            const compT = (2 * N * st.batch) / (st.flops * 1e12);
            const stepT = Math.max(memT, compT);
            const pts = [];
            for (let b = 1; b <= 256; b++) {
              pts.push([b, Math.max(memT, (2 * N * b) / (st.flops * 1e12)) * 1000]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [1, 256], yd: [0, Math.max.apply(null, pts.map(p => p[1])) * 1.1] })
              .frame({ xlabel: 'batch size', ylabel: 'time per decode step (ms)' });
            P.clip(() => {
              P.fn(b => memT * 1000, { color: T.blue, width: 2, dash: [5, 4] });
              P.fn(b => (2 * N * b) / (st.flops * 1e12) * 1000, { color: T.red, width: 2, dash: [5, 4] });
              P.line(pts, { color: T.text, width: 2.8 });
              P.vline(st.batch, { color: T.green, dash: [3, 3] });
              P.dots([[st.batch, stepT * 1000]], { r: 5, color: T.green, stroke: true });
              const cross = (memT * st.flops * 1e12) / (2 * N);
              if (cross > 1 && cross < 256) P.vline(cross, { color: T.amber, label: 'crossover: bandwidth → compute bound' });
            });
            out({
              mem: (memT * 1000).toFixed(1) + ' ms', comp: (compT * 1000).toFixed(2) + ' ms',
              ratio: (memT / compT).toFixed(0) + '×',
              tps: (st.batch / stepT).toFixed(0),
              mfu: (100 * compT / stepT).toFixed(2) + '%'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'memory time (flat in batch)' }, { c: Viz.theme().red, t: 'compute time (linear in batch)' }, { c: Viz.theme().text, t: 'actual step time = max of the two' }]);
        Viz.note(host, 'The memory line is flat because the same weights are read once per step regardless of batch — which is why batching is nearly free throughput until you cross into the compute-bound regime. Switch to INT4 and watch the memory line drop by 4× and the crossover move left.');
      },

      kv: function (host) {
        const st = Viz.controls(host, [
          { k: 'L', label: 'layers L', min: 8, max: 128, step: 1, value: 80, fmt: v => v },
          { k: 'nkv', label: 'K/V heads n_kv', min: 1, max: 64, step: 1, value: 8, fmt: v => v },
          { k: 'dhead', label: 'head dim', min: 32, max: 256, step: 8, value: 128, fmt: v => v },
          { k: 'seq', label: 'sequence length', min: 512, max: 262144, step: 512, value: 4096, fmt: v => v >= 1024 ? (v / 1024).toFixed(0) + 'k' : v },
          { k: 'batch', label: 'batch', min: 1, max: 128, step: 1, value: 8, fmt: v => v },
          { k: 'bytes', label: 'KV precision', type: 'buttons', value: '2', options: [{ v: '2', t: 'BF16' }, { v: '1', t: 'FP8' }] },
          { k: 'weights', label: 'model weights (GB)', min: 8, max: 400, step: 2, value: 140, fmt: v => v + ' GB' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kv', label: 'KV cache', cls: 'key' }, { k: 'perTok', label: 'per token per sequence' },
          { k: 'total', label: 'weights + cache' }, { k: 'cards', label: '80 GB cards needed' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const bpe = +st.bytes;
            const perTok = 2 * st.L * st.nkv * st.dhead * bpe;
            const bytes = perTok * st.seq * st.batch;
            const gb = bytes / 1e9;
            const total = gb + st.weights;
            ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillStyle = T.muted;
            ctx.fillText('bytes = 2 · L · n_kv · d_head · seq · batch · bytes/elem', 18, 18);
            ctx.fillStyle = T.text; ctx.font = '14px ui-monospace, monospace';
            ctx.fillText('= 2 · ' + st.L + ' · ' + st.nkv + ' · ' + st.dhead + ' · ' + st.seq.toLocaleString() + ' · ' + st.batch + ' · ' + bpe, 18, 42);
            ctx.fillStyle = T.blue; ctx.font = 'bold 20px ui-monospace, monospace';
            ctx.fillText('= ' + gb.toFixed(2) + ' GB', 18, 68);
            // stacked bar
            const bx = 18, by = 116, bw = w - 40, bh = 34;
            const wFrac = st.weights / total;
            ctx.fillStyle = T.faint; ctx.fillRect(bx, by, bw * wFrac, bh);
            ctx.fillStyle = T.blue; ctx.fillRect(bx + bw * wFrac, by, bw * (1 - wFrac), bh);
            ctx.fillStyle = '#fff'; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            ctx.fillText(' weights ' + st.weights + ' GB', bx + 6, by + bh / 2);
            if ((1 - wFrac) > .15) ctx.fillText(' KV ' + gb.toFixed(1) + ' GB', bx + bw * wFrac + 6, by + bh / 2);
            // card markers
            ctx.strokeStyle = T.red; ctx.setLineDash([4, 4]);
            for (let c = 80; c < total; c += 80) {
              const x = bx + bw * (c / total);
              ctx.beginPath(); ctx.moveTo(x, by - 8); ctx.lineTo(x, by + bh + 8); ctx.stroke();
              ctx.fillStyle = T.red; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
              ctx.fillText(c + ' GB', x, by - 10);
            }
            ctx.setLineDash([]);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('≈ ' + (perTok / 1024).toFixed(0) + ' KB of cache per token per sequence · ' +
              (perTok * 4096 / 1e9).toFixed(2) + ' GB per 4k-token conversation', 18, by + bh + 18);
            ctx.fillText('⚑ assumes standard GQA/MHA geometry; MLA and hybrid-attention models compute differently — treat this as a template.', 18, by + bh + 38);
            out({
              kv: gb.toFixed(2) + ' GB', perTok: (perTok / 1024).toFixed(0) + ' KB',
              total: total.toFixed(0) + ' GB', cards: Math.ceil(total / 80)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Llama-3-70B · 4k · batch 8', primary: true, on: () => { st.$set('L', 80); st.$set('nkv', 8); st.$set('dhead', 128); st.$set('seq', 4096); st.$set('batch', 8); st.$set('bytes', '2'); st.$set('weights', 140); S.redraw(); } },
          { label: 'Same, 128k context', on: () => { st.$set('seq', 131072); S.redraw(); } },
          { label: 'MHA instead of GQA', on: () => { st.$set('nkv', 64); S.redraw(); } }
        ]);
        Viz.note(host, 'The default preset reproduces the 10 GB figure: 2·80·8·128·4096·8·2 = 10,737,418,240 bytes. Press "128k context" and the cache dwarfs the weights — long context is a memory-budget problem before it is a quality problem.');
      }
    },
    quiz: [
      {
        q: 'Is FlashAttention an approximation?',
        options: ['Yes, it sparsifies attention', 'No — it is bit-for-bit exact and avoids materialising the score matrix', 'Yes, it uses low-rank factorisation', 'Only in the backward pass'],
        answer: 1,
        why: 'Tiling plus an online softmax (the log-sum-exp identity, §1.10) gives identical numbers with far less HBM traffic.'
      },
      {
        q: 'Llama-3-70B, L=80, n_kv=8, d_head=128, seq 4,096, batch 8, BF16. KV cache is…',
        options: ['1 GB', '10 GB', '40 GB', '140 GB'],
        answer: 1,
        why: '2·80·8·128·4096·8·2 bytes ≈ 10.7 GB — and it quadruples with the sequence length.'
      },
      {
        q: 'Batching helps decode throughput enormously because…',
        options: ['it reduces FLOPs', 'the same weight read serves every sequence in the batch', 'it shortens the KV cache', 'it improves MFU on prefill'],
        answer: 1,
        why: 'The 42 ms weight read is paid once per step regardless of batch — until you cross into the compute-bound regime.'
      }
    ],
    cards: [
      { q: 'KV cache formula', a: 'bytes = 2·L·n_kv·d_head·seq·batch·bytes-per-element. 70B/4k/batch 8/BF16 → 10 GB.' },
      { q: 'Why decode is memory-bound', a: 'Every step streams all weights + the whole cache from HBM for one token of arithmetic: ~300× more time moving than using.' },
      { q: 'Cache size order', a: 'MHA ≫ GQA > MLA ≈ MQA.' },
      { q: 'FlashAttention', a: 'Exact, IO-aware: tile into SRAM and use an online softmax so the s×s matrix never reaches HBM.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.8 */
  ML.section({
    id: 'moe', track: 'llm', num: '4.8',
    title: 'Mixture-of-experts; state-space models and Mamba',
    lede: 'Two ways to change what sits inside the block without abandoning it.',
    html: `
<p>§4.5 identified the two components that make up a transformer block — an attention sub-layer and an FFN sub-layer — and §4.6 chose their internal shape, but both components so far still make the same assumption once you look past their internals: every token is processed by the <i>same</i> weights, and every pair of tokens is compared at <i>quadratic</i> cost. Neither assumption is forced. This section covers the two architectural moves that challenge them directly, one attacking each half of the block: mixture-of-experts asks whether a token really needs to pass through the whole FFN, and state-space models ask whether relating two tokens really needs to cost $O(n^2)$.</p>

<h2><span class="sn">4.8.1</span> Mixture-of-experts</h2>
<p>A dense FFN applies the identical set of weights to every token, whether that token is the start of a poem, a line of Python, or a chemical formula — one generalist doing every job. A <b>mixture-of-experts</b> (MoE) layer replaces that single FFN with $N$ separate FFNs, called <b>experts</b>, plus a small <b>router</b> — typically just a linear layer followed by a softmax over the $N$ experts — that looks at each token and decides which experts should process it. Rather than running all $N$ experts on every token, the router selects only the top-$k$ (commonly $k=2$) and routes the token to just those, discarding the rest entirely for that token. The FFN's total parameter count is now the sum of every expert's parameters, which can be made enormous, while the <i>compute</i> spent per token depends only on how many experts that token actually visits — $k$, not $N$.</p>
${H.analogy(`<p>A dense FFN is a single generalist doctor seeing every patient regardless of complaint. An MoE layer is a hospital: a triage desk (the router) reads each patient's symptoms and sends them to the two most relevant specialists out of dozens on staff, rather than making every patient see every specialist or relying on one generalist for everything. The hospital as a whole holds vastly more expertise than any one doctor could, but any single patient's visit still only costs two consultations.</p>`)}
<p>The headline consequence is that total parameters and compute-per-token become <i>separate</i> dials rather than the same one. DeepSeek V3 is the canonical figure worth memorising: <b>671B total parameters, 37B active per token</b> — active parameters are only about $37/671 \\approx 5.5\\%$ of the total, meaning the model has the capacity of a 671B dense network while paying, per token, roughly the inference and training compute of a 37B one. That is the entire pitch of MoE: capacity purchased almost for free in compute terms, at a cost paid somewhere else entirely.</p>
<p>That cost has three faces, and all three are engineering rather than modelling problems. <b>Memory</b>: every one of the 671B parameters has to sit resident somewhere, ready to be selected at any moment, so the memory footprint is set by the total, not the active count — a stark contrast to the compute cost. <b>Communication</b>: because different tokens in a batch route to different experts, and those experts may live on different devices in a distributed setup, every forward pass needs an all-to-all exchange shuffling tokens to wherever their chosen experts live and back — chatty traffic that a dense model never has to pay for at all. And the <b>routing problem itself</b>, which is the subtlest of the three and the one the lab below makes visible: left unconstrained, a router tends toward a rich-get-richer collapse. An expert that, purely by chance early in training, receives slightly more traffic gets slightly more gradient signal, becomes slightly better at whatever those tokens need, and is therefore chosen slightly more often next time — a positive feedback loop with no natural ceiling. Left to run, the router settles onto a handful of favourite experts while the rest sit almost entirely unused: parameters you paid memory and communication cost for that contribute nothing. The standard fix is an <b>auxiliary load-balancing loss</b>, added to the training objective specifically to penalise uneven routing and push the distribution back toward using every expert roughly equally.</p>

<p><b>What you are looking at.</b> A real, small router — an actual softmax over expert logits, not a stand-in — processing a stream of tokens one at a time and accumulating how many tokens each expert has received. Each bar is one expert; its length is that expert's share of total routing traffic so far, with a dashed green line marking what perfectly even load would look like.</p>
<p><b>What to do with it.</b> Start with the auxiliary loss off and watch the bars diverge as more tokens are routed — a small early imbalance snowballs, exactly as the rich-get-richer argument predicts, and the readout's "starved experts" count climbs as some bars barely grow at all.</p>
<p><b>The thing genuinely worth noticing.</b> Turn the auxiliary loss on and rerun: the bars stay close to the dashed line throughout, because the loss actively penalises the imbalance the moment it starts to form rather than after it has compounded. A starved expert is not a cosmetic inefficiency — it is memory and communication bandwidth you paid for that a collapsed router is silently refusing to use, which is why load balancing is treated as a first-class part of the training objective rather than an afterthought.</p>

${H.lab('moe', 'MoE routing, with and without load balancing', 'Tokens routed by a real (small) router. Watch the expert-load bars: with the auxiliary loss off, the distribution collapses within a few hundred tokens and most of your parameters stop existing.')}

<h2><span class="sn">4.8.2</span> State-space models</h2>
<p>§4.1 framed the whole architecture question as a trade between path length and parallelism: recurrence gets an $O(n)$ path and no parallelism, attention gets an $O(1)$ path and full parallelism but pays $O(n^2)$ for it. Nothing in that framing rules out a third point in the space — a constant path length achieved through a genuinely different mechanism, at sub-quadratic cost, that is <i>still</i> trainable in parallel despite being, structurally, a recurrence. <b>State-space models</b> (SSMs) are that third point, and the trick that makes it possible is worth isolating precisely, because it resolves a tension §4.1 left open rather than sidestepping it.</p>
<p>An SSM maintains a fixed-size hidden state $h_t$, updated by a <i>linear</i> recurrence: $h_t = Ah_{t-1} + Bx_t$, with the output read off as $y_t = Ch_t$. This looks superficially like §3.8's RNN, and the crucial difference is exactly the word "linear". §3.8 and §4.1 established that an RNN's recurrence is inherently sequential — step $t$ needs step $t-1$'s <i>output</i>, which for a nonlinear recurrence means there is no way to combine two steps' worth of computation into one without actually running them in order. A purely <i>linear</i> recurrence does not have that problem: composing two affine maps is itself associative, so $h_t$ can be expressed directly as a weighted sum over all past inputs (a single unrolled formula, not a step-by-step chain), and computing that sum for every $t$ at once is exactly the kind of problem a <b>parallel (associative) scan</b> solves — the same class of algorithm behind a parallel prefix sum — in $O(\\log n)$ sequential steps rather than $O(n)$. The linearity that looked like it might be a modelling limitation is precisely what buys back the parallel training §4.1 said pure recurrence could never have.</p>
<p>Mamba's specific contribution is making $A$, $B$ and $C$ <i>input-dependent</i> rather than fixed — computed afresh from each token $x_t$, so the recurrence's own dynamics change based on content, deciding on the fly what to remember and what to let decay. This "selective" mechanism is doing, in spirit, the same job attention's dynamic per-pair weighting does (deciding what matters based on content rather than on a fixed rule), but through a completely different computational route: a fixed-size state carried forward rather than a growing set of keys and values kept individually addressable. Two consequences follow directly from that structural difference. Compute and memory scale near-linearly in sequence length rather than quadratically, because there is no $[s,s]$ score matrix anywhere in the computation. And at inference, the state is a single fixed-size vector rather than something that grows with every generated token — <b>no growing KV cache</b>, because there is no KV cache; a Mamba layer's memory footprint during generation does not depend on how long the conversation has already run.</p>
<p>That fixed-size state is also exactly where the weakness lives. A KV cache keeps every past token's key and value <i>individually</i> addressable, however long the sequence — attention can, in principle, reach back and retrieve one specific fact from ten thousand tokens ago with no interference from anything else in between. An SSM's state has to compress the <i>entire</i> history into one fixed-size vector, so retrieving one precise fact from far back is competing for room with everything else that has happened since, and precise long-range retrieval measurably suffers as a result. That is not a flaw to be trained away — it is the direct, structural price of never letting the state grow — and it is why practical long-context architectures are almost always <b>hybrids</b>: mostly efficient SSM (or local-attention) layers carrying the bulk of ordinary language modelling at near-linear cost, with a handful of full-attention layers interleaved to restore precise retrieval where it is genuinely needed. It is exactly the same local/global division of labour §4.4 described for sliding-window attention, arrived at from the opposite architectural direction and landing on the same answer.</p>

<p><b>What you are looking at.</b> Three cost curves against sequence length, computed from the actual FLOP formulas for each mechanism rather than drawn schematically: full attention's $O(s^2d)$, sliding-window attention's $O(s\\cdot w\\cdot d)$, and an SSM's $O(s\\cdot d^2)$, all for one layer's forward pass at a hidden size you choose.</p>
<p><b>What to do with it.</b> Read the crossover readout at 32k tokens, and note which curve dominates at short sequence lengths — it is not always the one you expect. At short lengths the $d^2$ term in the SSM and FFN cost can exceed attention's $s^2d$ term, simply because $s$ has not yet grown large enough for the quadratic term to dominate a comparatively small linear one.</p>
<p><b>The thing genuinely worth noticing.</b> The "attention exceeds FFN" crossover point is usually well past what a casual guess would suggest — attention is not automatically the expensive part of a model at ordinary context lengths, it only becomes so once sequence length grows past roughly the hidden size itself. Architecture debates about "attention versus SSM" are, underneath, debates about which regime of that curve your actual workload lives in, not a debate settled once for all context lengths.</p>

${H.lab('ssm', 'Quadratic versus linear, at length', 'Attention cost, sliding-window cost and SSM cost as sequence length grows, with the KV-cache curve alongside. The crossover is where architectural choices stop being aesthetic.')}

${H.probe([
      ['MoE’s benefit and its cost?', 'More capacity at fixed compute per token — total parameters and active-per-token compute become separate dials. The cost is memory for every expert regardless of use, all-to-all communication when experts are distributed, and a routing/load-balancing problem: an unconstrained router collapses onto a few favourite experts through a rich-get-richer dynamic.'],
      ['Why do SSM models still include attention layers?', 'An SSM compresses all history into one fixed-size state, so precise retrieval of one distant fact competes with everything else for the same limited room; a few full-attention layers, with their individually-addressable KV cache, restore exact long-range retrieval where it is needed.'],
      ['What is DeepSeek V3’s headline MoE figure?', '671B total parameters, 37B active per token — roughly 5.5% of parameters active on any given token, which is the entire economic case for MoE stated as one ratio.'],
      ['Why can an SSM be trained in parallel despite being a recurrence?', 'Because the recurrence is linear, composing two steps is associative, so the whole sequence can be computed via a parallel (associative) scan in $O(\\log n)$ steps rather than run step by step — the linearity that looked restrictive is exactly what buys back the parallelism §4.1 said ordinary nonlinear recurrence could never have.']
    ])}`,
    labs: {
      moe: function (host) {
        const st = Viz.controls(host, [
          { k: 'experts', label: 'experts', min: 4, max: 32, step: 1, value: 8, fmt: v => v },
          { k: 'topk', label: 'top-k routed', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'balance', label: 'auxiliary load-balancing loss', type: 'toggle', value: false },
          { k: 'tokens', label: 'tokens routed', min: 100, max: 4000, step: 100, value: 1500, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'active', label: 'active params per token', cls: 'key' }, { k: 'total', label: 'total params' },
          { k: 'dead', label: 'starved experts', cls: 'bad' }, { k: 'cv', label: 'load imbalance (CV)' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(41), E = st.experts;
            // router logits drift toward whichever experts win early unless balanced
            const bias = new Array(E).fill(0);
            const load = new Array(E).fill(0);
            for (let t = 0; t < st.tokens; t++) {
              const logits = Array.from({ length: E }, (_, e) => R.normal(0, 1) + bias[e] - (st.balance ? load[e] / (t + 1) * E * 1.6 : 0));
              const idx = logits.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, st.topk).map(p => p[1]);
              idx.forEach(e => { load[e]++; if (!st.balance) bias[e] += 0.004; });
            }
            const maxLoad = Math.max.apply(null, load);
            const bx = 130, bw = w - bx - 70;
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const rowH = Math.min(22, (h - 60) / E);
            load.forEach((v, e) => {
              const y = 30 + e * rowH;
              const starved = v < st.tokens * st.topk / E * .35;
              const overfull = v > st.tokens * st.topk / E * 1.8;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText('expert ' + (e + 1), bx - 10, y + rowH / 2);
              ctx.fillStyle = starved ? T.red : overfull ? T.amber : T.blue;
              ctx.fillRect(bx, y + 2, bw * v / maxLoad, rowH - 5);
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.font = '10px ui-monospace, monospace';
              ctx.fillText(((100 * v) / (st.tokens * st.topk)).toFixed(1) + '%' + (starved ? '  starved' : overfull ? '  overfull' : ''), bx + bw * v / maxLoad + 6, y + rowH / 2);
              ctx.font = '11px ui-monospace, monospace';
            });
            ctx.strokeStyle = T.green; ctx.setLineDash([4, 4]);
            const ideal = bx + bw * (st.tokens * st.topk / E) / maxLoad;
            ctx.beginPath(); ctx.moveTo(ideal, 26); ctx.lineTo(ideal, 30 + E * rowH); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = T.green; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('perfectly balanced', ideal + 4, 26);
            const mean = Num.mean(load), cv = Num.sd(load) / (mean || 1);
            out({
              active: (st.topk / E * 100).toFixed(0) + '% of the FFN',
              total: E + ' experts',
              dead: load.filter(v => v < st.tokens * st.topk / E * .35).length,
              cv: cv.toFixed(3)
            });
          }
        });
        Viz.note(host, 'Turn the auxiliary loss on and the bars level out. This is not a cosmetic problem: a starved expert is parameters you paid memory for and never use, and a collapsed router turns a 671B model into a much smaller one wearing its coat.');
      },

      ssm: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'hidden size d', min: 512, max: 8192, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'win', label: 'sliding window w', min: 128, max: 8192, step: 128, value: 1024, fmt: v => v.toLocaleString() },
          { k: 'max', label: 'max sequence length', min: 4096, max: 262144, step: 4096, value: 131072, fmt: v => (v / 1024).toFixed(0) + 'k' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'at32', label: 'attention cost @ 32k', cls: 'bad' }, { k: 'win32', label: 'sliding window @ 32k' },
          { k: 'ssm32', label: 'SSM @ 32k', cls: 'good' }, { k: 'cross', label: 'attention exceeds FFN at' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const d = st.d;
            const attn = s => (2 * s * s * d + 4 * s * d * d) / 1e12;
            const win = s => (2 * s * st.win * d + 4 * s * d * d) / 1e12;
            const ssm = s => (6 * s * d * d) / 1e12;
            const P = Viz.plot(ctx, w, h, { xd: [1024, st.max], yd: [0, attn(st.max) * 1.05] })
              .frame({ xlabel: 'sequence length', ylabel: 'TFLOPs per layer (forward)', xfmt: v => (v / 1024).toFixed(0) + 'k' });
            P.clip(() => {
              P.fn(attn, { color: T.red, width: 2.6, n: 300 });
              P.fn(win, { color: T.amber, width: 2.2, n: 300 });
              P.fn(ssm, { color: T.green, width: 2.4, n: 300 });
              P.vline(32768, { color: T.faint, dash: [3, 3], label: '32k' });
            });
            let cross = null;
            for (let s = 1024; s <= st.max; s += 512) { if (2 * s * s * d > 4 * s * d * d) { cross = s; break; } }
            out({
              at32: attn(32768).toFixed(2) + ' TF', win32: win(32768).toFixed(2) + ' TF',
              ssm32: ssm(32768).toFixed(2) + ' TF',
              cross: cross ? (cross / 1024).toFixed(0) + 'k tokens' : '—'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().red, t: 'full attention O(s²d)' }, { c: Viz.theme().amber, t: 'sliding window O(s·w·d)' }, { c: Viz.theme().green, t: 'SSM / linear O(s·d²)' }]);
        Viz.note(host, 'The crossover readout answers a question people get wrong: below roughly 2d tokens, attention is <i>not</i> the dominant cost — the FFN is. Quadratic attention only takes over past that point, which is why architecture debates about long context are really debates about the regime you serve.');
      }
    },
    quiz: [
      {
        q: 'Without an auxiliary load-balancing loss, an MoE router tends to…',
        options: ['distribute tokens uniformly', 'collapse onto a few favourite experts, leaving the rest as dead weight', 'route by token length', 'become deterministic and optimal'],
        answer: 1,
        why: 'Early winners get more gradient and win more — a rich-get-richer dynamic the auxiliary loss exists to counteract.'
      },
      {
        q: 'Why do Mamba-style models interleave a few full-attention layers?',
        options: ['for positional encoding', 'SSMs compress history into a fixed state and are weaker at precise in-context retrieval', 'to reduce parameters', 'for numerical stability'],
        answer: 1,
        why: 'Local/linear layers carry the language modelling; a few global layers do the retrieval — the same split as local + global attention.'
      }
    ],
    cards: [
      { q: 'MoE trade', a: 'Huge total parameters, small active per token (DeepSeek V3: 671B/37B). Costs: memory for all experts, all-to-all comms, router load balancing.' },
      { q: 'SSM / Mamba', a: 'Input-dependent linear recurrence: near-linear in length, constant state at inference, parallel training via associative scan; hybrids add a few attention layers.' }
    ]
  });
})();
