/* ============================================================
   PART 5 — Applied, continued: chunking (5.9), vector indexes
   (5.10), evals that do not lie (5.11), MLOps (5.12).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 5.9 */
  ML.section({
    id: 'chunking', track: 'applied', num: '5.9', level: 2,
    title: 'Chunking and the data layer',
    lede: 'Most RAG systems that fail do not fail at retrieval or generation. They fail because the answer was split across two chunks, or buried in a chunk about something else, or lost when a PDF table became a wall of numbers. This is the unglamorous half of §5.1, and it is the half that decides the outcome.',
    prereq: ['rag'],
    related: ['rag', 'vector-search', 'evals'],
    html: `
${H.tldr([
      'Chunk size trades <b>precision</b> (small chunks, the embedding is about one thing) against <b>completeness</b> (large chunks, the whole answer is inside one). There is no universal answer; there is a measurement.',
      'Overlap is the cheapest insurance against splitting an answer in half. 10–20% of the chunk size is the usual setting.',
      '<b>Parse before you chunk.</b> A table flattened into prose, a header lost, a footnote inlined mid-sentence — these destroy more retrieval quality than any embedding-model choice.'
    ])}

<h2><span class="sn">5.9.1</span> The pipeline nobody draws</h2>
<p>Every chunking decision is a bet about where meaning lives. A five-word chunk is almost always <i>about</i> exactly one thing, so its embedding is a sharp, unambiguous point in vector space — that is the precision half of the trade. A five-hundred-word chunk is almost always <i>complete</i>: if an answer exists in the document, it is probably somewhere inside this one chunk. But it is now about six different things at once, so its embedding sits near the average of all of them, and a sharp query about just one of those things may not land anywhere close to it. Push the chunk size up and you buy completeness at the cost of precision; push it down and you buy precision at the cost of completeness. There is no chunk size that buys both, because "small enough to be about one thing" and "large enough to contain the whole answer" are only the same number by coincidence.</p>
<p>It is tempting to treat this as an ordinary tuning problem — sweep chunk size on a validation set, pick the number that wins, move on. That undersells what is happening. However you tile a document into fixed-width windows, some answer somewhere in it is longer than your window or straddles the exact point between two windows, and no single choice of size and overlap fixes that for every possible answer. Overlap does not eliminate the boundary problem. It buys a guarantee, but only for spans up to a length you can compute in advance — and the computation is worth doing once, because it tells you exactly what you are and are not protected against.</p>
${H.deriv('the overlap that guarantees a span of length $L$ survives intact', [
      ['chunk $i$ covers tokens $[i s,\\, i s + C)$', 'chunks start $s$ tokens apart (the stride) and each spans $C$ tokens (the chunk size) — the definition of a sliding window.'],
      ['a span $[p,\\,p+L)$ sits fully inside chunk $i$ exactly when $i s \\le p$ and $p+L \\le is+C$', 'both ends of the span have to land inside the same window: expand "fully inside" into its two boundary conditions.'],
      ['solving both for $i$: $i \\in \\left[\\dfrac{p+L-C}{s},\\ \\dfrac{p}{s}\\right]$', 'divide each inequality through by $s$, and restate the first one so both bound $i$ from the same direction.'],
      ['that interval has width $\\dfrac{p}{s}-\\dfrac{p+L-C}{s}=\\dfrac{C-L}{s}$', 'subtract the lower bound from the upper bound. The $p$ cancels — the width does not depend on where the span starts.'],
      ['a valid whole-number chunk index exists whenever $\\dfrac{C-L}{s}\\ge 1$, i.e. $L \\le C-s$', 'a real interval of width $\\ge 1$ cannot avoid every integer — a standard pigeonhole fact, applied to the interval just derived.'],
      ['substitute the stride $s=C(1-o)$ for an overlap fraction $o$: $L \\le Co$', 'expand $C-s=C-C(1-o)=Co$: the guaranteed span length is exactly the overlap measured in tokens, not the overlap fraction itself.']
    ], 'Set $C=512$ tokens and $o=15\\%$ and the guarantee is $Co\\approx 77$ tokens: any answer of 77 tokens or fewer is captured whole in some chunk, wherever it happens to fall in the document. A 200-token answer has no such guarantee at these settings, however the boundaries land — and you rarely know the length of the answer you have not seen yet. That is the actual limit overlap runs into: buying a guarantee for longer spans means growing the overlap toward the chunk size itself, which means storing and embedding the same text two, three, four times over. Chunking a document is not a solved problem waiting for the right constant; it is a bet, and this is the arithmetic that tells you exactly how large a bet you are making.')}

${H.svg('the data layer, in order', '0 0 660 130', `
<defs><style>
.bx{fill:var(--panel);stroke:var(--line)}
.bx2{fill:color-mix(in oklab,var(--c1) 12%,var(--panel));stroke:var(--c1)}
.t{font:11px ui-sans-serif,system-ui;fill:var(--text);text-anchor:middle}
.s{font:9.5px ui-monospace,monospace;fill:var(--faint);text-anchor:middle}
.a{stroke:var(--faint);fill:none;stroke-width:1.4;marker-end:url(#h1)}
</style>
<marker id="h1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--faint)"/></marker></defs>
<g>
${['source', 'parse', 'clean', 'chunk', 'enrich', 'embed', 'index'].map((s, i) => {
      const x = 12 + i * 93;
      const sub = ['pdf, html, db', 'layout-aware', 'dedupe, strip', 'size + overlap', 'title, dates', 'model choice', 'ANN + filters'][i];
      const cls = (i === 3 || i === 1) ? 'bx2' : 'bx';
      return '<rect class="' + cls + '" x="' + x + '" y="34" width="80" height="46" rx="8"/>' +
        '<text class="t" x="' + (x + 40) + '" y="56">' + s + '</text>' +
        '<text class="s" x="' + (x + 40) + '" y="70">' + sub + '</text>' +
        (i < 6 ? '<path class="a" d="M' + (x + 82) + ' 57 H' + (x + 91) + '"/>' : '');
    }).join('')}
<text class="s" x="330" y="106">the two highlighted stages account for most of the quality variance, and get the least attention</text>
</g>
`, 'Everything downstream is bounded by what happens here. A brilliant reranker cannot recover an answer that the parser destroyed.')}

<h2><span class="sn">5.9.2</span> Chunking strategies</h2>
${H.table(['Strategy', 'How', 'Good for', 'Cost'], [
      ['Fixed-size', '$n$ tokens with $m$ overlap', 'a baseline you should always measure against', 'splits mid-sentence'],
      ['<b>Recursive character</b>', 'split on paragraphs, then sentences, then words, until it fits', '<b>the sane default</b>', 'none, really'],
      ['Structural / layout', 'split on markdown headings, HTML sections, PDF blocks', 'documentation, contracts, anything with structure', 'needs a good parser'],
      ['Semantic', 'split where consecutive-sentence embedding similarity drops', 'unstructured prose', 'an embedding call per sentence'],
      ['<b>Hierarchical (small-to-big)</b>', 'embed small chunks, <i>return</i> their larger parent', '<b>usually the best quality/effort ratio</b>', 'two stores, slightly more code'],
      ['Late chunking', 'embed the long document once, then pool per chunk', 'preserves cross-chunk context in each vector', 'needs a long-context embedding model'],
      ['Proposition / atomic', 'an LLM rewrites the text into standalone facts', 'dense factual corpora, FAQ generation', 'an LLM call per document, and drift risk']
    ])}
<p>Read that table against the derivation above. Fixed-size chunking places its seams at arbitrary token counts, blind to whatever is sitting at that position — it is the purest form of the problem just derived, betting entirely on overlap to save you. Recursive character splitting is the default not because it is cleverer, but because it moves the seam to the cheapest available proxy for a semantic boundary: a paragraph break, then a sentence end, then a word, in that order of preference. Cutting at a paragraph break rarely bisects a single fact, because a paragraph already exists as its author's own answer to "where does this thought end?" Structural and semantic splitting push the same idea further, using an explicit document structure or a measured drop in sentence-to-sentence similarity as a better proxy than punctuation. None of these strategies escapes the limit the derivation above puts a number on — a paragraph can still be longer than your chunk — they just place the unavoidable cut somewhere it is statistically less likely to matter.</p>
${H.analogy(`<p>Small-to-big is a library, not a filing cabinet. A filing cabinet stores one fact per drawer, indexed by exactly what is inside it — precise, but you can only look up what you already knew to file separately. A library stores whole books, but the catalogue card for each one is short and specific: a title, a subject, a call number. You search the catalogue, which is precise because each card is about one thing, and the catalogue card points you to the whole book, which is complete because the book contains everything the catalogue card could not.</p>
<p>Small-to-big is that catalogue-card trick applied to retrieval: embed the small, specific unit — a sentence, a short paragraph — because a small unit makes a sharp catalogue card. Then, once it matches, hand the model the whole shelf it came from, not just the card. The card and the book are different sizes on purpose; nobody expects one object to do both jobs.</p>`)}
${H.key('Small-to-big is the trick worth knowing: index sentence-level chunks so the embedding is precise, but hand the language model the surrounding section so it has enough context to answer. Precision on retrieval, completeness on generation, and you stop having to choose.')}

<p><b>What you are looking at.</b> The panel below chunks a real four-section API document — rate limits, tiers, retries, support — using the current size and overlap settings, and colours each resulting chunk so the seams are visible directly in the text. Beneath the chunked view, three real questions about that document are checked against the chunks actually produced: a check mark means the answer text sits whole inside a single chunk; a cross means it does not, with the offending question named explicitly.</p>
<p><b>What to do with it.</b> Start at the defaults and confirm all three questions come back checked. Now drag chunk size down toward the bottom of its range: past a point, the retry question — the one that needs both "backoff" and "six" inside the same chunk, sixteen words apart in the source — starts failing, exactly the failure mode the derivation above puts a number on. Nudge overlap on its own at that same small size and notice the rescue is not clean or monotonic; a small change in overlap recomputes the stride and shifts every boundary in the document at once, so a specific document's actual behaviour is noisier than the worst-case guarantee derived above, which only promises success for <i>every possible</i> position, not for this one. Push size back up past 70 words and every answer is safely contained again — but now read the chunk text itself: each one spans four unrelated topics.</p>
<p><b>The thing genuinely worth noticing.</b> The "answers inside one chunk" counter never moves when you toggle <b>small-to-big</b>, because in this widget small-to-big only changes what would be handed to the model after a match, not the boundaries used for matching — that is a faithful reflection of how the trick works in production, where it improves the chances a small, precise chunk gets <i>retrieved</i> in the first place, something a presence check like this one cannot show you. What the toggle does move is the token-cost readout: turning it on roughly triples the tokens sent to the model for this document, which is the price tag on the trick from the analogy above, made concrete.</p>
${H.lab('chunk', 'Chunk a real document and watch retrieval succeed or fail', 'A document with three planted answers, one of which straddles a natural boundary. Move the chunk size and overlap and watch which answers survive intact — and what that does to measured hit rate.')}

<h2><span class="sn">5.9.3</span> Metadata is half the system</h2>
${H.checklist([
      '<b>Prepend the document title and section path to the chunk text before embedding.</b> A chunk that says "the limit is 4,096" is meaningless; "API Reference → Rate Limits → the limit is 4,096" is retrievable. This one change is frequently worth more than upgrading the embedding model.',
      '<b>Store dates and versions as filterable fields</b>, not as prose. "What is the current policy?" is a filter, not a similarity question.',
      '<b>Keep a stable chunk ID and a pointer back to the source span</b>, so a citation can be verified and a stale chunk can be deleted.',
      '<b>Record the parser and chunker version</b> on every chunk. When you change either, you need to know what to re-index.',
      '<b>Deduplicate.</b> Corpora built from wikis and exports are 20–40% near-duplicate, and duplicates crowd out diversity in the top $k$.'
    ])}
${H.pitfall('Tables are where naive pipelines die. A financial table flattened to "Revenue 2024 2025 1,204 1,455" is retrievable by nobody and misreadable by everyone. The workable options are: extract tables separately and store them as markdown or HTML; generate a one-sentence natural-language summary per table for the embedding while keeping the structured form for the answer; or route table questions to SQL over the structured source instead of to retrieval at all.')}

${H.worked('a chunking budget, worked', `
<p>A 40,000-document corpus averaging 3,000 tokens each = 120M tokens.</p>
<ul>
<li>At 512-token chunks with 15% overlap: ~270,000 chunks. Embedding at $0.02/M tokens ≈ <b>$2.80</b>, once.</li>
<li>Storage at 1,024-dimensional float32: 270k × 4 KB ≈ <b>1.1 GB</b> — and that is why quantised vectors (int8, or binary with rescoring) matter at scale: the same index in int8 is 270 MB.</li>
<li>Re-embedding when you change the model or chunker: the same $2.80 and a few hours. <b>Cheap.</b> Re-parsing 40,000 PDFs correctly: possibly weeks.</li>
</ul>
<p>The lesson in those three numbers: embeddings are nearly free and easy to redo, parsing is expensive and hard to redo. Spend your effort accordingly.</p>`)}

${H.probe([
      ['What chunk size should we use?', 'Whatever your evaluation set says. Start at 400–600 tokens with 10–15% overlap and recursive splitting, then measure recall@k against a labelled question set. Anyone who gives you a number without asking about your documents is guessing.'],
      ['A user asks a question whose answer spans three sections. What breaks?', 'Single-chunk retrieval. You need larger or hierarchical chunks, multi-chunk retrieval with a higher $k$, or query decomposition into sub-questions.'],
      ['Why prepend the section heading to each chunk?', 'The embedding is of the chunk text alone; without context, an anaphoric or elliptical chunk has no retrievable signal. It is one line of code and a large measured gain.'],
      ['When would you not chunk at all?', 'Short documents that fit whole; or when the model’s context is large enough and the corpus small enough to just include everything — which is often cheaper than a retrieval system (§5.2).']
    ])}`,
    labs: {
      chunk: function (host) {
        const DOC = ('# Rate limits\\n' +
          'The API applies per-organisation quotas. Requests are counted per minute and per day. ' +
          'Exceeding a quota returns HTTP 429 with a Retry-After header. ' +
          '## Standard tier\\n' +
          'Standard organisations may issue up to 4096 requests per minute. ' +
          'The daily ceiling is 2 million requests. Bursts above the per-minute limit are rejected immediately rather than queued. ' +
          '## Enterprise tier\\n' +
          'Enterprise agreements raise the per-minute limit to 60000 and remove the daily ceiling entirely. ' +
          'Enterprise customers also receive a dedicated capacity pool that is not shared with other organisations. ' +
          '## Retries\\n' +
          'Clients should retry with exponential backoff starting at one second and doubling, ' +
          'with full jitter, up to a maximum of six attempts. Retrying without backoff will extend the throttling window. ' +
          '## Support\\n' +
          'Quota increases are requested through the console and are reviewed within two business days.').split(/\\s+/);

        const ANSWERS = [
          { q: 'What is the standard per-minute limit?', need: ['4096'] },
          { q: 'What is the enterprise per-minute limit?', need: ['60000'] },
          { q: 'How should clients retry, and how many times?', need: ['backoff', 'six'] }   // deliberately spans a sentence boundary
        ];

        const st = Viz.controls(host, [
          { k: 'size', label: 'chunk size (words)', min: 8, max: 80, step: 2, value: 26, fmt: v => v },
          { k: 'ov', label: 'overlap', min: 0, max: .5, step: .05, value: .1, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'head', label: 'prepend the section heading', type: 'toggle', value: false },
          { k: 'small2big', label: 'small-to-big (return the parent section)', type: 'toggle', value: false }
        ], () => draw());

        const view = ML.el('div');
        host.appendChild(view);
        const out = Viz.readout(host, [
          { k: 'n', label: 'chunks', cls: 'key' },
          { k: 'hit', label: 'answers fully inside one chunk', cls: 'good' },
          { k: 'split', label: 'answers split across chunks', cls: 'bad' },
          { k: 'tok', label: 'tokens sent to the model' }
        ]);

        function chunks() {
          const step = Math.max(1, Math.round(st.size * (1 - st.ov)));
          const out2 = [];
          let heading = '';
          for (let i = 0; i < DOC.length; i += step) {
            const words = DOC.slice(i, i + st.size);
            for (let j = Math.max(0, i); j >= 0; j--) if (DOC[j] && DOC[j][0] === '#') { heading = DOC.slice(j, j + 3).join(' ').replace(/#+/g, '').trim(); break; }
            out2.push({ from: i, to: Math.min(DOC.length, i + st.size), words: words, heading: heading });
            if (i + st.size >= DOC.length) break;
          }
          return out2;
        }

        function draw() {
          const cs = chunks();
          let hit = 0, split = 0;
          const marks = ANSWERS.map(a => {
            const inside = cs.some(c => a.need.every(n => c.words.join(' ').indexOf(n) >= 0));
            if (inside) hit++; else split++;
            return inside;
          });
          view.innerHTML = '<div style="font-family:var(--mono);font-size:11.5px;line-height:1.85;max-height:210px;overflow:auto;' +
            'border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:var(--panel)">' +
            cs.map((c, i) => '<span style="background:color-mix(in oklab,var(--c' + ((i % 6) + 1) + ') 16%,transparent);' +
              'border-left:2px solid var(--c' + ((i % 6) + 1) + ');padding:1px 4px;margin-right:2px;border-radius:3px">' +
              (st.head ? '<i style="color:var(--faint)">[' + ML.escapeHtml(c.heading) + '] </i>' : '') +
              ML.escapeHtml(c.words.join(' ')) + '</span> ').join('') + '</div>' +
            '<div style="margin-top:10px">' + ANSWERS.map((a, i) =>
              '<p class="small" style="margin:0 0 4px"><span style="color:var(--' + (marks[i] ? 'green">✓' : 'red">✗') +
              '</span> ' + a.q + (marks[i] ? '' : ' <i>— the answer is split across a chunk boundary</i>') + '</p>').join('') + '</div>';
          const perChunk = st.small2big ? st.size * 3 : st.size;
          out({
            n: cs.length,
            hit: hit + ' of ' + ANSWERS.length,
            split: split,
            tok: Math.round(perChunk * 4 * 1.35) + ' (top-4)'
          });
        }
        draw();
        Viz.note(host, 'Set the chunk size to 12 words: the embeddings become very precise and the retry answer — which needs both "backoff" and "six" — is torn in half. Push it to 70 and everything fits, but each chunk now covers four topics so the embedding is a blur and precision collapses. Turn on <b>small-to-big</b> and you get both: precise small chunks for matching, the full parent section for answering, at 3× the tokens sent to the model. <b>That token cost is the price of the trick, and it is usually worth paying.</b>');
      }
    },
    quiz: [
      {
        q: 'Small-to-big retrieval means…',
        options: ['start with a small model, escalate', 'embed small chunks for precise matching but return their larger parent for context', 'chunk smaller as the corpus grows', 'retrieve fewer documents over time'],
        answer: 1,
        why: 'Small-to-big works by decoupling the unit you match on from the unit you hand the model to answer with: index sentence-level chunks so each embedding is precisely about one thing, but return the surrounding parent section so the model has enough context to actually answer, which is the cleanest resolution of the precision/completeness trade §5.9.1 opens with. "Start with a small model, escalate" is the tempting wrong answer because "small-to-big" pattern-matches to a familiar idiom elsewhere in this course — start cheap and climb only when a cheaper option measurably fails, which is literally the decision ladder of §5.13 — but that idiom concerns model or method selection, and nothing about chunk retrieval swaps models at all. The general principle, from §5.9.2\'s table and the chunking lab, is that precision and completeness do not have to be traded against each other once you stop assuming the retrieval unit and the generation unit must be the same chunk.'
      },
      {
        q: 'The single cheapest improvement to chunk retrievability is usually…',
        options: ['a bigger embedding model', 'prepending the document title and section path to each chunk before embedding', 'more overlap', 'a larger k'],
        answer: 1,
        why: 'A chunk\'s embedding is computed from its own text alone, so a chunk that says "the limit is 4,096" carries no distinguishing signal to match a query against, while "API Reference → Rate Limits → the limit is 4,096" does — for the cost of one line of code prepending the heading before embedding (§5.9.3). "A bigger embedding model" is the tempting answer because a stronger embedding model is the reflexive fix for anything described as a retrieval problem, and it genuinely helps in general — but it solves the wrong layer here: a better model still faithfully embeds a contextless chunk as contextless, since it cannot invent context that chunking already discarded. "More overlap" only rescues an answer split across a chunk boundary; it does nothing for a chunk that is intact but ambiguous on its own, which is this failure mode. The general principle, from §5.9.1\'s pipeline diagram, is that quality lost upstream in parsing and chunking bounds everything downstream, and no amount of spend on the embedding model recovers it.'
      },
      {
        q: 'Re-embedding a 120M-token corpus costs roughly…',
        options: ['a few dollars', 'a few hundred dollars', 'a few thousand dollars', 'it is not feasible'],
        answer: 0,
        why: 'The worked box\'s own arithmetic gets you here: at 512-token chunks the 40,000-document corpus produces roughly 270,000 chunks, which at 512 tokens each is about 138M tokens actually sent to the embedding API — and at roughly $0.02 per million tokens that comes to about $2.80, comfortably "a few dollars." "A few hundred" or "a few thousand" dollars are the tempting, conservative-sounding guesses, since reprocessing an entire large corpus sounds like it should be expensive by default — but embedding is priced per token at a rate several orders of magnitude below most people\'s intuition for "processing 120 million tokens," and this specific number is worth memorising precisely so it never gets treated as a blocker to re-indexing. The general principle, stated directly at the end of the worked box, is that embeddings are nearly free and easy to redo while parsing is the expensive, hard-to-redo stage (§5.9.2) — which is exactly where re-engineering effort should go instead.'
      }
    ],
    cards: [
      { q: 'Default chunking recipe', a: 'Recursive splitting, 400–600 tokens, 10–15% overlap, heading prepended — then measure recall@k and tune from there.' },
      { q: 'Small-to-big', a: 'Index small chunks, return the parent section. Precision on retrieval, completeness on generation.' },
      { q: 'Why tables break RAG', a: 'Flattening destroys the row/column relationship. Extract separately as markdown, summarise for embedding, or route to SQL.' },
      { q: 'Where to spend effort', a: 'Parsing is expensive and hard to redo; embeddings are cheap and easy to redo. Optimise the parser.' }
    ]
  });

  /* ------------------------------------------------------------------ 5.10 */
  ML.section({
    id: 'vector-search', track: 'applied', num: '5.10', level: 3,
    title: 'Vector databases and approximate nearest neighbours',
    lede: 'Exact nearest-neighbour search over ten million vectors is a linear scan and takes seconds. Every vector database is one answer to the same question: how much recall will you give up for a thousandfold speedup, and how do you keep the filters honest while doing it?',
    prereq: ['rag', 'embeddings'],
    related: ['rag', 'chunking', 'pca'],
    html: `
${H.tldr([
      'ANN indexes trade <b>recall</b> for <b>latency</b>. The right question is never "which index is best" but "what recall do I need at what p99, and how much memory can I spend".',
      '<b>HNSW</b> — a navigable small-world graph with a hierarchy — is the default: excellent recall/latency, high memory, slow to build. <b>IVF-PQ</b> compresses vectors to a few bytes and is what you use when the index will not fit in RAM.',
      'Filtered search is the operationally hard part. Pre-filter and the graph loses connectivity; post-filter and you may return nothing. Every serious system implements <i>filtered traversal</i> instead.'
    ])}

<h2><span class="sn">5.10.1</span> The index families</h2>
${H.table(['Index', 'Idea', 'Recall@10', 'Build', 'Memory', 'Use when'], [
      ['Flat (exact)', 'scan everything', '100%', 'none', '1×', '<100k vectors — genuinely, just do this'],
      ['<b>HNSW</b>', 'layered proximity graph, greedy descent', '95–99%', 'slow', '<b>1.5–2×</b>', 'the default under ~10M vectors'],
      ['IVF-Flat', 'k-means partitions, scan the nearest few', '90–97%', 'fast', '1×', 'large, memory available'],
      ['<b>IVF-PQ</b>', 'partitions + product-quantised codes', '80–95%', 'fast', '<b>0.03–0.1×</b>', 'hundreds of millions of vectors'],
      ['ScaNN', 'anisotropic quantisation, score-aware', '95–98%', 'medium', 'low', 'Google-stack, very strong benchmarks'],
      ['DiskANN / Vamana', 'graph on SSD', '95%+', 'slow', 'tiny RAM', 'billions of vectors, cost-constrained'],
      ['Binary + rescore', '1-bit codes, then exact rerank of the top few hundred', '95%+', 'fast', '<b>0.03×</b>', 'a startlingly strong and underused baseline']
    ])}

<p><b>What you are looking at.</b> The left panel is a real navigable small-world graph, built live from the points plotted, with every edge the graph actually has drawn in faint grey. The large ringed dot is your query. The true 10 nearest neighbours — found here by brute-force comparison, purely to score the search against — are marked as large dots; the smaller highlighted dots are what the graph search actually returned. The bright path traces the real greedy walk, hop by hop, from the ringed entry point. The right panel plots the same idea numerically: recall@10 against distance computations performed, with a dashed vertical line marking what an exhaustive scan of every point would cost.</p>
<p><b>What to do with it.</b> Leave <code>M</code> and the corpus size at their defaults and sweep <code>efSearch</code> from 1 up to 30. Watch the highlighted dot on the right-hand curve slide along it, and watch the left-hand path grow longer and touch more of the graph as it goes. Then set <code>efSearch</code> back down and instead drag <code>M</code> toward 2: the graph on the left visibly thins into loosely connected pockets, and no amount of raising <code>efSearch</code> afterwards buys back the recall that fragmentation just cost you.</p>
<p><b>The thing genuinely worth noticing.</b> The right-hand curve is steep, then flat: most of the achievable recall arrives in the first few distance computations, and the last few percentage points cost disproportionately more to reach. That shape is the entire empirical case for tuning <code>efSearch</code> in production rather than <code>M</code> or <code>efConstruction</code> — it is the one parameter sitting on a curve you can move up and down per query, while the low-<code>M</code> experiment above shows the other two are decided once, at build time, and bound everything the search can ever achieve afterwards.</p>

${H.lab('ann', 'Build a proximity graph and watch greedy search walk it', 'A real navigable small-world graph, built here from the points shown, searched greedily from an entry point. The path is the actual traversal, and the distance-computation count is the actual cost.')}

${H.intuition(`<p>Why does a greedy walk on a graph find near neighbours? Because the graph has two kinds of edge: short ones to genuine neighbours, and a few long ones that act as motorways. The long edges get you into the right region in a handful of hops (the "small world" property — hop count grows like $\\log n$); the short ones do the local refinement. HNSW's hierarchy makes this explicit: upper layers are sparse and long-range, the bottom layer is dense and local, and you descend.</p>`)}
${H.analogy(`<p>If you already know a skip list, you already know the shape of this answer. A skip list speeds up search through a sorted, linked sequence by adding extra lanes above it: most nodes sit only in the bottom lane, a sparser sample also sits one lane up, an even sparser sample sits two lanes up, and so on. Search starts in the topmost, sparsest lane, walks along it until the next node would overshoot the target, drops down one lane, and repeats. Because each lane holds roughly half as many nodes as the one below it, the number of lanes — and so the number of hops — grows only as $\\log n$.</p>
<p>HNSW runs the same trick with one thing changed: there is no sorted order to walk, because vectors in a high-dimensional space do not line up along a single axis the way numbers on a list do. So instead of skipping several positions along a line, each layer skips across a proximity graph instead, where an edge connects two points that are near each other in the embedding space rather than adjacent in an artificial ordering. The upper layers are sparse graphs with long edges that cross large regions in one hop; the bottom layer is dense and does the fine work. Descend layer by layer, always moving to whichever neighbour is closest to the query, and a skip list's $\\log n$ search cost falls out of a space that never had an order to begin with.</p>`)}
${H.table(['HNSW parameter', 'Meaning', 'Effect'], [
      ['$M$', 'edges per node', 'higher = better recall, more memory, slower build. 16–48 typical'],
      ['<code>efConstruction</code>', 'candidate list size while building', 'higher = better graph, much slower build. 100–500'],
      ['<b><code>efSearch</code></b>', 'candidate list size at query time', '<b>the runtime recall/latency dial</b> — tune this, not the others'],
      ['layers', 'chosen randomly per node', 'geometric distribution; gives the $\\log n$ hop count']
    ])}
${H.key('<code>efSearch</code> is the one knob you tune in production, because it can be changed per query without rebuilding anything. Recall rises and latency rises with it, and the curve is steep at the start and flat at the end — which means there is almost always a setting that gets you 97% recall for a fraction of the cost of 99.5%.')}

<h2><span class="sn">5.10.2</span> Filtered search: the operationally hard part</h2>
${H.vs('Pre-filter', [
      'Restrict to matching IDs, then search',
      'Correct by construction',
      '<b>Destroys graph connectivity</b> when the filter is selective — the walk cannot reach the survivors',
      'Degenerates to a linear scan of the filtered set'
    ], 'Post-filter', [
      'Search, then drop non-matching results',
      'Fast, uses the index as built',
      '<b>May return far fewer than $k$</b>, or nothing at all',
      'Needs over-fetching by an unknown factor'
    ])}
${H.flag('The real answer is <b>filtered traversal</b>: walk the graph normally but only admit matching nodes into the result set, while still traversing through non-matching ones. Qdrant, Weaviate and pgvector-with-iterative-scan all implement a version of this, and each switches to a brute-force scan below some selectivity threshold. When you evaluate a vector database, <i>measure recall with your actual filters applied</i> — unfiltered benchmark numbers are close to meaningless for a real workload.')}
<p>Both failure modes are easiest to see as the filter gets stricter, so put a number on the one that looks like ordinary engineering — over-fetch-then-drop — rather than taking "may return far fewer than $k$" on faith.</p>
${H.deriv('how far you have to over-fetch to post-filter safely', [
      ['want $k$ results that also pass a filter matching a fraction $q$ of the corpus.', 'the setup: $k$ is what the caller asked for, $q$ is the filter\'s selectivity.'],
      ['assume, as a first approximation, that passing the filter is independent of how close an item is to the query.', 'true for a filter unrelated to relevance, such as a date range — flagged below for when it is not.'],
      ['the expected number of filter-matching items among the top $N$ retrieved is $\\approx qN$.', 'linearity of expectation: each of the $N$ items independently matches with probability $q$.'],
      ['set $qN=k$ and solve: $N=k/q$.', 'the smallest $N$ for which you expect to have collected $k$ matches.']
    ], 'At $q=1\\%$ and $k=10$, fetch about 1,000 candidates before filtering. At $q=0.01\\%$ — one match in ten thousand — the same $k$ needs about 100,000 candidates, more than most indexes are ever asked to return for a single query. The over-fetch factor is $1/q$, and it blows up exactly where the filter is doing the most work — which is the concrete, computable reason every real system abandons post-filtering below some selectivity and falls back to a brute-force scan of the filtered ID list instead. The independence assumption in step two is also the first thing to check when this estimate is wrong in practice: a filter correlated with relevance, such as "only the newest documents", breaks it in a direction that makes over-fetching even less reliable.')}
<p>Pre-filtering fails in the same low-selectivity region for an entirely different reason: it is not a cost problem but a structural one. HNSW's edges were built to route between whatever points happened to be near each other across the <i>whole</i> dataset; restrict the candidate set to a 0.01% slice before searching and most of the paths the greedy walk relies on lead to nodes that no longer count, so the walk can stall far from the true matching neighbours long before it would have found them. Filtered traversal survives both regimes because it never throws away the graph's connectivity — it walks through the 99.99% of nodes that do not match in order to keep reaching the ones that do, and only gives up on the graph altogether, in favour of a plain scan, once the maths above says a scan is cheaper anyway.</p>

<h2><span class="sn">5.10.3</span> Product quantisation, in one page</h2>
<p>Compression only ever does one thing: throw away precision you were not using. Product quantisation applies that idea to a vector's coordinates directly, and the byte count it lands on is not a claimed number — it falls straight out of counting what gets kept.</p>
${H.deriv('the product-quantisation byte budget', [
      ['$D=1{,}024$ dimensions stored as float32: $D\\times 4 = 4{,}096$ bytes per vector.', 'the uncompressed baseline — every one of the 1,024 numbers needs its own 4-byte float.'],
      ['split into $m=64$ sub-vectors of length $D/m = 1{,}024/64 = 16$.', 'group adjacent dimensions into sub-vectors so a codebook can be trained per group rather than per single dimension.'],
      ['run k-means with $k=256$ centroids on each sub-space; replace every sub-vector by the index of its nearest centroid.', '256 possible values is exactly what one byte addresses ($2^8=256$) — the codebook size is chosen to fit a byte, not the other way round.'],
      ['compressed size $= m \\times 1$ byte $=64$ bytes.', 'one byte per sub-vector, $m$ sub-vectors, nothing else is stored.'],
      ['compression ratio $=4{,}096/64=64\\times$.', 'divide the uncompressed size by the compressed size.']
    ], 'At query time this compression pays for itself twice: precompute the distance from the query\'s own sub-vectors to all 256 centroids per sub-space — an $m\\times256$ lookup table — and every candidate\'s distance becomes $m$ table lookups and an add, with no decompression step and a table small enough to live in cache. <b>Rescore</b> the top few hundred candidates against their full-precision vectors afterwards: that step is what turns an aggressive, lossy compression into an acceptable one, and skipping it is the single most common reason a team\'s measured recall lands far below the index\'s advertised numbers.')}
${H.pitfall('The recall lost to quantisation is not spread evenly across the dataset. A vector sitting near the middle of its centroid\'s cell survives compression almost losslessly; one sitting near the boundary between two centroids can be pulled into the wrong bucket entirely — and boundary vectors are disproportionately likely to be exactly the close, competitive neighbours a query actually struggles to distinguish. The easy, unambiguous matches were never at risk. This is why rescoring matters more than the headline recall figure suggests: it is cheap insurance concentrated precisely on the candidates most likely to have been quantised into the wrong place.')}
${H.worked('the memory arithmetic that picks your index', `
<p>10 million chunks, 1,024-dimensional embeddings:</p>
<ul>
<li><b>Flat float32</b>: 10M × 4 KB = <b>41 GB</b>. Exact, and needs a large machine.</li>
<li><b>HNSW float32, M=32</b>: 41 GB + graph ≈ <b>60 GB</b>. Fast and accurate, expensive.</li>
<li><b>int8 quantised + HNSW</b>: ~10 GB + graph ≈ <b>15 GB</b>, with ~1% recall loss. Usually the right answer.</li>
<li><b>IVF-PQ, 64 bytes/vector</b>: <b>0.64 GB</b>. Fits anywhere; 85–93% recall, restored to ~97% with rescoring.</li>
<li><b>Binary (1 bit/dim) + rescore</b>: 128 bytes/vector = <b>1.3 GB</b>, with a full-precision rescore of the top 500.</li>
</ul>
<p>The jump from 60 GB to 1.3 GB changes what machine you rent, which is usually a larger cost difference than the recall difference is a quality difference.</p>`)}

${H.probe([
      ['HNSW or IVF-PQ?', 'HNSW under about 10M vectors when memory is available — better recall/latency and no training step. IVF-PQ when the index must be small or the corpus is hundreds of millions.'],
      ['Which parameter do you tune in production?', '<code>efSearch</code>. It is per-query, needs no rebuild, and trades recall against latency along a curve that is steep at the start and flat at the end.'],
      ['Why is filtered vector search hard?', 'Pre-filtering breaks graph connectivity; post-filtering can return fewer than $k$ results. Filtered traversal — walk through non-matching nodes but only collect matching ones — is the standard fix, with a brute-force fallback below some selectivity.'],
      ['How do you measure whether your index is good enough?', 'Compute exact nearest neighbours on a sample of real queries, then measure recall@k of the index against them <i>with your filters applied</i>, alongside p50 and p99 latency.'],
      ['When do you not need a vector database at all?', 'Under ~100k vectors: a numpy matrix multiply is faster than a network round trip. Or when BM25 alone already answers the queries — measure before adding infrastructure.']
    ])}`,
    labs: {
      ann: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'vectors in the index', min: 40, max: 400, step: 20, value: 160, fmt: v => v },
          { k: 'M', label: 'edges per node M', min: 2, max: 12, step: 1, value: 4, fmt: v => v },
          { k: 'ef', label: 'efSearch', min: 1, max: 30, step: 1, value: 4, fmt: v => v },
          { k: 'qx', label: 'query x', min: -2.4, max: 2.4, step: .05, value: 1.1, fmt: v => v.toFixed(2) },
          { k: 'qy', label: 'query y', min: -2.4, max: 2.4, step: .05, value: -0.8, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rec', label: 'recall@10', cls: 'good' },
          { k: 'comps', label: 'distance computations', cls: 'key' },
          { k: 'exact', label: 'exact scan would need', cls: 'bad' },
          { k: 'speed', label: 'speedup' },
          { k: 'hops', label: 'greedy hops' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(12);
            const pts = [];
            for (let i = 0; i < st.n; i++) {
              const c = i % 4;
              const cx = [(-1.3), 1.4, -1.1, 1.2][c], cy = [1.2, 1.1, -1.2, -1.3][c];
              pts.push([cx + R.normal(0, .65), cy + R.normal(0, .65)]);
            }
            const idx = Num.nsw(pts, st.M, 3);
            const q = [st.qx, st.qy];
            const res = idx.search(q, st.ef, 0);
            const truth = idx.exact(q, 10);
            const found = res.nearest.slice(0, 10);
            const rec = truth.filter(t => found.indexOf(t) >= 0).length / Math.min(10, truth.length);

            const w1 = w * .58;
            const P = Viz.plot(ctx, w1, h, { xd: [-3.2, 3.2], yd: [-3, 3], pad: { l: 8, r: 8, t: 14, b: 26 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'the graph, and the walk' });
            P.clip(() => {
              ctx.strokeStyle = T.line; ctx.lineWidth = .6; ctx.globalAlpha = .55;
              idx.adj.forEach((nbrs, i) => nbrs.forEach(j => {
                if (j < i) return;
                ctx.beginPath(); ctx.moveTo(P.x(pts[i][0]), P.y(pts[i][1])); ctx.lineTo(P.x(pts[j][0]), P.y(pts[j][1])); ctx.stroke();
              }));
              ctx.globalAlpha = 1;
              pts.forEach(p => P.dots([[p[0], p[1]]], { r: 2, color: T.faint, alpha: .7 }));
              truth.forEach(i => P.dots([[pts[i][0], pts[i][1]]], { r: 4.2, color: T.c3, alpha: .9 }));
              found.forEach(i => P.dots([[pts[i][0], pts[i][1]]], { r: 2.6, color: T.c1 }));
              P.line(res.path.map(i => pts[i]), { color: T.c4, width: 2 });
              P.dots([pts[res.path[0]]], { r: 5, color: T.c4, stroke: true, strokeWidth: 1.6 });
              P.dots([q], { r: 6.5, color: T.c2, stroke: true, strokeWidth: 2 });
            });

            ctx.save(); ctx.translate(w1, 0);
            const curve = [];
            for (let e = 1; e <= 30; e += 1) {
              const r2 = idx.search(q, e, 0);
              const f2 = r2.nearest.slice(0, 10);
              curve.push([r2.comps, truth.filter(t => f2.indexOf(t) >= 0).length / Math.min(10, truth.length)]);
            }
            const P2 = Viz.plot(ctx, w - w1, h, {
              xd: [0, Math.max(20, st.n)], yd: [0, 1.05], pad: { l: 44, r: 12, t: 14, b: 40 }
            }).frame({ xlabel: 'distance computations', ylabel: 'recall@10', yfmt: v => (v * 100).toFixed(0) + '%' });
            P2.clip(() => {
              P2.line(curve.sort((a, b) => a[0] - b[0]), { color: T.c1, width: 2.4 });
              P2.dots(curve, { r: 2.4, color: T.c1 });
              P2.vline(st.n, { color: T.c2, dash: [4, 3], label: 'exact scan' });
              P2.dots([[res.comps, rec]], { r: 5, color: T.c2, stroke: true, strokeWidth: 2 });
            });
            ctx.restore();
            out({
              rec: (rec * 100).toFixed(0) + '%',
              comps: res.comps,
              exact: st.n,
              speed: (st.n / Math.max(1, res.comps)).toFixed(1) + '×',
              hops: res.path.length - 1
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c2)', t: 'query' }, { c: 'var(--c3)', t: 'true 10 nearest' },
          { c: 'var(--c1)', t: 'what the index returned' }, { c: 'var(--c4)', t: 'the greedy walk' }
        ]);
        Viz.note(host, 'The right-hand curve is the whole story of ANN search: <b>recall against work done</b>, with the vertical line marking what an exact scan would cost. Notice its shape — steep, then flat. Going from efSearch 1 to 6 buys most of the recall for a fraction of the scan cost; going from 20 to 30 buys almost nothing. Now drop <b>M</b> to 2: the graph fragments, the walk gets trapped in a local cluster, and recall collapses no matter how large you make efSearch. <b>Build quality bounds search quality</b>, which is why efConstruction is worth its build time.');
      }
    },
    quiz: [
      {
        q: 'The parameter you tune at query time to trade recall against latency in HNSW is…',
        options: ['M', 'efConstruction', 'efSearch', 'the number of layers'],
        answer: 2,
        why: 'efSearch controls the candidate-list size during the query itself, so it is the one HNSW parameter you can raise or lower per request with no rebuild, which is exactly what makes it the runtime recall/latency dial (§5.10.1). "efConstruction" is the most tempting wrong answer, since it shares nearly the same name and does genuinely determine how good the graph is — but it acts only at build time: changing it means re-running construction over the whole index, which is precisely what a per-query knob cannot require. "M" is baked in even more permanently, since it fixes how many edges each node gets when the graph is built, and changing it means building a structurally different graph. The general principle is to separate what is architecture, fixed once at build time, from what is a genuine runtime dial — the lab\'s recall-against-work curve is steep then flat specifically along the efSearch axis, which is what makes tuning it worthwhile.'
      },
      {
        q: 'Product quantisation with $m=64$ sub-spaces compresses a 1024-d float32 vector to…',
        options: ['1024 bytes', '256 bytes', '64 bytes', '16 bytes'],
        answer: 2,
        why: 'Each of the 64 sub-vectors is replaced by a single byte, the index into its own 256-centroid codebook (256 values fit exactly in one byte), so the uncompressed 1,024-dimensional float32 vector at 4,096 bytes becomes 64 bytes total — one byte per sub-space, a 64× reduction (§5.10.3\'s steps). "256 bytes" is the tempting wrong answer because 256 is the number sitting right there in the description, the centroid count per sub-space, and it is easy to conflate "256 possible codes per sub-space" with "256 bytes needed overall" if you do not track that the codebook size only determines what one byte can index, not how many bytes the compressed vector needs. "1,024 bytes" makes the related mistake of assigning one byte per dimension directly, skipping the step where quantisation first groups dimensions into sub-vectors. The general principle is the same one-byte-per-codebook-index arithmetic the section\'s memory table applies at scale, taking 10M vectors at 1,024-d from 41 GB flat down to 0.64 GB at 64 bytes per vector.'
      },
      {
        q: 'Pre-filtering a graph index by a highly selective predicate…',
        options: ['is the correct approach', 'breaks graph connectivity so the walk cannot reach the surviving nodes', 'is always faster', 'improves recall'],
        answer: 1,
        why: 'An HNSW-style graph\'s edges connect nodes based on proximity across the whole dataset, so restricting the candidate set to a highly selective filter before searching removes most of the paths the walk relies on, and the traversal can get stranded far from the true nearest matching neighbours — in the worst case degenerating to a linear scan of just the filtered survivors (§5.10.2). "Is always faster" is the tempting answer because narrowing the search space before doing the work sounds like it should obviously cost less, mirroring how filtering data down before processing usually saves effort elsewhere in engineering — but here the filter is applied to a graph\'s traversal structure, not to a flat scan, and a fragmented graph can force the walk to do more work while still returning a worse result. The general principle is that filtered vector search needs filtered traversal — walk through non-matching nodes but only collect matching ones, with a brute-force fallback below some selectivity threshold — precisely because neither pre-filtering nor post-filtering respects how the graph is actually connected.'
      },
      {
        q: 'With 60,000 vectors, the right index is usually…',
        options: ['HNSW', 'IVF-PQ', 'flat exact search', 'DiskANN'],
        answer: 2,
        why: 'At 60,000 vectors, a brute-force matrix multiply against the query completes in single-digit milliseconds on ordinary hardware — faster than the network round trip to a vector database would even take — so building an approximate index is paying its build time, memory overhead and recall loss to solve a latency problem that does not yet exist at this scale. "HNSW" is the tempting answer precisely because the section introduces it as the default, and defaults are exactly what people reach for without first checking whether the underlying problem — searching millions to hundreds of millions of vectors quickly — actually applies; at 60k it does not. "IVF-PQ" and "DiskANN" make the same mistake at an even larger scale mismatch, since both exist specifically for corpora orders of magnitude bigger than this one. The general principle, from §5.10\'s own probe, is that the right question is never "which index is best" in the abstract but what recall and latency you actually need, and under roughly 100k vectors the honest answer is frequently no index at all.'
      }
    ],
    cards: [
      { q: 'Index selection, in one line', a: '<100k: flat. <10M with RAM: HNSW (int8). Hundreds of millions: IVF-PQ or DiskANN. Always rescore.' },
      { q: 'HNSW parameters', a: 'M (edges), efConstruction (build quality), efSearch (the runtime recall/latency dial).' },
      { q: 'Product quantisation', a: 'Split into $m$ sub-vectors, 256-centroid codebook each → 1 byte per sub-vector; distances by table lookup; rescore the top few hundred.' },
      { q: 'Filtered ANN', a: 'Pre-filter breaks connectivity, post-filter under-returns. Filtered traversal with a brute-force fallback is the real answer — benchmark with your filters on.' }
    ]
  });

  /* ------------------------------------------------------------------ 5.11 */
  ML.section({
    id: 'evals', track: 'applied', num: '5.11', level: 2,
    title: 'Building evals that do not lie',
    lede: 'The eval is the only thing standing between you and shipping on vibes. It is also, in most teams, the least rigorous artefact in the codebase — a hundred hand-picked examples, graded by a model with known biases, reported without an interval.',
    prereq: ['llm-eval', 'intervals'],
    related: ['llm-eval', 'experimentation', 'mlops'],
    html: `
${H.tldr([
      'An eval set of 100 examples has a 95% interval of roughly ±10 points on a proportion near 50%. <b>Most reported "improvements" are inside the noise of the eval that measured them.</b>',
      'LLM judges have measurable, reproducible biases: position, verbosity, self-preference, and sensitivity to formatting. Every one of them has a cheap mitigation, and none of them goes away by choosing a better model.',
      'Build the eval from <b>production failures</b>, not from imagination. The examples you can think of are the ones the system already handles.'
    ])}

<h2><span class="sn">5.11.1</span> How big does the eval set need to be?</h2>
<p>Two prompts, a hundred examples each, and system B scores four points higher. Ship it? The honest answer turns on a number almost nobody computes before deciding: how much would that score have moved on this exact set, with neither system changed at all, purely from which hundred examples happened to get sampled? An eval score is not a fact about a system. It is an estimate, made from a finite sample, and every estimate carries an error bar whether or not anyone draws it.</p>
<p>Treat one graded item as a coin flip: it passes with the system's true, unknown pass rate $p$, or it does not. The measured pass rate $\\hat p$ over $n$ such items is the number you actually report, and this section answers exactly how far $\\hat p$ can be expected to wander from $p$ purely from which $n$ items you happened to draw.</p>
${H.deriv('the 95% interval on a measured pass rate', [
      ['each graded item is a Bernoulli trial: it passes with probability $p$, fails with probability $1-p$.', 'the noise model — one 0/1 outcome, the same coin-flip assumption behind logistic loss in §1.5.'],
      ['a single trial has variance $\\mathrm{Var}[X]=p(1-p)$.', 'for a 0/1 variable $X^2=X$, so $\\mathrm{Var}[X]=E[X^2]-E[X]^2=p-p^2=p(1-p)$.'],
      ['the measured rate $\\hat p=\\tfrac1n\\sum_i X_i$ is a mean of $n$ i.i.d. trials, so $\\mathrm{Var}[\\hat p]=p(1-p)/n$.', 'the variance of an average of independent variables is the variance of one, divided by $n$ — the same shrinkage the CLT relies on throughout Part 1.'],
      ['standard error: $\\mathrm{SE}(\\hat p)=\\sqrt{p(1-p)/n}$.', 'take the square root of the variance to get a quantity on the same scale as $\\hat p$ itself.'],
      ['for large enough $n$, $\\hat p$ is approximately Gaussian around $p$ (CLT), so a 95% interval is $\\hat p \\pm 1.96\\,\\mathrm{SE}(\\hat p)$.', '1.96 is the $z$-value enclosing the middle 95% of a standard normal — the same constant every other confidence interval in this course uses.']
    ], 'Plug in $p=0.7$, $n=100$: $\\mathrm{SE}=\\sqrt{0.7\\times0.3/100}\\approx0.0458$, so the 95% interval is $\\pm1.96\\times0.0458\\approx\\pm9$ points. A "3-point improvement" measured on 100 examples is not evidence of anything — it sits comfortably inside noise this size.')}
${H.worked('how many examples it actually takes to tell 70% from 75% apart', `<p>It is tempting to eyeball this from the interval above: roughly ±9 points at $n=100$, so surely $n=200$ or so comfortably resolves a 5-point gap. That reasoning is wrong, and worth seeing exactly why it is wrong. A ±9-point interval describes how far <i>one</i> measurement can drift from the truth; reliably telling two systems apart needs both measurements to be unlikely to have drifted past each other, which is a stricter, differently-shaped requirement — a power calculation, not a doubled interval.</p>
<p>Solving for the $n$ that reliably separates a 70% baseline from a 75% alternative, at the conventional 5% significance and 80% power, gives $n\\approx1{,}319$ per system when the two are scored on independent samples — over thirteen times the 100-example eval this section opened with, not two or three times it.</p>
<p>Score both systems on the <i>same</i> 100 items instead, and only the items where they disagree carry any information about which is better — an item both pass, or both fail, cancels out of the comparison entirely regardless of sample size. If the two systems disagree on 25% of items, the paired version of the same calculation needs only about 783 items: a real saving, but still nowhere near the 100 a team is routinely tempted to ship a decision on. There is no shortcut around needing several hundred to a couple of thousand examples to responsibly call a few-point difference — only a cheaper way to get there and a more expensive one.</p>`)}

<p><b>What you are looking at.</b> The curve plots the 95% confidence half-width against eval-set size $n$ on a log scale, for whatever baseline pass rate the first control sets. Three reference points mark $n=100$, $500$ and $2{,}000$ directly on the curve. The dashed horizontal line sits at half of the improvement you are trying to detect — a common rule of thumb is that the curve needs to dip below that line before a gap of that size is even resolvable in principle.</p>
<p><b>What to do with it.</b> Leave the baseline near 70% and raise "improvement to detect" from 1 point to 10, watching where the curve crosses the dashed line: a 10-point gap is resolved by a modest $n$, while a 1-point gap needs an eval set most teams will never build. Then look at the two readouts below the plot, "n needed, unpaired" against "n needed, paired", and drag "items where the two systems differ" down toward 10%: the paired requirement keeps shrinking as disagreement falls, because agreement between the two systems on an item is exactly the case that carries no comparative information.</p>
<p><b>The thing genuinely worth noticing.</b> At $n=100$ the interval half-width sits right around $\\pm9$ points for a baseline near 70%, matching the derivation above exactly — drag the baseline toward 50% and watch it grow further still, since $p(1-p)$ is largest at $p=0.5$. The paired-versus-unpaired ratio is the more actionable number on the panel: push disagreement down toward 10% and it climbs to roughly 4×, matching the several-fold reduction the key box below promises; push disagreement up toward "the two systems disagree on almost everything" and the advantage of pairing disappears, because at that point there are barely any agreeing items left for pairing to remove from the comparison.</p>
${H.lab('evalsize', 'How many examples do you actually need?', 'The interval on a single score, and the number of examples needed to call a difference between two systems. Paired evaluation on the same items is the free win here.')}
${H.key('Use the <b>same examples</b> for both systems and compare per-item. Paired comparison removes the item-difficulty variance, and typically needs 3–5× fewer examples than scoring the two systems on independent samples. It is the cheapest statistical improvement available to you.')}

<h2><span class="sn">5.11.2</span> LLM-as-judge, and its documented biases</h2>
<p>These biases are not bugs sitting in a careless prompt; they are inherited. A judge model is still an autoregressive language model doing next-token prediction, and the preference-tuning stage that shaped it into something willing to hand out verdicts at all — RLHF or DPO — was itself trained on human preference judgements, which carry exactly the same distortions before a judge ever sees a prompt. Position bias has a mechanical origin: the pairwise comparisons behind preference tuning were made by human raters under time pressure, who reliably anchor on whichever option they read first and look for a reason to reject the second rather than weighing both symmetrically; a judge distilled from that signal inherits the anchor along with everything else it learned. Verbosity bias is similarly traceable: human raters, and the reward models trained to imitate them, consistently rate longer, more structured-looking answers as more thorough even when the extra length adds nothing, so length became a learned proxy for effort during the very training process that produced the judge. Self-preference is the least mechanistically settled of the three — the leading account is that a model's own phrasing and rubric-following style is, in some sense, more "fluent" to itself as a next-token predictor, so text resembling its own family scores marginally higher independent of content.</p>
${H.flag('Self-preference bias is the most contested entry in this table. Some published evaluations find a strong, consistent same-family boost; others, controlling more carefully for confounds like formatting and length, find the effect shrinks close to zero or is not reliably reproducible across model pairs. Treat any single paper\'s self-preference number as provisional. Measure it directly on your own judge and your own outputs — a panel of judges from different families is the cheapest way — rather than importing someone else\'s estimate of how large the effect is.')}
${H.table(['Bias', 'What happens', 'Mitigation', 'Residual'], [
      ['<b>Position</b>', 'the first (or last) option is preferred regardless of content', '<b>run both orders, average</b> — or discard disagreements', 'largely solved'],
      ['<b>Verbosity</b>', 'longer answers score higher at equal quality', 'control for length; penalise it explicitly in the rubric', 'partly solved'],
      ['Self-preference', 'a judge prefers text from its own family', 'use a different family as judge; or a panel', 'reduced, not eliminated'],
      ['Formatting', 'markdown, bullet points and headers raise scores', 'normalise formatting before judging', 'easy'],
      ['Sycophancy', 'agrees with an assertion embedded in the prompt', 'never state the expected answer in the judge prompt', 'easy'],
      ['Score compression', 'everything gets a 4 out of 5', 'pairwise preference instead of absolute scoring', '<b>use pairwise by default</b>'],
      ['Rubric drift', 'the meaning of "good" moves as you edit the prompt', 'version the judge prompt; re-run a fixed calibration set', 'process, not modelling']
    ])}

<p><b>What you are looking at.</b> Two horizontal bars compare a system's true win rate — its actual quality, fixed by the "true quality of A" control — against what a biased judge would measure over 400 simulated pairwise comparisons. The vertical dashed line marks 50%, the boundary between "A wins" and "A loses" as a verdict. The readouts below translate the gap between the two bars into a signed error in points and a plain verdict: whether the measured result would lead you to the right system, the wrong one, or the right one by the wrong margin.</p>
<p><b>What to do with it.</b> Start at the defaults and watch the measured bar sit noticeably left of the true one — system A is being under-counted before any mitigation is applied. Turn on <b>evaluate both orders</b> and watch the measured bar jump back toward the true one, then read the "order-flip disagreement" readout that appears alongside it: that percentage is directly measuring how much of the raw judgement was ever presentation rather than content. Now turn on <b>normalise length</b> as well and watch the residual gap close further still.</p>
<p><b>The thing genuinely worth noticing.</b> Push both bias sliders toward their upper range with the mitigations off, and the verdict readout flips to "you would pick the WRONG system" while the true win rate is still comfortably above 50% — the measured number does not just look noisy, it points the wrong direction with complete apparent confidence. No amount of averaging over more comparisons fixes this, because the bias is systematic rather than random: it is the same mistake made consistently on every single comparison, which is exactly why §5.11.1's sample-size arithmetic cannot rescue a judge that is measuring the wrong thing precisely.</p>
${H.lab('judge', 'Simulating judge bias, and what mitigation recovers', 'Two systems of known true quality, evaluated by a biased judge. Turn the biases on and off, and turn the mitigations on and off, and see how far the measured win rate drifts from the truth.')}
${H.pitfall('Report <b>agreement with human labels</b> for your judge, on a held-out set, before you trust it. Cohen’s κ above 0.6 is workable; above 0.8 is good. Human–human agreement on the same task is the real ceiling — if two annotators agree only 70% of the time, a judge at 68% is doing fine and the task definition is the problem, not the judge.')}

<h2><span class="sn">5.11.3</span> What to actually build</h2>
${H.steps([
      '<b>A golden set from production failures.</b> Every incident, complaint and thumbs-down becomes a test case with the expected behaviour written down. This set only grows, and it is the most valuable artefact your team owns.',
      '<b>Deterministic assertions wherever possible.</b> Did it call the right tool? Is the JSON valid? Does the citation exist in the retrieved context? Is the number within tolerance? These need no judge, cost nothing, and never drift.',
      '<b>A held-out slice you never look at</b> until a release decision. The set you iterate against will be overfitted; that is not a failure of discipline, it is arithmetic.',
      '<b>Per-slice reporting.</b> Overall averages hide the group where you are failing. Break out by language, query type, document source, user tier.',
      '<b>Adversarial and safety cases</b>, maintained separately with their own bar. These are pass/fail gates, not averages.',
      '<b>Cost and latency in the same table as quality.</b> A 2-point quality gain for 3× the cost is a decision, and it cannot be made if the numbers live in different dashboards.'
    ])}
${H.vs('Offline eval', [
      'Fast, cheap, repeatable, runs in CI',
      'Catches regressions before users see them',
      'Measures a proxy — correlation with the real outcome is an assumption you should test',
      'Overfits as you iterate against it'
    ], 'Online eval', [
      'Measures what you actually care about',
      'Slow, expensive, and needs traffic',
      'Confounded unless randomised (§2.25)',
      'The final arbiter — but too slow to be the only loop'
    ])}
${H.flag('Public benchmark contamination is now severe enough that a strong score on a well-known benchmark is weak evidence about a model released after that benchmark was published. Prefer a private eval built from your own traffic; treat public leaderboards as a coarse filter for which models to bother testing.')}

${H.probe([
      ['You have 200 eval examples and system B scores 4 points higher. Is it better?', 'Probably not distinguishable. Paired on the same 200 items you might detect it if the per-item disagreement rate is low; unpaired you would need roughly 1,000+. Report the interval and the number of items where they differ.'],
      ['How do you control for position bias in an LLM judge?', 'Evaluate each pair in both orders and average, or count only the cases where the judgement is consistent across orders. Report the flip rate — it is a direct measure of how much of your signal is noise.'],
      ['Your judge agrees with humans 68% of the time. Is that bad?', 'Compare it with human–human agreement on the same task. If two humans agree 70% of the time, the task definition is ambiguous and 68% is close to the ceiling. Fix the rubric, not the judge.'],
      ['Where do good eval cases come from?', 'Production failures. The examples you invent are the ones the system already handles; the ones that broke are the ones that carry information.']
    ])}`,
    labs: {
      evalsize: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'baseline pass rate', min: .1, max: .95, step: .01, value: .72, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'delta', label: 'improvement to detect', min: .01, max: .2, step: .005, value: .04, fmt: v => '+' + (v * 100).toFixed(1) + ' pts' },
          { k: 'disagree', label: 'items where the two systems differ (paired)', min: .05, max: 1, step: .05, value: .25, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ci', label: '95% CI at n = 100', cls: 'warn' },
          { k: 'ci500', label: '… at n = 500' },
          { k: 'unpaired', label: 'n needed, unpaired', cls: 'bad' },
          { k: 'paired', label: 'n needed, paired', cls: 'good' },
          { k: 'ratio', label: 'paired is cheaper by' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const p = st.p;
            const P = Viz.plot(ctx, w, h, { xd: [1.3, 4], yd: [0, .2], pad: { l: 56, r: 14, t: 16, b: 42 } })
              .frame({
                xticks: [1.3, 2, 2.5, 3, 3.5, 4], xfmt: v => Math.round(Math.pow(10, v)).toLocaleString(),
                xlabel: 'eval set size n', ylabel: '95% CI half-width', yfmt: v => '±' + (v * 100).toFixed(0) + ' pts'
              });
            P.clip(() => {
              P.area(Num.linspace(1.3, 4, 80).map(lx => [lx, 1.96 * Math.sqrt(p * (1 - p) / Math.pow(10, lx))]), { color: T.c1, alpha: .12 });
              P.fn(lx => 1.96 * Math.sqrt(p * (1 - p) / Math.pow(10, lx)), { color: T.c1, width: 2.8, n: 120 });
              P.hline(st.delta / 2, { color: T.c2, dash: [5, 4], label: 'half your effect' });
              [100, 500, 2000].forEach(n => {
                P.dots([[Math.log10(n), 1.96 * Math.sqrt(p * (1 - p) / n)]], { r: 4.2, color: T.c4, stroke: true });
                P.text(Math.log10(n), 1.96 * Math.sqrt(p * (1 - p) / n), ' n=' + n, { dx: 7, color: T.c4, font: '10.5px ui-monospace' });
              });
            });
            const unp = Num.sampleSize(p, st.delta, .05, .8);
            // McNemar-style: only discordant pairs carry information
            const pd = st.disagree;
            const paired = Math.ceil(Math.pow(1.96 * Math.sqrt(pd) + 0.84 * Math.sqrt(pd - st.delta * st.delta), 2) / (st.delta * st.delta));
            out({
              ci: '±' + (1.96 * Math.sqrt(p * (1 - p) / 100) * 100).toFixed(1) + ' pts',
              ci500: '±' + (1.96 * Math.sqrt(p * (1 - p) / 500) * 100).toFixed(1) + ' pts',
              unpaired: unp.toLocaleString(),
              paired: paired.toLocaleString(),
              ratio: (unp / Math.max(1, paired)).toFixed(1) + '×'
            });
          }
        });
        Viz.note(host, 'At n = 100 and a pass rate near 70%, the 95% interval is about <b>±9 points</b>. A "3-point improvement" measured on 100 examples is indistinguishable from nothing. Now look at the paired row: when the two systems differ on only 25% of items, a paired test needs a small fraction of the sample — because the items they both get right or both get wrong carry no information about which is better, and pairing removes them from the variance.');
      },

      judge: function (host) {
        const st = Viz.controls(host, [
          { k: 'trueA', label: 'true quality of A', min: .3, max: .9, step: .02, value: .62, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'pos', label: 'position bias', min: 0, max: .3, step: .01, value: .12, fmt: v => '+' + (v * 100).toFixed(0) + ' pts to first' },
          { k: 'verb', label: 'verbosity bias (B is longer)', min: 0, max: .3, step: .01, value: .1, fmt: v => '+' + (v * 100).toFixed(0) + ' pts to B' },
          { k: 'swap', label: 'mitigation: evaluate both orders', type: 'toggle', value: false },
          { k: 'lennorm', label: 'mitigation: normalise length', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'truth', label: 'true win rate for A', cls: 'key' },
          { k: 'meas', label: 'measured win rate', cls: 'bad' },
          { k: 'err', label: 'error' },
          { k: 'flip', label: 'order-flip disagreement' },
          { k: 'verdict', label: 'conclusion you would draw' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(41);
            const N = 400;
            let winA = 0, flips = 0;
            for (let i = 0; i < N; i++) {
              const judgeOnce = (aFirst) => {
                let pA = st.trueA;
                pA += aFirst ? st.pos : -st.pos;
                if (!st.lennorm) pA -= st.verb;
                return R() < Math.max(0.01, Math.min(.99, pA));
              };
              if (st.swap) {
                const r1 = judgeOnce(true), r2 = judgeOnce(false);
                if (r1 !== r2) flips++;
                if ((r1 ? 1 : 0) + (r2 ? 1 : 0) >= 1 && r1 === r2) winA += r1 ? 1 : 0;
                else winA += 0.5 * ((r1 ? 1 : 0) + (r2 ? 1 : 0));
              } else {
                if (judgeOnce(true)) winA++;
              }
            }
            const measured = winA / N;
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [-.6, 1.6], pad: { l: 40, r: 14, t: 16, b: 40 } })
              .frame({ yticks: [], xlabel: 'win rate for system A', xfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.vline(.5, { color: T.faint, dash: [3, 3], width: 1 });
              // bars
              [[st.trueA, T.c3, 'true', 1.0], [measured, T.c2, 'measured', 0.3]].forEach(([v, c, lbl, y]) => {
                ctx.fillStyle = c; ctx.globalAlpha = .8;
                ctx.fillRect(P.x(0), P.y(y) - 16, P.x(v) - P.x(0), 32);
                ctx.globalAlpha = 1;
                ctx.fillStyle = T.text; ctx.font = '11.5px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText(lbl + '  ' + (v * 100).toFixed(1) + '%', P.x(v) + 8, P.y(y));
              });
            });
            const wrongCall = (st.trueA > .5) !== (measured > .5);
            out({
              truth: (st.trueA * 100).toFixed(0) + '%',
              meas: (measured * 100).toFixed(1) + '%',
              err: ((measured - st.trueA) * 100).toFixed(1) + ' pts',
              flip: st.swap ? (flips / N * 100).toFixed(0) + '%' : 'not measured',
              verdict: wrongCall ? '✗ you would pick the WRONG system' : (Math.abs(measured - st.trueA) > .05 ? '~ right winner, wrong margin' : '✓ close enough')
            });
          }
        });
        Viz.note(host, 'With position bias at 12 points and verbosity bias at 10, a system that genuinely wins 62% of the time can measure below 50% — <b>you would ship the worse one.</b> Turn on <b>evaluate both orders</b> and the position bias cancels exactly; turn on <b>normalise length</b> and the verbosity bias goes with it. The order-flip disagreement rate is worth reporting on its own: it tells you what fraction of your judgements were decided by presentation rather than content.');
      }
    },
    quiz: [
      {
        q: 'A 100-example eval with a pass rate near 70% has a 95% confidence interval of roughly…',
        options: ['±1 point', '±3 points', '±9 points', '±20 points'],
        answer: 2,
        why: 'Plugging into the standard-error formula for a proportion, $1.96\\sqrt{p(1-p)/n}$ with $p=0.7$ and $n=100$, gives roughly 0.09 — a 95% interval of about ±9 points, exactly what the eval-size lab\'s interval curve plots at n=100 (§5.11.1). "±3 points" is the tempting wrong answer because it is roughly the interval you would get after quintupling the sample to n≈500 — the lab\'s second reference line — so it is easy to misremember which sample size a "reasonably sized" eval set actually buys you. "±1 point" undershoots by close to an order of magnitude and would need an eval set in the thousands, not the hundreds. The general principle is that an eval score is a sample estimate with real sampling noise, and most reported 2–3 point gains measured on around 100 examples sit entirely inside this interval, indistinguishable from nothing (§5.11.1\'s tldr).'
      },
      {
        q: 'The cheapest fix for position bias in an LLM judge is…',
        options: ['a bigger judge model', 'evaluating each pair in both orders and averaging', 'raising the temperature', 'more examples'],
        answer: 1,
        why: 'Position bias is a fixed offset favouring whichever slot an option happens to occupy, so judging each pair in both orders and averaging makes that offset cancel exactly, regardless of its size, and hands you the order-flip rate for free as a direct measure of how much of the signal was ever presentation rather than content (§5.11.2, the judge lab). "A bigger judge model" is the tempting answer because a stronger model is the reflexive fix for most quality problems in this stack, but position bias is documented as reproducible across model families — it does not shrink by choosing a better judge because it is a property of the judging setup, not of judge competence. "More examples" fails for a different reason: more data helps when the problem is noise, but position bias is systematic, so averaging more biased judgements just measures the biased number more precisely, not the true one. The general principle, organising the whole bias table in §5.11.2, is to separate bias (fixed by a targeted mitigation) from variance (fixed by more samples) — position bias is squarely the first kind.'
      },
      {
        q: 'Paired evaluation on the same items needs fewer examples because…',
        options: ['it is more accurate per item', 'item-difficulty variance cancels; only items where the systems differ carry information', 'it uses a better judge', 'it avoids position bias'],
        answer: 1,
        why: 'When both systems are scored on the identical item, an item they both get right, or both get wrong, tells you nothing about which one is better — all the comparative information lives in the items where they disagree, so the effective sample size for detecting a difference tracks the discordance rate rather than the raw item count, which is exactly what the lab\'s McNemar-style formula computes. "It is more accurate per item" sounds plausible because pairing does feel like a more careful measurement, but the accuracy of any single item\'s score is unchanged by pairing — what changes is which items carry statistical weight once you aggregate. "It avoids position bias" conflates two independent fixes discussed in the same section: pairing on identical items is what buys sample efficiency, while judging in both orders is the separate technique that cancels position bias, and a paired comparison graded in only one order still carries the full bias. The general principle is §5.11.1\'s key box: paired comparison removes item-difficulty variance from the estimate, which is why it needs several-fold fewer examples than scoring two systems independently.'
      },
      {
        q: 'The most valuable source of eval cases is…',
        options: ['a public benchmark', 'production failures and complaints', 'synthetic examples from an LLM', 'the training set'],
        answer: 1,
        why: 'A case you can invent, hand-picked or synthetic, is almost by definition a case within the range of situations you already anticipated, and the system usually already handles those; a real production failure is proof, by construction, of a case the system does not yet handle, which is exactly the information an eval set exists to capture (§5.11.3\'s golden-set step). "A public benchmark" is tempting because it looks like the rigorous, externally validated option, but §5.11\'s own flag notes that contamination has made a strong public-benchmark score weak evidence about a model released after that benchmark was published — and even ignoring contamination, a public benchmark contains none of your tools, your documents, or your users\' actual failure modes. "Synthetic examples from an LLM" shares the same blind spot as hand-invented cases: an LLM generating test cases draws on the same distribution of "things people would think to test," never on what actually broke in your system. The general principle, stated in this section\'s tldr, is to build the eval from your own production traffic, because the examples you can think of are the ones that were never the problem.'
      }
    ],
    cards: [
      { q: 'Eval set size', a: '±9 points at n=100 near a 70% pass rate. Pair on the same items to cut the requirement several-fold.' },
      { q: 'The five judge biases', a: 'Position, verbosity, self-preference, formatting, sycophancy. Both-orders + length normalisation + a different model family fixes most of it.' },
      { q: 'Judge quality bar', a: 'Report Cohen’s κ against human labels; compare it against human–human agreement, which is the real ceiling.' },
      { q: 'What belongs in every eval report', a: 'Score, interval, baseline, per-slice breakdown, and cost and latency in the same table.' }
    ]
  });

  /* ------------------------------------------------------------------ 5.12 */
  ML.section({
    id: 'mlops', track: 'applied', num: '5.12', level: 2,
    title: 'MLOps: skew, shadow, canary, rollback',
    lede: 'The model is perhaps a fifth of the system. The rest is the machinery that gets features to it identically in training and serving, releases a new version without betting the business on it, and notices when the world has moved.',
    prereq: ['production'],
    related: ['production', 'evals', 'robustness'],
    html: `
${H.tldr([
      '<b>Training/serving skew</b> is the most common production ML bug and the hardest to see: the same feature computed two ways in two codebases. A feature store, or literally sharing the transformation code, is the structural fix.',
      'Release in stages: <b>shadow</b> (compute, do not act), <b>canary</b> (act for 1%, then 5%, then 25%), <b>full</b>. Each stage answers a different question, and each has an automatic rollback trigger defined <i>before</i> it starts.',
      'Monitor <b>inputs</b> (immediately available), <b>predictions</b> (immediately available) and <b>outcomes</b> (delayed by the label lag) as three separate systems. Only the last can detect concept drift.'
    ])}

<h2><span class="sn">5.12.1</span> Training/serving skew</h2>
<p>Training happens once, in a notebook or a batch job, against a dataset that already contains the label. Serving happens continuously, on a low-latency request path, against whatever the world looks like right now, with no label anywhere in sight. Nothing forces the code that computes a feature in the first setting to be the same code that computes it in the second — they are usually written by different people, in different languages, months apart, against different constraints — so "the same definition, computed twice" silently becomes "two definitions that happen to agree today." <b>Training/serving skew</b> is what happens the day they stop agreeing.</p>
<p>It is the hardest production bug to see because it does not look like a bug. Every input the model receives is still a valid number in a valid range; every prediction it produces still looks like a normal prediction; nothing crashes, no schema is violated, no alert fires. The only symptom is that live accuracy is worse than the offline number, by an amount that is easy to misattribute to "the world changed" — concept drift, covered properly in §5.12.3 — rather than to the real cause: a pipeline quietly computing something different from what the model was trained on. A training-time feature-statistics dashboard and a serving-time one can each look completely healthy in isolation while silently describing two different features that happen to share a name.</p>
${H.worked('a time-travel bug, worked through', `<p>A churn model is trained to predict cancellation using "the customer's support-ticket count in the last 30 days," computed once, offline, by joining every historical ticket against every historical customer snapshot as of the label date. Offline, this feature is a strong predictor: a customer with several tickets shortly before cancelling is a clear signal, and because the join has access to the customer's entire history, it can see tickets filed right up to the moment of cancellation.</p>
<p>At serving time the same feature has to be computed live, "as of right now" — and a ticket the customer files ten minutes after the prediction obviously cannot be in that count, because it has not happened yet. The training pipeline, built after the fact from a complete history table, had no such restriction, and quietly included tickets filed <i>after</i> the decision point it was supposedly predicting from. The model learned a version of the feature that leaks the future. Offline accuracy looks excellent; in production, the same feature is systematically smaller, the correlation the model learned is partly gone, and nothing about the input data itself looks wrong.</p>
<p>The fix is not a better model. It is a point-in-time correct join: every feature carries an as-of timestamp, and training is required to use only data that would genuinely have existed at that timestamp — the identical discipline §0.5's leakage lab teaches for fitting a transform inside each fold, applied here to feature computation instead of to model selection.</p>`)}
${H.table(['Cause', 'Example', 'Fix'], [
      ['Two implementations', 'pandas in training, Java in serving', '<b>share the transformation code</b>, or a feature store that serves both'],
      ['Time travel', 'a training feature computed with data that did not exist yet at decision time', 'point-in-time correct joins; every feature carries an as-of timestamp'],
      ['Different defaults', 'training fills NA with the median, serving with 0', 'the imputer is part of the artefact, not the notebook'],
      ['Schema drift', 'an upstream column changes units or type', 'schema validation on every batch; fail loudly'],
      ['Different aggregation windows', '"last 30 days" means different things in the two systems', 'define windows once, in one place'],
      ['Feedback loops', 'the model’s own decisions become its next training data', 'log counterfactuals; hold out a random control slice permanently']
    ])}
${H.key('If a feature is computed in two places, it is computed two ways. This is not pessimism — it is what happens over a year of independent maintenance on two codebases. The structural fix is one definition, two callers.')}

<h2><span class="sn">5.12.2</span> The release ladder</h2>
<p>Shadow, canary and holdback are not caution for its own sake. Each rung of the ladder exists to bound one specific, nameable cost of being wrong, and the ladder only makes sense once you can see exactly what each stage is paying for.</p>
${H.worked('what a staged rollout is actually buying', `<p>Suppose a new version is quietly worse by some amount, and it takes $t_{\\text{detect}}$ hours of traffic before that regression clears whatever alert threshold is set. In any given hour $t$, the fraction of traffic running the new version is $\\mathrm{share}(t)$, so the number of requests served by the worse model is the running total $\\text{affected} = \\sum_{t=0}^{t_{\\text{detect}}} \\mathrm{share}(t)\\times\\text{traffic\\_per\\_hour}$ — exactly what the widget below accumulates, live, on every frame.</p>
<p>Big-bang release sets $\\mathrm{share}(t)=100\\%$ from hour zero, so the regression is loud, sampled from the entire traffic volume, and detection tends to be fast — but every hour before the alert fires costs 100% exposure. A staged rollout instead holds $\\mathrm{share}(t)$ near 1% for the first several hours: the same regression is now measured on a smaller, noisier slice of traffic, so it typically takes longer to clear the same threshold — but each of those hours costs only 1% of what a big-bang release would have cost. The trade is not "caution versus speed" in the abstract; it is detection latency, traded directly against the exposure multiplier at each stage, and the sum above is what lets you put a number on which side wins for a given regression size.</p>`)}

<p><b>What you are looking at.</b> The shaded backdrop marks what share of traffic runs the new version at each hour, under whichever rollout plan is selected; the solid line is the observed metric, carrying realistic noise that shrinks as more traffic is sampled at once. Two dashed horizontal lines mark the old version's baseline and the alert threshold below it; a solid vertical line appears at the hour the regression is first detected, if it ever clears that threshold at all.</p>
<p><b>What to do with it.</b> Set the plan to <b>big bang</b> with the default 3-point regression and note the detection hour and the affected-users count. Switch to <b>staged</b> at the same regression size and watch detection take noticeably longer — the 1% arm is a noisier sample, so the signal needs more hours to clear the same threshold — while the affected-users count drops by roughly two orders of magnitude. Now shrink the regression toward 0.5 points on either plan and watch detection slip away entirely: a small enough regression on a noisy metric can sit permanently below the threshold, undetected, regardless of which plan you chose.</p>
<p><b>The thing genuinely worth noticing.</b> Staged rollout does not make detection faster or the regression easier to see — if anything it makes detection slower, because it deliberately waits for a signal measured on a small sample rather than a large one. What it buys instead is a smaller exposure multiplier for every hour that passes before that signal arrives, which is exactly the $\\mathrm{share}(t)\\times\\text{traffic}$ arithmetic above. That makes it a <b>containment</b> strategy, not a <b>detection</b> strategy, and confusing the two is the most common reason a team over-trusts a staged rollout to catch something a small, slow regression will simply sail past.</p>
${H.lab('rollout', 'A staged rollout with an injected regression', 'A new model version is 3% worse on a metric you monitor. Choose the rollout plan and the alert threshold, and see how many users are affected before the automatic rollback fires. There is a real trade here between caution and speed.')}
${H.table(['Stage', 'Question it answers', 'Traffic', 'Typical duration'], [
      ['<b>Shadow</b>', 'does it run, at what latency, and do its outputs look sane?', '100% computed, 0% acted on', '1–7 days'],
      ['<b>Canary 1%</b>', 'does it break anything catastrophically?', '1%', 'hours to a day'],
      ['Canary 5–25%', 'is the metric moving in the right direction?', '5–25%', 'until powered (§2.25)'],
      ['Full', '—', '100%', '—'],
      ['<b>Permanent holdback</b>', 'what is the model worth, cumulatively?', '0.5–2% never treated', 'forever']
    ])}
${H.note('The permanent holdback is the one most teams skip and most regret skipping. Without it you cannot answer "what is this model actually contributing?" a year later, and you have no clean data to retrain on that is free of the model’s own influence.')}

<h2><span class="sn">5.12.3</span> Monitoring, in three layers</h2>
${H.table(['Layer', 'Signal', 'Latency', 'Detects'], [
      ['<b>Inputs</b>', 'PSI/KS per feature, null rates, cardinality, range violations', 'immediate', 'covariate shift, upstream breakage'],
      ['<b>Predictions</b>', 'score distribution, alert volume, positive rate, confidence histogram', 'immediate', 'a sudden change in behaviour'],
      ['<b>Outcomes</b>', 'AUC, calibration, business metric, per-slice', 'delayed by the label lag', '<b>concept drift</b> — nothing else can'],
      ['System', 'p50/p99 latency, error rate, cost per request, cache hit rate', 'immediate', 'everything the SRE cares about']
    ])}
${H.intuition(`<p>The three layers are not three redundant views of the same health signal; they are watching three different, and separable, pieces of one joint distribution, $p(x,y)=p(x)\\,p(y\\mid x)$. Input monitoring watches $p(x)$ directly — is the world sending the model the same kind of requests it always has? Prediction monitoring watches the model's own output distribution, which is really watching its learned approximation $\\hat p(y\\mid x)$, but only from the inside: a confident, well-formed, wrong prediction looks identical to a confident, well-formed, right one from this vantage point alone. Only outcome monitoring ever compares a prediction against the true $y$, which is the only way to notice that the actual relationship between $x$ and $y$ has moved.</p>
<p>That is why a model can be failing badly while every input and prediction dashboard stays green: if $p(x)$ has not moved and the model still produces the same kind of confident output it always has, the first two layers have nothing to say, because neither of them was ever built to see $p(y\\mid x)$ in the first place. The label lag is not a shortcoming of outcome monitoring to be engineered away — it is the price of being the only layer that looks at the one thing that can actually be wrong.</p>`)}
${H.pitfall('PSI is a rule of thumb, not a test: >0.1 "investigate", >0.25 "significant shift". It is sensitive to binning and to sample size, and it will fire on a harmless seasonal pattern while missing a genuine concept drift that leaves the input marginals unchanged. Alert on it, but never let it be the only thing you watch — <b>input monitoring cannot detect a change in $p(y\\mid x)$.</b>')}
${H.checklist([
      'Every model artefact is versioned with its training data snapshot, code commit and hyperparameters — reproducible from those three alone.',
      'Rollback is a config change, not a redeploy, and it has been tested this quarter.',
      'Alert thresholds are defined <i>before</i> the rollout begins, with an owner and a runbook.',
      'A permanent randomised holdback exists and is excluded from training data.',
      'Retraining is scheduled and its output is gated by the same eval as a manual release — <b>automated retraining without an automated gate is an automated way to ship a bad model.</b>',
      'The feature transformation code is shared between training and serving, or served from one store.'
    ])}

${H.probe([
      ['What is training/serving skew and how do you eliminate it?', 'The same feature computed differently in the two paths. Eliminate it structurally: one implementation with two callers, or a feature store; and add a monitor comparing a sample of serving-time features against recomputed training-time values.'],
      ['Why shadow before canary?', 'Shadow answers the operational questions — does it run, at what latency, are the outputs sane — with zero user risk. Canary answers the quality question and necessarily has user risk. Do them in that order.'],
      ['Input drift is flat but accuracy fell. What happened?', 'Concept drift: $p(y\\mid x)$ changed while $p(x)$ did not. Only labelled outcomes reveal it, which is why outcome monitoring cannot be replaced by input monitoring.'],
      ['Why keep a permanent holdback?', 'To measure the model’s cumulative contribution and to retain a slice of data uncontaminated by the model’s own decisions.'],
      ['Automated retraining ran and the model got worse. What was missing?', 'A gate. Retraining must be blocked by the same eval suite, baseline comparison and canary process as a human release.']
    ])}`,
    labs: {
      rollout: function (host) {
        const st = Viz.controls(host, [
          { k: 'reg', label: 'regression in the new version', min: 0, max: .1, step: .005, value: .03, fmt: v => '−' + (v * 100).toFixed(1) + ' pts' },
          { k: 'plan', label: 'rollout plan', type: 'select', value: 'staged', options: [
            { v: 'big', t: 'big bang — 100% immediately' }, { v: 'fast', t: 'fast: 10% → 100%' },
            { v: 'staged', t: 'staged: 1% → 5% → 25% → 100%' }, { v: 'slow', t: 'cautious: shadow → 1% → 5% → 25% → 50% → 100%' }] },
          { k: 'thresh', label: 'alert threshold', min: .005, max: .06, step: .005, value: .02, fmt: v => (v * 100).toFixed(1) + ' pts' },
          { k: 'traffic', label: 'requests per hour', min: 1000, max: 500000, step: 1000, value: 60000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'detect', label: 'detected at hour', cls: 'key' },
          { k: 'affected', label: 'users served a worse model', cls: 'bad' },
          { k: 'pct', label: 'as a share of traffic' },
          { k: 'delay', label: 'time to full rollout if healthy', cls: 'good' },
          { k: 'note', label: 'verdict' }
        ]);
        const PLANS = {
          big: [[0, 1]],
          fast: [[0, .1], [6, 1]],
          staged: [[0, .01], [6, .05], [18, .25], [36, 1]],
          slow: [[0, 0], [12, .01], [24, .05], [42, .25], [66, .5], [90, 1]]
        };
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const plan = PLANS[st.plan];
            const HOURS = 120;
            const R = Num.rng(3);
            const base = .82;
            const share = t => { let s = 0; plan.forEach(p => { if (t >= p[0]) s = p[1]; }); return s; };
            const series = [], shares = [];
            let detected = null, affected = 0;
            for (let t = 0; t < HOURS; t++) {
              const sh = detected == null ? share(t) : 0;
              shares.push([t, sh]);
              const obs = base - st.reg * sh + R.normal(0, .004 / Math.sqrt(Math.max(.01, st.traffic / 20000)));
              series.push([t, obs]);
              if (detected == null) affected += sh * st.traffic;
              // detection needs the drop to exceed the threshold, and enough traffic on the new arm
              if (detected == null && t > 1 && (base - obs) > st.thresh && sh > 0) detected = t;
            }
            const P = Viz.plot(ctx, w, h, {
              xd: [0, HOURS], yd: [base - .12, base + .02], pad: { l: 56, r: 46, t: 16, b: 40 }
            }).frame({ xlabel: 'hours since release', ylabel: 'observed metric', yfmt: v => (v * 100).toFixed(1) + '%' });
            P.clip(() => {
              // traffic share as a shaded backdrop
              shares.forEach((s, i) => {
                if (!s[1]) return;
                ctx.fillStyle = T.c1; ctx.globalAlpha = .10 + .18 * s[1];
                ctx.fillRect(P.x(s[0]), P.pad.t, Math.max(1, P.x(1) - P.x(0)), P.ph);
                ctx.globalAlpha = 1;
              });
              P.line(series, { color: T.c1, width: 1.8 });
              P.hline(base, { color: T.c3, dash: [4, 4], label: 'old version' });
              P.hline(base - st.thresh, { color: T.c2, dash: [4, 4], label: 'alert threshold' });
              if (detected != null) P.vline(detected, { color: T.c2, width: 2.4, dash: false, label: 'rollback' });
            });
            const full = plan[plan.length - 1][0];
            out({
              detect: detected == null ? 'never (regression too small)' : 'hour ' + detected,
              affected: Math.round(affected).toLocaleString(),
              pct: (affected / (st.traffic * HOURS) * 100).toFixed(2) + '%',
              delay: full + ' h',
              note: detected == null ? '⚠ below the noise floor — needs a longer window or more traffic'
                : (affected < st.traffic * 2 ? '✓ contained' : 'many users exposed')
            });
          }
        });
        Viz.note(host, 'Try <b>big bang</b> with a 3-point regression: it is detected within an hour or two, and by then everyone has had it. Try <b>staged</b>: detection takes longer, because 1% of traffic is a noisy sample — but the number of affected users is two orders of magnitude smaller. <b>That is the actual trade, and it is not "caution versus speed" — it is exposure versus detection latency.</b> Now shrink the regression to 0.5 points and watch it slip past every plan: a small regression on a noisy metric is invisible at canary scale, which is why staged rollout is a containment strategy, not a detection strategy.');
      }
    },
    quiz: [
      {
        q: 'Training/serving skew is best prevented by…',
        options: ['more tests on the serving code', 'one feature-transformation implementation used by both paths', 'retraining more often', 'monitoring accuracy'],
        answer: 1,
        why: 'Skew arises because the same feature is computed by two independently maintained codebases (§5.12.1\'s table gives pandas-in-training, Java-in-serving as the canonical case), and any two implementations of the same logic drift apart over a year of separate maintenance no matter how carefully either team writes it — so the structural fix is to leave nothing to drift, one implementation with two callers, or a feature store serving both. "More tests on the serving code" is tempting because more testing is the general-purpose answer to "stop code from behaving unexpectedly," but tests written against one side of a two-codebase problem cannot detect that the other side computes something subtly different; catching that requires comparing the two outputs directly, which is really a weaker version of just sharing the implementation. "Monitoring accuracy" only catches skew after it has already degraded predictions in production, which is detection, not prevention. The general principle, stated in this section\'s key box, is that a feature computed in two places is a feature computed two ways — eliminate the duplication rather than trying to police it.'
      },
      {
        q: 'Shadow deployment means…',
        options: ['deploying to a staging environment', 'running the new model on real traffic but not acting on its outputs', 'deploying to 1% of users', 'A/B testing two models'],
        answer: 1,
        why: 'Shadow mode runs the new model against real, live traffic, computing predictions and measuring latency and output sanity, while only the old model\'s outputs are ever acted on — so it answers the purely operational questions at zero user risk, because nothing the shadow model produces ever reaches a user (§5.12.2\'s release-ladder table). "Deploying to a staging environment" is tempting because it also sounds like a safe pre-production step, but staging traffic is synthetic or replayed rather than the real, messy production distribution shadow mode is deliberately exposed to, so it cannot surface the production-only edge cases shadow mode exists to catch. "Deploying to 1% of users" is the next rung up, canary, and it does carry real user risk — exactly the distinction the ladder is organised around, since shadow answers "does it run" before canary risks answering "is it good." The general principle is that each stage of the release ladder answers a different question, and skipping the order means taking on user risk before ruling out the cheap, riskless failures shadow mode is built to find.'
      },
      {
        q: 'Input drift monitoring cannot detect…',
        options: ['covariate shift', 'concept drift, where $p(y|x)$ changes but $p(x)$ does not', 'schema changes', 'null-rate spikes'],
        answer: 1,
        why: 'Input monitoring — PSI, null rates, cardinality — only ever measures the distribution of $x$, so it is blind by construction to a change in the relationship between $x$ and $y$: if the inputs look statistically identical to before but the world has changed how $y$ depends on them, nothing in the input distribution moves and nothing fires, which is exactly why outcome monitoring, delayed by the label lag, exists as a separate, unavoidable layer (§5.12.3\'s three-layer table). "Covariate shift" is the wrong pick specifically because it names the thing input monitoring is designed to catch — a shift in $p(x)$ itself is precisely what PSI and null-rate checks measure, so choosing it has the mechanism backwards. "Schema changes" and "null-rate spikes" are likewise both squarely input-layer phenomena this monitoring catches immediately, not the delayed, label-dependent case the question is asking about. The general principle is to track which of $p(x)$, $p(y)$, or $p(y\\mid x)$ has changed, and to know that only labelled outcomes can ever reveal a change in the last one, however sophisticated the input monitoring becomes.'
      },
      {
        q: 'The main benefit of a permanent randomised holdback is…',
        options: ['faster inference', 'measuring the model’s cumulative value and keeping uncontaminated training data', 'reducing cost', 'satisfying auditors'],
        answer: 1,
        why: 'A slice of traffic permanently excluded from the model\'s decisions supplies two things nothing else can: a genuine counterfactual against which to measure what the model is actually worth, cumulatively, long after launch, and a pool of training data untouched by the model\'s own past decisions — which matters because a model retrained partly on the outcomes of its own earlier choices is training on a feedback loop, not on the world (§5.12.1\'s feedback-loop row and §5.12.2\'s note). "Satisfying auditors" is tempting because holdbacks do come up in compliance conversations, but that is a downstream consequence rather than the reason the practice exists, or the reason teams that skip it come to regret it. "Faster inference" and "reducing cost" both mistake a holdback, which excludes a slice of users from treatment rather than changing how the model runs, for an engineering optimisation it has nothing to do with. The general principle, stated directly in this section\'s note, is that both benefits — attribution and clean retraining data — are essentially impossible to reconstruct retroactively once the whole population has already been exposed to the model.'
      }
    ],
    cards: [
      { q: 'Training/serving skew', a: 'The same feature computed two ways. Fix structurally: one implementation, two callers, or a feature store.' },
      { q: 'The release ladder', a: 'Shadow → canary 1% → 5–25% → full, plus a permanent holdback. Thresholds and owners defined before the rollout starts.' },
      { q: 'Three monitoring layers', a: 'Inputs (immediate, covariate shift), predictions (immediate, behaviour change), outcomes (delayed, concept drift).' },
      { q: 'PSI thresholds', a: '>0.1 investigate, >0.25 significant. A rule of thumb sensitive to binning — never the only signal.' },
      { q: 'Automated retraining', a: 'Must be gated by the same eval, baseline and canary as a human release, or it is an automated way to ship a bad model.' }
    ]
  });
})();
