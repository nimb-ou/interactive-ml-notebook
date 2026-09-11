/* Formula test: parses every $…$ and $$…$$ in the corpus through KaTeX itself.
   This exists because test/smoke.js deliberately discards katex warnings, so a
   malformed formula renders as red error text in the browser while the suite
   still reports errors : 0. Run: node test/formulas.js  */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const katex = require(path.join(ROOT, 'vendor/katex/dist/katex.min.js'));

/* Flatten helper args to raw strings. Never JSON.stringify here: that
   re-escapes every backslash and turns \top into \\top, which then fails to
   parse and produces a flood of false positives. */
function flat(x) {
  if (typeof x === 'string') return x;
  if (Array.isArray(x)) return x.map(flat).join(' ');
  if (x && typeof x === 'object') return Object.values(x).map(flat).join(' ');
  return '';
}

const sections = [];
global.window = global;
global.ML = { section: s => sections.push(s), track: () => {} };
global.H = new Proxy({}, { get: () => (...a) => a.map(flat).join(' ') });
global.Viz = new Proxy({}, { get: () => () => {} });
global.Num = new Proxy({}, { get: () => () => {} });

const dir = path.join(ROOT, 'js/content');
const loadErrors = [];
for (const f of fs.readdirSync(dir).filter(n => n.endsWith('.js'))) {
  try { (0, eval)(fs.readFileSync(path.join(dir, f), 'utf8')); }
  catch (e) { loadErrors.push(f + ': ' + e.message.split('\n')[0]); }
}

function mathsIn(text) {
  const out = [];
  if (typeof text !== 'string') return out;
  text.replace(/\$\$([\s\S]+?)\$\$/g, (m, body) => { out.push({ body, display: true }); return ''; });
  text.replace(/\$\$[\s\S]+?\$\$/g, '')
      .replace(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g, (m, body) => { out.push({ body, display: false }); return ''; });
  return out;
}

function walk(v, acc) {
  if (typeof v === 'string') acc.push(v);
  else if (Array.isArray(v)) v.forEach(x => walk(x, acc));
  else if (v && typeof v === 'object') Object.values(v).forEach(x => walk(x, acc));
  return acc;
}

let total = 0;
const failures = [];
sections.forEach(s => {
  walk({ html: s.html, lede: s.lede, title: s.title, quiz: s.quiz, cards: s.cards, rests: s.rests }, [])
    .forEach(text => mathsIn(text).forEach(({ body, display }) => {
      total++;
      try { katex.renderToString(body, { displayMode: display, throwOnError: true, strict: false }); }
      catch (e) {
        failures.push({
          sec: (s.num || '?') + ' ' + s.id,
          body: body.replace(/\s+/g, ' ').slice(0, 90),
          err: String(e.message).replace(/\s+/g, ' ').slice(0, 130)
        });
      }
    }));
});

console.log('sections  :', sections.length);
console.log('formulas  :', total);
console.log('errors    :', failures.length + loadErrors.length);
loadErrors.forEach(e => console.log('  - load ' + e));
failures.slice(0, 40).forEach(f => console.log('  - §' + f.sec + '  ' + f.body + '\n      ' + f.err));
process.exit(failures.length + loadErrors.length ? 1 : 0);
