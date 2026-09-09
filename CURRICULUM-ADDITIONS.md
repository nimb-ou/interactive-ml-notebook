# Planned additions and reframings

From a curriculum audit of all 116 sections. Ordered by value, not by section number.

**Verification rule.** Several items below rest on fast-moving 2026 claims. Nothing here goes
into the course as a flat assertion until it has been checked against a primary source. Where
a claim is genuinely contested or moving, it belongs behind `H.flag`, which is the existing
convention for exactly this. A course whose selling point is that its numbers survive being
questioned cannot afford a section written from a summary.

## New sections — tier 1

Free numbering slots exist at 1.11, 2.20, 3.10, 4.18, 5.8 and 6.5. Five of the six are filled
by the list below, which also removes the visible jumps in the sidebar.

| § | Title | Track | Why |
|---|---|---|---|
| 1.11 | Heterogeneous treatment effects: CATE, uplift, double ML | foundations | §1.7 stops at average treatment effects. Targeting, pricing and personalisation are all CATE problems. Bridges §1.7 to §2.25. |
| 4.18 | Attention alternatives in production: linear attention, gated DeltaNet, hybrid layouts | llm | §4.8 covers Mamba framed as experimental. Hybrid linear/full layouts are now a mainstream production choice, not a curiosity. |
| 5.8 | Agentic RL: training the loop, not just prompting it | applied | Absent entirely. §5.3 is harness engineering, §4.12 is single-turn, §6.1 is textbook MDPs. Multi-turn credit assignment under sparse outcome rewards is the gap. |
| 2.20 | Tabular foundation models, and when they beat boosting | classical | Part 2 spends four sections making boosting the tabular default. In-context learning over a synthetic prior is conceptually load-bearing and links Part 2 to Part 4. Attach the caveat about single-model vs tuned-ensemble comparisons. |
| 6.5 | World models, video generation, vision-language-action | frontier | Part 6's stated remit is restoring deliberate exclusions. This is one. |

## Extensions to existing sections — tier 1

- **§4.12 `post-training` — the GRPO family (DAPO, GSPO, Dr.GRPO).** The section already
  teaches vanilla GRPO *including* its degenerate-unanimous-group failure, and then stops.
  DAPO's dynamic sampling is the published fix for that precise failure, GSPO's sequence-level
  ratio is what stabilises RL on MoE models, Dr.GRPO removes a length bias. The highest
  value-per-word addition in the audit, because the setup is already written.
- **§5.11 `evals` — pass@k versus pass^k.** `pass@k` appears nowhere in the corpus. Agents are
  stochastic across runs, and $\text{pass}^k = p^k$ means a 90% agent is 57% reliable at
  $k=8$. Slots beside the existing eval-sample-size lab.
- **§4.17 `safety` — circuits: induction heads, attribution graphs, steering.** The course
  asserts in-context learning repeatedly and never explains the mechanism. §4.17.4 is
  currently SAEs only. Note honestly that SAE feature stability is contested.

## Tier 2

- **§4.14 / §5.4** — prompt caching as the primary cost lever, and cache-hit rate as an
  explicit optimisation target when constructing agent context.
- **§3.13 / §4.12** — on-policy distillation, where the teacher grades the student's own
  rollouts. Architecturally different from the classic soft-label KD already covered.
- **§3.10** — kernels and compilers: Triton, `torch.compile`, and where the FLOPs actually go.
  The course teaches bandwidth arithmetic well but never how anyone acts on it. Fold in
  FP4/NVFP4 microscaling; §3.13 currently stops at NF4.
- **§4.15 / §6.2** — diffusion language models. §6.2 is images-only and §4.15 is
  autoregressive-only; this unifies them.
- **§2.2** — grokking and emergent abilities. Double descent is covered; these complete the
  generalisation story.

## Reframings

- **§4.8 `moe`** — routing has moved past the auxiliary loss to auxiliary-loss-free balancing
  with shared experts. The lede understates hybrids.
- **§4.12 `post-training`** — the SFT→RLHF→DPO→GRPO→RLVR title reads as a lineage. Preference
  optimisation is better framed as a polish step than as RLHF's successor.
- **§5.3 `agents` / §5.4** — "long-term memory is a vector store plus periodic summarisation"
  is now contested. Under prompt caching, retaining full history can beat compaction on cost,
  latency and recall at once. Reframe compaction as a response to a named constraint rather
  than a default. Add just-in-time retrieval and sub-agent context isolation.
- **§4.20 `reasoning`** — predates thinking budgets as a user-facing parameter and interleaved
  thinking between tool calls, which supersedes the plain ReAct framing in §5.3.
- **§2.7 / §2.8** — attach the tabular-foundation-model caveat to "boosting is the default".

## Bug to fix

- **`js/content/70-frontier.js`, §6.9 `exclusions`** — internal headings render as 6.5.1–6.5.4
  while the section is numbered 6.9. A renumbering leftover, and the only section-number
  inconsistency in the corpus.

## Audit result on existing content

Roughly 25 numeric claims were independently recomputed — Hoeffding row counts, the Bayes
base-rate example, speculative decoding tokens per pass, the conformal quantile index, A/B
sample size per arm, Adam optimiser state in GB, Condorcet probabilities — and all of them
check out. Every `§X.Y` cross-reference resolves to a real section. The existing material is
accurate; this is an expansion job, not a correction job.
