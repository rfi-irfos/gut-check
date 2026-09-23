# gut-check: Scaling Phase 2 — Design

## Context

Phase 1 (this repo's first ~24h) shipped: the core gate library, the Claude-Code + Hermes mining pipelines, a teacher-labeling pipeline that scaled Stage C from 55 hand-verified items to 5,400 teacher-labeled items (1.1% measured teacher error rate on the latest 88-item verification sample — well under the 47% heuristic baseline), a growing gold eval set (74 items, up from the original 55), a live-harness that tests real running Hermes sessions and found two genuine CONTRADICTED failures (both arithmetic/counting slips under multi-step load, not fabrication), a draft-but-tested lauras-agents-kernel integration (PR #8, 335 tests green, FlagOnly), and a structurally-tested (12/12) hermes-agent native plugin that directly imports `core/gate`.

Best fine-tuned checkpoint so far: 51.4% gold-set accuracy (seed 4, 5,400-item corpus, 74-item gold set) — but this has never been compared against a heuristic baseline recomputed on the *current* 74-item gold set. The known 52.7% figure is stale, measured only against the original 55 items.

This phase has two goals running in sequence with gates between them: (1) make the data/eval foundation trustworthy and growing on its own, (2) generalize the integration layer from "hermes-specific plugin" to "any MCP-capable agent host," and bring the already-built kernel integration live.

## Sequencing

Four blocks, each gating the next:

1. **Heuristic rebaseline** — recompute the heuristic baseline against the current (and future) gold set, not a stale one-time number.
2. **Stage C scaling** — deepen Hermes mining across all non-cron sessions, turn mining + labeling into a recurring automated job instead of one-off manual batches.
3. **MCP server** — expose `core/gate` as a generic MCP tool usable by any MCP-capable host (Claude Code, Codex, Gemini CLI, hermes-agent), published to Smithery. Consolidate hermes-agent's existing native plugin onto this path.
4. **lauras-agents-kernel PR merge** — bring PR #8 out of draft, live-enable the sidecar (still FlagOnly, no behavior change beyond visibility).

Blocks 1-2 improve the checkpoint and its measurement before block 3 exposes it to more consumers. Block 4 is independent of 1-3 (Rust path doesn't depend on the MCP server) and is gated mainly on team review, not more building.

## Block 1: Heuristic rebaseline

**Problem:** the 52.7% heuristic baseline is stale (measured once against the original 55-item gold set). The gold set has since grown to 74 items, 19 of which are live-harness-sourced Hermes items that don't carry the Claude-Code-trace tool metadata (`is_error` flags, structured `nearest_action`/`nearest_observation` fields) the original heuristic script expects.

**Design:**
- New `training/eval/heuristic_baseline.py` in this repo — a clean-room port of the original heuristic logic (is_error check + the four known fix classes already documented in `common_relevance_filter.py`: policy-block false positives, error-word-anywhere false positives, negation polarity, topical irrelevance).
- Generalized to operate on `claim_text` + the normalized `context` field already present on every `gold_eval.jsonl` row (regardless of original source), rather than requiring raw structured tool-call metadata. This means extracting an "is_error"-equivalent signal from context text patterns for Hermes-sourced items, where no structured flag exists.
- Wired into `human_verify_sample.py score` (or a new `score.py`) so the heuristic baseline is recomputed automatically every time the gold set grows, instead of being a stale hardcoded number quoted from memory.

**Done when:** running the baseline script against the current 74-item gold set produces a real, reproducible accuracy/error-rate number, and the fine-tuned checkpoint's gold accuracy is reported alongside it (not a stale comparison).

## Block 2: Stage C scaling — recurring mining + labeling

**Problem:** the Claude-Code corpus's naturally-occurring CONTRADICTED/VERIFIED-heuristic candidates are nearly exhausted (3-4 left unlabeled out of 5,704 mined). The Hermes corpus has only been shallowly mined (116 candidates from 78 non-cron sessions, never scaled). Growth has so far been one-off manual batches ("Vollgas" runs), not a sustained process.

**Design:**
- Deepen `extract_hermes_traces.py`'s reach across all non-cron Hermes sessions (cron sessions remain a held-out eval stratum per the original plan, not a training source).
- Convert mining from one-off scripts into a recurring cron job (matching this org's existing cron-as-production pattern): periodically re-runs both Claude-Code and Hermes extraction, dedupes against already-labeled `claim_text`s, appends new candidates to the prioritized queue (CONTRADICTED-heuristic first, as already established).
- The same cron job also triggers small, periodic teacher-labeling batches (not just mining) — fully automated end-to-end growth, not a manually-triggered step. Concurrency stays conservative (5-8, per the established NIM rate-limit ceiling) and batch size stays small enough to not blow through NIM free-tier quota unattended.
- Every batch still respects the mandatory go/no-go verification gate (stratified human-review sample, scored against the current heuristic baseline from Block 1) before being trusted for fine-tuning — this doesn't change, it just runs on a schedule instead of on demand.

**Done when:** a cron job exists that mines, labels, and appends to the Stage C corpus on a recurring schedule without manual triggering, and at least one automated cycle has run end-to-end successfully.

## Block 3: MCP server — generic agent-host integration

**Problem:** the current integration path is a hermes-agent-specific pip package that directly imports `core/gate`. This doesn't generalize to other agent hosts (Claude Code, Codex, Gemini CLI) without writing a bespoke plugin for each one.

**Design:**
- New `adapters/mcp_server/` package in gut-check. Exposes `core/gate` via a single MCP tool, `verify_claim(claim_text, context) -> GateVerdict` (label, confidence, whether escalation is recommended).
- The MCP server itself never calls a System-2 LLM — consistent with `core/gate`'s existing philosophy (classify and signal, never verify or block on its own). On low confidence, the verdict says so; the calling host decides whether/how to escalate.
- Transport: stdio for local CLI hosts (Claude Code, Codex, Gemini CLI). Consider an HTTP/SSE variant if hermes-agent's architecture makes that a better fit than stdio.
- Packaged as a standalone Python package (`laya` + `core` as dependencies), with a `smithery.yaml` for publishing to the Smithery MCP registry.
- The existing `adapters/hermes_plugin` (native, direct `core/gate` import, 12/12 tests green) gets converted to an MCP client calling this server, consolidating hermes-agent onto the same integration pattern as every other host rather than keeping a separate bespoke path.

**Done when:** the MCP server runs locally and responds correctly to `verify_claim` calls from at least one real MCP client (e.g., this Claude Code session, or a manual MCP inspector call), the hermes plugin's test suite passes against the MCP-client version, and the package is ready to publish (`smithery.yaml` present, README documents the tool contract).

## Block 4: lauras-agents-kernel PR merge

**Problem:** PR #8 (`KernelComponent::Laya`, `laya_gate.rs`, wired into all 3 Mission Control dispatch loops) is built, tested (335 passing, no regressions), and still in draft. No further building is needed — this is a review/merge/deploy step.

**Design:**
- No code changes. Bring PR #8 out of draft, route it through the team's existing multi-round review culture for `kernel_events.rs` changes (documented in-repo).
- On merge, set `LAYA_GATE_URL` in the relevant deployment environment so `laya_gate.rs` makes real sidecar calls instead of falling back to a no-op skip verdict.
- Stays FlagOnly per the existing `Block → FlagOnly, never Block → Block` doctrine — this is a visibility change, not a behavior change to the agent runs themselves.

**Done when:** PR #8 is merged to `main` and `LAYA_GATE_URL` is set in production, with real (non-fallback) sidecar traffic observable in logs.

## Open items deferred to implementation time

- Whether the MCP server needs an HTTP/SSE transport for hermes-agent or stdio suffices — decide when actually wiring hermes-agent's MCP client.
- Exact cron schedule/batch size for the recurring Stage C job — start conservative (e.g., daily, small batch) and tune based on observed NIM quota headroom.
- Whether the heuristic-baseline generalization (Block 1) fully closes the gap for Hermes-sourced context, or whether some gold items remain structurally inapplicable to the heuristic (e.g., multi-turn live-harness items with no single "nearest tool observation") — acceptable to report a partial-coverage baseline if so, as long as it's labeled honestly rather than papered over.
