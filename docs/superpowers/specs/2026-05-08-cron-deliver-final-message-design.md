# Cron `deliver_final_message_only` — Design Spec

**Date:** 2026-05-08  
**Status:** Approved  
**Scope:** `crates/zeroclaw-runtime`, `crates/zeroclaw-config`

---

## Problem

When a cron job runs with `delivery.mode = "announce"`, the runtime delivers the full
concatenated transcript of all assistant text turns to the channel. For verbose models
(Claude Opus 4.x, GLM-4, Qwen, DeepSeek with reasoning) that interleave prose narration
with tool calls, this causes intermediate reasoning to leak into the channel output
alongside the agent's intended final message.

Prompt-engineering workarounds are insufficient — the behaviour is structural to how
these models interleave text and tool calls inside a single API response.

---

## Goals

- Add an opt-in per-job flag that delivers only the **final** assistant message when
  `delivery.mode = "announce"`.
- `cron_runs.output` continues to store the **full** transcript for operator debugging.
- Default `false` — existing deployments are unaffected.
- No changes to the agent loop's core behaviour or prompt construction.

---

## Non-goals

- Modifying how the agent loop produces text (no model-side prompting changes).
- Stripping `<think>` tags — that is tracked separately in #6040 / #6174.
- Restructuring `cron_runs.output` storage.
- Changing default behaviour for any existing configuration.

---

## Design

### 1. Config layer

Add `deliver_final_message_only: bool` (default `false`) to two structs:

**`crates/zeroclaw-runtime/src/cron/types.rs` — `DeliveryConfig`:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeliveryConfig {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    #[serde(default = "default_true")]
    pub best_effort: bool,
    #[serde(default)]
    pub deliver_final_message_only: bool,   // NEW
}
```

**`crates/zeroclaw-config/src/schema.rs` — `DeliveryConfigDecl`** (TOML declarative form):

```rust
pub struct DeliveryConfigDecl {
    #[serde(default = "default_delivery_mode")]
    pub mode: String,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    #[serde(default = "default_true")]
    pub best_effort: bool,
    #[serde(default)]
    pub deliver_final_message_only: bool,   // NEW
}
```

The field is propagated wherever `DeliveryConfigDecl` is converted to `DeliveryConfig`
during the declarative job sync.

**Example TOML usage:**

```toml
[[cron.jobs]]
id = "morning-digest"
job_type = "agent"
schedule = { kind = "cron", expr = "0 7 * * *" }
prompt = "Produce the morning status summary."

[cron.jobs.delivery]
mode = "announce"
channel = "telegram"
to = "123456789"
deliver_final_message_only = true
```

---

### 2. `AgentRunOutput` struct and loop tracking

A new public struct is added in `crates/zeroclaw-runtime/src/agent/loop_.rs`:

```rust
#[derive(Debug, Default)]
pub struct AgentRunOutput {
    /// Full concatenated transcript — stored in cron_runs.output.
    pub full_text: String,
    /// Last non-empty display_text from the run — used for announce delivery.
    pub last_message: String,
}
```

Inside the iteration loop, a `last_display_text` tracker is added alongside
`accumulated_display_text`:

```rust
let mut accumulated_display_text = String::new();
let mut last_display_text = String::new();
```

At every site that pushes to `accumulated_display_text`, `last_display_text` is updated
when the new value is non-empty:

```rust
if !display_text.is_empty() {
    last_display_text = display_text.clone();
}
accumulated_display_text.push_str(&display_text);
```

There are four push sites (final-response path ~line 1460, intermediate tool-call path
~line 1495, and two paths in the max-iterations forced-summary section ~lines 2044).

`run()` return type changes from `Result<String>` to `Result<AgentRunOutput>`:

```rust
return Ok(AgentRunOutput {
    last_message: if last_display_text.is_empty() {
        accumulated_display_text.clone()   // fallback when all turns were tool-only
    } else {
        last_display_text
    },
    full_text: accumulated_display_text,
});
```

**Fallback rule:** if `last_display_text` is still empty after the run (every iteration
was tool-only with no prose), `last_message` falls back to `full_text` so the channel
always receives something.

---

### 3. Call chain updates

**`scheduler.rs:run_agent_job`** — return type `(bool, String)` → `(bool, String, String)`:

```rust
Ok(output) => (
    true,
    if output.full_text.trim().is_empty() { "agent job executed".to_string() }
    else { output.full_text },
    output.last_message,
),
Err(e) => (false, format!("agent job failed: {e}"), String::new()),
```

**`execute_job_with_retry`** — return type `(bool, String)` → `(bool, String, String)`;
threads both strings through without modification.

**`execute_and_persist_job`** — unpacks the tuple and passes both strings to
`persist_job_result`:

```rust
let (success, full_output, last_message) = execute_job_with_retry(...).await;
persist_job_result(config, job, success, &full_output, &last_message, ...).await
```

**`persist_job_result`** — gains `last_message: &str`. Storage (`record_run`) uses
`full_output` unchanged. Delivery passes `last_message`:

```rust
deliver_if_configured(config, job, &full_output, last_message).await
// record_run still receives full_output
```

For shell jobs the `last_message` parameter is passed as an empty string; shell jobs
never use it (the flag is only meaningful for agent jobs).

**`deliver_if_configured`** — selects the delivery string based on the flag:

```rust
async fn deliver_if_configured(
    config: &Config,
    job: &CronJob,
    output: &str,
    last_message: &str,
) -> Result<()> {
    let delivery = &job.delivery;
    if !delivery.mode.eq_ignore_ascii_case("announce") {
        return Ok(());
    }
    let text = if delivery.deliver_final_message_only && !last_message.is_empty() {
        last_message
    } else {
        output
    };
    let channel = /* ... */;
    let target = /* ... */;
    deliver_announcement(config, channel, target, text).await
}
```

**`daemon/mod.rs`** — two heartbeat callers of `crate::agent::run()` change mechanically:

```rust
// before:
Ok(response) => { ... use response ... }
// after:
Ok(output) => { let response = output.full_text; ... use response ... }
```

---

### 4. Testing

**Unit test — `last_message` vs `full_text` extraction:**

Construct `AgentRunOutput` values directly and assert:
- When `last_display_text` is non-empty, `last_message` equals only that final text.
- When `last_display_text` is empty (tool-only run), `last_message` equals `full_text`
  (fallback).

**Integration test — delivery filter in `persist_job_result`:**

Using the existing test harness in `scheduler.rs`:
1. Register a test delivery handler via `register_delivery_fn` that captures the
   delivered string into `Arc<Mutex<String>>`.
2. Create an agent job with `deliver_final_message_only = true`.
3. Call `persist_job_result` with `full_output` containing multi-paragraph narration and
   `last_message` containing only the final summary line.
4. Assert the captured delivery payload equals `last_message`.
5. Assert `cron_runs.output` (via `list_runs`) stores `full_output`.

Existing tests are unaffected: shell-job tests pass an empty `last_message`; the flag
defaults `false` so `deliver_if_configured` takes the `output` branch as before.

---

## Architecture impact summary

| File | Change |
|---|---|
| `crates/zeroclaw-runtime/src/cron/types.rs` | Add `deliver_final_message_only` to `DeliveryConfig` |
| `crates/zeroclaw-config/src/schema.rs` | Add `deliver_final_message_only` to `DeliveryConfigDecl` |
| `crates/zeroclaw-runtime/src/agent/loop_.rs` | Add `AgentRunOutput` struct; track `last_display_text`; change `run()` return type |
| `crates/zeroclaw-runtime/src/cron/scheduler.rs` | Thread `last_message` through call chain; filter in `deliver_if_configured` |
| `crates/zeroclaw-runtime/src/daemon/mod.rs` | Mechanical update: `output.full_text` in two heartbeat call sites |

---

## Risk and rollback

**Risk: low.** Opt-in flag, default `false`. All existing deployments unaffected.

**Rollback:** Remove `deliver_final_message_only = true` from a job's delivery config.
Behaviour reverts immediately on next run, no redeployment required.

**Edge cases covered:**
- Tool-only final turn → `last_message` falls back to `full_text`.
- All-tool-only run (no prose anywhere) → `last_display_text` is empty; fallback to
  `full_text` ensures something is always delivered.
- Long final message (> Telegram 4096 chars) → existing draft-splitting logic in the
  channel layer is unaffected; it operates on whatever string is delivered.

---

## Follow-on: future `AgentRunOutput` fields

The following fields are candidates for a follow-on commit once the struct exists:

| Field | Type | Purpose |
|---|---|---|
| `was_truncated` | `bool` | Whether max iterations was hit and a forced summary was used |
| `iteration_count` | `u32` | Tool-loop iterations consumed |
| `exit_reason` | `AgentExitReason` enum | `FinalResponse` / `MaxIterationsReached` / `Cancelled` / `ModelSwitch` |
| `tool_calls_executed` | `Vec<String>` | Tool names called — for auditing and observability |
| `model_used` | `String` | Actual model used (may differ after a mid-run model switch) |
| `total_cost_usd` | `Option<f64>` | Run-level cost estimate |
