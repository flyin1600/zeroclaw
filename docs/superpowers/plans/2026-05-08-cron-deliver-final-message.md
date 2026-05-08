# Cron `deliver_final_message_only` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-job `deliver_final_message_only` flag that, when set, delivers only the final assistant message to the channel instead of the full accumulated reasoning transcript.

**Architecture:** The agent loop in `loop_.rs` already accumulates all display text into `accumulated_display_text`; we add a parallel `last_display_text` tracker that updates on every non-empty turn. `run()` return type changes from `Result<String>` to `Result<AgentRunOutput>` (a new struct with `full_text` and `last_message` fields). The scheduler threads `last_message` through to `deliver_if_configured`, which selects which string to send based on the per-job flag. Storage (`cron_runs.output`) always receives `full_text`.

**Tech Stack:** Rust, Tokio, Serde, rusqlite (existing stack — no new dependencies).

**Spec:** `docs/superpowers/specs/2026-05-08-cron-deliver-final-message-design.md`

---

## File Map

| File | Change |
|---|---|
| `crates/zeroclaw-runtime/src/cron/types.rs` | Add `deliver_final_message_only: bool` to `DeliveryConfig` |
| `crates/zeroclaw-config/src/schema.rs` | Add `deliver_final_message_only: bool` to `DeliveryConfigDecl` |
| `crates/zeroclaw-runtime/src/cron/store.rs` | Update `convert_delivery_decl` to map the new field |
| `crates/zeroclaw-runtime/src/agent/loop_.rs` | Add `AgentRunOutput` struct; add `last_display_text` tracker; change `run()` return type |
| `crates/zeroclaw-runtime/src/cron/scheduler.rs` | Thread `last_message` through call chain; add `select_delivery_text` helper; wire flag in `deliver_if_configured`; update `execute_job_now` |
| `crates/zeroclaw-runtime/src/daemon/mod.rs` | Mechanical: extract `.full_text` from `AgentRunOutput` at two heartbeat call sites |

---

## Task 1: Add `deliver_final_message_only` to `DeliveryConfig`

**Files:**
- Modify: `crates/zeroclaw-runtime/src/cron/types.rs:103-128`

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block at the bottom of `crates/zeroclaw-runtime/src/cron/types.rs` (after line 222):

```rust
#[test]
fn delivery_config_deserializes_deliver_final_message_only_true() {
    let json = serde_json::json!({
        "mode": "announce",
        "channel": "telegram",
        "to": "123456",
        "best_effort": false,
        "deliver_final_message_only": true
    });
    let config: DeliveryConfig = serde_json::from_value(json).unwrap();
    assert!(config.deliver_final_message_only);
}

#[test]
fn delivery_config_deliver_final_message_only_defaults_false() {
    let json = serde_json::json!({
        "mode": "announce",
        "channel": "telegram",
        "to": "123456"
    });
    let config: DeliveryConfig = serde_json::from_value(json).unwrap();
    assert!(!config.deliver_final_message_only);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p zeroclaw-runtime cron::types::tests::delivery_config_deserializes_deliver_final_message_only_true 2>&1 | tail -5
```

Expected: compile error — field `deliver_final_message_only` does not exist on `DeliveryConfig`.

- [ ] **Step 3: Add the field to `DeliveryConfig`**

In `crates/zeroclaw-runtime/src/cron/types.rs`, update the `DeliveryConfig` struct (lines 103-113):

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
    pub deliver_final_message_only: bool,
}
```

- [ ] **Step 4: Fix all `DeliveryConfig { ... }` struct literals that now need the new field**

Search for every `DeliveryConfig {` literal in the runtime crate and add `deliver_final_message_only: false` to each one:

```bash
grep -rn "DeliveryConfig {" crates/zeroclaw-runtime/src/ --include="*.rs"
```

Each site should look like:
```rust
DeliveryConfig {
    mode: "announce".into(),
    channel: Some("telegram".into()),
    to: Some("123456".into()),
    best_effort: true,
    deliver_final_message_only: false,
}
```

- [ ] **Step 5: Run the new tests and confirm they pass**

```bash
cargo test -p zeroclaw-runtime cron::types 2>&1 | tail -15
```

Expected: all tests in `cron::types` pass.

- [ ] **Step 6: Commit**

```bash
git add crates/zeroclaw-runtime/src/cron/types.rs
git commit -m "feat(cron): add deliver_final_message_only to DeliveryConfig"
```

---

## Task 2: Add `deliver_final_message_only` to `DeliveryConfigDecl` and `convert_delivery_decl`

**Files:**
- Modify: `crates/zeroclaw-config/src/schema.rs:6484-6497`
- Modify: `crates/zeroclaw-runtime/src/cron/store.rs:845-853`

- [ ] **Step 1: Write the failing test**

Add to the `mod tests` block in `crates/zeroclaw-runtime/src/cron/store.rs`:

```rust
#[test]
fn convert_delivery_decl_maps_deliver_final_message_only() {
    use zeroclaw_config::schema::DeliveryConfigDecl;
    let decl = DeliveryConfigDecl {
        mode: "announce".to_string(),
        channel: Some("telegram".to_string()),
        to: Some("chat-id".to_string()),
        best_effort: false,
        deliver_final_message_only: true,
    };
    // Access the private function through the public sync path:
    // We test via add_agent_job round-trip instead since convert_delivery_decl is private.
    let tmp = TempDir::new().unwrap();
    let config = test_config(&tmp);
    let job = add_agent_job(
        &config,
        Some("dfmo-test".into()),
        Schedule::Cron { expr: "*/5 * * * *".into(), tz: None },
        "summarize",
        SessionTarget::Isolated,
        None,
        Some(crate::cron::DeliveryConfig {
            mode: "announce".into(),
            channel: Some("telegram".into()),
            to: Some("chat-id".into()),
            best_effort: false,
            deliver_final_message_only: true,
        }),
        false,
        None,
    )
    .unwrap();
    assert!(job.delivery.deliver_final_message_only);
    // Verify it round-trips through SQLite serialization
    let stored = crate::cron::get_job(&config, &job.id).unwrap();
    assert!(stored.delivery.deliver_final_message_only);
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cargo test -p zeroclaw-runtime cron::store::tests::convert_delivery_decl_maps_deliver_final_message_only 2>&1 | tail -5
```

Expected: compile error — field `deliver_final_message_only` does not exist on `DeliveryConfigDecl`.

- [ ] **Step 3: Add the field to `DeliveryConfigDecl` in `schema.rs`**

In `crates/zeroclaw-config/src/schema.rs`, update `DeliveryConfigDecl` (lines 6484-6497):

```rust
pub struct DeliveryConfigDecl {
    /// Delivery mode: `"none"` or `"announce"`.
    #[serde(default = "default_delivery_mode")]
    pub mode: String,
    /// Channel name (e.g. `"telegram"`, `"discord"`).
    #[serde(default)]
    pub channel: Option<String>,
    /// Target/recipient identifier.
    #[serde(default)]
    pub to: Option<String>,
    /// Best-effort delivery. Default: `true`.
    #[serde(default = "default_true")]
    pub best_effort: bool,
    /// When `true` and `mode = "announce"`, deliver only the final assistant
    /// message instead of the full accumulated transcript. Default: `false`.
    #[serde(default)]
    pub deliver_final_message_only: bool,
}
```

- [ ] **Step 4: Update `convert_delivery_decl` in `store.rs`**

In `crates/zeroclaw-runtime/src/cron/store.rs`, update `convert_delivery_decl` (lines 846-853):

```rust
fn convert_delivery_decl(decl: &zeroclaw_config::schema::DeliveryConfigDecl) -> DeliveryConfig {
    DeliveryConfig {
        mode: decl.mode.clone(),
        channel: decl.channel.clone(),
        to: decl.to.clone(),
        best_effort: decl.best_effort,
        deliver_final_message_only: decl.deliver_final_message_only,
    }
}
```

- [ ] **Step 5: Run the test and confirm it passes**

```bash
cargo test -p zeroclaw-runtime cron::store::tests::convert_delivery_decl_maps_deliver_final_message_only 2>&1 | tail -10
```

Expected: test passes.

- [ ] **Step 6: Run the full runtime test suite to catch regressions**

```bash
cargo test -p zeroclaw-runtime 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/zeroclaw-config/src/schema.rs crates/zeroclaw-runtime/src/cron/store.rs
git commit -m "feat(cron): add deliver_final_message_only to DeliveryConfigDecl and convert_delivery_decl"
```

---

## Task 3: Add `AgentRunOutput` struct to `loop_.rs`

**Files:**
- Modify: `crates/zeroclaw-runtime/src/agent/loop_.rs` (near line 517)

- [ ] **Step 1: Write the unit tests**

Locate the `#[cfg(test)]` module near the bottom of `crates/zeroclaw-runtime/src/agent/loop_.rs` (search for `fn resolve_display_text_hides_raw_payload`). Add these tests inside the `mod tests` block:

```rust
#[test]
fn agent_run_output_fields_are_independent() {
    let output = super::AgentRunOutput {
        full_text: "narration\nfinal summary".to_string(),
        last_message: "final summary".to_string(),
    };
    assert_eq!(output.last_message, "final summary");
    assert!(output.full_text.starts_with("narration"));
}

#[test]
fn agent_run_output_default_is_empty() {
    let output = super::AgentRunOutput::default();
    assert!(output.full_text.is_empty());
    assert!(output.last_message.is_empty());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p zeroclaw-runtime agent::loop_::tests::agent_run_output_fields_are_independent 2>&1 | tail -5
```

Expected: compile error — `AgentRunOutput` not found.

- [ ] **Step 3: Add the `AgentRunOutput` struct**

In `crates/zeroclaw-runtime/src/agent/loop_.rs`, add after the `ToolLoopCancelled` struct (after line 518):

```rust
/// Output produced by a completed agent run.
#[derive(Debug, Default)]
pub struct AgentRunOutput {
    /// Full concatenated transcript across all iterations.
    /// Stored verbatim in `cron_runs.output` for operator debugging.
    pub full_text: String,
    /// Last non-empty display text produced during the run.
    /// Used for `deliver_final_message_only` announce-mode delivery.
    /// Falls back to `full_text` when no iteration produced non-empty text.
    pub last_message: String,
}
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cargo test -p zeroclaw-runtime agent::loop_::tests::agent_run_output 2>&1 | tail -10
```

Expected: both new tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/zeroclaw-runtime/src/agent/loop_.rs
git commit -m "feat(agent): add AgentRunOutput struct"
```

---

## Task 4: Change `run()` return type and update all callers (behavior-equivalent)

This task changes the return type of `run()` from `Result<String>` to `Result<AgentRunOutput>` without changing behavior yet — `last_message` is set to `String::new()` as a placeholder. All existing callers are updated to extract `.full_text`. The `last_display_text` tracker and `last_message` population happen in Task 5.

**Files:**
- Modify: `crates/zeroclaw-runtime/src/agent/loop_.rs:2092-2102`
- Modify: `crates/zeroclaw-runtime/src/cron/scheduler.rs:247-373`
- Modify: `crates/zeroclaw-runtime/src/daemon/mod.rs` (two sites ~lines 503, 626)

- [ ] **Step 1: Change `run()` return type in `loop_.rs`**

At line 2102, change `-> Result<String>` to `-> Result<AgentRunOutput>`:

```rust
pub async fn run(
    config: Config,
    message: Option<String>,
    provider_override: Option<String>,
    model_override: Option<String>,
    temperature: f64,
    peripheral_overrides: Vec<String>,
    interactive: bool,
    session_state_file: Option<PathBuf>,
    allowed_tools: Option<Vec<String>>,
) -> Result<AgentRunOutput> {
```

- [ ] **Step 2: Update the two `Ok(accumulated_display_text)` return sites in `loop_.rs`**

Site 1 — the normal final-response exit (line 1491). Change:
```rust
return Ok(accumulated_display_text);
```
to:
```rust
return Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message: String::new(),  // populated in Task 5
});
```

Site 2 — the max-iterations forced-summary exit (line 2045). Change:
```rust
Ok(accumulated_display_text)
```
to:
```rust
Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message: String::new(),  // populated in Task 5
})
```

- [ ] **Step 3: Update `run_agent_job` in `scheduler.rs` to return `(bool, String, String)`**

`run_agent_job` at line 247 currently returns `(bool, String)`. Change its signature and the match arm at lines 347-355:

```rust
async fn run_agent_job(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
) -> (bool, String, String) {
```

Update the `match run_result` block (lines 347-372):

```rust
match run_result {
    Ok(output) => (
        true,
        if output.full_text.trim().is_empty() {
            "agent job executed".to_string()
        } else {
            output.full_text
        },
        output.last_message,
    ),
    Err(e) => {
        let mem_session_key = format!("cli:{}", session_path.display());
        if let Ok(mem) = zeroclaw_memory::create_memory(
            &config.memory,
            &config.workspace_dir,
            config
                .providers
                .fallback_provider()
                .and_then(|e| e.api_key.as_deref()),
        ) {
            let _ = mem.purge_session(&mem_session_key).await;
        }
        (false, format!("agent job failed: {e}"), String::new())
    }
}
```

- [ ] **Step 4: Update `execute_job_with_retry` in `scheduler.rs` to return `(bool, String, String)`**

Change the function signature at line 143 and update the body:

```rust
async fn execute_job_with_retry(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
) -> (bool, String, String) {
    let mut last_output = String::new();
    let mut last_msg = String::new();
    let retries = config.reliability.scheduler_retries;
    let mut backoff_ms = config.reliability.provider_backoff_ms.max(200);

    for attempt in 0..=retries {
        let (success, output, msg) = match job.job_type {
            JobType::Shell => {
                let (s, o) = run_job_command(config, security, job).await;
                (s, o, String::new())
            }
            JobType::Agent => Box::pin(run_agent_job(config, security, job)).await,
        };
        last_output = output;
        last_msg = msg;

        if success {
            return (true, last_output, last_msg);
        }

        if last_output.starts_with("blocked by security policy:") {
            return (false, last_output, last_msg);
        }

        if attempt < retries {
            let jitter_ms = u64::from(Utc::now().timestamp_subsec_millis() % 250);
            time::sleep(Duration::from_millis(backoff_ms + jitter_ms)).await;
            backoff_ms = (backoff_ms.saturating_mul(2)).min(30_000);
        }
    }

    (false, last_output, last_msg)
}
```

- [ ] **Step 5: Update `execute_and_persist_job` in `scheduler.rs` to unpack the triple**

Change lines 232-244:

```rust
async fn execute_and_persist_job(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
    component: &str,
) -> (String, bool, String) {
    crate::health::mark_component_ok(component);
    warn_if_high_frequency_agent_job(job);

    let started_at = Utc::now();
    let (success, full_output, last_message) =
        Box::pin(execute_job_with_retry(config, security, job)).await;
    let finished_at = Utc::now();
    let success = Box::pin(persist_job_result(
        config,
        job,
        success,
        &full_output,
        &last_message,
        started_at,
        finished_at,
    ))
    .await;

    (job.id.clone(), success, full_output)
}
```

Note: `execute_and_persist_job` return type stays `(String, bool, String)` — it returns `full_output` for the SSE broadcast in `process_due_jobs`, unchanged.

- [ ] **Step 6: Update `persist_job_result` signature in `scheduler.rs`**

Change the function at line 375 to accept `last_message`:

```rust
async fn persist_job_result(
    config: &Config,
    job: &CronJob,
    mut success: bool,
    output: &str,
    last_message: &str,
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
) -> bool {
    let duration_ms = (finished_at - started_at).num_milliseconds();

    if let Err(e) = deliver_if_configured(config, job, output, last_message).await {
        if job.delivery.best_effort {
            tracing::warn!("Cron delivery failed (best_effort): {e}");
        } else {
            success = false;
            tracing::warn!("Cron delivery failed: {e}");
        }
    }

    let _ = record_run(
        config,
        &job.id,
        started_at,
        finished_at,
        if success { "ok" } else { "error" },
        Some(output),   // always full transcript
        duration_ms,
    );
    // ... rest unchanged
```

- [ ] **Step 7: Update existing direct callers of `persist_job_result` in test helpers**

Search for any test or helper that calls `persist_job_result` directly and add `""` as the new `last_message` argument:

```bash
grep -n "persist_job_result(" crates/zeroclaw-runtime/src/cron/scheduler.rs
```

Update each call site in the test module to pass `""` for `last_message`:

```rust
// example: persist_job_result(&config, &job, true, "ok", "", started, finished).await
```

- [ ] **Step 8: Update two daemon call sites in `daemon/mod.rs`**

Site 1 (~line 503) — phase1 heartbeat decision:

```rust
// before:
Ok(response) => {
    let indices = HeartbeatEngine::parse_decision_response(&response, tasks.len());
// after:
Ok(run_output) => {
    let response = run_output.full_text;
    let indices = HeartbeatEngine::parse_decision_response(&response, tasks.len());
```

Site 2 (~line 626) — phase2 heartbeat task execution:

```rust
// before:
Ok(output) => {
    crate::health::mark_component_ok("heartbeat");
    // ... uses output.as_str() and output.chars().count() and output.len()
// after:
Ok(run_output) => {
    let output = run_output.full_text;
    crate::health::mark_component_ok("heartbeat");
    // ... uses output.as_str() and output.chars().count() and output.len() — unchanged
```

- [ ] **Step 9: Update `execute_job_now` in `scheduler.rs` to keep its `(bool, String)` return type**

`execute_job_now` is a public function called by the cron_run tool (`crates/zeroclaw-runtime/src/tools/cron_run.rs:119`). It wraps `execute_job_with_retry` which now returns a triple. Keep the public signature unchanged by discarding `_last_message`:

```rust
pub async fn execute_job_now(config: &Config, job: &CronJob) -> (bool, String) {
    let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);
    let (success, full_output, _last_message) =
        Box::pin(execute_job_with_retry(config, &security, job)).await;
    (success, full_output)
}
```

`cron_run.rs` is not changed — it already calls `deliver_announcement` directly with the output string and is unaffected.

- [ ] **Step 10: Cargo check to confirm it compiles**

```bash
cargo check -p zeroclaw-runtime 2>&1 | tail -20
```

Expected: no errors.

- [ ] **Step 11: Run the full test suite to confirm no regressions**

```bash
cargo test -p zeroclaw-runtime 2>&1 | tail -20
```

Expected: all tests pass. Behavior is identical to before — `last_message` is `String::new()` everywhere.

- [ ] **Step 12: Commit**

```bash
git add crates/zeroclaw-runtime/src/agent/loop_.rs \
        crates/zeroclaw-runtime/src/cron/scheduler.rs \
        crates/zeroclaw-runtime/src/daemon/mod.rs
git commit -m "refactor(agent): change run() return type to AgentRunOutput struct"
```

---

## Task 5: Add `last_display_text` tracker and wire the delivery filter

This task adds the actual behavior: tracking the last non-empty display text across iterations and using it in `deliver_if_configured` when the flag is set.

**Files:**
- Modify: `crates/zeroclaw-runtime/src/agent/loop_.rs`
- Modify: `crates/zeroclaw-runtime/src/cron/scheduler.rs`

- [ ] **Step 1: Write the failing unit tests for `select_delivery_text`**

Add these tests to the `mod tests` block in `crates/zeroclaw-runtime/src/cron/scheduler.rs`:

```rust
#[test]
fn select_delivery_text_returns_last_message_when_flag_enabled() {
    let delivery = DeliveryConfig {
        mode: "announce".into(),
        channel: Some("telegram".into()),
        to: Some("123".into()),
        best_effort: true,
        deliver_final_message_only: true,
    };
    let result = super::select_delivery_text(
        &delivery,
        "Let me check disk usage.\n\nDisk at 73%. Now search maintenance.",
        "🌅 Morning Status: all clear.",
    );
    assert_eq!(result, "🌅 Morning Status: all clear.");
}

#[test]
fn select_delivery_text_returns_full_output_when_flag_disabled() {
    let delivery = DeliveryConfig {
        mode: "announce".into(),
        channel: Some("telegram".into()),
        to: Some("123".into()),
        best_effort: true,
        deliver_final_message_only: false,
    };
    let full = "Let me check.\n\nFinal summary.";
    let result = super::select_delivery_text(&delivery, full, "Final summary.");
    assert_eq!(result, full);
}

#[test]
fn select_delivery_text_falls_back_to_full_when_last_message_empty() {
    let delivery = DeliveryConfig {
        mode: "announce".into(),
        channel: Some("telegram".into()),
        to: Some("123".into()),
        best_effort: true,
        deliver_final_message_only: true,
    };
    let full = "Tool-only run produced this.";
    let result = super::select_delivery_text(&delivery, full, "");
    assert_eq!(result, full);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p zeroclaw-runtime cron::scheduler::tests::select_delivery_text 2>&1 | tail -5
```

Expected: compile error — function `select_delivery_text` not found.

- [ ] **Step 3: Add `select_delivery_text` helper and update `deliver_if_configured` in `scheduler.rs`**

Add this private function just before `deliver_if_configured` (before line 471):

```rust
fn select_delivery_text<'a>(
    delivery: &DeliveryConfig,
    output: &'a str,
    last_message: &'a str,
) -> &'a str {
    if delivery.deliver_final_message_only && !last_message.is_empty() {
        last_message
    } else {
        output
    }
}
```

Update `deliver_if_configured` to accept and use `last_message`:

```rust
async fn deliver_if_configured(
    config: &Config,
    job: &CronJob,
    output: &str,
    last_message: &str,
) -> Result<()> {
    let delivery: &DeliveryConfig = &job.delivery;
    if !delivery.mode.eq_ignore_ascii_case("announce") {
        return Ok(());
    }

    let channel = delivery
        .channel
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("delivery.channel is required for announce mode"))?;
    let target = delivery
        .to
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("delivery.to is required for announce mode"))?;

    let text = select_delivery_text(delivery, output, last_message);
    deliver_announcement(config, channel, target, text).await
}
```

- [ ] **Step 4: Update the existing `deliver_if_configured_handles_none_mode` test**

The existing test at the bottom of `scheduler.rs` calls `deliver_if_configured` with the old 3-arg signature. Update it to pass the new `last_message` argument:

```rust
#[tokio::test]
async fn deliver_if_configured_handles_none_mode() {
    let tmp = TempDir::new().unwrap();
    let config = test_config(&tmp).await;
    let job = test_job("echo ok");

    // Default delivery mode is not "announce", so should be a no-op.
    assert!(deliver_if_configured(&config, &job, "x", "").await.is_ok());
}
```

- [ ] **Step 5: Run all unit tests and confirm they pass**

```bash
cargo test -p zeroclaw-runtime cron::scheduler::tests::select_delivery_text 2>&1 | tail -10
cargo test -p zeroclaw-runtime cron::scheduler::tests::deliver_if_configured 2>&1 | tail -10
```

Expected: all four tests (three new + one updated) pass.

- [ ] **Step 6: Add `last_display_text` tracker to the iteration loop in `loop_.rs`**

In `crates/zeroclaw-runtime/src/agent/loop_.rs`, after line 904 where `accumulated_display_text` is declared, add:

```rust
let mut accumulated_display_text = String::new();
let mut last_display_text = String::new();
```

- [ ] **Step 7: Update all four push sites to also track `last_display_text`**

**Site 1** — final-response path, around line 1459-1460. Replace:
```rust
// No tool calls — this is the final response.
accumulated_display_text.push_str(&display_text);
```
with:
```rust
// No tool calls — this is the final response.
if !display_text.is_empty() {
    last_display_text = display_text.clone();
}
accumulated_display_text.push_str(&display_text);
```

**Site 2** — intermediate tool-call path, around line 1494-1495. Replace:
```rust
// Accumulate text from this iteration (tool calls present, loop continues).
accumulated_display_text.push_str(&display_text);
```
with:
```rust
// Accumulate text from this iteration (tool calls present, loop continues).
if !display_text.is_empty() {
    last_display_text = display_text.clone();
}
accumulated_display_text.push_str(&display_text);
```

**Site 3** — max-iterations forced-summary path, around line 2040-2044. Replace:
```rust
accumulated_display_text.push_str(&text);
Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message: String::new(),
})
```
with:
```rust
if !text.is_empty() {
    last_display_text = text.clone();
}
accumulated_display_text.push_str(&text);
Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message: if last_display_text.is_empty() {
        accumulated_display_text.clone()
    } else {
        last_display_text
    },
})
```

Note: `accumulated_display_text` is moved into the struct, so the `clone()` in the fallback must happen on the version before the move. Rewrite as:

```rust
if !text.is_empty() {
    last_display_text = text.clone();
}
accumulated_display_text.push_str(&text);
let last_message = if last_display_text.is_empty() {
    accumulated_display_text.clone()
} else {
    last_display_text
};
Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message,
})
```

- [ ] **Step 8: Update the final-response return site in `loop_.rs`**

At the normal exit site (was line 1491), replace:
```rust
return Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message: String::new(),
});
```
with:
```rust
let last_message = if last_display_text.is_empty() {
    accumulated_display_text.clone()
} else {
    last_display_text.clone()
};
return Ok(AgentRunOutput {
    full_text: accumulated_display_text,
    last_message,
});
```

- [ ] **Step 9: Cargo check**

```bash
cargo check -p zeroclaw-runtime 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 10: Run the full test suite**

```bash
cargo test -p zeroclaw-runtime 2>&1 | tail -20
```

Expected: all tests pass, including the three new `select_delivery_text` tests.

- [ ] **Step 11: Commit**

```bash
git add crates/zeroclaw-runtime/src/agent/loop_.rs \
        crates/zeroclaw-runtime/src/cron/scheduler.rs
git commit -m "feat(cron): deliver_final_message_only — track last_display_text and wire delivery filter"
```

---

## Task 6: Final validation

- [ ] **Step 1: Run the full workspace test suite**

```bash
cargo test 2>&1 | tail -30
```

Expected: all tests pass across all crates.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets -- -D warnings 2>&1 | tail -20
```

Expected: no warnings.

- [ ] **Step 3: Run fmt check**

```bash
cargo fmt --all -- --check 2>&1 | tail -10
```

Expected: no formatting issues. If there are any, run `cargo fmt --all` and re-run the check.

- [ ] **Step 4: Commit fmt fixes if needed**

```bash
git add -p
git commit -m "style: cargo fmt"
```

---

## Summary

Five tasks, six commits. The change is entirely opt-in (`deliver_final_message_only` defaults `false`) and the only behavioral change for existing users is none. The `AgentRunOutput` struct is the foundation for future observability fields (`was_truncated`, `iteration_count`, `exit_reason`) tracked in the spec's follow-on section.
