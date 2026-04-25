use super::embeddings::EmbeddingProvider;
use super::traits::{ExportFilter, Memory, MemoryCategory, MemoryEntry, ProceduralMessage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::OnceCell;
use uuid::Uuid;

/// OpenBrain memory backend.
///
/// Stores memories in a Supabase-hosted PostgreSQL+pgvector instance
/// via the PostgREST API. ZeroClaw-specific fields (key, category,
/// session_id, namespace, importance) are packed into the `metadata`
/// JSONB column of the `thoughts` table under `zeroclaw_*` keys,
/// coexisting non-destructively with OpenBrain's own extracted metadata.
///
/// # Requirements
///
/// The embedding provider MUST be configured with `text-embedding-3-small`
/// at 1536 dimensions — the same model OpenBrain uses. The factory
/// (`zeroclaw_memory::create_memory`) enforces this and will refuse to
/// construct this backend with any other model.
pub struct OpenBrainMemory {
    client: reqwest::Client,
    /// Base Supabase URL, e.g. "https://xyz.supabase.co" (no trailing slash).
    base_url: String,
    service_role_key: String,
    match_threshold: f64,
    embedder: Arc<dyn EmbeddingProvider>,
    /// Lazy-init: verified on first operation.
    initialized: OnceCell<()>,
}

// ── Internal wire types ──────────────────────────────────────────────────────

/// The `metadata` JSONB object stored in every `thoughts` row.
/// OpenBrain's own extraction fields coexist with `zeroclaw_*` fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ThoughtMetadata {
    // ── OpenBrain native fields ──────────────────────────────────────────
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    thought_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topics: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,

    // ── ZeroClaw-specific fields ─────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    zeroclaw_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zeroclaw_category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zeroclaw_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zeroclaw_namespace: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zeroclaw_importance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zeroclaw_superseded_by: Option<String>,
}

/// A row as returned by `GET /rest/v1/thoughts`.
#[derive(Debug, Deserialize)]
struct ThoughtRow {
    id: String,
    content: String,
    #[serde(default)]
    metadata: ThoughtMetadata,
    created_at: String,
}

/// A scored row as returned by the `match_thoughts` RPC.
#[derive(Debug, Deserialize)]
struct MatchedThoughtRow {
    id: String,
    content: String,
    #[serde(default)]
    metadata: ThoughtMetadata,
    created_at: String,
    similarity: f64,
}


// ── Constructor & helpers ────────────────────────────────────────────────────

impl OpenBrainMemory {
    /// Create with lazy initialization. The Supabase connection is not
    /// tested until the first operation.
    pub fn new_lazy(
        url: &str,
        service_role_key: &str,
        match_threshold: f64,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        let base_url = url.trim_end_matches('/').to_string();
        let client = zeroclaw_config::schema::build_runtime_proxy_client("memory.openbrain");
        Self {
            client,
            base_url,
            service_role_key: service_role_key.to_string(),
            match_threshold,
            embedder,
            initialized: OnceCell::new(),
        }
    }

    async fn ensure_initialized(&self) -> Result<()> {
        self.initialized
            .get_or_try_init(|| async {
                let resp = self
                    .rest_get("/thoughts")
                    .query(&[("limit", "0")])
                    .send()
                    .await
                    .context("OpenBrain: connectivity check failed")?;
                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    anyhow::bail!(
                        "OpenBrain: Supabase PostgREST not reachable ({status}): {body}"
                    );
                }
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    /// Build an authenticated GET request against `/rest/v1{path}`.
    fn rest_get(&self, path: &str) -> reqwest::RequestBuilder {
        self.rest_request(reqwest::Method::GET, path)
    }

    /// Build an authenticated request against `/rest/v1{path}`.
    fn rest_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/rest/v1{}", self.base_url, path);
        self.client
            .request(method, &url)
            .header("Authorization", format!("Bearer {}", self.service_role_key))
            .header("apikey", &self.service_role_key)
            .header("Content-Type", "application/json")
    }

    /// Build an authenticated request against `/rest/v1/rpc{path}`.
    fn rpc_request(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/rest/v1/rpc{}", self.base_url, path);
        self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.service_role_key))
            .header("apikey", &self.service_role_key)
            .header("Content-Type", "application/json")
    }

    fn parse_category(s: &str) -> MemoryCategory {
        match s {
            "core" => MemoryCategory::Core,
            "daily" => MemoryCategory::Daily,
            "conversation" => MemoryCategory::Conversation,
            other => MemoryCategory::Custom(other.to_string()),
        }
    }

    fn row_to_entry(row: ThoughtRow, score: Option<f64>) -> MemoryEntry {
        let key = row
            .metadata
            .zeroclaw_key
            .clone()
            .unwrap_or_else(|| row.id.clone());
        let category = row
            .metadata
            .zeroclaw_category
            .as_deref()
            .map(Self::parse_category)
            .unwrap_or(MemoryCategory::Core);
        MemoryEntry {
            id: row.id,
            key,
            content: row.content,
            category,
            timestamp: row.created_at,
            session_id: row.metadata.zeroclaw_session_id,
            score,
            namespace: row
                .metadata
                .zeroclaw_namespace
                .unwrap_or_else(|| "default".to_string()),
            importance: row.metadata.zeroclaw_importance,
            superseded_by: row.metadata.zeroclaw_superseded_by,
        }
    }

    fn matched_row_to_entry(row: MatchedThoughtRow) -> MemoryEntry {
        Self::row_to_entry(
            ThoughtRow {
                id: row.id,
                content: row.content,
                metadata: row.metadata,
                created_at: row.created_at,
            },
            Some(row.similarity),
        )
    }

    /// Core store logic shared by `store()` and `store_with_metadata()`.
    async fn store_internal(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        namespace: Option<&str>,
        importance: Option<f64>,
    ) -> Result<()> {
        self.ensure_initialized().await?;

        let embedding = self.embedder.embed_one(content).await?;

        let thought_type = match &category {
            MemoryCategory::Core => "reference",
            MemoryCategory::Daily => "observation",
            MemoryCategory::Conversation => "observation",
            MemoryCategory::Custom(_) => "observation",
        };

        let metadata = ThoughtMetadata {
            thought_type: Some(thought_type.to_string()),
            topics: Some(vec![category.to_string()]),
            source: Some("zeroclaw".to_string()),
            zeroclaw_key: Some(key.to_string()),
            zeroclaw_category: Some(category.to_string()),
            zeroclaw_session_id: session_id.map(str::to_string),
            zeroclaw_namespace: Some(
                namespace.unwrap_or("default").to_string(),
            ),
            zeroclaw_importance: importance,
            zeroclaw_superseded_by: None,
        };

        // Check whether a row with this key already exists.
        let exists_resp = self
            .rest_get("/thoughts")
            .query(&[
                ("metadata->>zeroclaw_key", format!("eq.{key}")),
                ("select", "id".to_string()),
                ("limit", "1".to_string()),
            ])
            .send()
            .await
            .context("OpenBrain: existence check failed")?;

        if !exists_resp.status().is_success() {
            let status = exists_resp.status();
            let body = exists_resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenBrain: store (existence check) failed ({status}): {body}");
        }

        let existing: Vec<serde_json::Value> = exists_resp
            .json()
            .await
            .context("OpenBrain: existence check parse failed")?;

        let embedding_json: Vec<serde_json::Value> =
            embedding.iter().map(|f| serde_json::json!(f)).collect();

        if existing.is_empty() {
            // INSERT
            let body = serde_json::json!({
                "id": Uuid::new_v4().to_string(),
                "content": content,
                "embedding": embedding_json,
                "metadata": metadata,
            });
            let resp = self
                .rest_request(reqwest::Method::POST, "/thoughts")
                .header("Prefer", "return=minimal")
                .json(&body)
                .send()
                .await
                .context("OpenBrain: insert failed")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenBrain: store (insert) failed ({status}): {text}");
            }
        } else {
            // UPDATE
            let body = serde_json::json!({
                "content": content,
                "embedding": embedding_json,
                "metadata": metadata,
            });
            let resp = self
                .rest_request(reqwest::Method::PATCH, "/thoughts")
                .query(&[("metadata->>zeroclaw_key", format!("eq.{key}"))])
                .header("Prefer", "return=minimal")
                .json(&body)
                .send()
                .await
                .context("OpenBrain: update failed")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenBrain: store (update) failed ({status}): {text}");
            }
        }

        Ok(())
    }

    /// DELETE rows matching a PostgREST filter and return the count deleted.
    async fn delete_where(&self, filter_key: &str, filter_val: &str) -> Result<usize> {
        let resp = self
            .rest_request(reqwest::Method::DELETE, "/thoughts")
            .query(&[(filter_key, filter_val)])
            .header("Prefer", "return=representation")
            .send()
            .await
            .context("OpenBrain: delete failed")?;

        // 204 No Content — nothing deleted
        if resp.status() == reqwest::StatusCode::NO_CONTENT {
            return Ok(0);
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenBrain: delete failed ({status}): {text}");
        }

        let deleted: Vec<serde_json::Value> =
            resp.json().await.context("OpenBrain: delete parse failed")?;
        Ok(deleted.len())
    }
}

// ── Memory trait implementation ──────────────────────────────────────────────

#[async_trait]
impl Memory for OpenBrainMemory {
    fn name(&self) -> &str {
        "openbrain"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> Result<()> {
        self.store_internal(key, content, category, session_id, None, None)
            .await
    }

    async fn store_with_metadata(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        namespace: Option<&str>,
        importance: Option<f64>,
    ) -> Result<()> {
        self.store_internal(key, content, category, session_id, namespace, importance)
            .await
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let embedding = self.embedder.embed_one(query).await?;
        let embedding_json: Vec<serde_json::Value> =
            embedding.iter().map(|f| serde_json::json!(f)).collect();

        // Oversample to allow post-filtering by session / time bounds.
        let match_count = (limit * 3).max(20);

        let body = serde_json::json!({
            "query_embedding": embedding_json,
            "match_threshold": self.match_threshold,
            "match_count": match_count,
            "filter": {},
        });

        let resp = self
            .rpc_request("/match_thoughts")
            .json(&body)
            .send()
            .await
            .context("OpenBrain: recall failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenBrain: recall failed ({status}): {text}");
        }

        let rows: Vec<MatchedThoughtRow> =
            resp.json().await.context("OpenBrain: recall parse failed")?;

        let mut entries: Vec<MemoryEntry> = rows
            .into_iter()
            .map(Self::matched_row_to_entry)
            .collect();

        // Post-filter by session, time bounds.
        if let Some(sid) = session_id {
            entries.retain(|e| e.session_id.as_deref() == Some(sid));
        }
        if let Some(s) = since {
            entries.retain(|e| e.timestamp.as_str() >= s);
        }
        if let Some(u) = until {
            entries.retain(|e| e.timestamp.as_str() <= u);
        }

        entries.truncate(limit);
        Ok(entries)
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        self.ensure_initialized().await?;

        let resp = self
            .rest_get("/thoughts")
            .query(&[
                ("metadata->>zeroclaw_key", format!("eq.{key}")),
                ("limit", "1".to_string()),
            ])
            .send()
            .await
            .context("OpenBrain: get failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenBrain: get failed ({status}): {text}");
        }

        let mut rows: Vec<ThoughtRow> =
            resp.json().await.context("OpenBrain: get parse failed")?;
        Ok(rows.pop().map(|r| Self::row_to_entry(r, None)))
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let mut params: Vec<(&str, String)> = vec![
            ("order", "created_at.asc".to_string()),
            ("limit", "1000".to_string()),
        ];

        if let Some(cat) = category {
            params.push(("metadata->>zeroclaw_category", format!("eq.{cat}")));
        }
        if let Some(sid) = session_id {
            params.push(("metadata->>zeroclaw_session_id", format!("eq.{sid}")));
        }

        let resp = self
            .rest_get("/thoughts")
            .query(&params)
            .send()
            .await
            .context("OpenBrain: list failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenBrain: list failed ({status}): {text}");
        }

        let rows: Vec<ThoughtRow> =
            resp.json().await.context("OpenBrain: list parse failed")?;
        Ok(rows
            .into_iter()
            .map(|r| Self::row_to_entry(r, None))
            .collect())
    }

    async fn forget(&self, key: &str) -> Result<bool> {
        self.ensure_initialized().await?;
        let count = self
            .delete_where("metadata->>zeroclaw_key", &format!("eq.{key}"))
            .await?;
        Ok(count > 0)
    }

    async fn count(&self) -> Result<usize> {
        self.ensure_initialized().await?;

        let resp = self
            .rest_get("/thoughts")
            .query(&[("select", "id"), ("limit", "1")])
            .header("Prefer", "count=exact")
            .send()
            .await
            .context("OpenBrain: count failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenBrain: count failed ({status}): {text}");
        }

        // Parse total from Content-Range: 0-0/42
        let count = resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split('/').nth(1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        Ok(count)
    }

    async fn health_check(&self) -> bool {
        // Intentionally does NOT call ensure_initialized() to avoid side effects.
        let resp = self
            .rest_get("/thoughts")
            .query(&[("limit", "0")])
            .send()
            .await;
        resp.map(|r| r.status().is_success()).unwrap_or(false)
    }

    async fn purge_namespace(&self, namespace: &str) -> Result<usize> {
        self.ensure_initialized().await?;
        self.delete_where(
            "metadata->>zeroclaw_namespace",
            &format!("eq.{namespace}"),
        )
        .await
    }

    async fn purge_session(&self, session_id: &str) -> Result<usize> {
        self.ensure_initialized().await?;
        self.delete_where(
            "metadata->>zeroclaw_session_id",
            &format!("eq.{session_id}"),
        )
        .await
    }

    async fn recall_namespaced(
        &self,
        namespace: &str,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        let mut entries = self
            .recall(query, limit * 2, session_id, since, until)
            .await?;
        entries.retain(|e| e.namespace == namespace);
        entries.truncate(limit);
        Ok(entries)
    }

    async fn store_procedural(
        &self,
        _messages: &[ProceduralMessage],
        _session_id: Option<&str>,
    ) -> Result<()> {
        // No-op: OpenBrain does not have a procedural memory concept.
        Ok(())
    }

    async fn export(&self, filter: &ExportFilter) -> Result<Vec<MemoryEntry>> {
        let entries = self
            .list(filter.category.as_ref(), filter.session_id.as_deref())
            .await?;
        let filtered = entries
            .into_iter()
            .filter(|e| {
                if let Some(ref ns) = filter.namespace
                    && e.namespace != *ns
                {
                    return false;
                }
                if let Some(ref since) = filter.since
                    && e.timestamp.as_str() < since.as_str()
                {
                    return false;
                }
                if let Some(ref until) = filter.until
                    && e.timestamp.as_str() > until.as_str()
                {
                    return false;
                }
                true
            })
            .collect();
        Ok(filtered)
    }
}
