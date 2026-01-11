//! Core types for the swarm system.
//!
//! This module defines the fundamental data structures used throughout the swarm:
//! - [`AgentId`]: Unique cryptographic identifier for agents
//! - [`Query`]: Incoming query with embedding and metadata
//! - [`Response`]: Agent response with confidence score
//! - [`Feedback`]: User feedback for learning

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for an agent, derived from Ed25519 public key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    /// Create a new random AgentId.
    ///
    /// In production, this should be derived from the Ed25519 public key.
    /// This method generates a UUID-based ID for testing purposes.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create an AgentId from a public key bytes.
    ///
    /// The ID is the hex-encoded first 16 bytes of the public key.
    pub fn from_public_key(public_key: &[u8]) -> Self {
        let hex = public_key
            .iter()
            .take(16)
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        Self(hex)
    }

    /// Get the string representation of the AgentId.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}


/// Unique identifier for a query.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(pub Uuid);

impl QueryId {
    /// Create a new random QueryId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for QueryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a response.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResponseId(pub Uuid);

impl ResponseId {
    /// Create a new random ResponseId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ResponseId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ResponseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of query being processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryType {
    /// Retrieve information from knowledge base.
    Knowledge,
    /// How to perform a procedure.
    Procedure,
    /// Diagnose and troubleshoot a problem.
    Troubleshoot,
    /// Configure system parameters.
    Configure,
    /// Explain concepts.
    Explain,
}

impl Default for QueryType {
    fn default() -> Self {
        Self::Knowledge
    }
}

/// Context information for a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    /// Previous queries in the conversation.
    pub history: Vec<String>,
    /// User preferences or settings.
    pub preferences: Option<serde_json::Value>,
    /// Session identifier.
    pub session_id: Option<String>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            preferences: None,
            session_id: None,
        }
    }
}


/// A query submitted to the swarm for processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Unique identifier for this query.
    pub id: QueryId,
    /// The query content/text.
    pub content: String,
    /// Vector embedding of the query (128 dimensions).
    pub embedding: Vec<f32>,
    /// Type of query.
    pub query_type: QueryType,
    /// Optional context information.
    pub context: Option<Context>,
    /// Timestamp when query was created.
    pub timestamp: DateTime<Utc>,
}

impl Query {
    /// Create a new query with the given content.
    ///
    /// The embedding is initialized to zeros and should be computed
    /// by an embedding provider before routing.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: QueryId::new(),
            content: content.into(),
            embedding: vec![0.0; 128],
            query_type: QueryType::default(),
            context: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a new query with a specific type.
    pub fn with_type(content: impl Into<String>, query_type: QueryType) -> Self {
        Self {
            id: QueryId::new(),
            content: content.into(),
            embedding: vec![0.0; 128],
            query_type,
            context: None,
            timestamp: Utc::now(),
        }
    }

    /// Set the embedding for this query.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Set the context for this query.
    pub fn with_context(mut self, context: Context) -> Self {
        self.context = Some(context);
        self
    }
}

/// Source information for a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Identifier of the source document/entry.
    pub id: String,
    /// Title or name of the source.
    pub title: Option<String>,
    /// Relevance score (0.0 to 1.0).
    pub relevance: f32,
}

/// A response generated by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Unique identifier for this response.
    pub id: ResponseId,
    /// The query this response is for.
    pub query_id: QueryId,
    /// The response content.
    pub content: String,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Sources used to generate the response.
    pub sources: Vec<Source>,
    /// Suggested follow-up actions.
    pub suggested_actions: Vec<String>,
    /// Time taken to generate response in milliseconds.
    pub latency_ms: u32,
    /// Timestamp when response was created.
    pub timestamp: DateTime<Utc>,
}

impl Response {
    /// Create a new response.
    pub fn new(query_id: QueryId, content: impl Into<String>, confidence: f32) -> Self {
        Self {
            id: ResponseId::new(),
            query_id,
            content: content.into(),
            confidence: confidence.clamp(0.0, 1.0),
            sources: Vec::new(),
            suggested_actions: Vec::new(),
            latency_ms: 0,
            timestamp: Utc::now(),
        }
    }

    /// Set the latency for this response.
    pub fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Add a source to this response.
    pub fn with_source(mut self, source: Source) -> Self {
        self.sources.push(source);
        self
    }

    /// Add sources to this response.
    pub fn with_sources(mut self, sources: Vec<Source>) -> Self {
        self.sources = sources;
        self
    }
}


/// User feedback for a response, used for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback {
    /// The response this feedback is for.
    pub response_id: ResponseId,
    /// User rating (-1.0 to 1.0).
    pub rating: f32,
    /// Whether the query was resolved.
    pub resolution: bool,
    /// Optional user comments.
    pub comments: Option<String>,
    /// Timestamp when feedback was provided.
    pub timestamp: DateTime<Utc>,
}

impl Feedback {
    /// Create new feedback for a response.
    pub fn new(response_id: ResponseId, rating: f32, resolution: bool) -> Self {
        Self {
            response_id,
            rating: rating.clamp(-1.0, 1.0),
            resolution,
            comments: None,
            timestamp: Utc::now(),
        }
    }

    /// Add comments to this feedback.
    pub fn with_comments(mut self, comments: impl Into<String>) -> Self {
        self.comments = Some(comments.into());
        self
    }
}

/// Timestamp type alias for consistency.
pub type Timestamp = DateTime<Utc>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_creation() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2, "Agent IDs should be unique");
    }

    #[test]
    fn test_agent_id_from_public_key() {
        let key_bytes = [1u8; 32];
        let id = AgentId::from_public_key(&key_bytes);
        assert_eq!(id.as_str().len(), 32, "Hex ID should be 32 chars (16 bytes)");
    }

    #[test]
    fn test_query_creation() {
        let query = Query::new("Test query");
        assert_eq!(query.content, "Test query");
        assert_eq!(query.embedding.len(), 128);
        assert_eq!(query.query_type, QueryType::Knowledge);
    }

    #[test]
    fn test_query_with_type() {
        let query = Query::with_type("How to configure?", QueryType::Configure);
        assert_eq!(query.query_type, QueryType::Configure);
    }

    #[test]
    fn test_response_creation() {
        let query = Query::new("Test");
        let response = Response::new(query.id.clone(), "Answer", 0.95);
        assert_eq!(response.query_id, query.id);
        assert_eq!(response.confidence, 0.95);
    }

    #[test]
    fn test_response_confidence_clamping() {
        let query = Query::new("Test");
        let response = Response::new(query.id, "Answer", 1.5);
        assert_eq!(response.confidence, 1.0, "Confidence should be clamped to 1.0");
    }

    #[test]
    fn test_feedback_creation() {
        let response_id = ResponseId::new();
        let feedback = Feedback::new(response_id.clone(), 0.8, true);
        assert_eq!(feedback.response_id, response_id);
        assert_eq!(feedback.rating, 0.8);
        assert!(feedback.resolution);
    }

    #[test]
    fn test_feedback_rating_clamping() {
        let response_id = ResponseId::new();
        let feedback = Feedback::new(response_id, -2.0, false);
        assert_eq!(feedback.rating, -1.0, "Rating should be clamped to -1.0");
    }

    #[test]
    fn test_feedback_with_comments() {
        let response_id = ResponseId::new();
        let feedback = Feedback::new(response_id, 0.5, true)
            .with_comments("Great response!");
        assert_eq!(feedback.comments, Some("Great response!".to_string()));
    }
}
