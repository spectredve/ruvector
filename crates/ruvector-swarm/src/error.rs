//! Error types for the swarm system.

use thiserror::Error;

/// Result type alias for swarm operations.
pub type Result<T> = std::result::Result<T, SwarmError>;

/// Errors that can occur in the swarm system.
#[derive(Error, Debug)]
pub enum SwarmError {
    /// Error during Ed25519 keypair generation.
    #[error("Failed to generate identity keypair: {0}")]
    IdentityGenerationError(String),

    /// Agent ID already exists in the registry.
    #[error("Duplicate agent identity: {0}")]
    DuplicateIdentityError(String),

    /// Message signature verification failed.
    #[error("Signature verification failed: {0}")]
    SignatureVerificationError(String),

    /// P2P connection error.
    #[error("Peer connection failed: {0}")]
    PeerConnectionError(String),

    /// Vector search error.
    #[error("Vector search failed: {0}")]
    VectorSearchError(String),

    /// Database persistence error.
    #[error("Persistence failed: {0}")]
    PersistenceError(String),

    /// Data deserialization error.
    #[error("Deserialization failed: {0}")]
    DeserializationError(String),

    /// Memory limit exceeded.
    #[error("Memory limit exceeded: {0}")]
    MemoryLimitError(String),

    /// Invalid reward value.
    #[error("Invalid reward value: {0}")]
    InvalidRewardError(String),

    /// Trajectory data corruption.
    #[error("Trajectory corrupted: {0}")]
    TrajectoryCorruptionError(String),

    /// Pattern merge conflict.
    #[error("Merge conflict: {0}")]
    MergeConflictError(String),

    /// Compression error.
    #[error("Compression failed: {0}")]
    CompressionError(String),

    /// No suitable agent found for routing.
    #[error("Routing failed: {0}")]
    RoutingError(String),

    /// Operation timed out.
    #[error("Operation timed out: {0}")]
    TimeoutError(String),

    /// Gossip protocol error.
    #[error("Gossip protocol error: {0}")]
    GossipError(String),

    /// Encryption/decryption error.
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid query.
    #[error("Invalid query: {0}")]
    InvalidQueryError(String),
}
