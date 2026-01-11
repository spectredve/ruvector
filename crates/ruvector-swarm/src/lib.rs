//! # Ruvector Swarm
//!
//! Agentic AI Self-Learning Swarm with distributed coordination and federated learning.
//!
//! ## Overview
//!
//! This crate implements a distributed swarm of AI agents that can:
//! - Coordinate through P2P communication
//! - Learn continuously from interactions
//! - Share knowledge through federated learning
//! - Run on edge devices with zero cloud dependency
//!
//! ## Core Types
//!
//! - [`AgentId`]: Unique cryptographic identifier for agents
//! - [`Query`]: Incoming query with embedding and metadata
//! - [`Response`]: Agent response with confidence score
//! - [`Feedback`]: User feedback for learning

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod identity;
pub mod registry;
pub mod types;

#[cfg(test)]
mod identity_props;

// Re-exports
pub use error::{Result, SwarmError};
pub use identity::{AgentIdentity, AgentIdentityInfo};
pub use registry::{AgentInfo, AgentRegistry, AgentStatus, CapabilityMatch};
pub use types::{AgentId, Feedback, Query, QueryType, Response};

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty(), "Version should not be empty");
    }
}
