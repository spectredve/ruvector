//! Agent Registry for managing agent identities and swarm membership.
//!
//! This module provides:
//! - Agent registration with duplicate detection
//! - Identity verification via Ed25519 signatures
//! - Capability-based agent discovery using HNSW index

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{Result, SwarmError};
use crate::identity::{AgentIdentity, AgentIdentityInfo};
use crate::types::AgentId;

/// Status of an agent in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is online and available.
    Online,
    /// Agent is offline.
    Offline,
    /// Agent is busy processing.
    Busy,
    /// Agent is in maintenance mode.
    Maintenance,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Online
    }
}

/// Information about a registered agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Unique agent identifier.
    pub agent_id: AgentId,
    /// Ed25519 public key bytes (32 bytes).
    pub public_key: [u8; 32],
    /// 128-dimension capability vector.
    #[serde(with = "capability_vector_serde")]
    pub capability_vector: [f32; 128],
    /// Last time the agent was seen.
    pub last_seen: DateTime<Utc>,
    /// Current status of the agent.
    pub status: AgentStatus,
}

/// Custom serde module for [f32; 128] arrays
mod capability_vector_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(data: &[f32; 128], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        data.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; 128], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<f32> = Vec::deserialize(deserializer)?;
        if vec.len() != 128 {
            return Err(serde::de::Error::custom(format!(
                "expected 128 elements, got {}",
                vec.len()
            )));
        }
        let mut arr = [0.0f32; 128];
        arr.copy_from_slice(&vec);
        Ok(arr)
    }
}

impl AgentInfo {
    /// Create a new AgentInfo from an AgentIdentity.
    pub fn from_identity(identity: &AgentIdentity) -> Self {
        Self {
            agent_id: identity.agent_id().clone(),
            public_key: *identity.public_key_bytes(),
            capability_vector: *identity.capability_vector(),
            last_seen: Utc::now(),
            status: AgentStatus::Online,
        }
    }

    /// Get the verifying key for signature verification.
    pub fn verifying_key(&self) -> Result<VerifyingKey> {
        VerifyingKey::from_bytes(&self.public_key)
            .map_err(|e| SwarmError::SignatureVerificationError(e.to_string()))
    }
}

impl From<&AgentIdentityInfo> for AgentInfo {
    fn from(info: &AgentIdentityInfo) -> Self {
        Self {
            agent_id: info.agent_id.clone(),
            public_key: info.public_key,
            capability_vector: info.capability_vector,
            last_seen: Utc::now(),
            status: AgentStatus::Online,
        }
    }
}

/// Result of a capability search.
#[derive(Debug, Clone)]
pub struct CapabilityMatch {
    /// The matching agent info.
    pub agent_info: AgentInfo,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
}

/// Cosine distance function for HNSW.
#[cfg(feature = "native")]
struct CosineDistance;

#[cfg(feature = "native")]
impl hnsw_rs::prelude::Distance<f32> for CosineDistance {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0 // Maximum distance for zero vectors
        } else {
            1.0 - (dot / (norm_a * norm_b)) // Cosine distance = 1 - cosine similarity
        }
    }
}

/// Agent Registry for managing swarm membership.
///
/// The registry maintains:
/// - A HashMap of registered agents by AgentId
/// - An HNSW index for capability-based search (native only)
pub struct AgentRegistry {
    /// Registered agents by ID.
    agents: HashMap<AgentId, AgentInfo>,
    /// HNSW index for capability search (native only).
    #[cfg(feature = "native")]
    capability_index: Option<CapabilityIndex>,
}

/// HNSW-based capability index for semantic agent discovery.
#[cfg(feature = "native")]
struct CapabilityIndex {
    hnsw: hnsw_rs::prelude::Hnsw<'static, f32, CosineDistance>,
    /// Mapping from HNSW data ID to AgentId.
    id_map: Vec<AgentId>,
}

#[cfg(feature = "native")]
impl CapabilityIndex {
    /// Create a new capability index.
    fn new() -> Self {
        // HNSW parameters: max_nb_connection=16, ef_construction=200
        let hnsw = hnsw_rs::prelude::Hnsw::new(
            16,   // max_nb_connection
            1000, // max_elements (initial capacity)
            128,  // dimensions
            200,  // ef_construction
            CosineDistance,
        );
        Self {
            hnsw,
            id_map: Vec::new(),
        }
    }

    /// Insert a capability vector for an agent.
    fn insert(&mut self, agent_id: &AgentId, capability_vector: &[f32; 128]) {
        let data_id = self.id_map.len();
        self.id_map.push(agent_id.clone());
        self.hnsw.insert((&capability_vector[..], data_id));
    }

    /// Search for agents with similar capabilities.
    fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(AgentId, f32)> {
        let results = self.hnsw.search(query, k, ef_search);
        results
            .into_iter()
            .map(|neighbor| {
                let agent_id = self.id_map[neighbor.d_id].clone();
                // Convert distance to similarity (cosine distance to similarity)
                let similarity = 1.0 - neighbor.distance;
                (agent_id, similarity)
            })
            .collect()
    }
}

impl AgentRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            #[cfg(feature = "native")]
            capability_index: Some(CapabilityIndex::new()),
        }
    }

    /// Register an agent in the registry.
    ///
    /// Returns an error if an agent with the same ID is already registered.
    ///
    /// # Requirements
    /// - 1.2: Verify agent identity before accepting
    /// - 1.4: Reject duplicate identities
    pub fn register(&mut self, identity: &AgentIdentity) -> Result<()> {
        let agent_id = identity.agent_id();

        // Check for duplicate identity
        if self.agents.contains_key(agent_id) {
            warn!(agent_id = %agent_id, "Duplicate agent registration attempt");
            return Err(SwarmError::DuplicateIdentityError(agent_id.to_string()));
        }

        let agent_info = AgentInfo::from_identity(identity);

        // Add to HNSW index (native only)
        #[cfg(feature = "native")]
        if let Some(ref mut index) = self.capability_index {
            index.insert(agent_id, identity.capability_vector());
        }

        self.agents.insert(agent_id.clone(), agent_info);
        info!(agent_id = %agent_id, "Agent registered successfully");

        Ok(())
    }

    /// Register an agent from AgentIdentityInfo (for remote agents).
    pub fn register_info(&mut self, info: &AgentIdentityInfo) -> Result<()> {
        // Check for duplicate identity
        if self.agents.contains_key(&info.agent_id) {
            warn!(agent_id = %info.agent_id, "Duplicate agent registration attempt");
            return Err(SwarmError::DuplicateIdentityError(info.agent_id.to_string()));
        }

        let agent_info = AgentInfo::from(info);

        // Add to HNSW index (native only)
        #[cfg(feature = "native")]
        if let Some(ref mut index) = self.capability_index {
            index.insert(&info.agent_id, &info.capability_vector);
        }

        self.agents.insert(info.agent_id.clone(), agent_info);
        info!(agent_id = %info.agent_id, "Agent registered successfully");

        Ok(())
    }

    /// Verify an agent's identity by checking a signature.
    ///
    /// # Requirements
    /// - 1.2: Verify agent identity signature before accepting connection
    pub fn verify_identity(
        &self,
        agent_id: &AgentId,
        message: &[u8],
        signature: &Signature,
    ) -> Result<bool> {
        let agent_info = self.agents.get(agent_id).ok_or_else(|| {
            SwarmError::SignatureVerificationError(format!("Agent not found: {}", agent_id))
        })?;

        let verifying_key = agent_info.verifying_key()?;

        match verifying_key.verify_strict(message, signature) {
            Ok(()) => Ok(true),
            Err(e) => {
                warn!(
                    agent_id = %agent_id,
                    error = %e,
                    "Signature verification failed"
                );
                Ok(false)
            }
        }
    }

    /// Find agents by capability similarity using HNSW index.
    ///
    /// Returns the k most similar agents based on capability vector similarity.
    ///
    /// # Requirements
    /// - 1.2: Find agents by capability for routing
    #[cfg(feature = "native")]
    pub fn find_by_capability(&self, query_embedding: &[f32], k: usize) -> Vec<CapabilityMatch> {
        if query_embedding.len() != 128 {
            warn!(
                "Query embedding has wrong dimension: {} (expected 128)",
                query_embedding.len()
            );
            return Vec::new();
        }

        let Some(ref index) = self.capability_index else {
            return Vec::new();
        };

        let ef_search = k.max(50); // ef_search should be >= k
        let results = index.search(query_embedding, k, ef_search);

        results
            .into_iter()
            .filter_map(|(agent_id, similarity)| {
                self.agents.get(&agent_id).map(|info| CapabilityMatch {
                    agent_info: info.clone(),
                    similarity,
                })
            })
            .collect()
    }

    /// Find agents by capability (WASM fallback - linear search).
    #[cfg(not(feature = "native"))]
    pub fn find_by_capability(&self, query_embedding: &[f32], k: usize) -> Vec<CapabilityMatch> {
        if query_embedding.len() != 128 {
            return Vec::new();
        }

        let mut matches: Vec<_> = self
            .agents
            .values()
            .map(|info| {
                let similarity = cosine_similarity(query_embedding, &info.capability_vector);
                CapabilityMatch {
                    agent_info: info.clone(),
                    similarity,
                }
            })
            .collect();

        // Sort by similarity descending
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);
        matches
    }

    /// Remove an agent from the registry.
    pub fn remove(&mut self, agent_id: &AgentId) -> Result<()> {
        if self.agents.remove(agent_id).is_some() {
            info!(agent_id = %agent_id, "Agent removed from registry");
            // Note: HNSW doesn't support removal, so we just remove from HashMap
            // The HNSW entry will be orphaned but won't match any agent
            Ok(())
        } else {
            Err(SwarmError::SignatureVerificationError(format!(
                "Agent not found: {}",
                agent_id
            )))
        }
    }

    /// Get an agent's info by ID.
    pub fn get(&self, agent_id: &AgentId) -> Option<&AgentInfo> {
        self.agents.get(agent_id)
    }

    /// Get the number of registered agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Update an agent's status.
    pub fn update_status(&mut self, agent_id: &AgentId, status: AgentStatus) -> Result<()> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| {
            SwarmError::SignatureVerificationError(format!("Agent not found: {}", agent_id))
        })?;
        agent.status = status;
        agent.last_seen = Utc::now();
        Ok(())
    }

    /// Update an agent's last seen timestamp.
    pub fn touch(&mut self, agent_id: &AgentId) -> Result<()> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| {
            SwarmError::SignatureVerificationError(format!("Agent not found: {}", agent_id))
        })?;
        agent.last_seen = Utc::now();
        Ok(())
    }

    /// Get all registered agent IDs.
    pub fn agent_ids(&self) -> Vec<AgentId> {
        self.agents.keys().cloned().collect()
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors.
#[cfg(not(feature = "native"))]
fn cosine_similarity(a: &[f32], b: &[f32; 128]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AgentRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_agent_registration() {
        let mut registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");

        assert!(registry.register(&identity).is_ok());
        assert_eq!(registry.len(), 1);
        assert!(registry.get(identity.agent_id()).is_some());
    }

    #[test]
    fn test_duplicate_registration_rejected() {
        let mut registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");

        assert!(registry.register(&identity).is_ok());

        // Second registration should fail
        let result = registry.register(&identity);
        assert!(result.is_err());

        if let Err(SwarmError::DuplicateIdentityError(id)) = result {
            assert_eq!(id, identity.agent_id().to_string());
        } else {
            panic!("Expected DuplicateIdentityError");
        }
    }

    #[test]
    fn test_identity_verification() {
        let mut registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");
        registry.register(&identity).expect("Should register");

        let message = b"test message";
        let signature = identity.sign(message);

        // Valid signature should verify
        let result = registry.verify_identity(identity.agent_id(), message, &signature);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_identity_verification_wrong_message() {
        let mut registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");
        registry.register(&identity).expect("Should register");

        let message = b"test message";
        let wrong_message = b"wrong message";
        let signature = identity.sign(message);

        // Wrong message should fail verification
        let result = registry.verify_identity(identity.agent_id(), wrong_message, &signature);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_identity_verification_unknown_agent() {
        let registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");

        let message = b"test message";
        let signature = identity.sign(message);

        // Unknown agent should return error
        let result = registry.verify_identity(identity.agent_id(), message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_by_capability() {
        let mut registry = AgentRegistry::new();

        // Create agents with different capability vectors
        let mut cap1 = [0.0f32; 128];
        cap1[0] = 1.0;
        let identity1 = AgentIdentity::new()
            .expect("Should create identity")
            .with_capability_vector(cap1);

        let mut cap2 = [0.0f32; 128];
        cap2[1] = 1.0;
        let identity2 = AgentIdentity::new()
            .expect("Should create identity")
            .with_capability_vector(cap2);

        registry.register(&identity1).expect("Should register");
        registry.register(&identity2).expect("Should register");

        // Search for agents similar to cap1
        let query = cap1.to_vec();
        let results = registry.find_by_capability(&query, 2);

        assert_eq!(results.len(), 2);
        // First result should be the most similar (identity1)
        assert_eq!(results[0].agent_info.agent_id, *identity1.agent_id());
    }

    #[test]
    fn test_agent_removal() {
        let mut registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");

        registry.register(&identity).expect("Should register");
        assert_eq!(registry.len(), 1);

        registry.remove(identity.agent_id()).expect("Should remove");
        assert_eq!(registry.len(), 0);
        assert!(registry.get(identity.agent_id()).is_none());
    }

    #[test]
    fn test_status_update() {
        let mut registry = AgentRegistry::new();
        let identity = AgentIdentity::new().expect("Should create identity");

        registry.register(&identity).expect("Should register");

        registry
            .update_status(identity.agent_id(), AgentStatus::Busy)
            .expect("Should update status");

        let info = registry.get(identity.agent_id()).expect("Should get agent");
        assert_eq!(info.status, AgentStatus::Busy);
    }
}
