//! Agent identity module with Ed25519 cryptographic support.
//!
//! This module provides cryptographic identity for agents in the swarm:
//! - Ed25519 keypair generation
//! - Message signing and verification
//! - Immutable binding between agent_id, public_key, and capability_vector

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SwarmError};
use crate::types::AgentId;

/// Agent identity with Ed25519 keypair and capability vector.
///
/// The identity maintains an immutable binding between:
/// - `agent_id`: Derived from the public key
/// - `public_key`: Ed25519 verifying key
/// - `capability_vector`: 128-dimension embedding describing agent capabilities
#[derive(Debug)]
pub struct AgentIdentity {
    /// Unique agent identifier derived from public key
    agent_id: AgentId,
    /// Ed25519 signing key (private)
    signing_key: SigningKey,
    /// Ed25519 verifying key (public)
    verifying_key: VerifyingKey,
    /// 128-dimension capability vector
    capability_vector: [f32; 128],
}

impl AgentIdentity {
    /// Create a new agent identity with a fresh Ed25519 keypair.
    ///
    /// Generates a unique keypair and derives the agent_id from the public key.
    /// The capability vector is initialized to zeros and should be set via
    /// `with_capability_vector`.
    pub fn new() -> Result<Self> {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let agent_id = AgentId::from_public_key(verifying_key.as_bytes());

        Ok(Self {
            agent_id,
            signing_key,
            verifying_key,
            capability_vector: [0.0; 128],
        })
    }

    /// Create an agent identity with a specific capability vector.
    pub fn with_capability_vector(mut self, capability_vector: [f32; 128]) -> Self {
        self.capability_vector = capability_vector;
        self
    }


    /// Get the agent's unique identifier.
    pub fn agent_id(&self) -> &AgentId {
        &self.agent_id
    }

    /// Get the agent's public key bytes.
    pub fn public_key_bytes(&self) -> &[u8; 32] {
        self.verifying_key.as_bytes()
    }

    /// Get the agent's verifying (public) key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Get the agent's capability vector.
    pub fn capability_vector(&self) -> &[f32; 128] {
        &self.capability_vector
    }

    /// Sign a message with the agent's private key.
    ///
    /// Returns the Ed25519 signature.
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Verify a signature against a message using this agent's public key.
    ///
    /// Returns `Ok(())` if the signature is valid, or an error otherwise.
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<()> {
        self.verifying_key
            .verify(message, signature)
            .map_err(|e| SwarmError::SignatureVerificationError(e.to_string()))
    }

    /// Verify a signature using a provided public key.
    ///
    /// This is useful for verifying messages from other agents.
    pub fn verify_with_key(
        public_key: &VerifyingKey,
        message: &[u8],
        signature: &Signature,
    ) -> Result<()> {
        public_key
            .verify(message, signature)
            .map_err(|e| SwarmError::SignatureVerificationError(e.to_string()))
    }
}

impl Default for AgentIdentity {
    fn default() -> Self {
        Self::new().expect("Failed to generate default identity")
    }
}

/// Serializable representation of agent identity (without private key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentityInfo {
    /// Unique agent identifier
    pub agent_id: AgentId,
    /// Public key bytes (32 bytes)
    pub public_key: [u8; 32],
    /// 128-dimension capability vector
    #[serde(with = "capability_vector_serde")]
    pub capability_vector: [f32; 128],
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

impl From<&AgentIdentity> for AgentIdentityInfo {
    fn from(identity: &AgentIdentity) -> Self {
        Self {
            agent_id: identity.agent_id.clone(),
            public_key: *identity.public_key_bytes(),
            capability_vector: identity.capability_vector,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_creation() {
        let identity = AgentIdentity::new().expect("Should create identity");
        assert!(!identity.agent_id().as_str().is_empty());
    }

    #[test]
    fn test_sign_and_verify() {
        let identity = AgentIdentity::new().expect("Should create identity");
        let message = b"test message";
        let signature = identity.sign(message);
        assert!(identity.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_verify_wrong_message_fails() {
        let identity = AgentIdentity::new().expect("Should create identity");
        let message = b"test message";
        let wrong_message = b"wrong message";
        let signature = identity.sign(message);
        assert!(identity.verify(wrong_message, &signature).is_err());
    }

    #[test]
    fn test_unique_identities() {
        let id1 = AgentIdentity::new().expect("Should create identity");
        let id2 = AgentIdentity::new().expect("Should create identity");
        assert_ne!(id1.agent_id(), id2.agent_id());
        assert_ne!(id1.public_key_bytes(), id2.public_key_bytes());
    }
}
