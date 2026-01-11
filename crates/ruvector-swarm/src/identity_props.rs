//! Property-based tests for agent identity.
//!
//! **Feature: agentic-swarm-learning, Property 1: Agent Identity Integrity**
//! **Validates: Requirements 1.1, 1.3**

#[cfg(test)]
mod property_tests {
    use crate::identity::AgentIdentity;
    use crate::types::AgentId;
    use proptest::prelude::*;
    use std::collections::HashSet;

    /// Generate a random capability vector (128 dimensions, values in [-1.0, 1.0])
    fn arb_capability_vector() -> impl Strategy<Value = [f32; 128]> {
        proptest::collection::vec(-1.0f32..=1.0f32, 128)
            .prop_map(|v| {
                let mut arr = [0.0f32; 128];
                arr.copy_from_slice(&v);
                arr
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agentic-swarm-learning, Property 1: Agent Identity Integrity**
        ///
        /// For any agent initialization, the Agent_Registry SHALL generate a valid
        /// Ed25519 keypair with a unique public key, and the binding between agent_id,
        /// public_key, and capability_vector SHALL remain immutable throughout the
        /// agent's lifecycle.
        ///
        /// **Validates: Requirements 1.1, 1.3**
        #[test]
        fn prop_agent_identity_integrity(capability_vector in arb_capability_vector()) {
            // Create identity with capability vector
            let identity = AgentIdentity::new()
                .expect("Should create identity")
                .with_capability_vector(capability_vector);

            // 1. Verify valid Ed25519 keypair by signing and verifying
            let test_message = b"integrity test message";
            let signature = identity.sign(test_message);
            prop_assert!(
                identity.verify(test_message, &signature).is_ok(),
                "Signature verification should succeed for valid keypair"
            );

            // 2. Verify agent_id is derived from public key
            let expected_agent_id = AgentId::from_public_key(identity.public_key_bytes());
            prop_assert_eq!(
                identity.agent_id(),
                &expected_agent_id,
                "Agent ID should be derived from public key"
            );

            // 3. Verify capability vector binding is maintained
            prop_assert_eq!(
                identity.capability_vector(),
                &capability_vector,
                "Capability vector should remain unchanged"
            );

            // 4. Verify public key is 32 bytes (Ed25519 standard)
            prop_assert_eq!(
                identity.public_key_bytes().len(),
                32,
                "Public key should be 32 bytes"
            );
        }

        /// Property: Multiple identity generations produce unique keypairs
        ///
        /// **Validates: Requirements 1.1**
        #[test]
        fn prop_unique_identity_generation(_seed in 0u64..1000) {
            // Generate multiple identities and verify uniqueness
            let mut agent_ids = HashSet::new();
            let mut public_keys = HashSet::new();

            for _ in 0..10 {
                let identity = AgentIdentity::new().expect("Should create identity");
                
                // Verify each identity has unique agent_id
                let agent_id = identity.agent_id().clone();
                prop_assert!(
                    agent_ids.insert(agent_id.as_str().to_string()),
                    "Agent IDs should be unique"
                );

                // Verify each identity has unique public key
                let pk_hex: String = identity.public_key_bytes()
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                prop_assert!(
                    public_keys.insert(pk_hex),
                    "Public keys should be unique"
                );
            }
        }

        /// Property: Sign-verify round trip always succeeds for valid messages
        ///
        /// **Validates: Requirements 1.1**
        #[test]
        fn prop_sign_verify_roundtrip(message in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let identity = AgentIdentity::new().expect("Should create identity");
            
            let signature = identity.sign(&message);
            prop_assert!(
                identity.verify(&message, &signature).is_ok(),
                "Signature verification should succeed for the same message"
            );
        }

        /// Property: Verification fails for tampered messages
        ///
        /// **Validates: Requirements 1.1**
        #[test]
        fn prop_verify_fails_for_tampered_message(
            message in proptest::collection::vec(any::<u8>(), 1..1024),
            tamper_index in 0usize..1024,
        ) {
            let identity = AgentIdentity::new().expect("Should create identity");
            let signature = identity.sign(&message);

            // Tamper with the message
            let mut tampered = message.clone();
            let idx = tamper_index % tampered.len();
            tampered[idx] = tampered[idx].wrapping_add(1);

            // Verification should fail for tampered message
            prop_assert!(
                identity.verify(&tampered, &signature).is_err(),
                "Signature verification should fail for tampered message"
            );
        }
    }
}
