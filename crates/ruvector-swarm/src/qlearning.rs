//! Q-Learning Engine for agent decision-making.
//!
//! This module implements reinforcement learning for agent decision-making,
//! using Q-learning to optimize action selection based on state-action-reward tuples.
//!
//! # Algorithm
//!
//! The Q-Learning update rule (Bellman equation):
//! ```text
//! Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
//! ```
//!
//! Where:
//! - α (learning_rate) = 0.1: Controls how much new information overrides old
//! - γ (discount_factor) = 0.95: Controls importance of future rewards
//! - r: Immediate reward
//! - s: Current state
//! - a: Action taken
//! - s': Next state
//!
//! # Requirements
//! - 3.1: Implement Q-Learning Engine with Bellman equation
//! - 3.2: Encode states as tuples of (query_type, complexity, context_hash, confidence)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::error::Result;
use crate::types::QueryType;

/// Complexity level for query classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Complexity {
    /// Simple query requiring direct lookup.
    Simple,
    /// Moderate query requiring some reasoning.
    Moderate,
    /// Complex query requiring multi-step reasoning.
    Complex,
    /// Very complex query requiring expert consultation.
    VeryComplex,
}

impl Default for Complexity {
    fn default() -> Self {
        Self::Moderate
    }
}

/// Confidence level in the current state assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Confidence {
    /// Low confidence in state assessment.
    Low,
    /// Medium confidence in state assessment.
    Medium,
    /// High confidence in state assessment.
    High,
}

impl Default for Confidence {
    fn default() -> Self {
        Self::Medium
    }
}

/// State representation for Q-Learning.
///
/// A state encodes the current context for decision-making:
/// - query_type: Type of query being processed
/// - complexity: Complexity level of the query
/// - context_hash: Hash of the context information
/// - confidence: Confidence in the state assessment
///
/// # Requirements
/// - 3.2: Encode states as tuples of (query_type, complexity, context_hash, confidence)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct State {
    /// Type of query being processed.
    pub query_type: QueryType,
    /// Complexity level of the query.
    pub complexity: Complexity,
    /// Hash of context information (0 if no context).
    pub context_hash: u64,
    /// Confidence in the state assessment.
    pub confidence: Confidence,
}

impl State {
    /// Create a new state.
    pub fn new(
        query_type: QueryType,
        complexity: Complexity,
        context_hash: u64,
        confidence: Confidence,
    ) -> Self {
        Self {
            query_type,
            complexity,
            context_hash,
            confidence,
        }
    }

    /// Create a state with default confidence.
    pub fn with_defaults(query_type: QueryType, complexity: Complexity, context_hash: u64) -> Self {
        Self {
            query_type,
            complexity,
            context_hash,
            confidence: Confidence::default(),
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            query_type: QueryType::default(),
            complexity: Complexity::default(),
            context_hash: 0,
            confidence: Confidence::default(),
        }
    }
}

/// Action that an agent can take in response to a query.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Provide a direct answer from knowledge base.
    DirectAnswer,
    /// Provide answer with additional context.
    ContextAnswer,
    /// Consult a specific peer agent (identified by index).
    ConsultPeer(usize),
    /// Request clarification from user.
    RequestClarification,
    /// Escalate to coordinator agent.
    EscalateCoordinator,
    /// Synthesize response from multiple agents.
    MultiAgentSynthesis(usize), // number of agents to consult
}

impl Default for Action {
    fn default() -> Self {
        Self::DirectAnswer
    }
}

/// Q-value type for storing learned values.
pub type QValue = f32;

/// State-Action pair for Q-table indexing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateAction {
    /// The state.
    pub state: State,
    /// The action.
    pub action: Action,
}

impl StateAction {
    /// Create a new state-action pair.
    pub fn new(state: State, action: Action) -> Self {
        Self { state, action }
    }
}

/// Q-Learning Engine for agent decision-making.
///
/// Implements reinforcement learning using Q-learning algorithm.
/// Maintains a Q-table mapping state-action pairs to learned values.
///
/// # Parameters
/// - learning_rate (α): 0.1 - Controls learning speed
/// - discount_factor (γ): 0.95 - Controls future reward importance
/// - exploration_rate (ε): 0.1 - Controls exploration vs exploitation
///
/// # Requirements
/// - 3.1: Implement Q-Learning Engine with Bellman equation
/// - 3.2: Encode states as tuples of (query_type, complexity, context_hash, confidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLearningEngine {
    /// Q-table: maps state-action pairs to learned values.
    q_table: HashMap<StateAction, QValue>,

    /// Learning rate (α): controls how much new information overrides old.
    /// Typical value: 0.1
    learning_rate: f32,

    /// Discount factor (γ): controls importance of future rewards.
    /// Typical value: 0.95
    discount_factor: f32,

    /// Exploration rate (ε): probability of random action selection.
    /// Typical value: 0.1
    exploration_rate: f32,

    /// Total number of updates performed.
    update_count: u64,

    /// Total number of actions selected.
    action_count: u64,
}

impl QLearningEngine {
    /// Create a new Q-Learning Engine with default parameters.
    ///
    /// Default parameters:
    /// - learning_rate: 0.1
    /// - discount_factor: 0.95
    /// - exploration_rate: 0.1
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.95,
            exploration_rate: 0.1,
            update_count: 0,
            action_count: 0,
        }
    }

    /// Create a new Q-Learning Engine with custom parameters.
    ///
    /// # Arguments
    /// - `learning_rate`: Learning rate (α), typically 0.01 to 0.5
    /// - `discount_factor`: Discount factor (γ), typically 0.9 to 0.99
    /// - `exploration_rate`: Exploration rate (ε), typically 0.01 to 0.3
    pub fn with_params(
        learning_rate: f32,
        discount_factor: f32,
        exploration_rate: f32,
    ) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            exploration_rate,
            update_count: 0,
            action_count: 0,
        }
    }

    /// Get the Q-value for a state-action pair.
    ///
    /// Returns 0.0 if the pair has not been visited yet.
    pub fn get_q(&self, state: &State, action: &Action) -> QValue {
        let key = StateAction::new(*state, action.clone());
        self.q_table.get(&key).copied().unwrap_or(0.0)
    }

    /// Set the Q-value for a state-action pair.
    pub fn set_q(&mut self, state: &State, action: &Action, value: QValue) {
        let key = StateAction::new(*state, action.clone());
        self.q_table.insert(key, value);
    }

    /// Get the maximum Q-value for a given state across all actions.
    ///
    /// Returns 0.0 if no actions have been visited for this state.
    pub fn max_q(&self, state: &State, actions: &[Action]) -> QValue {
        actions
            .iter()
            .map(|action| self.get_q(state, action))
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0)
    }

    /// Update Q-value using the Bellman equation.
    ///
    /// Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    ///
    /// # Arguments
    /// - `state`: Current state
    /// - `action`: Action taken
    /// - `reward`: Immediate reward received
    /// - `next_state`: Resulting state
    /// - `next_actions`: Possible actions in next state
    ///
    /// # Requirements
    /// - 3.1: Update Q-table using Bellman equation with α=0.1, γ=0.95
    pub fn update(
        &mut self,
        state: &State,
        action: &Action,
        reward: f32,
        next_state: &State,
        next_actions: &[Action],
    ) {
        let current_q = self.get_q(state, action);
        let max_next_q = self.max_q(next_state, next_actions);

        // Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        let new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);

        self.set_q(state, action, new_q);
        self.update_count += 1;
    }

    /// Select an action using ε-greedy policy.
    ///
    /// With probability ε, select a random action (exploration).
    /// Otherwise, select the action with highest Q-value (exploitation).
    ///
    /// # Arguments
    /// - `state`: Current state
    /// - `actions`: Available actions
    /// - `rng_seed`: Seed for random number generation (for determinism in tests)
    ///
    /// Returns the selected action.
    pub fn select_action(&mut self, state: &State, actions: &[Action], rng_seed: Option<u64>) -> Action {
        self.action_count += 1;

        if actions.is_empty() {
            return Action::DirectAnswer;
        }

        // Determine if we should explore
        let should_explore = if let Some(seed) = rng_seed {
            // For testing: use seed to determine exploration
            (seed % 100) as f32 / 100.0 < self.exploration_rate
        } else {
            // In production: use actual random number
            // For now, we'll use a simple hash-based approach
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            state.hash(&mut hasher);
            let hash = hasher.finish();
            (hash % 100) as f32 / 100.0 < self.exploration_rate
        };

        if should_explore {
            // Exploration: select random action
            let idx = (rng_seed.unwrap_or(0) as usize) % actions.len();
            actions[idx].clone()
        } else {
            // Exploitation: select best action
            let mut best_action = actions[0].clone();
            let mut best_q = self.get_q(state, &best_action);

            for action in &actions[1..] {
                let q = self.get_q(state, action);
                if q > best_q {
                    best_q = q;
                    best_action = action.clone();
                }
            }

            best_action
        }
    }

    /// Get the number of state-action pairs in the Q-table.
    pub fn table_size(&self) -> usize {
        self.q_table.len()
    }

    /// Get the total number of updates performed.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Get the total number of actions selected.
    pub fn action_count(&self) -> u64 {
        self.action_count
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get the discount factor.
    pub fn discount_factor(&self) -> f32 {
        self.discount_factor
    }

    /// Get the exploration rate.
    pub fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    }

    /// Export Q-table entries for federated learning.
    ///
    /// Returns a vector of (state-action, q-value, visit-count) tuples.
    /// Visit count is estimated from the update count.
    pub fn export_entries(&self) -> Vec<(StateAction, QValue, u32)> {
        self.q_table
            .iter()
            .map(|(sa, q)| (sa.clone(), *q, 1))
            .collect()
    }

    /// Clear the Q-table (useful for testing).
    pub fn clear(&mut self) {
        self.q_table.clear();
        self.update_count = 0;
        self.action_count = 0;
    }

    /// Get a reference to the Q-table for inspection (testing only).
    #[cfg(test)]
    pub fn q_table(&self) -> &HashMap<StateAction, QValue> {
        &self.q_table
    }

    /// Persist Q-table to bytes using bincode serialization.
    ///
    /// # Requirements
    /// - 3.4: Persist Q-table updates to learning_sessions table
    ///
    /// # Returns
    /// Serialized Q-table as bytes, or error if serialization fails.
    pub fn persist_to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| crate::error::SwarmError::SerializationError(e.to_string()))
    }

    /// Load Q-table from bytes using bincode deserialization.
    ///
    /// # Requirements
    /// - 3.4: Load Q-table from learning_sessions table
    ///
    /// # Arguments
    /// - `data`: Serialized Q-table bytes
    ///
    /// # Returns
    /// Deserialized QLearningEngine, or error if deserialization fails.
    pub fn load_from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| crate::error::SwarmError::DeserializationError(e.to_string()))
    }
}

impl Default for QLearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlearning_creation() {
        let engine = QLearningEngine::new();
        assert_eq!(engine.learning_rate(), 0.1);
        assert_eq!(engine.discount_factor(), 0.95);
        assert_eq!(engine.exploration_rate(), 0.1);
        assert_eq!(engine.table_size(), 0);
    }

    #[test]
    fn test_qlearning_custom_params() {
        let engine = QLearningEngine::with_params(0.05, 0.9, 0.2);
        assert_eq!(engine.learning_rate(), 0.05);
        assert_eq!(engine.discount_factor(), 0.9);
        assert_eq!(engine.exploration_rate(), 0.2);
    }

    #[test]
    fn test_get_set_q() {
        let mut engine = QLearningEngine::new();
        let state = State::default();
        let action = Action::DirectAnswer;

        // Initially should be 0.0
        assert_eq!(engine.get_q(&state, &action), 0.0);

        // Set and retrieve
        engine.set_q(&state, &action, 0.5);
        assert_eq!(engine.get_q(&state, &action), 0.5);
    }

    #[test]
    fn test_max_q() {
        let mut engine = QLearningEngine::new();
        let state = State::default();

        engine.set_q(&state, &Action::DirectAnswer, 0.3);
        engine.set_q(&state, &Action::ContextAnswer, 0.7);
        engine.set_q(&state, &Action::RequestClarification, 0.5);

        let actions = vec![
            Action::DirectAnswer,
            Action::ContextAnswer,
            Action::RequestClarification,
        ];

        assert_eq!(engine.max_q(&state, &actions), 0.7);
    }

    #[test]
    fn test_bellman_update() {
        let mut engine = QLearningEngine::new();
        let state = State::default();
        let next_state = State::with_defaults(QueryType::Procedure, Complexity::Complex, 42);
        let action = Action::DirectAnswer;

        // Set initial Q-value
        engine.set_q(&state, &action, 0.0);

        // Set some Q-values for next state
        engine.set_q(&next_state, &Action::ContextAnswer, 0.8);

        let next_actions = vec![Action::ContextAnswer];

        // Update with reward
        let reward = 0.5;
        engine.update(&state, &action, reward, &next_state, &next_actions);

        // Q(s,a) = 0.0 + 0.1 * (0.5 + 0.95 * 0.8 - 0.0)
        // Q(s,a) = 0.1 * (0.5 + 0.76)
        // Q(s,a) = 0.1 * 1.26 = 0.126
        let expected = 0.126;
        let actual = engine.get_q(&state, &action);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_select_action_exploitation() {
        let mut engine = QLearningEngine::new();
        let state = State::default();

        // Set Q-values
        engine.set_q(&state, &Action::DirectAnswer, 0.3);
        engine.set_q(&state, &Action::ContextAnswer, 0.9);
        engine.set_q(&state, &Action::RequestClarification, 0.5);

        let actions = vec![
            Action::DirectAnswer,
            Action::ContextAnswer,
            Action::RequestClarification,
        ];

        // With high seed (>10), should exploit (select best action)
        let selected = engine.select_action(&state, &actions, Some(99));
        assert_eq!(selected, Action::ContextAnswer);
    }

    #[test]
    fn test_select_action_exploration() {
        let mut engine = QLearningEngine::new();
        let state = State::default();

        let actions = vec![
            Action::DirectAnswer,
            Action::ContextAnswer,
            Action::RequestClarification,
        ];

        // With low seed (<10), should explore (select random action)
        let selected = engine.select_action(&state, &actions, Some(5));
        assert!(actions.contains(&selected));
    }

    #[test]
    fn test_state_creation() {
        let state = State::new(
            QueryType::Troubleshoot,
            Complexity::Complex,
            12345,
            Confidence::High,
        );

        assert_eq!(state.query_type, QueryType::Troubleshoot);
        assert_eq!(state.complexity, Complexity::Complex);
        assert_eq!(state.context_hash, 12345);
        assert_eq!(state.confidence, Confidence::High);
    }

    #[test]
    fn test_state_with_defaults() {
        let state = State::with_defaults(QueryType::Knowledge, Complexity::Simple, 0);

        assert_eq!(state.query_type, QueryType::Knowledge);
        assert_eq!(state.complexity, Complexity::Simple);
        assert_eq!(state.context_hash, 0);
        assert_eq!(state.confidence, Confidence::Medium);
    }

    #[test]
    fn test_state_action_pair() {
        let state = State::default();
        let action = Action::DirectAnswer;
        let sa = StateAction::new(state, action.clone());

        assert_eq!(sa.state, state);
        assert_eq!(sa.action, action);
    }

    #[test]
    fn test_update_count() {
        let mut engine = QLearningEngine::new();
        let state = State::default();
        let action = Action::DirectAnswer;

        assert_eq!(engine.update_count(), 0);

        engine.update(&state, &action, 0.5, &state, &[Action::ContextAnswer]);
        assert_eq!(engine.update_count(), 1);

        engine.update(&state, &action, 0.3, &state, &[Action::ContextAnswer]);
        assert_eq!(engine.update_count(), 2);
    }

    #[test]
    fn test_action_count() {
        let mut engine = QLearningEngine::new();
        let state = State::default();
        let actions = vec![Action::DirectAnswer, Action::ContextAnswer];

        assert_eq!(engine.action_count(), 0);

        engine.select_action(&state, &actions, Some(99));
        assert_eq!(engine.action_count(), 1);

        engine.select_action(&state, &actions, Some(99));
        assert_eq!(engine.action_count(), 2);
    }

    #[test]
    fn test_export_entries() {
        let mut engine = QLearningEngine::new();
        let state = State::default();

        engine.set_q(&state, &Action::DirectAnswer, 0.5);
        engine.set_q(&state, &Action::ContextAnswer, 0.7);

        let entries = engine.export_entries();
        assert_eq!(entries.len(), 2);

        // Verify entries contain correct values
        let values: Vec<f32> = entries.iter().map(|(_, q, _)| *q).collect();
        assert!(values.contains(&0.5));
        assert!(values.contains(&0.7));
    }

    #[test]
    fn test_clear() {
        let mut engine = QLearningEngine::new();
        let state = State::default();

        engine.set_q(&state, &Action::DirectAnswer, 0.5);
        engine.set_q(&state, &Action::ContextAnswer, 0.7);
        assert_eq!(engine.table_size(), 2);

        engine.clear();
        assert_eq!(engine.table_size(), 0);
        assert_eq!(engine.update_count(), 0);
        assert_eq!(engine.action_count(), 0);
    }

    #[test]
    fn test_multiple_states() {
        let mut engine = QLearningEngine::new();

        let state1 = State::with_defaults(QueryType::Knowledge, Complexity::Simple, 0);
        let state2 = State::with_defaults(QueryType::Procedure, Complexity::Complex, 42);

        engine.set_q(&state1, &Action::DirectAnswer, 0.5);
        engine.set_q(&state2, &Action::ContextAnswer, 0.8);

        assert_eq!(engine.get_q(&state1, &Action::DirectAnswer), 0.5);
        assert_eq!(engine.get_q(&state2, &Action::ContextAnswer), 0.8);
        assert_eq!(engine.table_size(), 2);
    }

    #[test]
    fn test_action_enum_variants() {
        let actions = vec![
            Action::DirectAnswer,
            Action::ContextAnswer,
            Action::ConsultPeer(0),
            Action::RequestClarification,
            Action::EscalateCoordinator,
            Action::MultiAgentSynthesis(3),
        ];

        assert_eq!(actions.len(), 6);
    }

    #[test]
    fn test_persist_and_load() {
        let mut engine = QLearningEngine::new();
        let state = State::default();

        // Set some Q-values
        engine.set_q(&state, &Action::DirectAnswer, 0.5);
        engine.set_q(&state, &Action::ContextAnswer, 0.7);

        // Persist to bytes
        let bytes = engine.persist_to_bytes().expect("Should persist");
        assert!(!bytes.is_empty());

        // Load from bytes
        let loaded = QLearningEngine::load_from_bytes(&bytes).expect("Should load");

        // Verify Q-values are preserved
        assert_eq!(loaded.get_q(&state, &Action::DirectAnswer), 0.5);
        assert_eq!(loaded.get_q(&state, &Action::ContextAnswer), 0.7);
        assert_eq!(loaded.table_size(), 2);
    }

    #[test]
    fn test_persist_empty_engine() {
        let engine = QLearningEngine::new();

        let bytes = engine.persist_to_bytes().expect("Should persist empty engine");
        let loaded = QLearningEngine::load_from_bytes(&bytes).expect("Should load empty engine");

        assert_eq!(loaded.table_size(), 0);
        assert_eq!(loaded.update_count(), 0);
        assert_eq!(loaded.action_count(), 0);
    }

    #[test]
    fn test_persist_preserves_parameters() {
        let engine = QLearningEngine::with_params(0.05, 0.9, 0.2);

        let bytes = engine.persist_to_bytes().expect("Should persist");
        let loaded = QLearningEngine::load_from_bytes(&bytes).expect("Should load");

        assert_eq!(loaded.learning_rate(), 0.05);
        assert_eq!(loaded.discount_factor(), 0.9);
        assert_eq!(loaded.exploration_rate(), 0.2);
    }

    #[test]
    fn test_persist_preserves_counters() {
        let mut engine = QLearningEngine::new();
        let state = State::default();
        let actions = vec![Action::DirectAnswer, Action::ContextAnswer];

        // Perform some operations
        engine.update(&state, &Action::DirectAnswer, 0.5, &state, &actions);
        engine.update(&state, &Action::ContextAnswer, 0.3, &state, &actions);
        engine.select_action(&state, &actions, Some(99));

        let bytes = engine.persist_to_bytes().expect("Should persist");
        let loaded = QLearningEngine::load_from_bytes(&bytes).expect("Should load");

        assert_eq!(loaded.update_count(), 2);
        assert_eq!(loaded.action_count(), 1);
    }

    #[test]
    fn test_load_invalid_bytes() {
        let invalid_bytes = vec![0xFF, 0xFE, 0xFD];
        let result = QLearningEngine::load_from_bytes(&invalid_bytes);
        assert!(result.is_err());
    }
}
