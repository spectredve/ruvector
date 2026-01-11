# ruvector-swarm

Agentic AI Self-Learning Swarm with distributed coordination and federated learning.

## Overview

This crate implements a distributed swarm of AI agents that can:
- Coordinate through P2P communication
- Learn continuously from interactions
- Share knowledge through federated learning
- Run on edge devices with zero cloud dependency

## Features

- **Agent Identity**: Ed25519 cryptographic identity for each agent
- **Q-Learning Engine**: Reinforcement learning for decision optimization
- **ReasoningBank**: Pattern storage for successful trajectories
- **Semantic Router**: Query routing based on agent capabilities
- **Federated Sync**: Privacy-preserving knowledge sharing
- **WASM Support**: Browser and edge deployment

## Usage

```rust
use ruvector_swarm::{AgentId, Query, Response, Feedback};

// Create a new agent identity
let agent_id = AgentId::new();

// Create a query
let query = Query::new("How do I configure the system?");

// Process and get response
let response = Response::new(query.id, "Configuration steps...", 0.95);

// Provide feedback
let feedback = Feedback::new(response.id, 0.8, true);
```

## Feature Flags

- `native` (default): Native runtime with tokio async support
- `wasm`: WebAssembly support for browser deployment
