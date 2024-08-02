# L-SAGA: A Learning Hyper-Heuristic Architecture for the Permutation Flow-Shop Problem

## Overview
L-SAGA (Learning Simulated Annealing Genetic Algorithm) is a generative hyper-heuristic designed to address the Permutation Flow-Shop Problem (PFSP), an NP-hard problem significant in manufacturing and production environments.

## Components
L-SAGA combines two main components:
- **Low-Level Genetic Algorithm**: Handles job sequencing with operations like initialization, selection, crossover, mutation, and replacement.
- **High-Level Simulated Annealing**: Optimizes hyperparameters of the genetic algorithm, incorporating a learning component to enhance the search process.
The integration of these components allows L-SAGA to efficiently find optimal or sub-optimal solutions for PFSP instances, adapting to various problem sizes and maintaining high solution quality.

## Key Features
- **Generative Hyper-Heuristic:** Automatically generates and selects heuristics for diverse PFSP instances.
- **Hybrid Approach**: Combines simulated annealing and genetic algorithms to balance exploration and exploitation.
- **Adaptability**: Can handle various problem instances with different numbers of jobs and machines.
- **Performance**: Demonstrates potential for high-quality solutions as shown by benchmark tests.
