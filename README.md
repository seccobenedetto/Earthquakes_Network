# Earthquake Network Analysis in Italy

## Overview
This project explores seismic activity in Italy using tools from network science. The goal is to model earthquakes as complex networks and study their structural and statistical properties.

## Objectives
- Perform preliminary data analysis (magnitude, depth, temporal trends)
- Validate empirical seismic laws:
  - Gutenberg–Richter law
  - Omori law (aftershock decay)

## Network Construction
- Model based on Abe–Suzuki approach:
  - Divide space into 3D cells (e.g., 5x5x5 km, 10x10x10 km)
  - Nodes: spatial cells
  - Edges: consecutive earthquakes
  - Directed network (optionally treated as undirected)
  - Self-loops allowed

## Network Analysis
- Adjacency matrix of spatial grid
- Degree distribution (identify hubs)
- Diameter and average path length
- Clustering coefficient (aftershock patterns)
- Connectivity and giant component size

## Network Models Comparison
- Compare with:
  - Erdős–Rényi random graphs
  - Preferential attachment models
- Evaluate scale-free behavior (power-law degree distribution)

## Centrality Measures
- Degree centrality: most active regions
- Betweenness: bridges between seismic zones
- Closeness: globally influential areas
- PageRank: importance in seismic propagation

## Additional Analyses
- Assortativity (homophily):
  - Test similarity in magnitude, depth, etc.
- Robustness:
  - Random node removal
  - Targeted attacks on central nodes

## Community Detection
- Identify seismic regions and fault systems using:
  - Louvain
  - InfoMap
  - Spectral clustering
  - HDBSCAN

## Visualization
- Plot earthquakes on maps of Italy
- Use Gephi for network visualization and analysis

## Extensions
- Baiesi–Paczuski model:
  - Correlation-based links between earthquakes
- Telesca–Lovallo model:
  - Visibility graph based on magnitude time series

## Status
Work in progress. Future updates will include implementation details and results.
