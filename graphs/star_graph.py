"""
Star graph construction module for constellation detection.

This module implements k-Nearest Neighbors (k-NN) graph construction from
detected star positions, with PCA normalization for rotation and scale invariance.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class StarGraph:
    """
    Represents a star graph with nodes (star positions) and edges (connections).
    
    Attributes
    ----------
    nodes : np.ndarray, shape (N, 2)
        Star positions as (x, y) coordinates.
    edges : np.ndarray, shape (M, 2)
        Edge list with indices into nodes array.
    edge_weights : np.ndarray, shape (M,)
        Euclidean distances for each edge.
    normalized_nodes : np.ndarray, shape (N, 2), optional
        PCA-normalized node positions for rotation/scale invariance.
    pca_transform : dict, optional
        PCA transformation parameters (mean, components, scale).
    """
    nodes: np.ndarray
    edges: np.ndarray
    edge_weights: np.ndarray
    normalized_nodes: Optional[np.ndarray] = None
    pca_transform: Optional[Dict] = None
    
    def to_networkx(self) -> 'nx.Graph':
        """Convert to NetworkX graph (if available)."""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required. Install with: pip install networkx")
        
        G = nx.Graph()
        
        # Add nodes with positions
        for i, (x, y) in enumerate(self.nodes):
            G.add_node(i, pos=(x, y))
        
        # Add edges with weights
        for (i, j), weight in zip(self.edges, self.edge_weights):
            G.add_edge(int(i), int(j), weight=weight)
        
        return G
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix representation."""
        n = len(self.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        
        for (i, j), weight in zip(self.edges, self.edge_weights):
            i, j = int(i), int(j)
            adj[i, j] = weight
            adj[j, i] = weight  # Undirected graph
        
        return adj
    
    def get_num_components(self) -> int:
        """Get number of connected components in the graph."""
        if not HAS_NETWORKX:
            # Simple DFS-based component counting
            visited = np.zeros(len(self.nodes), dtype=bool)
            components = 0
            
            def dfs(node):
                visited[node] = True
                for edge in self.edges:
                    if edge[0] == node and not visited[edge[1]]:
                        dfs(edge[1])
                    elif edge[1] == node and not visited[edge[0]]:
                        dfs(edge[0])
            
            for i in range(len(self.nodes)):
                if not visited[i]:
                    dfs(i)
                    components += 1
            
            return components
        else:
            G = self.to_networkx()
            return nx.number_connected_components(G)


class StarGraphBuilder:
    """
    Build k-Nearest Neighbors star graphs from detected star positions.
    
    Parameters
    ----------
    k : int, default=4
        Number of nearest neighbors to connect for each star.
    use_pca_normalization : bool, default=True
        Whether to apply PCA normalization for rotation/scale invariance.
    directed : bool, default=False
        Whether to create directed edges. If False, creates undirected graph.
    """
    
    def __init__(
        self,
        k: int = 4,
        use_pca_normalization: bool = True,
        directed: bool = False,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.use_pca_normalization = use_pca_normalization
        self.directed = directed
    
    def build(self, centroids: np.ndarray) -> StarGraph:
        """
        Build a star graph from detected star centroids.
        
        Parameters
        ----------
        centroids : np.ndarray, shape (N, 2)
            Star centroids as (x, y) coordinates.
        
        Returns
        -------
        StarGraph
            Constructed star graph with nodes, edges, and edge weights.
        """
        if len(centroids) == 0:
            return StarGraph(
                nodes=np.empty((0, 2), dtype=np.float32),
                edges=np.empty((0, 2), dtype=np.int32),
                edge_weights=np.empty((0,), dtype=np.float32),
            )
        
        if len(centroids.shape) != 2 or centroids.shape[1] != 2:
            raise ValueError("centroids must be shape (N, 2)")
        
        # Build k-NN graph
        edges, edge_weights = self._build_knn_graph(centroids)
        
        # Normalize nodes if requested
        normalized_nodes = None
        pca_transform = None
        if self.use_pca_normalization:
            normalized_nodes, pca_transform = self._normalize_with_pca(centroids)
        
        return StarGraph(
            nodes=centroids.astype(np.float32),
            edges=edges,
            edge_weights=edge_weights,
            normalized_nodes=normalized_nodes,
            pca_transform=pca_transform,
        )
    
    def _build_knn_graph(
        self, centroids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph from centroids.
        
        Returns
        -------
        edges : np.ndarray, shape (M, 2)
            Edge list with node indices.
        edge_weights : np.ndarray, shape (M,)
            Euclidean distances for each edge.
        """
        n = len(centroids)
        
        # Handle edge case: fewer stars than k
        k_actual = min(self.k, n - 1)
        if k_actual < 1:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        
        edges_list = []
        weights_list = []
        
        if HAS_SCIPY:
            # Use KDTree for efficient k-NN search
            tree = KDTree(centroids)
            
            for i in range(n):
                # Find k+1 nearest neighbors (includes self)
                distances, indices = tree.query(centroids[i], k=k_actual + 1)
                
                # Remove self (first result)
                if len(indices.shape) == 0:
                    # Single neighbor case
                    neighbor_idx = indices
                    if neighbor_idx != i:
                        edges_list.append([i, neighbor_idx])
                        dist = np.linalg.norm(centroids[i] - centroids[neighbor_idx])
                        weights_list.append(dist)
                else:
                    for j, neighbor_idx in enumerate(indices):
                        if neighbor_idx != i:
                            edges_list.append([i, neighbor_idx])
                            weights_list.append(distances[j])
        else:
            # Fallback to numpy-based distance computation
            for i in range(n):
                # Compute distances to all other points
                distances = np.linalg.norm(centroids - centroids[i], axis=1)
                
                # Set self-distance to infinity to exclude it
                distances[i] = np.inf
                
                # Find k nearest neighbors
                nearest_indices = np.argsort(distances)[:k_actual]
                
                for neighbor_idx in nearest_indices:
                    if neighbor_idx != i:
                        edges_list.append([i, neighbor_idx])
                        weights_list.append(distances[neighbor_idx])
        
        if len(edges_list) == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        
        edges = np.array(edges_list, dtype=np.int32)
        edge_weights = np.array(weights_list, dtype=np.float32)
        
        # Make undirected if requested
        if not self.directed:
            edges, edge_weights = self._make_undirected(edges, edge_weights)
        
        return edges, edge_weights
    
    def _make_undirected(
        self, edges: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert directed edges to undirected (remove duplicates)."""
        # Sort edges to ensure consistent ordering
        sorted_edges = np.sort(edges, axis=1)
        
        # Remove duplicates while preserving weights
        unique_edges = []
        unique_weights = []
        seen = set()
        
        for edge, weight in zip(sorted_edges, weights):
            edge_tuple = tuple(edge)
            if edge_tuple not in seen:
                seen.add(edge_tuple)
                unique_edges.append(edge)
                unique_weights.append(weight)
        
        if len(unique_edges) == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        
        return np.array(unique_edges, dtype=np.int32), np.array(unique_weights, dtype=np.float32)
    
    def _normalize_with_pca(
        self, centroids: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize star positions using PCA for rotation and scale invariance.
        
        Steps:
        1. Center the points (subtract mean)
        2. Compute PCA
        3. Rotate to align with principal axes
        4. Scale to unit variance
        
        Returns
        -------
        normalized_nodes : np.ndarray
            Normalized node positions.
        pca_transform : dict
            Transformation parameters (mean, components, scale).
        """
        if len(centroids) < 2:
            # Can't do PCA with < 2 points
            return centroids.copy(), {
                'mean': np.mean(centroids, axis=0),
                'components': np.eye(2),
                'scale': np.array([1.0, 1.0]),
            }
        
        # Center the points
        mean = np.mean(centroids, axis=0)
        centered = centroids - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Rotate to align with principal axes
        rotated = centered @ eigenvectors
        
        # Scale to unit variance (or use standard deviation)
        std = np.std(rotated, axis=0)
        std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
        scale = 1.0 / std
        normalized = rotated * scale
        
        pca_transform = {
            'mean': mean,
            'components': eigenvectors,
            'scale': scale,
            'eigenvalues': eigenvalues,
        }
        
        return normalized.astype(np.float32), pca_transform
    
    def apply_pca_transform(
        self, centroids: np.ndarray, pca_transform: Dict
    ) -> np.ndarray:
        """
        Apply a previously computed PCA transform to new centroids.
        
        Useful for normalizing template graphs or test graphs using
        the same transformation.
        
        Parameters
        ----------
        centroids : np.ndarray, shape (N, 2)
            Star centroids to transform.
        pca_transform : dict
            PCA transformation parameters from _normalize_with_pca.
        
        Returns
        -------
        np.ndarray
            Transformed centroids.
        """
        # Center
        centered = centroids - pca_transform['mean']
        
        # Rotate
        rotated = centered @ pca_transform['components']
        
        # Scale
        normalized = rotated * pca_transform['scale']
        
        return normalized.astype(np.float32)

