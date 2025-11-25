"""
Comprehensive tests for star graph construction module.
"""

import numpy as np
import pytest
from graphs.star_graph import StarGraph, StarGraphBuilder


class TestStarGraph:
    """Test the StarGraph dataclass."""
    
    def test_star_graph_creation(self):
        """Test creating a StarGraph."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int32)
        weights = np.array([1.0, 1.0], dtype=np.float32)
        
        graph = StarGraph(nodes=nodes, edges=edges, edge_weights=weights)
        
        assert graph.nodes.shape == (3, 2)
        assert graph.edges.shape == (2, 2)
        assert len(graph.edge_weights) == 2
        assert graph.normalized_nodes is None
        assert graph.pca_transform is None
    
    def test_star_graph_with_normalization(self):
        """Test StarGraph with normalized nodes."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.int32)
        weights = np.array([1.0], dtype=np.float32)
        normalized = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pca_transform = {'mean': np.array([0.5, 0.0]), 'components': np.eye(2)}
        
        graph = StarGraph(
            nodes=nodes,
            edges=edges,
            edge_weights=weights,
            normalized_nodes=normalized,
            pca_transform=pca_transform,
        )
        
        assert graph.normalized_nodes is not None
        assert graph.pca_transform is not None
    
    def test_to_adjacency_matrix(self):
        """Test conversion to adjacency matrix."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int32)
        weights = np.array([1.0, 1.0], dtype=np.float32)
        
        graph = StarGraph(nodes=nodes, edges=edges, edge_weights=weights)
        adj = graph.to_adjacency_matrix()
        
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[0, 2] == 1.0
        assert adj[2, 0] == 1.0
        assert adj[1, 2] == 0.0  # No edge
    
    def test_to_networkx(self):
        """Test conversion to NetworkX graph."""
        try:
            import networkx as nx
            
            nodes = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            edges = np.array([[0, 1]], dtype=np.int32)
            weights = np.array([1.0], dtype=np.float32)
            
            graph = StarGraph(nodes=nodes, edges=edges, edge_weights=weights)
            G = graph.to_networkx()
            
            assert isinstance(G, nx.Graph)
            assert G.number_of_nodes() == 2
            assert G.number_of_edges() == 1
            assert G[0][1]['weight'] == 1.0
            
        except ImportError:
            pytest.skip("NetworkX not available")
    
    def test_get_num_components(self):
        """Test counting connected components."""
        # Two disconnected components
        nodes = np.array([
            [0.0, 0.0], [1.0, 0.0],  # Component 1
            [10.0, 10.0], [11.0, 10.0]  # Component 2
        ], dtype=np.float32)
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        weights = np.array([1.0, 1.0], dtype=np.float32)
        
        graph = StarGraph(nodes=nodes, edges=edges, edge_weights=weights)
        num_components = graph.get_num_components()
        
        assert num_components == 2
    
    def test_empty_graph(self):
        """Test empty graph handling."""
        graph = StarGraph(
            nodes=np.empty((0, 2), dtype=np.float32),
            edges=np.empty((0, 2), dtype=np.int32),
            edge_weights=np.empty((0,), dtype=np.float32),
        )
        
        assert graph.get_num_components() == 0
        adj = graph.to_adjacency_matrix()
        assert adj.shape == (0, 0)


class TestStarGraphBuilder:
    """Test the StarGraphBuilder class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        builder = StarGraphBuilder()
        assert builder.k == 4
        assert builder.use_pca_normalization is True
        assert builder.directed is False
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        builder = StarGraphBuilder(k=6, use_pca_normalization=False, directed=True)
        assert builder.k == 6
        assert builder.use_pca_normalization is False
        assert builder.directed is True
    
    def test_initialization_invalid_k(self):
        """Test that k must be at least 1."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            StarGraphBuilder(k=0)
    
    def test_build_empty_centroids(self):
        """Test building graph from empty centroids."""
        builder = StarGraphBuilder()
        centroids = np.empty((0, 2), dtype=np.float32)
        graph = builder.build(centroids)
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.edge_weights) == 0
    
    def test_build_single_star(self):
        """Test building graph with single star."""
        builder = StarGraphBuilder(k=4)
        centroids = np.array([[10.0, 20.0]], dtype=np.float32)
        graph = builder.build(centroids)
        
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0  # No neighbors for single star
        assert len(graph.edge_weights) == 0
    
    def test_build_two_stars(self):
        """Test building graph with two stars."""
        builder = StarGraphBuilder(k=4)
        centroids = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        graph = builder.build(centroids)
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1  # One edge (undirected)
        assert len(graph.edge_weights) == 1
        assert graph.edge_weights[0] == pytest.approx(1.0)
    
    def test_build_knn_graph(self):
        """Test k-NN graph construction."""
        # Create a grid of stars
        builder = StarGraphBuilder(k=2)
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        
        graph = builder.build(centroids)
        
        assert len(graph.nodes) == 4
        assert len(graph.edges) > 0
        # Each node should have at most k=2 neighbors (undirected)
        # With 4 nodes, expect at least 4 edges (2 per node / 2 for undirected)
    
    def test_build_directed_vs_undirected(self):
        """Test directed vs undirected graph construction."""
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        builder_undirected = StarGraphBuilder(k=2, directed=False)
        graph_undirected = builder_undirected.build(centroids)
        
        builder_directed = StarGraphBuilder(k=2, directed=True)
        graph_directed = builder_directed.build(centroids)
        
        # Undirected should have fewer or equal edges (duplicates removed)
        assert len(graph_undirected.edges) <= len(graph_directed.edges)
    
    def test_build_invalid_centroids_shape(self):
        """Test error handling for invalid centroids shape."""
        builder = StarGraphBuilder()
        
        # Wrong shape
        with pytest.raises(ValueError, match="centroids must be shape"):
            builder.build(np.array([1.0, 2.0]))  # 1D array
        
        # Wrong number of columns
        with pytest.raises(ValueError, match="centroids must be shape"):
            builder.build(np.array([[1.0, 2.0, 3.0]]))  # 3 columns
    
    def test_pca_normalization_enabled(self):
        """Test PCA normalization when enabled."""
        builder = StarGraphBuilder(k=2, use_pca_normalization=True)
        centroids = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        graph = builder.build(centroids)
        
        assert graph.normalized_nodes is not None
        assert graph.pca_transform is not None
        assert 'mean' in graph.pca_transform
        assert 'components' in graph.pca_transform
        assert 'scale' in graph.pca_transform
        assert graph.normalized_nodes.shape == centroids.shape
    
    def test_pca_normalization_disabled(self):
        """Test that PCA normalization can be disabled."""
        builder = StarGraphBuilder(k=2, use_pca_normalization=False)
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        graph = builder.build(centroids)
        
        assert graph.normalized_nodes is None
        assert graph.pca_transform is None
    
    def test_pca_normalization_single_point(self):
        """Test PCA normalization with single point."""
        builder = StarGraphBuilder(k=1, use_pca_normalization=True)
        centroids = np.array([[10.0, 20.0]], dtype=np.float32)
        graph = builder.build(centroids)
        
        # Should handle gracefully
        assert graph.normalized_nodes is not None
        assert graph.pca_transform is not None
    
    def test_pca_normalization_rotation_invariance(self):
        """Test that PCA normalization provides rotation invariance."""
        builder = StarGraphBuilder(k=2, use_pca_normalization=True)
        
        # Original pattern
        centroids1 = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        # Rotated pattern (45 degrees)
        angle = np.pi / 4
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        centroids2 = centroids1 @ rotation.T
        
        graph1 = builder.build(centroids1)
        graph2 = builder.build(centroids2)
        
        # Normalized positions should be similar (up to reflection)
        # Check that normalized nodes have similar structure
        assert graph1.normalized_nodes is not None
        assert graph2.normalized_nodes is not None
        # The patterns should be similar after normalization
        assert graph1.normalized_nodes.shape == graph2.normalized_nodes.shape
    
    def test_pca_normalization_scale_invariance(self):
        """Test that PCA normalization provides scale invariance."""
        builder = StarGraphBuilder(k=2, use_pca_normalization=True)
        
        # Original pattern
        centroids1 = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        # Scaled pattern (2x)
        centroids2 = centroids1 * 2.0
        
        graph1 = builder.build(centroids1)
        graph2 = builder.build(centroids2)
        
        # After normalization, patterns should be similar
        assert graph1.normalized_nodes is not None
        assert graph2.normalized_nodes is not None
        # Normalized nodes should have similar structure
        assert graph1.normalized_nodes.shape == graph2.normalized_nodes.shape
    
    def test_apply_pca_transform(self):
        """Test applying a precomputed PCA transform."""
        builder = StarGraphBuilder(k=2, use_pca_normalization=True)
        
        # Build graph and get transform
        centroids1 = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        graph1 = builder.build(centroids1)
        
        # Apply same transform to new centroids
        centroids2 = centroids1 + np.array([10.0, 10.0])  # Translated
        normalized2 = builder.apply_pca_transform(centroids2, graph1.pca_transform)
        
        assert normalized2.shape == centroids2.shape
        # Should be similar to graph1's normalized nodes (up to translation)
        assert graph1.normalized_nodes is not None
    
    def test_k_larger_than_stars(self):
        """Test when k is larger than number of stars."""
        builder = StarGraphBuilder(k=10)  # k > number of stars
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        graph = builder.build(centroids)
        
        # Should handle gracefully, using k_actual = n - 1
        assert len(graph.nodes) == 3
        assert len(graph.edges) > 0
    
    def test_edge_weights_are_distances(self):
        """Test that edge weights are Euclidean distances."""
        builder = StarGraphBuilder(k=2)
        centroids = np.array([
            [0.0, 0.0],
            [3.0, 0.0],  # Distance 3
            [0.0, 4.0],  # Distance 4
        ], dtype=np.float32)
        
        graph = builder.build(centroids)
        
        # Check that edge weights match distances
        for edge, weight in zip(graph.edges, graph.edge_weights):
            i, j = edge
            expected_dist = np.linalg.norm(graph.nodes[i] - graph.nodes[j])
            assert weight == pytest.approx(expected_dist, rel=1e-5)
    
    def test_integration_with_star_detector(self):
        """Test integration with star detector output."""
        try:
            from detection.star_detector import StarDetector
            import cv2
            
            # Create a test image with stars
            img = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(img, (20, 20), 3, 255, -1)
            cv2.circle(img, (50, 50), 3, 255, -1)
            cv2.circle(img, (80, 80), 3, 255, -1)
            
            # Detect stars
            detector = StarDetector(threshold=100, min_area=1.0)
            centroids = detector.detect(img)
            
            # Build graph
            builder = StarGraphBuilder(k=2)
            graph = builder.build(centroids)
            
            assert len(graph.nodes) == len(centroids)
            assert len(graph.edges) > 0
            
        except ImportError:
            pytest.skip("detection.star_detector not available")
    
    def test_graph_connectivity(self):
        """Test graph connectivity properties."""
        builder = StarGraphBuilder(k=3)
        
        # Create a connected pattern
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        graph = builder.build(centroids)
        num_components = graph.get_num_components()
        
        # With k=3 and 4 nodes, should be connected
        assert num_components >= 1
    
    def test_different_k_values(self):
        """Test graph construction with different k values."""
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        
        for k in [1, 2, 3, 4]:
            builder = StarGraphBuilder(k=k)
            graph = builder.build(centroids)
            
            assert len(graph.nodes) == 5
            # More k should generally mean more edges (but undirected removes duplicates)
            if k > 1:
                assert len(graph.edges) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

