import networkx as nx
import matplotlib.pyplot as plt
import pickle
from itertools import islice
from sklearn.metrics import precision_score, recall_score

# Load your knowledge graph
def load_kg(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    

# Structural Metrics 
def analyze_structure(G):
    print("=== Structural Analysis ===")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {sum(degrees)/len(degrees):.2f}")
    
    # Density
    print(f"Density: {nx.density(G):.4f}")
    
    # Clustering coefficient
    print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    # Path length (use largest connected component for disconnected graphs) [[6]][[10]]
    if not nx.is_connected(G):
        print("Graph is disconnected. Analyzing largest connected component...")
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
    else:
        G_connected = G
    
    print(f"Average shortest path length (largest component): {nx.average_shortest_path_length(G_connected):.2f}")

def analyze_structure2(G):
    print("=== Structural Analysis ===")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {sum(degrees)/len(degrees):.2f}")
    
    # Density
    print(f"Density: {nx.density(G):.4f}")
    
    # Clustering coefficient
    print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    # Path length (sample 100 nodes for large graphs)
    sample_nodes = list(islice(G.nodes, 100))
    print(f"Average shortest path length (sample): {nx.average_shortest_path_length(G.subgraph(sample_nodes)):.2f}")

# Semantic Metrics (using synthetic ground truth) [[6]]
def semantic_evaluation(G):
    print("\n=== Semantic Evaluation ===")
    
    # Synthetic ground truth (example)
    ground_truth = {
        'edges': [('A', 'B'), ('B', 'C')],
        'categories': {'A': 'tech', 'B': 'finance', 'C': 'health'}
    }
    
    # Precision/Recall simulation
    predicted_edges = list(G.edges())[:2]  # First 2 edges for demo
    y_true = [1 if e in ground_truth['edges'] else 0 for e in predicted_edges]
    y_pred = [1]*len(predicted_edges)
    
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")

# Link Prediction [[2]]
def link_prediction(G):
    print("\n=== Link Prediction ===")
    test_node = list(G.nodes())[0]
    print(f"Predictions for node {test_node}:")
    
    # Use common neighbors algorithm
    preds = nx.common_neighbor_centrality(G, [(test_node, n) for n in G.nodes()])
    for u, v, p in preds:
        print(f"({u}, {v}): {p:.2f}")

# Entity Categorization [[5]]
def categorize_entities(G):
    print("\n=== Entity Categorization ===")
    categories = {'tech': 0, 'finance': 0, 'health': 0}
    
    for node in G.nodes(data=True):
        text = node[1].get('text', '').lower()
        if 'blockchain' in text:
            categories['tech'] += 1
        elif 'token' in text:
            categories['finance'] += 1
        elif 'risk' in text:
            categories['health'] += 1
            
    print(f"Category distribution: {categories}")

# Consistency Check [[8]]
def check_consistency(G):
    print("\n=== Consistency Check ===")
    conflicts = 0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) != len(set(neighbors)):
            conflicts += 1
    print(f"Nodes with duplicate relationships: {conflicts}")

# Visualization [[5]]
def visualize_kg(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_size=50, alpha=0.8)
    plt.show()

if __name__ == "__main__":
    kg = load_kg("../../faiss_index_download/graph.pkl")
    
    analyze_structure(kg)
    semantic_evaluation(kg)
    link_prediction(kg)
    categorize_entities(kg)
    check_consistency(kg)
    visualize_kg(kg)