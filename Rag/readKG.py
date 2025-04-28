import networkx as nx
import pickle

# Path to the .pkl file
pkl_file_path = "../faiss_index_download/graph.pkl"

try:
    # Use the pickle module to load the graph directly
    with open(pkl_file_path, "rb") as file:
        knowledge_graph = pickle.load(file)
    
    # Ensure the loaded object is a NetworkX graph
    if not isinstance(knowledge_graph, nx.Graph):
        raise ValueError("The loaded object is not a NetworkX graph.")
    
    # Print basic information about the graph
    print(f"Type of the loaded object: {type(knowledge_graph)}")
    print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
    print(f"Number of edges: {knowledge_graph.number_of_edges()}")
    
    # Inspect the first few nodes and their attributes (if any)
    print("\nSample nodes and attributes:")
    for node, attributes in list(knowledge_graph.nodes(data=True))[:5]:
        print(f"Node: {node}, Attributes: {attributes}")
    
    # Inspect the first few edges and their attributes (if any)
    print("\nSample edges and attributes:")
    for edge in list(knowledge_graph.edges(data=True))[:5]:
        print(f"Edge: {edge}")

except FileNotFoundError:
    print(f"Error: The file '{pkl_file_path}' was not found.") [[2]]
except Exception as e:
    print(f"An error occurred while reading the .pkl file: {str(e)}") [[4]]