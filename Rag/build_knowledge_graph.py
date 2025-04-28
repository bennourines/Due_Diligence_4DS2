"""
Build a knowledge graph from FAISS index and metadata or from a JSON chunk file.
Nodes: chunk IDs; Edges: k-nearest neighbors based on embedding similarity.
Optionally, connect chunks sharing metadata features (e.g., same source).
"""
import os
import json
import pickle
import faiss
import networkx as nx
import numpy as np

# Define paths and settings manually
FAISS_INDEX_PATH = '../faiss_index_download/index.faiss'
METADATA_PATH = '../faiss_index_download/merged_metadata.json'
CHUNK_JSON_PATH = None  # Set this if using chunk JSON
OUTPUT_PATH = '../faiss_index_download/graph.pkl'
TOP_K_NEIGHBORS = 5

def load_faiss_index(index_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    return index

def load_metadata(metadata_path: str):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata

def load_chunk_json(chunk_json_path: str):
    if not os.path.exists(chunk_json_path):
        raise FileNotFoundError(f"Chunk JSON not found: {chunk_json_path}")
    with open(chunk_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

def build_graph_from_faiss(index_path: str, metadata_path: str, k: int = 5) -> nx.Graph:
    index = load_faiss_index(index_path)
    meta = load_metadata(metadata_path)
    ntotal = index.ntotal
    if ntotal != len(meta):
        print(f"Warning: index size {ntotal} != metadata length {len(meta)}")

    G = nx.Graph()
    for i, item in enumerate(meta):
        chunk_id = item.get('metadata', {}).get('chunk_id', str(i))
        G.add_node(chunk_id, **item)

    batch_size = 1024
    for start in range(0, ntotal, batch_size):
        end = min(ntotal, start + batch_size)
        emb_batch = np.zeros((end - start, index.d), dtype='float32')
        for j in range(start, end):
            emb_batch[j - start] = index.reconstruct(j)
        D, I = index.search(emb_batch, k + 1)
        for i_local, neighbors in enumerate(I):
            src_idx = start + i_local
            src_id = meta[src_idx]['metadata'].get('chunk_id', str(src_idx))
            for nbr_pos in neighbors:
                if nbr_pos == src_idx:
                    continue
                dst_id = meta[nbr_pos]['metadata'].get('chunk_id', str(nbr_pos))
                dist = float(D[i_local][np.where(neighbors == nbr_pos)][0])
                G.add_edge(src_id, dst_id, weight=1.0/(1.0+dist))
    return G

def build_graph_from_json(chunk_json_path: str, k: int = 5) -> nx.Graph:
    chunks = load_chunk_json(chunk_json_path)
    embeddings = np.array([item['embedding'] for item in chunks], dtype='float32')
    chunk_ids = [item['chunk_id'] for item in chunks]
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    G = nx.Graph()
    for item in chunks:
        node_id = item['chunk_id']
        attrs = {k: v for k, v in item.items() if k not in ('chunk_id', 'embedding')}
        G.add_node(node_id, **attrs)

    D, I = index.search(embeddings, k + 1)
    for i, neighbors in enumerate(I):
        src = chunk_ids[i]
        for j in neighbors:
            if j == i:
                continue
            dst = chunk_ids[j]
            dist = float(D[i][np.where(neighbors == j)][0])
            G.add_edge(src, dst, weight=1.0/(1.0+dist))
    return G

def save_graph(G: nx.Graph, output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Knowledge graph saved to {output_path}")

def main():
    if FAISS_INDEX_PATH and METADATA_PATH:
        G = build_graph_from_faiss(FAISS_INDEX_PATH, METADATA_PATH, TOP_K_NEIGHBORS)
    elif CHUNK_JSON_PATH:
        G = build_graph_from_json(CHUNK_JSON_PATH, TOP_K_NEIGHBORS)
    else:
        raise ValueError('Please set paths for either FAISS+metadata or chunk JSON.')

    save_graph(G, OUTPUT_PATH)

if __name__ == '__main__':
    main()
