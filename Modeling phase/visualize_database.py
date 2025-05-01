import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from text_processing_pipeline import VectorDatabase
import os
import pandas as pd

def export_database(vector_db, output_dir="vector_db_export"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with embeddings, metadata, and documents
    data = {
        'source': [meta['source'] for meta in vector_db.metadata],
        'content': vector_db.documents,
        'embedding': [vector_db.index._fit_X[i] for i in range(len(vector_db.documents))]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save embeddings separately as numpy array
    np.save(os.path.join(output_dir, 'embeddings.npy'), vector_db.index._fit_X)
    
    # Save metadata and content as CSV
    df.to_csv(os.path.join(output_dir, 'database.csv'), index=False)
    
    print(f"\nDatabase exported to {output_dir}/")
    print(f"- embeddings.npy: Contains all embeddings")
    print(f"- database.csv: Contains source, content, and embedding references")
    print(f"Total records exported: {len(df)}")

def visualize_database(db_path="vector_db"):
    # Load the vector database
    vector_db = VectorDatabase()
    vector_db.load(db_path)
    
    # Export the database
    export_database(vector_db)
    
    # Get embeddings and metadata
    embeddings = vector_db.index._fit_X
    metadata = vector_db.metadata
    
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create a scatter plot
    plt.figure(figsize=(15, 10))
    
    # Get unique sources for coloring
    sources = [meta['source'] for meta in metadata]
    unique_sources = list(set(sources))[:20]  # Limit to 20 sources for better visualization
    
    # Create a color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))
    
    # Plot points for each source
    for i, source in enumerate(unique_sources):
        mask = [s == source for s in sources]
        plt.scatter(reduced_embeddings[mask, 0], 
                   reduced_embeddings[mask, 1],
                   c=[colors[i]], 
                   label=os.path.splitext(source)[0],  # Remove .txt extension
                   alpha=0.6,
                   s=50)  # Increase point size
    
    plt.title('Document Chunks in 2D Space (PCA)\nShowing top 20 whitepapers', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add legend with smaller font
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add some sample text annotations for interesting points
    # Find points that are far from the center
    distances = np.linalg.norm(reduced_embeddings, axis=1)
    interesting_indices = np.argsort(distances)[-5:]  # Get 5 most distant points
    
    for idx in interesting_indices:
        x, y = reduced_embeddings[idx]
        text = vector_db.documents[idx][:50] + '...' if len(vector_db.documents[idx]) > 50 else vector_db.documents[idx]
        source = metadata[idx]['source']
        plt.annotate(f"{os.path.splitext(source)[0]}\n{text}", 
                    (x, y),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/vector_database_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to visualizations/vector_database_visualization.png")
    print(f"Number of documents visualized: {len(embeddings)}")
    print(f"Number of unique sources: {len(unique_sources)}")
    print("\nVisualization shows the distribution of document chunks in 2D space.")
    print("Each point represents a chunk of text, colored by its source whitepaper.")
    print("The plot shows how different chunks relate to each other semantically.")
    print("Chunks that are closer together in the plot have more similar content.")

if __name__ == "__main__":
    visualize_database() 