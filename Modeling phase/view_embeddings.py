import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def view_embeddings():
    # Load the embeddings
    embeddings = np.load('vector_db_export/embeddings.npy')
    
    print("\n=== Embeddings Information ===")
    print(f"Shape of embeddings array: {embeddings.shape}")
    print(f"Number of dimensions per embedding: {embeddings.shape[1]}")
    print(f"Total number of embeddings: {embeddings.shape[0]}")
    
    # Load the corresponding metadata from CSV
    df = pd.read_csv('vector_db_export/database.csv')
    
    while True:
        print("\n=== Embeddings Viewer ===")
        print("1. View sample embeddings")
        print("2. View embedding statistics")
        print("3. Visualize embeddings in 2D")
        print("4. Visualize embeddings in 3D")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            # View sample embeddings
            num_samples = int(input("\nHow many embeddings to view? (1-5): "))
            num_samples = min(max(1, num_samples), 5)
            
            print("\nSample Embeddings:")
            for i in range(num_samples):
                print(f"\n--- Embedding {i+1} ---")
                print(f"Source: {df['source'].iloc[i]}")
                print(f"First 10 dimensions: {embeddings[i][:10]}")
                print(f"Content preview: {df['content'].iloc[i][:100]}...")
        
        elif choice == "2":
            # View statistics
            print("\nEmbedding Statistics:")
            print(f"Mean: {np.mean(embeddings):.4f}")
            print(f"Standard Deviation: {np.std(embeddings):.4f}")
            print(f"Min value: {np.min(embeddings):.4f}")
            print(f"Max value: {np.max(embeddings):.4f}")
            
            # Show distribution of values
            plt.figure(figsize=(10, 6))
            plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
            plt.title('Distribution of Embedding Values')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig('visualizations/embeddings_distribution.png')
            plt.close()
            print("\nDistribution plot saved to visualizations/embeddings_distribution.png")
        
        elif choice == "3":
            # Visualize in 2D
            print("\nReducing dimensions for visualization...")
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 8))
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
            plt.title('Embeddings in 2D Space (PCA)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            
            # Add some sample points with labels
            sample_indices = np.random.choice(len(embeddings), 5, replace=False)
            for idx in sample_indices:
                plt.annotate(df['source'].iloc[idx], 
                           (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]),
                           fontsize=8)
            
            plt.grid(True, alpha=0.3)
            plt.savefig('visualizations/embeddings_2d.png')
            plt.close()
            print("\n2D visualization saved to visualizations/embeddings_2d.png")

        elif choice == "4":
            # Visualize in 3D
            print("\nReducing dimensions for 3D visualization...")
            pca = PCA(n_components=3)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Create 3D plot
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create scatter plot
            scatter = ax.scatter(reduced_embeddings[:, 0], 
                               reduced_embeddings[:, 1], 
                               reduced_embeddings[:, 2],
                               alpha=0.6,
                               c=np.arange(len(embeddings)),  # Color by index for variety
                               cmap='viridis')
            
            # Add labels
            ax.set_title('Embeddings in 3D Space (PCA)', fontsize=14, pad=20)
            ax.set_xlabel('Principal Component 1', fontsize=12, labelpad=10)
            ax.set_ylabel('Principal Component 2', fontsize=12, labelpad=10)
            ax.set_zlabel('Principal Component 3', fontsize=12, labelpad=10)
            
            # Add colorbar
            plt.colorbar(scatter, label='Document Index')
            
            # Add sample labels
            sample_indices = np.random.choice(len(embeddings), 5, replace=False)
            for idx in sample_indices:
                ax.text(reduced_embeddings[idx, 0], 
                       reduced_embeddings[idx, 1], 
                       reduced_embeddings[idx, 2],
                       df['source'].iloc[idx],
                       fontsize=8)
            
            # Adjust the view
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('visualizations/embeddings_3d.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n3D visualization saved to visualizations/embeddings_3d.png")
            print("Note: The 3D plot shows the first three principal components of the embeddings")
            print("Colors indicate document indices, and some points are labeled with their source")
        
        elif choice == "5":
            print("Exiting embeddings viewer...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    view_embeddings() 