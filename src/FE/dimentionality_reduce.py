"""
Dimensionality Reduction Algorithms

Key Details:
-------------
PCA: Works well for linear dimensionality reduction. Outputs components with the highest variance.
t-SNE: Effective for visualization in 2D/3D but less suitable for large datasets due to high computation time.
UMAP: Scalable and fast, preserving both local and global structures. Great for visualization and clustering tasks.
LDA: Supervised; reduces dimensions while maximizing class separability.
SVD: Similar to PCA but works with sparse matrices.
Autoencoders: Neural network-based, customizable for nonlinear dimensionality reduction.

You can choose a method based on the dataset size, the problem, and the purpose (e.g., visualization or preprocessing).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as umap
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def dimensionality_reduction(X, method="pca", n_components=2, y=None):
    """
    Apply dimensionality reduction to a dataset.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        method (str): Dimensionality reduction method ('pca', 'tsne', 'umap', 'lda', 'svd', 'autoencoder').
        n_components (int): Number of components to reduce to.
        y (np.ndarray): Target values (required for 'lda').

    Returns:
        np.ndarray: Transformed dataset with reduced dimensions.
    """
    try:
        if method == "pca":
            # Principal Component Analysis
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(X)
            print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

        elif method == "tsne":
            # t-SNE
            tsne = TSNE(n_components=n_components, random_state=42)
            reduced_data = tsne.fit_transform(X)

        elif method == "umap":
            # UMAP
            umap_model = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = umap_model.fit_transform(X)

        elif method == "lda":
            if y is None:
                raise ValueError("y (target) is required for LDA.")
            # Linear Discriminant Analysis
            lda = LDA(n_components=n_components)
            reduced_data = lda.fit_transform(X, y)

        elif method == "svd":
            # Singular Value Decomposition
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_data = svd.fit_transform(X)

        elif method == "autoencoder":
            # Autoencoder
            input_dim = X.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(n_components, activation="relu")(input_layer)
            decoded = Dense(input_dim, activation="sigmoid")(encoded)
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            encoder = Model(inputs=input_layer, outputs=encoded)
            autoencoder.compile(optimizer="adam", loss="mse")
            autoencoder.fit(X, X, epochs=50, batch_size=32, verbose=0)
            reduced_data = encoder.predict(X)

        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        return reduced_data

    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        return None


# Example Usage
if __name__ == "__main__":
    # Sample data
    from sklearn.datasets import load_iris

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Apply dimensionality reduction
    methods = ["pca", "tsne", "umap", "lda", "svd", "autoencoder"]
    for method in methods:
        print(f"\nDimensionality Reduction Method: {method}")
        if method == "lda":
            reduced = dimensionality_reduction(X, method=method, n_components=2, y=y)
        else:
            reduced = dimensionality_reduction(X, method=method, n_components=2)
        print(f"Reduced Shape: {reduced.shape}")
