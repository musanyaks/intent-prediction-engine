"""User and product embedding generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserEmbeddingModel(nn.Module):
    """Simple neural network to learn user embeddings from behavior."""

    def __init__(self, num_categories: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_user_embeddings(
    user_category_matrix: pd.DataFrame, embedding_dim: int = 128, epochs: int = 100
) -> pd.DataFrame:
    """
    Train user embeddings from category affinity matrix using autoencoder-style training.
    """
    from sklearn.decomposition import TruncatedSVD

    # Use TruncatedSVD as a fast, interpretable baseline
    category_cols = [c for c in user_category_matrix.columns if c != "user_id"]
    X = user_category_matrix[category_cols].values

    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    embeddings = svd.fit_transform(X)

    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"user_emb_{i}" for i in range(embedding_dim)],
    )
    embedding_df["user_id"] = user_category_matrix["user_id"].values

    explained = sum(svd.explained_variance_ratio_)
    logger.info(f"User embeddings trained. Explained variance: {explained:.3f}")
    return embedding_df


def compute_product_embeddings(product_catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Compute or fetch pre-trained product embeddings.
    Assumes catalog has 'embedding_vector' column or generates from text.
    """
    if "embedding_vector" in product_catalog.columns:
        logger.info("Using pre-computed product embeddings")
        return product_catalog[["product_id", "embedding_vector"]]

    # Fallback: simple TF-IDF on product name
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=128, stop_words="english")
    tfidf = vectorizer.fit_transform(product_catalog["product_name"].fillna(""))
    embedding_df = pd.DataFrame(
        tfidf.toarray(),
        columns=[f"prod_emb_{i}" for i in range(tfidf.shape[1])],
    )
    embedding_df["product_id"] = product_catalog["product_id"].values

    logger.info(f"Computed TF-IDF product embeddings: {tfidf.shape[1]} dimensions")
    return embedding_df