# Baseline Design

This project uses eight baselines to compare clustering behavior across lexical,
classical dense, pretrained neural, and contrastive multilingual embeddings.

1. **TF-IDF + K-Means** is the lexical lower-bound baseline. It measures how far
   sparse word-level overlap can go without neural sentence representations.

2. **TF-IDF + SVD + K-Means** is a classical dense semantic baseline. It applies
   latent semantic analysis to TF-IDF vectors before centroid clustering.

3. **MiniLM + K-Means** is a lightweight multilingual sentence-transformer
   baseline. It tests whether compact pretrained sentence embeddings improve
   clustering over lexical features.

4. **MiniLM + Agglomerative** keeps the MiniLM embedding fixed and changes the
   clustering algorithm. This isolates the effect of hierarchical clustering
   from the embedding model.

5. **mBERT + K-Means** uses raw multilingual transformer encoder embeddings
   with centroid clustering. It is a direct neural encoder baseline without
   supervised contrastive fine-tuning.

6. **mBERT + DBSCAN** is the closest control baseline to the proposed method
   because it uses the same base encoder family and a density-based clustering
   algorithm.

7. **LaBSE + K-Means** is a strong cross-lingual sentence embedding baseline.
   It tests whether a pretrained model built for cross-lingual sentence
   alignment performs better than general mBERT embeddings.

8. **mSimCSE + DBSCAN** is a contrastive sentence embedding benchmark. It
   compares the proposed supervised contrastive fine-tuning against an existing
   contrastive sentence representation approach when a compatible model is
   provided.

The comparison covers four main dimensions:

- Lexical vs neural embeddings.
- General multilingual encoders vs cross-lingual sentence embeddings.
- K-Means vs DBSCAN vs Agglomerative clustering.
- Zero-shot pretrained embeddings vs supervised contrastive fine-tuning.

The goal is fair comparison, not artificial improvement. If a baseline performs
poorly or cannot run because a model is unavailable locally, that result should
be reported transparently in `outputs/metrics/baseline_summary.csv`.
