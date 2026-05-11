from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score

def evaluate_kmeans(embeddings, true_labels, n_clusters=3):
    """Runs KMeans clustering and returns evaluation metrics."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
    
    sil_score = silhouette_score(embeddings, cluster_labels)
    homog_score = homogeneity_score(true_labels, cluster_labels)
    
    return {
        "silhouette_score": sil_score,
        "homogeneity_score": homog_score
    }
