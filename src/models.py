from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_baseline(df_encoded):
    """
    Pokreće K-Means klasterovanje (baseline) koristeći odabrane atribute.
    Vraća Silhouette rezultat (mjeru kvaliteta klastera).
    """
    X_cluster = df_encoded[['Caffeine_mg', 'Sleep_Hours']]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    klasteri = kmeans.fit_predict(X_cluster)
    
    score = silhouette_score(X_cluster, klasteri)
    
    return score