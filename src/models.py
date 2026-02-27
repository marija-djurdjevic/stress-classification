from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler 

def run_kmeans_baseline(df_encoded):
    """
    Pokreće K-Means klasterovanje (baseline) koristeći odabrane atribute.
    Vraća Silhouette rezultat, dodeljene klastere, istrenirani model i scaler.
    """
    X_cluster = df_encoded[['Caffeine_mg', 'Sleep_Hours']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    klasteri = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, klasteri)
    
    return score, klasteri, kmeans, scaler, X_cluster