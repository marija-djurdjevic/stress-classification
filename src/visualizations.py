import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")

def plot_stress_distribution(df):
    """Prikazuje distribuciju klasa stresa (Low, Medium, High)."""
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Stress_Level', order=['Low', 'Medium', 'High'])
    plt.title('Distribucija nivoa stresa kod korisnika', fontsize=14)
    plt.xlabel('Nivo stresa', fontsize=12)
    plt.ylabel('Broj korisnika', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_caffeine_vs_sleep(df):
    """Scatter plot: Unos kofeina vs Sati sna, obojeno po nivou stresa."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Caffeine_mg', y='Sleep_Hours', hue='Stress_Level', 
                    palette='coolwarm', alpha=0.7)
    plt.title('Odnos unosa kofeina i dužine sna u zavisnosti od stresa', fontsize=14)
    plt.xlabel('Kofein (mg)', fontsize=12)
    plt.ylabel('Sati sna', fontsize=12)
    plt.legend(title='Stres')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df_encoded):
    """Heatmap korelacije za sve atribute u skupu podataka."""
    plt.figure(figsize=(14, 10))
    
    df_for_corr = df_encoded.drop(columns=['ID'], errors='ignore')
    corr = df_for_corr.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f", 
                annot_kws={"size": 9}, linewidths=.5)
    plt.title('Korelaciona matrica svih atributa', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\nKorelacija atributa sa nivoom stresa (Stress_Level):")
    stres_korelacija = corr['Stress_Level'].sort_values(ascending=False)
    print(stres_korelacija)

def plot_kmeans_clusters(df_original, klasteri, kmeans, scaler):
    """Vizuelizacija K-Means klastera sa obeleženim centroidima."""
    
    centroidi_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(data=df_original, x='Caffeine_mg', y='Sleep_Hours', 
                    hue=klasteri, palette='viridis', alpha=0.7)
    
    plt.scatter(centroidi_original[:, 0], centroidi_original[:, 1], 
                c='red', s=200, marker='X', label='Centroidi klastera', linewidths=3)
    
    plt.title('Spontano grupisanje podataka: K-Means Klasteri (k=3)', fontsize=14)
    plt.xlabel('Kofein (mg)', fontsize=12)
    plt.ylabel('Sati sna', fontsize=12)
    plt.legend(title='K-Means Klaster')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm):
    """Prikazuje matricu konfuzije za model klasifikacije."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'], 
                yticklabels=['Low', 'Medium', 'High'])
    plt.title('Matrica konfuzije - SVM Baseline', fontsize=14)
    plt.xlabel('Predviđena klasa (Predicted)', fontsize=12)
    plt.ylabel('Stvarna klasa (Actual)', fontsize=12)
    plt.tight_layout()
    plt.show()