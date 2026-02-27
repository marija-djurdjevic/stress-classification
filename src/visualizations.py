import matplotlib.pyplot as plt
import seaborn as sns

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
    """Heatmap korelacije između numeričkih atributa."""
    plt.figure(figsize=(12, 8))
    cols = ['Caffeine_mg', 'Sleep_Hours', 'Sleep_Quality', 'Heart_Rate', 
            'Physical_Activity_Hours', 'Stress_Level']
    corr = df_encoded[cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Korelaciona matrica ključnih faktora', fontsize=14)
    plt.tight_layout()
    plt.show()