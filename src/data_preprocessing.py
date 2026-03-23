import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath):
    """Učitava CSV i radi osnovnu proveru nedostajućih vrednosti."""
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df

def encode_features(df):
    """Enkodira kategoričke varijable u numeričke."""
    df_encoded = df.copy()
    
    sleep_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    df_encoded['Sleep_Quality'] = df_encoded['Sleep_Quality'].map(sleep_mapping)
    
    stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_encoded['Stress_Level'] = df_encoded['Stress_Level'].map(stress_mapping)
    
    categorical_cols = ['Gender', 'Country', 'Health_Issues', 'Occupation']
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
    return df_encoded

def scale_features(df_encoded, features_to_scale):
    """Skalira numeričke vrednosti."""
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    return df_scaled

def add_real_world_noise(df, missing_pct=0.05, noise_std=0.1, label_noise_pct=0.1):
    """
    Kvari dataset simulirajući probleme iz stvarnog sveta:
    1. Gausov šum u senzorima
    2. Nedostajuće vrednosti (NaN)
    3. Greške u subjektivnim anketama (Label Noise)
    """
    df_noisy = df.copy()
    np.random.seed(42) 

    # 1. Gausov šum 
    num_cols = ['Caffeine_mg', 'Heart_Rate', 'BMI', 'Sleep_Hours']
    for col in num_cols:
        if col in df_noisy.columns:
            std_dev = df_noisy[col].std()
            noise = np.random.normal(0, std_dev * noise_std, size=len(df_noisy))
            df_noisy[col] = df_noisy[col] + noise
            df_noisy[col] = df_noisy[col].apply(lambda x: max(x, 0))

    # 2. Nedostajući podaci 
    cols_for_nan = ['Heart_Rate', 'Physical_Activity_Hours', 'Sleep_Quality']
    for col in cols_for_nan:
        if col in df_noisy.columns:
            mask = np.random.rand(len(df_noisy)) < missing_pct
            df_noisy.loc[mask, col] = np.nan

    # 3. Šum u labelama 
    if 'Stress_Level' in df_noisy.columns:
        n_label_noise = int(len(df_noisy) * label_noise_pct)
        noise_indices = np.random.choice(df_noisy.index, size=n_label_noise, replace=False)

        is_numeric = pd.api.types.is_numeric_dtype(df_noisy['Stress_Level'])
        classes = [0, 1, 2] if is_numeric else ['Low', 'Medium', 'High']

        for idx in noise_indices:
            current_label = df_noisy.loc[idx, 'Stress_Level']
            other_labels = [c for c in classes if c != current_label]
            df_noisy.loc[idx, 'Stress_Level'] = np.random.choice(other_labels)

    return df_noisy