import pandas as pd
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