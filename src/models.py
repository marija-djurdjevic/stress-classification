from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

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

def run_svm_baseline_extreme(df_encoded):
    """
    Ekstremni eksperiment: Izbacujemo sve atribute za koje smo 
    dokazale da se koriste za generisanje sintetičkih labela.
    """
    cols_to_drop = ['ID', 'Stress_Level', 'Coffee_Intake', 'Sleep_Quality', 
                    'Sleep_Hours', 'Gender', 'Country',
                    'Age', 'BMI', 'Health_Issues']
    
    X = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns], errors='ignore')
    y = df_encoded['Stress_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf', random_state=42)
    
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    return svm_model, acc, f1, cm, cv_scores, cv_scores.mean()



def run_svm_robust_pipeline(df_encoded):
    """
    Trenira SVM model koristeći scikit-learn Pipeline za bezbednu 
    imputaciju nedostajućih vrednosti i skaliranje podataka.
    """
    cols_to_drop = ['ID', 'Stress_Level', 'Coffee_Intake', 'Sleep_Quality']
    
    X = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns], errors='ignore')
    y = df_encoded['Stress_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  
        ('scaler', StandardScaler()),            
        ('svm', SVC(kernel='rbf', random_state=42)) 
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    return pipeline, acc, f1, cm, cv_scores, cv_scores.mean()


def run_svm_baseline(df_encoded):
    """
    Priprema podatke, vrši 70:30 podelu i trenira SVM baseline model.
    """
    cols_to_drop = ['ID', 'Stress_Level', 'Coffee_Intake', 'Sleep_Quality', 
                    'Gender', 'Country', 'Alcohol_Consumption', 'Smoking', 'Health_Issues', 'Occupation']
    
    X = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns], errors='ignore')
    y = df_encoded['Stress_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf', random_state=42)
    
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    return svm_model, acc, f1, cm, cv_scores, cv_scores.mean()

def run_svm_smote_pipeline(df_encoded):
    """Robusni SVM za realistične, pokvarene podatke (KT2)"""
    cols_to_drop = ['ID', 'Stress_Level', 'Coffee_Intake', 'Sleep_Quality']
    X = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns], errors='ignore')
    y = df_encoded['Stress_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipeline = ImbPipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  
        ('smote', SMOTE(random_state=42)),       
        ('scaler', StandardScaler()),            
        ('svm', SVC(kernel='rbf', random_state=42)) 
    ])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return pipeline, acc, f1, cm, cv_scores, cv_scores.mean()