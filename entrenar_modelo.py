import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# 1. CARGA DEL DATASET
print("Descargando dataset desde Kaggle...")
path = kagglehub.dataset_download("ighoshsubho/security-vulnerabilities-dataset")
csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csvs:
    raise FileNotFoundError("No se encontro ningun CSV en la descarga de Kaggle.")
df = pd.read_csv(csvs[0])
df.columns = [c.strip() for c in df.columns]
print("Columnas:", df.columns.tolist())

# 2. LIMPIEZA
initial_rows = len(df)
duplicados = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

target_candidates = [c for c in df.columns if "sever" in c.lower()]
if not target_candidates:
    raise ValueError(f"No se encontro columna de severidad. Columnas: {df.columns.tolist()}")
target_col = target_candidates[0]

df = df.dropna(subset=[target_col]).reset_index(drop=True)
conteo = df[target_col].astype(str).value_counts()
df = df[df[target_col].astype(str).isin(conteo[conteo >= 10].index)].reset_index(drop=True)
print(f"Filas iniciales: {initial_rows} | Duplicados: {duplicados} | Filas limpias: {len(df)}")
print(df[target_col].value_counts())

# 3. CARACTERISTICAS
# Texto: Title + Summary (lo que realmente describe la vulnerabilidad). Se descarta Link.
text_cols = [c for c in df.columns if c.lower() in ("title", "summary", "description")]
if not text_cols:
    raise ValueError("No se encontraron columnas de texto (Title/Summary).")
df["text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

# Fecha: solo anio y mes
date_cols = [c for c in df.columns if "date" in c.lower()]
num_features = []
if date_cols:
    d = pd.to_datetime(df[date_cols[0]], errors="coerce")
    df["Date_Year"] = d.dt.year
    df["Date_Month"] = d.dt.month
    num_features = ["Date_Year", "Date_Month"]

X = df[["text"] + num_features]
le = LabelEncoder()
y = le.fit_transform(df[target_col].astype(str))
class_names = list(le.classes_)

# 4. EDA
os.makedirs("graficos_eda", exist_ok=True)
plt.figure(figsize=(7, 4))
order = df[target_col].value_counts().index
sns.countplot(data=df, x=target_col, order=order, hue=target_col, palette="viridis", legend=False)
plt.title("Distribucion de Clases de Severidad")
plt.tight_layout()
plt.savefig("graficos_eda/1_distribucion_severidad.png")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

transformers = [("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                         min_df=2, stop_words="english",
                                         sublinear_tf=True), "text")]
if num_features:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]), num_features))
preprocessor = ColumnTransformer(transformers)

# 5. MODELOS (sin PCA: TF-IDF ya es esparso y PCA solo lo haria lento)
models = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                             tree_method="hist", eval_metric="mlogloss",
                             n_jobs=-1, random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
trained = {}

for name, clf in models.items():
    print(f"Entrenando: {name} ...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    cv = cross_validate(pipe, X_train, y_train, cv=skf, scoring=["f1_weighted"],
                        return_train_score=True, n_jobs=1)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    train_f1 = cv["train_f1_weighted"].mean()
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    results.append({
        "Modelo": name,
        "CV Val F1": round(cv["test_f1_weighted"].mean(), 4),
        "Test F1-Weighted": round(test_f1, 4),
        "Brecha Overfitting (%)": round((train_f1 - test_f1) * 100, 2),
    })
    trained[name] = pipe

df_results = pd.DataFrame(results).sort_values("Test F1-Weighted", ascending=False)
print("\n--- RESULTADOS COMPARATIVOS ---")
print(df_results.to_string(index=False))

best_name = df_results.iloc[0]["Modelo"]
best_pipe = trained[best_name]
print(f"\nMejor modelo: {best_name}")
print(classification_report(y_test, best_pipe.predict(X_test), target_names=class_names))

# 6. EXPORTAR
bundle = {
    "pipeline": best_pipe,
    "label_encoder": le,
    "num_features": num_features,
    "feature_defaults": {c: float(X_train[c].median()) for c in num_features},
    "metrics_summary": df_results.to_dict(orient="records"),
    "best_model_name": best_name,
}
joblib.dump(bundle, "vulnerability_model.joblib")
print("Modelo guardado en 'vulnerability_model.joblib'.")
print("Ejecuta la app con: python -m streamlit run app.py")