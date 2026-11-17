import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# 1. CARGA DEL DATASET
csv_files = glob.glob("*.csv")
if not csv_files:
    raise FileNotFoundError("Coloca el archivo .csv de Kaggle en esta misma carpeta antes de ejecutar.")

file_path = csv_files[0]
print(f"Cargando dataset: {file_path}")
df = pd.read_csv(file_path)
initial_rows = len(df)

# 2. LIMPIEZA Y AUDITORÍA
duplicados = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

df.columns = [c.strip() for c in df.columns]
target_col = [c for c in df.columns if "sever" in c.lower()][0]

nulos_target = df[target_col].isna().sum()
df = df.dropna(subset=[target_col]).reset_index(drop=True)

date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
for dc in date_cols:
    df[dc] = pd.to_datetime(df[dc], errors="coerce")

num_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Filas iniciales: {initial_rows} | Duplicados eliminados: {duplicados} | Target nulo eliminado: {nulos_target} | Filas limpias: {len(df)}")

# 3. EDA (GUARDA GRÁFICOS EN CARPETA 'graficos_eda')
os.makedirs("graficos_eda", exist_ok=True)

plt.figure(figsize=(7, 4))
order = df[target_col].value_counts().index
sns.countplot(data=df, x=target_col, order=order, palette="viridis")
plt.title("Distribucion de Clases de Severidad")
plt.tight_layout()
plt.savefig("graficos_eda/1_distribucion_severidad.png")
plt.close()

if num_cols_raw:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=target_col, y=num_cols_raw[0], order=order, palette="Set2")
    plt.title(f"{num_cols_raw[0]} por Severidad")
    plt.tight_layout()
    plt.savefig("graficos_eda/2_bivariado_boxplot.png")
    plt.close()

if len(num_cols_raw) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[num_cols_raw].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matriz de Correlacion")
    plt.tight_layout()
    plt.savefig("graficos_eda/3_correlacion.png")
    plt.close()

# 4. INGENIERÍA DE CARACTERÍSTICAS
df_feat = df.copy()
for dc in date_cols:
    if pd.api.types.is_datetime64_any_dtype(df_feat[dc]):
        df_feat[f"{dc}_Year"] = df_feat[dc].dt.year
        df_feat[f"{dc}_Month"] = df_feat[dc].dt.month
        df_feat[f"{dc}_Days_Elapsed"] = (df_feat[dc].max() - df_feat[dc]).dt.days
        df_feat = df_feat.drop(columns=[dc])

cols_to_drop = [
    col for col in df_feat.columns
    if col != target_col and (("id" in col.lower() and "cwe" not in col.lower()) or (df_feat[col].dtype == "object" and df_feat[col].nunique() > 40))
]
df_feat = df_feat.drop(columns=cols_to_drop)

X = df_feat.drop(columns=[target_col])
y_raw = df_feat[target_col].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Análisis de PCA (95% varianza)
X_train_proc = preprocessor.fit_transform(X_train)
pca_full = PCA().fit(X_train_proc)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_95 = int(np.argmax(cum_var >= 0.95) + 1)
print(f"Features preprocesadas: {X_train_proc.shape[1]} | Componentes PCA (95% varianza): {n_comp_95}")

# 5. MODELADO Y VALIDACIÓN CRUZADA (K=5)
models = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, eval_metric="mlogloss", random_state=42)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
trained_pipelines = {}

for use_pca in [False, True]:
    tag = "Con PCA (95%)" if use_pca else "Sin PCA"
    for name, clf in models.items():
        steps = [("preprocessor", preprocessor)]
        if use_pca:
            steps.append(("pca", PCA(n_components=0.95, random_state=42)))
        steps.append(("classifier", clf))

        pipe = Pipeline(steps)
        cv_res = cross_validate(pipe, X_train, y_train, cv=skf, scoring=["accuracy", "f1_weighted"], return_train_score=True)

        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)

        train_f1 = cv_res["train_f1_weighted"].mean()
        val_f1 = cv_res["test_f1_weighted"].mean()
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1_w = f1_score(y_test, y_pred_test, average="weighted")
        test_f1_m = f1_score(y_test, y_pred_test, average="macro")
        overfit_gap = (train_f1 - test_f1_w) * 100

        results.append({
            "Modelo": name,
            "Configuracion": tag,
            "CV Val F1": round(val_f1, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test F1-Weighted": round(test_f1_w, 4),
            "Test F1-Macro": round(test_f1_m, 4),
            "Brecha Overfitting (%)": round(overfit_gap, 2)
        })
        trained_pipelines[f"{name} | {tag}"] = pipe

df_results = pd.DataFrame(results).sort_values(by="Test F1-Weighted", ascending=False)
print("\n--- RESULTADOS COMPARATIVOS ---")
print(df_results.to_string(index=False))

# 6. EXPORTAR MEJOR MODELO PARA STREAMLIT
best_row = df_results.iloc[0]
best_key = f"{best_row['Modelo']} | {best_row['Configuracion']}"
best_pipeline = trained_pipelines[best_key]

y_pred_best = best_pipeline.predict(X_test)
print(f"\nMejor modelo: {best_key}")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Matriz de Confusion - {best_key}")
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("graficos_eda/4_matriz_confusion.png")
plt.close()

model_bundle = {
    "pipeline": best_pipeline,
    "label_encoder": le,
    "num_features": num_features,
    "cat_features": cat_features,
    "cat_options": {col: sorted(X_train[col].dropna().unique().tolist()) for col in cat_features},
    "feature_ranges": {col: (float(X_train[col].min()), float(X_train[col].max()), float(X_train[col].median())) for col in num_features},
    "metrics_summary": df_results.to_dict(orient="records"),
    "best_model_name": best_key
}

joblib.dump(model_bundle, "vulnerability_model.joblib")
print("Modelo guardado en 'vulnerability_model.joblib'. Ya puedes ejecutar: streamlit run app.py")
