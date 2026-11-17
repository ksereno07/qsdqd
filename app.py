import json
import re
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Escáner de repositorios (simulación)", page_icon="🛡️", layout="wide")

MAX_VULNS = 40  # limite para que la demo sea rapida


@st.cache_resource
def cargar_modelo():
    ruta = Path(__file__).parent / "vulnerability_model.joblib"
    if not ruta.exists():
        st.error("No se encontró 'vulnerability_model.joblib'. Súbelo al repositorio junto a app.py.")
        st.stop()
    return joblib.load(ruta)


def parsear_url(url):
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        return None
    return m.group(1), m.group(2).removesuffix(".git")


def descargar(owner, repo, ruta):
    r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{ruta}", timeout=15)
    return r.text if r.status_code == 200 else None


def leer_requirements(texto):
    paquetes = []
    for linea in texto.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)", linea)
        if m:
            paquetes.append((m.group(1), m.group(2), "PyPI"))
    return paquetes


def leer_package_json(texto):
    paquetes = []
    try:
        data = json.loads(texto)
    except json.JSONDecodeError:
        return paquetes
    for seccion in ("dependencies", "devDependencies"):
        for nombre, ver in data.get(seccion, {}).items():
            m = re.search(r"\d+(\.\d+)*", str(ver))
            if m:
                paquetes.append((nombre, m.group(0), "npm"))
    return paquetes


def consultar_osv(paquetes):
    queries = [{"package": {"name": n, "ecosystem": e}, "version": v} for n, v, e in paquetes]
    hallazgos = []
    for i in range(0, len(queries), 100):
        lote = queries[i:i + 100]
        r = requests.post("https://api.osv.dev/v1/querybatch", json={"queries": lote}, timeout=30)
        r.raise_for_status()
        for j, res in enumerate(r.json().get("results", [])):
            for v in res.get("vulns", []):
                hallazgos.append((paquetes[i + j], v["id"]))
    return hallazgos


def detalle_vuln(vid):
    r = requests.get(f"https://api.osv.dev/v1/vulns/{vid}", timeout=20)
    return r.json() if r.status_code == 200 else {}


def version_corregida(vuln, nombre):
    for a in vuln.get("affected", []):
        if a.get("package", {}).get("name", "").lower() == nombre.lower():
            for rg in a.get("ranges", []):
                for ev in rg.get("events", []):
                    if "fixed" in ev:
                        return ev["fixed"]
    return None


bundle = cargar_modelo()
pipeline = bundle["pipeline"]
le = bundle["label_encoder"]
num_features = bundle["num_features"]
defaults = bundle.get("feature_defaults", {})

st.title("🛡️ Escáner de repositorios GitHub (simulación)")
st.caption(f"Severidad estimada con el modelo entrenado: {bundle['best_model_name']}. "
           "Las vulnerabilidades conocidas vienen de OSV.dev.")

url = st.text_input("URL del repositorio", placeholder="https://github.com/usuario/repositorio")

if st.button("Escanear") and url:
    datos = parsear_url(url)
    if not datos:
        st.error("No parece una URL de GitHub válida.")
        st.stop()
    owner, repo = datos

    with st.spinner("Descargando archivos de dependencias..."):
        paquetes = []
        encontrados = []
        for ruta, lector in (("requirements.txt", leer_requirements), ("package.json", leer_package_json)):
            texto = descargar(owner, repo, ruta)
            if texto:
                encontrados.append(ruta)
                paquetes += lector(texto)

    if not encontrados:
        st.warning("No encontré requirements.txt ni package.json en la raíz del repositorio "
                   "(o el repo es privado).")
        st.stop()

    st.write(f"Archivos analizados: {', '.join(encontrados)} | Dependencias con versión fija: {len(paquetes)}")
    if not paquetes:
        st.info("Las dependencias no tienen versión fija (==), así que no se pueden comprobar.")
        st.stop()

    with st.spinner("Consultando base de vulnerabilidades..."):
        try:
            hallazgos = consultar_osv(paquetes)[:MAX_VULNS]
        except requests.RequestException as e:
            st.error(f"No pude consultar OSV.dev: {e}")
            st.stop()

        filas = []
        for (nombre, version, eco), vid in hallazgos:
            v = detalle_vuln(vid)
            resumen = v.get("summary") or (v.get("details", "")[:300])
            texto = f"{resumen} {v.get('details', '')[:500]}".strip()
            fijo = version_corregida(v, nombre)
            filas.append({
                "Paquete": nombre,
                "Versión actual": version,
                "ID": vid,
                "Descripción": resumen,
                "_texto": texto,
                "Solución": f"Actualizar {nombre} a {fijo} o superior" if fijo
                            else "Revisar el aviso y buscar una versión más reciente",
            })

    if not filas:
        st.success("No se encontraron vulnerabilidades conocidas en las dependencias analizadas. ✅")
        st.stop()

    df = pd.DataFrame(filas)
    entrada = pd.DataFrame({"text": df["_texto"]})
    for c in num_features:
        entrada[c] = defaults.get(c)
    df["Severidad (modelo)"] = le.inverse_transform(pipeline.predict(entrada[["text"] + num_features]))
    df = df.drop(columns=["_texto"])

    st.error(f"Se encontraron {len(df)} vulnerabilidades")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df["Severidad (modelo)"].value_counts())

with st.expander("Métricas de los modelos entrenados"):
    st.dataframe(pd.DataFrame(bundle["metrics_summary"]))