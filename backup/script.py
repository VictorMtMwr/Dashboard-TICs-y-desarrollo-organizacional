
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
from pyvis.network import Network
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import multiprocessing as mp
import numpy as np
from collections import Counter, defaultdict
import math
import time
import re
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# --------------------------
# Configuración Streamlit
# --------------------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Analítica Bibliométrica de TICs y Desarrollo Organizacional")
st.markdown("**Análisis:** ¿Cómo las TICs median el desarrollo organizacional?")
st.markdown("Versión completa con análisis avanzados y predictivos")

# --------------------------
# Parámetros configurables
# --------------------------
CPU_USAGE_RATIO = 0.75
MAX_NETWORK_NODES = 100
MIN_EDGE_WEIGHT = 2
TOP_KEYWORDS = 50
TOP_AUTHORS = 20
TOP_COUNTRIES = 20
TOP_INSTITUTIONS = 30
TOP_SOURCES = 20

# --------------------------
# Funciones de normalización
# --------------------------
def normalize_author_name(name):
    if not name or pd.isna(name):
        return None
    name = str(name).strip()
    name = re.sub(r'\.(?=\s|$)', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.title()
    return name if name else None

def normalize_keyword(keyword):
    if not keyword or pd.isna(keyword):
        return None
    kw = str(keyword).strip().lower()
    kw = re.sub(r'^[^\w]+|[^\w]+$', '', kw)
    kw = re.sub(r'[-\s]+', ' ', kw)
    return kw if kw else None

def normalize_country(country):
    if not country or pd.isna(country):
        return None
    country = str(country).strip().title()
    country_map = {
        'Usa': 'United States', 'United States Of America': 'United States',
        'U.S.A.': 'United States', 'U.S.': 'United States',
        'Uk': 'United Kingdom', 'U.K.': 'United Kingdom',
        'England': 'United Kingdom', 'Scotland': 'United Kingdom',
        'Wales': 'United Kingdom', 'Prc': 'China',
        "People'S Republic Of China": 'China', 'Peoples R China': 'China',
        'Korea': 'South Korea', 'Republic Of Korea': 'South Korea',
        'Russian Federation': 'Russia'
    }
    return country_map.get(country, country)

def country_to_iso3(country_name):
    """Convierte nombre de país a código ISO-3."""
    if not country_name:
        return None
    
    # Mapeo básico de países comunes a códigos ISO-3
    iso3_map = {
        'United States': 'USA',
        'United Kingdom': 'GBR',
        'China': 'CHN',
        'Germany': 'DEU',
        'France': 'FRA',
        'Italy': 'ITA',
        'Spain': 'ESP',
        'Canada': 'CAN',
        'Australia': 'AUS',
        'Japan': 'JPN',
        'South Korea': 'KOR',
        'Brazil': 'BRA',
        'India': 'IND',
        'Russia': 'RUS',
        'Netherlands': 'NLD',
        'Sweden': 'SWE',
        'Switzerland': 'CHE',
        'Belgium': 'BEL',
        'Poland': 'POL',
        'Portugal': 'PRT',
        'Turkey': 'TUR',
        'Mexico': 'MEX',
        'Argentina': 'ARG',
        'Chile': 'CHL',
        'Colombia': 'COL',
        'Peru': 'PER',
        'Venezuela': 'VEN',
        'Ecuador': 'ECU',
        'Uruguay': 'URY',
        'Paraguay': 'PRY',
        'Bolivia': 'BOL',
        'South Africa': 'ZAF',
        'Egypt': 'EGY',
        'Nigeria': 'NGA',
        'Kenya': 'KEN',
        'Saudi Arabia': 'SAU',
        'United Arab Emirates': 'ARE',
        'Israel': 'ISR',
        'Iran': 'IRN',
        'Pakistan': 'PAK',
        'Bangladesh': 'BGD',
        'Thailand': 'THA',
        'Malaysia': 'MYS',
        'Singapore': 'SGP',
        'Indonesia': 'IDN',
        'Philippines': 'PHL',
        'Vietnam': 'VNM',
        'New Zealand': 'NZL',
        'Norway': 'NOR',
        'Denmark': 'DNK',
        'Finland': 'FIN',
        'Ireland': 'IRL',
        'Austria': 'AUT',
        'Czech Republic': 'CZE',
        'Greece': 'GRC',
        'Romania': 'ROU',
        'Hungary': 'HUN'
    }
    
    # Normalizar el nombre del país primero
    normalized = normalize_country(country_name)
    
    # Retornar el código ISO-3 o el nombre original si no se encuentra
    return iso3_map.get(normalized, normalized)

def normalize_institution(affiliation):
    if not affiliation or pd.isna(affiliation):
        return None
    parts = str(affiliation).split(',')
    if not parts:
        return None
    inst = parts[0].strip()
    inst = re.sub(r'\([^)]*\)', '', inst)
    inst = re.sub(r'[0-9]', '', inst)
    inst = re.sub(r'\s+', ' ', inst).strip()
    inst = inst.replace('Univ ', 'University ')
    inst = inst.replace('Inst ', 'Institute ')
    return inst.title() if inst and len(inst) > 3 else None

def normalize_source(source):
    if not source or pd.isna(source):
        return None
    source = str(source).strip()
    source = re.sub(r'\s+', ' ', source)
    return source.title() if source else None

# --------------------------
# Utilidades
# --------------------------
def split_authors(text):
    if not text or pd.isna(text):
        return []
    separators = [",", ";", " and "]
    s = str(text)
    for sep in separators:
        s = s.replace(sep, "|")
    parts = [normalize_author_name(p) for p in s.split("|") if p.strip()]
    return [p for p in parts if p]

def split_keywords(text):
    if not text or pd.isna(text):
        return []
    parts = [normalize_keyword(p) for p in str(text).replace(";", ",").split(",") if p.strip()]
    return [p for p in parts if p]

def extract_countries_from_affiliation(text):
    if not text or pd.isna(text):
        return []
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    countries = []
    for p in parts:
        toks = p.split(",")
        country_candidate = toks[-1].strip()
        if len(country_candidate) > 1 and not any(ch.isdigit() for ch in country_candidate):
            normalized = normalize_country(country_candidate)
            if normalized:
                countries.append(normalized)
    return list(dict.fromkeys(countries))

def extract_institutions_from_affiliation(text):
    if not text or pd.isna(text):
        return []
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    institutions = []
    for p in parts:
        inst = normalize_institution(p)
        if inst:
            institutions.append(inst)
    return list(dict.fromkeys(institutions))

# --------------------------
# Paralelización
# --------------------------
def process_chunk_authors(series_chunk):
    result = []
    for val in series_chunk:
        result.extend(split_authors(val))
    return result

def process_chunk_keywords(series_chunk):
    result = []
    for val in series_chunk:
        result.extend(split_keywords(val))
    return result

def process_chunk_countries(series_chunk):
    result = []
    for val in series_chunk:
        result.extend(extract_countries_from_affiliation(val))
    return result

def process_chunk_institutions(series_chunk):
    result = []
    for val in series_chunk:
        result.extend(extract_institutions_from_affiliation(val))
    return result

def parallel_map_series(series, worker_func, n_cores):
    series = series.fillna("").astype(str)
    n = len(series)
    if n == 0:
        return []
    chunk_sizes = int(math.ceil(n / n_cores))
    chunks = [series[i:i+chunk_sizes].tolist() for i in range(0, n, chunk_sizes)]
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker_func, chunks)
    flat = [item for sub in results for item in sub]
    return flat

# --------------------------
# Cargar CSV
# --------------------------
RUTA_CSV = "data.csv"
try:
    df = pd.read_csv(RUTA_CSV, dtype=str)
    st.success(f"Archivo '{RUTA_CSV}' cargado correctamente con {len(df)} registros.")
except FileNotFoundError:
    st.error(f"No se encontró el archivo '{RUTA_CSV}'.")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# Detectar columnas
author_full_col = "Author full names" if "Author full names" in df.columns else None
author_col = author_full_col or next((c for c in df.columns if c.lower().startswith("authors")), None)
keywords_cols = [c for c in df.columns if "keyword" in c.lower()]
aff_col = next((c for c in df.columns if "affil" in c.lower()), None)
year_col = next((c for c in df.columns if c.lower() == "year"), None)
source_col = next((c for c in df.columns if "source" in c.lower() or "journal" in c.lower()), None)
doctype_col = next((c for c in df.columns if "document type" in c.lower() or "type" in c.lower()), None)
citations_col = next((c for c in df.columns if "cited" in c.lower() or "citation" in c.lower()), None)
language_col = next((c for c in df.columns if "language" in c.lower()), None)
funding_col = next((c for c in df.columns if "funding" in c.lower() or "sponsor" in c.lower()), None)

with st.expander("Columnas detectadas en el CSV"):
    st.write({
        "Autores": author_col,
        "Palabras clave": keywords_cols,
        "Afiliaciones": aff_col,
        "Año": year_col,
        "Fuente": source_col,
        "Tipo documento": doctype_col,
        "Citaciones": citations_col,
        "Idioma": language_col,
        "Financiamiento": funding_col
    })
    st.dataframe(df.head(5))

# --------------------------
# Cores
# --------------------------
total_cores = mp.cpu_count()
usable_cores = max(1, int(total_cores * CPU_USAGE_RATIO))
st.sidebar.info(f"CPU: {total_cores} cores | Usando: {usable_cores} cores")

# --------------------------
# Filtros avanzados
# --------------------------
st.sidebar.header("Filtros Avanzados")
years = sorted(df[year_col].dropna().unique()) if year_col else []
selected_years = st.sidebar.multiselect("Filtrar por año", options=years, default=years)

# Filtro por tipo de documento
if doctype_col:
    doc_types = sorted(df[doctype_col].dropna().unique())
    selected_doctypes = st.sidebar.multiselect("Tipo de documento", options=doc_types, default=doc_types)
else:
    selected_doctypes = []

# Filtro por idioma
if language_col:
    languages = sorted(df[language_col].dropna().unique())
    selected_languages = st.sidebar.multiselect("Idioma", options=languages, default=languages)
else:
    selected_languages = []

# Aplicar filtros
df_filtered = df.copy()
if year_col and selected_years:
    df_filtered = df_filtered[df_filtered[year_col].isin(selected_years)]
if doctype_col and selected_doctypes:
    df_filtered = df_filtered[df_filtered[doctype_col].isin(selected_doctypes)]
if language_col and selected_languages:
    df_filtered = df_filtered[df_filtered[language_col].isin(selected_languages)]

st.sidebar.metric("Registros filtrados", f"{len(df_filtered)} / {len(df)}")

# --------------------------
# Procesamiento paralelo
# --------------------------
progress_text = st.empty()
progress_bar = st.progress(0)
start_time = time.time()

progress_text.text("Procesando autores...")
authors_series = df_filtered[author_col] if author_col else pd.Series(dtype=str)
all_authors = parallel_map_series(authors_series, process_chunk_authors, usable_cores) if len(authors_series)>0 else []
progress_bar.progress(15)

progress_text.text("Procesando palabras clave...")
all_keywords = []
keywords_by_year = defaultdict(list)
if keywords_cols:
    for col in keywords_cols:
        series_col = df_filtered[col] if col in df_filtered.columns else pd.Series(dtype=str)
        if len(series_col) > 0:
            kws = parallel_map_series(series_col, process_chunk_keywords, usable_cores)
            all_keywords.extend(kws)
            # Keywords por año
            if year_col:
                for idx, row in df_filtered.iterrows():
                    year = row[year_col]
                    kw_text = row[col]
                    if pd.notna(year) and pd.notna(kw_text):
                        kws_row = split_keywords(kw_text)
                        keywords_by_year[year].extend(kws_row)
progress_bar.progress(30)

progress_text.text("Procesando países...")
all_countries = []
if aff_col:
    aff_series = df_filtered[aff_col] if aff_col in df_filtered.columns else pd.Series(dtype=str)
    if len(aff_series) > 0:
        all_countries = parallel_map_series(aff_series, process_chunk_countries, usable_cores)
progress_bar.progress(45)

progress_text.text("Procesando instituciones...")
all_institutions = []
institutions_by_year = {}
institution_country_pairs = []
if aff_col:
    all_institutions = parallel_map_series(aff_series, process_chunk_institutions, usable_cores)
    if year_col:
        for idx, row in df_filtered.iterrows():
            year = row[year_col]
            aff = row[aff_col]
            if pd.notna(year) and pd.notna(aff):
                insts = extract_institutions_from_affiliation(aff)
                countries = extract_countries_from_affiliation(aff)
                for inst in insts:
                    if year not in institutions_by_year:
                        institutions_by_year[year] = []
                    institutions_by_year[year].append(inst)
                # Pares institución-país
                for inst in insts:
                    for country in countries:
                        institution_country_pairs.append((inst, country))
progress_bar.progress(60)

progress_text.text("Procesando fuentes...")
all_sources = []
if source_col:
    for val in df_filtered[source_col].dropna():
        src = normalize_source(val)
        if src:
            all_sources.append(src)
progress_bar.progress(75)

progress_text.text("Analizando co-ocurrencias...")
# Co-ocurrencia de keywords
keyword_pairs = []
if keywords_cols:
    for idx, row in df_filtered.iterrows():
        all_kws_row = []
        for col in keywords_cols:
            if col in row and pd.notna(row[col]):
                all_kws_row.extend(split_keywords(row[col]))
        all_kws_row = list(dict.fromkeys(all_kws_row))  # unique
        if len(all_kws_row) >= 2:
            for pair in combinations(sorted(all_kws_row), 2):
                keyword_pairs.append(pair)
progress_bar.progress(90)

elapsed = time.time() - start_time
counter_authors = Counter(all_authors)
counter_keywords = Counter(all_keywords)
counter_countries = Counter(all_countries)
counter_institutions = Counter(all_institutions)
counter_sources = Counter(all_sources)
counter_keyword_pairs = Counter(keyword_pairs)

progress_bar.progress(100)
progress_text.text(f"Procesamiento completado en {elapsed:.1f}s")
time.sleep(0.5)
progress_bar.empty()
progress_text.empty()

# --------------------------
# Tabs del dashboard
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Evolución",
    "Autores",
    "Keywords",
    "Geografía",
    "Coautorías",
    "Instituciones",
    "Fuentes",
    "Co-ocurrencia",
    "Métricas",
    "Tendencias"
])

# TAB 1: Evolución
with tab1:
    st.header("Evolución Anual de Publicaciones")
    if year_col:
        pubs_per_year = df_filtered[year_col].value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pubs_per_year.index, y=pubs_per_year.values,
            name='Publicaciones', marker_color='lightblue',
            hovertemplate='<b>Año %{x}</b><br>Publicaciones: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=pubs_per_year.index, y=pubs_per_year.values,
            mode='lines+markers', name='Tendencia',
            line=dict(color='darkblue', width=2), marker=dict(size=8),
            hovertemplate='<b>Año %{x}</b><br>Publicaciones: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title='Tendencia de Publicaciones por Año',
            xaxis_title='Año', yaxis_title='Número de Publicaciones',
            hovermode='x unified', height=500
        )
        st.plotly_chart(fig, width='stretch')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Publicaciones", len(df_filtered))
        col2.metric("Año Máx. Productivo", pubs_per_year.idxmax() if len(pubs_per_year)>0 else "N/A")
        col3.metric("Promedio Anual", f"{pubs_per_year.mean():.1f}")
        growth_rate = ((pubs_per_year.iloc[-1] - pubs_per_year.iloc[0]) / pubs_per_year.iloc[0] * 100) if len(pubs_per_year) > 1 else 0
        col4.metric("Crecimiento %", f"{growth_rate:.1f}%")
        
        # Tipos de documento por año
        if doctype_col:
            st.subheader("Distribución de Tipos de Documento por Año")
            doc_year_counts = df_filtered.groupby([year_col, doctype_col]).size().reset_index(name='count')
            fig_doc = px.bar(doc_year_counts, x=year_col, y='count', color=doctype_col,
                            title='Evolución de Tipos de Documento',
                            labels={year_col: 'Año', 'count': 'Publicaciones'})
            fig_doc.update_layout(height=400)
            st.plotly_chart(fig_doc, width='stretch')

# TAB 2: Autores
with tab2:
    st.header("Autores Más Productivos")
    if counter_authors:
        top_authors = pd.Series(dict(counter_authors)).sort_values(ascending=False).head(TOP_AUTHORS)
        
        fig = go.Figure(go.Bar(
            x=top_authors.values[::-1],
            y=[(a[:50] + "..." if len(a)>50 else a) for a in top_authors.index[::-1]],
            orientation='h', marker_color='steelblue',
            hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<extra></extra>'
        ))
        fig.update_layout(
            title=f'Top {TOP_AUTHORS} Autores Más Productivos',
            xaxis_title='Número de Publicaciones', yaxis_title='Autor',
            height=700, yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=200, r=20, t=50, b=50)
        )
        st.plotly_chart(fig, width='stretch')
        
        # Índice H simplificado (si hay citaciones)
        if citations_col:
            st.subheader("Análisis de Impacto")
            st.info("Calculando métricas de impacto basadas en citaciones...")
        
        with st.expander("Ver tabla completa de autores"):
            df_authors = pd.DataFrame(counter_authors.most_common(50), columns=['Autor', 'Publicaciones'])
            st.dataframe(df_authors, use_container_width=True)

# TAB 3: Keywords
with tab3:
    st.header("Análisis de Palabras Clave")
    if counter_keywords:
        top_kws = pd.Series(dict(counter_keywords)).sort_values(ascending=False).head(TOP_KEYWORDS)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Nube de Palabras")
            wc_text = " ".join([k for k,c in top_kws.items() for _ in range(min(5,c))])
            wc = WordCloud(width=800, height=400, background_color="white", 
                          colormap='viridis').generate(wc_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10,5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        
        with col2:
            st.subheader(f"Top {min(20, TOP_KEYWORDS)} Keywords")
            fig = go.Figure(go.Bar(
                x=top_kws.values[:20][::-1], y=top_kws.index[:20][::-1],
                orientation='h', marker_color='teal',
                hovertemplate='<b>%{y}</b><br>Frecuencia: %{x}<extra></extra>'
            ))
            fig.update_layout(
                height=500, xaxis_title='Frecuencia', yaxis_title='Palabra Clave',
                yaxis={'categoryorder': 'total ascending'}, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, width='stretch')

# TAB 4: Geografía
with tab4:
    st.header("Análisis Geográfico")
    if counter_countries:
        top_countries = pd.Series(dict(counter_countries)).sort_values(ascending=False).head(TOP_COUNTRIES)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Países Más Productivos")
            fig = go.Figure(go.Bar(
                x=top_countries.values[::-1], y=top_countries.index[::-1],
                orientation='h', marker_color='coral',
                hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<extra></extra>'
            ))
            fig.update_layout(
                height=600, xaxis_title='Publicaciones', yaxis_title='País',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Mapa Mundial")
            country_df = pd.DataFrame(counter_countries.most_common(100), columns=['Country','Publications'])
            # Convertir nombres de países a códigos ISO-3
            country_df['ISO3'] = country_df['Country'].apply(country_to_iso3)
            # Filtrar filas donde no se pudo convertir a ISO-3 (mantener solo las que tienen código válido)
            country_df_iso3 = country_df[country_df['ISO3'].notna()].copy()
            fig_map = px.choropleth(
                country_df_iso3, locations='ISO3', locationmode='ISO-3',
                color='Publications', title='Distribución Global',
                color_continuous_scale='Blues',
                hover_data={'Country': True, 'Publications': True, 'ISO3': False}
            )
            fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_map, width='stretch')
        
        # Red internacional
        st.subheader("Red de Colaboración Internacional")
        if aff_col:
            country_pairs = []
            aff_series = df_filtered[aff_col].dropna().astype(str)
            for aff in aff_series:
                countries = extract_countries_from_affiliation(aff)
                countries = list(dict.fromkeys(countries))
                for i in range(len(countries)):
                    for j in range(i+1, len(countries)):
                        country_pairs.append(tuple(sorted([countries[i], countries[j]])))
            
            if country_pairs:
                cc = Counter(country_pairs)
                Gc = nx.Graph()
                for (a,b),w in cc.items():
                    if w >= MIN_EDGE_WEIGHT:
                        Gc.add_edge(a, b, weight=int(w), title=f"{a} ↔ {b}: {w} colaboraciones")
                
                if Gc.number_of_nodes() > 0:
                    if Gc.number_of_nodes() > MAX_NETWORK_NODES:
                        degrees = dict(Gc.degree(weight='weight'))
                        top_n = sorted(degrees.items(), key=lambda x:x[1], reverse=True)[:MAX_NETWORK_NODES]
                        top_names = [n for n,_ in top_n]
                        subGc = Gc.subgraph(top_names).copy()
                    else:
                        subGc = Gc
                    
                    netc = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                    netc.from_nx(subGc)
                    netc.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -30000}}}""")
                    tmp_country = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
                    netc.save_graph(tmp_country.name)
                    tmp_country.close()
                    with open(tmp_country.name, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
                    os.unlink(tmp_country.name)
                    st.info(f"Red: {subGc.number_of_nodes()} países, {subGc.number_of_edges()} colaboraciones")

# TAB 5: Coautorías
with tab5:
    st.header("Red de Coautorías")
    if counter_authors:
        top_nodes = [a for a,_ in counter_authors.most_common(MAX_NETWORK_NODES)]
        G = nx.Graph()
        
        auth_series = df_filtered[author_col].dropna().astype(str)
        for val in auth_series:
            authors = split_authors(val)
            authors = [a for a in authors if a in top_nodes]
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a1, a2 = authors[i], authors[j]
                    if G.has_edge(a1,a2):
                        G[a1][a2]['weight'] += 1
                    else:
                        G.add_edge(a1, a2, weight=1)
        
        edges_to_remove = [(u,v) for u,v,d in G.edges(data=True) if d.get('weight',0) < MIN_EDGE_WEIGHT]
        G.remove_edges_from(edges_to_remove)
        
        if G.number_of_nodes() > 0:
            for u, v, data in G.edges(data=True):
                data['title'] = f"{u} ↔ {v}: {data['weight']} colaboraciones"
            
            net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
            
            if G.number_of_nodes() > MAX_NETWORK_NODES:
                degrees = dict(G.degree(weight='weight'))
                top_n = sorted(degrees.items(), key=lambda x:x[1], reverse=True)[:MAX_NETWORK_NODES]
                top_names = [n for n,_ in top_n]
                subG = G.subgraph(top_names).copy()
            else:
                subG = G
            
            net.from_nx(subG)
            net.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -8000}}}""")
            
            path_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
            net.save_graph(path_tmp.name)
            path_tmp.close()
            with open(path_tmp.name, 'r', encoding='utf-8') as f:
                html = f.read()
            st.components.v1.html(html, height=700)
            os.unlink(path_tmp.name)
            
            st.info(f"Red: {subG.number_of_nodes()} autores, {subG.number_of_edges()} colaboraciones")

# TAB 6: Instituciones
with tab6:
    st.header("Análisis Institucional")
    
    if counter_institutions:
        top_insts = pd.Series(dict(counter_institutions)).sort_values(ascending=False).head(TOP_INSTITUTIONS)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Instituciones Más Productivas")
            fig = go.Figure(go.Bar(
                x=top_insts.values[::-1],
                y=[(inst[:50] + "..." if len(inst)>50 else inst) for inst in top_insts.index[::-1]],
                orientation='h', marker_color='purple',
                hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<extra></extra>'
            ))
            fig.update_layout(
                height=700, xaxis_title='Publicaciones', yaxis_title='Institución',
                yaxis={'categoryorder': 'total ascending'}, margin=dict(l=250, r=20, t=30, b=50)
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Mapa de Calor: Instituciones vs Años")
            if institutions_by_year and year_col:
                years_list = sorted(institutions_by_year.keys())
                top_inst_names = [inst for inst in top_insts.index[:20]]
                
                heatmap_data = []
                for inst in top_inst_names:
                    row = []
                    for year in years_list:
                        count = institutions_by_year[year].count(inst) if year in institutions_by_year else 0
                        row.append(count)
                    heatmap_data.append(row)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=years_list,
                    y=[inst[:40] + "..." if len(inst)>40 else inst for inst in top_inst_names],
                    colorscale='YlOrRd',
                    hovertemplate='<b>%{y}</b><br>Año: %{x}<br>Publicaciones: %{z}<extra></extra>'
                ))
                fig_heat.update_layout(
                    title='Evolución de Publicaciones por Institución',
                    xaxis_title='Año', yaxis_title='Institución', height=700
                )
                st.plotly_chart(fig_heat, width='stretch')
        
        # Matriz Institución-País (Sankey)
        if institution_country_pairs:
            st.subheader("Relación Institución-País")
            inst_country_counter = Counter(institution_country_pairs)
            top_pairs = inst_country_counter.most_common(30)
            
            institutions_sankey = [p[0][0] for p in top_pairs]
            countries_sankey = [p[0][1] for p in top_pairs]
            values_sankey = [p[1] for p in top_pairs]
            
            all_nodes = list(dict.fromkeys(institutions_sankey + countries_sankey))
            inst_indices = [all_nodes.index(inst) for inst in institutions_sankey]
            country_indices = [all_nodes.index(country) for country in countries_sankey]
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[n[:40] + "..." if len(n)>40 else n for n in all_nodes],
                    color="blue"
                ),
                link=dict(
                    source=inst_indices,
                    target=country_indices,
                    value=values_sankey,
                    hovertemplate='%{source.label} → %{target.label}<br>Publicaciones: %{value}<extra></extra>'
                )
            )])
            fig_sankey.update_layout(title="Flujo Institución → País", height=600)
            st.plotly_chart(fig_sankey, width='stretch')

# TAB 7: Fuentes
with tab7:
    st.header("Análisis de Fuentes de Publicación")
    
    if counter_sources:
        top_sources = pd.Series(dict(counter_sources)).sort_values(ascending=False).head(TOP_SOURCES)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top Revistas/Fuentes")
            fig = go.Figure(go.Bar(
                x=top_sources.values[::-1],
                y=[(src[:60] + "..." if len(src)>60 else src) for src in top_sources.index[::-1]],
                orientation='h', marker_color='darkgreen',
                hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<extra></extra>'
            ))
            fig.update_layout(
                height=700, xaxis_title='Publicaciones', yaxis_title='Fuente',
                yaxis={'categoryorder': 'total ascending'}, margin=dict(l=300, r=20, t=30, b=50)
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Distribución por Tipo")
            if doctype_col:
                doc_types = df_filtered[doctype_col].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=doc_types.index,
                    values=doc_types.values,
                    hovertemplate='<b>%{label}</b><br>Publicaciones: %{value}<br>Porcentaje: %{percent}<extra></extra>'
                )])
                fig_pie.update_layout(title='Tipos de Documento', height=400)
                st.plotly_chart(fig_pie, width='stretch')
            
            if language_col:
                st.subheader("Distribución por Idioma")
                lang_counts = df_filtered[language_col].value_counts().head(10)
                fig_lang = go.Figure(data=[go.Pie(
                    labels=lang_counts.index,
                    values=lang_counts.values,
                    hovertemplate='<b>%{label}</b><br>Publicaciones: %{value}<extra></extra>'
                )])
                fig_lang.update_layout(title='Idiomas de Publicación', height=400)
                st.plotly_chart(fig_lang, width='stretch')
    
    # Estadísticas de concentración
    if counter_sources:
        total_pubs = sum(counter_sources.values())
        top_10_pubs = sum([v for k,v in counter_sources.most_common(10)])
        concentration = (top_10_pubs / total_pubs) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Fuentes", len(counter_sources))
        col2.metric("Top 10 concentra", f"{concentration:.1f}%")
        col3.metric("Fuentes únicas", len([k for k,v in counter_sources.items() if v == 1]))

# TAB 8: Co-ocurrencia
with tab8:
    st.header("Red de Co-ocurrencia de Palabras Clave")
    
    if counter_keyword_pairs:
        st.info("Mostrando las palabras clave que aparecen juntas con mayor frecuencia")
        
        # Red de co-ocurrencia
        top_pairs = counter_keyword_pairs.most_common(100)
        G_cooc = nx.Graph()
        
        for (kw1, kw2), weight in top_pairs:
            if weight >= MIN_EDGE_WEIGHT:
                G_cooc.add_edge(kw1, kw2, weight=weight, title=f"{kw1} ↔ {kw2}: {weight} veces")
        
        if G_cooc.number_of_nodes() > 0:
            # Detectar comunidades
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G_cooc)
            
            st.metric("Clusters temáticos detectados", len(communities))
            
            # Limitar nodos
            if G_cooc.number_of_nodes() > MAX_NETWORK_NODES:
                degrees = dict(G_cooc.degree(weight='weight'))
                top_nodes_cooc = sorted(degrees.items(), key=lambda x:x[1], reverse=True)[:MAX_NETWORK_NODES]
                top_kw_names = [n for n,_ in top_nodes_cooc]
                subG_cooc = G_cooc.subgraph(top_kw_names).copy()
            else:
                subG_cooc = G_cooc
            
            net_cooc = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
            net_cooc.from_nx(subG_cooc)
            net_cooc.set_options("""
            {
              "nodes": {"font": {"size": 14}},
              "edges": {"smooth": {"type": "continuous"}},
              "physics": {"barnesHut": {"gravitationalConstant": -5000, "springLength": 200}}
            }
            """)
            
            tmp_cooc = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
            net_cooc.save_graph(tmp_cooc.name)
            tmp_cooc.close()
            with open(tmp_cooc.name, 'r', encoding='utf-8') as f:
                html = f.read()
            st.components.v1.html(html, height=700)
            os.unlink(tmp_cooc.name)
            
            st.info(f"Red: {subG_cooc.number_of_nodes()} keywords, {subG_cooc.number_of_edges()} co-ocurrencias")
            
            # Top pares
            with st.expander("Top 20 pares de keywords más frecuentes"):
                df_pairs = pd.DataFrame(
                    [(f"{k1} + {k2}", v) for (k1, k2), v in counter_keyword_pairs.most_common(20)],
                    columns=['Par de Keywords', 'Frecuencia']
                )
                st.dataframe(df_pairs, use_container_width=True)
        else:
            st.info("No hay suficientes co-ocurrencias para mostrar la red")
    else:
        st.warning("No se encontraron co-ocurrencias de keywords")

# TAB 9: Métricas
with tab9:
    st.header("Métricas e Indicadores Bibliométricos")
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Publicaciones", len(df_filtered))
    col2.metric("Total Autores", len(counter_authors))
    col3.metric("Total Países", len(counter_countries))
    col4.metric("Total Keywords", len(counter_keywords))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Instituciones", len(counter_institutions))
    col2.metric("Fuentes", len(counter_sources))
    avg_authors_per_pub = len(all_authors) / len(df_filtered) if len(df_filtered) > 0 else 0
    col3.metric("Autores/Pub promedio", f"{avg_authors_per_pub:.2f}")
    collab_index = (avg_authors_per_pub - 1) if avg_authors_per_pub > 1 else 0
    col4.metric("Índice Colaboración", f"{collab_index:.2f}")
    
    # Análisis de citaciones (si existe)
    if citations_col:
        st.subheader("Análisis de Impacto (Citaciones)")
        
        # Convertir citaciones a numérico
        df_filtered['citations_num'] = pd.to_numeric(df_filtered[citations_col], errors='coerce')
        citations_data = df_filtered['citations_num'].dropna()
        
        if len(citations_data) > 0:
            # Calcular Índice H
            citations_sorted = sorted(citations_data.values, reverse=True)
            h_index = 0
            for i, cites in enumerate(citations_sorted, 1):
                if cites >= i:
                    h_index = i
                else:
                    break
            
            # Calcular Factor de Impacto (últimos 2 años si hay datos de año)
            impact_factor = 0
            if year_col:
                current_year = int(max(df_filtered[year_col].dropna()))
                recent_years = [str(current_year), str(current_year - 1)]
                
                # Publicaciones en últimos 2 años
                recent_pubs = df_filtered[df_filtered[year_col].isin(recent_years)]
                num_recent_pubs = len(recent_pubs)
                
                # Citaciones recibidas por esas publicaciones
                if num_recent_pubs > 0:
                    recent_citations = pd.to_numeric(recent_pubs[citations_col], errors='coerce').sum()
                    impact_factor = recent_citations / num_recent_pubs
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Citaciones", f"{int(citations_data.sum())}")
            col2.metric("Índice H", f"{h_index}")
            col3.metric("Factor de Impacto", f"{impact_factor:.2f}" if impact_factor > 0 else "N/A")
            col4.metric("Promedio Citas/Pub", f"{citations_data.mean():.2f}")
            col5.metric("Máx Citaciones", f"{int(citations_data.max())}")
            
            # Índice i10 (artículos con al menos 10 citaciones)
            i10_index = len([c for c in citations_data if c >= 10])
            
            # Percentiles de citaciones
            p25 = citations_data.quantile(0.25)
            p50 = citations_data.quantile(0.50)
            p75 = citations_data.quantile(0.75)
            p90 = citations_data.quantile(0.90)
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Índice i10", f"{i10_index}")
            col2.metric("Mediana Citas", f"{p50:.0f}")
            col3.metric("Percentil 75", f"{p75:.0f}")
            col4.metric("Percentil 90", f"{p90:.0f}")
            
            # Información sobre las métricas
            with st.expander("Información sobre las métricas de impacto"):
                st.markdown(f"""
                **Índice H**: Un investigador tiene índice h si h de sus publicaciones tienen al menos h citaciones cada una.
                - Mide tanto productividad como impacto
                - Un h-index de {h_index} significa que hay {h_index} artículos con al menos {h_index} citaciones
                
                **Factor de Impacto**: Promedio de citaciones recibidas por artículos publicados en los últimos 2 años.
                - Se calcula: (Citaciones a artículos de últimos 2 años) / (Número de artículos publicados en esos 2 años)
                - Indica el impacto promedio reciente de las publicaciones
                
                **Índice i10**: Número de publicaciones con al menos 10 citaciones.
                - Útil para identificar trabajos con impacto significativo
                """)
            
            # Top artículos más citados
            st.subheader("Top 10 Artículos Más Citados")
            title_col = next((c for c in df_filtered.columns if 'title' in c.lower()), None)
            if title_col:
                top_cited = df_filtered.nlargest(10, 'citations_num')[[title_col, author_col, year_col, citations_col]]
                top_cited.columns = ['Título', 'Autores', 'Año', 'Citaciones']
                st.dataframe(top_cited, use_container_width=True)
            
            # Top artículos más relevantes según el índice H
            st.subheader("Top 10 Artículos Más Relevantes Según el Índice H")
            if title_col:
                # Crear DataFrame con citaciones y filtrar los que tienen citaciones válidas
                df_with_citations = df_filtered[df_filtered['citations_num'].notna()].copy()
                df_with_citations = df_with_citations.sort_values('citations_num', ascending=False).reset_index(drop=True)
                
                # Los artículos más relevantes según el índice H son aquellos que están en el "core" del índice H
                # Estos son los primeros h artículos que tienen citaciones >= a su posición en el ranking
                h_relevant_articles = []
                for idx, row in df_with_citations.iterrows():
                    position = idx + 1  # Posición 1-indexed (ranking por citaciones)
                    citations = int(row['citations_num'])
                    
                    # Los artículos en el core del índice H son aquellos donde:
                    # - Están en las primeras h posiciones (posición <= h_index)
                    # - Tienen citaciones >= a su posición (condición del índice H)
                    if position <= h_index and citations >= position:
                        h_relevant_articles.append({
                            'Posición en Índice H': position,
                            'Título': row[title_col] if title_col else 'N/A',
                            'Autores': row[author_col] if author_col else 'N/A',
                            'Año': row[year_col] if year_col else 'N/A',
                            'Citaciones': citations
                        })
                    # Si no tenemos suficientes del core, agregamos los más citados que siguen
                    elif len(h_relevant_articles) < 10:
                        h_relevant_articles.append({
                            'Posición en Índice H': position,
                            'Título': row[title_col] if title_col else 'N/A',
                            'Autores': row[author_col] if author_col else 'N/A',
                            'Año': row[year_col] if year_col else 'N/A',
                            'Citaciones': citations
                        })
                    
                    # Limitar a los top 10
                    if len(h_relevant_articles) >= 10:
                        break
                
                if h_relevant_articles:
                    df_h_relevant = pd.DataFrame(h_relevant_articles)
                    st.dataframe(df_h_relevant, use_container_width=True)
                    
                    # Contar cuántos están en el core del índice H
                    core_count = sum(1 for art in h_relevant_articles if art['Posición en Índice H'] <= h_index)
                    
                    if core_count > 0:
                        st.info(f"**{core_count} de estos artículos** están en el núcleo del índice H (h={h_index}). "
                               f"Son aquellos que tienen al menos tantas citaciones como su posición en el ranking, "
                               f"y contribuyen directamente al cálculo del índice H.")
                    else:
                        st.info(f"Estos son los **{len(h_relevant_articles)} artículos más citados** "
                               f"ordenados por relevancia. El índice H actual es {h_index}.")
                else:
                    st.warning("No se encontraron artículos con citaciones suficientes para el análisis del índice H.")
            
            # Distribución de citaciones
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_cit = go.Figure()
                fig_cit.add_trace(go.Histogram(
                    x=citations_data[citations_data <= citations_data.quantile(0.95)],
                    nbinsx=30,
                    marker_color='indianred',
                    hovertemplate='Rango: %{x}<br>Publicaciones: %{y}<extra></extra>'
                ))
                fig_cit.update_layout(
                    title='Distribución de Citaciones (95% percentil)',
                    xaxis_title='Número de Citaciones',
                    yaxis_title='Frecuencia',
                    height=400
                )
                st.plotly_chart(fig_cit, width='stretch')
            
            with col2:
                # Gráfico de índice H
                h_data = pd.DataFrame({
                    'Posición': range(1, min(len(citations_sorted), 50) + 1),
                    'Citaciones': citations_sorted[:50]
                })
                
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=h_data['Posición'],
                    y=h_data['Citaciones'],
                    mode='lines+markers',
                    name='Citaciones',
                    marker_color='blue'
                ))
                # Línea diagonal y=x
                fig_h.add_trace(go.Scatter(
                    x=[0, 50],
                    y=[0, 50],
                    mode='lines',
                    name='y=x',
                    line=dict(dash='dash', color='red')
                ))
                # Marcar el índice H
                fig_h.add_vline(x=h_index, line_dash="dot", line_color="green",
                               annotation_text=f"h={h_index}")
                fig_h.update_layout(
                    title='Índice H (Top 50 artículos)',
                    xaxis_title='Ranking de artículos',
                    yaxis_title='Citaciones',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_h, width='stretch')
    
    # Índices de productividad
    st.subheader("Ley de Lotka (Distribución de Productividad)")
    if counter_authors:
        productivity_dist = Counter(counter_authors.values())
        prod_df = pd.DataFrame(list(productivity_dist.items()), columns=['Publicaciones', 'Num_Autores'])
        prod_df = prod_df.sort_values('Publicaciones')
        
        fig_lotka = go.Figure()
        fig_lotka.add_trace(go.Scatter(
            x=prod_df['Publicaciones'],
            y=prod_df['Num_Autores'],
            mode='lines+markers',
            marker_color='purple',
            hovertemplate='<b>%{x}</b> publicaciones<br>%{y} autores<extra></extra>'
        ))
        fig_lotka.update_layout(
            title='Ley de Lotka: Distribución de Productividad de Autores',
            xaxis_title='Número de Publicaciones',
            yaxis_title='Número de Autores',
            xaxis_type='log',
            yaxis_type='log',
            height=400
        )
        st.plotly_chart(fig_lotka, width='stretch')
    
    # Distribución de Bradford (fuentes)
    if counter_sources:
        st.subheader("Ley de Bradford (Núcleo de Revistas)")
        sources_counts = sorted(counter_sources.values(), reverse=True)
        cumsum_sources = np.cumsum(sources_counts)
        total_articles = cumsum_sources[-1]
        
        bradford_zones = []
        zone_limit = total_articles / 3
        current_zone = 1
        
        for i, cum in enumerate(cumsum_sources):
            if cum <= zone_limit:
                bradford_zones.append(1)
            elif cum <= 2 * zone_limit:
                bradford_zones.append(2)
            else:
                bradford_zones.append(3)
        
        zone_counts = Counter(bradford_zones)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Zona 1 (Núcleo)", f"{zone_counts.get(1, 0)} revistas")
        col2.metric("Zona 2", f"{zone_counts.get(2, 0)} revistas")
        col3.metric("Zona 3 (Periferia)", f"{zone_counts.get(3, 0)} revistas")

# TAB 10: Tendencias y Predicción
with tab10:
    st.header("Análisis de Tendencias y Predicción")
    
    # Evolución temporal de keywords
    if keywords_by_year and year_col:
        st.subheader("Evolución de Keywords Emergentes")
        
        # Calcular crecimiento de keywords
        years_sorted = sorted(keywords_by_year.keys())
        if len(years_sorted) >= 3:
            recent_years = years_sorted[-3:]
            old_years = years_sorted[:-3] if len(years_sorted) > 3 else years_sorted[:1]
            
            recent_kws = []
            for y in recent_years:
                recent_kws.extend(keywords_by_year[y])
            
            old_kws = []
            for y in old_years:
                old_kws.extend(keywords_by_year[y])
            
            recent_counter = Counter(recent_kws)
            old_counter = Counter(old_kws)
            
            # Keywords emergentes (crecimiento)
            emerging = {}
            for kw, recent_count in recent_counter.most_common(50):
                old_count = old_counter.get(kw, 0)
                if old_count > 0:
                    growth = ((recent_count - old_count) / old_count) * 100
                else:
                    growth = 100
                if recent_count >= 3:  # Mínimo 3 apariciones
                    emerging[kw] = growth
            
            emerging_sorted = sorted(emerging.items(), key=lambda x: x[1], reverse=True)[:15]
            
            if emerging_sorted:
                df_emerging = pd.DataFrame(emerging_sorted, columns=['Keyword', 'Crecimiento %'])
                
                fig_emerging = go.Figure(go.Bar(
                    x=df_emerging['Crecimiento %'][::-1],
                    y=df_emerging['Keyword'][::-1],
                    orientation='h',
                    marker_color='lightgreen',
                    hovertemplate='<b>%{y}</b><br>Crecimiento: %{x:.1f}%<extra></extra>'
                ))
                fig_emerging.update_layout(
                    title=f'Keywords Emergentes (últimos {len(recent_years)} años vs anteriores)',
                    xaxis_title='Crecimiento %',
                    yaxis_title='Keyword',
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_emerging, width='stretch')
        
        # Heatmap temporal de keywords
        st.subheader("Mapa de Calor: Keywords vs Tiempo")
        top_kws_temporal = [kw for kw, _ in counter_keywords.most_common(15)]
        years_list = sorted(keywords_by_year.keys())
        
        heatmap_kw_data = []
        for kw in top_kws_temporal:
            row = []
            for year in years_list:
                count = keywords_by_year[year].count(kw)
                row.append(count)
            heatmap_kw_data.append(row)
        
        fig_heat_kw = go.Figure(data=go.Heatmap(
            z=heatmap_kw_data,
            x=years_list,
            y=top_kws_temporal,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b><br>Año: %{x}<br>Frecuencia: %{z}<extra></extra>'
        ))
        fig_heat_kw.update_layout(
            title='Evolución Temporal de Keywords Principales',
            xaxis_title='Año',
            yaxis_title='Keyword',
            height=500
        )
        st.plotly_chart(fig_heat_kw, width='stretch')
    
    # Predicción simple de publicaciones
    if year_col and len(pubs_per_year) >= 3:
        st.subheader("Proyección de Publicaciones")
        
        years_num = pubs_per_year.index.astype(int).values
        pubs_values = pubs_per_year.values
        
        # Regresión lineal simple
        z = np.polyfit(years_num, pubs_values, 1)
        p = np.poly1d(z)
        
        # Proyectar 3 años
        future_years = np.arange(years_num[-1] + 1, years_num[-1] + 4)
        future_pubs = p(future_years)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=years_num,
            y=pubs_values,
            mode='lines+markers',
            name='Real',
            marker_color='blue',
            hovertemplate='Año: %{x}<br>Publicaciones: %{y}<extra></extra>'
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_years,
            y=future_pubs,
            mode='lines+markers',
            name='Proyección',
            line=dict(dash='dash'),
            marker_color='red',
            hovertemplate='Año: %{x}<br>Proyección: %{y:.0f}<extra></extra>'
        ))
        fig_pred.update_layout(
            title='Proyección de Publicaciones (Regresión Lineal)',
            xaxis_title='Año',
            yaxis_title='Publicaciones',
            height=400
        )
        st.plotly_chart(fig_pred, width='stretch')
        
        st.info(f"Tendencia: ~{z[0]:.1f} publicaciones adicionales por año")
    
    # Insights automáticos
    st.subheader("Insights Estratégicos")
    
    insights = []
    
    if counter_countries:
        top_country = counter_countries.most_common(1)[0]
        insights.append(f"**Liderazgo geográfico**: {top_country[0]} domina con {top_country[1]} publicaciones ({top_country[1]/len(df_filtered)*100:.1f}%)")
    
    if counter_keywords:
        top_kw = counter_keywords.most_common(1)[0]
        insights.append(f"**Tema dominante**: '{top_kw[0]}' aparece {top_kw[1]} veces")
    
    if counter_authors:
        prolific = len([a for a, c in counter_authors.items() if c >= 3])
        insights.append(f"**Autores prolíficos**: {prolific} autores con 3+ publicaciones")
    
    if year_col and len(pubs_per_year) >= 2:
        recent_growth = pubs_per_year.iloc[-1] - pubs_per_year.iloc[-2]
        if recent_growth > 0:
            insights.append(f"**Crecimiento reciente**: +{recent_growth} publicaciones en el último año")
        else:
            insights.append(f"**Tendencia**: {recent_growth} publicaciones en el último año")
    
    if counter_sources:
        core_journals = len([s for s, c in counter_sources.items() if c >= 5])
        insights.append(f"**Revistas core**: {core_journals} fuentes con 5+ artículos")
    
    for insight in insights:
        st.markdown(f"- {insight}")

# Footer
st.markdown("---")
st.markdown("**Dashboard** | Desarrollado con Streamlit, Plotly y NetworkX, sklearn, WordCloud, PyVis, pandas| Autor: Victor A. Martínez | [GitHub](")
st.markdown("*Análisis bibliométrico completo con métricas avanzadas, redes de colaboración y predicciones*")