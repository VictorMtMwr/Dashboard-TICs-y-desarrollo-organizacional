"""
Dashboard Bibliométrico - Aplicación principal de Streamlit
"""
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
import time
from itertools import combinations
import re

# Importar módulos locales
import config
from utils.normalizers import (
    normalize_author_name,
    normalize_keyword,
    normalize_country,
    country_to_iso3,
    normalize_institution,
    normalize_source,
    split_authors,
    split_keywords,
    extract_countries_from_affiliation,
    extract_institutions_from_affiliation
)
from utils.data_processing import (
    process_chunk_authors,
    process_chunk_keywords,
    process_chunk_countries,
    process_chunk_institutions,
    parallel_map_series,
    detect_columns,
    load_data
)
from utils.metrics import (
    calculate_h_index,
    calculate_impact_factor,
    calculate_i10_index,
    get_citation_percentiles,
    get_top_h_relevant_articles,
    calculate_g_index,
    calculate_e_index,
    calculate_m_index,
    calculate_collaboration_coefficient,
    calculate_international_collaboration_index,
    calculate_price_index,
    calculate_author_h_index
)

# --------------------------
# Configuración Streamlit
# --------------------------
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.PAGE_LAYOUT)
st.title(config.MAIN_TITLE)
st.markdown(config.SUBTITLE)
st.markdown(config.DESCRIPTION)

# --------------------------
# Parámetros configurables (importados de config)
# --------------------------
CPU_USAGE_RATIO = config.CPU_USAGE_RATIO
MAX_NETWORK_NODES = config.MAX_NETWORK_NODES
MIN_EDGE_WEIGHT = config.MIN_EDGE_WEIGHT
TOP_KEYWORDS = config.TOP_KEYWORDS
TOP_AUTHORS = config.TOP_AUTHORS
TOP_COUNTRIES = config.TOP_COUNTRIES
TOP_INSTITUTIONS = config.TOP_INSTITUTIONS
TOP_SOURCES = config.TOP_SOURCES

# --------------------------
# Cargar CSV
# --------------------------
df, error = load_data(config.RUTA_CSV)
if error:
    st.error(error)
    st.stop()
else:
    st.success(f"Archivo '{config.RUTA_CSV}' cargado correctamente con {len(df)} registros.")

# Detectar columnas
columns_info = detect_columns(df)
author_col = columns_info['author_col']
author_full_col = columns_info['author_full_col']
keywords_cols = columns_info['keywords_cols']
aff_col = columns_info['aff_col']
year_col = columns_info['year_col']
source_col = columns_info['source_col']
doctype_col = columns_info['doctype_col']
citations_col = columns_info['citations_col']
language_col = columns_info['language_col']
funding_col = columns_info['funding_col']
abstract_col = columns_info['abstract_col']
publisher_col = columns_info['publisher_col']
conference_col = columns_info['conference_col']
oa_col = columns_info['oa_col']

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
        "Financiamiento": funding_col,
        "Abstract": abstract_col,
        "Publisher": publisher_col,
        "Conferencia": conference_col,
        "Open Access": oa_col
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
        
        # H-index individual por autor
        if citations_col and author_col:
            st.markdown("---")
            st.subheader("H-index Individual por Autor (Top 20)")
            
            # Calcular H-index para cada autor
            top_authors_list = [author for author, _ in counter_authors.most_common(20)]
            authors_h_index = []
            
            for author in top_authors_list:
                h_idx = calculate_author_h_index(author, df_filtered, citations_col, author_col)
                pub_count = counter_authors[author]
                authors_h_index.append({
                    'Autor': author,
                    'Publicaciones': pub_count,
                    'H-index': h_idx
                })
            
            if authors_h_index:
                df_authors_h = pd.DataFrame(authors_h_index)
                df_authors_h = df_authors_h.sort_values('H-index', ascending=False)
                
                # Tabla comparativa
                st.dataframe(df_authors_h, use_container_width=True)
                
                # Gráfico H-index vs publicaciones (dispersión)
                fig_h_scatter = go.Figure()
                fig_h_scatter.add_trace(go.Scatter(
                    x=df_authors_h['Publicaciones'],
                    y=df_authors_h['H-index'],
                    mode='markers+text',
                    text=df_authors_h['Autor'].str[:30],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=df_authors_h['H-index'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="H-index")
                    ),
                    hovertemplate='<b>%{text}</b><br>Publicaciones: %{x}<br>H-index: %{y}<extra></extra>'
                ))
                fig_h_scatter.update_layout(
                    title='H-index vs Publicaciones (Top 20 Autores)',
                    xaxis_title='Número de Publicaciones',
                    yaxis_title='H-index',
                    height=500
                )
                st.plotly_chart(fig_h_scatter, width='stretch')
        
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
    
    # Análisis de editores (Publishers)
    if publisher_col:
        st.markdown("---")
        st.subheader("Análisis de Editores (Publishers)")
        
        all_publishers = []
        for val in df_filtered[publisher_col].dropna():
            publisher = str(val).strip()
            if publisher and publisher.lower() != 'nan':
                all_publishers.append(publisher)
        
        if all_publishers:
            counter_publishers = Counter(all_publishers)
            top_publishers = pd.Series(dict(counter_publishers)).sort_values(ascending=False).head(10)
            
            fig_publishers = go.Figure(go.Bar(
                x=top_publishers.values[::-1],
                y=[(pub[:60] + "..." if len(pub)>60 else pub) for pub in top_publishers.index[::-1]],
                orientation='h',
                marker_color='darkcyan',
                hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<extra></extra>'
            ))
            fig_publishers.update_layout(
                title='Top 10 Editores por Publicaciones',
                xaxis_title='Publicaciones',
                yaxis_title='Editor',
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=300, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_publishers, width='stretch')
            
            with st.expander("Ver tabla completa de editores"):
                df_publishers = pd.DataFrame(counter_publishers.most_common(50), columns=['Editor', 'Publicaciones'])
                st.dataframe(df_publishers, use_container_width=True)
    
    # Análisis de conferencias
    if conference_col:
        st.markdown("---")
        st.subheader("Análisis de Conferencias")
        
        all_conferences = []
        for val in df_filtered[conference_col].dropna():
            conference = str(val).strip()
            if conference and conference.lower() != 'nan':
                all_conferences.append(conference)
        
        if all_conferences:
            counter_conferences = Counter(all_conferences)
            top_conferences = pd.Series(dict(counter_conferences)).sort_values(ascending=False).head(10)
            
            fig_conferences = go.Figure(go.Bar(
                x=top_conferences.values[::-1],
                y=[(conf[:60] + "..." if len(conf)>60 else conf) for conf in top_conferences.index[::-1]],
                orientation='h',
                marker_color='crimson',
                hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<extra></extra>'
            ))
            fig_conferences.update_layout(
                title='Top 10 Conferencias',
                xaxis_title='Frecuencia de Publicaciones',
                yaxis_title='Conferencia',
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=300, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_conferences, width='stretch')
            
            with st.expander("Ver tabla completa de conferencias"):
                df_conferences = pd.DataFrame(counter_conferences.most_common(50), columns=['Conferencia', 'Frecuencia'])
                st.dataframe(df_conferences, use_container_width=True)

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
            # Calcular métricas bibliométricas usando módulos
            h_index, citations_sorted = calculate_h_index(citations_data)
            impact_factor = calculate_impact_factor(df_filtered, citations_col, year_col)
            i10_index = calculate_i10_index(citations_data)
            percentiles = get_citation_percentiles(citations_data)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Citaciones", f"{int(citations_data.sum())}")
            col2.metric("Índice H", f"{h_index}")
            col3.metric("Factor de Impacto", f"{impact_factor:.2f}" if impact_factor > 0 else "N/A")
            col4.metric("Promedio Citas/Pub", f"{citations_data.mean():.2f}")
            col5.metric("Máx Citaciones", f"{int(citations_data.max())}")
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Índice i10", f"{i10_index}")
            col2.metric("Mediana Citas", f"{percentiles['p50']:.0f}")
            col3.metric("Percentil 75", f"{percentiles['p75']:.0f}")
            col4.metric("Percentil 90", f"{percentiles['p90']:.0f}")
            
            # Métricas avanzadas de impacto (G-index, E-index, M-index)
            st.markdown("---")
            st.subheader("Métricas de Impacto Avanzadas")
            
            g_index = calculate_g_index(citations_data)
            e_index = calculate_e_index(citations_data, h_index)
            m_index = calculate_m_index(h_index, year_col, df_filtered)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Índice G", f"{g_index:.0f}", 
                       help="Mayor número g tal que los g artículos más citados tienen al menos g² citaciones en total")
            col2.metric("Índice E", f"{e_index:.2f}",
                       help="Exceso de citaciones sobre el H-index: √(suma citaciones core H - h²)")
            col3.metric("Índice M", f"{m_index:.3f}",
                       help="H-index normalizado por años de actividad: H / años de carrera")
            
            # Comparativa visual de índices
            st.markdown("#### Comparativa Visual de Índices")
            indices_df = pd.DataFrame({
                'Índice': ['H', 'G', 'E', 'M'],
                'Valor': [h_index, g_index, e_index, m_index * 10]  # Multiplicar M por 10 para mejor visualización
            })
            
            fig_indices = go.Figure()
            fig_indices.add_trace(go.Bar(
                x=indices_df['Índice'],
                y=indices_df['Valor'],
                marker_color=['blue', 'green', 'orange', 'purple'],
                hovertemplate='<b>%{x}-index</b><br>Valor: %{y:.2f}<extra></extra>'
            ))
            fig_indices.update_layout(
                title='Comparativa de Índices de Impacto',
                xaxis_title='Índice',
                yaxis_title='Valor',
                height=400
            )
            st.plotly_chart(fig_indices, width='stretch')
            st.caption("Nota: El índice M está multiplicado por 10 para mejor visualización")
            
            # Información sobre las métricas
            with st.expander("Información sobre las métricas de impacto"):
                # Calcular años de carrera para la explicación del M-index
                try:
                    years_data = pd.to_numeric(df_filtered[year_col], errors='coerce').dropna()
                    career_years = years_data.max() - years_data.min() + 1 if len(years_data) > 0 else 0
                except:
                    career_years = 0
                
                st.markdown(f"""
                **Índice H**: Un investigador tiene índice h si h de sus publicaciones tienen al menos h citaciones cada una.
                - Mide tanto productividad como impacto
                - Un h-index de {h_index} significa que hay {h_index} artículos con al menos {h_index} citaciones
                - Limita la atención solo a los artículos del "core" del índice H
                
                **Índice G**: Complementa al H-index, considera el impacto acumulado.
                - Mayor número g tal que los g artículos más citados tienen al menos g² citaciones en total
                - Un g-index de {g_index:.0f} significa que los {g_index:.0f} artículos más citados tienen al menos {g_index:.0f}² = {g_index*g_index:.0f} citaciones acumuladas
                - Da más peso a los artículos altamente citados que el H-index
                - Generalmente g-index ≥ h-index
                
                **Índice E**: Mide el exceso de citaciones sobre el H-index.
                - Se calcula como: E-index = √(suma de citaciones del core H - h²)
                - Un E-index de {e_index:.2f} indica que hay un exceso de {e_index:.2f}² = {e_index*e_index:.2f} citaciones sobre el mínimo requerido para el H-index
                - Complementa el H-index mostrando la intensidad de las citaciones en el núcleo
                - Cuanto mayor sea el E-index, mayor es la concentración de citaciones en los artículos del core H
                
                **Índice M**: H-index normalizado por años de actividad (también conocido como m-quotient).
                - Se calcula como: M-index = H-index / años de carrera
                - Un M-index de {m_index:.3f} indica un H-index de {h_index} distribuido a lo largo de {career_years} años de actividad
                - Permite comparar investigadores con diferentes longitudes de carrera
                - Útil para identificar investigadores jóvenes con alto impacto potencial
                - Un M-index > 1 indica alta productividad e impacto sostenido
                
                **Factor de Impacto**: Promedio de citaciones recibidas por artículos publicados en los últimos 2 años.
                - Se calcula: (Citaciones a artículos de últimos 2 años) / (Número de artículos publicados en esos 2 años)
                - Indica el impacto promedio reciente de las publicaciones
                - Mide la frescura y relevancia actual del trabajo
                
                **Índice i10**: Número de publicaciones con al menos 10 citaciones.
                - Útil para identificar trabajos con impacto significativo
                - Mide la consistencia del impacto, no solo los trabajos más destacados
                - Complementa al H-index dando una visión del número de trabajos de alto impacto
                
                **Comparativa de Índices**:
                - **H-index**: Equilibrio entre productividad e impacto, pero limita la atención al core
                - **G-index**: Mayor sensibilidad a los artículos altamente citados, considera impacto acumulado
                - **E-index**: Mide la intensidad de las citaciones en el núcleo del H-index
                - **M-index**: Normaliza el H-index por tiempo, permite comparaciones justas entre investigadores
                """)
            
            # Top artículos más citados
            st.subheader("Top 10 Artículos Más Citados")
            title_col = next((c for c in df_filtered.columns if 'title' in c.lower()), None)
            if title_col:
                top_cited = df_filtered.nlargest(10, 'citations_num')[[title_col, author_col, year_col, citations_col]]
                top_cited.columns = ['Título', 'Autores', 'Año', 'Citaciones']
                st.dataframe(top_cited, use_container_width=True)
            
            # Top artículos más relevantes según el índice H (usando módulo de métricas)
            st.subheader("Top 10 Artículos Más Relevantes Según el Índice H")
            df_h_relevant, core_count = get_top_h_relevant_articles(
                df_filtered, citations_col, year_col, author_col, h_index, top_n=10
            )
            
            if df_h_relevant is not None and len(df_h_relevant) > 0:
                st.dataframe(df_h_relevant, use_container_width=True)
                
                if core_count > 0:
                    st.info(f"**{core_count} de estos artículos** están en el núcleo del índice H (h={h_index}). "
                           f"Son aquellos que tienen al menos tantas citaciones como su posición en el ranking, "
                           f"y contribuyen directamente al cálculo del índice H.")
                else:
                    st.info(f"Estos son los **{len(df_h_relevant)} artículos más citados** "
                           f"ordenados por relevancia. El índice H actual es {h_index}.")
            else:
                st.warning("No se encontraron artículos con citaciones suficientes para el análisis del índice H.")
            
            # Distribución de citaciones
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_cit = go.Figure()
                # Usar percentil 95 para filtrar outliers
                percentile_95 = citations_data.quantile(0.95)
                fig_cit.add_trace(go.Histogram(
                    x=citations_data[citations_data <= percentile_95],
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
    
    # Análisis de colaboración mejorado
    st.markdown("---")
    st.subheader("Análisis de Colaboración Avanzado")
    
    if author_col:
        # Coeficiente de colaboración
        cc = calculate_collaboration_coefficient(df_filtered, author_col)
        
        # Evolución temporal del CC
        cc_by_year = {}
        if year_col:
            for year in sorted(df_filtered[year_col].dropna().unique()):
                df_year = df_filtered[df_filtered[year_col] == year]
                cc_year = calculate_collaboration_coefficient(df_year, author_col)
                cc_by_year[year] = cc_year
        
        # Índice de colaboración internacional
        ici = calculate_international_collaboration_index(df_filtered, aff_col) if aff_col else 0.0
        
        # Promedio de autores por publicación
        avg_authors_per_pub = len(all_authors) / len(df_filtered) if len(df_filtered) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Coeficiente de Colaboración (CC)", f"{cc:.2%}",
                   help="Proporción de publicaciones colaborativas (más de 1 autor)")
        col2.metric("Índice Colaboración Internacional (ICI)", f"{ici:.2%}",
                   help="Proporción de publicaciones con múltiples países")
        col3.metric("Promedio Autores/Pub", f"{avg_authors_per_pub:.2f}")
        col4.metric("Índice Colaboración", f"{collab_index:.2f}")
        
        # Evolución temporal del CC
        if cc_by_year and len(cc_by_year) > 1:
            st.markdown("#### Evolución Temporal del Coeficiente de Colaboración")
            cc_df = pd.DataFrame(list(cc_by_year.items()), columns=['Año', 'CC'])
            cc_df = cc_df.sort_values('Año')
            
            fig_cc = go.Figure()
            fig_cc.add_trace(go.Scatter(
                x=cc_df['Año'],
                y=cc_df['CC'] * 100,
                mode='lines+markers',
                name='CC (%)',
                marker_color='teal',
                hovertemplate='Año: %{x}<br>CC: %{y:.2f}%<extra></extra>'
            ))
            fig_cc.update_layout(
                title='Evolución del Coeficiente de Colaboración',
                xaxis_title='Año',
                yaxis_title='Coeficiente de Colaboración (%)',
                height=400
            )
            st.plotly_chart(fig_cc, width='stretch')
    
    # Índice de Price
    st.markdown("---")
    st.subheader("Índice de Price (Frescura de la Literatura)")
    
    price_index_5 = calculate_price_index(df_filtered, year_col, years=5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Índice de Price (5 años)", f"{price_index_5:.2f}%",
                 help="Porcentaje de publicaciones en los últimos 5 años")
    
    with col2:
        if price_index_5 >= 50:
            st.success("📈 Alta frescura: más del 50% de las publicaciones son recientes")
        elif price_index_5 >= 30:
            st.info("📊 Frescura moderada: entre 30-50% de publicaciones recientes")
        else:
            st.warning("📉 Baja frescura: menos del 30% de publicaciones recientes")
    
    # Análisis de acceso abierto
    if oa_col:
        st.markdown("---")
        st.subheader("Análisis de Acceso Abierto")
        
        # Distribución OA vs No-OA
        oa_counts = df_filtered[oa_col].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_oa = go.Figure(data=[go.Pie(
                labels=oa_counts.index,
                values=oa_counts.values,
                hovertemplate='<b>%{label}</b><br>Publicaciones: %{value}<br>Porcentaje: %{percent}<extra></extra>'
            )])
            fig_oa.update_layout(title='Distribución OA vs No-OA', height=400)
            st.plotly_chart(fig_oa, width='stretch')
        
        with col2:
            if citations_col:
                # Comparativa de impacto: citaciones promedio OA vs No-OA
                oa_with_cites = df_filtered[df_filtered[oa_col].notna()].copy()
                oa_with_cites['citations_num'] = pd.to_numeric(oa_with_cites[citations_col], errors='coerce')
                
                oa_citations = {}
                for oa_type in oa_counts.index:
                    oa_pubs = oa_with_cites[oa_with_cites[oa_col] == oa_type]
                    avg_cites = oa_pubs['citations_num'].mean() if len(oa_pubs) > 0 else 0
                    oa_citations[oa_type] = avg_cites
                
                if len(oa_citations) >= 2:
                    oa_types = list(oa_citations.keys())
                    avg_cites_list = [oa_citations[t] for t in oa_types]
                    
                    fig_oa_impact = go.Figure(go.Bar(
                        x=oa_types,
                        y=avg_cites_list,
                        marker_color=['green', 'blue'],
                        hovertemplate='<b>%{x}</b><br>Promedio citaciones: %{y:.2f}<extra></extra>'
                    ))
                    fig_oa_impact.update_layout(
                        title='Impacto Promedio: OA vs No-OA',
                        xaxis_title='Tipo de Acceso',
                        yaxis_title='Promedio de Citaciones',
                        height=350
                    )
                    st.plotly_chart(fig_oa_impact, width='stretch')
                    
                    # Diferencia porcentual
                    if len(avg_cites_list) == 2 and avg_cites_list[1] > 0:
                        diff_pct = ((avg_cites_list[0] - avg_cites_list[1]) / avg_cites_list[1]) * 100
                        st.metric("Diferencia porcentual", f"{diff_pct:+.1f}%")
    
    # Análisis de financiamiento
    if funding_col:
        st.markdown("---")
        st.subheader("Análisis de Financiamiento")
        
        # Artículos con/sin financiamiento
        funded = df_filtered[funding_col].notna() & (df_filtered[funding_col].astype(str).str.lower() != 'nan')
        funded_count = funded.sum()
        not_funded_count = len(df_filtered) - funded_count
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            funding_counts = pd.Series({
                'Con Financiamiento': funded_count,
                'Sin Financiamiento': not_funded_count
            })
            
            fig_fund = go.Figure(data=[go.Pie(
                labels=funding_counts.index,
                values=funding_counts.values,
                hovertemplate='<b>%{label}</b><br>Publicaciones: %{value}<br>Porcentaje: %{percent}<extra></extra>'
            )])
            fig_fund.update_layout(title='Distribución de Financiamiento', height=400)
            st.plotly_chart(fig_fund, width='stretch')
        
        with col2:
            if citations_col:
                # Comparativa de impacto
                df_funding = df_filtered.copy()
                df_funding['citations_num'] = pd.to_numeric(df_funding[citations_col], errors='coerce')
                df_funding['Funded'] = funded
                
                funded_cites = df_funding[df_funding['Funded']]['citations_num'].mean()
                not_funded_cites = df_funding[~df_funding['Funded']]['citations_num'].mean()
                
                fig_fund_impact = go.Figure(go.Bar(
                    x=['Con Financiamiento', 'Sin Financiamiento'],
                    y=[funded_cites, not_funded_cites],
                    marker_color=['orange', 'gray'],
                    hovertemplate='<b>%{x}</b><br>Promedio citaciones: %{y:.2f}<extra></extra>'
                ))
                fig_fund_impact.update_layout(
                    title='Impacto: Financiado vs No Financiado',
                    xaxis_title='Tipo',
                    yaxis_title='Promedio de Citaciones',
                    height=350
                )
                st.plotly_chart(fig_fund_impact, width='stretch')
    
    # Índices de productividad
    st.markdown("---")
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
    
    # Velocidad de citación temporal
    if citations_col and year_col:
        st.markdown("---")
        st.subheader("Velocidad de Citación Temporal")
        
        # Calcular promedio de citaciones por año (últimos 10 años)
        df_cites_temporal = df_filtered.copy()
        df_cites_temporal['citations_num'] = pd.to_numeric(df_cites_temporal[citations_col], errors='coerce')
        df_cites_temporal['year_num'] = pd.to_numeric(df_cites_temporal[year_col], errors='coerce')
        
        # Filtrar últimos 10 años
        if len(df_cites_temporal[df_cites_temporal['year_num'].notna()]) > 0:
            max_year = df_cites_temporal['year_num'].max()
            min_year = max(max_year - 9, df_cites_temporal['year_num'].min())
            df_recent = df_cites_temporal[(df_cites_temporal['year_num'] >= min_year) & 
                                         (df_cites_temporal['year_num'] <= max_year)]
            
            # Agrupar por año y calcular promedio de citaciones
            citations_by_year = df_recent.groupby('year_num')['citations_num'].agg(['mean', 'sum', 'count']).reset_index()
            citations_by_year.columns = ['Año', 'Promedio Citaciones', 'Total Citaciones', 'Número Publicaciones']
            citations_by_year = citations_by_year.sort_values('Año')
            
            if len(citations_by_year) > 0:
                # Visualización de tendencia temporal
                fig_cite_speed = go.Figure()
                fig_cite_speed.add_trace(go.Scatter(
                    x=citations_by_year['Año'],
                    y=citations_by_year['Promedio Citaciones'],
                    mode='lines+markers',
                    name='Promedio de Citaciones',
                    marker_color='blue',
                    hovertemplate='<b>Año %{x}</b><br>Promedio: %{y:.2f} citaciones<br>Total: %{customdata[0]}<br>Publicaciones: %{customdata[1]}<extra></extra>',
                    customdata=citations_by_year[['Total Citaciones', 'Número Publicaciones']].values
                ))
                fig_cite_speed.update_layout(
                    title='Promedio de Citaciones por Año (Últimos 10 años)',
                    xaxis_title='Año',
                    yaxis_title='Promedio de Citaciones',
                    height=400
                )
                st.plotly_chart(fig_cite_speed, width='stretch')
                
                # Tabla de datos
                with st.expander("Ver datos detallados de velocidad de citación"):
                    st.dataframe(citations_by_year, use_container_width=True)
    
    # Predicción simple de publicaciones
    if year_col and len(pubs_per_year) >= 3:
        st.subheader("Proyección de Publicaciones")
        
        years_num = pubs_per_year.index.astype(int).values
        pubs_values = pubs_per_year.values
        
        # Regresión lineal simple
        z = np.polyfit(years_num, pubs_values, 1)
        p = np.poly1d(z)
        
        # Proyectar 20 años
        future_years = np.arange(years_num[-1] + 1, years_num[-1] + 21)
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
        
        # Métricas de proyección a 10, 15 y 20 años
        last_year_pubs = pubs_values[-1]
        year_10_proj = p(years_num[-1] + 10)
        year_15_proj = p(years_num[-1] + 15)
        year_20_proj = p(years_num[-1] + 20)
        
        diff_10 = year_10_proj - last_year_pubs
        diff_15 = year_15_proj - last_year_pubs
        diff_20 = year_20_proj - last_year_pubs
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_10 = f"+{diff_10:.0f}" if diff_10 > 0 else f"{diff_10:.0f}" if diff_10 < 0 else "0"
            direction_10 = "Sube" if diff_10 > 0 else "Baja" if diff_10 < 0 else "Igual"
            col1.metric(
                f"Proyección a 10 años ({years_num[-1] + 10})",
                f"{year_10_proj:.0f} publicaciones",
                delta=f"{delta_10} ({direction_10})",
                delta_color="normal" if diff_10 > 0 else "inverse" if diff_10 < 0 else "off"
            )
        
        with col2:
            delta_15 = f"+{diff_15:.0f}" if diff_15 > 0 else f"{diff_15:.0f}" if diff_15 < 0 else "0"
            direction_15 = "Sube" if diff_15 > 0 else "Baja" if diff_15 < 0 else "Igual"
            col2.metric(
                f"Proyección a 15 años ({years_num[-1] + 15})",
                f"{year_15_proj:.0f} publicaciones",
                delta=f"{delta_15} ({direction_15})",
                delta_color="normal" if diff_15 > 0 else "inverse" if diff_15 < 0 else "off"
            )
        
        with col3:
            delta_20 = f"+{diff_20:.0f}" if diff_20 > 0 else f"{diff_20:.0f}" if diff_20 < 0 else "0"
            direction_20 = "Sube" if diff_20 > 0 else "Baja" if diff_20 < 0 else "Igual"
            col3.metric(
                f"Proyección a 20 años ({years_num[-1] + 20})",
                f"{year_20_proj:.0f} publicaciones",
                delta=f"{delta_20} ({direction_20})",
                delta_color="normal" if diff_20 > 0 else "inverse" if diff_20 < 0 else "off"
            )
    
    # Análisis de contenido (Abstracts)
    if abstract_col:
        st.markdown("---")
        st.subheader("Análisis de Contenido (Abstracts)")
        
        # Obtener abstracts válidos
        abstracts = df_filtered[abstract_col].dropna().astype(str)
        abstracts = abstracts[abstracts.str.lower() != 'nan']
        
        if len(abstracts) > 0:
            # Longitud promedio de abstracts
            abstract_lengths = abstracts.str.split().str.len()
            avg_length = abstract_lengths.mean()
            median_length = abstract_lengths.median()
            
            col1, col2 = st.columns(2)
            col1.metric("Longitud Promedio de Abstracts", f"{avg_length:.0f} palabras")
            col2.metric("Mediana de Palabras", f"{median_length:.0f} palabras")
            
            # Palabras más frecuentes en abstracts (Top 30)
            st.markdown("#### Palabras Más Frecuentes en Abstracts (Top 30)")
            
            # Unir todos los abstracts
            all_text = ' '.join(abstracts.str.lower())
            
            # Limpiar y tokenizar
            # Eliminar puntuación y números, mantener solo palabras
            words = re.findall(r'\b[a-záéíóúñ]{3,}\b', all_text)
            
            # Filtrar palabras comunes (stopwords en español e inglés)
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'con', 'por',
                'para', 'que', 'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel',
                'aquella', 'aquellos', 'aquellas', 'the', 'this', 'that', 'these', 'those', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'must', 'can', 'not', 'from', 'as', 'its',
                'it', 'which', 'who', 'when', 'where', 'what', 'why', 'how', 'all', 'each', 'every',
                'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                'now', 'use', 'also', 'new', 'paper', 'study', 'results', 'research', 'method', 'data',
                'analysis', 'article', 'abstract', 'introduction', 'conclusion', 'present', 'show',
                'propose', 'present', 'presented', 'proposed', 'shown', 'based', 'using', 'used'
            }
            
            words_filtered = [w for w in words if w not in stopwords and len(w) > 3]
            
            if words_filtered:
                word_counter = Counter(words_filtered)
                top_words = word_counter.most_common(30)
                
                # Visualización
                words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
                
                fig_words = go.Figure(go.Bar(
                    x=words_df['Frecuencia'][::-1],
                    y=words_df['Palabra'][::-1],
                    orientation='h',
                    marker_color='steelblue',
                    hovertemplate='<b>%{y}</b><br>Frecuencia: %{x}<extra></extra>'
                ))
                fig_words.update_layout(
                    title='Top 30 Palabras Más Frecuentes en Abstracts',
                    xaxis_title='Frecuencia',
                    yaxis_title='Palabra',
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_words, width='stretch')
                
                # Tabla de palabras
                with st.expander("Ver tabla completa de palabras (Top 30)"):
                    st.dataframe(words_df, use_container_width=True)
                
                # Análisis de texto básico
                st.markdown("#### Análisis de Texto Básico")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Abstracts", len(abstracts))
                col2.metric("Palabras Únicas", len(set(words_filtered)))
                col3.metric("Total Palabras", len(words_filtered))
                col4.metric("Palabras/Abstract Promedio", f"{len(words_filtered)/len(abstracts):.0f}")
    
    # Insights automáticos
    st.markdown("---")
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