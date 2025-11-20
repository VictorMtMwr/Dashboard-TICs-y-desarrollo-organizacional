"""
Funciones para el cálculo de métricas bibliométricas.
"""
import pandas as pd
import numpy as np
from collections import Counter


def calculate_h_index(citations_data):
    """Calcula el índice H basado en un array de citaciones."""
    if len(citations_data) == 0:
        return 0
    
    citations_sorted = sorted(citations_data.values if hasattr(citations_data, 'values') else citations_data, reverse=True)
    h_index = 0
    
    for i, cites in enumerate(citations_sorted, 1):
        if cites >= i:
            h_index = i
        else:
            break
    
    return h_index, citations_sorted


def calculate_impact_factor(df_filtered, citations_col, year_col):
    """Calcula el factor de impacto basado en los últimos 2 años."""
    if not year_col or not citations_col:
        return 0
    
    try:
        current_year = int(max(df_filtered[year_col].dropna()))
        recent_years = [str(current_year), str(current_year - 1)]
        
        # Publicaciones en últimos 2 años
        recent_pubs = df_filtered[df_filtered[year_col].isin(recent_years)]
        num_recent_pubs = len(recent_pubs)
        
        # Citaciones recibidas por esas publicaciones
        if num_recent_pubs > 0:
            recent_citations = pd.to_numeric(recent_pubs[citations_col], errors='coerce').sum()
            impact_factor = recent_citations / num_recent_pubs
            return impact_factor
    except:
        pass
    
    return 0


def calculate_i10_index(citations_data):
    """Calcula el índice i10 (artículos con al menos 10 citaciones)."""
    return len([c for c in citations_data if c >= 10])


def get_citation_percentiles(citations_data):
    """Calcula los percentiles de citaciones."""
    return {
        'p25': citations_data.quantile(0.25),
        'p50': citations_data.quantile(0.50),
        'p75': citations_data.quantile(0.75),
        'p90': citations_data.quantile(0.90)
    }


def get_top_h_relevant_articles(df_filtered, citations_col, year_col, author_col, h_index, top_n=10):
    """Obtiene los artículos más relevantes según el índice H."""
    title_col = next((c for c in df_filtered.columns if 'title' in c.lower()), None)
    
    if not title_col or citations_col not in df_filtered.columns:
        return None
    
    # Crear DataFrame con citaciones y filtrar los que tienen citaciones válidas
    df_with_citations = df_filtered[df_filtered['citations_num'].notna()].copy()
    df_with_citations = df_with_citations.sort_values('citations_num', ascending=False).reset_index(drop=True)
    
    h_relevant_articles = []
    for idx, row in df_with_citations.iterrows():
        position = idx + 1  # Posición 1-indexed
        citations = int(row['citations_num'])
        
        # Los artículos en el core del índice H
        if position <= h_index and citations >= position:
            h_relevant_articles.append({
                'Posición en Índice H': position,
                'Título': row[title_col] if title_col else 'N/A',
                'Autores': row[author_col] if author_col else 'N/A',
                'Año': row[year_col] if year_col else 'N/A',
                'Citaciones': citations
            })
        # Si no tenemos suficientes del core, agregamos los más citados que siguen
        elif len(h_relevant_articles) < top_n:
            h_relevant_articles.append({
                'Posición en Índice H': position,
                'Título': row[title_col] if title_col else 'N/A',
                'Autores': row[author_col] if author_col else 'N/A',
                'Año': row[year_col] if year_col else 'N/A',
                'Citaciones': citations
            })
        
        # Limitar a los top N
        if len(h_relevant_articles) >= top_n:
            break
    
    if h_relevant_articles:
        df_h_relevant = pd.DataFrame(h_relevant_articles)
        core_count = sum(1 for art in h_relevant_articles if art['Posición en Índice H'] <= h_index)
        return df_h_relevant, core_count
    
    return None, 0


def calculate_g_index(citations_data):
    """Calcula el índice G basado en un array de citaciones.
    El índice G es el mayor número g tal que los g artículos más citados 
    tienen al menos g^2 citaciones en total."""
    if len(citations_data) == 0:
        return 0
    
    citations_sorted = sorted(citations_data.values if hasattr(citations_data, 'values') else citations_data, reverse=True)
    g_index = 0
    cumulative_citations = 0
    
    for i, cites in enumerate(citations_sorted, 1):
        cumulative_citations += cites
        if cumulative_citations >= i * i:
            g_index = i
        else:
            break
    
    return g_index


def calculate_e_index(citations_data, h_index):
    """Calcula el índice E (exceso de citaciones sobre el H-index).
    E-index = sqrt(sum(citaciones de artículos en el core H) - h^2)"""
    if len(citations_data) == 0 or h_index == 0:
        return 0.0
    
    citations_sorted = sorted(citations_data.values if hasattr(citations_data, 'values') else citations_data, reverse=True)
    
    # Sumar citaciones de los primeros h artículos
    h_core_citations = sum(citations_sorted[:h_index])
    
    # E-index = sqrt(h_core_citations - h^2)
    excess = h_core_citations - (h_index * h_index)
    
    if excess > 0:
        return np.sqrt(excess)
    return 0.0


def calculate_m_index(h_index, year_col, df_filtered):
    """Calcula el índice M (H-index normalizado por años de actividad).
    M-index = H-index / años de carrera"""
    if h_index == 0 or not year_col:
        return 0.0
    
    try:
        # Calcular años de actividad
        years = pd.to_numeric(df_filtered[year_col], errors='coerce').dropna()
        if len(years) == 0:
            return 0.0
        
        career_years = years.max() - years.min() + 1  # +1 para incluir ambos años
        if career_years == 0:
            return 0.0
        
        m_index = h_index / career_years
        return m_index
    except:
        return 0.0


def calculate_collaboration_coefficient(df_filtered, author_col):
    """Calcula el coeficiente de colaboración (CC).
    CC = número de publicaciones colaborativas / total publicaciones"""
    if not author_col:
        return 0.0
    
    total_pubs = len(df_filtered)
    if total_pubs == 0:
        return 0.0
    
    # Publicaciones colaborativas (más de 1 autor)
    collaborative_pubs = 0
    for idx, row in df_filtered.iterrows():
        authors_text = row[author_col] if pd.notna(row[author_col]) else ""
        if pd.notna(authors_text):
            # Contar autores (asumiendo separación por ; o ,)
            authors_list = [a.strip() for a in str(authors_text).replace(';', ',').split(',') if a.strip()]
            if len(authors_list) > 1:
                collaborative_pubs += 1
    
    return collaborative_pubs / total_pubs if total_pubs > 0 else 0.0


def calculate_international_collaboration_index(df_filtered, aff_col):
    """Calcula el índice de colaboración internacional.
    ICI = número de publicaciones con múltiples países / total publicaciones"""
    if not aff_col:
        return 0.0
    
    total_pubs = len(df_filtered)
    if total_pubs == 0:
        return 0.0
    
    from utils.normalizers import extract_countries_from_affiliation
    
    international_pubs = 0
    for idx, row in df_filtered.iterrows():
        aff_text = row[aff_col] if pd.notna(row[aff_col]) else ""
        if pd.notna(aff_text):
            countries = extract_countries_from_affiliation(str(aff_text))
            countries = list(set(countries))  # Países únicos
            if len(countries) > 1:
                international_pubs += 1
    
    return international_pubs / total_pubs if total_pubs > 0 else 0.0


def calculate_price_index(df_filtered, year_col, years=5):
    """Calcula el índice de Price.
    Porcentaje de publicaciones en los últimos N años."""
    if not year_col:
        return 0.0
    
    try:
        total_pubs = len(df_filtered)
        if total_pubs == 0:
            return 0.0
        
        # Calcular el año más reciente
        years_data = pd.to_numeric(df_filtered[year_col], errors='coerce').dropna()
        if len(years_data) == 0:
            return 0.0
        
        current_year = years_data.max()
        threshold_year = current_year - years + 1
        
        # Contar publicaciones en los últimos N años
        recent_pubs = df_filtered[pd.to_numeric(df_filtered[year_col], errors='coerce') >= threshold_year]
        
        price_index = (len(recent_pubs) / total_pubs) * 100
        return price_index
    except:
        return 0.0


def calculate_author_h_index(author_name, df_filtered, citations_col, author_col):
    """Calcula el H-index individual para un autor específico."""
    # Filtrar publicaciones del autor
    author_pubs = df_filtered[df_filtered[author_col].str.contains(author_name, case=False, na=False)]
    
    if len(author_pubs) == 0:
        return 0
    
    # Obtener citaciones
    citations = pd.to_numeric(author_pubs[citations_col], errors='coerce').dropna()
    
    if len(citations) == 0:
        return 0
    
    h_index, _ = calculate_h_index(citations)
    return h_index

