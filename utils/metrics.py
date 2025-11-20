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

