"""
Funciones de procesamiento de datos y paralelización.
"""
import math
import multiprocessing as mp
import pandas as pd
from .normalizers import (
    split_authors,
    split_keywords,
    extract_countries_from_affiliation,
    extract_institutions_from_affiliation
)


def process_chunk_authors(series_chunk):
    """Procesa un chunk de autores en paralelo."""
    result = []
    for val in series_chunk:
        result.extend(split_authors(val))
    return result


def process_chunk_keywords(series_chunk):
    """Procesa un chunk de keywords en paralelo."""
    result = []
    for val in series_chunk:
        result.extend(split_keywords(val))
    return result


def process_chunk_countries(series_chunk):
    """Procesa un chunk de países en paralelo."""
    result = []
    for val in series_chunk:
        result.extend(extract_countries_from_affiliation(val))
    return result


def process_chunk_institutions(series_chunk):
    """Procesa un chunk de instituciones en paralelo."""
    result = []
    for val in series_chunk:
        result.extend(extract_institutions_from_affiliation(val))
    return result


def parallel_map_series(series, worker_func, n_cores):
    """Ejecuta una función worker en paralelo sobre una serie de pandas."""
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


def detect_columns(df):
    """Detecta y retorna las columnas relevantes del DataFrame."""
    df.columns = [c.strip() for c in df.columns]
    
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
    
    return {
        'author_col': author_col,
        'author_full_col': author_full_col,
        'keywords_cols': keywords_cols,
        'aff_col': aff_col,
        'year_col': year_col,
        'source_col': source_col,
        'doctype_col': doctype_col,
        'citations_col': citations_col,
        'language_col': language_col,
        'funding_col': funding_col
    }


def load_data(file_path):
    """Carga el archivo CSV y retorna el DataFrame."""
    try:
        df = pd.read_csv(file_path, dtype=str)
        return df, None
    except FileNotFoundError:
        return None, f"No se encontró el archivo '{file_path}'."
    except Exception as e:
        return None, f"Error al cargar el archivo: {str(e)}"

