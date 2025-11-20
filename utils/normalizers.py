"""
Funciones de normalización de datos bibliométricos.
"""
import pandas as pd
import re


def normalize_author_name(name):
    """Normaliza el nombre de un autor."""
    if not name or pd.isna(name):
        return None
    name = str(name).strip()
    name = re.sub(r'\.(?=\s|$)', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.title()
    return name if name else None


def normalize_keyword(keyword):
    """Normaliza una palabra clave."""
    if not keyword or pd.isna(keyword):
        return None
    kw = str(keyword).strip().lower()
    kw = re.sub(r'^[^\w]+|[^\w]+$', '', kw)
    kw = re.sub(r'[-\s]+', ' ', kw)
    return kw if kw else None


def normalize_country(country):
    """Normaliza el nombre de un país."""
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
    """Normaliza el nombre de una institución."""
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
    """Normaliza el nombre de una fuente/jornal."""
    if not source or pd.isna(source):
        return None
    source = str(source).strip()
    source = re.sub(r'\s+', ' ', source)
    return source.title() if source else None


def split_authors(text):
    """Divide un texto de autores en una lista de autores normalizados."""
    if not text or pd.isna(text):
        return []
    separators = [",", ";", " and "]
    s = str(text)
    for sep in separators:
        s = s.replace(sep, "|")
    parts = [normalize_author_name(p) for p in s.split("|") if p.strip()]
    return [p for p in parts if p]


def split_keywords(text):
    """Divide un texto de keywords en una lista de keywords normalizadas."""
    if not text or pd.isna(text):
        return []
    parts = [normalize_keyword(p) for p in str(text).replace(";", ",").split(",") if p.strip()]
    return [p for p in parts if p]


def extract_countries_from_affiliation(text):
    """Extrae países de un texto de afiliación."""
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
    """Extrae instituciones de un texto de afiliación."""
    if not text or pd.isna(text):
        return []
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    institutions = []
    for p in parts:
        inst = normalize_institution(p)
        if inst:
            institutions.append(inst)
    return list(dict.fromkeys(institutions))

