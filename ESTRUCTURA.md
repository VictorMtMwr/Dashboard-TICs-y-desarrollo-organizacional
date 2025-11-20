# Estructura Modular del Dashboard Bibliométrico

Este proyecto ha sido modularizado para mejorar la organización y mantenibilidad del código.

## Estructura de Carpetas

```
Dashboard-TICs-y-desarrollo-organizacional/
├── app.py                    # Archivo principal de Streamlit
├── script.py                 # Script original (respaldo)
├── config.py                 # Configuraciones y constantes
├── requirements.txt          # Dependencias del proyecto
├── data/
│   └── data.csv             # Datos bibliométricos
├── utils/                    # Módulos de utilidades
│   ├── __init__.py          # Inicialización del paquete
│   ├── normalizers.py       # Funciones de normalización
│   ├── data_processing.py   # Procesamiento de datos y paralelización
│   └── metrics.py           # Cálculos de métricas bibliométricas
└── README.md                # Documentación del proyecto
```

## Descripción de Módulos

### `config.py`
Contiene todas las constantes y configuraciones del dashboard:
- Parámetros configurables (CPU_USAGE_RATIO, MAX_NETWORK_NODES, etc.)
- Rutas de archivos
- Configuración de Streamlit

### `utils/normalizers.py`
Funciones para normalizar y limpiar datos bibliométricos:
- `normalize_author_name()` - Normaliza nombres de autores
- `normalize_keyword()` - Normaliza palabras clave
- `normalize_country()` - Normaliza nombres de países
- `country_to_iso3()` - Convierte nombres de países a códigos ISO-3
- `normalize_institution()` - Normaliza nombres de instituciones
- `normalize_source()` - Normaliza nombres de fuentes/jornales
- `split_authors()` - Divide texto de autores en lista
- `split_keywords()` - Divide texto de keywords en lista
- `extract_countries_from_affiliation()` - Extrae países de afiliaciones
- `extract_institutions_from_affiliation()` - Extrae instituciones de afiliaciones

### `utils/data_processing.py`
Funciones de procesamiento de datos y paralelización:
- `process_chunk_authors()` - Procesa chunk de autores en paralelo
- `process_chunk_keywords()` - Procesa chunk de keywords en paralelo
- `process_chunk_countries()` - Procesa chunk de países en paralelo
- `process_chunk_institutions()` - Procesa chunk de instituciones en paralelo
- `parallel_map_series()` - Ejecuta función worker en paralelo sobre serie
- `detect_columns()` - Detecta columnas relevantes del DataFrame
- `load_data()` - Carga el archivo CSV

### `utils/metrics.py`
Funciones para calcular métricas bibliométricas:
- `calculate_h_index()` - Calcula el índice H
- `calculate_impact_factor()` - Calcula el factor de impacto
- `calculate_i10_index()` - Calcula el índice i10
- `get_citation_percentiles()` - Obtiene percentiles de citaciones
- `get_top_h_relevant_articles()` - Obtiene artículos más relevantes según índice H

### `app.py`
Archivo principal de Streamlit que:
- Importa y utiliza los módulos creados
- Define la interfaz del dashboard
- Maneja las pestañas y visualizaciones
- Coordina el procesamiento de datos

## Cómo Usar

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el dashboard:**
   ```bash
   streamlit run app.py
   ```

3. **Ejecutar el script original (respaldo):**
   ```bash
   streamlit run script.py
   ```

## Ventajas de la Estructura Modular

1. **Mantenibilidad**: El código está organizado en módulos lógicos
2. **Reutilización**: Las funciones pueden ser importadas y reutilizadas
3. **Testabilidad**: Cada módulo puede ser probado independientemente
4. **Escalabilidad**: Fácil agregar nuevas funcionalidades sin modificar código existente
5. **Legibilidad**: Código más fácil de leer y entender

## Notas

- El archivo `script.py` se mantiene como respaldo del código original
- Todos los módulos están en Python estándar y no requieren instalación adicional
- La estructura es compatible con Streamlit y puede ser desplegada en servicios como Streamlit Cloud

