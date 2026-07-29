"""
Funciones reutilizables para cargar, diagnosticar, limpiar y transformar
el dataset de movilidad.

El módulo está diseñado para utilizarse tanto en Google Colab como en
una aplicación de Streamlit.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Nombres genéricos de columnas
# ---------------------------------------------------------------------
COL_ID = "id_viaje"
COL_FECHA = "fecha_hora"
COL_CIUDAD = "ciudad"
COL_ORIGEN = "estacion_origen"
COL_DESTINO = "estacion_destino"
COL_USUARIO = "usuario_id"
COL_GENERO = "genero"
COL_EDAD = "edad"
COL_TIPO_USUARIO = "tipo_usuario"
COL_CATEGORIA = "tipo_vehiculo"
COL_DURACION = "duracion_min"
COL_DISTANCIA = "distancia_km"
COL_TARIFA = "tarifa_base"
COL_DESCUENTO = "descuento"
COL_VALOR = "valor_pagado"
COL_METODO_PAGO = "metodo_pago"
COL_CLIMA = "clima"
COL_TEMPERATURA = "temperatura_c"
COL_SATISFACCION = "satisfaccion"
COL_INCIDENTE = "incidente"

COLUMNAS_REQUERIDAS = [
    COL_ID,
    COL_FECHA,
    COL_CIUDAD,
    COL_ORIGEN,
    COL_DESTINO,
    COL_USUARIO,
    COL_GENERO,
    COL_EDAD,
    COL_TIPO_USUARIO,
    COL_CATEGORIA,
    COL_DURACION,
    COL_DISTANCIA,
    COL_TARIFA,
    COL_DESCUENTO,
    COL_VALOR,
    COL_METODO_PAGO,
    COL_CLIMA,
    COL_TEMPERATURA,
    COL_SATISFACCION,
    COL_INCIDENTE,
]

COLUMNAS_NUMERICAS = [
    COL_EDAD,
    COL_DURACION,
    COL_DISTANCIA,
    COL_TARIFA,
    COL_DESCUENTO,
    COL_VALOR,
    COL_TEMPERATURA,
    COL_SATISFACCION,
]

MESES_ES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]

DIAS_ES = [
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
]

MAPA_CIUDADES = {
    "bogota": "Bogotá",
    "medellin": "Medellín",
    "cali": "Cali",
    "barranquilla": "Barranquilla",
}


def validar_columnas(df: pd.DataFrame) -> None:
    """Comprueba que el DataFrame contenga las columnas necesarias."""
    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if faltantes:
        raise ValueError(
            "El archivo no contiene todas las columnas requeridas. "
            f"Faltan: {', '.join(faltantes)}"
        )


def cargar_datos(
    fuente: str | Path | BinaryIO | BytesIO,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Carga el CSV desde una ruta o un archivo subido."""
    try:
        df = pd.read_csv(fuente, encoding=encoding)
    except UnicodeDecodeError:
        if hasattr(fuente, "seek"):
            fuente.seek(0)
        df = pd.read_csv(fuente, encoding="latin-1")

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    validar_columnas(df)
    return df


def diagnosticar_datos(df: pd.DataFrame) -> dict[str, Any]:
    """Genera un diagnóstico básico de calidad del dataset."""
    validar_columnas(df)

    resumen_nulos = (
        pd.DataFrame(
            {
                "columna": df.columns,
                "tipo": [str(tipo) for tipo in df.dtypes],
                "nulos": df.isna().sum().values,
                "porcentaje_nulos": (
                    df.isna().mean().mul(100).round(2).values
                ),
                "valores_unicos": [
                    df[col].nunique(dropna=False) for col in df.columns
                ],
            }
        )
        .sort_values(["nulos", "columna"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "duplicados_completos": int(df.duplicated().sum()),
        "ids_duplicados": int(df.duplicated(subset=[COL_ID]).sum()),
        "resumen_columnas": resumen_nulos,
    }


def _clave_sin_tildes(serie: pd.Series) -> pd.Series:
    """Crea una clave minúscula y sin tildes para normalizar textos."""
    return (
        serie.astype("string")
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
    )


def _normalizar_ciudad(serie: pd.Series) -> pd.Series:
    original = serie.astype("string").str.strip()
    clave = _clave_sin_tildes(original)
    normalizada = clave.map(MAPA_CIUDADES)
    return normalizada.fillna(original.str.title())


def _normalizar_incidente(serie: pd.Series) -> pd.Series:
    clave = _clave_sin_tildes(serie)
    mapa = {
        "si": "Sí",
        "s": "Sí",
        "1": "Sí",
        "true": "Sí",
        "no": "No",
        "n": "No",
        "0": "No",
        "false": "No",
    }
    return clave.map(mapa).fillna(serie.astype("string").str.strip())


def convertir_fecha_mixta(serie: pd.Series) -> pd.Series:
    """
    Convierte fechas ISO y fechas colombianas DD/MM/AAAA.

    Los valores que comienzan con AAAA-MM-DD se interpretan como ISO.
    Los demás se convierten con dayfirst=True.
    """
    texto = serie.astype("string").str.strip()
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")

    es_iso = texto.str.match(r"^\d{4}-\d{1,2}-\d{1,2}", na=False)
    resultado.loc[es_iso] = pd.to_datetime(
        texto.loc[es_iso], errors="coerce", yearfirst=True
    )
    resultado.loc[~es_iso] = pd.to_datetime(
        texto.loc[~es_iso], errors="coerce", dayfirst=True
    )
    return resultado


def _imputar_mediana_por_grupos(
    df: pd.DataFrame,
    columna: str,
    grupos: list[str],
) -> pd.Series:
    """Imputa con mediana de grupo y luego con mediana global."""
    resultado = df[columna].copy()

    for nivel in range(len(grupos), 0, -1):
        columnas_grupo = grupos[:nivel]
        medianas = df.groupby(columnas_grupo, dropna=False)[columna].transform(
            "median"
        )
        resultado = resultado.fillna(medianas)

    return resultado.fillna(df[columna].median())


def _moda_segura(serie: pd.Series) -> Any:
    moda = serie.mode(dropna=True)
    return moda.iloc[0] if not moda.empty else pd.NA


def _limites_iqr(serie: pd.Series, factor: float = 1.5) -> tuple[float, float]:
    datos = pd.to_numeric(serie, errors="coerce").dropna()
    if datos.empty:
        return (np.nan, np.nan)

    q1 = float(datos.quantile(0.25))
    q3 = float(datos.quantile(0.75))
    iqr = q3 - q1
    return (q1 - factor * iqr, q3 + factor * iqr)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset sin eliminar automáticamente los valores atípicos válidos.

    Los outliers se conservan y se marcan mediante variables indicadoras.
    """
    validar_columnas(df)
    datos = df.copy()

    # 1. Espacios y duplicados
    columnas_texto = datos.select_dtypes(include=["object", "string"]).columns
    for columna in columnas_texto:
        datos[columna] = datos[columna].astype("string").str.strip()

    datos = datos.drop_duplicates().copy()
    datos = datos.drop_duplicates(subset=[COL_ID], keep="first").copy()

    # 2. Tipos de datos
    for columna in COLUMNAS_NUMERICAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos[COL_FECHA] = convertir_fecha_mixta(datos[COL_FECHA])

    # 3. Normalización de categorías
    datos[COL_CIUDAD] = _normalizar_ciudad(datos[COL_CIUDAD])
    datos[COL_INCIDENTE] = _normalizar_incidente(datos[COL_INCIDENTE])

    # 4. Validación de rangos
    datos.loc[~datos[COL_EDAD].between(14, 100), COL_EDAD] = np.nan
    datos.loc[datos[COL_DURACION] <= 0, COL_DURACION] = np.nan
    datos.loc[datos[COL_DISTANCIA] <= 0, COL_DISTANCIA] = np.nan
    datos.loc[datos[COL_TARIFA] < 0, COL_TARIFA] = np.nan
    datos.loc[~datos[COL_DESCUENTO].between(0, 1), COL_DESCUENTO] = np.nan
    datos.loc[datos[COL_VALOR] < 0, COL_VALOR] = np.nan
    datos.loc[~datos[COL_TEMPERATURA].between(-10, 50), COL_TEMPERATURA] = np.nan
    datos.loc[~datos[COL_SATISFACCION].between(1, 5), COL_SATISFACCION] = np.nan

    # Las filas sin información estructural indispensable no se pueden analizar.
    datos = datos.dropna(
        subset=[
            COL_ID,
            COL_FECHA,
            COL_CIUDAD,
            COL_ORIGEN,
            COL_DESTINO,
            COL_DURACION,
            COL_DISTANCIA,
            COL_TARIFA,
            COL_DESCUENTO,
            COL_VALOR,
        ]
    ).copy()

    # 5. Indicadores para identificar valores imputados
    datos["edad_fue_imputada"] = datos[COL_EDAD].isna()
    datos["clima_fue_imputado"] = datos[COL_CLIMA].isna()
    datos["temperatura_fue_imputada"] = datos[COL_TEMPERATURA].isna()
    datos["satisfaccion_fue_imputada"] = datos[COL_SATISFACCION].isna()

    # 6. Imputación de edad por ciudad y tipo de usuario
    datos[COL_EDAD] = _imputar_mediana_por_grupos(
        datos,
        COL_EDAD,
        [COL_CIUDAD, COL_TIPO_USUARIO],
    ).round()

    # 7. Imputación de clima por moda de ciudad
    moda_por_ciudad = datos.groupby(COL_CIUDAD)[COL_CLIMA].agg(_moda_segura)
    moda_global_clima = _moda_segura(datos[COL_CLIMA])
    datos[COL_CLIMA] = (
        datos[COL_CLIMA]
        .fillna(datos[COL_CIUDAD].map(moda_por_ciudad))
        .fillna(moda_global_clima)
    )

    # 8. Imputación de temperatura por ciudad y mes
    datos["_mes_auxiliar"] = datos[COL_FECHA].dt.month
    datos[COL_TEMPERATURA] = _imputar_mediana_por_grupos(
        datos,
        COL_TEMPERATURA,
        [COL_CIUDAD, "_mes_auxiliar"],
    ).round(1)
    datos = datos.drop(columns="_mes_auxiliar")

    # 9. Imputación de satisfacción por tipo de usuario y vehículo
    datos[COL_SATISFACCION] = _imputar_mediana_por_grupos(
        datos,
        COL_SATISFACCION,
        [COL_TIPO_USUARIO, COL_CATEGORIA],
    ).round().clip(1, 5)

    # 10. Coherencia entre estaciones
    datos["misma_estacion"] = datos[COL_ORIGEN].eq(datos[COL_DESTINO])

    # 11. Validación contable. Se marca, pero no se reemplaza el valor pagado.
    datos["valor_estimado"] = (
        datos[COL_TARIFA] * (1 - datos[COL_DESCUENTO])
    ).round(2)
    datos["diferencia_valor"] = (
        datos[COL_VALOR] - datos["valor_estimado"]
    ).round(2)
    datos["valor_consistente"] = np.isclose(
        datos[COL_VALOR],
        datos["valor_estimado"],
        rtol=0.02,
        atol=50,
    )

    return datos.reset_index(drop=True)


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables temporales, operativas y de segmentación."""
    datos = df.copy()

    datos["fecha"] = datos[COL_FECHA].dt.date
    datos["anio"] = datos[COL_FECHA].dt.year.astype("Int64")
    datos["mes_numero"] = datos[COL_FECHA].dt.month.astype("Int64")
    datos["mes"] = pd.Categorical(
        datos["mes_numero"].map(dict(enumerate(MESES_ES, start=1))),
        categories=MESES_ES,
        ordered=True,
    )
    datos["dia_mes"] = datos[COL_FECHA].dt.day.astype("Int64")
    datos["dia_semana_numero"] = datos[COL_FECHA].dt.dayofweek.astype("Int64")
    datos["dia_semana"] = pd.Categorical(
        datos["dia_semana_numero"].map(dict(enumerate(DIAS_ES))),
        categories=DIAS_ES,
        ordered=True,
    )
    datos["hora"] = datos[COL_FECHA].dt.hour.astype("Int64")
    datos["es_fin_semana"] = datos["dia_semana_numero"].isin([5, 6])
    datos["es_hora_pico"] = datos["hora"].between(6, 9) | datos["hora"].between(
        16, 19
    )

    datos["franja_horaria"] = pd.cut(
        datos["hora"].astype(float),
        bins=[-1, 5, 11, 17, 23],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
    )

    datos["velocidad_promedio_kmh"] = (
        datos[COL_DISTANCIA] / (datos[COL_DURACION] / 60)
    ).round(2)
    datos["descuento_valor"] = (
        datos[COL_TARIFA] * datos[COL_DESCUENTO]
    ).round(2)
    datos["ruta"] = (
        datos[COL_ORIGEN].astype(str)
        + " → "
        + datos[COL_DESTINO].astype(str)
    )
    datos["tiene_incidente"] = datos[COL_INCIDENTE].eq("Sí")

    datos["segmento_distancia"] = pd.cut(
        datos[COL_DISTANCIA],
        bins=[0, 3, 7, np.inf],
        labels=["Corta", "Media", "Larga"],
        include_lowest=True,
    )
    datos["segmento_duracion"] = pd.cut(
        datos[COL_DURACION],
        bins=[0, 15, 30, np.inf],
        labels=["Corta", "Media", "Larga"],
        include_lowest=True,
    )

    limite_inf_duracion, limite_sup_duracion = _limites_iqr(datos[COL_DURACION])
    limite_inf_distancia, limite_sup_distancia = _limites_iqr(datos[COL_DISTANCIA])

    datos["es_atipico_duracion"] = ~datos[COL_DURACION].between(
        limite_inf_duracion, limite_sup_duracion
    )
    datos["es_atipico_distancia"] = ~datos[COL_DISTANCIA].between(
        limite_inf_distancia, limite_sup_distancia
    )

    return datos


def procesar_datos(
    fuente: pd.DataFrame | str | Path | BinaryIO | BytesIO,
) -> pd.DataFrame:
    """Ejecuta carga, limpieza e ingeniería de variables."""
    if isinstance(fuente, pd.DataFrame):
        df_original = fuente.copy()
        validar_columnas(df_original)
    else:
        df_original = cargar_datos(fuente)

    df_limpio = limpiar_datos(df_original)
    return crear_variables_derivadas(df_limpio)


def calcular_indicadores(df: pd.DataFrame) -> dict[str, float | int]:
    """Calcula KPI generales para el dashboard."""
    total_viajes = int(len(df))
    incidentes = int(df["tiene_incidente"].sum()) if total_viajes else 0

    return {
        "total_viajes": total_viajes,
        "ingresos_totales": float(df[COL_VALOR].sum()),
        "duracion_promedio": float(df[COL_DURACION].mean()),
        "distancia_promedio": float(df[COL_DISTANCIA].mean()),
        "satisfaccion_promedio": float(df[COL_SATISFACCION].mean()),
        "incidentes": incidentes,
        "tasa_incidentes": (
            float(incidentes / total_viajes * 100) if total_viajes else 0.0
        ),
        "usuarios_unicos": int(df[COL_USUARIO].nunique()),
    }


def generar_resumen(
    df: pd.DataFrame,
    agrupador: str,
    *,
    metrica: str | None = None,
    operacion: str = "count",
    nombre_resultado: str = "resultado",
) -> pd.DataFrame:
    """
    Crea tablas de resumen genéricas para gráficos.

    operacion admite: count, sum, mean y nunique.
    """
    if agrupador not in df.columns:
        raise KeyError(f"No existe la columna de agrupación: {agrupador}")

    if operacion == "count":
        resumen = (
            df.groupby(agrupador, observed=False)
            .size()
            .rename(nombre_resultado)
            .reset_index()
        )
    else:
        if metrica is None or metrica not in df.columns:
            raise KeyError("Debe indicar una métrica válida.")
        if operacion not in {"sum", "mean", "nunique"}:
            raise ValueError("Operación no soportada.")
        resumen = (
            df.groupby(agrupador, observed=False)[metrica]
            .agg(operacion)
            .rename(nombre_resultado)
            .reset_index()
        )

    return resumen.sort_values(nombre_resultado, ascending=False).reset_index(
        drop=True
    )


def filtrar_datos(
    df: pd.DataFrame,
    *,
    fecha_inicio: Any | None = None,
    fecha_fin: Any | None = None,
    ciudades: list[str] | None = None,
    categorias: list[str] | None = None,
    tipos_usuario: list[str] | None = None,
    climas: list[str] | None = None,
    metodos_pago: list[str] | None = None,
    incidentes: list[str] | None = None,
) -> pd.DataFrame:
    """Aplica filtros comunes del dashboard."""
    datos = df.copy()

    if fecha_inicio is not None:
        datos = datos[
            datos[COL_FECHA].dt.date >= pd.to_datetime(fecha_inicio).date()
        ]
    if fecha_fin is not None:
        datos = datos[
            datos[COL_FECHA].dt.date <= pd.to_datetime(fecha_fin).date()
        ]

    filtros = {
        COL_CIUDAD: ciudades,
        COL_CATEGORIA: categorias,
        COL_TIPO_USUARIO: tipos_usuario,
        COL_CLIMA: climas,
        COL_METODO_PAGO: metodos_pago,
        COL_INCIDENTE: incidentes,
    }
    for columna, seleccion in filtros.items():
        if seleccion:
            datos = datos[datos[columna].isin(seleccion)]

    return datos.reset_index(drop=True)


__all__ = [
    "COL_ID",
    "COL_FECHA",
    "COL_CIUDAD",
    "COL_ORIGEN",
    "COL_DESTINO",
    "COL_USUARIO",
    "COL_GENERO",
    "COL_EDAD",
    "COL_TIPO_USUARIO",
    "COL_CATEGORIA",
    "COL_DURACION",
    "COL_DISTANCIA",
    "COL_TARIFA",
    "COL_DESCUENTO",
    "COL_VALOR",
    "COL_METODO_PAGO",
    "COL_CLIMA",
    "COL_TEMPERATURA",
    "COL_SATISFACCION",
    "COL_INCIDENTE",
    "MESES_ES",
    "DIAS_ES",
    "cargar_datos",
    "diagnosticar_datos",
    "limpiar_datos",
    "crear_variables_derivadas",
    "procesar_datos",
    "calcular_indicadores",
    "generar_resumen",
    "filtrar_datos",
]
