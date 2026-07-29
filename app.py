from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from procesamiento import (
    COL_CATEGORIA,
    COL_CIUDAD,
    COL_CLIMA,
    COL_DESTINO,
    COL_DISTANCIA,
    COL_DURACION,
    COL_EDAD,
    COL_FECHA,
    COL_INCIDENTE,
    COL_METODO_PAGO,
    COL_ORIGEN,
    COL_SATISFACCION,
    COL_TEMPERATURA,
    COL_TIPO_USUARIO,
    COL_USUARIO,
    COL_VALOR,
    calcular_indicadores,
    diagnosticar_datos,
    filtrar_datos,
    generar_resumen,
    procesar_datos,
)


# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de movilidad",
    page_icon="🚲",
    layout="wide",
)

st.title("Dashboard de movilidad urbana")
st.caption(
    "Análisis de viajes, ingresos, operación, satisfacción e incidentes."
)


# ---------------------------------------------------------------------
# Funciones auxiliares de presentación
# ---------------------------------------------------------------------
def formatear_moneda(valor: float) -> str:
    return f"${valor:,.0f}".replace(",", ".")


def formatear_numero(valor: float, decimales: int = 1) -> str:
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner=False)
def cargar_y_procesar_archivo(archivo) -> pd.DataFrame:
    """Procesa el archivo subido y conserva el resultado en caché."""
    return procesar_datos(archivo)


@st.cache_data(show_spinner=False)
def cargar_y_procesar_ruta(ruta: str) -> pd.DataFrame:
    """Procesa un CSV almacenado junto a la aplicación."""
    return procesar_datos(ruta)


def obtener_archivo_local() -> Path | None:
    candidatos = [
        Path("movilidad.csv"),
        Path("movilidad(1).csv"),
        Path("data/movilidad.csv"),
    ]
    return next((ruta for ruta in candidatos if ruta.exists()), None)


def grafico_sin_datos(mensaje: str = "No hay datos para este gráfico.") -> None:
    st.info(mensaje)


# ---------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Datos")
    archivo_subido = st.file_uploader(
        "Carga el archivo CSV",
        type=["csv"],
        help=(
            "Debe contener las columnas originales del dataset de movilidad. "
            "La limpieza se ejecuta automáticamente."
        ),
    )

try:
    if archivo_subido is not None:
        df_original = pd.read_csv(archivo_subido)
        archivo_subido.seek(0)
        reporte_calidad = diagnosticar_datos(df_original)
        df_trabajo = cargar_y_procesar_archivo(archivo_subido)
        origen_datos = archivo_subido.name
    else:
        archivo_local = obtener_archivo_local()
        if archivo_local is None:
            st.warning(
                "Carga el CSV desde la barra lateral o agrega "
                "`movilidad.csv` junto a `app.py`."
            )
            st.stop()

        df_original = pd.read_csv(archivo_local)
        reporte_calidad = diagnosticar_datos(df_original)
        df_trabajo = cargar_y_procesar_ruta(str(archivo_local))
        origen_datos = archivo_local.name
except (ValueError, KeyError, pd.errors.ParserError) as error:
    st.error(f"No fue posible procesar el archivo: {error}")
    st.stop()


# ---------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("Filtros")

    fecha_minima = df_trabajo[COL_FECHA].min().date()
    fecha_maxima = df_trabajo[COL_FECHA].max().date()

    fecha_inicio = st.date_input(
        "Fecha inicial",
        value=fecha_minima,
        min_value=fecha_minima,
        max_value=fecha_maxima,
    )
    fecha_fin = st.date_input(
        "Fecha final",
        value=fecha_maxima,
        min_value=fecha_minima,
        max_value=fecha_maxima,
    )

    if fecha_inicio > fecha_fin:
        st.error("La fecha inicial no puede ser posterior a la fecha final.")
        st.stop()

    opciones_ciudad = sorted(df_trabajo[COL_CIUDAD].dropna().unique())
    opciones_categoria = sorted(df_trabajo[COL_CATEGORIA].dropna().unique())
    opciones_tipo_usuario = sorted(
        df_trabajo[COL_TIPO_USUARIO].dropna().unique()
    )
    opciones_clima = sorted(df_trabajo[COL_CLIMA].dropna().unique())
    opciones_pago = sorted(df_trabajo[COL_METODO_PAGO].dropna().unique())
    opciones_incidente = sorted(df_trabajo[COL_INCIDENTE].dropna().unique())

    ciudades = st.multiselect("Ciudad", opciones_ciudad)
    categorias = st.multiselect("Tipo de vehículo", opciones_categoria)
    tipos_usuario = st.multiselect(
        "Tipo de usuario",
        opciones_tipo_usuario,
    )
    climas = st.multiselect("Clima", opciones_clima)
    metodos_pago = st.multiselect("Método de pago", opciones_pago)
    incidentes = st.multiselect("Incidente", opciones_incidente)

    st.divider()
    st.caption(f"Fuente: {origen_datos}")

df_filtrado = filtrar_datos(
    df_trabajo,
    fecha_inicio=fecha_inicio,
    fecha_fin=fecha_fin,
    ciudades=ciudades,
    categorias=categorias,
    tipos_usuario=tipos_usuario,
    climas=climas,
    metodos_pago=metodos_pago,
    incidentes=incidentes,
)

if df_filtrado.empty:
    st.warning("No hay registros que cumplan los filtros seleccionados.")
    st.stop()


# ---------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------
indicadores = calcular_indicadores(df_filtrado)

fila_kpi_1 = st.columns(4)
fila_kpi_1[0].metric(
    "Total de viajes",
    f"{indicadores['total_viajes']:,}".replace(",", "."),
)
fila_kpi_1[1].metric(
    "Ingresos totales",
    formatear_moneda(indicadores["ingresos_totales"]),
)
fila_kpi_1[2].metric(
    "Duración promedio",
    f"{formatear_numero(indicadores['duracion_promedio'])} min",
)
fila_kpi_1[3].metric(
    "Distancia promedio",
    f"{formatear_numero(indicadores['distancia_promedio'], 2)} km",
)

fila_kpi_2 = st.columns(4)
fila_kpi_2[0].metric(
    "Satisfacción promedio",
    f"{formatear_numero(indicadores['satisfaccion_promedio'], 2)} / 5",
)
fila_kpi_2[1].metric(
    "Incidentes",
    f"{indicadores['incidentes']:,}".replace(",", "."),
)
fila_kpi_2[2].metric(
    "Tasa de incidentes",
    f"{formatear_numero(indicadores['tasa_incidentes'], 2)} %",
)
fila_kpi_2[3].metric(
    "Usuarios únicos",
    f"{indicadores['usuarios_unicos']:,}".replace(",", "."),
)


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
tab_general, tab_tiempo, tab_operacion, tab_calidad = st.tabs(
    [
        "Resumen general",
        "Comportamiento temporal",
        "Operación e incidentes",
        "Calidad y datos",
    ]
)


with tab_general:
    columna_1, columna_2 = st.columns(2)

    with columna_1:
        resumen_ciudad = generar_resumen(
            df_filtrado,
            COL_CIUDAD,
            nombre_resultado="viajes",
        )
        figura_ciudad = px.bar(
            resumen_ciudad,
            x=COL_CIUDAD,
            y="viajes",
            title="Cantidad de viajes por ciudad",
            text_auto=True,
        )
        figura_ciudad.update_layout(xaxis_title="", yaxis_title="Viajes")
        st.plotly_chart(figura_ciudad, use_container_width=True)

    with columna_2:
        resumen_ingresos_ciudad = generar_resumen(
            df_filtrado,
            COL_CIUDAD,
            metrica=COL_VALOR,
            operacion="sum",
            nombre_resultado="ingresos",
        )
        figura_ingresos = px.bar(
            resumen_ingresos_ciudad,
            x=COL_CIUDAD,
            y="ingresos",
            title="Ingresos por ciudad",
            text_auto=".2s",
        )
        figura_ingresos.update_layout(
            xaxis_title="",
            yaxis_title="Ingresos",
        )
        st.plotly_chart(figura_ingresos, use_container_width=True)

    columna_3, columna_4 = st.columns(2)

    with columna_3:
        resumen_categoria = generar_resumen(
            df_filtrado,
            COL_CATEGORIA,
            nombre_resultado="viajes",
        )
        figura_categoria = px.pie(
            resumen_categoria,
            names=COL_CATEGORIA,
            values="viajes",
            hole=0.45,
            title="Participación por tipo de vehículo",
        )
        st.plotly_chart(figura_categoria, use_container_width=True)

    with columna_4:
        resumen_pago = generar_resumen(
            df_filtrado,
            COL_METODO_PAGO,
            nombre_resultado="viajes",
        )
        figura_pago = px.bar(
            resumen_pago,
            x="viajes",
            y=COL_METODO_PAGO,
            orientation="h",
            title="Métodos de pago",
            text_auto=True,
        )
        figura_pago.update_layout(
            xaxis_title="Viajes",
            yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(figura_pago, use_container_width=True)

    columna_5, columna_6 = st.columns(2)

    with columna_5:
        figura_edad = px.histogram(
            df_filtrado,
            x=COL_EDAD,
            nbins=15,
            title="Distribución de edades",
        )
        figura_edad.update_layout(
            xaxis_title="Edad",
            yaxis_title="Frecuencia",
        )
        st.plotly_chart(figura_edad, use_container_width=True)

    with columna_6:
        resumen_satisfaccion = (
            df_filtrado.groupby(COL_SATISFACCION, observed=False)
            .size()
            .rename("viajes")
            .reset_index()
            .sort_values(COL_SATISFACCION)
        )
        figura_satisfaccion = px.bar(
            resumen_satisfaccion,
            x=COL_SATISFACCION,
            y="viajes",
            title="Distribución de la satisfacción",
            text_auto=True,
        )
        figura_satisfaccion.update_layout(
            xaxis_title="Satisfacción",
            yaxis_title="Viajes",
        )
        st.plotly_chart(figura_satisfaccion, use_container_width=True)


with tab_tiempo:
    resumen_diario = (
        df_filtrado.assign(
            fecha_grafico=df_filtrado[COL_FECHA].dt.floor("D")
        )
        .groupby("fecha_grafico", observed=False)
        .agg(
            viajes=(COL_FECHA, "size"),
            ingresos=(COL_VALOR, "sum"),
        )
        .reset_index()
        .sort_values("fecha_grafico")
    )

    figura_tendencia = px.line(
        resumen_diario,
        x="fecha_grafico",
        y="viajes",
        title="Evolución diaria de los viajes",
        markers=True,
    )
    figura_tendencia.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Viajes",
    )
    st.plotly_chart(figura_tendencia, use_container_width=True)

    columna_1, columna_2 = st.columns(2)

    with columna_1:
        resumen_mes = (
            df_filtrado.groupby(
                ["mes_numero", "mes"],
                observed=False,
            )
            .agg(
                viajes=(COL_FECHA, "size"),
                ingresos=(COL_VALOR, "sum"),
            )
            .reset_index()
            .sort_values("mes_numero")
        )
        figura_mes = px.line(
            resumen_mes,
            x="mes",
            y="ingresos",
            markers=True,
            title="Ingresos mensuales",
        )
        figura_mes.update_layout(
            xaxis_title="",
            yaxis_title="Ingresos",
        )
        st.plotly_chart(figura_mes, use_container_width=True)

    with columna_2:
        resumen_hora = (
            df_filtrado.groupby("hora", observed=False)
            .size()
            .rename("viajes")
            .reset_index()
            .sort_values("hora")
        )
        figura_hora = px.area(
            resumen_hora,
            x="hora",
            y="viajes",
            title="Demanda por hora del día",
        )
        figura_hora.update_layout(
            xaxis_title="Hora",
            yaxis_title="Viajes",
        )
        st.plotly_chart(figura_hora, use_container_width=True)

    tabla_calor = (
        df_filtrado.groupby(
            ["dia_semana", "hora"],
            observed=False,
        )
        .size()
        .rename("viajes")
        .reset_index()
        .pivot(index="dia_semana", columns="hora", values="viajes")
        .fillna(0)
    )

    figura_calor = px.imshow(
        tabla_calor,
        aspect="auto",
        labels={
            "x": "Hora",
            "y": "Día de la semana",
            "color": "Viajes",
        },
        title="Demanda por día y hora",
    )
    st.plotly_chart(figura_calor, use_container_width=True)


with tab_operacion:
    columna_1, columna_2 = st.columns(2)

    with columna_1:
        resumen_origen = generar_resumen(
            df_filtrado,
            COL_ORIGEN,
            nombre_resultado="viajes",
        ).head(10)
        figura_origen = px.bar(
            resumen_origen.sort_values("viajes"),
            x="viajes",
            y=COL_ORIGEN,
            orientation="h",
            title="Top 10 estaciones de origen",
            text_auto=True,
        )
        figura_origen.update_layout(
            xaxis_title="Viajes",
            yaxis_title="",
        )
        st.plotly_chart(figura_origen, use_container_width=True)

    with columna_2:
        resumen_rutas = generar_resumen(
            df_filtrado,
            "ruta",
            nombre_resultado="viajes",
        ).head(10)
        figura_rutas = px.bar(
            resumen_rutas.sort_values("viajes"),
            x="viajes",
            y="ruta",
            orientation="h",
            title="Top 10 rutas",
            text_auto=True,
        )
        figura_rutas.update_layout(
            xaxis_title="Viajes",
            yaxis_title="",
        )
        st.plotly_chart(figura_rutas, use_container_width=True)

    figura_relacion = px.scatter(
        df_filtrado,
        x=COL_DISTANCIA,
        y=COL_DURACION,
        color=COL_CATEGORIA,
        size=COL_VALOR,
        hover_data=[
            COL_CIUDAD,
            COL_ORIGEN,
            COL_DESTINO,
            "velocidad_promedio_kmh",
        ],
        title="Relación entre distancia y duración",
        opacity=0.65,
    )
    figura_relacion.update_layout(
        xaxis_title="Distancia (km)",
        yaxis_title="Duración (min)",
    )
    st.plotly_chart(figura_relacion, use_container_width=True)

    columna_3, columna_4 = st.columns(2)

    with columna_3:
        resumen_incidentes_clima = (
            df_filtrado.groupby(COL_CLIMA, observed=False)
            .agg(
                viajes=(COL_INCIDENTE, "size"),
                incidentes=("tiene_incidente", "sum"),
            )
            .reset_index()
        )
        resumen_incidentes_clima["tasa_incidentes"] = (
            resumen_incidentes_clima["incidentes"]
            / resumen_incidentes_clima["viajes"]
            * 100
        )
        figura_incidentes_clima = px.bar(
            resumen_incidentes_clima.sort_values(
                "tasa_incidentes",
                ascending=False,
            ),
            x=COL_CLIMA,
            y="tasa_incidentes",
            title="Tasa de incidentes por clima",
            text_auto=".2f",
        )
        figura_incidentes_clima.update_layout(
            xaxis_title="",
            yaxis_title="Tasa de incidentes (%)",
        )
        st.plotly_chart(figura_incidentes_clima, use_container_width=True)

    with columna_4:
        resumen_incidentes_categoria = (
            df_filtrado.groupby(COL_CATEGORIA, observed=False)
            .agg(
                viajes=(COL_INCIDENTE, "size"),
                incidentes=("tiene_incidente", "sum"),
            )
            .reset_index()
        )
        resumen_incidentes_categoria["tasa_incidentes"] = (
            resumen_incidentes_categoria["incidentes"]
            / resumen_incidentes_categoria["viajes"]
            * 100
        )
        figura_incidentes_categoria = px.bar(
            resumen_incidentes_categoria.sort_values(
                "tasa_incidentes",
                ascending=False,
            ),
            x=COL_CATEGORIA,
            y="tasa_incidentes",
            title="Tasa de incidentes por vehículo",
            text_auto=".2f",
        )
        figura_incidentes_categoria.update_layout(
            xaxis_title="",
            yaxis_title="Tasa de incidentes (%)",
        )
        st.plotly_chart(figura_incidentes_categoria, use_container_width=True)

    columna_5, columna_6 = st.columns(2)

    with columna_5:
        figura_temperatura = px.box(
            df_filtrado,
            x=COL_CIUDAD,
            y=COL_TEMPERATURA,
            title="Temperatura por ciudad",
        )
        figura_temperatura.update_layout(
            xaxis_title="",
            yaxis_title="Temperatura (°C)",
        )
        st.plotly_chart(figura_temperatura, use_container_width=True)

    with columna_6:
        resumen_satisfaccion_ciudad = generar_resumen(
            df_filtrado,
            COL_CIUDAD,
            metrica=COL_SATISFACCION,
            operacion="mean",
            nombre_resultado="satisfaccion_promedio",
        )
        figura_satisfaccion_ciudad = px.bar(
            resumen_satisfaccion_ciudad,
            x=COL_CIUDAD,
            y="satisfaccion_promedio",
            title="Satisfacción promedio por ciudad",
            text_auto=".2f",
        )
        figura_satisfaccion_ciudad.update_layout(
            xaxis_title="",
            yaxis_title="Satisfacción promedio",
            yaxis_range=[0, 5],
        )
        st.plotly_chart(
            figura_satisfaccion_ciudad,
            use_container_width=True,
        )


with tab_calidad:
    st.subheader("Diagnóstico del archivo original")

    columna_1, columna_2, columna_3, columna_4 = st.columns(4)
    columna_1.metric("Filas originales", reporte_calidad["filas"])
    columna_2.metric("Columnas", reporte_calidad["columnas"])
    columna_3.metric(
        "Duplicados completos",
        reporte_calidad["duplicados_completos"],
    )
    columna_4.metric(
        "ID duplicados",
        reporte_calidad["ids_duplicados"],
    )

    st.dataframe(
        reporte_calidad["resumen_columnas"],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Datos procesados y filtrados")
    columnas_visibles = [
        COL_FECHA,
        COL_CIUDAD,
        COL_ORIGEN,
        COL_DESTINO,
        COL_USUARIO,
        COL_CATEGORIA,
        COL_DURACION,
        COL_DISTANCIA,
        COL_VALOR,
        COL_CLIMA,
        COL_SATISFACCION,
        COL_INCIDENTE,
    ]
    st.dataframe(
        df_filtrado[columnas_visibles],
        use_container_width=True,
        hide_index=True,
    )

    csv_filtrado = df_filtrado.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar datos procesados",
        data=csv_filtrado,
        file_name="movilidad_procesada.csv",
        mime="text/csv",
    )

    with st.expander("Variables creadas durante el procesamiento"):
        st.write(
            [
                "fecha",
                "anio",
                "mes_numero",
                "mes",
                "dia_mes",
                "dia_semana",
                "hora",
                "franja_horaria",
                "es_fin_semana",
                "es_hora_pico",
                "velocidad_promedio_kmh",
                "descuento_valor",
                "ruta",
                "tiene_incidente",
                "segmento_distancia",
                "segmento_duracion",
                "es_atipico_duracion",
                "es_atipico_distancia",
                "valor_estimado",
                "diferencia_valor",
                "valor_consistente",
                "indicadores de imputación",
            ]
        )
