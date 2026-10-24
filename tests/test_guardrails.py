"""Pruebas de los guardrails de entrada (``ValidadorEntrada``) y de las excepciones."""

from __future__ import annotations

import json
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ACCION_IMPUTACION_LINEAL,
    ACCION_SIN_CAMBIOS,
    COLUMNA_TIEMPO_RUTA_A,
    COLUMNA_TIEMPO_RUTA_B,
    PASOS_VENTANA_ENTRADA,
)
from src.guardrails import (
    COLUMNAS_ESQUEMA_ENTRADA,
    ErrorBaseDatos,
    ErrorIntegridadSecuencial,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
    ValidadorEntrada,
)


def construir_serie(horas: int, inicio: str = "2025-03-03 00:00:00") -> pd.DataFrame:
    """Construye una serie horaria válida con tiempos constantes dentro de rango."""
    fechas = pd.date_range(inicio, periods=horas, freq="h")
    ultimo_dia = fechas.is_month_end
    return pd.DataFrame(
        {
            "fecha_hora": fechas,
            "dia_semana": fechas.weekday.astype(int),
            "hora_decimal": fechas.hour.astype(float),
            "es_fin_de_semana": fechas.weekday >= 5,
            "es_quincena_o_fin_mes": (fechas.day == 15) | ultimo_dia,
            "clima_adverso": False,
            COLUMNA_TIEMPO_RUTA_A: 40.0,
            COLUMNA_TIEMPO_RUTA_B: 35.0,
            "origen_dato": "SINTETICO",
        }
    )


@pytest.fixture
def validador() -> ValidadorEntrada:
    """Validador con los límites por defecto de ``config``."""
    return ValidadorEntrada()


# -----------------------------------------------------------------------------
# Excepciones
# -----------------------------------------------------------------------------


def test_jerarquia_de_excepciones() -> None:
    assert issubclass(ErrorIntegridadSecuencial, ErrorValidacionEntrada)
    assert issubclass(ErrorValidacionEntrada, ErrorSistemaTrafico)
    assert issubclass(ErrorBaseDatos, ErrorSistemaTrafico)


def test_excepcion_serializable_para_auditoria() -> None:
    error = ErrorValidacionEntrada("fallo", detalles={"columna": "x", "conteo": 3})
    assert error.a_diccionario() == {
        "codigo": "ErrorValidacionEntrada",
        "mensaje": "fallo",
        "detalles": {"columna": "x", "conteo": 3},
    }
    json.dumps(error.a_diccionario())


# -----------------------------------------------------------------------------
# Parámetros del validador
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"tiempo_min": 50.0, "tiempo_max": 10.0},
        {"umbral_corrupcion": 1.0},
        {"pasos_ventana": 0},
    ],
)
def test_parametros_incoherentes_rechazados(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ValidadorEntrada(**kwargs)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Capa 1: esquema
# -----------------------------------------------------------------------------


def test_serie_limpia_sin_cambios(validador: ValidadorEntrada) -> None:
    resultado = validador.validar_serie(construir_serie(PASOS_VENTANA_ENTRADA * 2))
    assert resultado.reporte.accion == ACCION_SIN_CAMBIOS
    assert resultado.reporte.total_imputados == 0
    assert list(resultado.datos.columns) == list(COLUMNAS_ESQUEMA_ENTRADA)


def test_columnas_adicionales_se_descartan(validador: ValidadorEntrada) -> None:
    datos = construir_serie(48).assign(id=range(48))
    assert "id" not in validador.validar_serie(datos).datos.columns


def test_columna_faltante(validador: ValidadorEntrada) -> None:
    with pytest.raises(ErrorValidacionEntrada) as error:
        validador.validar_serie(construir_serie(48).drop(columns=["clima_adverso"]))
    assert error.value.detalles["columnas_faltantes"] == ["clima_adverso"]


def test_dataframe_vacio_o_tipo_incorrecto(validador: ValidadorEntrada) -> None:
    with pytest.raises(ErrorValidacionEntrada):
        validador.validar_serie(construir_serie(1).iloc[0:0])
    with pytest.raises(ErrorValidacionEntrada):
        validador.validar_serie([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("columna", "valor"),
    [
        (COLUMNA_TIEMPO_RUTA_A, "40"),
        ("clima_adverso", 1),
        ("origen_dato", "OTRO"),
        ("dia_semana", 6),
        ("es_fin_de_semana", True),
        ("hora_decimal", 23.5),
    ],
)
def test_violaciones_de_esquema(validador: ValidadorEntrada, columna: str, valor: object) -> None:
    datos = construir_serie(48).astype({columna: object})
    datos.at[3, columna] = valor
    with pytest.raises(ErrorValidacionEntrada) as error:
        validador.validar_serie(datos)
    assert error.value.detalles["errores"][0]["fila"] == 3


def test_fecha_con_zona_horaria_rechazada(validador: ValidadorEntrada) -> None:
    datos = construir_serie(24)
    datos["fecha_hora"] = datos["fecha_hora"].dt.tz_localize(timezone.utc)
    with pytest.raises(ErrorValidacionEntrada):
        validador.validar_esquema(datos)


# -----------------------------------------------------------------------------
# Capa 2: integridad secuencial
# -----------------------------------------------------------------------------


def test_marca_duplicada(validador: ValidadorEntrada) -> None:
    datos = construir_serie(48)
    datos = pd.concat([datos, datos.iloc[[10]]], ignore_index=True)
    with pytest.raises(ErrorIntegridadSecuencial, match="duplicadas"):
        validador.validar_serie(datos)


def test_orden_cronologico(validador: ValidadorEntrada) -> None:
    datos = construir_serie(48).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ErrorIntegridadSecuencial, match="orden"):
        validador.validar_serie(datos)


def test_hueco_en_rejilla(validador: ValidadorEntrada) -> None:
    datos = construir_serie(48).drop(index=[20, 21, 22]).reset_index(drop=True)
    with pytest.raises(ErrorIntegridadSecuencial) as error:
        validador.validar_serie(datos)
    assert error.value.detalles["huecos"][0]["horas_faltantes"] == 3


def test_registro_no_alineado(validador: ValidadorEntrada) -> None:
    datos = construir_serie(48)
    datos.at[5, "fecha_hora"] = datos.at[5, "fecha_hora"] + pd.Timedelta(minutes=30)
    datos.at[5, "hora_decimal"] = datos.at[5, "hora_decimal"] + 0.5
    with pytest.raises(ErrorIntegridadSecuencial, match="alineados"):
        validador.validar_serie(datos)


def test_ventana_debe_medir_14_dias(validador: ValidadorEntrada) -> None:
    validador.validar_ventana(construir_serie(PASOS_VENTANA_ENTRADA))
    with pytest.raises(ErrorIntegridadSecuencial, match="336"):
        validador.validar_ventana(construir_serie(PASOS_VENTANA_ENTRADA - 1))


# -----------------------------------------------------------------------------
# Capa 3: calidad física e imputación
# -----------------------------------------------------------------------------


def test_imputacion_lineal_de_nan(validador: ValidadorEntrada) -> None:
    datos = construir_serie(PASOS_VENTANA_ENTRADA)
    datos.loc[9, COLUMNA_TIEMPO_RUTA_A] = 30.0
    datos.loc[10, COLUMNA_TIEMPO_RUTA_A] = np.nan
    datos.loc[11, COLUMNA_TIEMPO_RUTA_A] = 50.0
    resultado = validador.validar_ventana(datos)
    assert resultado.datos.loc[10, COLUMNA_TIEMPO_RUTA_A] == pytest.approx(40.0)
    assert resultado.reporte.accion == ACCION_IMPUTACION_LINEAL
    assert resultado.reporte.no_finitos_por_columna[COLUMNA_TIEMPO_RUTA_A] == 1
    assert not resultado.datos[COLUMNA_TIEMPO_RUTA_A].isna().any()


def test_fuera_de_rango_e_infinitos_se_imputan(validador: ValidadorEntrada) -> None:
    datos = construir_serie(PASOS_VENTANA_ENTRADA)
    datos.loc[[5, 6], COLUMNA_TIEMPO_RUTA_A] = [5.0, 500.0]
    datos.loc[7, COLUMNA_TIEMPO_RUTA_B] = np.inf
    resultado = validador.validar_ventana(datos)
    reporte = resultado.reporte
    assert reporte.fuera_de_rango_por_columna[COLUMNA_TIEMPO_RUTA_A] == 2
    assert reporte.no_finitos_por_columna[COLUMNA_TIEMPO_RUTA_B] == 1
    assert reporte.imputados_por_columna == {COLUMNA_TIEMPO_RUTA_A: 2, COLUMNA_TIEMPO_RUTA_B: 1}
    assert resultado.datos[COLUMNA_TIEMPO_RUTA_A].between(10.0, 240.0).all()
    assert np.isfinite(resultado.datos[COLUMNA_TIEMPO_RUTA_B]).all()


def test_limites_fisicos_inclusivos(validador: ValidadorEntrada) -> None:
    datos = construir_serie(48)
    datos.loc[[0, 1], COLUMNA_TIEMPO_RUTA_A] = [10.0, 240.0]
    assert validador.validar_serie(datos).reporte.accion == ACCION_SIN_CAMBIOS


def test_umbral_exacto_15_por_ciento_se_imputa(validador: ValidadorEntrada) -> None:
    datos = construir_serie(PASOS_VENTANA_ENTRADA)
    n_corruptos = int(PASOS_VENTANA_ENTRADA * 0.15)
    datos.loc[: n_corruptos - 1, COLUMNA_TIEMPO_RUTA_A] = np.nan
    resultado = validador.validar_ventana(datos)
    assert resultado.reporte.imputados_por_columna[COLUMNA_TIEMPO_RUTA_A] == n_corruptos
    assert resultado.reporte.fraccion_corrupta_max <= 0.15


def test_corrupcion_superior_al_umbral_aborta(validador: ValidadorEntrada) -> None:
    datos = construir_serie(PASOS_VENTANA_ENTRADA)
    n_corruptos = int(PASOS_VENTANA_ENTRADA * 0.15) + 1
    datos.loc[: n_corruptos - 1, COLUMNA_TIEMPO_RUTA_B] = np.nan
    with pytest.raises(ErrorValidacionEntrada, match="abortada") as error:
        validador.validar_ventana(datos)
    assert error.value.detalles["fracciones_corruptas_max"][COLUMNA_TIEMPO_RUTA_B] > 0.15


def test_regla_15_se_evalua_por_ventana_no_por_serie(validador: ValidadorEntrada) -> None:
    """Corrupción concentrada: 3 % en la serie, pero > 15 % en una ventana de 14 días."""
    datos = construir_serie(PASOS_VENTANA_ENTRADA * 10)
    datos.loc[1000 : 1000 + 60, COLUMNA_TIEMPO_RUTA_A] = np.nan
    assert datos[COLUMNA_TIEMPO_RUTA_A].isna().mean() < 0.15
    with pytest.raises(ErrorValidacionEntrada):
        validador.validar_serie(datos)


def test_nan_en_bordes_se_imputa(validador: ValidadorEntrada) -> None:
    datos = construir_serie(PASOS_VENTANA_ENTRADA)
    datos.loc[0, COLUMNA_TIEMPO_RUTA_A] = np.nan
    datos.loc[PASOS_VENTANA_ENTRADA - 1, COLUMNA_TIEMPO_RUTA_A] = np.nan
    resultado = validador.validar_ventana(datos)
    assert not resultado.datos[COLUMNA_TIEMPO_RUTA_A].isna().any()


def test_reporte_inmutable_y_serializable(validador: ValidadorEntrada) -> None:
    reporte = validador.validar_serie(construir_serie(48)).reporte
    json.dumps(reporte.model_dump(mode="json"))
    with pytest.raises(ValueError):
        reporte.accion = "OTRA"  # type: ignore[misc]


def test_entrada_original_no_se_modifica(validador: ValidadorEntrada) -> None:
    datos = construir_serie(PASOS_VENTANA_ENTRADA)
    datos.loc[10, COLUMNA_TIEMPO_RUTA_A] = np.nan
    validador.validar_ventana(datos)
    assert np.isnan(datos.loc[10, COLUMNA_TIEMPO_RUTA_A])
