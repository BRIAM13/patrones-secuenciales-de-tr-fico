"""Pruebas de los guardrails de entrada y salida y de las excepciones del dominio."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ACCION_IMPUTACION_LINEAL,
    ACCION_SIN_CAMBIOS,
    COLUMNA_TIEMPO_RUTA_A,
    COLUMNA_TIEMPO_RUTA_B,
    PASOS_VENTANA_ENTRADA,
    PASOS_VENTANA_SALIDA,
)
from src.guardrails import (
    COLUMNAS_ESQUEMA_ENTRADA,
    ErrorBaseDatos,
    ErrorContratoSalida,
    ErrorExtrapolacionSalida,
    ErrorIntegridadSecuencial,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
    RecomendacionTrafico,
    ValidadorEntrada,
    ValidadorSalida,
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


# =============================================================================
# Guardrails de salida (ValidadorSalida)
# =============================================================================


@pytest.fixture
def validador_salida() -> ValidadorSalida:
    """Validador de salida con los límites por defecto de ``config``."""
    return ValidadorSalida()


def fechas_objetivo() -> pd.DatetimeIndex:
    """Las 168 horas de un horizonte de 7 días."""
    return pd.date_range("2025-04-01 00:00:00", periods=PASOS_VENTANA_SALIDA, freq="h")


def campos_recomendacion(**cambios: Any) -> dict[str, Any]:
    """Campos válidos de una recomendación ``RED_LSTM`` sin alertas."""
    campos: dict[str, Any] = {
        "ruta_recomendada": "Ruta_B",
        "tiempo_estimado_min": 41.5,
        "ahorro_estimado_min": 6.0,
        "hora_optima_salida": "07:15",
        "nivel_confianza_pct": 82.0,
        "modo_inferencia": "RED_LSTM",
        "mae_modelo": 8.2,
    }
    campos.update(cambios)
    return campos


# --- Regla 1: clipping -------------------------------------------------------


def test_prediccion_en_rango_sin_alerta(validador_salida: ValidadorSalida) -> None:
    prediccion = np.full((PASOS_VENTANA_SALIDA, 2), 45.0)
    acotada = validador_salida.acotar_prediccion(prediccion, fechas_objetivo())
    assert not acotada.alerta_extrapolacion
    assert acotada.total_recortes == 0
    assert list(acotada.datos.columns) == [
        "fecha_hora",
        COLUMNA_TIEMPO_RUTA_A,
        COLUMNA_TIEMPO_RUTA_B,
    ]


def test_clipping_activa_alerta_extrapolacion(validador_salida: ValidadorSalida) -> None:
    prediccion = np.full((1, PASOS_VENTANA_SALIDA, 2), 45.0)  # forma de salida de Keras
    prediccion[0, 0, 0] = 3.0
    prediccion[0, 1, 1] = 250.0
    acotada = validador_salida.acotar_prediccion(prediccion, fechas_objetivo())
    assert acotada.alerta_extrapolacion
    assert acotada.recortes_por_columna == {COLUMNA_TIEMPO_RUTA_A: 1, COLUMNA_TIEMPO_RUTA_B: 1}
    assert acotada.datos[COLUMNA_TIEMPO_RUTA_A].min() == 10.0
    assert acotada.datos[COLUMNA_TIEMPO_RUTA_B].max() == 180.0
    assert acotada.minimo_original[COLUMNA_TIEMPO_RUTA_A] == 3.0
    assert acotada.maximo_original[COLUMNA_TIEMPO_RUTA_B] == 250.0


def test_limites_de_salida_inclusivos(validador_salida: ValidadorSalida) -> None:
    prediccion = np.full((PASOS_VENTANA_SALIDA, 2), 45.0)
    prediccion[0] = [10.0, 180.0]
    assert not validador_salida.acotar_prediccion(
        prediccion, fechas_objetivo()
    ).alerta_extrapolacion


def test_prediccion_no_finita_rechazada(validador_salida: ValidadorSalida) -> None:
    prediccion = np.full((PASOS_VENTANA_SALIDA, 2), 45.0)
    prediccion[5, 1] = np.nan
    with pytest.raises(ErrorExtrapolacionSalida, match="no finitos") as error:
        validador_salida.acotar_prediccion(prediccion, fechas_objetivo())
    assert error.value.detalles["fechas"] == [str(fechas_objetivo()[5])]


def test_forma_incompatible_rechazada(validador_salida: ValidadorSalida) -> None:
    with pytest.raises(ErrorExtrapolacionSalida, match="Forma"):
        validador_salida.acotar_prediccion(np.full((24, 2), 45.0), fechas_objetivo())


# --- Regla 2: protocolo fallback ---------------------------------------------


@pytest.mark.parametrize(
    ("confianza", "mae", "modo", "motivos"),
    [
        (0.82, 8.0, "RED_LSTM", ()),
        (0.75, 12.0, "RED_LSTM", ()),
        (0.7499, 8.0, "FALLBACK_HEURISTICO", ("CONFIANZA_BAJA",)),
        (0.90, 12.01, "FALLBACK_HEURISTICO", ("MAE_ALTO",)),
        (0.50, 20.0, "FALLBACK_HEURISTICO", ("CONFIANZA_BAJA", "MAE_ALTO")),
        (0.90, None, "FALLBACK_HEURISTICO", ("MAE_DESCONOCIDO",)),
        (None, 5.0, "FALLBACK_HEURISTICO", ("CONFIANZA_BAJA",)),
    ],
)
def test_protocolo_fallback(
    validador_salida: ValidadorSalida,
    confianza: float | None,
    mae: float | None,
    modo: str,
    motivos: tuple[str, ...],
) -> None:
    evaluacion = validador_salida.evaluar_confianza(confianza, mae)
    assert evaluacion.modo_inferencia == modo
    assert evaluacion.motivos_fallback == motivos
    assert evaluacion.requiere_fallback == (modo == "FALLBACK_HEURISTICO")


@pytest.mark.parametrize(
    ("confianza", "mae"), [(1.2, 5.0), (-0.1, 5.0), (0.8, -1.0), (0.8, np.nan)]
)
def test_confianza_o_mae_fuera_de_dominio(
    validador_salida: ValidadorSalida, confianza: float, mae: float
) -> None:
    with pytest.raises(ErrorContratoSalida):
        validador_salida.evaluar_confianza(confianza, mae)


# --- Regla 3: contrato JSON estricto -----------------------------------------


def test_contrato_valido_y_campos_calculados(validador_salida: ValidadorSalida) -> None:
    recomendacion = validador_salida.construir_recomendacion(**campos_recomendacion())
    assert recomendacion.estado_auditoria == "APROBADO"
    assert recomendacion.banderas == ()
    assert recomendacion.timestamp_consulta.tzinfo is not None


def test_alertas_cambian_estado_y_banderas(validador_salida: ValidadorSalida) -> None:
    recomendacion = validador_salida.construir_recomendacion(
        **campos_recomendacion(
            modo_inferencia="FALLBACK_HEURISTICO",
            nivel_confianza_pct=40.0,
            mae_modelo=None,
            motivos_fallback=("CONFIANZA_BAJA", "MAE_DESCONOCIDO"),
            alerta_extrapolacion=True,
            alerta_drift=True,
        )
    )
    assert recomendacion.estado_auditoria == "APROBADO_CON_ALERTAS"
    assert recomendacion.banderas == ("ALERTA_EXTRAPOLACION", "ALERTA_DRIFT", "FALLBACK_HEURISTICO")


@pytest.mark.parametrize(
    "cambios",
    [
        {"nivel_confianza_pct": 70.0},  # RED_LSTM con confianza baja → debía ser fallback
        {"mae_modelo": 15.0},  # RED_LSTM con MAE alto
        {"mae_modelo": None},  # RED_LSTM sin MAE
        {"motivos_fallback": ("MAE_ALTO",)},  # RED_LSTM con motivos
        {"modo_inferencia": "FALLBACK_HEURISTICO"},  # fallback sin motivos
        {"modo_inferencia": "FALLBACK_HEURISTICO", "motivos_fallback": ("POR_QUE_SI",)},
        {"ruta_recomendada": "Ruta_C"},
        {"tiempo_estimado_min": 200.0},  # fuera del clipping [10, 180]
        {"ahorro_estimado_min": -1.0},
        {"hora_optima_salida": "7:15"},
        {"hora_optima_salida": "24:00"},
        {"tiempo_estimado_min": "41.5"},  # modo estricto: sin coerción de tipos
        {"nivel_confianza_pct": float("nan")},
        {"timestamp_consulta": datetime(2025, 1, 1, 8, 0)},  # sin zona horaria
        {"estado_auditoria": "APROBADO"},  # campo calculado: no se acepta como entrada
    ],
)
def test_contrato_rechaza_violaciones(
    validador_salida: ValidadorSalida, cambios: dict[str, Any]
) -> None:
    with pytest.raises(ErrorContratoSalida):
        validador_salida.construir_recomendacion(**campos_recomendacion(**cambios))


def test_contrato_inmutable(validador_salida: ValidadorSalida) -> None:
    recomendacion = validador_salida.construir_recomendacion(**campos_recomendacion())
    with pytest.raises(ValueError):
        recomendacion.modo_inferencia = "FALLBACK_HEURISTICO"  # type: ignore[misc]


def test_json_ida_y_vuelta(validador_salida: ValidadorSalida) -> None:
    original = validador_salida.construir_recomendacion(**campos_recomendacion(alerta_drift=True))
    texto = validador_salida.serializar(original)
    contenido = json.loads(texto)
    assert contenido["estado_auditoria"] == "APROBADO_CON_ALERTAS"
    assert contenido["banderas"] == ["ALERTA_DRIFT"]
    assert validador_salida.validar_json(texto) == original


@pytest.mark.parametrize(
    "manipulacion",
    [
        {"estado_auditoria": "APROBADO"},
        {"banderas": []},
        {"tiempo_estimado_min": "41.5"},
        {"campo_extra": 1},
    ],
)
def test_json_manipulado_rechazado(
    validador_salida: ValidadorSalida, manipulacion: dict[str, Any]
) -> None:
    original = validador_salida.construir_recomendacion(**campos_recomendacion(alerta_drift=True))
    contenido = json.loads(validador_salida.serializar(original))
    contenido.update(manipulacion)
    with pytest.raises(ErrorContratoSalida):
        validador_salida.validar_json(json.dumps(contenido))


def test_json_mal_formado(validador_salida: ValidadorSalida) -> None:
    with pytest.raises(ErrorContratoSalida, match="mal formado"):
        validador_salida.validar_json("{no es json")
    with pytest.raises(ErrorContratoSalida, match="objeto"):
        validador_salida.validar_json("[1, 2]")


def test_proyeccion_a_registro_de_auditoria() -> None:
    recomendacion = RecomendacionTrafico(**campos_recomendacion())
    registro = recomendacion.a_registro_auditoria()
    assert registro["estado_auditoria"] == "APROBADO"
    assert registro["id_transaccion"] == str(recomendacion.id_transaccion)
    assert "banderas" not in registro and "alerta_extrapolacion" not in registro
