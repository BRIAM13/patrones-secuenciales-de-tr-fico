"""Pruebas de los agentes: datos, auditoría, estrategia de ruta y predictor LSTM."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from src.agents import (
    AgenteAuditor,
    AgenteDatos,
    AgenteEstrategaRuta,
    codificar_variables_ciclicas,
)
from src.config import (
    COLUMNA_TIEMPO_RUTA_A,
    COLUMNA_TIEMPO_RUTA_B,
    COLUMNAS_CARACTERISTICAS,
    ORIGEN_REAL,
    ORIGEN_SINTETICO,
    PASOS_VENTANA_ENTRADA,
    PASOS_VENTANA_SALIDA,
)
from src.database import GestorBaseDatos
from src.guardrails import (
    COLUMNAS_ESQUEMA_ENTRADA,
    AlertaDataDrift,
    ErrorBaseDatos,
    ErrorIntegridadSecuencial,
    ErrorModeloNoDisponible,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
    ValidadorEntrada,
    ValidadorSalida,
)

DIAS_PRUEBA = 60


@pytest.fixture
def gestor(tmp_path: Path) -> GestorBaseDatos:
    """Base de datos y log JSONL aislados en un directorio temporal."""
    return GestorBaseDatos(tmp_path / "data" / "prueba.db", tmp_path / "logs" / "auditoria.jsonl")


@pytest.fixture
def agente(gestor: GestorBaseDatos) -> AgenteDatos:
    """Agente de datos con semilla fija."""
    return AgenteDatos(gestor, semilla=7)


def fuente_real(horas: int = 48) -> pd.DataFrame:
    """Fuente externa mínima, como la entregaría un CSV de sensores."""
    fechas = pd.date_range("2025-06-02 00:00:00", periods=horas, freq="h")
    return pd.DataFrame(
        {
            "fecha_hora": fechas.astype(str),
            COLUMNA_TIEMPO_RUTA_A: np.linspace(30.0, 60.0, horas),
            COLUMNA_TIEMPO_RUTA_B: 40.0,
        }
    )


# -----------------------------------------------------------------------------
# Generación sintética
# -----------------------------------------------------------------------------


def test_generacion_sintetica_cumple_esquema_y_guardrails(agente: AgenteDatos) -> None:
    datos = agente.generar_datos_sinteticos(dias=DIAS_PRUEBA)
    assert len(datos) == DIAS_PRUEBA * 24
    assert list(datos.columns) == list(COLUMNAS_ESQUEMA_ENTRADA)
    assert (datos["origen_dato"] == ORIGEN_SINTETICO).all()
    resultado = ValidadorEntrada().validar_serie(datos)
    assert resultado.reporte.total_imputados == 0


def test_generacion_reproducible_con_semilla(gestor: GestorBaseDatos) -> None:
    primera = AgenteDatos(gestor, semilla=1).generar_datos_sinteticos(dias=7)
    segunda = AgenteDatos(gestor, semilla=1).generar_datos_sinteticos(dias=7)
    otra = AgenteDatos(gestor, semilla=2).generar_datos_sinteticos(dias=7)
    pd.testing.assert_frame_equal(primera, segunda)
    assert not primera[COLUMNA_TIEMPO_RUTA_A].equals(otra[COLUMNA_TIEMPO_RUTA_A])


def test_patrones_realistas_del_dominio(agente: AgenteDatos) -> None:
    datos = agente.generar_datos_sinteticos(dias=365)
    laborables = datos[~datos["es_fin_de_semana"]]
    pico = laborables[laborables["hora_decimal"] == 8.0][COLUMNA_TIEMPO_RUTA_A].median()
    valle = laborables[laborables["hora_decimal"] == 3.0][COLUMNA_TIEMPO_RUTA_A].median()
    assert valle == pytest.approx(25.0, abs=3.0)
    assert pico > 55.0

    finde = datos[datos["es_fin_de_semana"] & (datos["hora_decimal"] == 8.0)]
    assert finde[COLUMNA_TIEMPO_RUTA_A].median() < pico

    # Ruta_A volátil y Ruta_B estable; Ruta_B gana en hora punta y pierde en valle.
    cv = datos[[COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B]].pipe(lambda d: d.std() / d.mean())
    assert cv[COLUMNA_TIEMPO_RUTA_A] > cv[COLUMNA_TIEMPO_RUTA_B]
    pico_b = laborables[laborables["hora_decimal"] == 8.0][COLUMNA_TIEMPO_RUTA_B].median()
    valle_b = laborables[laborables["hora_decimal"] == 3.0][COLUMNA_TIEMPO_RUTA_B].median()
    assert pico_b < pico and valle_b > valle

    quincena = laborables[laborables["es_quincena_o_fin_mes"] & (laborables["hora_decimal"] == 8.0)]
    assert quincena[COLUMNA_TIEMPO_RUTA_A].median() > pico
    assert datos["clima_adverso"].any()


def test_inyeccion_de_corrupcion(agente: AgenteDatos) -> None:
    datos = agente.generar_datos_sinteticos(dias=14, fraccion_corrupcion=0.05)
    assert datos[COLUMNA_TIEMPO_RUTA_A].isna().sum() == round(0.05 * 14 * 24)


@pytest.mark.parametrize(
    "kwargs",
    [{"dias": 0}, {"fraccion_corrupcion": 1.0}, {"fecha_inicio": "2025-01-01 00:30:00"}],
)
def test_parametros_de_generacion_invalidos(agente: AgenteDatos, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        agente.generar_datos_sinteticos(**kwargs)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Codificación cíclica
# -----------------------------------------------------------------------------


def test_codificacion_ciclica() -> None:
    datos = pd.DataFrame({"hora_decimal": [0.0, 6.0, 23.0], "dia_semana": [0, 0, 6]})
    codificados = codificar_variables_ciclicas(datos)
    assert codificados.loc[0, "hora_sin"] == pytest.approx(0.0)
    assert codificados.loc[0, "hora_cos"] == pytest.approx(1.0)
    assert codificados.loc[1, "hora_sin"] == pytest.approx(1.0)
    # Las 23:00 quedan cerca de las 00:00 en el espacio cíclico.
    distancia = np.hypot(
        codificados.loc[2, "hora_sin"] - codificados.loc[0, "hora_sin"],
        codificados.loc[2, "hora_cos"] - codificados.loc[0, "hora_cos"],
    )
    assert distancia < 0.3
    assert "hora_sin" not in datos


def test_codificacion_ciclica_exige_columnas() -> None:
    with pytest.raises(ErrorValidacionEntrada):
        codificar_variables_ciclicas(pd.DataFrame({"hora_decimal": [1.0]}))


# -----------------------------------------------------------------------------
# Ingestión
# -----------------------------------------------------------------------------


def test_ingesta_persiste_corruptos_como_null_y_carga_imputada(
    agente: AgenteDatos, gestor: GestorBaseDatos
) -> None:
    fuente = fuente_real()
    fuente.loc[5, COLUMNA_TIEMPO_RUTA_A] = 999.0
    fuente.loc[6, COLUMNA_TIEMPO_RUTA_B] = np.nan
    assert agente.ingerir_dataframe(fuente) == 48

    crudo = gestor.obtener_historico()
    assert np.isnan(crudo.loc[5, COLUMNA_TIEMPO_RUTA_A])
    assert np.isnan(crudo.loc[6, COLUMNA_TIEMPO_RUTA_B])
    assert (crudo["origen_dato"] == ORIGEN_REAL).all()

    resultado = agente.cargar_historico()
    esperado = (fuente.loc[4, COLUMNA_TIEMPO_RUTA_A] + fuente.loc[6, COLUMNA_TIEMPO_RUTA_A]) / 2
    assert resultado.datos.loc[5, COLUMNA_TIEMPO_RUTA_A] == pytest.approx(esperado)
    assert resultado.reporte.total_imputados == 2
    assert len(agente.reportes_validacion) == 2


def test_ingesta_regulariza_horas_ausentes(agente: AgenteDatos, gestor: GestorBaseDatos) -> None:
    fuente = fuente_real().drop(index=[10, 11]).sample(frac=1.0, random_state=0)
    assert agente.ingerir_dataframe(fuente) == 48
    assert gestor.obtener_historico()[COLUMNA_TIEMPO_RUTA_A].isna().sum() == 2


def test_ingesta_rechaza_duplicados_y_horas_inexactas(agente: AgenteDatos) -> None:
    fuente = fuente_real()
    with pytest.raises(ErrorIntegridadSecuencial):
        agente.ingerir_dataframe(pd.concat([fuente, fuente.iloc[[0]]]))
    fuente.loc[3, "fecha_hora"] = "2025-06-02 03:15:00"
    with pytest.raises(ErrorIntegridadSecuencial):
        agente.ingerir_dataframe(fuente)


def test_ingesta_rechaza_tipos_invalidos(agente: AgenteDatos) -> None:
    fuente = fuente_real().astype({COLUMNA_TIEMPO_RUTA_A: object})
    fuente.loc[2, COLUMNA_TIEMPO_RUTA_A] = "lento"
    with pytest.raises(ErrorValidacionEntrada, match="no numéricos"):
        agente.ingerir_dataframe(fuente)
    with pytest.raises(ErrorValidacionEntrada, match="obligatorias"):
        agente.ingerir_dataframe(fuente_real().drop(columns=[COLUMNA_TIEMPO_RUTA_B]))
    with pytest.raises(ErrorValidacionEntrada, match="origen_dato"):
        agente.ingerir_dataframe(fuente_real(), origen_dato="SIMULADO")


def test_ingesta_corrupta_aborta_sin_persistir(
    agente: AgenteDatos, gestor: GestorBaseDatos
) -> None:
    fuente = fuente_real()
    fuente.loc[:20, COLUMNA_TIEMPO_RUTA_A] = np.nan
    with pytest.raises(ErrorValidacionEntrada, match="umbral"):
        agente.ingerir_dataframe(fuente)
    assert gestor.contar_registros_historico() == 0


def test_ingesta_csv_con_clima(
    agente: AgenteDatos, gestor: GestorBaseDatos, tmp_path: Path
) -> None:
    fuente = fuente_real(24)
    fuente["clima_adverso"] = ["Sí" if i % 6 == 0 else "no" for i in range(24)]
    ruta = tmp_path / "sensores.csv"
    fuente.to_csv(ruta, index=False)
    assert agente.ingerir_csv(ruta) == 24
    assert gestor.obtener_historico()["clima_adverso"].sum() == 4

    fuente["clima_adverso"] = "quizás"
    fuente.to_csv(ruta, index=False)
    with pytest.raises(ErrorValidacionEntrada, match="booleanos"):
        agente.ingerir_csv(ruta, reemplazar_existentes=True)
    with pytest.raises(ErrorValidacionEntrada, match="No se pudo leer"):
        agente.ingerir_csv(tmp_path / "no_existe.csv")


def test_inicializar_datos_es_idempotente(agente: AgenteDatos, gestor: GestorBaseDatos) -> None:
    resultado = agente.inicializar_datos(dias=DIAS_PRUEBA)
    assert len(resultado.datos) == DIAS_PRUEBA * 24
    agente.inicializar_datos(dias=DIAS_PRUEBA)
    assert gestor.contar_registros_historico() == DIAS_PRUEBA * 24
    agente.inicializar_datos(forzar_regeneracion=True, dias=30)
    assert gestor.contar_registros_historico() == 30 * 24


def test_cargar_historico_vacio(agente: AgenteDatos) -> None:
    with pytest.raises(ErrorValidacionEntrada, match="No hay registros"):
        agente.cargar_historico()


# -----------------------------------------------------------------------------
# Ventanas, escalado y anti-leakage
# -----------------------------------------------------------------------------


@pytest.fixture
def serie(agente: AgenteDatos) -> pd.DataFrame:
    """Serie sintética saneada de ``DIAS_PRUEBA`` días persistida en SQLite."""
    return agente.inicializar_datos(dias=DIAS_PRUEBA).datos


def test_construir_conjuntos_formas(agente: AgenteDatos, serie: pd.DataFrame) -> None:
    conjuntos = agente.construir_conjuntos(serie, guardar_scalers=False)
    for x, y in (
        (conjuntos.x_entrenamiento, conjuntos.y_entrenamiento),
        (conjuntos.x_validacion, conjuntos.y_validacion),
        (conjuntos.x_prueba, conjuntos.y_prueba),
    ):
        assert len(x) == len(y) > 0
        assert x.shape[1:] == (PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS))
        assert y.shape[1:] == (PASOS_VENTANA_SALIDA, 2)
        assert x.dtype == np.float32 and y.dtype == np.float32
        assert np.isfinite(x).all() and np.isfinite(y).all()


def test_sin_fuga_de_datos_entre_tramos(agente: AgenteDatos, serie: pd.DataFrame) -> None:
    conjuntos = agente.construir_conjuntos(serie, guardar_scalers=False)
    duracion = pd.Timedelta(hours=PASOS_VENTANA_SALIDA)
    fin_entrenamiento = conjuntos.inicio_objetivo_entrenamiento.max() + duracion
    fin_validacion = conjuntos.inicio_objetivo_validacion.max() + duracion
    assert conjuntos.inicio_objetivo_validacion.min() >= fin_entrenamiento
    assert conjuntos.inicio_objetivo_prueba.min() >= fin_validacion

    # Los scalers solo conocen el tramo de entrenamiento.
    n_entrenamiento = int(len(serie) * 0.70)
    tramo = serie.iloc[:n_entrenamiento]
    assert agente.scaler_salida is not None
    np.testing.assert_allclose(
        agente.scaler_salida.data_max_,
        tramo[[COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B]].max().to_numpy(),
    )
    assert conjuntos.y_entrenamiento.max() <= 1.0 + 1e-6


def test_ventana_corresponde_a_la_serie(agente: AgenteDatos, serie: pd.DataFrame) -> None:
    conjuntos = agente.construir_conjuntos(serie, guardar_scalers=False)
    y_minutos = agente.invertir_escala_salida(conjuntos.y_entrenamiento[:1])
    inicio = PASOS_VENTANA_ENTRADA
    esperado = serie[[COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B]].iloc[
        inicio : inicio + PASOS_VENTANA_SALIDA
    ]
    np.testing.assert_allclose(y_minutos[0], esperado.to_numpy(), rtol=1e-4)


def test_construir_conjuntos_exige_datos_saneados(agente: AgenteDatos, serie: pd.DataFrame) -> None:
    sucia = serie.copy()
    sucia.loc[3, COLUMNA_TIEMPO_RUTA_A] = np.nan
    with pytest.raises(ErrorValidacionEntrada, match="sin sanear"):
        agente.construir_conjuntos(sucia, guardar_scalers=False)


def test_historia_insuficiente(agente: AgenteDatos) -> None:
    corta = agente.inicializar_datos(dias=25).datos
    with pytest.raises(ErrorValidacionEntrada, match="insuficiente"):
        agente.construir_conjuntos(corta, guardar_scalers=False)


def test_ventana_de_inferencia(agente: AgenteDatos, serie: pd.DataFrame) -> None:
    with pytest.raises(ErrorSistemaTrafico, match="scaler"):
        agente.preparar_ventana_inferencia()
    agente.construir_conjuntos(serie, guardar_scalers=False)
    ventana = agente.preparar_ventana_inferencia()
    assert ventana.x.shape == (1, PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS))
    assert len(ventana.fechas_objetivo) == PASOS_VENTANA_SALIDA
    assert ventana.fechas_objetivo[0] == serie["fecha_hora"].iloc[-1] + pd.Timedelta(hours=1)
    assert ventana.reporte.alcance == "VENTANA"


def test_ventana_de_inferencia_incompleta(agente: AgenteDatos, serie: pd.DataFrame) -> None:
    agente.construir_conjuntos(serie, guardar_scalers=False)
    with pytest.raises(ErrorIntegridadSecuencial):
        agente.preparar_ventana_inferencia(serie.tail(PASOS_VENTANA_ENTRADA - 24))


def test_guardar_y_cargar_scalers(
    agente: AgenteDatos, gestor: GestorBaseDatos, serie: pd.DataFrame, tmp_path: Path
) -> None:
    rutas = (tmp_path / "artifacts" / "x.pkl", tmp_path / "artifacts" / "y.pkl")
    agente.construir_conjuntos(serie, guardar_scalers=False)
    agente.guardar_scalers(*rutas)

    nuevo = AgenteDatos(gestor)
    nuevo.cargar_scalers(*rutas)
    assert nuevo.scaler_entrada is not None and agente.scaler_entrada is not None
    np.testing.assert_array_equal(nuevo.scaler_entrada.data_min_, agente.scaler_entrada.data_min_)

    joblib.dump({"no": "scaler"}, rutas[0])
    with pytest.raises(ErrorSistemaTrafico, match="MinMaxScaler"):
        nuevo.cargar_scalers(*rutas)


# =============================================================================
# AgenteAuditor
# =============================================================================


@pytest.fixture
def auditor(gestor: GestorBaseDatos, tmp_path: Path) -> AgenteAuditor:
    """Auditor con log de eventos aislado."""
    return AgenteAuditor(gestor, ruta_log_eventos=tmp_path / "logs" / "eventos.jsonl")


def leer_jsonl(ruta: Path) -> list[dict[str, Any]]:
    """Lee un archivo JSONL completo."""
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines()]


def recomendacion(**cambios: Any) -> Any:
    """Recomendación válida (``RED_LSTM`` sin alertas) con cambios opcionales."""
    campos: dict[str, Any] = {
        "ruta_recomendada": "Ruta_A",
        "tiempo_estimado_min": 38.0,
        "ahorro_estimado_min": 4.5,
        "hora_optima_salida": "06:45",
        "nivel_confianza_pct": 88.0,
        "modo_inferencia": "RED_LSTM",
        "mae_modelo": 7.5,
    }
    campos.update(cambios)
    return ValidadorSalida().construir_recomendacion(**campos)


@pytest.mark.parametrize("kwargs", [{"umbral_p_valor": 0.0}, {"umbral_z_score": -1.0}])
def test_auditor_umbrales_invalidos(gestor: GestorBaseDatos, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AgenteAuditor(gestor, **kwargs)


# --- Trazabilidad -------------------------------------------------------------


def test_registrar_inferencia_en_sqlite_y_jsonl(
    auditor: AgenteAuditor, gestor: GestorBaseDatos
) -> None:
    rec = recomendacion(alerta_drift=True)
    id_transaccion = auditor.registrar_inferencia(rec)
    assert id_transaccion == str(rec.id_transaccion)

    fila = gestor.obtener_auditorias().iloc[0]
    assert fila["estado_auditoria"] == "APROBADO_CON_ALERTAS"
    assert bool(fila["alerta_drift"])
    lineas = leer_jsonl(gestor.ruta_log_jsonl)
    assert [linea["id_transaccion"] for linea in lineas] == [id_transaccion]


def test_registrar_reportes_validacion_vacia_la_cola(
    agente: AgenteDatos, auditor: AgenteAuditor
) -> None:
    agente.inicializar_datos(dias=20)
    pendientes = len(agente.reportes_validacion)
    assert pendientes >= 2  # ingesta + carga
    assert auditor.registrar_reportes_validacion(agente.reportes_validacion) == pendientes
    assert agente.reportes_validacion == []

    eventos = leer_jsonl(auditor.ruta_log_eventos)
    assert {e["tipo_evento"] for e in eventos} == {"VALIDACION_ENTRADA"}
    assert eventos[0]["datos"]["alcance"] == "SERIE"


def test_registrar_reportes_sin_vaciar(agente: AgenteDatos, auditor: AgenteAuditor) -> None:
    agente.inicializar_datos(dias=20)
    pendientes = list(agente.reportes_validacion)
    auditor.registrar_reportes_validacion(agente.reportes_validacion, vaciar=False)
    assert agente.reportes_validacion == pendientes


def test_registrar_error_del_dominio_y_generico(auditor: AgenteAuditor) -> None:
    auditor.registrar_error(
        ErrorValidacionEntrada("ventana corrupta", detalles={"fraccion": np.float64(0.2)}),
        contexto="inferencia",
    )
    auditor.registrar_error(RuntimeError("fallo inesperado"), contexto="entrenamiento")
    primero, segundo = leer_jsonl(auditor.ruta_log_eventos)
    assert primero["tipo_evento"] == "ERROR"
    assert primero["datos"] == {
        "codigo": "ErrorValidacionEntrada",
        "mensaje": "ventana corrupta",
        "detalles": {"fraccion": 0.2},
        "contexto": "inferencia",
    }
    assert segundo["datos"]["codigo"] == "RuntimeError"


def test_registrar_evento_invalido(auditor: AgenteAuditor, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        auditor.registrar_evento("INVENTADO", {})
    with pytest.raises(ErrorBaseDatos):
        auditor.registrar_evento("METRICAS_MODELO", {"mae": float("nan")})
    roto = AgenteAuditor(auditor.gestor_bd, ruta_log_eventos=tmp_path)  # es un directorio
    with pytest.raises(ErrorBaseDatos):
        roto.registrar_evento("METRICAS_MODELO", {"mae": 1.0})


# --- Métricas -----------------------------------------------------------------


def test_calcular_metricas_valores_conocidos(auditor: AgenteAuditor) -> None:
    reales = np.array([[10.0, 20.0], [30.0, 40.0]])
    predichos = np.array([[12.0, 18.0], [30.0, 44.0]])
    metricas = auditor.calcular_metricas(reales, predichos, etiqueta="prueba")
    assert metricas.n_observaciones == 2
    assert metricas.mae == pytest.approx(2.0)
    assert metricas.rmse == pytest.approx(np.sqrt(6.0))
    assert metricas.sesgo == pytest.approx(1.0)
    assert metricas.mae_por_ruta == pytest.approx(
        {COLUMNA_TIEMPO_RUTA_A: 1.0, COLUMNA_TIEMPO_RUTA_B: 3.0}
    )
    assert metricas.rmse_por_ruta[COLUMNA_TIEMPO_RUTA_B] == pytest.approx(np.sqrt(10.0))
    assert metricas.sesgo_por_ruta == pytest.approx(
        {COLUMNA_TIEMPO_RUTA_A: 1.0, COLUMNA_TIEMPO_RUTA_B: 1.0}
    )


def test_calcular_metricas_tensores_3d(auditor: AgenteAuditor) -> None:
    reales = np.full((3, PASOS_VENTANA_SALIDA, 2), 40.0)
    metricas = auditor.calcular_metricas(reales, reales + 5.0)
    assert metricas.n_observaciones == 3 * PASOS_VENTANA_SALIDA
    assert metricas.mae == pytest.approx(5.0) and metricas.sesgo == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("reales", "predichos"),
    [
        (np.zeros((4, 2)), np.zeros((5, 2))),
        (np.zeros((4, 3)), np.zeros((4, 3))),
        (np.zeros((0, 2)), np.zeros((0, 2))),
        (np.array([[np.nan, 1.0]]), np.array([[1.0, 1.0]])),
    ],
)
def test_calcular_metricas_entradas_invalidas(
    auditor: AgenteAuditor, reales: np.ndarray, predichos: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        auditor.calcular_metricas(reales, predichos)


def test_registrar_metricas(auditor: AgenteAuditor) -> None:
    metricas = auditor.calcular_metricas(np.ones((2, 2)) * 30, np.ones((2, 2)) * 33)
    auditor.registrar_metricas(metricas)
    (evento,) = leer_jsonl(auditor.ruta_log_eventos)
    assert evento["tipo_evento"] == "METRICAS_MODELO"
    assert evento["datos"]["mae"] == pytest.approx(3.0)


# --- Data Drift ---------------------------------------------------------------


@pytest.fixture
def serie_anual(agente: AgenteDatos) -> pd.DataFrame:
    """Un año sintético estacionario (sin persistir)."""
    return agente.generar_datos_sinteticos(dias=365)


def test_sin_drift_en_serie_estacionaria(auditor: AgenteAuditor, serie_anual: pd.DataFrame) -> None:
    reporte = auditor.evaluar_drift(serie_anual)
    assert not reporte.hay_drift
    assert {r.columna for r in reporte.resultados} == {COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B}
    assert reporte.fin_reciente == serie_anual["fecha_hora"].iloc[-1]
    assert (reporte.inicio_reciente - reporte.fin_referencia) == pd.Timedelta(hours=1)
    (evento,) = leer_jsonl(auditor.ruta_log_eventos)
    assert evento["tipo_evento"] == "DATA_DRIFT" and evento["datos"]["hay_drift"] is False


def test_drift_detectado_ante_desplazamiento(
    auditor: AgenteAuditor, serie_anual: pd.DataFrame
) -> None:
    desplazada = serie_anual.copy()
    recientes = desplazada.index[-14 * 24 :]
    desplazada.loc[recientes, COLUMNA_TIEMPO_RUTA_A] *= 1.25
    reporte = auditor.evaluar_drift(desplazada, registrar=False)
    assert reporte.hay_drift
    resultado_a = next(r for r in reporte.resultados if r.columna == COLUMNA_TIEMPO_RUTA_A)
    assert set(resultado_a.criterios_activados) == {"KOLMOGOROV_SMIRNOV", "Z_SCORE"}
    assert resultado_a.media_reciente > resultado_a.media_referencia
    assert not auditor.ruta_log_eventos.exists()


def test_drift_con_alerta_registra_antes_de_lanzar(
    auditor: AgenteAuditor, serie_anual: pd.DataFrame
) -> None:
    desplazada = serie_anual.copy()
    desplazada.loc[desplazada.index[-14 * 24 :], COLUMNA_TIEMPO_RUTA_B] += 15.0
    with pytest.raises(AlertaDataDrift) as error:
        auditor.evaluar_drift(desplazada, lanzar_alerta=True)
    assert error.value.detalles["hay_drift"] is True
    assert leer_jsonl(auditor.ruta_log_eventos)[0]["datos"]["hay_drift"] is True


def test_drift_historia_insuficiente_o_sucia(
    auditor: AgenteAuditor, serie_anual: pd.DataFrame
) -> None:
    with pytest.raises(ErrorValidacionEntrada, match="insuficiente"):
        auditor.evaluar_drift(serie_anual.tail(41 * 24))
    sucia = serie_anual.copy()
    sucia.loc[sucia.index[-1], COLUMNA_TIEMPO_RUTA_A] = np.nan
    with pytest.raises(ErrorValidacionEntrada, match="no finitos"):
        auditor.evaluar_drift(sucia)
    with pytest.raises(ErrorValidacionEntrada, match="días completos"):
        auditor.detectar_data_drift(np.ones(50), np.ones(48), COLUMNA_TIEMPO_RUTA_A)


# --- Resumen ------------------------------------------------------------------


def test_resumir_auditorias(auditor: AgenteAuditor) -> None:
    vacio = auditor.resumir_auditorias()
    assert vacio.total_inferencias == 0 and vacio.confianza_media_pct is None

    auditor.registrar_inferencia(recomendacion(nivel_confianza_pct=80.0, mae_modelo=6.0))
    auditor.registrar_inferencia(recomendacion(nivel_confianza_pct=90.0, mae_modelo=8.0))
    auditor.registrar_inferencia(
        recomendacion(
            modo_inferencia="FALLBACK_HEURISTICO",
            nivel_confianza_pct=40.0,
            mae_modelo=None,
            motivos_fallback=("MODELO_NO_DISPONIBLE",),
            alerta_drift=True,
        )
    )
    resumen = auditor.resumir_auditorias()
    assert resumen.total_inferencias == 3
    assert resumen.por_modo == {"RED_LSTM": 2, "FALLBACK_HEURISTICO": 1}
    assert resumen.tasa_fallback == pytest.approx(1 / 3)
    assert resumen.confianza_media_pct == pytest.approx(70.0)
    assert resumen.mae_medio == pytest.approx(7.0)
    assert resumen.alertas_drift == 1


# =============================================================================
# AgenteEstrategaRuta
# =============================================================================

INICIO_HORIZONTE = pd.Timestamp("2025-04-07 00:00:00")  # lunes


def prediccion_horaria(
    ruta_a: float | dict[int, float] = 50.0, ruta_b: float | dict[int, float] = 50.0
) -> pd.DataFrame:
    """Predicción de 168 h; cada ruta es un valor fijo o un mapa hora del día → minutos."""
    fechas = pd.date_range(INICIO_HORIZONTE, periods=PASOS_VENTANA_SALIDA, freq="h")

    def valores(definicion: float | dict[int, float]) -> list[float]:
        if isinstance(definicion, dict):
            return [definicion.get(f.hour, 30.0) for f in fechas]
        return [definicion] * len(fechas)

    return pd.DataFrame(
        {
            "fecha_hora": fechas,
            COLUMNA_TIEMPO_RUTA_A: valores(ruta_a),
            COLUMNA_TIEMPO_RUTA_B: valores(ruta_b),
        }
    )


@pytest.fixture
def estratega(gestor: GestorBaseDatos) -> AgenteEstrategaRuta:
    """Estratega con los desplazamientos por defecto [-30, -15, 0, +15]."""
    return AgenteEstrategaRuta(gestor)


def test_adelantar_salida_por_la_ruta_habitual(estratega: AgenteEstrategaRuta) -> None:
    prediccion = prediccion_horaria(ruta_a={7: 40.0, 8: 60.0}, ruta_b=55.0)
    plan = estratega.planificar_viaje(prediccion, date(2025, 4, 8), "07:30")
    assert plan.ruta_recomendada == "Ruta_A"
    assert plan.hora_optima_salida == "07:00"
    assert plan.mejor_opcion.desplazamiento_min == -30
    assert plan.tiempo_estimado_min == pytest.approx(40.0)
    assert plan.tiempo_ruta_habitual_min == pytest.approx(50.0)  # 07:30 interpolado
    assert plan.ahorro_estimado_min == pytest.approx(10.0)
    assert len(plan.alternativas) == 8


def test_interpolacion_lineal_entre_horas(estratega: AgenteEstrategaRuta) -> None:
    prediccion = prediccion_horaria(ruta_a={7: 40.0, 8: 60.0}, ruta_b=55.0)
    plan = estratega.planificar_viaje(prediccion, date(2025, 4, 8), time(7, 30))
    opcion = next(
        o for o in plan.alternativas if o.ruta == "Ruta_A" and o.desplazamiento_min == -15
    )
    assert opcion.hora_salida == "07:15"
    assert opcion.tiempo_estimado_min == pytest.approx(45.0)


def test_cambiar_a_ruta_alterna_sin_mover_la_salida(estratega: AgenteEstrategaRuta) -> None:
    plan = estratega.planificar_viaje(
        prediccion_horaria(ruta_a=50.0, ruta_b=30.0), date(2025, 4, 9), "18:00"
    )
    assert plan.ruta_recomendada == "Ruta_B"
    assert plan.mejor_opcion.desplazamiento_min == 0  # empate en tiempo → menor |d|
    assert plan.ahorro_estimado_min == pytest.approx(20.0)


def test_empate_entre_rutas_prefiere_el_orden_configurado(estratega: AgenteEstrategaRuta) -> None:
    plan = estratega.planificar_viaje(prediccion_horaria(), date(2025, 4, 9), "12:00")
    assert plan.ruta_recomendada == "Ruta_A" and plan.ahorro_estimado_min == 0.0


def test_salida_fuera_del_horizonte(estratega: AgenteEstrategaRuta) -> None:
    with pytest.raises(ErrorValidacionEntrada, match="horizonte"):
        estratega.planificar_viaje(prediccion_horaria(), date(2025, 4, 7), "00:10")
    with pytest.raises(ErrorValidacionEntrada, match="horizonte"):
        estratega.planificar_viaje(prediccion_horaria(), date(2025, 4, 20), "08:00")


@pytest.mark.parametrize("hora", ["7:3", "25:00", "siete"])
def test_hora_deseada_invalida(estratega: AgenteEstrategaRuta, hora: str) -> None:
    with pytest.raises(ErrorValidacionEntrada, match="HH:MM"):
        estratega.planificar_viaje(prediccion_horaria(), date(2025, 4, 8), hora)


def test_prediccion_invalida_para_planificar(estratega: AgenteEstrategaRuta) -> None:
    con_nan = prediccion_horaria()
    con_nan.loc[30, COLUMNA_TIEMPO_RUTA_A] = np.nan
    with pytest.raises(ErrorValidacionEntrada, match="acótela"):
        estratega.planificar_viaje(con_nan, date(2025, 4, 8), "08:00")
    with pytest.raises(ErrorValidacionEntrada, match="continua"):
        estratega.planificar_viaje(prediccion_horaria().drop(index=[5]), date(2025, 4, 8), "08:00")
    with pytest.raises(ErrorValidacionEntrada, match="ruta_habitual"):
        estratega.planificar_viaje(prediccion_horaria(), date(2025, 4, 8), "08:00", "Ruta_Z")


@pytest.mark.parametrize("desplazamientos", [(-15, 15), (0, 0, 15)])
def test_desplazamientos_invalidos(desplazamientos: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        AgenteEstrategaRuta(desplazamientos_min=desplazamientos)


def test_resumir_tendencia_de_7_dias(estratega: AgenteEstrategaRuta) -> None:
    prediccion = prediccion_horaria(ruta_a={7: 70.0, 8: 70.0, 9: 70.0}, ruta_b=45.0)
    tendencia = estratega.resumir_tendencia(prediccion)
    assert len(tendencia) == 7
    assert tendencia[0].fecha == date(2025, 4, 7)
    assert tendencia[0].manana_por_ruta == {"Ruta_A": 70.0, "Ruta_B": 45.0}
    assert tendencia[0].mejor_ruta_manana == "Ruta_B"
    assert tendencia[0].mejor_ruta_tarde == "Ruta_A"  # 30 min fuera de la punta de mañana


def historia_semanal(semanas: int, fin: pd.Timestamp = INICIO_HORIZONTE) -> pd.DataFrame:
    """Historia que repite un patrón semanal exacto: 20 + 2·día + hora."""
    fechas = pd.date_range(end=fin - pd.Timedelta(hours=1), periods=semanas * 168, freq="h")
    patron = 20.0 + 2.0 * fechas.weekday + fechas.hour
    return pd.DataFrame(
        {
            "fecha_hora": fechas,
            "dia_semana": fechas.weekday,
            "hora_decimal": fechas.hour.astype(float),
            COLUMNA_TIEMPO_RUTA_A: patron,
            COLUMNA_TIEMPO_RUTA_B: patron + 5.0,
        }
    )


def test_fallback_heuristico_usa_la_mediana_por_dia_y_hora(estratega: AgenteEstrategaRuta) -> None:
    historia = historia_semanal(4)
    antigua = historia_semanal(4, fin=INICIO_HORIZONTE - pd.Timedelta(weeks=4))
    antigua[[COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B]] = 200.0  # fuera de las 4 semanas
    futura = historia_semanal(1, fin=INICIO_HORIZONTE + pd.Timedelta(weeks=1))
    futura[[COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B]] = 999.0  # nunca debe usarse
    completa = pd.concat([antigua, historia, futura], ignore_index=True)

    fechas = pd.date_range(INICIO_HORIZONTE, periods=PASOS_VENTANA_SALIDA, freq="h")
    resultado = estratega.generar_prediccion_heuristica(fechas, historico=completa, semanas=4)
    esperado = 20.0 + 2.0 * fechas.weekday + fechas.hour
    np.testing.assert_allclose(resultado.datos[COLUMNA_TIEMPO_RUTA_A], esperado)
    np.testing.assert_allclose(resultado.datos[COLUMNA_TIEMPO_RUTA_B], esperado + 5.0)
    assert resultado.nivel_confianza == 1.0  # patrón exacto: dispersión nula
    assert (resultado.desviacion[COLUMNA_TIEMPO_RUTA_A] == 0.0).all()


def test_fallback_heuristico_historia_insuficiente(estratega: AgenteEstrategaRuta) -> None:
    historia = historia_semanal(1)
    historia = historia[historia["dia_semana"] != 2]  # sin miércoles
    fechas = pd.date_range(INICIO_HORIZONTE, periods=PASOS_VENTANA_SALIDA, freq="h")
    with pytest.raises(ErrorValidacionEntrada, match="insuficiente"):
        estratega.generar_prediccion_heuristica(fechas, historico=historia)
    with pytest.raises(ValueError):
        AgenteEstrategaRuta().generar_prediccion_heuristica(fechas)  # sin gestor ni historia


def test_fallback_heuristico_desde_sqlite_y_plan(
    agente: AgenteDatos, estratega: AgenteEstrategaRuta
) -> None:
    serie = agente.inicializar_datos(dias=DIAS_PRUEBA).datos
    ultima = serie["fecha_hora"].iloc[-1]
    fechas = pd.date_range(ultima + pd.Timedelta(hours=1), periods=PASOS_VENTANA_SALIDA, freq="h")
    heuristica = estratega.generar_prediccion_heuristica(fechas)
    assert 0.0 <= heuristica.nivel_confianza <= 1.0
    acotada = ValidadorSalida().acotar_prediccion(
        heuristica.datos[[COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B]].to_numpy(), fechas
    )
    dia = (fechas[0] + pd.Timedelta(days=1)).date()
    plan = estratega.planificar_viaje(acotada.datos, dia, "08:00")
    assert plan.ruta_recomendada == "Ruta_B"  # en hora punta la vía alterna es más rápida
    assert plan.ahorro_estimado_min > 0


# =============================================================================
# AgentePredictorLSTM (red diminuta para que la suite sea rápida)
# =============================================================================


@pytest.fixture(scope="module")
def entorno_lstm(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Entrena una única vez una red diminuta compartida por las pruebas del módulo."""
    pytest.importorskip("keras")
    from src.agents import AgentePredictorLSTM

    directorio = tmp_path_factory.mktemp("lstm")
    gestor_lstm = GestorBaseDatos(directorio / "t.db", directorio / "a.jsonl")
    datos = AgenteDatos(gestor_lstm, semilla=3)
    auditor_lstm = AgenteAuditor(gestor_lstm, ruta_log_eventos=directorio / "eventos.jsonl")
    conjuntos = datos.construir_conjuntos(
        datos.inicializar_datos(dias=DIAS_PRUEBA).datos, guardar_scalers=False
    )
    predictor = AgentePredictorLSTM(
        datos.invertir_escala_salida, auditor_lstm, unidades_lstm=4, unidades_densa=8
    )
    resultado = predictor.entrenar(conjuntos, epocas=2, tamano_lote=8)
    yield {
        "directorio": directorio,
        "gestor": gestor_lstm,
        "datos": datos,
        "auditor": auditor_lstm,
        "conjuntos": conjuntos,
        "predictor": predictor,
        "resultado": resultado,
        "clase": AgentePredictorLSTM,
    }


def test_entrenamiento_y_metricas_en_minutos(entorno_lstm: dict[str, Any]) -> None:
    resultado = entorno_lstm["resultado"]
    predictor = entorno_lstm["predictor"]
    assert resultado.epocas_ejecutadas == 2
    assert {"loss", "val_loss", "mae", "val_mae"} <= set(resultado.historial)
    assert predictor.mae_validacion == pytest.approx(resultado.metricas_validacion.mae)
    assert 0.0 < resultado.metricas_validacion.mae < 100.0  # minutos, no escala [0, 1]
    eventos = leer_jsonl(entorno_lstm["auditor"].ruta_log_eventos)
    etiquetas = [e["datos"]["etiqueta"] for e in eventos if e["tipo_evento"] == "METRICAS_MODELO"]
    assert etiquetas == ["validacion", "prueba"]


def test_arquitectura_bidireccional(entorno_lstm: dict[str, Any]) -> None:
    modelo = entorno_lstm["predictor"].modelo
    nombres = [capa.name for capa in modelo.layers]
    assert "bilstm" in nombres and "dropout_lstm" in nombres and "dropout_densa" in nombres
    assert type(modelo.get_layer("bilstm")).__name__ == "Bidirectional"
    assert modelo.get_layer("dropout_lstm").rate == pytest.approx(0.25)
    assert modelo.input_shape == (None, PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS))
    assert modelo.output_shape == (None, PASOS_VENTANA_SALIDA, 2)
    assert type(modelo.loss).__name__ == "Huber"


def test_prediccion_determinista(entorno_lstm: dict[str, Any]) -> None:
    predictor, conjuntos = entorno_lstm["predictor"], entorno_lstm["conjuntos"]
    primera = predictor.predecir(conjuntos.x_prueba[:2])
    assert primera.shape == (2, PASOS_VENTANA_SALIDA, 2)
    np.testing.assert_allclose(primera, predictor.predecir(conjuntos.x_prueba[:2]))


def test_monte_carlo_dropout(entorno_lstm: dict[str, Any]) -> None:
    predictor, conjuntos = entorno_lstm["predictor"], entorno_lstm["conjuntos"]
    incertidumbre = predictor.predecir_con_incertidumbre(conjuntos.x_prueba[0], iteraciones=10)
    assert incertidumbre.media.shape == (PASOS_VENTANA_SALIDA, 2)
    assert (incertidumbre.desviacion > 0).any()  # el dropout está activo
    assert (incertidumbre.limite_inferior <= incertidumbre.media).all()
    assert (incertidumbre.media <= incertidumbre.limite_superior).all()
    assert 0.0 <= incertidumbre.nivel_confianza <= 1.0
    with pytest.raises(ErrorValidacionEntrada, match="única"):
        predictor.predecir_con_incertidumbre(conjuntos.x_prueba[:2])
    with pytest.raises(ValueError):
        predictor.predecir_con_incertidumbre(conjuntos.x_prueba[0], iteraciones=1)


def test_validacion_de_entradas_del_predictor(entorno_lstm: dict[str, Any]) -> None:
    predictor = entorno_lstm["predictor"]
    with pytest.raises(ErrorValidacionEntrada, match="forma"):
        predictor.predecir(np.zeros((1, 100, len(COLUMNAS_CARACTERISTICAS))))
    con_nan = np.zeros((1, PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS)))
    con_nan[0, 0, 0] = np.nan
    with pytest.raises(ErrorValidacionEntrada, match="no finitos"):
        predictor.predecir(con_nan)


def test_guardar_y_cargar_modelo(entorno_lstm: dict[str, Any]) -> None:
    directorio: Path = entorno_lstm["directorio"]
    predictor, conjuntos = entorno_lstm["predictor"], entorno_lstm["conjuntos"]
    ruta_modelo, ruta_meta = directorio / "modelo.keras", directorio / "modelo.json"
    predictor.guardar(ruta_modelo, ruta_meta, resultado=entorno_lstm["resultado"])
    metadatos = json.loads(ruta_meta.read_text(encoding="utf-8"))
    assert metadatos["mae_validacion"] == pytest.approx(predictor.mae_validacion)
    assert metadatos["hiperparametros"]["unidades_lstm"] == 4

    cargado = entorno_lstm["clase"](
        entorno_lstm["datos"].invertir_escala_salida, entorno_lstm["auditor"]
    )
    cargado.cargar(ruta_modelo, ruta_meta)
    assert cargado.mae_validacion == pytest.approx(predictor.mae_validacion)
    np.testing.assert_allclose(
        cargado.predecir(conjuntos.x_prueba[:1]),
        predictor.predecir(conjuntos.x_prueba[:1]),
        rtol=1e-5,
    )

    sin_meta = entorno_lstm["clase"](
        entorno_lstm["datos"].invertir_escala_salida, entorno_lstm["auditor"]
    )
    sin_meta.cargar(ruta_modelo, directorio / "no_existe.json")
    assert sin_meta.mae_validacion is None  # → MAE_DESCONOCIDO en el protocolo fallback

    with pytest.raises(ValueError, match=".keras"):
        predictor.guardar(directorio / "modelo.h5", ruta_meta)


def test_modelo_no_disponible(entorno_lstm: dict[str, Any]) -> None:
    import keras

    clase, directorio = entorno_lstm["clase"], entorno_lstm["directorio"]
    vacio = clase(entorno_lstm["datos"].invertir_escala_salida, entorno_lstm["auditor"])
    assert not vacio.esta_entrenado
    with pytest.raises(ErrorModeloNoDisponible):
        vacio.predecir(entorno_lstm["conjuntos"].x_prueba[:1])
    with pytest.raises(ErrorModeloNoDisponible, match="No existe"):
        vacio.cargar(directorio / "inexistente.keras")

    entrada = keras.Input(shape=(10, 3))
    ajeno = keras.Model(entrada, keras.layers.Dense(1)(entrada))
    ajeno.save(directorio / "ajeno.keras")
    with pytest.raises(ErrorModeloNoDisponible, match="incompatible"):
        vacio.cargar(directorio / "ajeno.keras")


@pytest.mark.parametrize(
    "kwargs", [{"tasa_dropout": 0.0}, {"unidades_lstm": 0}, {"tasa_aprendizaje": -1.0}]
)
def test_hiperparametros_invalidos(entorno_lstm: dict[str, Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        entorno_lstm["clase"](
            entorno_lstm["datos"].invertir_escala_salida, entorno_lstm["auditor"], **kwargs
        )


def test_flujo_de_inferencia_de_punta_a_punta(entorno_lstm: dict[str, Any]) -> None:
    """Datos → LSTM + MC Dropout → guardrails de salida → estratega → contrato → auditoría."""
    datos, predictor, auditor_lstm = (
        entorno_lstm["datos"],
        entorno_lstm["predictor"],
        entorno_lstm["auditor"],
    )
    ventana = datos.preparar_ventana_inferencia()
    incertidumbre = predictor.predecir_con_incertidumbre(ventana.x, iteraciones=10)

    validador = ValidadorSalida()
    acotada = validador.acotar_prediccion(incertidumbre.media, ventana.fechas_objetivo)
    evaluacion = validador.evaluar_confianza(
        incertidumbre.nivel_confianza, predictor.mae_validacion
    )
    plan = AgenteEstrategaRuta(entorno_lstm["gestor"]).planificar_viaje(
        acotada.datos, (ventana.fechas_objetivo[0] + pd.Timedelta(days=1)).date(), "07:30"
    )
    recomendacion_final = validador.construir_recomendacion(
        ruta_recomendada=plan.ruta_recomendada,
        tiempo_estimado_min=plan.tiempo_estimado_min,
        ahorro_estimado_min=plan.ahorro_estimado_min,
        hora_optima_salida=plan.hora_optima_salida,
        nivel_confianza_pct=round(incertidumbre.nivel_confianza * 100.0, 2),
        modo_inferencia=evaluacion.modo_inferencia,
        mae_modelo=predictor.mae_validacion,
        motivos_fallback=evaluacion.motivos_fallback,
        alerta_extrapolacion=acotada.alerta_extrapolacion,
    )
    id_transaccion = auditor_lstm.registrar_inferencia(recomendacion_final)
    assert entorno_lstm["gestor"].obtener_auditorias().iloc[0]["id_transaccion"] == id_transaccion
