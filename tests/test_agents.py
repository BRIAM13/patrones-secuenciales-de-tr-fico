"""Pruebas del ``AgenteDatos``: generación, ingestión, codificación y ventanas."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.agents import AgenteDatos, codificar_variables_ciclicas
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
    ErrorIntegridadSecuencial,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
    ValidadorEntrada,
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
