"""Agente Auditor (``AgenteAuditor``): trazabilidad, métricas y *Data Drift*.

Responsabilidades (Fase 3 de ``CONTEXT.md``):

* **Trazabilidad de inferencias:** cada ``RecomendacionTrafico`` se registra
  en ``auditoria_inferencias`` y, en la misma transacción, en
  ``logs/auditoria_trafico.jsonl`` (vía ``GestorBaseDatos``).
* **Eventos MLOps:** reportes de los guardrails de entrada, errores del
  dominio, métricas y resultados de drift se anexan como líneas JSON
  inmutables en ``logs/eventos_mlops.jsonl`` (Regla 3.2.1: ninguna anomalía
  queda sin registrar).
* **Métricas de rendimiento:** ``MAE``, ``RMSE`` y sesgo (``predicho − real``),
  globales y por ruta.
* **Detección de *Data Drift*:** compara los últimos ``14`` días con los
  ``28`` previos mediante la prueba de Kolmogorov-Smirnov sobre los datos
  horarios y un Z-score de la media diaria; basta con que uno de los dos
  criterios se active para declarar drift.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping, MutableSequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import (
    COLUMNA_FECHA_HORA,
    COLUMNAS_OBJETIVO,
    DIAS_RECIENTES_DRIFT,
    DIAS_REFERENCIA_DRIFT,
    EVENTO_DRIFT,
    EVENTO_ERROR,
    EVENTO_METRICAS,
    EVENTO_VALIDACION_ENTRADA,
    HORAS_POR_DIA,
    MODO_FALLBACK_HEURISTICO,
    MODOS_INFERENCIA_VALIDOS,
    RUTA_LOG_EVENTOS,
    TIPOS_EVENTO_VALIDOS,
    UMBRAL_P_VALOR_KS,
    UMBRAL_Z_SCORE_DRIFT,
)
from src.database.db_manager import GestorBaseDatos
from src.guardrails.exceptions import (
    AlertaDataDrift,
    ErrorBaseDatos,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
)
from src.guardrails.input_validator import ReporteValidacion
from src.guardrails.output_validator import RecomendacionTrafico

registro_log: logging.Logger = logging.getLogger(__name__)

_EPSILON_DESVIACION: Final[float] = 1e-9
"""Desviación mínima para evitar divisiones por cero en el Z-score."""

CRITERIO_KS: Final[str] = "KOLMOGOROV_SMIRNOV"
CRITERIO_Z_SCORE: Final[str] = "Z_SCORE"


# =============================================================================
# Modelos de resultados (serializables a JSON)
# =============================================================================


class MetricasRendimiento(BaseModel):
    """Métricas de error de un conjunto de predicciones (minutos).

    ``sesgo`` es la media de ``predicho − real``: positivo indica que el
    modelo sobrestima los tiempos de traslado.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    etiqueta: str
    n_observaciones: int = Field(ge=1)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    sesgo: float
    mae_por_ruta: dict[str, float]
    rmse_por_ruta: dict[str, float]
    sesgo_por_ruta: dict[str, float]


class ResultadoDriftColumna(BaseModel):
    """Resultado de las pruebas de drift sobre una ruta."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columna: str
    estadistico_ks: float = Field(ge=0.0, le=1.0)
    p_valor_ks: float = Field(ge=0.0, le=1.0)
    z_score: float
    media_referencia: float
    media_reciente: float
    criterios_activados: tuple[str, ...]

    @property
    def hay_drift(self) -> bool:
        """``True`` si al menos un criterio detectó drift."""
        return bool(self.criterios_activados)


class ReporteDrift(BaseModel):
    """Reporte de drift de todas las rutas para un par de periodos."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inicio_referencia: datetime
    fin_referencia: datetime
    inicio_reciente: datetime
    fin_reciente: datetime
    umbral_p_valor: float
    umbral_z_score: float
    resultados: tuple[ResultadoDriftColumna, ...]
    hay_drift: bool
    timestamp_evaluacion: datetime


class ResumenAuditorias(BaseModel):
    """Resumen agregado del registro ``auditoria_inferencias``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_inferencias: int = Field(ge=0)
    por_modo: dict[str, int]
    tasa_fallback: float = Field(ge=0.0, le=1.0)
    confianza_media_pct: float | None
    mae_medio: float | None
    alertas_drift: int = Field(ge=0)


# =============================================================================
# Agente Auditor
# =============================================================================


class AgenteAuditor:
    """Agente de trazabilidad, métricas y detección de *Data Drift*.

    Attributes:
        gestor_bd: Gestor SQLite donde se registran las inferencias.
        ruta_log_eventos: Log JSONL inmutable de eventos MLOps.
        umbral_p_valor: p-valor KS bajo el cual se declara drift.
        umbral_z_score: |Z| de la media diaria sobre el cual se declara drift.

    Example:
        >>> auditor = AgenteAuditor(gestor)
        >>> auditor.registrar_reportes_validacion(agente_datos.reportes_validacion)
        >>> reporte = auditor.evaluar_drift(serie_saneada)
        >>> auditor.registrar_inferencia(recomendacion)
    """

    def __init__(
        self,
        gestor_bd: GestorBaseDatos,
        ruta_log_eventos: Path | str = RUTA_LOG_EVENTOS,
        umbral_p_valor: float = UMBRAL_P_VALOR_KS,
        umbral_z_score: float = UMBRAL_Z_SCORE_DRIFT,
    ) -> None:
        """Inicializa el auditor.

        Args:
            gestor_bd: Gestor SQLite (crea también el log de inferencias).
            ruta_log_eventos: Ruta del log de eventos MLOps.
            umbral_p_valor: p-valor KS (``(0, 1)``).
            umbral_z_score: Umbral de |Z| (``> 0``).

        Raises:
            ValueError: Si los umbrales están fuera de dominio.
        """
        if not 0.0 < umbral_p_valor < 1.0:
            raise ValueError(f"'umbral_p_valor' debe estar en (0, 1): {umbral_p_valor}")
        if umbral_z_score <= 0.0:
            raise ValueError(f"'umbral_z_score' debe ser positivo: {umbral_z_score}")
        self.gestor_bd: GestorBaseDatos = gestor_bd
        self.ruta_log_eventos: Path = Path(ruta_log_eventos)
        self.umbral_p_valor: float = umbral_p_valor
        self.umbral_z_score: float = umbral_z_score

    # ------------------------------------------------------------------
    # Trazabilidad
    # ------------------------------------------------------------------

    def registrar_inferencia(self, recomendacion: RecomendacionTrafico) -> str:
        """Registra una recomendación en SQLite y en el log inmutable de auditoría.

        Args:
            recomendacion: Recomendación que ya cumple el contrato de salida.

        Returns:
            ``id_transaccion`` persistido.

        Raises:
            ErrorBaseDatos: Si falla SQLite o la escritura del JSONL (en ese
                caso la transacción se revierte).
        """
        return self.gestor_bd.registrar_auditoria(recomendacion.a_registro_auditoria())

    def registrar_evento(self, tipo_evento: str, datos: Mapping[str, Any]) -> dict[str, Any]:
        """Anexa un evento MLOps como línea JSON y lo fuerza a disco.

        Args:
            tipo_evento: Uno de ``TIPOS_EVENTO_VALIDOS``.
            datos: Contenido del evento, serializable a JSON y sin ``NaN``.

        Returns:
            El evento escrito (``tipo_evento``, ``timestamp`` y ``datos``).

        Raises:
            ValueError: Si ``tipo_evento`` no es válido.
            ErrorBaseDatos: Si el evento no es serializable o no se puede escribir.
        """
        if tipo_evento not in TIPOS_EVENTO_VALIDOS:
            raise ValueError(
                f"'tipo_evento'={tipo_evento!r} inválido; permitidos: {TIPOS_EVENTO_VALIDOS}"
            )
        evento = {
            "tipo_evento": tipo_evento,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datos": dict(datos),
        }
        try:
            linea = json.dumps(evento, ensure_ascii=False, allow_nan=False)
            self.ruta_log_eventos.parent.mkdir(parents=True, exist_ok=True)
            with self.ruta_log_eventos.open("a", encoding="utf-8") as archivo:
                archivo.write(linea + "\n")
                archivo.flush()
                os.fsync(archivo.fileno())
        except (TypeError, ValueError, OSError) as error:
            raise ErrorBaseDatos(
                f"No se pudo registrar el evento {tipo_evento} en "
                f"'{self.ruta_log_eventos}': {error}"
            ) from error
        return evento

    def registrar_reportes_validacion(
        self,
        reportes: MutableSequence[ReporteValidacion],
        vaciar: bool = True,
    ) -> int:
        """Registra los reportes de los guardrails de entrada.

        Pensado para consumir ``AgenteDatos.reportes_validacion``; al vaciar la
        lista se evita registrar dos veces el mismo reporte.

        Args:
            reportes: Reportes pendientes de auditoría.
            vaciar: Si es ``True``, vacía ``reportes`` tras registrarlos.

        Returns:
            Número de reportes registrados.

        Raises:
            ErrorBaseDatos: Si falla la escritura (los reportes no se vacían).
        """
        registrados = 0
        for reporte in reportes:
            self.registrar_evento(EVENTO_VALIDACION_ENTRADA, reporte.model_dump(mode="json"))
            registrados += 1
        if vaciar:
            del reportes[:]
        return registrados

    def registrar_error(self, error: Exception, contexto: str) -> dict[str, Any]:
        """Registra una anomalía (Regla 3.2.1: prohibido el fallo silencioso).

        Args:
            error: Excepción capturada; las del dominio aportan ``detalles``.
            contexto: Etapa del pipeline donde ocurrió (p. ej. ``'inferencia'``).

        Returns:
            El evento escrito.

        Raises:
            ErrorBaseDatos: Si no se puede escribir el evento.
        """
        if isinstance(error, ErrorSistemaTrafico):
            contenido = error.a_diccionario()
        else:
            contenido = {"codigo": type(error).__name__, "mensaje": str(error), "detalles": {}}
        contenido["contexto"] = contexto
        registro_log.error("[%s] %s: %s", contexto, contenido["codigo"], contenido["mensaje"])
        return self.registrar_evento(EVENTO_ERROR, _serializable(contenido))

    # ------------------------------------------------------------------
    # Métricas de rendimiento
    # ------------------------------------------------------------------

    def calcular_metricas(
        self,
        reales: npt.ArrayLike,
        predichos: npt.ArrayLike,
        etiqueta: str = "evaluacion",
    ) -> MetricasRendimiento:
        """Calcula ``MAE``, ``RMSE`` y sesgo, globales y por ruta.

        Args:
            reales: Tiempos observados en minutos; último eje = rutas
                (p. ej. ``(n, 168, 2)``).
            predichos: Tiempos predichos, con la misma forma que ``reales``.
            etiqueta: Nombre del conjunto evaluado (``'validacion'``, ``'prueba'``…).

        Returns:
            Métricas inmutables.

        Raises:
            ValueError: Si las formas difieren, están vacías, el último eje no
                corresponde a las rutas o hay valores no finitos.
        """
        y_real = np.asarray(reales, dtype=float)
        y_pred = np.asarray(predichos, dtype=float)
        n_rutas = len(COLUMNAS_OBJETIVO)
        if y_real.shape != y_pred.shape:
            raise ValueError(f"Formas distintas: reales {y_real.shape} vs predichos {y_pred.shape}")
        if y_real.size == 0 or y_real.shape[-1] != n_rutas:
            raise ValueError(
                f"Se esperaba un arreglo no vacío con último eje {n_rutas}: {y_real.shape}"
            )
        if not (np.isfinite(y_real).all() and np.isfinite(y_pred).all()):
            raise ValueError("Las métricas requieren valores finitos.")

        planos_real = y_real.reshape(-1, n_rutas)
        planos_pred = y_pred.reshape(-1, n_rutas)
        mae_ruta, rmse_ruta, sesgo_ruta = {}, {}, {}
        for j, columna in enumerate(COLUMNAS_OBJETIVO):
            mae_ruta[columna] = float(mean_absolute_error(planos_real[:, j], planos_pred[:, j]))
            rmse_ruta[columna] = float(
                root_mean_squared_error(planos_real[:, j], planos_pred[:, j])
            )
            sesgo_ruta[columna] = float(np.mean(planos_pred[:, j] - planos_real[:, j]))

        return MetricasRendimiento(
            etiqueta=etiqueta,
            n_observaciones=int(planos_real.shape[0]),
            mae=float(mean_absolute_error(planos_real.ravel(), planos_pred.ravel())),
            rmse=float(root_mean_squared_error(planos_real.ravel(), planos_pred.ravel())),
            sesgo=float(np.mean(planos_pred - planos_real)),
            mae_por_ruta=mae_ruta,
            rmse_por_ruta=rmse_ruta,
            sesgo_por_ruta=sesgo_ruta,
        )

    def registrar_metricas(self, metricas: MetricasRendimiento) -> dict[str, Any]:
        """Registra un conjunto de métricas en el log de eventos.

        Args:
            metricas: Métricas calculadas con ``calcular_metricas``.

        Returns:
            El evento escrito.

        Raises:
            ErrorBaseDatos: Si no se puede escribir el evento.
        """
        registro_log.info(
            "Métricas [%s]: MAE=%.2f min, RMSE=%.2f min, sesgo=%+.2f min.",
            metricas.etiqueta,
            metricas.mae,
            metricas.rmse,
            metricas.sesgo,
        )
        return self.registrar_evento(EVENTO_METRICAS, metricas.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # Data Drift
    # ------------------------------------------------------------------

    def detectar_data_drift(
        self,
        referencia: npt.ArrayLike,
        reciente: npt.ArrayLike,
        columna: str,
    ) -> ResultadoDriftColumna:
        """Compara dos periodos horarios de una ruta.

        * **KS:** ``ks_2samp`` sobre los valores horarios; drift si
          ``p < umbral_p_valor``.
        * **Z-score:** ``(μ_reciente − μ_ref) / (σ_ref / √n_días_recientes)``
          sobre medias **diarias** (eliminan la estacionalidad intradía);
          drift si ``|Z| > umbral_z_score``.

        Args:
            referencia: Valores horarios del periodo de referencia (días completos).
            reciente: Valores horarios del periodo reciente (días completos).
            columna: Nombre de la ruta evaluada.

        Returns:
            Estadísticos y criterios activados.

        Raises:
            ErrorValidacionEntrada: Si algún periodo no tiene al menos 2 días
                completos o contiene valores no finitos.
        """
        serie_ref = _validar_periodo(referencia, "referencia", columna)
        serie_rec = _validar_periodo(reciente, "reciente", columna)

        prueba_ks = ks_2samp(serie_ref, serie_rec)
        estadistico = float(prueba_ks.statistic)
        p_valor = float(prueba_ks.pvalue)

        diarias_ref = serie_ref.reshape(-1, HORAS_POR_DIA).mean(axis=1)
        diarias_rec = serie_rec.reshape(-1, HORAS_POR_DIA).mean(axis=1)
        error_estandar = max(float(diarias_ref.std(ddof=1)), _EPSILON_DESVIACION) / math.sqrt(
            diarias_rec.size
        )
        z_score = float((diarias_rec.mean() - diarias_ref.mean()) / error_estandar)

        criterios: list[str] = []
        if p_valor < self.umbral_p_valor:
            criterios.append(CRITERIO_KS)
        if abs(z_score) > self.umbral_z_score:
            criterios.append(CRITERIO_Z_SCORE)
        return ResultadoDriftColumna(
            columna=columna,
            estadistico_ks=estadistico,
            p_valor_ks=p_valor,
            z_score=z_score,
            media_referencia=float(serie_ref.mean()),
            media_reciente=float(serie_rec.mean()),
            criterios_activados=tuple(criterios),
        )

    def evaluar_drift(
        self,
        datos: pd.DataFrame,
        dias_referencia: int = DIAS_REFERENCIA_DRIFT,
        dias_recientes: int = DIAS_RECIENTES_DRIFT,
        registrar: bool = True,
        lanzar_alerta: bool = False,
    ) -> ReporteDrift:
        """Evalúa el drift de los últimos ``dias_recientes`` frente a los previos.

        Usa el final de la serie: ``[-(ref + rec), -rec)`` como referencia y
        ``[-rec, fin]`` como periodo reciente.

        Args:
            datos: Serie horaria saneada y continua (salida del ``AgenteDatos``).
            dias_referencia: Días del periodo de referencia (``>= 2``).
            dias_recientes: Días del periodo reciente (``>= 2``).
            registrar: Si es ``True``, escribe el reporte en el log de eventos.
            lanzar_alerta: Si es ``True`` y hay drift, lanza ``AlertaDataDrift``
                (después de registrar el reporte).

        Returns:
            Reporte de drift de ambas rutas.

        Raises:
            ErrorValidacionEntrada: Si la serie es demasiado corta, le faltan
                columnas o contiene valores no finitos.
            AlertaDataDrift: Si ``lanzar_alerta`` y se detecta drift.
            ErrorBaseDatos: Si falla el registro del evento.
        """
        if dias_referencia < 2 or dias_recientes < 2:
            raise ErrorValidacionEntrada("Los periodos de drift deben tener al menos 2 días.")
        faltantes = [c for c in (COLUMNA_FECHA_HORA, *COLUMNAS_OBJETIVO) if c not in datos.columns]
        if faltantes:
            raise ErrorValidacionEntrada(
                f"Faltan columnas para evaluar drift: {faltantes}",
                detalles={"columnas_faltantes": faltantes},
            )
        horas_ref = dias_referencia * HORAS_POR_DIA
        horas_rec = dias_recientes * HORAS_POR_DIA
        if len(datos) < horas_ref + horas_rec:
            raise ErrorValidacionEntrada(
                f"Historia insuficiente para drift: {len(datos)} registros < "
                f"{horas_ref + horas_rec} ({dias_referencia} + {dias_recientes} días).",
                detalles={"registros": len(datos), "requeridos": horas_ref + horas_rec},
            )

        tramo = datos.iloc[-(horas_ref + horas_rec) :]
        referencia, reciente = tramo.iloc[:horas_ref], tramo.iloc[horas_ref:]
        resultados = tuple(
            self.detectar_data_drift(
                referencia[columna].to_numpy(dtype=float),
                reciente[columna].to_numpy(dtype=float),
                columna,
            )
            for columna in COLUMNAS_OBJETIVO
        )
        fechas_ref = pd.DatetimeIndex(referencia[COLUMNA_FECHA_HORA])
        fechas_rec = pd.DatetimeIndex(reciente[COLUMNA_FECHA_HORA])
        reporte = ReporteDrift(
            inicio_referencia=fechas_ref[0].to_pydatetime(),
            fin_referencia=fechas_ref[-1].to_pydatetime(),
            inicio_reciente=fechas_rec[0].to_pydatetime(),
            fin_reciente=fechas_rec[-1].to_pydatetime(),
            umbral_p_valor=self.umbral_p_valor,
            umbral_z_score=self.umbral_z_score,
            resultados=resultados,
            hay_drift=any(resultado.hay_drift for resultado in resultados),
            timestamp_evaluacion=datetime.now(timezone.utc),
        )

        if reporte.hay_drift:
            registro_log.warning(
                "Data Drift detectado: %s",
                {r.columna: r.criterios_activados for r in resultados if r.hay_drift},
            )
        if registrar:
            self.registrar_evento(EVENTO_DRIFT, reporte.model_dump(mode="json"))
        if lanzar_alerta and reporte.hay_drift:
            raise AlertaDataDrift(
                "La distribución reciente difiere de la de referencia.",
                detalles=reporte.model_dump(mode="json"),
            )
        return reporte

    # ------------------------------------------------------------------
    # Resumen del registro de auditoría
    # ------------------------------------------------------------------

    def resumir_auditorias(self, limite: int | None = None) -> ResumenAuditorias:
        """Agrega el registro ``auditoria_inferencias``.

        Args:
            limite: Considera solo las ``limite`` inferencias más recientes.

        Returns:
            Conteos por modo, tasa de fallback, confianza y MAE medios y
            número de alertas de drift.

        Raises:
            ErrorBaseDatos: Si falla la consulta.
        """
        auditorias = self.gestor_bd.obtener_auditorias(limite=limite)
        total = len(auditorias)
        conteos = auditorias["modo_inferencia"].value_counts()
        por_modo = {modo: int(conteos.get(modo, 0)) for modo in MODOS_INFERENCIA_VALIDOS}
        mae = auditorias["mae_modelo"].dropna()
        return ResumenAuditorias(
            total_inferencias=total,
            por_modo=por_modo,
            tasa_fallback=por_modo[MODO_FALLBACK_HEURISTICO] / total if total else 0.0,
            confianza_media_pct=float(auditorias["nivel_confianza_pct"].mean()) if total else None,
            mae_medio=float(mae.mean()) if len(mae) else None,
            alertas_drift=int(auditorias["alerta_drift"].sum()),
        )


# =============================================================================
# Utilidades internas
# =============================================================================


def _validar_periodo(valores: npt.ArrayLike, nombre: str, columna: str) -> npt.NDArray[np.float64]:
    """Valida un periodo horario para las pruebas de drift.

    Args:
        valores: Valores horarios.
        nombre: ``'referencia'`` o ``'reciente'`` (para el mensaje).
        columna: Ruta evaluada (para el mensaje).

    Returns:
        Arreglo 1-D ``float64``.

    Raises:
        ErrorValidacionEntrada: Si no son al menos 2 días completos o hay
            valores no finitos.
    """
    arreglo = np.asarray(valores, dtype=float).ravel()
    if arreglo.size < 2 * HORAS_POR_DIA or arreglo.size % HORAS_POR_DIA:
        raise ErrorValidacionEntrada(
            f"El periodo {nombre} de '{columna}' debe contener días completos (>= 2): "
            f"{arreglo.size} horas.",
            detalles={"periodo": nombre, "columna": columna, "horas": int(arreglo.size)},
        )
    if not np.isfinite(arreglo).all():
        raise ErrorValidacionEntrada(
            f"El periodo {nombre} de '{columna}' contiene valores no finitos; sanee los datos.",
            detalles={"periodo": nombre, "columna": columna},
        )
    return arreglo


def _serializable(valor: Any) -> Any:
    """Convierte recursivamente un valor a tipos compatibles con JSON estricto.

    Args:
        valor: Estructura con posibles escalares de NumPy, fechas o ``NaN``.

    Returns:
        Estructura equivalente serializable (``NaN``/infinitos → ``None``,
        objetos desconocidos → ``str``).
    """
    if isinstance(valor, Mapping):
        return {str(clave): _serializable(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [_serializable(item) for item in valor]
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return float(valor) if math.isfinite(float(valor)) else None
    if valor is None or isinstance(valor, str):
        return valor
    return str(valor)
