"""Reglas duras de salida (Output Guardrails) y contrato JSON de la recomendación.

Implementa la Regla 3.2.3 de ``CONTEXT.md`` mediante el ``ValidadorSalida``:

1. **Clipping de seguridad** (``acotar_prediccion``): toda predicción se
   acota a ``[10, 180]`` min; si hubo recorte se activa
   ``ALERTA_EXTRAPOLACION``. Los valores no finitos no son acotables y se
   rechazan con ``ErrorExtrapolacionSalida``.
2. **Protocolo fallback** (``evaluar_confianza``): si la confianza es
   ``< 75 %`` o el MAE de validación es ``> 12`` min (o desconocido), el modo
   sugerido pasa de ``RED_LSTM`` a ``FALLBACK_HEURISTICO``.
3. **Contrato estricto** (``RecomendacionTrafico``): modelo Pydantic v2
   inmutable que garantiza la coherencia entre modo, motivos, confianza, MAE,
   alertas y ``estado_auditoria``; es imposible serializar una recomendación
   ``RED_LSTM`` que debía haber conmutado a fallback.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from src.config import (
    BANDERA_ALERTA_DRIFT,
    BANDERA_ALERTA_EXTRAPOLACION,
    BANDERA_FALLBACK,
    COLUMNA_FECHA_HORA,
    COLUMNAS_OBJETIVO,
    ESTADO_AUDITORIA_APROBADO,
    ESTADO_AUDITORIA_CON_ALERTAS,
    MAX_ERRORES_REPORTADOS,
    MODO_FALLBACK_HEURISTICO,
    MODO_RED_LSTM,
    MODOS_INFERENCIA_VALIDOS,
    MOTIVO_CONFIANZA_BAJA,
    MOTIVO_MAE_ALTO,
    MOTIVO_MAE_DESCONOCIDO,
    MOTIVOS_FALLBACK_VALIDOS,
    RUTAS_DISPONIBLES,
    TIEMPO_MAX_SALIDA_MIN,
    TIEMPO_MIN_SALIDA_MIN,
    TOLERANCIA_INCERTIDUMBRE_ABSOLUTA_MIN,
    TOLERANCIA_INCERTIDUMBRE_RELATIVA,
    UMBRAL_CONFIANZA_MIN,
    UMBRAL_MAE_MAX_MIN,
    Z_INTERVALO_CONFIANZA,
)
from src.guardrails.exceptions import ErrorContratoSalida, ErrorExtrapolacionSalida

registro_log: logging.Logger = logging.getLogger(__name__)

_PATRON_HORA_SALIDA: Final[str] = r"^([01]\d|2[0-3]):[0-5]\d$"


# =============================================================================
# Estimación de confianza
# =============================================================================


def estimar_nivel_confianza(
    media: npt.ArrayLike,
    desviacion: npt.ArrayLike,
    z: float = Z_INTERVALO_CONFIANZA,
    tolerancia_relativa: float = TOLERANCIA_INCERTIDUMBRE_RELATIVA,
    tolerancia_absoluta: float = TOLERANCIA_INCERTIDUMBRE_ABSOLUTA_MIN,
) -> float:
    """Convierte una dispersión predictiva en un nivel de confianza ``[0, 1]``.

    Una hora predicha es *confiable* si el semiancho de su intervalo al 95 %
    (``z·σ``) no supera ``max(tolerancia_absoluta, tolerancia_relativa·μ)``.
    La confianza es la fracción de horas (de ambas rutas) confiables, por lo
    que se interpreta como "porcentaje del horizonte con incertidumbre
    acotada". La usan tanto el *Monte Carlo Dropout* de la red como la
    dispersión histórica del modo heurístico.

    Args:
        media: Tiempos predichos (minutos).
        desviacion: Desviación estándar asociada, con la misma forma.
        z: Cuantil normal del intervalo.
        tolerancia_relativa: Semiancho máximo relativo al tiempo predicho.
        tolerancia_absoluta: Piso absoluto de la tolerancia (minutos).

    Returns:
        Fracción de horas confiables en ``[0, 1]``.

    Raises:
        ValueError: Si las formas difieren, están vacías o hay valores no
            finitos o desviaciones negativas.
    """
    mu = np.asarray(media, dtype=float)
    sigma = np.asarray(desviacion, dtype=float)
    if mu.shape != sigma.shape or mu.size == 0:
        raise ValueError(
            f"Formas incompatibles o vacías: media {mu.shape}, desviación {sigma.shape}"
        )
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all()) or (sigma < 0).any():
        raise ValueError("La media y la desviación deben ser finitas y la desviación >= 0.")
    tolerancia = np.maximum(tolerancia_absoluta, tolerancia_relativa * np.abs(mu))
    return float(np.mean(z * sigma <= tolerancia))


# =============================================================================
# Contrato JSON de salida
# =============================================================================


def determinar_estado_auditoria(
    modo_inferencia: str, alerta_extrapolacion: bool, alerta_drift: bool
) -> str:
    """Deriva el ``estado_auditoria`` de una recomendación.

    Args:
        modo_inferencia: ``'RED_LSTM'`` o ``'FALLBACK_HEURISTICO'``.
        alerta_extrapolacion: Si la predicción fue recortada.
        alerta_drift: Si se detectó *Data Drift*.

    Returns:
        ``'APROBADO'`` si la red respondió sin alertas; en cualquier otro caso
        (fallback o alguna alerta), ``'APROBADO_CON_ALERTAS'``.
    """
    con_alertas = alerta_extrapolacion or alerta_drift or modo_inferencia != MODO_RED_LSTM
    return ESTADO_AUDITORIA_CON_ALERTAS if con_alertas else ESTADO_AUDITORIA_APROBADO


class RecomendacionTrafico(BaseModel):
    """Contrato JSON estricto de la recomendación entregada al usuario.

    ``estado_auditoria`` y ``banderas`` son campos calculados: no se aceptan
    como entrada, de modo que nunca pueden contradecir a las alertas.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    id_transaccion: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp_consulta: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ruta_recomendada: str
    tiempo_estimado_min: float = Field(ge=TIEMPO_MIN_SALIDA_MIN, le=TIEMPO_MAX_SALIDA_MIN)
    ahorro_estimado_min: float = Field(ge=0.0, le=TIEMPO_MAX_SALIDA_MIN - TIEMPO_MIN_SALIDA_MIN)
    hora_optima_salida: str = Field(pattern=_PATRON_HORA_SALIDA)
    nivel_confianza_pct: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    modo_inferencia: str
    mae_modelo: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    alerta_extrapolacion: bool = False
    alerta_drift: bool = False
    motivos_fallback: tuple[str, ...] = ()

    @field_validator("timestamp_consulta")
    @classmethod
    def _validar_zona_horaria(cls, valor: datetime) -> datetime:
        """Exige una marca de tiempo con zona horaria (se recomienda UTC)."""
        if valor.tzinfo is None:
            raise ValueError("debe incluir zona horaria (p. ej. UTC)")
        return valor

    @field_validator("ruta_recomendada")
    @classmethod
    def _validar_ruta(cls, valor: str) -> str:
        """Restringe la ruta a ``Ruta_A`` o ``Ruta_B``."""
        if valor not in RUTAS_DISPONIBLES:
            raise ValueError(f"debe ser una de {RUTAS_DISPONIBLES}")
        return valor

    @field_validator("modo_inferencia")
    @classmethod
    def _validar_modo(cls, valor: str) -> str:
        """Restringe el modo a ``RED_LSTM`` o ``FALLBACK_HEURISTICO``."""
        if valor not in MODOS_INFERENCIA_VALIDOS:
            raise ValueError(f"debe ser uno de {MODOS_INFERENCIA_VALIDOS}")
        return valor

    @field_validator("motivos_fallback")
    @classmethod
    def _validar_motivos(cls, valor: tuple[str, ...]) -> tuple[str, ...]:
        """Restringe los motivos a los códigos de ``config`` y sin duplicados."""
        invalidos = [motivo for motivo in valor if motivo not in MOTIVOS_FALLBACK_VALIDOS]
        if invalidos:
            raise ValueError(
                f"motivos inválidos {invalidos}; permitidos: {MOTIVOS_FALLBACK_VALIDOS}"
            )
        if len(set(valor)) != len(valor):
            raise ValueError("contiene motivos duplicados")
        return valor

    @model_validator(mode="after")
    def _validar_protocolo_fallback(self) -> RecomendacionTrafico:
        """Impone el protocolo fallback sobre el contrato.

        * ``RED_LSTM`` exige confianza ``>= 75 %``, MAE conocido ``<= 12`` min
          y ningún motivo de fallback.
        * ``FALLBACK_HEURISTICO`` exige al menos un motivo.
        """
        if self.modo_inferencia == MODO_RED_LSTM:
            if self.motivos_fallback:
                raise ValueError("el modo RED_LSTM no admite motivos de fallback")
            if self.nivel_confianza_pct < UMBRAL_CONFIANZA_MIN * 100.0:
                raise ValueError(
                    f"confianza {self.nivel_confianza_pct:.1f} % < "
                    f"{UMBRAL_CONFIANZA_MIN:.0%}: debía conmutar a {MODO_FALLBACK_HEURISTICO}"
                )
            if self.mae_modelo is None or self.mae_modelo > UMBRAL_MAE_MAX_MIN:
                raise ValueError(
                    f"MAE {self.mae_modelo} fuera de [0, {UMBRAL_MAE_MAX_MIN}] min: "
                    f"debía conmutar a {MODO_FALLBACK_HEURISTICO}"
                )
        elif not self.motivos_fallback:
            raise ValueError("el modo FALLBACK_HEURISTICO requiere al menos un motivo")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estado_auditoria(self) -> str:
        """Estado de auditoría derivado del modo y de las alertas."""
        return determinar_estado_auditoria(
            self.modo_inferencia, self.alerta_extrapolacion, self.alerta_drift
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def banderas(self) -> tuple[str, ...]:
        """Banderas activas, en orden estable, para consumo de la CLI."""
        activas: list[str] = []
        if self.alerta_extrapolacion:
            activas.append(BANDERA_ALERTA_EXTRAPOLACION)
        if self.alerta_drift:
            activas.append(BANDERA_ALERTA_DRIFT)
        if self.modo_inferencia == MODO_FALLBACK_HEURISTICO:
            activas.append(BANDERA_FALLBACK)
        return tuple(activas)

    def a_registro_auditoria(self) -> dict[str, Any]:
        """Proyecta el contrato sobre las columnas de ``auditoria_inferencias``.

        Returns:
            Diccionario aceptado por ``GestorBaseDatos.registrar_auditoria``.
        """
        return {
            "id_transaccion": str(self.id_transaccion),
            "timestamp_consulta": self.timestamp_consulta,
            "ruta_recomendada": self.ruta_recomendada,
            "tiempo_estimado_min": self.tiempo_estimado_min,
            "ahorro_estimado_min": self.ahorro_estimado_min,
            "hora_optima_salida": self.hora_optima_salida,
            "nivel_confianza_pct": self.nivel_confianza_pct,
            "modo_inferencia": self.modo_inferencia,
            "estado_auditoria": self.estado_auditoria,
            "mae_modelo": self.mae_modelo,
            "alerta_drift": self.alerta_drift,
        }


# =============================================================================
# Resultados intermedios
# =============================================================================


@dataclass(frozen=True)
class PrediccionAcotada:
    """Predicción recortada al rango físico de salida.

    Attributes:
        datos: ``DataFrame`` con ``fecha_hora`` y una columna por ruta (minutos).
        alerta_extrapolacion: ``True`` si algún valor fue recortado.
        recortes_por_columna: Número de valores recortados por ruta.
        minimo_original: Valor mínimo antes del recorte, por ruta.
        maximo_original: Valor máximo antes del recorte, por ruta.
    """

    datos: pd.DataFrame
    alerta_extrapolacion: bool
    recortes_por_columna: dict[str, int]
    minimo_original: dict[str, float]
    maximo_original: dict[str, float]

    @property
    def total_recortes(self) -> int:
        """Número total de valores recortados."""
        return sum(self.recortes_por_columna.values())


@dataclass(frozen=True)
class EvaluacionConfianza:
    """Decisión del protocolo fallback.

    Attributes:
        modo_inferencia: Modo sugerido para el orquestador.
        motivos_fallback: Causas de la conmutación (vacío en ``RED_LSTM``).
        nivel_confianza: Confianza evaluada (fracción ``[0, 1]``) o ``None``.
        mae_validacion: MAE de validación evaluado (minutos) o ``None``.
    """

    modo_inferencia: str
    motivos_fallback: tuple[str, ...] = field(default_factory=tuple)
    nivel_confianza: float | None = None
    mae_validacion: float | None = None

    @property
    def requiere_fallback(self) -> bool:
        """``True`` si el orquestador debe usar el modo heurístico."""
        return self.modo_inferencia == MODO_FALLBACK_HEURISTICO


# =============================================================================
# Validador de salida
# =============================================================================


class ValidadorSalida:
    """Aplica las reglas duras de salida y construye el contrato JSON.

    Attributes:
        tiempo_min: Límite inferior del clipping (minutos).
        tiempo_max: Límite superior del clipping (minutos).
        umbral_confianza: Confianza mínima (fracción) para aceptar la red.
        umbral_mae: MAE de validación máximo (minutos) para aceptar la red.

    Example:
        >>> validador = ValidadorSalida()
        >>> acotada = validador.acotar_prediccion(prediccion_168x2, fechas_objetivo)
        >>> evaluacion = validador.evaluar_confianza(0.82, mae_validacion=8.4)
        >>> evaluacion.requiere_fallback
        False
    """

    def __init__(
        self,
        tiempo_min: float = TIEMPO_MIN_SALIDA_MIN,
        tiempo_max: float = TIEMPO_MAX_SALIDA_MIN,
        umbral_confianza: float = UMBRAL_CONFIANZA_MIN,
        umbral_mae: float = UMBRAL_MAE_MAX_MIN,
    ) -> None:
        """Configura los límites (por defecto, los de ``config``).

        Args:
            tiempo_min: Límite inferior del clipping (minutos).
            tiempo_max: Límite superior del clipping (minutos).
            umbral_confianza: Confianza mínima (``(0, 1]``).
            umbral_mae: MAE máximo aceptado (``> 0`` minutos).

        Raises:
            ValueError: Si los parámetros son incoherentes.
        """
        if not 0.0 < tiempo_min < tiempo_max:
            raise ValueError(
                f"Se requiere 0 < tiempo_min < tiempo_max ({tiempo_min}, {tiempo_max})"
            )
        if not 0.0 < umbral_confianza <= 1.0:
            raise ValueError(f"'umbral_confianza' debe estar en (0, 1]: {umbral_confianza}")
        if umbral_mae <= 0.0:
            raise ValueError(f"'umbral_mae' debe ser positivo: {umbral_mae}")
        self.tiempo_min: float = tiempo_min
        self.tiempo_max: float = tiempo_max
        self.umbral_confianza: float = umbral_confianza
        self.umbral_mae: float = umbral_mae

    # ------------------------------------------------------------------
    # Regla 1: clipping de seguridad
    # ------------------------------------------------------------------

    def acotar_prediccion(
        self,
        prediccion: npt.ArrayLike,
        fechas_objetivo: pd.DatetimeIndex | list[datetime],
    ) -> PrediccionAcotada:
        """Recorta la predicción al rango físico ``[tiempo_min, tiempo_max]``.

        Args:
            prediccion: Arreglo ``(pasos, 2)`` en minutos (``Ruta_A``, ``Ruta_B``),
                o ``(1, pasos, 2)`` tal como lo devuelve Keras para una ventana.
            fechas_objetivo: Una marca de tiempo por paso predicho.

        Returns:
            Predicción acotada con su bandera de extrapolación.

        Raises:
            ErrorExtrapolacionSalida: Si la forma es incompatible o hay valores
                no finitos (la red no produjo una predicción utilizable).
        """
        arreglo = np.asarray(prediccion, dtype=float)
        if arreglo.ndim == 3 and arreglo.shape[0] == 1:
            arreglo = arreglo[0]
        fechas = pd.DatetimeIndex(fechas_objetivo)
        forma_esperada = (len(fechas), len(COLUMNAS_OBJETIVO))
        if arreglo.shape != forma_esperada:
            raise ErrorExtrapolacionSalida(
                f"Forma de predicción {arreglo.shape} incompatible; se esperaba {forma_esperada}.",
                detalles={"forma": list(arreglo.shape), "esperada": list(forma_esperada)},
            )

        no_finitos = ~np.isfinite(arreglo)
        if no_finitos.any():
            filas = np.flatnonzero(no_finitos.any(axis=1))
            raise ErrorExtrapolacionSalida(
                f"La predicción contiene {int(no_finitos.sum())} valor(es) no finitos.",
                detalles={
                    "no_finitos": int(no_finitos.sum()),
                    "fechas": [str(fechas[i]) for i in filas[:MAX_ERRORES_REPORTADOS]],
                },
            )

        acotado = np.clip(arreglo, self.tiempo_min, self.tiempo_max)
        recortados = acotado != arreglo
        recortes = {
            columna: int(recortados[:, j].sum()) for j, columna in enumerate(COLUMNAS_OBJETIVO)
        }
        alerta = any(recortes.values())
        if alerta:
            registro_log.warning(
                "%s: %d valor(es) recortados a [%.0f, %.0f] min %s.",
                BANDERA_ALERTA_EXTRAPOLACION,
                sum(recortes.values()),
                self.tiempo_min,
                self.tiempo_max,
                recortes,
            )

        datos = pd.DataFrame(acotado, columns=list(COLUMNAS_OBJETIVO))
        datos.insert(0, COLUMNA_FECHA_HORA, fechas)
        return PrediccionAcotada(
            datos=datos,
            alerta_extrapolacion=alerta,
            recortes_por_columna=recortes,
            minimo_original={
                c: float(arreglo[:, j].min()) for j, c in enumerate(COLUMNAS_OBJETIVO)
            },
            maximo_original={
                c: float(arreglo[:, j].max()) for j, c in enumerate(COLUMNAS_OBJETIVO)
            },
        )

    # ------------------------------------------------------------------
    # Regla 2: protocolo fallback
    # ------------------------------------------------------------------

    def evaluar_confianza(
        self,
        nivel_confianza: float | None,
        mae_validacion: float | None,
    ) -> EvaluacionConfianza:
        """Decide si la inferencia neuronal es aceptable o requiere fallback.

        Args:
            nivel_confianza: Confianza estimada (fracción ``[0, 1]``, p. ej. por
                MC Dropout); ``None`` se trata como confianza insuficiente.
            mae_validacion: MAE de validación del modelo (minutos); ``None``
                significa que se desconoce y obliga al fallback.

        Returns:
            Modo sugerido y motivos de la decisión.

        Raises:
            ErrorContratoSalida: Si los valores están fuera de dominio
                (confianza fuera de ``[0, 1]``, MAE negativo o no finitos).
        """
        if nivel_confianza is not None and not (
            math.isfinite(nivel_confianza) and 0.0 <= nivel_confianza <= 1.0
        ):
            raise ErrorContratoSalida(
                f"'nivel_confianza' debe ser una fracción en [0, 1]: {nivel_confianza}"
            )
        if mae_validacion is not None and not (
            math.isfinite(mae_validacion) and mae_validacion >= 0.0
        ):
            raise ErrorContratoSalida(f"'mae_validacion' debe ser finito y >= 0: {mae_validacion}")

        motivos: list[str] = []
        if nivel_confianza is None or nivel_confianza < self.umbral_confianza:
            motivos.append(MOTIVO_CONFIANZA_BAJA)
        if mae_validacion is None:
            motivos.append(MOTIVO_MAE_DESCONOCIDO)
        elif mae_validacion > self.umbral_mae:
            motivos.append(MOTIVO_MAE_ALTO)

        modo = MODO_FALLBACK_HEURISTICO if motivos else MODO_RED_LSTM
        if motivos:
            registro_log.warning(
                "Protocolo fallback activado %s (confianza=%s, MAE=%s).",
                motivos,
                nivel_confianza,
                mae_validacion,
            )
        return EvaluacionConfianza(
            modo_inferencia=modo,
            motivos_fallback=tuple(motivos),
            nivel_confianza=nivel_confianza,
            mae_validacion=mae_validacion,
        )

    # ------------------------------------------------------------------
    # Regla 3: contrato JSON estricto
    # ------------------------------------------------------------------

    def construir_recomendacion(self, **campos: Any) -> RecomendacionTrafico:
        """Construye y valida la recomendación final.

        Args:
            **campos: Campos de ``RecomendacionTrafico`` (los ``float`` deben
                ser nativos de Python; convierta escalares de NumPy con ``float()``).

        Returns:
            Recomendación inmutable que cumple el contrato.

        Raises:
            ErrorContratoSalida: Si algún campo viola el contrato, incluido el
                intento de emitir ``RED_LSTM`` cuando correspondía fallback.
        """
        try:
            return RecomendacionTrafico(**campos)
        except ValidationError as error:
            raise _error_contrato(error) from error

    @staticmethod
    def serializar(recomendacion: RecomendacionTrafico) -> str:
        """Serializa la recomendación al JSON del contrato.

        Args:
            recomendacion: Recomendación validada.

        Returns:
            Cadena JSON (incluye ``estado_auditoria`` y ``banderas``).
        """
        return recomendacion.model_dump_json()

    @staticmethod
    def validar_json(texto: str) -> RecomendacionTrafico:
        """Reconstruye y valida una recomendación desde su JSON.

        Los campos calculados (``estado_auditoria``, ``banderas``) se verifican
        contra los recalculados para detectar contratos manipulados.

        Args:
            texto: JSON producido por ``serializar``.

        Returns:
            Recomendación validada.

        Raises:
            ErrorContratoSalida: Si el JSON es inválido o incoherente.
        """
        try:
            crudo = json.loads(texto)
        except json.JSONDecodeError as error:
            raise ErrorContratoSalida(f"JSON de contrato mal formado: {error}") from error
        if not isinstance(crudo, dict):
            raise ErrorContratoSalida("El JSON del contrato debe ser un objeto.")

        calculados = {campo: crudo.pop(campo, None) for campo in ("estado_auditoria", "banderas")}
        try:
            recomendacion = RecomendacionTrafico.model_validate_json(json.dumps(crudo))
        except ValidationError as error:
            raise _error_contrato(error) from error
        if calculados["estado_auditoria"] not in (None, recomendacion.estado_auditoria) or (
            calculados["banderas"] is not None
            and tuple(calculados["banderas"]) != recomendacion.banderas
        ):
            raise ErrorContratoSalida(
                "Los campos calculados del JSON no coinciden con las alertas del contrato.",
                detalles={"recibidos": calculados},
            )
        return recomendacion


def _error_contrato(error: ValidationError) -> ErrorContratoSalida:
    """Convierte un ``ValidationError`` de Pydantic en ``ErrorContratoSalida``.

    Args:
        error: Error de validación original.

    Returns:
        Excepción del dominio con los primeros errores como detalle.
    """
    errores = [
        {
            "campo": ".".join(str(parte) for parte in detalle["loc"]) or None,
            "mensaje": detalle["msg"],
        }
        for detalle in error.errors(include_url=False)[:MAX_ERRORES_REPORTADOS]
    ]
    return ErrorContratoSalida(
        f"{error.error_count()} violación(es) del contrato de salida; primera: {errores[0]}",
        detalles={"total_errores": error.error_count(), "errores": errores},
    )
