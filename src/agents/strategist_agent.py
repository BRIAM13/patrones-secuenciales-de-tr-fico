"""Agente Estratega de Ruta (``AgenteEstrategaRuta``).

Responsabilidades (Fase 4 de ``CONTEXT.md``):

* **Optimización de ruta y hora de salida:** para un día y una hora de salida
  deseada, simula los desplazamientos ``[-30, -15, 0, +15]`` min sobre
  ``Ruta_A`` y ``Ruta_B`` (interpolando linealmente la predicción horaria) y
  elige la combinación de menor tiempo de traslado. El ahorro se mide contra
  salir a la hora deseada por la ruta habitual (``Ruta_A``).
* **Tendencia de 7 días:** resume, día a día, las horas punta de mañana y
  tarde de cada ruta.
* **Motor ``FALLBACK_HEURISTICO``:** predice cada hora futura con la mediana
  histórica de su mismo día de semana y hora, y estima su confianza a partir
  de la dispersión histórica.

El agente no conoce el origen de la predicción (red o heurística): ambas
llegan como un ``DataFrame`` horario ``fecha_hora`` + una columna por ruta,
que debe haber pasado antes por ``ValidadorSalida.acotar_prediccion``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import (
    COLUMNA_FECHA_HORA,
    COLUMNAS_OBJETIVO,
    DESPLAZAMIENTOS_HORARIO_MIN,
    FORMATO_HORA_SALIDA,
    RUTA_HABITUAL,
    RUTAS_DISPONIBLES,
    SEMANAS_HISTORIA_FALLBACK,
    VENTANA_MANANA_FIN_H,
    VENTANA_MANANA_INICIO_H,
    VENTANA_TARDE_FIN_H,
    VENTANA_TARDE_INICIO_H,
)
from src.database.db_manager import GestorBaseDatos
from src.guardrails.exceptions import ErrorValidacionEntrada
from src.guardrails.output_validator import estimar_nivel_confianza

registro_log: logging.Logger = logging.getLogger(__name__)

COLUMNA_POR_RUTA: Final[dict[str, str]] = dict(
    zip(RUTAS_DISPONIBLES, COLUMNAS_OBJETIVO, strict=True)
)
"""Correspondencia ``Ruta_A → tiempo_ruta_a_min``, ``Ruta_B → tiempo_ruta_b_min``."""

_DECIMALES_SALIDA: Final[int] = 2
_PATRON_HORA: Final[re.Pattern[str]] = re.compile(r"\d{1,2}:\d{2}")


# =============================================================================
# Modelos de resultados
# =============================================================================


class OpcionSalida(BaseModel):
    """Una combinación simulada de ruta y hora de salida."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ruta: str
    desplazamiento_min: int
    salida: datetime
    tiempo_estimado_min: float = Field(gt=0.0)

    @property
    def hora_salida(self) -> str:
        """Hora de salida en formato ``HH:MM``."""
        return self.salida.strftime(FORMATO_HORA_SALIDA)


class PlanViaje(BaseModel):
    """Recomendación de ruta y hora de salida para un viaje concreto."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    salida_deseada: datetime
    ruta_habitual: str
    mejor_opcion: OpcionSalida
    tiempo_ruta_habitual_min: float = Field(gt=0.0)
    """Tiempo de salir a la hora deseada por la ruta habitual."""
    ahorro_estimado_min: float = Field(ge=0.0)
    alternativas: tuple[OpcionSalida, ...]
    """Todas las opciones simuladas, de la mejor a la peor."""

    @property
    def ruta_recomendada(self) -> str:
        """Ruta de la mejor opción."""
        return self.mejor_opcion.ruta

    @property
    def hora_optima_salida(self) -> str:
        """Hora de salida óptima en formato ``HH:MM``."""
        return self.mejor_opcion.hora_salida

    @property
    def tiempo_estimado_min(self) -> float:
        """Tiempo de traslado de la mejor opción (minutos)."""
        return self.mejor_opcion.tiempo_estimado_min


class TendenciaDiaria(BaseModel):
    """Resumen de un día del horizonte en las franjas de hora punta."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fecha: date
    manana_por_ruta: dict[str, float]
    tarde_por_ruta: dict[str, float]
    mejor_ruta_manana: str
    mejor_ruta_tarde: str


@dataclass(frozen=True)
class PrediccionHeuristica:
    """Predicción del modo ``FALLBACK_HEURISTICO``.

    Attributes:
        datos: ``fecha_hora`` + una columna por ruta con la mediana histórica.
        desviacion: Desviación estándar histórica de cada hora y ruta.
        nivel_confianza: Confianza ``[0, 1]`` según la dispersión histórica.
        semanas_usadas: Semanas de historia consideradas.
    """

    datos: pd.DataFrame
    desviacion: pd.DataFrame
    nivel_confianza: float
    semanas_usadas: int


# =============================================================================
# Agente Estratega
# =============================================================================


class AgenteEstrategaRuta:
    """Optimiza ruta y hora de salida y provee el motor heurístico de respaldo.

    Attributes:
        gestor_bd: Fuente de la historia para el modo heurístico (opcional si
            la historia se pasa explícitamente).
        desplazamientos_min: Desplazamientos de salida simulados (minutos).

    Example:
        >>> estratega = AgenteEstrategaRuta(gestor)
        >>> plan = estratega.planificar_viaje(acotada.datos, date(2026, 1, 2), "07:30")
        >>> plan.ruta_recomendada, plan.hora_optima_salida
        ('Ruta_B', '07:00')
    """

    def __init__(
        self,
        gestor_bd: GestorBaseDatos | None = None,
        desplazamientos_min: Sequence[int] = DESPLAZAMIENTOS_HORARIO_MIN,
    ) -> None:
        """Inicializa el estratega.

        Args:
            gestor_bd: Gestor SQLite para cargar la historia del fallback.
            desplazamientos_min: Desplazamientos a simular; debe incluir ``0``.

        Raises:
            ValueError: Si los desplazamientos están vacíos, repetidos o no
                incluyen ``0``.
        """
        desplazamientos = tuple(int(d) for d in desplazamientos_min)
        if 0 not in desplazamientos or len(set(desplazamientos)) != len(desplazamientos):
            raise ValueError(f"Desplazamientos inválidos (únicos y con 0): {desplazamientos}")
        self.gestor_bd: GestorBaseDatos | None = gestor_bd
        self.desplazamientos_min: tuple[int, ...] = desplazamientos

    # ------------------------------------------------------------------
    # Optimización de ruta y hora de salida
    # ------------------------------------------------------------------

    def planificar_viaje(
        self,
        prediccion: pd.DataFrame,
        fecha_viaje: date,
        hora_salida_deseada: time | str,
        ruta_habitual: str = RUTA_HABITUAL,
    ) -> PlanViaje:
        """Elige la ruta y la hora de salida de menor tiempo de traslado.

        Para cada ruta y desplazamiento ``d`` se estima el tiempo de salir a
        ``hora_deseada + d`` interpolando linealmente entre las horas
        predichas. Desempate: menor ``|d|`` y luego el orden de
        ``RUTAS_DISPONIBLES``.

        Args:
            prediccion: Predicción horaria acotada (``fecha_hora`` + rutas).
            fecha_viaje: Día del viaje (dentro del horizonte predicho).
            hora_salida_deseada: Hora deseada (``time`` o ``'HH:MM'``).
            ruta_habitual: Ruta de referencia para el ahorro.

        Returns:
            Plan con la mejor opción, el ahorro y todas las alternativas.

        Raises:
            ErrorValidacionEntrada: Si la predicción es inválida, la hora no
                tiene formato ``HH:MM`` o alguna salida simulada cae fuera del
                horizonte predicho.
        """
        if ruta_habitual not in RUTAS_DISPONIBLES:
            raise ErrorValidacionEntrada(
                f"'ruta_habitual'={ruta_habitual!r} inválida; permitidas: {RUTAS_DISPONIBLES}"
            )
        tabla = _validar_prediccion(prediccion)
        salida_deseada = datetime.combine(fecha_viaje, _a_hora(hora_salida_deseada))

        candidatas = [salida_deseada + timedelta(minutes=d) for d in self.desplazamientos_min]
        inicio, fin = tabla.index[0], tabla.index[-1]
        fuera = [str(c) for c in candidatas if not inicio <= c <= fin]
        if fuera:
            raise ErrorValidacionEntrada(
                f"Salidas simuladas fuera del horizonte predicho [{inicio}, {fin}]: {fuera}",
                detalles={"inicio": str(inicio), "fin": str(fin), "fuera": fuera},
            )

        eje = _a_nanosegundos(tabla.index).astype(float)
        consultas = _a_nanosegundos(pd.DatetimeIndex(candidatas)).astype(float)
        opciones: list[OpcionSalida] = []
        tiempos_habitual: dict[int, float] = {}
        for ruta in RUTAS_DISPONIBLES:
            tiempos = np.interp(consultas, eje, tabla[COLUMNA_POR_RUTA[ruta]].to_numpy(dtype=float))
            for desplazamiento, salida, tiempo in zip(
                self.desplazamientos_min, candidatas, tiempos, strict=True
            ):
                opciones.append(
                    OpcionSalida(
                        ruta=ruta,
                        desplazamiento_min=desplazamiento,
                        salida=salida,
                        tiempo_estimado_min=round(float(tiempo), _DECIMALES_SALIDA),
                    )
                )
                if ruta == ruta_habitual:
                    tiempos_habitual[desplazamiento] = round(float(tiempo), _DECIMALES_SALIDA)

        orden_ruta = {ruta: i for i, ruta in enumerate(RUTAS_DISPONIBLES)}
        opciones.sort(
            key=lambda o: (o.tiempo_estimado_min, abs(o.desplazamiento_min), orden_ruta[o.ruta])
        )
        mejor = opciones[0]
        tiempo_habitual = tiempos_habitual[0]
        plan = PlanViaje(
            salida_deseada=salida_deseada,
            ruta_habitual=ruta_habitual,
            mejor_opcion=mejor,
            tiempo_ruta_habitual_min=tiempo_habitual,
            ahorro_estimado_min=round(max(tiempo_habitual - mejor.tiempo_estimado_min, 0.0), 2),
            alternativas=tuple(opciones),
        )
        registro_log.info(
            "Plan de viaje %s: %s a las %s (%.1f min, ahorro %.1f min frente a %s a las %s).",
            fecha_viaje,
            plan.ruta_recomendada,
            plan.hora_optima_salida,
            plan.tiempo_estimado_min,
            plan.ahorro_estimado_min,
            ruta_habitual,
            salida_deseada.strftime(FORMATO_HORA_SALIDA),
        )
        return plan

    def resumir_tendencia(self, prediccion: pd.DataFrame) -> tuple[TendenciaDiaria, ...]:
        """Resume cada día del horizonte en las franjas de hora punta.

        Las franjas son inclusivas: mañana ``[07:00, 09:00]`` y tarde
        ``[18:00, 20:00]``. Los días sin horas en alguna franja se omiten.

        Args:
            prediccion: Predicción horaria acotada (``fecha_hora`` + rutas).

        Returns:
            Una ``TendenciaDiaria`` por día con ambas franjas, en orden.

        Raises:
            ErrorValidacionEntrada: Si la predicción es inválida.
        """
        tabla = _validar_prediccion(prediccion)
        indice = pd.DatetimeIndex(tabla.index)
        horas = indice.hour + indice.minute / 60.0
        en_manana = (horas >= VENTANA_MANANA_INICIO_H) & (horas <= VENTANA_MANANA_FIN_H)
        en_tarde = (horas >= VENTANA_TARDE_INICIO_H) & (horas <= VENTANA_TARDE_FIN_H)
        dias: list[date] = sorted(set(indice.date))

        tendencia: list[TendenciaDiaria] = []
        for dia in dias:
            del_dia = indice.date == dia
            manana = tabla[del_dia & en_manana]
            tarde = tabla[del_dia & en_tarde]
            if manana.empty or tarde.empty:
                continue
            media_manana = {
                r: round(float(manana[c].mean()), 2) for r, c in COLUMNA_POR_RUTA.items()
            }
            media_tarde = {r: round(float(tarde[c].mean()), 2) for r, c in COLUMNA_POR_RUTA.items()}
            tendencia.append(
                TendenciaDiaria(
                    fecha=dia,
                    manana_por_ruta=media_manana,
                    tarde_por_ruta=media_tarde,
                    mejor_ruta_manana=min(media_manana, key=media_manana.__getitem__),
                    mejor_ruta_tarde=min(media_tarde, key=media_tarde.__getitem__),
                )
            )
        return tuple(tendencia)

    # ------------------------------------------------------------------
    # Motor FALLBACK_HEURISTICO
    # ------------------------------------------------------------------

    def generar_prediccion_heuristica(
        self,
        fechas_objetivo: pd.DatetimeIndex,
        historico: pd.DataFrame | None = None,
        semanas: int = SEMANAS_HISTORIA_FALLBACK,
    ) -> PrediccionHeuristica:
        """Predice con la mediana histórica por día de semana y hora.

        Solo se usan las ``semanas`` más recientes anteriores a la primera
        hora objetivo (nunca datos del propio horizonte).

        Args:
            fechas_objetivo: Horas a predecir.
            historico: Serie histórica (``fecha_hora``, ``dia_semana``,
                ``hora_decimal`` y rutas); si es ``None`` se lee de
                ``gestor_bd``.
            semanas: Semanas recientes a considerar (``>= 1``).

        Returns:
            Medianas, desviaciones y confianza heurística.

        Raises:
            ErrorValidacionEntrada: Si no hay historia suficiente para alguna
                combinación día de semana-hora, o faltan columnas.
            ValueError: Si ``semanas < 1`` o no hay fuente de historia.
        """
        if semanas < 1:
            raise ValueError(f"'semanas' debe ser >= 1, se recibió {semanas}")
        fechas = pd.DatetimeIndex(fechas_objetivo)
        if fechas.empty:
            raise ErrorValidacionEntrada("No hay horas objetivo que predecir.")
        if historico is None:
            if self.gestor_bd is None:
                raise ValueError(
                    "Se requiere 'historico' o un 'gestor_bd' para el modo heurístico."
                )
            historico = self.gestor_bd.obtener_historico()

        columnas = [COLUMNA_FECHA_HORA, "dia_semana", "hora_decimal", *COLUMNAS_OBJETIVO]
        faltantes = [c for c in columnas if c not in historico.columns]
        if faltantes:
            raise ErrorValidacionEntrada(
                f"Faltan columnas en la historia del fallback: {faltantes}",
                detalles={"columnas_faltantes": faltantes},
            )

        limite_superior = fechas[0]
        limite_inferior = limite_superior - pd.Timedelta(weeks=semanas)
        fechas_hist = pd.to_datetime(historico[COLUMNA_FECHA_HORA])
        reciente = historico.loc[(fechas_hist >= limite_inferior) & (fechas_hist < limite_superior)]

        agrupado = reciente.groupby(["dia_semana", "hora_decimal"])[list(COLUMNAS_OBJETIVO)]
        medianas = agrupado.median()
        desviaciones = agrupado.std(ddof=1).fillna(0.0)

        claves = pd.MultiIndex.from_arrays(
            [fechas.weekday, fechas.hour + fechas.minute / 60.0],
            names=["dia_semana", "hora_decimal"],
        )
        valores = medianas.reindex(claves)
        dispersion = desviaciones.reindex(claves)
        sin_historia = valores.isna().any(axis=1).to_numpy()
        if sin_historia.any():
            raise ErrorValidacionEntrada(
                f"Historia insuficiente para el fallback: {int(sin_historia.sum())} hora(s) "
                f"objetivo sin observaciones del mismo día de semana y hora en las últimas "
                f"{semanas} semanas.",
                detalles={"horas_sin_historia": [str(f) for f in fechas[sin_historia][:10]]},
            )

        datos = pd.DataFrame(valores.to_numpy(dtype=float), columns=list(COLUMNAS_OBJETIVO))
        datos.insert(0, COLUMNA_FECHA_HORA, fechas)
        desviacion = pd.DataFrame(dispersion.to_numpy(dtype=float), columns=list(COLUMNAS_OBJETIVO))
        desviacion.insert(0, COLUMNA_FECHA_HORA, fechas)
        confianza = estimar_nivel_confianza(
            datos[list(COLUMNAS_OBJETIVO)].to_numpy(),
            desviacion[list(COLUMNAS_OBJETIVO)].to_numpy(),
        )
        registro_log.info(
            "Predicción heurística: %d hora(s) con la mediana de %d semana(s); confianza %.1f %%.",
            len(fechas),
            semanas,
            confianza * 100.0,
        )
        return PrediccionHeuristica(
            datos=datos, desviacion=desviacion, nivel_confianza=confianza, semanas_usadas=semanas
        )


# =============================================================================
# Utilidades internas
# =============================================================================


def _validar_prediccion(prediccion: pd.DataFrame) -> pd.DataFrame:
    """Verifica una predicción horaria y la indexa por ``fecha_hora``.

    Args:
        prediccion: ``DataFrame`` con ``fecha_hora`` y una columna por ruta.

    Returns:
        Copia indexada por ``fecha_hora`` con solo las columnas de rutas.

    Raises:
        ErrorValidacionEntrada: Si faltan columnas, está vacía, no es una
            rejilla horaria continua o tiene tiempos no finitos o no positivos.
    """
    columnas = [COLUMNA_FECHA_HORA, *COLUMNAS_OBJETIVO]
    faltantes = [c for c in columnas if c not in prediccion.columns]
    if faltantes:
        raise ErrorValidacionEntrada(
            f"La predicción no tiene las columnas {faltantes}.",
            detalles={"columnas_faltantes": faltantes},
        )
    if len(prediccion) < 2:
        raise ErrorValidacionEntrada("La predicción debe contener al menos 2 horas.")
    tabla = prediccion.loc[:, columnas].set_index(pd.DatetimeIndex(prediccion[COLUMNA_FECHA_HORA]))
    tabla = tabla.drop(columns=[COLUMNA_FECHA_HORA])
    pasos = np.diff(_a_nanosegundos(tabla.index))
    if not (pasos == pd.Timedelta(hours=1).value).all():
        raise ErrorValidacionEntrada(
            "La predicción debe ser una rejilla horaria continua y ordenada."
        )
    valores = tabla.to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores <= 0).any():
        raise ErrorValidacionEntrada(
            "La predicción contiene tiempos no finitos o no positivos; acótela con "
            "ValidadorSalida.acotar_prediccion antes de planificar."
        )
    return tabla


def _a_hora(valor: time | str) -> time:
    """Normaliza la hora deseada.

    Args:
        valor: ``time`` o cadena ``'HH:MM'`` (se admite ``'H:MM'``).

    Returns:
        ``time`` sin segundos.

    Raises:
        ErrorValidacionEntrada: Si la cadena no tiene formato ``HH:MM``.
    """
    if isinstance(valor, time):
        return valor.replace(second=0, microsecond=0)
    try:
        if not _PATRON_HORA.fullmatch(valor.strip()):
            raise ValueError("formato no reconocido")
        return datetime.strptime(valor.strip(), FORMATO_HORA_SALIDA).time()
    except (ValueError, AttributeError) as error:
        raise ErrorValidacionEntrada(
            f"'hora_salida_deseada' debe tener formato HH:MM: {valor!r}"
        ) from error


def _a_nanosegundos(indice: pd.Index) -> npt.NDArray[np.int64]:
    """Convierte un índice temporal en nanosegundos desde la época (para ``np.interp``).

    Args:
        indice: Índice de marcas de tiempo.

    Returns:
        Arreglo ``int64`` de nanosegundos.
    """
    return pd.DatetimeIndex(indice).to_numpy(dtype="datetime64[ns]").astype(np.int64)
