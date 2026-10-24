"""Reglas duras de entrada (Input Guardrails) del sistema de predicción.

Implementa la Regla 3.2.2 de ``CONTEXT.md`` mediante el ``ValidadorEntrada``,
que aplica tres capas de control en orden:

1. **Esquema** (``validar_esquema``): contrato Pydantic v2 estricto por
   registro: tipos exactos, rangos de calendario, categorías válidas y
   coherencia entre ``fecha_hora`` y sus columnas derivadas.
2. **Integridad secuencial** (``validar_continuidad``): marcas de tiempo
   únicas, ordenadas, alineadas a la hora y sin huecos; las ventanas de
   inferencia deben medir exactamente ``14`` días (``336`` pasos horarios).
3. **Calidad física** (``sanear_tiempos``): un tiempo es *corrupto* si es
   ``NaN``/infinito o está fuera de ``[10, 240]`` min. Si la fracción
   corrupta de cualquier ventana de ``14`` días es ``<= 15 %`` se imputa por
   interpolación lineal temporal y se emite un ``ReporteValidacion`` para
   auditoría; si la supera, se aborta con ``ErrorValidacionEntrada``.
"""

from __future__ import annotations

import calendar
import itertools
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import (
    ACCION_IMPUTACION_LINEAL,
    ACCION_SIN_CAMBIOS,
    ACCIONES_VALIDACION,
    ALCANCE_VALIDACION_SERIE,
    ALCANCE_VALIDACION_VENTANA,
    ALCANCES_VALIDACION,
    COLUMNA_FECHA_HORA,
    COLUMNAS_OBJETIVO,
    DIA_QUINCENA,
    DIAS_POR_SEMANA,
    MAX_ERRORES_REPORTADOS,
    METODO_INTERPOLACION,
    ORIGENES_DATO_VALIDOS,
    PASOS_VENTANA_ENTRADA,
    PERIODO_HORA,
    TIEMPO_MAX_ENTRADA_MIN,
    TIEMPO_MIN_ENTRADA_MIN,
    UMBRAL_DATOS_CORRUPTOS,
)
from src.guardrails.exceptions import ErrorIntegridadSecuencial, ErrorValidacionEntrada

registro_log: logging.Logger = logging.getLogger(__name__)

_TOLERANCIA_HORA_DECIMAL: Final[float] = 1e-6
_PASO_HORARIO: Final[pd.Timedelta] = pd.Timedelta(hours=1)


# =============================================================================
# Contratos de datos (Pydantic v2)
# =============================================================================


class RegistroTraficoEntrada(BaseModel):
    """Contrato estricto de un registro horario de tráfico.

    Los tiempos admiten ``None``/``NaN`` a nivel de esquema: su tratamiento
    (imputación o rechazo) corresponde a la capa de calidad física.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    fecha_hora: datetime
    dia_semana: int = Field(ge=0, le=DIAS_POR_SEMANA - 1)
    hora_decimal: float = Field(ge=0.0, lt=PERIODO_HORA)
    es_fin_de_semana: bool
    es_quincena_o_fin_mes: bool
    clima_adverso: bool
    tiempo_ruta_a_min: float | None
    tiempo_ruta_b_min: float | None
    origen_dato: str

    @field_validator("fecha_hora")
    @classmethod
    def _validar_fecha_ingenua(cls, valor: datetime) -> datetime:
        """Exige fecha-hora sin zona horaria (las series se guardan en hora local)."""
        if valor.tzinfo is not None:
            raise ValueError("debe ser ingenua (hora local sin zona horaria)")
        return valor

    @field_validator("origen_dato")
    @classmethod
    def _validar_origen(cls, valor: str) -> str:
        """Restringe ``origen_dato`` a las categorías de ``config``."""
        if valor not in ORIGENES_DATO_VALIDOS:
            raise ValueError(f"debe ser uno de {ORIGENES_DATO_VALIDOS}")
        return valor

    @model_validator(mode="after")
    def _validar_coherencia_calendario(self) -> RegistroTraficoEntrada:
        """Verifica que las columnas de calendario se deriven de ``fecha_hora``."""
        fecha = self.fecha_hora
        if self.dia_semana != fecha.weekday():
            raise ValueError(
                f"dia_semana={self.dia_semana} no coincide con {fecha} "
                f"(esperado {fecha.weekday()}, lunes=0)"
            )
        hora_esperada = fecha.hour + fecha.minute / 60.0 + fecha.second / 3600.0
        if not math.isclose(self.hora_decimal, hora_esperada, abs_tol=_TOLERANCIA_HORA_DECIMAL):
            raise ValueError(f"hora_decimal={self.hora_decimal} no coincide con {fecha}")
        if self.es_fin_de_semana != (fecha.weekday() >= 5):
            raise ValueError(f"es_fin_de_semana={self.es_fin_de_semana} no coincide con {fecha}")
        es_quincena = fecha.day in (DIA_QUINCENA, calendar.monthrange(fecha.year, fecha.month)[1])
        if self.es_quincena_o_fin_mes != es_quincena:
            raise ValueError(
                f"es_quincena_o_fin_mes={self.es_quincena_o_fin_mes} no coincide con {fecha}"
            )
        return self


COLUMNAS_ESQUEMA_ENTRADA: Final[tuple[str, ...]] = tuple(RegistroTraficoEntrada.model_fields)
"""Columnas del contrato de entrada, en orden canónico."""

_ADAPTADOR_REGISTROS: Final[TypeAdapter[list[RegistroTraficoEntrada]]] = TypeAdapter(
    list[RegistroTraficoEntrada]
)


class ReporteValidacion(BaseModel):
    """Resultado auditable de aplicar los guardrails de entrada.

    Es inmutable y serializable con ``model_dump(mode="json")`` para que el
    ``AgenteAuditor`` lo registre en SQLite / JSONL.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alcance: str
    n_registros: int = Field(ge=1)
    fecha_inicio: datetime
    fecha_fin: datetime
    no_finitos_por_columna: dict[str, int]
    fuera_de_rango_por_columna: dict[str, int]
    imputados_por_columna: dict[str, int]
    fraccion_corrupta_max: float = Field(ge=0.0, le=1.0)
    """Peor fracción corrupta observada en una ventana de ``14`` días."""
    umbral_corrupcion: float = Field(ge=0.0, le=1.0)
    accion: str
    timestamp_validacion: datetime

    @field_validator("alcance")
    @classmethod
    def _validar_alcance(cls, valor: str) -> str:
        """Restringe ``alcance`` a ``SERIE`` o ``VENTANA``."""
        if valor not in ALCANCES_VALIDACION:
            raise ValueError(f"debe ser uno de {ALCANCES_VALIDACION}")
        return valor

    @field_validator("accion")
    @classmethod
    def _validar_accion(cls, valor: str) -> str:
        """Restringe ``accion`` a las acciones definidas en ``config``."""
        if valor not in ACCIONES_VALIDACION:
            raise ValueError(f"debe ser una de {ACCIONES_VALIDACION}")
        return valor

    @property
    def total_imputados(self) -> int:
        """Número total de celdas imputadas en todas las columnas objetivo."""
        return sum(self.imputados_por_columna.values())


@dataclass(frozen=True)
class ResultadoValidacion:
    """Datos saneados junto con su reporte de auditoría.

    Attributes:
        datos: ``DataFrame`` con las columnas de ``COLUMNAS_ESQUEMA_ENTRADA``,
            sin ``NaN`` ni valores fuera de rango en las columnas objetivo.
        reporte: Reporte inmutable de la validación realizada.
    """

    datos: pd.DataFrame
    reporte: ReporteValidacion


# =============================================================================
# Validador de entrada
# =============================================================================


class ValidadorEntrada:
    """Aplica las reglas duras de entrada sobre series y ventanas horarias.

    Attributes:
        tiempo_min: Tiempo físico mínimo aceptado (minutos).
        tiempo_max: Tiempo físico máximo aceptado (minutos).
        umbral_corrupcion: Fracción máxima imputable por ventana.
        pasos_ventana: Longitud de la ventana de entrada (pasos horarios).

    Example:
        >>> validador = ValidadorEntrada()
        >>> resultado = validador.validar_ventana(ultimos_14_dias)
        >>> resultado.reporte.accion
        'SIN_CAMBIOS'
    """

    def __init__(
        self,
        tiempo_min: float = TIEMPO_MIN_ENTRADA_MIN,
        tiempo_max: float = TIEMPO_MAX_ENTRADA_MIN,
        umbral_corrupcion: float = UMBRAL_DATOS_CORRUPTOS,
        pasos_ventana: int = PASOS_VENTANA_ENTRADA,
    ) -> None:
        """Configura los límites del validador (por defecto, los de ``config``).

        Args:
            tiempo_min: Tiempo físico mínimo aceptado (minutos).
            tiempo_max: Tiempo físico máximo aceptado (minutos).
            umbral_corrupcion: Fracción máxima de datos corruptos imputables
                en una ventana (``[0, 1)``).
            pasos_ventana: Longitud de ventana usada para medir la corrupción
                y exigida por ``validar_ventana``.

        Raises:
            ValueError: Si los parámetros son incoherentes.
        """
        if not 0.0 < tiempo_min < tiempo_max:
            raise ValueError(
                f"Se requiere 0 < tiempo_min < tiempo_max ({tiempo_min}, {tiempo_max})"
            )
        if not 0.0 <= umbral_corrupcion < 1.0:
            raise ValueError(f"'umbral_corrupcion' debe estar en [0, 1): {umbral_corrupcion}")
        if pasos_ventana <= 0:
            raise ValueError(f"'pasos_ventana' debe ser positivo: {pasos_ventana}")
        self.tiempo_min: float = tiempo_min
        self.tiempo_max: float = tiempo_max
        self.umbral_corrupcion: float = umbral_corrupcion
        self.pasos_ventana: int = pasos_ventana

    # ------------------------------------------------------------------
    # Flujos completos
    # ------------------------------------------------------------------

    def validar_serie(self, datos: pd.DataFrame) -> ResultadoValidacion:
        """Valida una serie histórica completa (entrenamiento o ingesta).

        Aplica esquema, continuidad (sin longitud fija) y la regla del 15 %
        sobre cada ventana deslizante de ``pasos_ventana`` registros.

        Args:
            datos: Serie horaria con las columnas de ``COLUMNAS_ESQUEMA_ENTRADA``.

        Returns:
            Datos saneados y su reporte con ``alcance='SERIE'``.

        Raises:
            ErrorValidacionEntrada: Si se viola el esquema o la corrupción de
                alguna ventana supera el umbral.
            ErrorIntegridadSecuencial: Si la serie no es continua.
        """
        datos_esquema = self.validar_esquema(datos)
        self.validar_continuidad(datos_esquema)
        return self.sanear_tiempos(datos_esquema, alcance=ALCANCE_VALIDACION_SERIE)

    def validar_ventana(self, datos: pd.DataFrame) -> ResultadoValidacion:
        """Valida una ventana de inferencia de exactamente ``14`` días.

        Args:
            datos: Ventana horaria con las columnas de ``COLUMNAS_ESQUEMA_ENTRADA``.

        Returns:
            Ventana saneada y su reporte con ``alcance='VENTANA'``.

        Raises:
            ErrorValidacionEntrada: Si se viola el esquema o la corrupción
                supera el umbral.
            ErrorIntegridadSecuencial: Si la ventana no es continua o no mide
                ``pasos_ventana`` registros.
        """
        datos_esquema = self.validar_esquema(datos)
        self.validar_continuidad(datos_esquema, longitud_esperada=self.pasos_ventana)
        return self.sanear_tiempos(datos_esquema, alcance=ALCANCE_VALIDACION_VENTANA)

    # ------------------------------------------------------------------
    # Capa 1: esquema
    # ------------------------------------------------------------------

    def validar_esquema(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Valida cada registro contra ``RegistroTraficoEntrada``.

        Las columnas adicionales (p. ej. ``id`` de SQLite) se descartan.

        Args:
            datos: Tabla a validar.

        Returns:
            Copia con únicamente ``COLUMNAS_ESQUEMA_ENTRADA``, en orden canónico,
            con ``fecha_hora`` como ``datetime64`` y tiempos como ``float``
            (``NaN`` para faltantes).

        Raises:
            ErrorValidacionEntrada: Si no es un ``DataFrame``, está vacío,
                faltan columnas o algún registro viola el contrato.
        """
        if not isinstance(datos, pd.DataFrame):
            raise ErrorValidacionEntrada(
                f"Se esperaba un pandas.DataFrame, se recibió {type(datos).__name__}"
            )
        if datos.empty:
            raise ErrorValidacionEntrada("El conjunto de datos de entrada está vacío.")

        faltantes = [
            columna for columna in COLUMNAS_ESQUEMA_ENTRADA if columna not in datos.columns
        ]
        if faltantes:
            raise ErrorValidacionEntrada(
                f"Faltan columnas obligatorias: {faltantes}",
                detalles={"columnas_faltantes": faltantes},
            )

        seleccion = datos.loc[:, list(COLUMNAS_ESQUEMA_ENTRADA)]
        try:
            _ADAPTADOR_REGISTROS.validate_python(seleccion.to_dict(orient="records"))
        except ValidationError as error:
            errores = [
                {
                    "fila": int(detalle["loc"][0]) if detalle["loc"] else None,
                    "campo": ".".join(str(parte) for parte in detalle["loc"][1:]) or None,
                    "mensaje": detalle["msg"],
                }
                for detalle in error.errors(include_url=False)[:MAX_ERRORES_REPORTADOS]
            ]
            raise ErrorValidacionEntrada(
                f"{error.error_count()} violación(es) del esquema de entrada; "
                f"primera: {errores[0]}",
                detalles={"total_errores": error.error_count(), "errores": errores},
            ) from error

        resultado = seleccion.reset_index(drop=True).copy()
        resultado[COLUMNA_FECHA_HORA] = pd.to_datetime(resultado[COLUMNA_FECHA_HORA])
        for columna in COLUMNAS_OBJETIVO:
            resultado[columna] = resultado[columna].astype(float)
        return resultado

    # ------------------------------------------------------------------
    # Capa 2: integridad secuencial
    # ------------------------------------------------------------------

    def validar_continuidad(
        self,
        datos: pd.DataFrame,
        longitud_esperada: int | None = None,
    ) -> None:
        """Verifica que la serie sea una rejilla horaria continua.

        Args:
            datos: Tabla con la columna ``fecha_hora``.
            longitud_esperada: Si se indica, número exacto de registros exigido.

        Raises:
            ErrorIntegridadSecuencial: Ante duplicados, desorden, registros no
                alineados a la hora, huecos o longitud incorrecta.
        """
        fechas = pd.DatetimeIndex(pd.to_datetime(datos[COLUMNA_FECHA_HORA]))

        if longitud_esperada is not None and len(fechas) != longitud_esperada:
            raise ErrorIntegridadSecuencial(
                f"La ventana tiene {len(fechas)} registros; se esperaban {longitud_esperada}.",
                detalles={"registros": len(fechas), "esperados": longitud_esperada},
            )

        duplicadas = fechas[fechas.duplicated()]
        if len(duplicadas):
            raise ErrorIntegridadSecuencial(
                f"{len(duplicadas)} marca(s) de tiempo duplicadas.",
                detalles={"duplicadas": _fechas_a_texto(duplicadas)},
            )

        if not fechas.is_monotonic_increasing:
            raise ErrorIntegridadSecuencial("Las marcas de tiempo no están en orden cronológico.")

        desalineadas = fechas[fechas != fechas.floor("h")]
        if len(desalineadas):
            raise ErrorIntegridadSecuencial(
                f"{len(desalineadas)} registro(s) no alineados a la hora en punto.",
                detalles={"desalineadas": _fechas_a_texto(desalineadas)},
            )

        diferencias = fechas[1:] - fechas[:-1]
        posiciones_hueco = np.flatnonzero(diferencias != _PASO_HORARIO)
        if posiciones_hueco.size:
            huecos = [
                {
                    "desde": str(fechas[posicion]),
                    "hasta": str(fechas[posicion + 1]),
                    "horas_faltantes": int(diferencias[posicion] / _PASO_HORARIO) - 1,
                }
                for posicion in posiciones_hueco[:MAX_ERRORES_REPORTADOS]
            ]
            raise ErrorIntegridadSecuencial(
                f"{posiciones_hueco.size} hueco(s) en la rejilla horaria; primero: {huecos[0]}",
                detalles={"total_huecos": int(posiciones_hueco.size), "huecos": huecos},
            )

    # ------------------------------------------------------------------
    # Capa 3: calidad física e imputación controlada
    # ------------------------------------------------------------------

    def mascaras_corruptos(self, datos: pd.DataFrame) -> dict[str, npt.NDArray[np.bool_]]:
        """Identifica los tiempos corruptos de cada columna objetivo.

        Un valor es corrupto si es ``NaN``/infinito o está fuera de
        ``[tiempo_min, tiempo_max]``.

        Args:
            datos: Tabla con las columnas de ``COLUMNAS_OBJETIVO``.

        Returns:
            Diccionario ``columna -> máscara booleana`` alineada con las filas.
        """
        mascaras: dict[str, npt.NDArray[np.bool_]] = {}
        for columna in COLUMNAS_OBJETIVO:
            valores = datos[columna].to_numpy(dtype=float)
            no_finito = ~np.isfinite(valores)
            with np.errstate(invalid="ignore"):
                fuera = (valores < self.tiempo_min) | (valores > self.tiempo_max)
            mascaras[columna] = no_finito | fuera
        return mascaras

    def sanear_tiempos(self, datos: pd.DataFrame, alcance: str) -> ResultadoValidacion:
        """Aplica la regla del 15 % e imputa por interpolación lineal temporal.

        Se asume que ``datos`` ya superó ``validar_esquema`` y
        ``validar_continuidad``.

        Args:
            datos: Serie o ventana horaria continua.
            alcance: ``'SERIE'`` o ``'VENTANA'`` (para el reporte).

        Returns:
            Datos sin valores corruptos en las columnas objetivo y su reporte.

        Raises:
            ErrorValidacionEntrada: Si alguna ventana de ``pasos_ventana``
                registros (o la serie completa, si es más corta) supera el
                umbral de corrupción en cualquiera de las columnas objetivo.
        """
        saneados = datos.copy()
        mascaras = self.mascaras_corruptos(saneados)

        no_finitos: dict[str, int] = {}
        fuera_de_rango: dict[str, int] = {}
        fracciones: dict[str, float] = {}
        for columna, mascara in mascaras.items():
            valores = saneados[columna].to_numpy(dtype=float)
            no_finitos[columna] = int((~np.isfinite(valores)).sum())
            fuera_de_rango[columna] = int(mascara.sum()) - no_finitos[columna]
            fracciones[columna] = self._fraccion_maxima_por_ventana(mascara)

        fraccion_max = max(fracciones.values())
        detalles_calidad: dict[str, Any] = {
            "alcance": alcance,
            "fracciones_corruptas_max": fracciones,
            "umbral": self.umbral_corrupcion,
            "no_finitos": no_finitos,
            "fuera_de_rango": fuera_de_rango,
        }
        if fraccion_max > self.umbral_corrupcion:
            peor_columna = max(fracciones, key=fracciones.__getitem__)
            raise ErrorValidacionEntrada(
                f"Datos corruptos en '{peor_columna}': {fraccion_max:.1%} de una ventana "
                f"supera el umbral de {self.umbral_corrupcion:.0%}; ejecución abortada.",
                detalles=detalles_calidad,
            )

        imputados: dict[str, int] = {}
        indice_temporal = pd.DatetimeIndex(saneados[COLUMNA_FECHA_HORA])
        for columna, mascara in mascaras.items():
            imputados[columna] = int(mascara.sum())
            if not imputados[columna]:
                continue
            serie = pd.Series(saneados[columna].to_numpy(dtype=float), index=indice_temporal)
            serie[mascara] = np.nan
            serie = serie.interpolate(method=METODO_INTERPOLACION, limit_direction="both")
            saneados[columna] = serie.to_numpy()

        accion = ACCION_IMPUTACION_LINEAL if any(imputados.values()) else ACCION_SIN_CAMBIOS
        reporte = ReporteValidacion(
            alcance=alcance,
            n_registros=len(saneados),
            fecha_inicio=indice_temporal[0].to_pydatetime(),
            fecha_fin=indice_temporal[-1].to_pydatetime(),
            no_finitos_por_columna=no_finitos,
            fuera_de_rango_por_columna=fuera_de_rango,
            imputados_por_columna=imputados,
            fraccion_corrupta_max=fraccion_max,
            umbral_corrupcion=self.umbral_corrupcion,
            accion=accion,
            timestamp_validacion=datetime.now(timezone.utc),
        )
        if accion == ACCION_IMPUTACION_LINEAL:
            registro_log.warning(
                "Guardrail de entrada (%s): %d valor(es) imputados por interpolación lineal %s "
                "(peor ventana: %.2f%% corrupto).",
                alcance,
                reporte.total_imputados,
                imputados,
                fraccion_max * 100.0,
            )
        return ResultadoValidacion(datos=saneados, reporte=reporte)

    def _fraccion_maxima_por_ventana(self, mascara: npt.NDArray[np.bool_]) -> float:
        """Calcula la peor fracción corrupta entre todas las ventanas deslizantes.

        Args:
            mascara: Máscara booleana de valores corruptos.

        Returns:
            Máximo de ``corruptos / pasos_ventana`` sobre todas las ventanas
            de ``pasos_ventana`` registros; si la serie es más corta, la
            fracción de la serie completa.
        """
        if mascara.size <= self.pasos_ventana:
            return float(mascara.mean())
        acumulado = np.concatenate(([0], np.cumsum(mascara, dtype=np.int64)))
        por_ventana = acumulado[self.pasos_ventana :] - acumulado[: -self.pasos_ventana]
        return float(por_ventana.max() / self.pasos_ventana)


def _fechas_a_texto(fechas: Iterable[Any]) -> list[str]:
    """Convierte hasta ``MAX_ERRORES_REPORTADOS`` fechas a texto para auditoría.

    Args:
        fechas: Marcas de tiempo (``DatetimeIndex``, lista, etc.).

    Returns:
        Lista de fechas en texto.
    """
    return [str(fecha) for fecha in itertools.islice(fechas, MAX_ERRORES_REPORTADOS)]
