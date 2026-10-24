"""Gestor de persistencia SQLite del sistema de predicción de tráfico.

Implementa la Sección 3.3 de ``CONTEXT.md``:

* Base de datos ``data/trafico_cotidiano.db`` gestionada con ``sqlite3`` nativo.
* Todas las operaciones usan *context managers* que abren la conexión, ejecutan
  una transacción explícita (``BEGIN IMMEDIATE`` ... ``COMMIT``/``ROLLBACK``) y
  cierran la conexión de forma garantizada.
* Tablas ``historico_trafico`` y ``auditoria_inferencias`` con restricciones
  ``CHECK`` construidas a partir de las constantes de ``src.config``.
* Cada registro de ``auditoria_inferencias`` se escribe simultáneamente como
  una línea JSON en ``logs/auditoria_trafico.jsonl`` dentro de la misma
  transacción: si la escritura del log falla, la inserción se revierte.

Las validaciones de este módulo son **estructurales** (tipos, dominios de
categorías, coherencia calendario-fecha). Los límites físicos del dominio
(``[10, 240]`` min, umbral de corrupción del 15 %, etc.) pertenecen al
``ValidadorEntrada`` de ``src.guardrails``; por eso los tiempos no finitos se
persisten como ``NULL`` (dato faltante) para que el guardrail decida si
imputarlos o abortar.
"""

from __future__ import annotations

import calendar
import json
import logging
import math
import numbers
import os
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import numpy as np
import pandas as pd

from src.config import (
    COLUMNA_FECHA_HORA,
    DIA_QUINCENA,
    DIAS_POR_SEMANA,
    ESTADOS_AUDITORIA_VALIDOS,
    FORMATO_FECHA_HORA,
    FORMATO_HORA_SALIDA,
    MODOS_INFERENCIA_VALIDOS,
    ORIGENES_DATO_VALIDOS,
    PERIODO_HORA,
    RUTA_BASE_DATOS,
    RUTA_LOG_AUDITORIA,
    RUTAS_DISPONIBLES,
    TABLA_AUDITORIA,
    TABLA_HISTORICO,
    TAMANO_LOTE_INSERCION,
    TIEMPO_ESPERA_BLOQUEO_BD_S,
)
from src.guardrails.exceptions import ErrorBaseDatos

registro_log: logging.Logger = logging.getLogger(__name__)

_TOLERANCIA_HORA_DECIMAL: Final[float] = 1e-6


# =============================================================================
# Tipos de registros
# =============================================================================


class RegistroHistorico(TypedDict, total=False):
    """Registro horario de la tabla ``historico_trafico``.

    Claves obligatorias: ``fecha_hora``, ``tiempo_ruta_a_min``,
    ``tiempo_ruta_b_min`` y ``origen_dato``. Las demás se derivan de
    ``fecha_hora`` si se omiten (``clima_adverso`` toma ``False``).
    """

    fecha_hora: datetime | str
    dia_semana: int
    hora_decimal: float
    es_fin_de_semana: bool
    es_quincena_o_fin_mes: bool
    clima_adverso: bool
    tiempo_ruta_a_min: float | None
    tiempo_ruta_b_min: float | None
    origen_dato: str


class RegistroAuditoria(TypedDict, total=False):
    """Registro de la tabla ``auditoria_inferencias``.

    ``id_transaccion`` (UUID) y ``timestamp_consulta`` (UTC) se generan
    automáticamente si se omiten; ``mae_modelo`` admite ``None`` (p. ej. en
    modo ``FALLBACK_HEURISTICO``) y ``alerta_drift`` toma ``False`` por defecto.
    """

    id_transaccion: str
    timestamp_consulta: datetime | str
    ruta_recomendada: str
    tiempo_estimado_min: float
    ahorro_estimado_min: float
    hora_optima_salida: time | str
    nivel_confianza_pct: float
    modo_inferencia: str
    estado_auditoria: str
    mae_modelo: float | None
    alerta_drift: bool


# =============================================================================
# Esquema SQL
# =============================================================================


def _lista_sql(valores: Sequence[str]) -> str:
    """Convierte una tupla de constantes de ``config`` en una lista SQL literal.

    Args:
        valores: Valores categóricos permitidos (constantes internas, no
            entrada de usuario).

    Returns:
        Cadena del tipo ``'A', 'B'`` apta para una cláusula ``IN (...)``.
    """
    return ", ".join("'" + valor.replace("'", "''") + "'" for valor in valores)


_COLUMNAS_HISTORICO: Final[tuple[str, ...]] = (
    "fecha_hora",
    "dia_semana",
    "hora_decimal",
    "es_fin_de_semana",
    "es_quincena_o_fin_mes",
    "clima_adverso",
    "tiempo_ruta_a_min",
    "tiempo_ruta_b_min",
    "origen_dato",
)

_COLUMNAS_AUDITORIA: Final[tuple[str, ...]] = (
    "id_transaccion",
    "timestamp_consulta",
    "ruta_recomendada",
    "tiempo_estimado_min",
    "ahorro_estimado_min",
    "hora_optima_salida",
    "nivel_confianza_pct",
    "modo_inferencia",
    "estado_auditoria",
    "mae_modelo",
    "alerta_drift",
)

_COLUMNAS_BANDERA_HISTORICO: Final[tuple[str, ...]] = (
    "es_fin_de_semana",
    "es_quincena_o_fin_mes",
    "clima_adverso",
)

_ESQUEMA_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {TABLA_HISTORICO} (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_hora             TEXT    NOT NULL UNIQUE,
    dia_semana             INTEGER NOT NULL
                           CHECK (dia_semana BETWEEN 0 AND {DIAS_POR_SEMANA - 1}),
    hora_decimal           REAL    NOT NULL
                           CHECK (hora_decimal >= 0 AND hora_decimal < {PERIODO_HORA}),
    es_fin_de_semana       INTEGER NOT NULL CHECK (es_fin_de_semana IN (0, 1)),
    es_quincena_o_fin_mes  INTEGER NOT NULL CHECK (es_quincena_o_fin_mes IN (0, 1)),
    clima_adverso          INTEGER NOT NULL CHECK (clima_adverso IN (0, 1)),
    tiempo_ruta_a_min      REAL    CHECK (tiempo_ruta_a_min IS NULL OR tiempo_ruta_a_min > 0),
    tiempo_ruta_b_min      REAL    CHECK (tiempo_ruta_b_min IS NULL OR tiempo_ruta_b_min > 0),
    origen_dato            TEXT    NOT NULL
                           CHECK (origen_dato IN ({_lista_sql(ORIGENES_DATO_VALIDOS)}))
);

CREATE INDEX IF NOT EXISTS idx_historico_origen_fecha
    ON {TABLA_HISTORICO} (origen_dato, fecha_hora);

CREATE INDEX IF NOT EXISTS idx_historico_franja
    ON {TABLA_HISTORICO} (dia_semana, hora_decimal);

CREATE TABLE IF NOT EXISTS {TABLA_AUDITORIA} (
    id_transaccion         TEXT    PRIMARY KEY CHECK (length(id_transaccion) = 36),
    timestamp_consulta     TEXT    NOT NULL,
    ruta_recomendada       TEXT    NOT NULL
                           CHECK (ruta_recomendada IN ({_lista_sql(RUTAS_DISPONIBLES)})),
    tiempo_estimado_min    REAL    NOT NULL CHECK (tiempo_estimado_min > 0),
    ahorro_estimado_min    REAL    NOT NULL CHECK (ahorro_estimado_min >= 0),
    hora_optima_salida     TEXT    NOT NULL CHECK (length(hora_optima_salida) = 5),
    nivel_confianza_pct    REAL    NOT NULL CHECK (nivel_confianza_pct BETWEEN 0 AND 100),
    modo_inferencia        TEXT    NOT NULL
                           CHECK (modo_inferencia IN ({_lista_sql(MODOS_INFERENCIA_VALIDOS)})),
    estado_auditoria       TEXT    NOT NULL
                           CHECK (estado_auditoria IN ({_lista_sql(ESTADOS_AUDITORIA_VALIDOS)})),
    mae_modelo             REAL    CHECK (mae_modelo IS NULL OR mae_modelo >= 0),
    alerta_drift           INTEGER NOT NULL CHECK (alerta_drift IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_auditoria_timestamp
    ON {TABLA_AUDITORIA} (timestamp_consulta);
"""


# =============================================================================
# Funciones de normalización (validación estructural)
# =============================================================================


def _a_fecha_hora(valor: Any) -> datetime:
    """Convierte un valor a ``datetime`` ingenuo (sin zona horaria).

    Args:
        valor: ``datetime``, ``pandas.Timestamp``, ``numpy.datetime64`` o
            cadena ISO-8601.

    Returns:
        ``datetime`` sin información de zona horaria.

    Raises:
        ErrorBaseDatos: Si el valor no es interpretable como fecha-hora o
            incluye zona horaria (las series se guardan en hora local).
    """
    try:
        if isinstance(valor, datetime):
            fecha = valor
        elif isinstance(valor, str):
            fecha = datetime.fromisoformat(valor.strip())
        elif isinstance(valor, np.datetime64):
            fecha = pd.Timestamp(valor).to_pydatetime()
        else:
            raise TypeError(f"tipo no soportado: {type(valor).__name__}")
    except (TypeError, ValueError) as error:
        raise ErrorBaseDatos(f"'fecha_hora' inválida ({valor!r}): {error}") from error

    if isinstance(fecha, pd.Timestamp):
        fecha = fecha.to_pydatetime()
    if fecha.tzinfo is not None:
        raise ErrorBaseDatos(f"'fecha_hora' debe ser ingenua (hora local sin zona): {valor!r}")
    return fecha


def _a_bandera(valor: Any, nombre: str) -> int:
    """Convierte un indicador booleano a entero ``0``/``1`` para SQLite.

    Args:
        valor: ``bool``, ``numpy.bool_`` o número igual a ``0`` o ``1``.
        nombre: Nombre del campo (para el mensaje de error).

    Returns:
        ``0`` o ``1``.

    Raises:
        ErrorBaseDatos: Si el valor no representa un booleano.
    """
    if isinstance(valor, (bool, np.bool_, numbers.Real)) and valor in (0, 1):
        return 1 if valor else 0
    raise ErrorBaseDatos(f"'{nombre}' debe ser booleano (0/1), se recibió {valor!r}")


def _a_real(valor: Any, nombre: str) -> float:
    """Convierte un valor numérico real y finito a ``float``.

    Args:
        valor: Número real (``int``, ``float``, escalares de ``numpy``).
        nombre: Nombre del campo (para el mensaje de error).

    Returns:
        Valor como ``float``.

    Raises:
        ErrorBaseDatos: Si el valor no es numérico o no es finito.
    """
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, numbers.Real):
        raise ErrorBaseDatos(f"'{nombre}' debe ser numérico, se recibió {valor!r}")
    numero = float(valor)
    if not math.isfinite(numero):
        raise ErrorBaseDatos(f"'{nombre}' debe ser finito, se recibió {valor!r}")
    return numero


def _a_tiempo_opcional(valor: Any, nombre: str) -> tuple[float | None, bool]:
    """Normaliza un tiempo de traslado admitiendo datos faltantes.

    Los valores ``None``, ``NaN``, ``pd.NA`` e infinitos se convierten en
    ``None`` (``NULL`` en SQLite) para que el ``ValidadorEntrada`` aplique la
    regla de imputación del 15 %.

    Args:
        valor: Tiempo en minutos o marcador de dato faltante.
        nombre: Nombre del campo (para el mensaje de error).

    Returns:
        Tupla ``(tiempo, era_no_finito)``; ``era_no_finito`` es ``True`` si el
        valor original era ``NaN``/infinito (para contabilizarlo en el log).

    Raises:
        ErrorBaseDatos: Si el valor no es numérico o es menor o igual a cero.
    """
    if valor is None or valor is pd.NA:
        return None, False
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, numbers.Real):
        raise ErrorBaseDatos(f"'{nombre}' debe ser numérico, se recibió {valor!r}")
    numero = float(valor)
    if not math.isfinite(numero):
        return None, True
    if numero <= 0.0:
        raise ErrorBaseDatos(f"'{nombre}' debe ser positivo, se recibió {numero}")
    return numero, False


def _es_quincena_o_fin_mes(fecha: datetime) -> bool:
    """Indica si la fecha cae en día de quincena o en el último día del mes.

    Args:
        fecha: Fecha a evaluar.

    Returns:
        ``True`` si ``fecha.day`` es ``DIA_QUINCENA`` o el último día del mes.
    """
    ultimo_dia = calendar.monthrange(fecha.year, fecha.month)[1]
    return fecha.day in (DIA_QUINCENA, ultimo_dia)


def _hora_decimal(fecha: datetime) -> float:
    """Calcula la hora decimal ``h ∈ [0, 24)`` de una fecha-hora.

    Args:
        fecha: Fecha-hora de referencia.

    Returns:
        Hora expresada en fracción decimal (p. ej. ``07:30`` → ``7.5``).
    """
    return fecha.hour + fecha.minute / 60.0 + fecha.second / 3600.0


def _normalizar_registro_historico(registro: Mapping[str, Any]) -> tuple[tuple[Any, ...], bool]:
    """Valida y transforma un registro histórico en la tupla de inserción SQL.

    Args:
        registro: Diccionario con las claves de ``RegistroHistorico``.

    Returns:
        Tupla ``(fila, tuvo_no_finitos)`` con los valores en el orden de
        ``_COLUMNAS_HISTORICO`` y si algún tiempo era ``NaN``/infinito.

    Raises:
        ErrorBaseDatos: Si faltan claves obligatorias o algún campo es
            estructuralmente inválido o incoherente con ``fecha_hora``.
    """
    obligatorias = ("fecha_hora", "tiempo_ruta_a_min", "tiempo_ruta_b_min", "origen_dato")
    faltantes = [clave for clave in obligatorias if clave not in registro]
    if faltantes:
        raise ErrorBaseDatos(f"Registro histórico sin claves obligatorias: {faltantes}")

    fecha = _a_fecha_hora(registro["fecha_hora"])

    dia_semana_esperado = fecha.weekday()
    dia_semana_valor = registro.get("dia_semana", dia_semana_esperado)
    if (
        isinstance(dia_semana_valor, (bool, np.bool_))
        or not isinstance(dia_semana_valor, numbers.Integral)
        or int(dia_semana_valor) != dia_semana_esperado
    ):
        raise ErrorBaseDatos(
            f"'dia_semana'={dia_semana_valor!r} no coincide con {fecha} "
            f"(esperado {dia_semana_esperado}, lunes=0)"
        )

    hora_esperada = _hora_decimal(fecha)
    hora_valor = _a_real(registro.get("hora_decimal", hora_esperada), "hora_decimal")
    if not math.isclose(hora_valor, hora_esperada, abs_tol=_TOLERANCIA_HORA_DECIMAL):
        raise ErrorBaseDatos(
            f"'hora_decimal'={hora_valor} no coincide con {fecha} (esperado {hora_esperada})"
        )

    banderas_por_defecto: dict[str, bool] = {
        "es_fin_de_semana": dia_semana_esperado >= 5,
        "es_quincena_o_fin_mes": _es_quincena_o_fin_mes(fecha),
        "clima_adverso": False,
    }
    banderas = {
        nombre: _a_bandera(registro.get(nombre, banderas_por_defecto[nombre]), nombre)
        for nombre in _COLUMNAS_BANDERA_HISTORICO
    }

    tiempo_a, no_finito_a = _a_tiempo_opcional(registro["tiempo_ruta_a_min"], "tiempo_ruta_a_min")
    tiempo_b, no_finito_b = _a_tiempo_opcional(registro["tiempo_ruta_b_min"], "tiempo_ruta_b_min")

    origen = registro["origen_dato"]
    if origen not in ORIGENES_DATO_VALIDOS:
        raise ErrorBaseDatos(
            f"'origen_dato'={origen!r} inválido; valores permitidos: {ORIGENES_DATO_VALIDOS}"
        )

    fila = (
        fecha.strftime(FORMATO_FECHA_HORA),
        dia_semana_esperado,
        hora_esperada,
        banderas["es_fin_de_semana"],
        banderas["es_quincena_o_fin_mes"],
        banderas["clima_adverso"],
        tiempo_a,
        tiempo_b,
        origen,
    )
    return fila, no_finito_a or no_finito_b


def _normalizar_registro_auditoria(registro: Mapping[str, Any]) -> dict[str, Any]:
    """Valida y normaliza un registro de auditoría de inferencia.

    Args:
        registro: Diccionario con las claves de ``RegistroAuditoria``.

    Returns:
        Diccionario con todas las columnas de ``auditoria_inferencias`` en
        tipos serializables (``alerta_drift`` como ``bool``).

    Raises:
        ErrorBaseDatos: Si faltan claves obligatorias o algún campo es inválido.
    """
    obligatorias = (
        "ruta_recomendada",
        "tiempo_estimado_min",
        "ahorro_estimado_min",
        "hora_optima_salida",
        "nivel_confianza_pct",
        "modo_inferencia",
        "estado_auditoria",
    )
    faltantes = [clave for clave in obligatorias if clave not in registro]
    if faltantes:
        raise ErrorBaseDatos(f"Registro de auditoría sin claves obligatorias: {faltantes}")

    id_crudo = registro.get("id_transaccion")
    try:
        id_transaccion = str(uuid.uuid4()) if id_crudo is None else str(uuid.UUID(str(id_crudo)))
    except ValueError as error:
        raise ErrorBaseDatos(f"'id_transaccion' no es un UUID válido: {id_crudo!r}") from error

    marca_cruda = registro.get("timestamp_consulta")
    if marca_cruda is None:
        timestamp_consulta = datetime.now(timezone.utc).isoformat(timespec="seconds")
    elif isinstance(marca_cruda, datetime):
        timestamp_consulta = marca_cruda.isoformat(timespec="seconds")
    elif isinstance(marca_cruda, str):
        try:
            timestamp_consulta = datetime.fromisoformat(marca_cruda.strip()).isoformat(
                timespec="seconds"
            )
        except ValueError as error:
            raise ErrorBaseDatos(f"'timestamp_consulta' no es ISO-8601: {marca_cruda!r}") from error
    else:
        raise ErrorBaseDatos(f"'timestamp_consulta' de tipo no soportado: {marca_cruda!r}")

    ruta = registro["ruta_recomendada"]
    if ruta not in RUTAS_DISPONIBLES:
        raise ErrorBaseDatos(
            f"'ruta_recomendada'={ruta!r} inválida; permitidas: {RUTAS_DISPONIBLES}"
        )

    tiempo_estimado = _a_real(registro["tiempo_estimado_min"], "tiempo_estimado_min")
    if tiempo_estimado <= 0.0:
        raise ErrorBaseDatos(f"'tiempo_estimado_min' debe ser positivo: {tiempo_estimado}")

    ahorro = _a_real(registro["ahorro_estimado_min"], "ahorro_estimado_min")
    if ahorro < 0.0:
        raise ErrorBaseDatos(f"'ahorro_estimado_min' no puede ser negativo: {ahorro}")

    hora_cruda = registro["hora_optima_salida"]
    if isinstance(hora_cruda, (time, datetime)):
        hora_optima = hora_cruda.strftime(FORMATO_HORA_SALIDA)
    elif isinstance(hora_cruda, str):
        try:
            hora_optima = datetime.strptime(hora_cruda.strip(), FORMATO_HORA_SALIDA).strftime(
                FORMATO_HORA_SALIDA
            )
        except ValueError as error:
            raise ErrorBaseDatos(
                f"'hora_optima_salida' debe tener formato HH:MM: {hora_cruda!r}"
            ) from error
    else:
        raise ErrorBaseDatos(f"'hora_optima_salida' de tipo no soportado: {hora_cruda!r}")

    confianza = _a_real(registro["nivel_confianza_pct"], "nivel_confianza_pct")
    if not 0.0 <= confianza <= 100.0:
        raise ErrorBaseDatos(f"'nivel_confianza_pct' fuera de [0, 100]: {confianza}")

    modo = registro["modo_inferencia"]
    if modo not in MODOS_INFERENCIA_VALIDOS:
        raise ErrorBaseDatos(
            f"'modo_inferencia'={modo!r} inválido; permitidos: {MODOS_INFERENCIA_VALIDOS}"
        )

    estado = registro["estado_auditoria"]
    if estado not in ESTADOS_AUDITORIA_VALIDOS:
        raise ErrorBaseDatos(
            f"'estado_auditoria'={estado!r} inválido; permitidos: {ESTADOS_AUDITORIA_VALIDOS}"
        )

    mae_crudo = registro.get("mae_modelo")
    mae_modelo: float | None = None
    if mae_crudo is not None:
        mae_modelo = _a_real(mae_crudo, "mae_modelo")
        if mae_modelo < 0.0:
            raise ErrorBaseDatos(f"'mae_modelo' no puede ser negativo: {mae_modelo}")

    alerta_drift = bool(_a_bandera(registro.get("alerta_drift", False), "alerta_drift"))

    return {
        "id_transaccion": id_transaccion,
        "timestamp_consulta": timestamp_consulta,
        "ruta_recomendada": ruta,
        "tiempo_estimado_min": tiempo_estimado,
        "ahorro_estimado_min": ahorro,
        "hora_optima_salida": hora_optima,
        "nivel_confianza_pct": confianza,
        "modo_inferencia": modo,
        "estado_auditoria": estado,
        "mae_modelo": mae_modelo,
        "alerta_drift": alerta_drift,
    }


# =============================================================================
# Gestor de base de datos
# =============================================================================


class GestorBaseDatos:
    """Gestor transaccional de la base de datos SQLite del sistema.

    Cada método público abre su propia conexión mediante un *context manager*,
    ejecuta sus sentencias en una única transacción atómica y cierra la
    conexión al terminar, de modo que la instancia es segura para reutilizarse
    durante todo el pipeline.

    Attributes:
        ruta_bd: Ruta del archivo SQLite.
        ruta_log_jsonl: Ruta del log inmutable de auditoría (``.jsonl``).

    Example:
        >>> gestor = GestorBaseDatos()
        >>> gestor.insertar_lote_historico(registros)
        >>> df = gestor.obtener_ultimos_dias(14)
    """

    def __init__(
        self,
        ruta_bd: Path | str = RUTA_BASE_DATOS,
        ruta_log_jsonl: Path | str = RUTA_LOG_AUDITORIA,
        inicializar: bool = True,
    ) -> None:
        """Configura el gestor y, opcionalmente, crea el esquema.

        Args:
            ruta_bd: Ruta del archivo SQLite (por defecto
                ``data/trafico_cotidiano.db``).
            ruta_log_jsonl: Ruta del log JSONL de auditoría (por defecto
                ``logs/auditoria_trafico.jsonl``).
            inicializar: Si es ``True``, crea directorios, tablas e índices.

        Raises:
            ErrorBaseDatos: Si no es posible crear los directorios o el esquema.
        """
        self.ruta_bd: Path = Path(ruta_bd)
        self.ruta_log_jsonl: Path = Path(ruta_log_jsonl)
        if inicializar:
            self.inicializar_esquema()

    def __repr__(self) -> str:
        """Representación legible del gestor."""
        return f"GestorBaseDatos(ruta_bd='{self.ruta_bd}', ruta_log_jsonl='{self.ruta_log_jsonl}')"

    # ------------------------------------------------------------------
    # Conexión y transacciones
    # ------------------------------------------------------------------

    def _abrir_conexion(self) -> sqlite3.Connection:
        """Abre una conexión SQLite en modo de transacciones explícitas.

        Returns:
            Conexión con ``isolation_level=None`` (las transacciones se
            controlan con ``BEGIN``/``COMMIT``/``ROLLBACK``), claves foráneas
            activas y filas accesibles por nombre.

        Raises:
            ErrorBaseDatos: Si SQLite no puede abrir el archivo.
        """
        try:
            conexion = sqlite3.connect(
                self.ruta_bd,
                timeout=TIEMPO_ESPERA_BLOQUEO_BD_S,
                isolation_level=None,
            )
        except sqlite3.Error as error:
            raise ErrorBaseDatos(
                f"No se pudo abrir la base de datos '{self.ruta_bd}': {error}"
            ) from error
        conexion.row_factory = sqlite3.Row
        conexion.execute("PRAGMA foreign_keys = ON")
        return conexion

    @contextmanager
    def conexion(self) -> Iterator[sqlite3.Connection]:
        """Context manager de solo lectura que garantiza el cierre de la conexión.

        Yields:
            Conexión SQLite abierta.

        Raises:
            ErrorBaseDatos: Si ocurre un ``sqlite3.Error`` (o el
                ``DatabaseError`` con que ``pandas`` lo envuelve) durante su uso.
        """
        conexion = self._abrir_conexion()
        try:
            yield conexion
        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            raise ErrorBaseDatos(f"Error de lectura en SQLite: {error}") from error
        finally:
            conexion.close()

    @contextmanager
    def transaccion(self) -> Iterator[sqlite3.Connection]:
        """Context manager de escritura con transacción atómica.

        Ejecuta ``BEGIN IMMEDIATE`` al entrar, ``COMMIT`` si el bloque termina
        sin errores y ``ROLLBACK`` ante cualquier excepción; la conexión se
        cierra siempre.

        Yields:
            Conexión SQLite con una transacción abierta.

        Raises:
            ErrorBaseDatos: Si ocurre un ``sqlite3.Error`` u ``OSError``
                (se revierte la transacción antes de propagar).
        """
        conexion = self._abrir_conexion()
        try:
            conexion.execute("BEGIN IMMEDIATE")
            yield conexion
            conexion.execute("COMMIT")
        except (sqlite3.Error, OSError) as error:
            self._revertir(conexion)
            raise ErrorBaseDatos(f"Transacción revertida: {error}") from error
        except BaseException:
            self._revertir(conexion)
            raise
        finally:
            conexion.close()

    @staticmethod
    def _revertir(conexion: sqlite3.Connection) -> None:
        """Ejecuta ``ROLLBACK`` si la conexión mantiene una transacción abierta.

        Args:
            conexion: Conexión cuya transacción se debe revertir.
        """
        if conexion.in_transaction:
            conexion.execute("ROLLBACK")

    def inicializar_esquema(self) -> None:
        """Crea directorios, tablas e índices si no existen (idempotente).

        Raises:
            ErrorBaseDatos: Si falla la creación de directorios o del esquema.
        """
        try:
            self.ruta_bd.parent.mkdir(parents=True, exist_ok=True)
            self.ruta_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ErrorBaseDatos(
                f"No se pudieron crear los directorios de persistencia: {error}"
            ) from error

        conexion = self._abrir_conexion()
        try:
            conexion.executescript(f"BEGIN;\n{_ESQUEMA_SQL}\nCOMMIT;")
        except sqlite3.Error as error:
            self._revertir(conexion)
            raise ErrorBaseDatos(
                f"No se pudo crear el esquema de la base de datos: {error}"
            ) from error
        finally:
            conexion.close()
        registro_log.info("Esquema SQLite verificado en '%s'.", self.ruta_bd)

    # ------------------------------------------------------------------
    # historico_trafico: escritura
    # ------------------------------------------------------------------

    def insertar_lote_historico(
        self,
        registros: Iterable[Mapping[str, Any]],
        reemplazar_existentes: bool = False,
    ) -> int:
        """Inserta registros horarios por lotes en una única transacción atómica.

        Si cualquier registro es inválido, no se persiste ninguno.

        Args:
            registros: Registros con las claves de ``RegistroHistorico``.
            reemplazar_existentes: Si es ``True``, un registro con una
                ``fecha_hora`` ya existente actualiza la fila (UPSERT); si es
                ``False``, se conserva la fila existente y se omite el nuevo.

        Returns:
            Número de filas insertadas o actualizadas.

        Raises:
            ErrorBaseDatos: Si algún registro es inválido o falla SQLite.
        """
        filas: list[tuple[Any, ...]] = []
        total_no_finitos = 0
        for indice, registro in enumerate(registros):
            try:
                fila, tuvo_no_finitos = _normalizar_registro_historico(registro)
            except ErrorBaseDatos as error:
                raise ErrorBaseDatos(f"Registro histórico #{indice} inválido: {error}") from error
            filas.append(fila)
            total_no_finitos += int(tuvo_no_finitos)

        if not filas:
            return 0
        if total_no_finitos:
            registro_log.warning(
                "%d registro(s) con tiempos NaN/infinitos se persistieron como NULL "
                "para su tratamiento por el ValidadorEntrada.",
                total_no_finitos,
            )

        columnas = ", ".join(_COLUMNAS_HISTORICO)
        marcadores = ", ".join("?" for _ in _COLUMNAS_HISTORICO)
        if reemplazar_existentes:
            actualizaciones = ", ".join(
                f"{columna} = excluded.{columna}"
                for columna in _COLUMNAS_HISTORICO
                if columna != COLUMNA_FECHA_HORA
            )
            sentencia = (
                f"INSERT INTO {TABLA_HISTORICO} ({columnas}) VALUES ({marcadores}) "
                f"ON CONFLICT({COLUMNA_FECHA_HORA}) DO UPDATE SET {actualizaciones}"
            )
        else:
            sentencia = (
                f"INSERT INTO {TABLA_HISTORICO} ({columnas}) VALUES ({marcadores}) "
                f"ON CONFLICT({COLUMNA_FECHA_HORA}) DO NOTHING"
            )

        with self.transaccion() as conexion:
            cambios_iniciales = conexion.total_changes
            for inicio in range(0, len(filas), TAMANO_LOTE_INSERCION):
                conexion.executemany(sentencia, filas[inicio : inicio + TAMANO_LOTE_INSERCION])
            filas_afectadas = conexion.total_changes - cambios_iniciales

        registro_log.info(
            "historico_trafico: %d/%d fila(s) persistidas (reemplazar=%s).",
            filas_afectadas,
            len(filas),
            reemplazar_existentes,
        )
        return filas_afectadas

    def insertar_dataframe_historico(
        self,
        datos: pd.DataFrame,
        reemplazar_existentes: bool = False,
    ) -> int:
        """Inserta un ``DataFrame`` de registros horarios por lotes.

        ``fecha_hora`` puede ser una columna o el índice del ``DataFrame``.

        Args:
            datos: Tabla con las columnas de ``RegistroHistorico``.
            reemplazar_existentes: Ver ``insertar_lote_historico``.

        Returns:
            Número de filas insertadas o actualizadas.

        Raises:
            ErrorBaseDatos: Si falta ``fecha_hora`` o algún registro es inválido.
        """
        if COLUMNA_FECHA_HORA not in datos.columns:
            if datos.index.name != COLUMNA_FECHA_HORA:
                raise ErrorBaseDatos(
                    f"El DataFrame debe contener '{COLUMNA_FECHA_HORA}' como columna o índice."
                )
            datos = datos.reset_index()
        columnas_presentes = [
            columna for columna in _COLUMNAS_HISTORICO if columna in datos.columns
        ]
        registros = cast(list[dict[str, Any]], datos[columnas_presentes].to_dict(orient="records"))
        return self.insertar_lote_historico(registros, reemplazar_existentes=reemplazar_existentes)

    def eliminar_historico(self, origen_dato: str | None = None) -> int:
        """Elimina registros históricos, opcionalmente filtrando por origen.

        Args:
            origen_dato: ``'SINTETICO'`` o ``'REAL'``; ``None`` elimina todo.

        Returns:
            Número de filas eliminadas.

        Raises:
            ErrorBaseDatos: Si ``origen_dato`` es inválido o falla SQLite.
        """
        clausula, parametros = self._filtro_origen(origen_dato)
        with self.transaccion() as conexion:
            cursor = conexion.execute(f"DELETE FROM {TABLA_HISTORICO}{clausula}", parametros)
            eliminadas = cursor.rowcount
        registro_log.info(
            "historico_trafico: %d fila(s) eliminadas (origen=%s).", eliminadas, origen_dato
        )
        return eliminadas

    # ------------------------------------------------------------------
    # historico_trafico: consultas
    # ------------------------------------------------------------------

    def obtener_historico(
        self,
        fecha_inicio: datetime | str | None = None,
        fecha_fin: datetime | str | None = None,
        origen_dato: str | None = None,
    ) -> pd.DataFrame:
        """Consulta la serie histórica ordenada cronológicamente.

        Args:
            fecha_inicio: Límite inferior inclusivo de ``fecha_hora``.
            fecha_fin: Límite superior inclusivo de ``fecha_hora``.
            origen_dato: Filtra por ``'SINTETICO'`` o ``'REAL'``.

        Returns:
            ``DataFrame`` con ``fecha_hora`` como ``datetime64``, banderas como
            ``bool`` y tiempos como ``float`` (``NaN`` para datos faltantes).

        Raises:
            ErrorBaseDatos: Si algún filtro es inválido o falla SQLite.
        """
        condiciones: list[str] = []
        parametros: list[Any] = []
        if fecha_inicio is not None:
            condiciones.append(f"{COLUMNA_FECHA_HORA} >= ?")
            parametros.append(_a_fecha_hora(fecha_inicio).strftime(FORMATO_FECHA_HORA))
        if fecha_fin is not None:
            condiciones.append(f"{COLUMNA_FECHA_HORA} <= ?")
            parametros.append(_a_fecha_hora(fecha_fin).strftime(FORMATO_FECHA_HORA))
        if origen_dato is not None:
            self._validar_origen(origen_dato)
            condiciones.append("origen_dato = ?")
            parametros.append(origen_dato)

        clausula = f" WHERE {' AND '.join(condiciones)}" if condiciones else ""
        return self._consultar_historico(clausula, parametros)

    def obtener_ultimos_dias(self, n_dias: int, origen_dato: str | None = None) -> pd.DataFrame:
        """Obtiene los últimos ``n_dias`` de historia respecto al registro más reciente.

        Útil para construir la ventana de entrada de ``14`` días de la LSTM.

        Args:
            n_dias: Número de días (``> 0``) a recuperar.
            origen_dato: Filtra por ``'SINTETICO'`` o ``'REAL'``.

        Returns:
            ``DataFrame`` cronológico con ``fecha_hora`` en
            ``(max_fecha - n_dias, max_fecha]``; vacío si no hay datos.

        Raises:
            ErrorBaseDatos: Si ``n_dias`` no es positivo o falla SQLite.
        """
        if isinstance(n_dias, bool) or not isinstance(n_dias, int) or n_dias <= 0:
            raise ErrorBaseDatos(f"'n_dias' debe ser un entero positivo, se recibió {n_dias!r}")

        rango = self.obtener_rango_fechas(origen_dato)
        if rango is None:
            return self._consultar_historico(" WHERE 0", [])
        limite_inferior = rango[1] - timedelta(days=n_dias)

        condiciones = [f"{COLUMNA_FECHA_HORA} > ?"]
        parametros: list[Any] = [limite_inferior.strftime(FORMATO_FECHA_HORA)]
        if origen_dato is not None:
            condiciones.append("origen_dato = ?")
            parametros.append(origen_dato)
        return self._consultar_historico(f" WHERE {' AND '.join(condiciones)}", parametros)

    def obtener_historico_por_franja(
        self,
        hora_inicio: float,
        hora_fin: float,
        dias_semana: Sequence[int] | None = None,
        origen_dato: str | None = None,
    ) -> pd.DataFrame:
        """Consulta registros de una franja horaria y días de la semana dados.

        Base del cálculo de la mediana histórica del modo
        ``FALLBACK_HEURISTICO``.

        Args:
            hora_inicio: Hora decimal inicial inclusiva (``[0, 24)``).
            hora_fin: Hora decimal final inclusiva (``[0, 24)``, ``>= hora_inicio``).
            dias_semana: Días a incluir (lunes=0 … domingo=6); ``None`` = todos.
            origen_dato: Filtra por ``'SINTETICO'`` o ``'REAL'``.

        Returns:
            ``DataFrame`` cronológico con los registros de la franja.

        Raises:
            ErrorBaseDatos: Si la franja o los días son inválidos o falla SQLite.
        """
        inicio = _a_real(hora_inicio, "hora_inicio")
        fin = _a_real(hora_fin, "hora_fin")
        if not 0.0 <= inicio <= fin < PERIODO_HORA:
            raise ErrorBaseDatos(
                f"Franja horaria inválida [{inicio}, {fin}]; se requiere 0 <= inicio <= fin < 24."
            )

        condiciones = ["hora_decimal BETWEEN ? AND ?"]
        parametros: list[Any] = [inicio, fin]
        if dias_semana is not None:
            dias = sorted({int(dia) for dia in dias_semana})
            if not dias or any(not 0 <= dia < DIAS_POR_SEMANA for dia in dias):
                raise ErrorBaseDatos(f"'dias_semana' inválidos: {list(dias_semana)!r}")
            condiciones.append(f"dia_semana IN ({', '.join('?' for _ in dias)})")
            parametros.extend(dias)
        if origen_dato is not None:
            self._validar_origen(origen_dato)
            condiciones.append("origen_dato = ?")
            parametros.append(origen_dato)
        return self._consultar_historico(f" WHERE {' AND '.join(condiciones)}", parametros)

    def contar_registros_historico(self, origen_dato: str | None = None) -> int:
        """Cuenta los registros de ``historico_trafico``.

        Args:
            origen_dato: Filtra por ``'SINTETICO'`` o ``'REAL'``.

        Returns:
            Número de registros.

        Raises:
            ErrorBaseDatos: Si ``origen_dato`` es inválido o falla SQLite.
        """
        clausula, parametros = self._filtro_origen(origen_dato)
        with self.conexion() as conexion:
            fila = conexion.execute(
                f"SELECT COUNT(*) FROM {TABLA_HISTORICO}{clausula}", parametros
            ).fetchone()
        return int(fila[0])

    def obtener_rango_fechas(
        self, origen_dato: str | None = None
    ) -> tuple[datetime, datetime] | None:
        """Obtiene la primera y la última ``fecha_hora`` almacenadas.

        Args:
            origen_dato: Filtra por ``'SINTETICO'`` o ``'REAL'``.

        Returns:
            Tupla ``(fecha_minima, fecha_maxima)`` o ``None`` si la tabla está vacía.

        Raises:
            ErrorBaseDatos: Si ``origen_dato`` es inválido o falla SQLite.
        """
        clausula, parametros = self._filtro_origen(origen_dato)
        with self.conexion() as conexion:
            fila = conexion.execute(
                f"SELECT MIN({COLUMNA_FECHA_HORA}), MAX({COLUMNA_FECHA_HORA}) "
                f"FROM {TABLA_HISTORICO}{clausula}",
                parametros,
            ).fetchone()
        if fila[0] is None:
            return None
        return (
            datetime.strptime(fila[0], FORMATO_FECHA_HORA),
            datetime.strptime(fila[1], FORMATO_FECHA_HORA),
        )

    # ------------------------------------------------------------------
    # auditoria_inferencias
    # ------------------------------------------------------------------

    def registrar_auditoria(self, registro: Mapping[str, Any]) -> str:
        """Registra una inferencia en SQLite y en el log inmutable JSONL.

        Ambas escrituras ocurren dentro de la misma transacción: la fila se
        inserta, se anexa la línea JSON (con ``flush`` + ``fsync``) y solo
        entonces se hace ``COMMIT``. Si el log falla, la fila se revierte.

        Args:
            registro: Diccionario con las claves de ``RegistroAuditoria``.

        Returns:
            ``id_transaccion`` (UUID) del registro persistido.

        Raises:
            ErrorBaseDatos: Si el registro es inválido, el UUID ya existe,
                falla SQLite o no se puede escribir el log JSONL.
        """
        fila = _normalizar_registro_auditoria(registro)
        columnas = ", ".join(_COLUMNAS_AUDITORIA)
        marcadores = ", ".join(f":{columna}" for columna in _COLUMNAS_AUDITORIA)
        parametros = {**fila, "alerta_drift": int(fila["alerta_drift"])}

        with self.transaccion() as conexion:
            conexion.execute(
                f"INSERT INTO {TABLA_AUDITORIA} ({columnas}) VALUES ({marcadores})", parametros
            )
            self._anexar_linea_jsonl(fila)

        registro_log.info(
            "Auditoría %s registrada (modo=%s, estado=%s).",
            fila["id_transaccion"],
            fila["modo_inferencia"],
            fila["estado_auditoria"],
        )
        return str(fila["id_transaccion"])

    def obtener_auditorias(
        self,
        limite: int | None = None,
        modo_inferencia: str | None = None,
    ) -> pd.DataFrame:
        """Consulta el registro de auditorías, de la más reciente a la más antigua.

        Args:
            limite: Número máximo de filas (``> 0``); ``None`` = todas.
            modo_inferencia: Filtra por ``'RED_LSTM'`` o ``'FALLBACK_HEURISTICO'``.

        Returns:
            ``DataFrame`` con las columnas de ``auditoria_inferencias``
            (``alerta_drift`` como ``bool``).

        Raises:
            ErrorBaseDatos: Si algún filtro es inválido o falla SQLite.
        """
        condiciones: list[str] = []
        parametros: list[Any] = []
        if modo_inferencia is not None:
            if modo_inferencia not in MODOS_INFERENCIA_VALIDOS:
                raise ErrorBaseDatos(
                    f"'modo_inferencia'={modo_inferencia!r} inválido; "
                    f"permitidos: {MODOS_INFERENCIA_VALIDOS}"
                )
            condiciones.append("modo_inferencia = ?")
            parametros.append(modo_inferencia)
        clausula = f" WHERE {' AND '.join(condiciones)}" if condiciones else ""

        sentencia = (
            f"SELECT {', '.join(_COLUMNAS_AUDITORIA)} FROM {TABLA_AUDITORIA}{clausula} "
            "ORDER BY timestamp_consulta DESC"
        )
        if limite is not None:
            if isinstance(limite, bool) or not isinstance(limite, int) or limite <= 0:
                raise ErrorBaseDatos(f"'limite' debe ser un entero positivo, se recibió {limite!r}")
            sentencia += " LIMIT ?"
            parametros.append(limite)

        with self.conexion() as conexion:
            datos = pd.read_sql_query(sentencia, conexion, params=parametros)
        datos["alerta_drift"] = datos["alerta_drift"].astype(bool)
        datos["mae_modelo"] = datos["mae_modelo"].astype(float)
        return datos

    def contar_auditorias(self) -> int:
        """Cuenta los registros de ``auditoria_inferencias``.

        Returns:
            Número de inferencias auditadas.

        Raises:
            ErrorBaseDatos: Si falla SQLite.
        """
        with self.conexion() as conexion:
            fila = conexion.execute(f"SELECT COUNT(*) FROM {TABLA_AUDITORIA}").fetchone()
        return int(fila[0])

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _anexar_linea_jsonl(self, fila: Mapping[str, Any]) -> None:
        """Anexa un registro como línea JSON al log inmutable y fuerza su escritura a disco.

        Args:
            fila: Registro de auditoría normalizado (serializable a JSON).

        Raises:
            OSError: Si no se puede escribir el archivo (la transacción
                envolvente lo convierte en ``ErrorBaseDatos`` tras revertir).
        """
        linea = json.dumps(dict(fila), ensure_ascii=False, allow_nan=False)
        with self.ruta_log_jsonl.open("a", encoding="utf-8") as archivo:
            archivo.write(linea + "\n")
            archivo.flush()
            os.fsync(archivo.fileno())

    @staticmethod
    def _validar_origen(origen_dato: str) -> None:
        """Verifica que ``origen_dato`` sea una categoría permitida.

        Args:
            origen_dato: Valor a verificar.

        Raises:
            ErrorBaseDatos: Si el valor no está en ``ORIGENES_DATO_VALIDOS``.
        """
        if origen_dato not in ORIGENES_DATO_VALIDOS:
            raise ErrorBaseDatos(
                f"'origen_dato'={origen_dato!r} inválido; "
                f"valores permitidos: {ORIGENES_DATO_VALIDOS}"
            )

    def _filtro_origen(self, origen_dato: str | None) -> tuple[str, list[Any]]:
        """Construye la cláusula ``WHERE`` opcional por origen del dato.

        Args:
            origen_dato: ``'SINTETICO'``, ``'REAL'`` o ``None``.

        Returns:
            Tupla ``(clausula_sql, parametros)``.

        Raises:
            ErrorBaseDatos: Si ``origen_dato`` es inválido.
        """
        if origen_dato is None:
            return "", []
        self._validar_origen(origen_dato)
        return " WHERE origen_dato = ?", [origen_dato]

    def _consultar_historico(self, clausula: str, parametros: Sequence[Any]) -> pd.DataFrame:
        """Ejecuta un ``SELECT`` sobre ``historico_trafico`` y tipa el resultado.

        Args:
            clausula: Cláusula ``WHERE`` (con espacio inicial) o cadena vacía.
            parametros: Parámetros posicionales de la cláusula.

        Returns:
            ``DataFrame`` cronológico con tipos de ``pandas`` normalizados.

        Raises:
            ErrorBaseDatos: Si falla SQLite.
        """
        sentencia = (
            f"SELECT id, {', '.join(_COLUMNAS_HISTORICO)} FROM {TABLA_HISTORICO}"
            f"{clausula} ORDER BY {COLUMNA_FECHA_HORA} ASC"
        )
        with self.conexion() as conexion:
            datos = pd.read_sql_query(sentencia, conexion, params=list(parametros))

        datos[COLUMNA_FECHA_HORA] = pd.to_datetime(
            datos[COLUMNA_FECHA_HORA], format=FORMATO_FECHA_HORA
        )
        datos["dia_semana"] = datos["dia_semana"].astype(int)
        datos["hora_decimal"] = datos["hora_decimal"].astype(float)
        for columna in _COLUMNAS_BANDERA_HISTORICO:
            datos[columna] = datos[columna].astype(bool)
        for columna in ("tiempo_ruta_a_min", "tiempo_ruta_b_min"):
            datos[columna] = datos[columna].astype(float)
        return datos
