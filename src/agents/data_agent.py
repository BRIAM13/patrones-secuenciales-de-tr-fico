"""Agente de Datos e Ingestión (``AgenteDatos``).

Responsabilidades (Fase 3 de ``CONTEXT.md``):

* **Generación sintética** de un año horario realista: picos gaussianos en
  las horas punta, efecto quincena / fin de mes, fines de semana más ligeros,
  episodios de clima adverso, incidentes aleatorios en ``Ruta_A`` (volátil)
  y una ``Ruta_B`` más lenta en valle pero más estable.
* **Ingestión** de registros históricos reales (CSV o ``DataFrame``):
  regularización a una rejilla horaria, derivación de columnas de calendario
  y paso obligatorio por el ``ValidadorEntrada`` antes de persistir en SQLite.
* **Preparación para la LSTM**: codificación cíclica seno/coseno,
  ``MinMaxScaler`` ajustado solo con el tramo de entrenamiento y ventanas
  deslizantes ``14 → 7`` días cuyo objetivo nunca cruza la frontera entre
  entrenamiento, validación y prueba (sin *data leakage*).

Toda serie que sale de este agente hacia la red ha pasado por los guardrails
de entrada; cada ``ReporteValidacion`` queda en ``reportes_validacion`` para
que el ``AgenteAuditor`` lo registre.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    AMPLITUD_PICO_MANANA_MIN,
    AMPLITUD_PICO_TARDE_MIN,
    ATENUACION_PICO_RUTA_B,
    CENTRO_PICO_MANANA_H,
    CENTRO_PICO_TARDE_H,
    COLUMNA_FECHA_HORA,
    COLUMNAS_CARACTERISTICAS,
    COLUMNAS_OBJETIVO,
    COLUMNAS_OBLIGATORIAS_INGESTA,
    DECIMALES_TIEMPO,
    DEMORA_MEDIA_INCIDENTE_MIN,
    DIA_QUINCENA,
    DIAS_DATOS_SINTETICOS,
    DIAS_VENTANA_ENTRADA,
    DURACION_MAX_CLIMA_H,
    DURACION_MIN_CLIMA_H,
    FACTOR_CLIMA_ADVERSO,
    FACTOR_FIN_DE_SEMANA,
    FACTOR_QUINCENA,
    FECHA_INICIO_SINTETICA,
    FRECUENCIA_MUESTREO,
    HORAS_POR_DIA,
    ORIGEN_REAL,
    ORIGEN_SINTETICO,
    ORIGENES_DATO_VALIDOS,
    PASO_VENTANA_DESLIZANTE,
    PASOS_VENTANA_ENTRADA,
    PASOS_VENTANA_SALIDA,
    PENALIZACION_BASE_RUTA_B_MIN,
    PERIODO_DIA_SEMANA,
    PERIODO_HORA,
    PROBABILIDAD_CLIMA_ADVERSO,
    PROBABILIDAD_INCIDENTE_RUTA_A,
    PROPORCION_PRUEBA,
    PROPORCION_VALIDACION,
    RANGO_ESCALADO,
    RUTA_SCALER_ENTRADA,
    RUTA_SCALER_SALIDA,
    SEMILLA_ALEATORIA,
    SIGMA_PICO_H,
    TIEMPO_BASE_MIN,
    TIEMPO_MAX_ENTRADA_MIN,
    TIEMPO_MIN_ENTRADA_MIN,
    VOLATILIDAD_RUTA_A,
    VOLATILIDAD_RUTA_B,
)
from src.database.db_manager import GestorBaseDatos
from src.guardrails.exceptions import (
    ErrorIntegridadSecuencial,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
)
from src.guardrails.input_validator import (
    COLUMNAS_ESQUEMA_ENTRADA,
    ReporteValidacion,
    ResultadoValidacion,
    ValidadorEntrada,
)

registro_log: logging.Logger = logging.getLogger(__name__)

MatrizFlotante = npt.NDArray[np.float32]
"""Alias de los tensores ``float32`` que consume Keras."""

_TEXTOS_BOOLEANOS: Final[Mapping[str, bool]] = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "si": True,
    "sí": True,
    "no": False,
}


# =============================================================================
# Estructuras de salida
# =============================================================================


@dataclass(frozen=True)
class ConjuntosEntrenamiento:
    """Tensores listos para Keras, separados cronológicamente sin fuga de datos.

    Formas: ``x_*`` → ``(n_ventanas, 336, n_caracteristicas)``;
    ``y_*`` → ``(n_ventanas, 168, 2)``. Los valores están escalados con los
    ``MinMaxScaler`` ajustados solo sobre el tramo de entrenamiento.

    Attributes:
        x_entrenamiento: Ventanas de entrada de entrenamiento.
        y_entrenamiento: Objetivos de entrenamiento.
        x_validacion: Ventanas de entrada de validación.
        y_validacion: Objetivos de validación.
        x_prueba: Ventanas de entrada de prueba.
        y_prueba: Objetivos de prueba.
        inicio_objetivo_entrenamiento: Primera hora de cada objetivo de entrenamiento.
        inicio_objetivo_validacion: Primera hora de cada objetivo de validación.
        inicio_objetivo_prueba: Primera hora de cada objetivo de prueba.
        columnas_caracteristicas: Orden de las características en el eje 2 de ``x``.
        columnas_objetivo: Orden de las variables en el eje 2 de ``y``.
    """

    x_entrenamiento: MatrizFlotante
    y_entrenamiento: MatrizFlotante
    x_validacion: MatrizFlotante
    y_validacion: MatrizFlotante
    x_prueba: MatrizFlotante
    y_prueba: MatrizFlotante
    inicio_objetivo_entrenamiento: pd.DatetimeIndex
    inicio_objetivo_validacion: pd.DatetimeIndex
    inicio_objetivo_prueba: pd.DatetimeIndex
    columnas_caracteristicas: tuple[str, ...] = COLUMNAS_CARACTERISTICAS
    columnas_objetivo: tuple[str, ...] = COLUMNAS_OBJETIVO


@dataclass(frozen=True)
class VentanaInferencia:
    """Ventana de los últimos ``14`` días lista para predecir los ``7`` siguientes.

    Attributes:
        x: Tensor escalado de forma ``(1, 336, n_caracteristicas)``.
        datos: Ventana saneada (sin escalar) para el estratega y el fallback.
        reporte: Reporte de los guardrails de entrada sobre la ventana.
        fechas_objetivo: Las ``168`` horas futuras que se van a predecir.
    """

    x: MatrizFlotante
    datos: pd.DataFrame
    reporte: ReporteValidacion
    fechas_objetivo: pd.DatetimeIndex


# =============================================================================
# Funciones de transformación (puras)
# =============================================================================


def derivar_columnas_calendario(fechas: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Deriva las columnas de calendario a partir de las marcas de tiempo.

    Args:
        fechas: Marcas de tiempo horarias ingenuas.

    Returns:
        ``DataFrame`` con ``dia_semana`` (lunes=0), ``hora_decimal``,
        ``es_fin_de_semana`` y ``es_quincena_o_fin_mes`` (día 15 o último día
        del mes), alineado posicionalmente con ``fechas``.
    """
    indice = pd.DatetimeIndex(fechas)
    dia_semana = indice.weekday.to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "dia_semana": dia_semana,
            "hora_decimal": (indice.hour + indice.minute / 60.0 + indice.second / 3600.0).to_numpy(
                dtype=float
            ),
            "es_fin_de_semana": dia_semana >= 5,
            "es_quincena_o_fin_mes": np.asarray(
                (indice.day == DIA_QUINCENA) | indice.is_month_end, dtype=bool
            ),
        }
    )


def codificar_variables_ciclicas(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade la codificación seno/coseno de la hora y del día (Regla 3.1.3).

    ``hora_sin = sin(2π·h/24)``, ``hora_cos = cos(2π·h/24)``,
    ``dia_sin = sin(2π·d/7)``, ``dia_cos = cos(2π·d/7)``.

    Args:
        datos: Tabla con ``hora_decimal`` (``[0, 24)``) y ``dia_semana`` (``[0, 7)``).

    Returns:
        Copia de ``datos`` con las columnas ``hora_sin``, ``hora_cos``,
        ``dia_sin`` y ``dia_cos``.

    Raises:
        ErrorValidacionEntrada: Si faltan ``hora_decimal`` o ``dia_semana``.
    """
    faltantes = [columna for columna in ("hora_decimal", "dia_semana") if columna not in datos]
    if faltantes:
        raise ErrorValidacionEntrada(
            f"No se puede aplicar la codificación cíclica; faltan columnas {faltantes}.",
            detalles={"columnas_faltantes": faltantes},
        )
    codificados = datos.copy()
    angulo_hora = 2.0 * np.pi * codificados["hora_decimal"].to_numpy(dtype=float) / PERIODO_HORA
    angulo_dia = 2.0 * np.pi * codificados["dia_semana"].to_numpy(dtype=float) / PERIODO_DIA_SEMANA
    codificados["hora_sin"] = np.sin(angulo_hora)
    codificados["hora_cos"] = np.cos(angulo_hora)
    codificados["dia_sin"] = np.sin(angulo_dia)
    codificados["dia_cos"] = np.cos(angulo_dia)
    return codificados


def _a_serie_booleana(serie: pd.Series, nombre: str) -> pd.Series:
    """Convierte una columna externa a booleanos de forma estricta.

    Acepta ``bool``, ``0``/``1`` y los textos ``true/false``, ``1/0`` y
    ``si/sí/no`` (sin distinguir mayúsculas).

    Args:
        serie: Columna a convertir.
        nombre: Nombre de la columna (para el mensaje de error).

    Returns:
        Serie de tipo ``bool``.

    Raises:
        ErrorValidacionEntrada: Si algún valor no representa un booleano.
    """
    if pd.api.types.is_bool_dtype(serie):
        return serie.astype(bool)

    def convertir(valor: Any) -> bool | None:
        if isinstance(valor, (bool, np.bool_)):
            return bool(valor)
        if isinstance(valor, (int, float, np.integer, np.floating)) and valor in (0, 1):
            return bool(valor)
        if isinstance(valor, str):
            return _TEXTOS_BOOLEANOS.get(valor.strip().lower())
        return None

    convertida = serie.map(convertir)
    invalidos = serie[convertida.isna()]
    if len(invalidos):
        raise ErrorValidacionEntrada(
            f"La columna '{nombre}' contiene {len(invalidos)} valor(es) no booleanos; "
            f"ejemplos: {invalidos.head(5).tolist()}",
            detalles={"columna": nombre, "ejemplos": [str(v) for v in invalidos.head(5)]},
        )
    return convertida.astype(bool)


# =============================================================================
# Agente de Datos
# =============================================================================


class AgenteDatos:
    """Agente responsable de generar, ingerir, validar y preparar los datos.

    Attributes:
        gestor_bd: Gestor de persistencia SQLite.
        validador: Guardrail de entrada aplicado a toda serie o ventana.
        semilla: Semilla del generador aleatorio (reproducibilidad).
        scaler_entrada: ``MinMaxScaler`` de las características (``None``
            hasta ajustarse o cargarse).
        scaler_salida: ``MinMaxScaler`` de los tiempos objetivo.
        reportes_validacion: Reportes emitidos, pendientes de auditoría.

    Example:
        >>> agente = AgenteDatos(GestorBaseDatos())
        >>> resultado = agente.inicializar_datos()
        >>> conjuntos = agente.construir_conjuntos(resultado.datos)
        >>> ventana = agente.preparar_ventana_inferencia()
    """

    def __init__(
        self,
        gestor_bd: GestorBaseDatos,
        validador: ValidadorEntrada | None = None,
        semilla: int = SEMILLA_ALEATORIA,
    ) -> None:
        """Inicializa el agente con sus dependencias.

        Args:
            gestor_bd: Gestor de persistencia SQLite.
            validador: Guardrail de entrada; por defecto, uno con los límites
                de ``config``.
            semilla: Semilla para la generación sintética.
        """
        self.gestor_bd: GestorBaseDatos = gestor_bd
        self.validador: ValidadorEntrada = validador or ValidadorEntrada()
        self.semilla: int = semilla
        self.scaler_entrada: MinMaxScaler | None = None
        self.scaler_salida: MinMaxScaler | None = None
        self.reportes_validacion: list[ReporteValidacion] = []

    # ------------------------------------------------------------------
    # Generación sintética
    # ------------------------------------------------------------------

    def generar_datos_sinteticos(
        self,
        dias: int = DIAS_DATOS_SINTETICOS,
        fecha_inicio: str | pd.Timestamp = FECHA_INICIO_SINTETICA,
        fraccion_corrupcion: float = 0.0,
    ) -> pd.DataFrame:
        """Genera una serie horaria sintética y realista de ambas rutas.

        Modelo por hora ``h``::

            congestion = (A_m·N(h; 8, σ) + A_t·N(h; 19, σ)) · f_finde · f_quincena
            Ruta_A = (25 + congestion) · f_clima · ruido(σ_A) + incidentes
            Ruta_B = (25 + 8 + 0.55·congestion) · f_clima · ruido(σ_B)

        donde ``N`` es una campana gaussiana normalizada a 1 en su centro.

        Args:
            dias: Número de días a generar (``>= 1``).
            fecha_inicio: Primera hora de la serie (debe caer en hora en punto).
            fraccion_corrupcion: Fracción de tiempos que se sustituyen por
                ``NaN`` en cada ruta para ejercitar los guardrails
                (``[0, 1)``; por defecto ``0``).

        Returns:
            ``DataFrame`` con todas las columnas de ``COLUMNAS_ESQUEMA_ENTRADA``
            y ``origen_dato='SINTETICO'``.

        Raises:
            ValueError: Si ``dias`` o ``fraccion_corrupcion`` son inválidos o
                ``fecha_inicio`` no está alineada a la hora.
        """
        if dias < 1:
            raise ValueError(f"'dias' debe ser >= 1, se recibió {dias}")
        if not 0.0 <= fraccion_corrupcion < 1.0:
            raise ValueError(f"'fraccion_corrupcion' debe estar en [0, 1): {fraccion_corrupcion}")
        inicio = pd.Timestamp(fecha_inicio)
        if inicio != inicio.floor("h") or inicio.tzinfo is not None:
            raise ValueError(f"'fecha_inicio' debe ser ingenua y en hora en punto: {fecha_inicio}")

        generador = np.random.default_rng(self.semilla)
        fechas = pd.date_range(inicio, periods=dias * HORAS_POR_DIA, freq=FRECUENCIA_MUESTREO)
        calendario = derivar_columnas_calendario(fechas)
        n_registros = len(fechas)
        hora = calendario["hora_decimal"].to_numpy()

        pico = AMPLITUD_PICO_MANANA_MIN * _campana(hora, CENTRO_PICO_MANANA_H) + (
            AMPLITUD_PICO_TARDE_MIN * _campana(hora, CENTRO_PICO_TARDE_H)
        )
        factor_dia = np.where(calendario["es_fin_de_semana"], FACTOR_FIN_DE_SEMANA, 1.0)
        factor_quincena = np.where(calendario["es_quincena_o_fin_mes"], FACTOR_QUINCENA, 1.0)
        congestion = pico * factor_dia * factor_quincena

        clima_adverso = self._simular_clima_adverso(dias, generador)
        factor_clima = np.where(clima_adverso, FACTOR_CLIMA_ADVERSO, 1.0)

        incidentes = generador.random(n_registros) < PROBABILIDAD_INCIDENTE_RUTA_A
        demora_incidentes = generador.exponential(DEMORA_MEDIA_INCIDENTE_MIN, n_registros)
        tiempo_a = (TIEMPO_BASE_MIN + congestion) * factor_clima * generador.normal(
            1.0, VOLATILIDAD_RUTA_A, n_registros
        ) + np.where(incidentes, demora_incidentes, 0.0)
        tiempo_b = (
            (TIEMPO_BASE_MIN + PENALIZACION_BASE_RUTA_B_MIN + ATENUACION_PICO_RUTA_B * congestion)
            * factor_clima
            * generador.normal(1.0, VOLATILIDAD_RUTA_B, n_registros)
        )

        datos = pd.DataFrame({COLUMNA_FECHA_HORA: fechas})
        datos = pd.concat([datos, calendario], axis=1)
        datos["clima_adverso"] = clima_adverso
        for columna, valores in (
            (COLUMNAS_OBJETIVO[0], tiempo_a),
            (COLUMNAS_OBJETIVO[1], tiempo_b),
        ):
            acotados = np.clip(valores, TIEMPO_MIN_ENTRADA_MIN, TIEMPO_MAX_ENTRADA_MIN)
            datos[columna] = np.round(acotados, DECIMALES_TIEMPO)
            if fraccion_corrupcion > 0.0:
                n_corruptos = int(round(fraccion_corrupcion * n_registros))
                posiciones = generador.choice(n_registros, size=n_corruptos, replace=False)
                datos.loc[posiciones, columna] = np.nan
        datos["origen_dato"] = ORIGEN_SINTETICO

        registro_log.info(
            "Serie sintética generada: %d registros (%s → %s), %d hora(s) con clima adverso, "
            "%d incidente(s) en Ruta_A.",
            n_registros,
            fechas[0],
            fechas[-1],
            int(clima_adverso.sum()),
            int(incidentes.sum()),
        )
        return datos.loc[:, list(COLUMNAS_ESQUEMA_ENTRADA)]

    def _simular_clima_adverso(
        self, dias: int, generador: np.random.Generator
    ) -> npt.NDArray[np.bool_]:
        """Simula episodios contiguos de clima adverso.

        Cada día inicia un episodio con probabilidad ``PROBABILIDAD_CLIMA_ADVERSO``,
        con hora de inicio uniforme y duración entre ``DURACION_MIN_CLIMA_H`` y
        ``DURACION_MAX_CLIMA_H`` horas (puede extenderse al día siguiente).

        Args:
            dias: Número de días simulados.
            generador: Generador aleatorio del agente.

        Returns:
            Máscara horaria de longitud ``dias * 24``.
        """
        n_registros = dias * HORAS_POR_DIA
        mascara = np.zeros(n_registros, dtype=bool)
        dias_con_episodio = np.flatnonzero(generador.random(dias) < PROBABILIDAD_CLIMA_ADVERSO)
        for dia in dias_con_episodio:
            inicio = dia * HORAS_POR_DIA + int(generador.integers(0, HORAS_POR_DIA))
            duracion = int(generador.integers(DURACION_MIN_CLIMA_H, DURACION_MAX_CLIMA_H + 1))
            mascara[inicio : min(inicio + duracion, n_registros)] = True
        return mascara

    # ------------------------------------------------------------------
    # Ingestión y persistencia
    # ------------------------------------------------------------------

    def ingerir_csv(
        self,
        ruta_csv: Path | str,
        origen_dato: str = ORIGEN_REAL,
        reemplazar_existentes: bool = False,
        separador: str = ",",
    ) -> int:
        """Lee un CSV de registros históricos, lo valida y lo persiste.

        Columnas mínimas: ``fecha_hora``, ``tiempo_ruta_a_min`` y
        ``tiempo_ruta_b_min``; ``clima_adverso`` es opcional (``False``).

        Args:
            ruta_csv: Ruta del archivo CSV.
            origen_dato: ``'REAL'`` (por defecto) o ``'SINTETICO'``.
            reemplazar_existentes: Si es ``True``, sobrescribe horas ya guardadas.
            separador: Separador de columnas del CSV.

        Returns:
            Número de filas insertadas o actualizadas en SQLite.

        Raises:
            ErrorValidacionEntrada: Si el archivo no se puede leer o los datos
                violan los guardrails de entrada.
            ErrorIntegridadSecuencial: Si hay marcas de tiempo duplicadas o no
                alineadas a la hora.
            ErrorBaseDatos: Si falla la persistencia.
        """
        ruta = Path(ruta_csv)
        try:
            datos = pd.read_csv(ruta, sep=separador)
        except (
            OSError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
        ) as error:
            raise ErrorValidacionEntrada(
                f"No se pudo leer el CSV '{ruta}': {error}", detalles={"ruta": str(ruta)}
            ) from error
        return self.ingerir_dataframe(datos, origen_dato, reemplazar_existentes)

    def ingerir_dataframe(
        self,
        datos: pd.DataFrame,
        origen_dato: str = ORIGEN_REAL,
        reemplazar_existentes: bool = False,
    ) -> int:
        """Normaliza, valida y persiste una tabla de registros históricos.

        Flujo: columnas mínimas → tipos → rejilla horaria continua (las horas
        ausentes se insertan como faltantes) → columnas de calendario →
        ``ValidadorEntrada.validar_serie`` (aborta si alguna ventana supera el
        15 % de corrupción) → persistencia. Los valores corruptos se guardan
        como ``NULL`` (nunca el valor inválido) y se imputan de forma
        reproducible al cargar.

        Args:
            datos: Tabla con al menos ``COLUMNAS_OBLIGATORIAS_INGESTA``.
            origen_dato: ``'REAL'`` (por defecto) o ``'SINTETICO'``.
            reemplazar_existentes: Si es ``True``, sobrescribe horas ya guardadas.

        Returns:
            Número de filas insertadas o actualizadas en SQLite.

        Raises:
            ErrorValidacionEntrada: Si faltan columnas, los tipos son inválidos
                o la corrupción supera el umbral.
            ErrorIntegridadSecuencial: Si hay duplicados o registros no
                alineados a la hora.
            ErrorBaseDatos: Si falla la persistencia.
        """
        normalizados = self._normalizar_fuente(datos, origen_dato)
        resultado = self.validador.validar_serie(normalizados)
        self.reportes_validacion.append(resultado.reporte)

        a_persistir = normalizados.copy()
        for columna, mascara in self.validador.mascaras_corruptos(a_persistir).items():
            a_persistir.loc[mascara, columna] = np.nan
        filas = self.gestor_bd.insertar_dataframe_historico(
            a_persistir, reemplazar_existentes=reemplazar_existentes
        )
        registro_log.info(
            "Ingesta %s: %d registro(s) validados, %d persistidos, %d valor(es) corruptos "
            "guardados como NULL.",
            origen_dato,
            len(a_persistir),
            filas,
            resultado.reporte.total_imputados,
        )
        return filas

    def _normalizar_fuente(self, datos: pd.DataFrame, origen_dato: str) -> pd.DataFrame:
        """Convierte una fuente externa al esquema canónico sobre una rejilla horaria.

        Args:
            datos: Tabla externa.
            origen_dato: Origen a asignar a todos los registros.

        Returns:
            ``DataFrame`` continuo con ``COLUMNAS_ESQUEMA_ENTRADA``.

        Raises:
            ErrorValidacionEntrada: Si faltan columnas, el origen es inválido
                o hay valores no convertibles.
            ErrorIntegridadSecuencial: Si hay duplicados o horas no exactas.
        """
        if not isinstance(datos, pd.DataFrame) or datos.empty:
            raise ErrorValidacionEntrada("La fuente de datos debe ser un DataFrame no vacío.")
        if origen_dato not in ORIGENES_DATO_VALIDOS:
            raise ErrorValidacionEntrada(
                f"'origen_dato'={origen_dato!r} inválido; permitidos: {ORIGENES_DATO_VALIDOS}"
            )
        faltantes = [c for c in COLUMNAS_OBLIGATORIAS_INGESTA if c not in datos.columns]
        if faltantes:
            raise ErrorValidacionEntrada(
                f"Faltan columnas obligatorias en la fuente: {faltantes}",
                detalles={"columnas_faltantes": faltantes},
            )

        try:
            fechas = pd.to_datetime(datos[COLUMNA_FECHA_HORA])
        except (ValueError, TypeError) as error:
            raise ErrorValidacionEntrada(
                f"'fecha_hora' contiene valores no válidos: {error}"
            ) from error
        if fechas.isna().any():
            raise ErrorValidacionEntrada(
                f"'fecha_hora' contiene {int(fechas.isna().sum())} valor(es) vacíos."
            )
        if fechas.dt.tz is not None:
            raise ErrorValidacionEntrada("'fecha_hora' debe estar en hora local sin zona horaria.")

        tabla = pd.DataFrame({COLUMNA_FECHA_HORA: fechas})
        for columna in COLUMNAS_OBJETIVO:
            try:
                tabla[columna] = pd.to_numeric(datos[columna], errors="raise").astype(float)
            except (ValueError, TypeError) as error:
                raise ErrorValidacionEntrada(
                    f"La columna '{columna}' contiene valores no numéricos: {error}",
                    detalles={"columna": columna},
                ) from error
        if "clima_adverso" in datos.columns:
            tabla["clima_adverso"] = _a_serie_booleana(datos["clima_adverso"], "clima_adverso")
        else:
            tabla["clima_adverso"] = False

        duplicadas = tabla[COLUMNA_FECHA_HORA][tabla[COLUMNA_FECHA_HORA].duplicated()]
        if len(duplicadas):
            raise ErrorIntegridadSecuencial(
                f"La fuente contiene {len(duplicadas)} marca(s) de tiempo duplicadas.",
                detalles={"duplicadas": [str(f) for f in duplicadas.head(10)]},
            )
        desalineadas = tabla[COLUMNA_FECHA_HORA][
            tabla[COLUMNA_FECHA_HORA] != tabla[COLUMNA_FECHA_HORA].dt.floor("h")
        ]
        if len(desalineadas):
            raise ErrorIntegridadSecuencial(
                f"La fuente contiene {len(desalineadas)} registro(s) fuera de la hora en punto.",
                detalles={"desalineadas": [str(f) for f in desalineadas.head(10)]},
            )

        tabla = tabla.sort_values(COLUMNA_FECHA_HORA).set_index(COLUMNA_FECHA_HORA)
        rejilla = pd.date_range(tabla.index[0], tabla.index[-1], freq=FRECUENCIA_MUESTREO)
        horas_ausentes = len(rejilla) - len(tabla)
        tabla = tabla.reindex(rejilla)
        tabla.index.name = COLUMNA_FECHA_HORA
        if horas_ausentes:
            registro_log.warning(
                "La fuente tenía %d hora(s) ausentes; se insertan como datos faltantes "
                "(sujetos a la regla del 15 %% del ValidadorEntrada).",
                horas_ausentes,
            )
        tabla["clima_adverso"] = tabla["clima_adverso"].eq(True)  # horas insertadas → False
        tabla = tabla.reset_index()

        calendario = derivar_columnas_calendario(tabla[COLUMNA_FECHA_HORA])
        tabla = pd.concat([tabla, calendario], axis=1)
        tabla["origen_dato"] = origen_dato
        return tabla.loc[:, list(COLUMNAS_ESQUEMA_ENTRADA)]

    # ------------------------------------------------------------------
    # Carga validada desde SQLite
    # ------------------------------------------------------------------

    def cargar_historico(
        self,
        origen_dato: str | None = None,
        fecha_inicio: pd.Timestamp | str | None = None,
        fecha_fin: pd.Timestamp | str | None = None,
    ) -> ResultadoValidacion:
        """Lee la serie histórica de SQLite y le aplica los guardrails de entrada.

        Args:
            origen_dato: Filtra por ``'SINTETICO'`` o ``'REAL'``.
            fecha_inicio: Límite inferior inclusivo.
            fecha_fin: Límite superior inclusivo.

        Returns:
            Serie saneada (``NULL`` imputados) y su reporte.

        Raises:
            ErrorValidacionEntrada: Si no hay datos o se violan los guardrails.
            ErrorIntegridadSecuencial: Si la serie almacenada no es continua.
            ErrorBaseDatos: Si falla la consulta.
        """
        datos = self.gestor_bd.obtener_historico(
            fecha_inicio=_a_fecha_opcional(fecha_inicio),
            fecha_fin=_a_fecha_opcional(fecha_fin),
            origen_dato=origen_dato,
        )
        if datos.empty:
            raise ErrorValidacionEntrada(
                "No hay registros históricos para los filtros indicados.",
                detalles={"origen_dato": origen_dato},
            )
        resultado = self.validador.validar_serie(datos)
        self.reportes_validacion.append(resultado.reporte)
        return resultado

    def inicializar_datos(
        self,
        forzar_regeneracion: bool = False,
        dias: int = DIAS_DATOS_SINTETICOS,
    ) -> ResultadoValidacion:
        """Garantiza que exista historia en SQLite y la devuelve validada.

        Si la base está vacía (o se fuerza la regeneración), genera la serie
        sintética, la ingiere por el mismo flujo validado que los datos reales
        y luego carga la historia completa.

        Args:
            forzar_regeneracion: Si es ``True``, borra los datos sintéticos
                existentes y los vuelve a generar.
            dias: Días de historia sintética a generar.

        Returns:
            Serie histórica saneada y su reporte.

        Raises:
            ErrorValidacionEntrada: Si se violan los guardrails de entrada.
            ErrorBaseDatos: Si falla la persistencia.
        """
        if forzar_regeneracion:
            self.gestor_bd.eliminar_historico(ORIGEN_SINTETICO)
        if self.gestor_bd.contar_registros_historico() == 0:
            sinteticos = self.generar_datos_sinteticos(dias=dias)
            self.ingerir_dataframe(sinteticos, origen_dato=ORIGEN_SINTETICO)
        return self.cargar_historico()

    # ------------------------------------------------------------------
    # Preparación para la LSTM
    # ------------------------------------------------------------------

    def construir_conjuntos(
        self,
        datos: pd.DataFrame,
        paso: int = PASO_VENTANA_DESLIZANTE,
        proporcion_validacion: float = PROPORCION_VALIDACION,
        proporcion_prueba: float = PROPORCION_PRUEBA,
        guardar_scalers: bool = True,
    ) -> ConjuntosEntrenamiento:
        """Construye ventanas ``14 → 7`` días separadas sin fuga de información.

        Garantías anti-*leakage*:

        1. La serie se divide cronológicamente en tramos de entrenamiento,
           validación y prueba **antes** de crear ventanas.
        2. Los ``MinMaxScaler`` se ajustan solo con el tramo de entrenamiento.
        3. El objetivo ``y`` de cada ventana cae íntegramente dentro de su
           tramo; las ventanas cuyo objetivo cruza una frontera se descartan.
           (La entrada ``x`` de validación/prueba puede incluir horas previas,
           que en producción también serían pasado conocido.)

        Args:
            datos: Serie saneada (salida de ``cargar_historico`` o
                ``inicializar_datos``).
            paso: Desplazamiento en horas entre ventanas consecutivas.
            proporcion_validacion: Fracción cronológica para validación.
            proporcion_prueba: Fracción cronológica final para prueba.
            guardar_scalers: Si es ``True``, persiste los scalers en ``artifacts/``.

        Returns:
            Tensores ``float32`` de los tres conjuntos.

        Raises:
            ErrorValidacionEntrada: Si los datos no están saneados, las
                proporciones son inválidas o algún tramo no alcanza para una
                ventana completa.
            ErrorIntegridadSecuencial: Si la serie no es continua.
        """
        if paso < 1:
            raise ErrorValidacionEntrada(f"'paso' debe ser >= 1, se recibió {paso}")
        if not (
            0.0 < proporcion_validacion < 1.0
            and 0.0 < proporcion_prueba < 1.0
            and proporcion_validacion + proporcion_prueba < 1.0
        ):
            raise ErrorValidacionEntrada(
                "Proporciones inválidas: se requiere 0 < validación, prueba y validación + "
                f"prueba < 1 ({proporcion_validacion}, {proporcion_prueba})."
            )
        self._exigir_datos_saneados(datos)
        self.validador.validar_continuidad(datos)

        ordenados = datos.reset_index(drop=True)
        caracteristicas = self._matriz_caracteristicas(ordenados)
        objetivos = ordenados.loc[:, list(COLUMNAS_OBJETIVO)].to_numpy(dtype=float)
        fechas = pd.DatetimeIndex(ordenados[COLUMNA_FECHA_HORA])

        n_registros = len(ordenados)
        fin_entrenamiento = int(n_registros * (1.0 - proporcion_validacion - proporcion_prueba))
        fin_validacion = int(n_registros * (1.0 - proporcion_prueba))

        self.scaler_entrada = MinMaxScaler(feature_range=RANGO_ESCALADO)
        self.scaler_salida = MinMaxScaler(feature_range=RANGO_ESCALADO)
        self.scaler_entrada.fit(caracteristicas[:fin_entrenamiento])
        self.scaler_salida.fit(objetivos[:fin_entrenamiento])
        x_escalado = self.scaler_entrada.transform(caracteristicas).astype(np.float32)
        y_escalado = self.scaler_salida.transform(objetivos).astype(np.float32)

        largo_total = PASOS_VENTANA_ENTRADA + PASOS_VENTANA_SALIDA
        inicios = np.arange(0, n_registros - largo_total + 1, paso)
        inicio_objetivo = inicios + PASOS_VENTANA_ENTRADA
        fin_objetivo = inicio_objetivo + PASOS_VENTANA_SALIDA
        tramos = {
            "entrenamiento": inicios[fin_objetivo <= fin_entrenamiento],
            "validacion": inicios[
                (inicio_objetivo >= fin_entrenamiento) & (fin_objetivo <= fin_validacion)
            ],
            "prueba": inicios[inicio_objetivo >= fin_validacion],
        }
        vacios = [nombre for nombre, seleccion in tramos.items() if seleccion.size == 0]
        if vacios:
            raise ErrorValidacionEntrada(
                f"Historia insuficiente: los tramos {vacios} no contienen ninguna ventana "
                f"completa de {DIAS_VENTANA_ENTRADA} + {PASOS_VENTANA_SALIDA // HORAS_POR_DIA} "
                f"días ({n_registros} registros disponibles).",
                detalles={"registros": n_registros, "tramos_vacios": vacios},
            )

        def ventanas(seleccion: npt.NDArray[np.int_]) -> tuple[MatrizFlotante, MatrizFlotante]:
            x = np.stack([x_escalado[i : i + PASOS_VENTANA_ENTRADA] for i in seleccion])
            y = np.stack(
                [y_escalado[i + PASOS_VENTANA_ENTRADA : i + largo_total] for i in seleccion]
            )
            return x, y

        x_ent, y_ent = ventanas(tramos["entrenamiento"])
        x_val, y_val = ventanas(tramos["validacion"])
        x_pru, y_pru = ventanas(tramos["prueba"])

        if guardar_scalers:
            self.guardar_scalers()

        registro_log.info(
            "Ventanas construidas (paso=%dh): entrenamiento=%d, validación=%d, prueba=%d; "
            "forma x=%s, y=%s.",
            paso,
            len(x_ent),
            len(x_val),
            len(x_pru),
            x_ent.shape[1:],
            y_ent.shape[1:],
        )
        return ConjuntosEntrenamiento(
            x_entrenamiento=x_ent,
            y_entrenamiento=y_ent,
            x_validacion=x_val,
            y_validacion=y_val,
            x_prueba=x_pru,
            y_prueba=y_pru,
            inicio_objetivo_entrenamiento=fechas[tramos["entrenamiento"] + PASOS_VENTANA_ENTRADA],
            inicio_objetivo_validacion=fechas[tramos["validacion"] + PASOS_VENTANA_ENTRADA],
            inicio_objetivo_prueba=fechas[tramos["prueba"] + PASOS_VENTANA_ENTRADA],
        )

    def preparar_ventana_inferencia(self, datos: pd.DataFrame | None = None) -> VentanaInferencia:
        """Prepara la ventana de los últimos ``14`` días para la predicción.

        Args:
            datos: Ventana de exactamente ``336`` horas; si es ``None`` se leen
                los últimos ``14`` días de SQLite.

        Returns:
            Ventana escalada, ventana saneada, reporte y horas objetivo.

        Raises:
            ErrorSistemaTrafico: Si los scalers no están ajustados ni cargados.
            ErrorValidacionEntrada: Si la ventana viola los guardrails.
            ErrorIntegridadSecuencial: Si la ventana no es continua o no mide
                ``14`` días.
        """
        scaler_entrada = self._exigir_scaler_entrada()
        if datos is None:
            datos = self.gestor_bd.obtener_ultimos_dias(DIAS_VENTANA_ENTRADA)
            if datos.empty:
                raise ErrorValidacionEntrada("No hay historia en SQLite para construir la ventana.")

        resultado = self.validador.validar_ventana(datos)
        self.reportes_validacion.append(resultado.reporte)

        caracteristicas = self._matriz_caracteristicas(resultado.datos)
        x = scaler_entrada.transform(caracteristicas).astype(np.float32)[np.newaxis, ...]
        ultima_hora = pd.Timestamp(resultado.datos[COLUMNA_FECHA_HORA].iloc[-1])
        fechas_objetivo = pd.date_range(
            ultima_hora + pd.Timedelta(hours=1),
            periods=PASOS_VENTANA_SALIDA,
            freq=FRECUENCIA_MUESTREO,
        )
        return VentanaInferencia(
            x=x, datos=resultado.datos, reporte=resultado.reporte, fechas_objetivo=fechas_objetivo
        )

    def invertir_escala_salida(self, y_escalado: npt.NDArray[np.floating[Any]]) -> MatrizFlotante:
        """Devuelve predicciones escaladas a minutos reales.

        Args:
            y_escalado: Arreglo cuyo último eje tiene tamaño ``2``
                (``Ruta_A``, ``Ruta_B``), p. ej. ``(n, 168, 2)``.

        Returns:
            Arreglo de la misma forma en minutos.

        Raises:
            ErrorSistemaTrafico: Si el scaler de salida no está disponible.
            ValueError: Si el último eje no coincide con las columnas objetivo.
        """
        if self.scaler_salida is None:
            raise ErrorSistemaTrafico(
                "El scaler de salida no está ajustado: ejecute construir_conjuntos() o "
                "cargar_scalers()."
            )
        arreglo = np.asarray(y_escalado, dtype=float)
        if arreglo.shape[-1] != len(COLUMNAS_OBJETIVO):
            raise ValueError(
                f"El último eje debe medir {len(COLUMNAS_OBJETIVO)}, se recibió {arreglo.shape}"
            )
        planos = self.scaler_salida.inverse_transform(arreglo.reshape(-1, len(COLUMNAS_OBJETIVO)))
        return planos.reshape(arreglo.shape).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistencia de scalers
    # ------------------------------------------------------------------

    def guardar_scalers(
        self,
        ruta_entrada: Path | str = RUTA_SCALER_ENTRADA,
        ruta_salida: Path | str = RUTA_SCALER_SALIDA,
    ) -> None:
        """Guarda los scalers ajustados con ``joblib``.

        Args:
            ruta_entrada: Destino del scaler de características.
            ruta_salida: Destino del scaler de objetivos.

        Raises:
            ErrorSistemaTrafico: Si los scalers no están ajustados.
            OSError: Si no se pueden escribir los archivos.
        """
        if self.scaler_entrada is None or self.scaler_salida is None:
            raise ErrorSistemaTrafico("No hay scalers ajustados que guardar.")
        for scaler, ruta in (
            (self.scaler_entrada, ruta_entrada),
            (self.scaler_salida, ruta_salida),
        ):
            destino = Path(ruta)
            destino.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, destino)
        registro_log.info("Scalers guardados en '%s' y '%s'.", ruta_entrada, ruta_salida)

    def cargar_scalers(
        self,
        ruta_entrada: Path | str = RUTA_SCALER_ENTRADA,
        ruta_salida: Path | str = RUTA_SCALER_SALIDA,
    ) -> None:
        """Carga los scalers persistidos por ``guardar_scalers``.

        Solo deben cargarse artefactos generados por este propio sistema:
        ``joblib`` deserializa con ``pickle``.

        Args:
            ruta_entrada: Archivo del scaler de características.
            ruta_salida: Archivo del scaler de objetivos.

        Raises:
            FileNotFoundError: Si algún archivo no existe.
            ErrorSistemaTrafico: Si el contenido no es un ``MinMaxScaler``
                compatible con las columnas configuradas.
        """
        cargados: list[MinMaxScaler] = []
        for ruta, n_columnas in (
            (ruta_entrada, len(COLUMNAS_CARACTERISTICAS)),
            (ruta_salida, len(COLUMNAS_OBJETIVO)),
        ):
            objeto = joblib.load(Path(ruta))
            if not isinstance(objeto, MinMaxScaler) or objeto.n_features_in_ != n_columnas:
                raise ErrorSistemaTrafico(
                    f"'{ruta}' no contiene un MinMaxScaler ajustado para {n_columnas} columnas.",
                    detalles={"ruta": str(ruta)},
                )
            cargados.append(objeto)
        self.scaler_entrada, self.scaler_salida = cargados
        registro_log.info("Scalers cargados desde '%s' y '%s'.", ruta_entrada, ruta_salida)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _matriz_caracteristicas(datos: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Construye la matriz de características en el orden de ``COLUMNAS_CARACTERISTICAS``.

        Args:
            datos: Serie saneada con columnas de calendario y objetivos.

        Returns:
            Matriz ``(n_registros, n_caracteristicas)`` en ``float64``.
        """
        codificados = codificar_variables_ciclicas(datos)
        return codificados.loc[:, list(COLUMNAS_CARACTERISTICAS)].to_numpy(dtype=float)

    def _exigir_datos_saneados(self, datos: pd.DataFrame) -> None:
        """Verifica que los datos ya hayan pasado por el ``ValidadorEntrada``.

        Args:
            datos: Serie a verificar.

        Raises:
            ErrorValidacionEntrada: Si faltan columnas o quedan tiempos corruptos.
        """
        faltantes = [c for c in COLUMNAS_ESQUEMA_ENTRADA if c not in datos.columns]
        if faltantes:
            raise ErrorValidacionEntrada(
                f"Faltan columnas para construir los conjuntos: {faltantes}",
                detalles={"columnas_faltantes": faltantes},
            )
        corruptos = {
            columna: int(mascara.sum())
            for columna, mascara in self.validador.mascaras_corruptos(datos).items()
            if mascara.any()
        }
        if corruptos:
            raise ErrorValidacionEntrada(
                "Los datos contienen tiempos corruptos sin sanear; páselos antes por "
                "ValidadorEntrada (cargar_historico / validar_serie).",
                detalles={"corruptos_por_columna": corruptos},
            )

    def _exigir_scaler_entrada(self) -> MinMaxScaler:
        """Devuelve el scaler de entrada o falla si no está disponible.

        Returns:
            Scaler de características ajustado.

        Raises:
            ErrorSistemaTrafico: Si no se ha ajustado ni cargado.
        """
        if self.scaler_entrada is None:
            raise ErrorSistemaTrafico(
                "El scaler de entrada no está ajustado: ejecute construir_conjuntos() o "
                "cargar_scalers()."
            )
        return self.scaler_entrada


def _campana(hora: npt.NDArray[np.float64], centro: float) -> npt.NDArray[np.float64]:
    """Campana gaussiana normalizada a ``1`` en ``centro`` con ancho ``SIGMA_PICO_H``.

    Args:
        hora: Horas decimales.
        centro: Hora del máximo del pico.

    Returns:
        Intensidad relativa del pico en ``[0, 1]``.
    """
    return np.exp(-0.5 * ((hora - centro) / SIGMA_PICO_H) ** 2)


def _a_fecha_opcional(valor: pd.Timestamp | str | None) -> pd.Timestamp | None:
    """Normaliza un límite de fecha opcional a ``pd.Timestamp``.

    Args:
        valor: Fecha en texto, ``Timestamp`` o ``None``.

    Returns:
        ``pd.Timestamp`` o ``None``.
    """
    return None if valor is None else pd.Timestamp(valor)
