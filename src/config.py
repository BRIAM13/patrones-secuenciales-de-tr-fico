"""Configuración central del sistema de predicción de tráfico cotidiano.

Única fuente de constantes del proyecto (Regla 3.1.1 de ``CONTEXT.md``):
parámetros del dominio, límites de los guardrails de entrada y salida,
hiperparámetros de la red ``Bidirectional LSTM``, parámetros del generador
sintético, umbrales de *Data Drift* y rutas de directorios/archivos.

Ningún otro módulo debe definir constantes "mágicas"; todos deben importarlas
desde aquí, por ejemplo::

    from src.config import UMBRAL_CONFIANZA_MIN, RUTA_BASE_DATOS
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

# =============================================================================
# 1. Rutas de directorios y archivos
# =============================================================================

DIR_RAIZ_PROYECTO: Final[Path] = Path(__file__).resolve().parent.parent
"""Directorio raíz del repositorio (padre de ``src/``)."""

DIR_DATOS: Final[Path] = DIR_RAIZ_PROYECTO / "data"
DIR_ARTEFACTOS: Final[Path] = DIR_RAIZ_PROYECTO / "artifacts"
DIR_LOGS: Final[Path] = DIR_RAIZ_PROYECTO / "logs"

RUTA_BASE_DATOS: Final[Path] = DIR_DATOS / "trafico_cotidiano.db"
RUTA_LOG_AUDITORIA: Final[Path] = DIR_LOGS / "auditoria_trafico.jsonl"

RUTA_MODELO_LSTM: Final[Path] = DIR_ARTEFACTOS / "modelo_lstm_trafico.keras"
RUTA_SCALER_ENTRADA: Final[Path] = DIR_ARTEFACTOS / "scaler_entrada.pkl"
RUTA_SCALER_SALIDA: Final[Path] = DIR_ARTEFACTOS / "scaler_salida.pkl"
RUTA_METRICAS_ENTRENAMIENTO: Final[Path] = DIR_ARTEFACTOS / "metricas_entrenamiento.json"
RUTA_GRAFICA_APRENDIZAJE: Final[Path] = DIR_ARTEFACTOS / "curvas_aprendizaje.png"
RUTA_GRAFICA_PREDICCION: Final[Path] = DIR_ARTEFACTOS / "prediccion_vs_real.png"

DIRECTORIOS_EJECUCION: Final[tuple[Path, ...]] = (DIR_DATOS, DIR_ARTEFACTOS, DIR_LOGS)
"""Directorios que deben existir antes de ejecutar el pipeline."""

# =============================================================================
# 2. Parámetros del dominio (traslado cotidiano)
# =============================================================================

RUTA_A: Final[str] = "Ruta_A"
"""Vía principal: rápida pero volátil."""

RUTA_B: Final[str] = "Ruta_B"
"""Vía alterna secundaria: más estable."""

RUTAS_DISPONIBLES: Final[tuple[str, ...]] = (RUTA_A, RUTA_B)

TIEMPO_BASE_MIN: Final[float] = 25.0
"""Tiempo de traslado sin tráfico (minutos)."""

TIEMPO_CRITICO_MIN: Final[float] = 70.0
"""Tiempo de traslado en hora punta severa (minutos)."""

TIEMPO_ALERTA_FISICA_MIN: Final[float] = 120.0
"""Límite físico de alerta del tiempo de traslado (minutos)."""

VENTANA_MANANA_INICIO_H: Final[float] = 7.0
VENTANA_MANANA_FIN_H: Final[float] = 9.0
VENTANA_TARDE_INICIO_H: Final[float] = 18.0
VENTANA_TARDE_FIN_H: Final[float] = 20.0

VENTANAS_HORA_PUNTA: Final[tuple[tuple[float, float], ...]] = (
    (VENTANA_MANANA_INICIO_H, VENTANA_MANANA_FIN_H),
    (VENTANA_TARDE_INICIO_H, VENTANA_TARDE_FIN_H),
)
"""Franjas horarias clave ``(inicio, fin)`` en horas decimales."""

HORAS_POR_DIA: Final[int] = 24
DIAS_POR_SEMANA: Final[int] = 7
PERIODO_HORA: Final[float] = 24.0
"""Periodo para la codificación cíclica de ``hora_decimal`` (h ∈ [0, 24))."""

PERIODO_DIA_SEMANA: Final[float] = 7.0
"""Periodo para la codificación cíclica de ``dia_semana`` (d ∈ [0, 7))."""

FRECUENCIA_MUESTREO: Final[str] = "h"
"""Frecuencia de los registros históricos (alias de ``pandas``: horario)."""

DIA_QUINCENA: Final[int] = 15
"""Día de pago de quincena; junto con el último día del mes activa
``es_quincena_o_fin_mes``."""

# =============================================================================
# 3. Ventanas temporales X → y
# =============================================================================

DIAS_VENTANA_ENTRADA: Final[int] = 14
DIAS_VENTANA_SALIDA: Final[int] = 7
PASOS_VENTANA_ENTRADA: Final[int] = DIAS_VENTANA_ENTRADA * HORAS_POR_DIA
"""Longitud de la secuencia de entrada en pasos horarios (336)."""

PASOS_VENTANA_SALIDA: Final[int] = DIAS_VENTANA_SALIDA * HORAS_POR_DIA
"""Longitud de la secuencia de salida en pasos horarios (168)."""

# =============================================================================
# 4. Esquema de datos y características
# =============================================================================

ORIGEN_SINTETICO: Final[str] = "SINTETICO"
ORIGEN_REAL: Final[str] = "REAL"
ORIGENES_DATO_VALIDOS: Final[tuple[str, ...]] = (ORIGEN_SINTETICO, ORIGEN_REAL)

FORMATO_FECHA_HORA: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Formato ISO-8601 (sin zona) usado para persistir ``fecha_hora``."""

FORMATO_HORA_SALIDA: Final[str] = "%H:%M"
"""Formato de ``hora_optima_salida`` en el contrato de salida."""

COLUMNA_FECHA_HORA: Final[str] = "fecha_hora"
COLUMNA_TIEMPO_RUTA_A: Final[str] = "tiempo_ruta_a_min"
COLUMNA_TIEMPO_RUTA_B: Final[str] = "tiempo_ruta_b_min"

COLUMNAS_OBJETIVO: Final[tuple[str, ...]] = (COLUMNA_TIEMPO_RUTA_A, COLUMNA_TIEMPO_RUTA_B)
"""Variables a predecir por la LSTM."""

COLUMNAS_CALENDARIO: Final[tuple[str, ...]] = (
    "dia_semana",
    "hora_decimal",
    "es_fin_de_semana",
    "es_quincena_o_fin_mes",
    "clima_adverso",
)

COLUMNAS_CICLICAS: Final[tuple[str, ...]] = ("hora_sin", "hora_cos", "dia_sin", "dia_cos")
"""Columnas resultantes de la codificación seno/coseno (Regla 3.1.3)."""

COLUMNAS_CARACTERISTICAS: Final[tuple[str, ...]] = (
    *COLUMNAS_CICLICAS,
    "es_fin_de_semana",
    "es_quincena_o_fin_mes",
    "clima_adverso",
    *COLUMNAS_OBJETIVO,
)
"""Características de entrada de la red, en el orden exacto del tensor X."""

# =============================================================================
# 5. Guardrails de entrada (ValidadorEntrada)
# =============================================================================

TIEMPO_MIN_ENTRADA_MIN: Final[float] = 10.0
TIEMPO_MAX_ENTRADA_MIN: Final[float] = 240.0
UMBRAL_DATOS_CORRUPTOS: Final[float] = 0.15
"""Fracción máxima de datos corruptos imputables por interpolación (15 %)."""

METODO_INTERPOLACION: Final[Literal["linear", "time"]] = "linear"
"""Método de interpolación temporal para la imputación controlada."""

COLUMNAS_OBLIGATORIAS_INGESTA: Final[tuple[str, ...]] = (
    COLUMNA_FECHA_HORA,
    COLUMNA_TIEMPO_RUTA_A,
    COLUMNA_TIEMPO_RUTA_B,
)
"""Columnas mínimas que debe traer una fuente externa (CSV / DataFrame)."""

MAX_ERRORES_REPORTADOS: Final[int] = 10
"""Máximo de errores individuales incluidos en el detalle de una excepción."""

ALCANCE_VALIDACION_SERIE: Final[str] = "SERIE"
ALCANCE_VALIDACION_VENTANA: Final[str] = "VENTANA"
ALCANCES_VALIDACION: Final[tuple[str, ...]] = (ALCANCE_VALIDACION_SERIE, ALCANCE_VALIDACION_VENTANA)

ACCION_SIN_CAMBIOS: Final[str] = "SIN_CAMBIOS"
ACCION_IMPUTACION_LINEAL: Final[str] = "IMPUTACION_LINEAL"
ACCIONES_VALIDACION: Final[tuple[str, ...]] = (ACCION_SIN_CAMBIOS, ACCION_IMPUTACION_LINEAL)

# =============================================================================
# 6. Guardrails de salida (ValidadorSalida) y modo fallback
# =============================================================================

TIEMPO_MIN_SALIDA_MIN: Final[float] = 10.0
TIEMPO_MAX_SALIDA_MIN: Final[float] = 180.0
UMBRAL_CONFIANZA_MIN: Final[float] = 0.75
"""Confianza mínima (fracción) para aceptar la inferencia de la red."""

UMBRAL_MAE_MAX_MIN: Final[float] = 12.0
"""MAE de validación máximo (minutos) antes de conmutar a fallback."""

BANDERA_ALERTA_EXTRAPOLACION: Final[str] = "ALERTA_EXTRAPOLACION"

MODO_RED_LSTM: Final[str] = "RED_LSTM"
MODO_FALLBACK_HEURISTICO: Final[str] = "FALLBACK_HEURISTICO"
MODOS_INFERENCIA_VALIDOS: Final[tuple[str, ...]] = (MODO_RED_LSTM, MODO_FALLBACK_HEURISTICO)

# =============================================================================
# 7. Hiperparámetros de la red Bidirectional LSTM (AgentePredictorLSTM)
# =============================================================================

SEMILLA_ALEATORIA: Final[int] = 42

UNIDADES_LSTM: Final[int] = 64
UNIDADES_DENSA_OCULTA: Final[int] = 128
TASA_DROPOUT: Final[float] = 0.25
TASA_APRENDIZAJE: Final[float] = 1e-3
DELTA_HUBER: Final[float] = 1.0

TAMANO_LOTE: Final[int] = 32
EPOCAS_MAXIMAS: Final[int] = 100
PROPORCION_VALIDACION: Final[float] = 0.15
PROPORCION_PRUEBA: Final[float] = 0.15
PASO_VENTANA_DESLIZANTE: Final[int] = HORAS_POR_DIA
"""Desplazamiento (en pasos horarios) entre ventanas consecutivas."""

PACIENCIA_EARLY_STOPPING: Final[int] = 10
PACIENCIA_REDUCE_LR: Final[int] = 5
FACTOR_REDUCE_LR: Final[float] = 0.5
TASA_APRENDIZAJE_MINIMA: Final[float] = 1e-6

RANGO_ESCALADO: Final[tuple[int, int]] = (0, 1)
"""Rango de salida de los ``MinMaxScaler`` de entrada y salida."""

ITERACIONES_MC_DROPOUT: Final[int] = 50
"""Número de pasadas estocásticas para estimar incertidumbre (MC Dropout)."""

# =============================================================================
# 8. Generador de datos sintéticos (AgenteDatos)
# =============================================================================

DIAS_DATOS_SINTETICOS: Final[int] = 365
FECHA_INICIO_SINTETICA: Final[str] = "2025-01-01 00:00:00"

AMPLITUD_PICO_MANANA_MIN: Final[float] = 40.0
AMPLITUD_PICO_TARDE_MIN: Final[float] = 35.0
CENTRO_PICO_MANANA_H: Final[float] = 8.0
CENTRO_PICO_TARDE_H: Final[float] = 19.0
SIGMA_PICO_H: Final[float] = 0.75
"""Desviación estándar (horas) de los picos gaussianos de hora punta."""

FACTOR_FIN_DE_SEMANA: Final[float] = 0.45
"""Multiplicador de la congestión en sábados y domingos."""

FACTOR_QUINCENA: Final[float] = 1.20
"""Multiplicador de la congestión en quincena / fin de mes."""

PROBABILIDAD_CLIMA_ADVERSO: Final[float] = 0.10
"""Probabilidad diaria de un episodio de clima adverso."""

DURACION_MIN_CLIMA_H: Final[int] = 3
DURACION_MAX_CLIMA_H: Final[int] = 9
"""Duración (horas, inclusiva) de un episodio de clima adverso."""

FACTOR_CLIMA_ADVERSO: Final[float] = 1.30

PROBABILIDAD_INCIDENTE_RUTA_A: Final[float] = 0.01
"""Probabilidad horaria de un incidente (choque, obra) en ``Ruta_A``."""

DEMORA_MEDIA_INCIDENTE_MIN: Final[float] = 15.0
"""Media (minutos) de la demora exponencial causada por un incidente."""

DECIMALES_TIEMPO: Final[int] = 2
"""Decimales con los que se redondean los tiempos sintéticos."""

VOLATILIDAD_RUTA_A: Final[float] = 0.18
"""Desviación relativa del ruido de ``Ruta_A`` (más volátil)."""

VOLATILIDAD_RUTA_B: Final[float] = 0.07
"""Desviación relativa del ruido de ``Ruta_B`` (más estable)."""

PENALIZACION_BASE_RUTA_B_MIN: Final[float] = 8.0
"""Sobrecosto fijo (minutos) de la vía alterna frente al tiempo base."""

ATENUACION_PICO_RUTA_B: Final[float] = 0.55
"""Fracción del pico de congestión que sufre ``Ruta_B`` respecto a ``Ruta_A``."""

# =============================================================================
# 9. Estratega de ruta (AgenteEstrategaRuta)
# =============================================================================

DESPLAZAMIENTOS_HORARIO_MIN: Final[tuple[int, ...]] = (-30, -15, 0, 15)
"""Desplazamientos simulados de la hora de salida (minutos)."""

# =============================================================================
# 10. Auditoría y Data Drift (AgenteAuditor)
# =============================================================================

UMBRAL_P_VALOR_KS: Final[float] = 0.05
"""p-valor bajo el cual la prueba Kolmogorov-Smirnov indica drift."""

UMBRAL_Z_SCORE_DRIFT: Final[float] = 3.0
"""|Z| de la media reciente sobre el cual se declara drift."""

DIAS_REFERENCIA_DRIFT: Final[int] = 28
"""Días de historia usados como distribución de referencia del drift."""

ESTADO_AUDITORIA_APROBADO: Final[str] = "APROBADO"
ESTADO_AUDITORIA_CON_ALERTAS: Final[str] = "APROBADO_CON_ALERTAS"
ESTADO_AUDITORIA_RECHAZADO: Final[str] = "RECHAZADO"
ESTADOS_AUDITORIA_VALIDOS: Final[tuple[str, ...]] = (
    ESTADO_AUDITORIA_APROBADO,
    ESTADO_AUDITORIA_CON_ALERTAS,
    ESTADO_AUDITORIA_RECHAZADO,
)

# =============================================================================
# 11. Base de datos SQLite
# =============================================================================

TABLA_HISTORICO: Final[str] = "historico_trafico"
TABLA_AUDITORIA: Final[str] = "auditoria_inferencias"
TIEMPO_ESPERA_BLOQUEO_BD_S: Final[float] = 10.0
"""Segundos que SQLite espera a que se libere un bloqueo antes de fallar."""

TAMANO_LOTE_INSERCION: Final[int] = 5_000
"""Filas por sentencia ``executemany`` en inserciones masivas."""


def asegurar_directorios() -> None:
    """Crea los directorios de ejecución (``data/``, ``artifacts/``, ``logs/``).

    La operación es idempotente: no falla si los directorios ya existen.

    Raises:
        OSError: Si el sistema de archivos impide crear algún directorio
            (permisos insuficientes, disco lleno, etc.).
    """
    for directorio in DIRECTORIOS_EJECUCION:
        directorio.mkdir(parents=True, exist_ok=True)
