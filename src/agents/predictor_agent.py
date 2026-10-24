"""Agente Predictor LSTM (``AgentePredictorLSTM``): núcleo neuronal del sistema.

Responsabilidades (Fase 4 de ``CONTEXT.md``):

* **Arquitectura:** ``Bidirectional(LSTM)`` → ``Dropout(0.25)`` →
  ``Dense(relu)`` → ``Dropout(0.25)`` → ``Dense`` → ``Reshape(168, 2)``: una
  ventana de ``14`` días (``336 × 9``) produce el horizonte de ``7`` días de
  ambas rutas en una sola pasada (estrategia *multi-output*, sin
  realimentación de errores).
* **Entrenamiento:** ``Adam`` + pérdida ``Huber`` (robusta a los picos de
  incidentes), ``EarlyStopping`` (restaura los mejores pesos) y
  ``ReduceLROnPlateau``. Al terminar, el ``AgenteAuditor`` calcula el ``MAE``
  de validación y prueba **en minutos** (tras invertir el escalado).
* **Persistencia:** modelo ``.keras`` y metadatos JSON (MAE de validación,
  métricas e hiperparámetros) en ``artifacts/``.
* **Incertidumbre:** *Monte Carlo Dropout* (``N`` pasadas con el dropout
  activo); la dispersión se traduce en un nivel de confianza ``[0, 1]`` con
  ``estimar_nivel_confianza``.

TensorFlow se importa al cargar este módulo; ``src.agents`` lo expone de forma
diferida para que los demás agentes no paguen ese coste.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402  (la variable de entorno debe fijarse antes)
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from src.agents.auditor_agent import AgenteAuditor, MetricasRendimiento  # noqa: E402
from src.agents.data_agent import ConjuntosEntrenamiento, MatrizFlotante  # noqa: E402
from src.config import (  # noqa: E402
    COLUMNAS_CARACTERISTICAS,
    COLUMNAS_OBJETIVO,
    DELTA_HUBER,
    EPOCAS_MAXIMAS,
    FACTOR_REDUCE_LR,
    ITERACIONES_MC_DROPOUT,
    PACIENCIA_EARLY_STOPPING,
    PACIENCIA_REDUCE_LR,
    PASOS_VENTANA_ENTRADA,
    PASOS_VENTANA_SALIDA,
    RUTA_METRICAS_ENTRENAMIENTO,
    RUTA_MODELO_LSTM,
    SEMILLA_ALEATORIA,
    TAMANO_LOTE,
    TASA_APRENDIZAJE,
    TASA_APRENDIZAJE_MINIMA,
    TASA_DROPOUT,
    UNIDADES_DENSA_OCULTA,
    UNIDADES_LSTM,
    VERBOSIDAD_ENTRENAMIENTO,
    Z_INTERVALO_CONFIANZA,
)
from src.guardrails.exceptions import (  # noqa: E402
    ErrorModeloNoDisponible,
    ErrorValidacionEntrada,
)
from src.guardrails.output_validator import estimar_nivel_confianza  # noqa: E402

registro_log: logging.Logger = logging.getLogger(__name__)

InversorEscala = Callable[[npt.NDArray[np.floating[Any]]], MatrizFlotante]
"""Función que lleva ``(..., 2)`` del espacio escalado a minutos
(p. ej. ``AgenteDatos.invertir_escala_salida``)."""

_N_RUTAS: Final[int] = len(COLUMNAS_OBJETIVO)


# =============================================================================
# Resultados
# =============================================================================


class ResultadoEntrenamiento(BaseModel):
    """Resumen auditable de un entrenamiento."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epocas_ejecutadas: int = Field(ge=1)
    mejor_epoca: int = Field(ge=1)
    duracion_s: float = Field(ge=0.0)
    historial: dict[str, list[float]]
    metricas_validacion: MetricasRendimiento
    metricas_prueba: MetricasRendimiento

    @property
    def mae_validacion(self) -> float:
        """MAE de validación en minutos (criterio del protocolo fallback)."""
        return self.metricas_validacion.mae


@dataclass(frozen=True)
class PrediccionIncertidumbre:
    """Predicción de una ventana con su incertidumbre MC Dropout (minutos).

    Attributes:
        media: Predicción media ``(168, 2)``.
        desviacion: Desviación estándar entre pasadas ``(168, 2)``.
        limite_inferior: ``media − z·σ``.
        limite_superior: ``media + z·σ``.
        nivel_confianza: Fracción ``[0, 1]`` de horas con incertidumbre acotada.
        iteraciones: Número de pasadas estocásticas.
    """

    media: MatrizFlotante
    desviacion: MatrizFlotante
    limite_inferior: MatrizFlotante
    limite_superior: MatrizFlotante
    nivel_confianza: float
    iteraciones: int


# =============================================================================
# Agente Predictor
# =============================================================================


class AgentePredictorLSTM:
    """Red ``Bidirectional LSTM`` con estimación de incertidumbre MC Dropout.

    Attributes:
        invertir_escala: Conversión de la salida escalada a minutos.
        auditor: Agente que calcula y registra las métricas.
        modelo: Modelo Keras (``None`` hasta construir, entrenar o cargar).
        mae_validacion: MAE de validación en minutos (``None`` si se desconoce).

    Example:
        >>> predictor = AgentePredictorLSTM(agente_datos.invertir_escala_salida, auditor)
        >>> resultado = predictor.entrenar(conjuntos)
        >>> predictor.guardar()
        >>> incertidumbre = predictor.predecir_con_incertidumbre(ventana.x)
    """

    def __init__(
        self,
        invertir_escala: InversorEscala,
        auditor: AgenteAuditor,
        unidades_lstm: int = UNIDADES_LSTM,
        unidades_densa: int = UNIDADES_DENSA_OCULTA,
        tasa_dropout: float = TASA_DROPOUT,
        tasa_aprendizaje: float = TASA_APRENDIZAJE,
        delta_huber: float = DELTA_HUBER,
        semilla: int = SEMILLA_ALEATORIA,
    ) -> None:
        """Configura el agente (el modelo se construye al entrenar o cargar).

        Args:
            invertir_escala: Conversión ``(..., 2)`` escalado → minutos.
            auditor: Agente auditor para métricas y trazabilidad.
            unidades_lstm: Unidades de cada dirección de la LSTM.
            unidades_densa: Unidades de la capa densa oculta.
            tasa_dropout: Tasa de las capas ``Dropout`` (``(0, 1)``; el MC
                Dropout la requiere positiva).
            tasa_aprendizaje: Tasa inicial de ``Adam``.
            delta_huber: Umbral de la pérdida ``Huber``.
            semilla: Semilla global de Keras/NumPy/TensorFlow.

        Raises:
            ValueError: Si algún hiperparámetro está fuera de dominio.
        """
        if unidades_lstm < 1 or unidades_densa < 1:
            raise ValueError("Las unidades de las capas deben ser >= 1.")
        if not 0.0 < tasa_dropout < 1.0:
            raise ValueError(f"'tasa_dropout' debe estar en (0, 1): {tasa_dropout}")
        if tasa_aprendizaje <= 0.0 or delta_huber <= 0.0:
            raise ValueError("'tasa_aprendizaje' y 'delta_huber' deben ser positivos.")
        self.invertir_escala: InversorEscala = invertir_escala
        self.auditor: AgenteAuditor = auditor
        self.unidades_lstm: int = unidades_lstm
        self.unidades_densa: int = unidades_densa
        self.tasa_dropout: float = tasa_dropout
        self.tasa_aprendizaje: float = tasa_aprendizaje
        self.delta_huber: float = delta_huber
        self.semilla: int = semilla
        self.modelo: keras.Model | None = None
        self.mae_validacion: float | None = None

    @property
    def esta_entrenado(self) -> bool:
        """``True`` si hay un modelo listo para predecir."""
        return self.modelo is not None

    def hiperparametros(self) -> dict[str, float | int]:
        """Hiperparámetros del agente (se guardan junto al modelo).

        Returns:
            Diccionario serializable a JSON.
        """
        return {
            "unidades_lstm": self.unidades_lstm,
            "unidades_densa": self.unidades_densa,
            "tasa_dropout": self.tasa_dropout,
            "tasa_aprendizaje": self.tasa_aprendizaje,
            "delta_huber": self.delta_huber,
            "semilla": self.semilla,
        }

    # ------------------------------------------------------------------
    # Construcción y entrenamiento
    # ------------------------------------------------------------------

    def construir_modelo(self) -> keras.Model:
        """Construye y compila la red (reinicia los pesos).

        Returns:
            Modelo compilado con entrada ``(336, 9)`` y salida ``(168, 2)``.
        """
        keras.utils.set_random_seed(self.semilla)
        entrada = keras.Input(
            shape=(PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS)), name="ventana_14_dias"
        )
        x = keras.layers.Bidirectional(keras.layers.LSTM(self.unidades_lstm), name="bilstm")(
            entrada
        )
        x = keras.layers.Dropout(self.tasa_dropout, name="dropout_lstm")(x)
        x = keras.layers.Dense(self.unidades_densa, activation="relu", name="densa_oculta")(x)
        x = keras.layers.Dropout(self.tasa_dropout, name="dropout_densa")(x)
        x = keras.layers.Dense(PASOS_VENTANA_SALIDA * _N_RUTAS, name="salida_plana")(x)
        salida = keras.layers.Reshape((PASOS_VENTANA_SALIDA, _N_RUTAS), name="horizonte_7_dias")(x)

        modelo = keras.Model(entrada, salida, name="bilstm_trafico")
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje),
            loss=keras.losses.Huber(delta=self.delta_huber),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        self.modelo = modelo
        self.mae_validacion = None
        return modelo

    def entrenar(
        self,
        conjuntos: ConjuntosEntrenamiento,
        epocas: int = EPOCAS_MAXIMAS,
        tamano_lote: int = TAMANO_LOTE,
        verbosidad: int = VERBOSIDAD_ENTRENAMIENTO,
    ) -> ResultadoEntrenamiento:
        """Entrena desde cero y evalúa en minutos sobre validación y prueba.

        Args:
            conjuntos: Tensores sin *data leakage* del ``AgenteDatos``.
            epocas: Máximo de épocas (``EarlyStopping`` puede cortar antes).
            tamano_lote: Tamaño de lote.
            verbosidad: ``verbose`` de ``model.fit``.

        Returns:
            Historial, épocas y métricas de validación/prueba (registradas
            por el ``AgenteAuditor``).

        Raises:
            ErrorValidacionEntrada: Si las formas de los conjuntos no coinciden
                con la arquitectura o contienen valores no finitos.
            ValueError: Si ``epocas`` o ``tamano_lote`` no son positivos.
        """
        if epocas < 1 or tamano_lote < 1:
            raise ValueError("'epocas' y 'tamano_lote' deben ser >= 1.")
        for nombre, x, y in (
            ("entrenamiento", conjuntos.x_entrenamiento, conjuntos.y_entrenamiento),
            ("validacion", conjuntos.x_validacion, conjuntos.y_validacion),
            ("prueba", conjuntos.x_prueba, conjuntos.y_prueba),
        ):
            _validar_tensor_entrada(x, nombre)
            _validar_tensor_salida(y, nombre)
            if len(x) != len(y):
                raise ErrorValidacionEntrada(
                    f"'{nombre}': {len(x)} entradas vs {len(y)} objetivos."
                )

        modelo = self.construir_modelo()
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PACIENCIA_EARLY_STOPPING, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=FACTOR_REDUCE_LR,
                patience=PACIENCIA_REDUCE_LR,
                min_lr=TASA_APRENDIZAJE_MINIMA,
            ),
        ]
        inicio = time.perf_counter()
        historia = modelo.fit(
            conjuntos.x_entrenamiento,
            conjuntos.y_entrenamiento,
            validation_data=(conjuntos.x_validacion, conjuntos.y_validacion),
            epochs=epocas,
            batch_size=tamano_lote,
            callbacks=callbacks,
            shuffle=True,
            verbose=verbosidad,
        )
        duracion = time.perf_counter() - inicio

        metricas_validacion = self._evaluar(
            conjuntos.x_validacion, conjuntos.y_validacion, "validacion"
        )
        metricas_prueba = self._evaluar(conjuntos.x_prueba, conjuntos.y_prueba, "prueba")
        self.mae_validacion = metricas_validacion.mae

        historial = {
            clave: [float(valor) for valor in valores]
            for clave, valores in historia.history.items()
        }
        perdidas_val = historial["val_loss"]
        resultado = ResultadoEntrenamiento(
            epocas_ejecutadas=len(perdidas_val),
            mejor_epoca=int(np.argmin(perdidas_val)) + 1,
            duracion_s=round(duracion, 3),
            historial=historial,
            metricas_validacion=metricas_validacion,
            metricas_prueba=metricas_prueba,
        )
        registro_log.info(
            "Entrenamiento: %d época(s) (mejor %d) en %.1f s; MAE val=%.2f min, prueba=%.2f min.",
            resultado.epocas_ejecutadas,
            resultado.mejor_epoca,
            duracion,
            metricas_validacion.mae,
            metricas_prueba.mae,
        )
        return resultado

    def _evaluar(
        self, x: MatrizFlotante, y_escalado: MatrizFlotante, etiqueta: str
    ) -> MetricasRendimiento:
        """Calcula y registra las métricas en minutos de un conjunto.

        Args:
            x: Ventanas de entrada.
            y_escalado: Objetivos escalados.
            etiqueta: Nombre del conjunto.

        Returns:
            Métricas del ``AgenteAuditor``.
        """
        predichos = self.predecir(x)
        reales = self.invertir_escala(y_escalado)
        metricas = self.auditor.calcular_metricas(reales, predichos, etiqueta=etiqueta)
        self.auditor.registrar_metricas(metricas)
        return metricas

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predecir(self, x: npt.ArrayLike) -> MatrizFlotante:
        """Predicción determinista (dropout inactivo) en minutos.

        Args:
            x: Tensor ``(n, 336, 9)`` escalado.

        Returns:
            Arreglo ``(n, 168, 2)`` en minutos.

        Raises:
            ErrorModeloNoDisponible: Si no hay modelo.
            ErrorValidacionEntrada: Si la forma no es válida o hay no finitos.
        """
        modelo = self._exigir_modelo()
        tensor = _validar_tensor_entrada(x, "inferencia")
        escalado = modelo.predict(tensor, verbose=0)
        return self.invertir_escala(np.asarray(escalado, dtype=np.float32))

    def predecir_con_incertidumbre(
        self, x: npt.ArrayLike, iteraciones: int = ITERACIONES_MC_DROPOUT
    ) -> PrediccionIncertidumbre:
        """Predice una ventana con *Monte Carlo Dropout*.

        Se ejecutan ``iteraciones`` pasadas con ``training=True`` (cada copia
        de la ventana recibe su propia máscara de dropout en un único lote) y
        se resumen en minutos.

        Args:
            x: Ventana escalada ``(1, 336, 9)`` o ``(336, 9)``.
            iteraciones: Número de pasadas estocásticas (``>= 2``).

        Returns:
            Media, desviación, intervalo al 95 % y nivel de confianza.

        Raises:
            ErrorModeloNoDisponible: Si no hay modelo.
            ErrorValidacionEntrada: Si la ventana no es válida o no es única.
            ValueError: Si ``iteraciones < 2``.
        """
        if iteraciones < 2:
            raise ValueError(f"'iteraciones' debe ser >= 2, se recibió {iteraciones}")
        modelo = self._exigir_modelo()
        tensor = np.asarray(x, dtype=np.float32)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, ...]
        tensor = _validar_tensor_entrada(tensor, "inferencia")
        if len(tensor) != 1:
            raise ErrorValidacionEntrada(
                f"MC Dropout requiere una única ventana, se recibieron {len(tensor)}."
            )

        repetida = np.repeat(tensor, iteraciones, axis=0)
        muestras_escaladas = keras.ops.convert_to_numpy(modelo(repetida, training=True))
        muestras = self.invertir_escala(np.asarray(muestras_escaladas, dtype=np.float32))
        media = muestras.mean(axis=0).astype(np.float32)
        desviacion = muestras.std(axis=0).astype(np.float32)
        semiancho = (Z_INTERVALO_CONFIANZA * desviacion).astype(np.float32)
        confianza = estimar_nivel_confianza(media, desviacion)
        registro_log.info(
            "MC Dropout (%d pasadas): σ media=%.2f min, confianza=%.1f %%.",
            iteraciones,
            float(desviacion.mean()),
            confianza * 100.0,
        )
        return PrediccionIncertidumbre(
            media=media,
            desviacion=desviacion,
            limite_inferior=media - semiancho,
            limite_superior=media + semiancho,
            nivel_confianza=confianza,
            iteraciones=iteraciones,
        )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def guardar(
        self,
        ruta_modelo: Path | str = RUTA_MODELO_LSTM,
        ruta_metadatos: Path | str = RUTA_METRICAS_ENTRENAMIENTO,
        resultado: ResultadoEntrenamiento | None = None,
    ) -> None:
        """Guarda el modelo (``.keras``) y sus metadatos (JSON).

        Args:
            ruta_modelo: Destino del modelo (debe terminar en ``.keras``).
            ruta_metadatos: Destino de los metadatos.
            resultado: Resultado del entrenamiento a incluir en los metadatos.

        Raises:
            ErrorModeloNoDisponible: Si no hay modelo que guardar.
            ValueError: Si ``ruta_modelo`` no tiene extensión ``.keras``.
            OSError: Si no se pueden escribir los archivos.
        """
        modelo = self._exigir_modelo()
        destino_modelo, destino_metadatos = Path(ruta_modelo), Path(ruta_metadatos)
        if destino_modelo.suffix != ".keras":
            raise ValueError(f"El modelo debe guardarse con extensión .keras: {destino_modelo}")
        destino_modelo.parent.mkdir(parents=True, exist_ok=True)
        destino_metadatos.parent.mkdir(parents=True, exist_ok=True)
        modelo.save(destino_modelo)

        metadatos: dict[str, Any] = {
            "fecha_guardado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mae_validacion": self.mae_validacion,
            "hiperparametros": self.hiperparametros(),
            "forma_entrada": [PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS)],
            "forma_salida": [PASOS_VENTANA_SALIDA, _N_RUTAS],
            "columnas_caracteristicas": list(COLUMNAS_CARACTERISTICAS),
            "columnas_objetivo": list(COLUMNAS_OBJETIVO),
        }
        if resultado is not None:
            metadatos["entrenamiento"] = resultado.model_dump(mode="json")
        destino_metadatos.write_text(
            json.dumps(metadatos, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        registro_log.info(
            "Modelo guardado en '%s' (metadatos: '%s').", destino_modelo, destino_metadatos
        )

    def cargar(
        self,
        ruta_modelo: Path | str = RUTA_MODELO_LSTM,
        ruta_metadatos: Path | str = RUTA_METRICAS_ENTRENAMIENTO,
    ) -> None:
        """Carga un modelo guardado y su MAE de validación.

        Si faltan los metadatos, el modelo se carga con ``mae_validacion=None``
        (el protocolo fallback lo tratará como ``MAE_DESCONOCIDO``).

        Args:
            ruta_modelo: Archivo ``.keras``.
            ruta_metadatos: Archivo JSON de metadatos.

        Raises:
            ErrorModeloNoDisponible: Si el modelo no existe, no se puede
                deserializar o su arquitectura es incompatible.
        """
        origen = Path(ruta_modelo)
        if not origen.is_file():
            raise ErrorModeloNoDisponible(
                f"No existe un modelo entrenado en '{origen}'.", detalles={"ruta": str(origen)}
            )
        try:
            modelo = keras.models.load_model(origen)
        except (OSError, ValueError, TypeError) as error:
            raise ErrorModeloNoDisponible(
                f"No se pudo cargar el modelo '{origen}': {error}", detalles={"ruta": str(origen)}
            ) from error

        entrada_esperada = (None, PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS))
        salida_esperada = (None, PASOS_VENTANA_SALIDA, _N_RUTAS)
        if tuple(modelo.input_shape) != entrada_esperada or tuple(modelo.output_shape) != (
            salida_esperada
        ):
            raise ErrorModeloNoDisponible(
                f"Arquitectura incompatible: entrada {modelo.input_shape}, salida "
                f"{modelo.output_shape}; se esperaba {entrada_esperada} → {salida_esperada}.",
                detalles={"ruta": str(origen)},
            )

        mae: float | None = None
        metadatos = Path(ruta_metadatos)
        if metadatos.is_file():
            try:
                valor = json.loads(metadatos.read_text(encoding="utf-8")).get("mae_validacion")
            except (OSError, json.JSONDecodeError, AttributeError) as error:
                registro_log.warning("Metadatos ilegibles en '%s': %s", metadatos, error)
            else:
                mae = float(valor) if isinstance(valor, (int, float)) else None
        else:
            registro_log.warning("Sin metadatos en '%s': MAE de validación desconocido.", metadatos)

        self.modelo = modelo
        self.mae_validacion = mae
        registro_log.info("Modelo cargado desde '%s' (MAE validación=%s).", origen, mae)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _exigir_modelo(self) -> keras.Model:
        """Devuelve el modelo o falla si no existe.

        Returns:
            Modelo Keras.

        Raises:
            ErrorModeloNoDisponible: Si no se ha entrenado ni cargado.
        """
        if self.modelo is None:
            raise ErrorModeloNoDisponible(
                "El AgentePredictorLSTM no tiene modelo: ejecute entrenar() o cargar()."
            )
        return self.modelo


def _validar_tensor_entrada(x: npt.ArrayLike, nombre: str) -> MatrizFlotante:
    """Verifica la forma ``(n, 336, 9)`` y la finitud de un tensor de entrada.

    Args:
        x: Tensor a validar.
        nombre: Conjunto al que pertenece (para el mensaje).

    Returns:
        Tensor ``float32``.

    Raises:
        ErrorValidacionEntrada: Si la forma es incorrecta, está vacío o hay
            valores no finitos.
    """
    tensor = np.asarray(x, dtype=np.float32)
    forma = (PASOS_VENTANA_ENTRADA, len(COLUMNAS_CARACTERISTICAS))
    if tensor.ndim != 3 or tensor.shape[1:] != forma or len(tensor) == 0:
        raise ErrorValidacionEntrada(
            f"Entrada '{nombre}' con forma {tensor.shape}; "
            f"se esperaba (n, {forma[0]}, {forma[1]}).",
            detalles={"forma": list(tensor.shape)},
        )
    if not np.isfinite(tensor).all():
        raise ErrorValidacionEntrada(f"La entrada '{nombre}' contiene valores no finitos.")
    return tensor


def _validar_tensor_salida(y: npt.ArrayLike, nombre: str) -> MatrizFlotante:
    """Verifica la forma ``(n, 168, 2)`` y la finitud de un tensor objetivo.

    Args:
        y: Tensor a validar.
        nombre: Conjunto al que pertenece (para el mensaje).

    Returns:
        Tensor ``float32``.

    Raises:
        ErrorValidacionEntrada: Si la forma es incorrecta, está vacío o hay
            valores no finitos.
    """
    tensor = np.asarray(y, dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[1:] != (PASOS_VENTANA_SALIDA, _N_RUTAS) or len(tensor) == 0:
        raise ErrorValidacionEntrada(
            f"Objetivo '{nombre}' con forma {tensor.shape}; se esperaba "
            f"(n, {PASOS_VENTANA_SALIDA}, {_N_RUTAS}).",
            detalles={"forma": list(tensor.shape)},
        )
    if not np.isfinite(tensor).all():
        raise ErrorValidacionEntrada(f"El objetivo '{nombre}' contiene valores no finitos.")
    return tensor
