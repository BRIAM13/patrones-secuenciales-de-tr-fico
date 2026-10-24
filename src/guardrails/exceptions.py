"""Jerarquía de excepciones del dominio y de MLOps.

Regla 3.2.1 de ``CONTEXT.md``: prohibido el fallo silencioso. Toda anomalía se
expresa con una de estas excepciones, que transportan un diccionario
``detalles`` serializable a JSON para que el ``AgenteAuditor`` pueda
registrarla de forma estructurada.

Jerarquía::

    ErrorSistemaTrafico
    ├── ErrorValidacionEntrada
    │   └── ErrorIntegridadSecuencial
    ├── ErrorExtrapolacionSalida
    ├── ErrorContratoSalida
    ├── AlertaDataDrift
    └── ErrorBaseDatos
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ErrorSistemaTrafico(Exception):
    """Excepción base de todo el sistema de predicción de tráfico.

    Attributes:
        mensaje: Descripción legible del problema.
        detalles: Contexto estructurado (serializable a JSON) para auditoría.
    """

    def __init__(self, mensaje: str, *, detalles: Mapping[str, Any] | None = None) -> None:
        """Construye la excepción con su contexto de auditoría.

        Args:
            mensaje: Descripción legible del problema.
            detalles: Contexto estructurado opcional (columnas, conteos,
                fechas, umbrales, etc.).
        """
        super().__init__(mensaje)
        self.mensaje: str = mensaje
        self.detalles: dict[str, Any] = dict(detalles) if detalles else {}

    @property
    def codigo(self) -> str:
        """Código estable del error para auditoría (nombre de la clase)."""
        return type(self).__name__

    def a_diccionario(self) -> dict[str, Any]:
        """Serializa la excepción para el registro de auditoría.

        Returns:
            Diccionario con ``codigo``, ``mensaje`` y ``detalles``.
        """
        return {"codigo": self.codigo, "mensaje": self.mensaje, "detalles": self.detalles}


class ErrorValidacionEntrada(ErrorSistemaTrafico):
    """Los datos de entrada violan el esquema o las reglas duras de entrada.

    Se lanza, entre otros casos, cuando faltan columnas, los tipos son
    incorrectos o la fracción de datos corruptos de una ventana supera el
    umbral permitido (15 %).
    """


class ErrorIntegridadSecuencial(ErrorValidacionEntrada):
    """La serie temporal no es continua o no tiene la longitud esperada.

    Ejemplos: marcas de tiempo duplicadas o desordenadas, huecos en la rejilla
    horaria, registros no alineados a la hora o una ventana distinta de
    ``14`` días.
    """


class ErrorExtrapolacionSalida(ErrorSistemaTrafico):
    """La predicción no es acotable al rango físico de salida.

    Se lanza cuando la red produce valores no finitos (``NaN``/infinito) o con
    una forma incompatible; los valores finitos fuera de rango se recortan y
    solo activan la bandera ``ALERTA_EXTRAPOLACION``.
    """


class ErrorContratoSalida(ErrorSistemaTrafico):
    """La recomendación viola el contrato JSON estricto de salida.

    Incluye el caso en que se pretende emitir en modo ``RED_LSTM`` una
    inferencia que, por confianza o MAE, debía conmutar a fallback.
    """


class AlertaDataDrift(ErrorSistemaTrafico):
    """La distribución reciente de los datos difiere de la de referencia."""


class ErrorBaseDatos(ErrorSistemaTrafico):
    """Error de persistencia: datos estructuralmente inválidos o fallo de SQLite.

    Envuelve ``sqlite3.Error`` y ``OSError`` (escritura del log JSONL) para que
    las capas superiores manejen un único tipo de excepción de persistencia y
    lo registren mediante el ``AgenteAuditor``.
    """
