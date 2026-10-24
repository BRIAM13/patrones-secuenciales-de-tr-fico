"""Reglas estrictas de entrada y salida (Guardrails) y excepciones del dominio.

Estado actual: excepciones y ``ValidadorEntrada`` implementados; el
``ValidadorSalida`` (``output_validator.py``) está pendiente.
"""

from src.guardrails.exceptions import (
    AlertaDataDrift,
    ErrorBaseDatos,
    ErrorExtrapolacionSalida,
    ErrorIntegridadSecuencial,
    ErrorSistemaTrafico,
    ErrorValidacionEntrada,
)
from src.guardrails.input_validator import (
    COLUMNAS_ESQUEMA_ENTRADA,
    RegistroTraficoEntrada,
    ReporteValidacion,
    ResultadoValidacion,
    ValidadorEntrada,
)

__all__: list[str] = [
    "COLUMNAS_ESQUEMA_ENTRADA",
    "AlertaDataDrift",
    "ErrorBaseDatos",
    "ErrorExtrapolacionSalida",
    "ErrorIntegridadSecuencial",
    "ErrorSistemaTrafico",
    "ErrorValidacionEntrada",
    "RegistroTraficoEntrada",
    "ReporteValidacion",
    "ResultadoValidacion",
    "ValidadorEntrada",
]
