"""Reglas estrictas de entrada y salida (Guardrails) y excepciones del dominio."""

from src.guardrails.exceptions import (
    AlertaDataDrift,
    ErrorBaseDatos,
    ErrorContratoSalida,
    ErrorExtrapolacionSalida,
    ErrorIntegridadSecuencial,
    ErrorModeloNoDisponible,
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
from src.guardrails.output_validator import (
    EvaluacionConfianza,
    PrediccionAcotada,
    RecomendacionTrafico,
    ValidadorSalida,
    determinar_estado_auditoria,
    estimar_nivel_confianza,
)

__all__: list[str] = [
    "COLUMNAS_ESQUEMA_ENTRADA",
    "AlertaDataDrift",
    "ErrorBaseDatos",
    "ErrorContratoSalida",
    "ErrorExtrapolacionSalida",
    "ErrorIntegridadSecuencial",
    "ErrorModeloNoDisponible",
    "ErrorSistemaTrafico",
    "ErrorValidacionEntrada",
    "EvaluacionConfianza",
    "PrediccionAcotada",
    "RecomendacionTrafico",
    "RegistroTraficoEntrada",
    "ReporteValidacion",
    "ResultadoValidacion",
    "ValidadorEntrada",
    "ValidadorSalida",
    "determinar_estado_auditoria",
    "estimar_nivel_confianza",
]
