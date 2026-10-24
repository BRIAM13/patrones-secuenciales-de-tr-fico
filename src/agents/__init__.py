"""Agentes especializados del sistema multi-agente de predicción de tráfico.

``AgentePredictorLSTM`` (y sus tipos) se importan de forma diferida: acceder a
ellos carga TensorFlow, mientras que los demás agentes no lo necesitan.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from src.agents.auditor_agent import (
    AgenteAuditor,
    MetricasRendimiento,
    ReporteDrift,
    ResultadoDriftColumna,
    ResumenAuditorias,
)
from src.agents.data_agent import (
    AgenteDatos,
    ConjuntosEntrenamiento,
    VentanaInferencia,
    codificar_variables_ciclicas,
    derivar_columnas_calendario,
)
from src.agents.strategist_agent import (
    AgenteEstrategaRuta,
    OpcionSalida,
    PlanViaje,
    PrediccionHeuristica,
    TendenciaDiaria,
)

if TYPE_CHECKING:
    from src.agents.predictor_agent import (
        AgentePredictorLSTM,
        PrediccionIncertidumbre,
        ResultadoEntrenamiento,
    )

_DIFERIDOS: frozenset[str] = frozenset(
    {"AgentePredictorLSTM", "PrediccionIncertidumbre", "ResultadoEntrenamiento"}
)


def __getattr__(nombre: str) -> Any:
    """Importa bajo demanda los símbolos que dependen de TensorFlow.

    Args:
        nombre: Atributo solicitado del paquete.

    Returns:
        El símbolo de ``src.agents.predictor_agent``.

    Raises:
        AttributeError: Si el atributo no existe en el paquete.
    """
    if nombre in _DIFERIDOS:
        return getattr(importlib.import_module("src.agents.predictor_agent"), nombre)
    raise AttributeError(f"module 'src.agents' has no attribute {nombre!r}")


__all__: list[str] = [
    "AgenteAuditor",
    "AgenteDatos",
    "AgenteEstrategaRuta",
    "AgentePredictorLSTM",
    "ConjuntosEntrenamiento",
    "MetricasRendimiento",
    "OpcionSalida",
    "PlanViaje",
    "PrediccionHeuristica",
    "PrediccionIncertidumbre",
    "ReporteDrift",
    "ResultadoDriftColumna",
    "ResultadoEntrenamiento",
    "ResumenAuditorias",
    "TendenciaDiaria",
    "VentanaInferencia",
    "codificar_variables_ciclicas",
    "derivar_columnas_calendario",
]
