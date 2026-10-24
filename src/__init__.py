"""Sistema Multi-Agente y Red Neuronal LSTM para Predicción de Tráfico Cotidiano.

Paquete raíz del proyecto. Contiene los subpaquetes:

* ``src.config``: constantes del dominio, límites de guardrails, hiperparámetros
  de la LSTM y rutas de directorios.
* ``src.database``: persistencia SQLite (``GestorBaseDatos``).
* ``src.guardrails``: reglas estrictas de entrada y salida.
* ``src.agents``: los 4 agentes especializados.
* ``src.orchestrator``: coordinación del flujo multi-agente y modo fallback.
* ``src.utils``: utilidades de visualización.
"""

__version__: str = "0.1.0"

__all__: list[str] = ["__version__"]
