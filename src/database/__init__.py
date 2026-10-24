"""Capa de persistencia SQLite del sistema de predicción de tráfico.

Expone el ``GestorBaseDatos`` (tablas ``historico_trafico`` y
``auditoria_inferencias``) y los tipos auxiliares de sus registros.
"""

from src.database.db_manager import (
    ErrorBaseDatos,
    GestorBaseDatos,
    RegistroAuditoria,
    RegistroHistorico,
)

__all__: list[str] = [
    "ErrorBaseDatos",
    "GestorBaseDatos",
    "RegistroAuditoria",
    "RegistroHistorico",
]
