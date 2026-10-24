# CONTEXT.md — Sistema Multi-Agente y Red Neuronal LSTM para Predicción de Tráfico Cotidiano

> **Repositorio:** `https://github.com/BRIAM13/patrones-secuenciales-de-tr-fico`  
> **Propósito de este documento:** Servir como fuente única de verdad (Single Source of Truth) y puente de contexto arquitectónico entre sesiones de desarrollo asistidas por IA (Claude Web / Gemini).

---

## 1. Resumen del Proyecto y Stack Tecnológico

### Descripción General
Sistema inteligente de grado producción (MLOps) diseñado para predecir la tendencia futura del tiempo de traslado cotidiano (hacia el trabajo o estudio) y recomendar la ruta y hora de salida óptimas. El núcleo predictivo utiliza una **Red Neuronal Recurrente Bidireccional (LSTM)** que analiza ventanas históricas de los últimos $14$ días para proyectar el comportamiento de los próximos $7$ días sobre dos rutas competitivas (`Ruta_A`: vía principal rápida pero volátil; `Ruta_B`: vía alterna secundaria más estable).

El sistema opera bajo una **Arquitectura de 4 Agentes Especializados** protegidos por **Reglas Estrictas de Entrada y Salida (Guardrails)**, un **Motor de Auditoría Continua** con detección de *Data Drift*, persistencia en base de datos local y un **Modo Fallback Heurístico** en caso de baja confianza neuronal.

### Parámetros del Dominio
* **Tiempo base (sin tráfico):** $25\text{ minutos}$.
* **Tiempo crítico (hora punta severa):** $70\text{ minutos}$ (límite físico de alerta: $120\text{ minutos}$).
* **Ventanas horarias clave:** Mañanas ($07:00 - 09:00$) y Tardes ($18:00 - 20:00$).
* **Ventana temporal ($X \rightarrow y$):** Secuencia de entrada de $14\text{ días}$ $\rightarrow$ Secuencia de salida de $7\text{ días}$.

### Stack Tecnológico
* **Lenguaje:** Python 3.10+ (con tipado estático estricto `typing`).
* **Deep Learning:** `TensorFlow 2.x` / `Keras` (Arquitectura `Bidirectional LSTM`, `Dropout`, pérdida `Huber`).
* **Preprocesamiento y Métricas:** `Scikit-Learn` (`MinMaxScaler`, evaluación $MAE$ y $RMSE$), `NumPy`, `Pandas`.
* **Contratos de Datos y Guardrails:** `Pydantic v2` (validación estricta de esquemas de entrada y salida).
* **Base de Datos y Persistencia:** `SQLite3` (almacenamiento de series históricas reales/sintéticas y registro transaccional de auditorías) + archivos `.jsonl` para trazabilidad inmutable.
* **Visualización:** `Matplotlib` (curvas de aprendizaje y comparativas de tendencias).
* **Testing:** `Pytest` (pruebas unitarias de guardrails, agentes y base de datos).

---

## 2. Estructura Exacta de Carpetas y Archivos

```text
patrones-secuenciales-de-tr-fico/
│
├── CONTEXT.md                       # Este documento de contexto y progreso
├── README.md                        # Documentación de uso e instalación para el usuario
├── requirements.txt                 # Dependencias del proyecto con versiones fijas
├── .gitignore                       # Archivos excluidos (entornos virtuales, __pycache__, etc.)
├── main.py                          # Punto de entrada principal (CLI y ejecución del pipeline)
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Hiperparámetros, rutas de archivos y constantes del dominio
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── db_manager.py            # Gestor SQLite (tablas: historico_trafico, auditoria_inferencias)
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── exceptions.py            # Excepciones personalizadas del dominio y MLOps
│   │   ├── input_validator.py       # Reglas de entrada (Pydantic, límites físicos, continuidad)
│   │   └── output_validator.py      # Reglas de salida (Clipping, umbral de confianza, contrato JSON)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── data_agent.py            # AgenteDatos: Generador sintético, codificación cíclica y escalado
│   │   ├── predictor_agent.py       # AgentePredictorLSTM: Red neuronal Keras e incertidumbre MC Dropout
│   │   ├── strategist_agent.py      # AgenteEstrategaRuta: Optimización de ruta y ventana de salida
│   │   └── auditor_agent.py         # AgenteAuditor: Trazabilidad, métricas (MAE/RMSE) y Data Drift
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── traffic_orchestrator.py  # OrquestadorTrafico: Coordina el flujo entre agentes y fallback
│   │
│   └── utils/
│       ├── __init__.py
│       └── visualizer.py            # Generación de gráficas de entrenamiento y predicción
│
├── data/                            # Datos locales y base de datos SQLite (creados en ejecución)
│   └── .gitkeep
│
├── artifacts/                       # Modelos entrenados (.keras), scalers (.pkl) y gráficas (.png)
│   └── .gitkeep
│
├── logs/                            # Archivos de auditoría inmutable (auditoria_trafico.jsonl)
│   └── .gitkeep
│
└── tests/                           # Suite de pruebas automatizadas
    ├── __init__.py
    ├── test_guardrails.py
    └── test_agents.py
```

---

## 3. Reglas de Código y Arquitectura

### 3.1. Convenciones de Nombres y Estilo
1. **Idioma del Código:** Nombres de clases, métodos, variables y docstrings en **español claro y técnico** (alineado con el diseño del dominio), siguiendo estrictamente **PEP 8**.
   * *Clases:* `PascalCase` (ej. `ValidadorEntrada`, `AgentePredictorLSTM`, `OrquestadorTrafico`).
   * *Funciones, métodos y variables:* `snake_case` (ej. `generar_datos_sinteticos`, `tiempo_ruta_a_min`).
   * *Constantes:* `UPPER_SNAKE_CASE` definidas exclusivamente en `src/config.py` (ej. `UMBRAL_CONFIANZA_MIN = 0.75`).
2. **Tipado Estático y Documentación:** Todas las funciones y métodos deben incluir `Type Hints` completos (argumentos y retorno) y docstrings en formato Google/NumPy explicando parámetros, salidas y excepciones.
3. **Transformación Cíclica Temporal:** Las variables `hora_decimal` ($h \in [0, 24)$) y `dia_semana` ($d \in [0, 7)$) deben codificarse mediante senos y cosenos antes de entrar a la red:
   $$\text{hora\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hora\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

### 3.2. Manejo de Errores, Guardrails y Resiliencia
1. **Prohibido el fallo silencioso:** Nunca usar bloques `except Exception: pass`. Toda anomalía debe capturarse con las excepciones de `src/guardrails/exceptions.py` (`ErrorValidacionEntrada`, `ErrorIntegridadSecuencial`, `ErrorExtrapolacionSalida`, `AlertaDataDrift`) y registrarse mediante el `AgenteAuditor`.
2. **Reglas Duras de Entrada (`ValidadorEntrada`):**
   * Rechazar o imputar tiempos menores a $10.0\text{ min}$ o mayores a $240.0\text{ min}$.
   * Verificar ausencia de `NaN` o valores infinitos. Si los datos corruptos en una ventana son $\le 15\%$, aplicar interpolación lineal temporal y auditar; si superan el $15\%$, abortar ejecución con `ErrorValidacionEntrada`.
3. **Reglas Duras de Salida (`ValidadorSalida`) y Modo Fallback:**
   * **Clipping de Seguridad:** Acotar cualquier predicción al rango físico $[10.0, 180.0]\text{ minutos}$ y activar la bandera `ALERTA_EXTRAPOLACION` si hubo recorte.
   * **Protocolo Fallback:** Si el nivel de confianza estimado es $< 75\%$ o el error esperado ($MAE$ de validación) es $> 12.0\text{ minutos}$, el `OrquestadorTrafico` debe conmutar automáticamente de `"RED_LSTM"` a `"FALLBACK_HEURISTICO"` (mediana histórica por día de semana y franja horaria).

### 3.3. Base de Datos y Persistencia (`SQLite3` + `JSONL`)
1. **Motor de Base de Datos:** Usar `sqlite3` nativo en `data/trafico_cotidiano.db` gestionado por `GestorBaseDatos` (`src/database/db_manager.py`) mediante context managers (`with`) para garantizar cierres seguros y transacciones atómicas.
2. **Tablas obligatorias:**
   * `historico_trafico`: Almacena cada registro horario (`id`, `fecha_hora` UNIQUE, `dia_semana`, `hora_decimal`, `es_fin_de_semana`, `es_quincena_o_fin_mes`, `clima_adverso`, `tiempo_ruta_a_min`, `tiempo_ruta_b_min`, `origen_dato` [`'SINTETICO'` | `'REAL'`]).
   * `auditoria_inferencias`: Registra cada consulta ejecutada (`id_transaccion` PK UUID, `timestamp_consulta`, `ruta_recomendada`, `tiempo_estimado_min`, `ahorro_estimado_min`, `hora_optima_salida`, `nivel_confianza_pct`, `modo_inferencia`, `estado_auditoria`, `mae_modelo`, `alerta_drift`).
3. **Log Inmutable:** Cada registro en `auditoria_inferencias` debe escribirse simultáneamente como una línea JSON en `logs/auditoria_trafico.jsonl`.

---

## 4. Roadmap de Desarrollo por Fases

> **Instrucción de Flujo de Trabajo:** Al completar cada fase, se debe marcar el checkbox correspondiente `[x]` en este archivo `CONTEXT.md` y actualizar la sección 5 antes de hacer commit.

### Fase 1: Cimientos, Configuración y Persistencia SQLite
- [x] Crear `requirements.txt` y `.gitignore`.
- [x] Crear `src/__init__.py` y `src/config.py` (con todos los parámetros de la ruta, límites de guardrails, hiperparámetros de la LSTM y rutas de directorios).
- [x] Crear `src/database/__init__.py` y `src/database/db_manager.py` (creación de tablas `historico_trafico` y `auditoria_inferencias`, inserción por lotes y consultas históricas).

### Fase 2: Contratos de Datos y Guardrails (Entrada y Salida)
- [x] Crear `src/guardrails/__init__.py` y `src/guardrails/exceptions.py` (jerarquía de excepciones personalizadas).
- [x] Crear `src/guardrails/input_validator.py` (clase `ValidadorEntrada` con esquemas Pydantic, verificación de límites $[10, 240]\text{ min}$, continuidad de ventana de $14\text{ días}$ e imputación controlada).
- [x] Crear `src/guardrails/output_validator.py` (clase `ValidadorSalida` con clipping $[10, 180]\text{ min}$, evaluación de umbral de confianza del $75\%$ y formato JSON de contrato estricto).

### Fase 3: Agente de Datos y Agente Auditor (MLOps)
- [x] Crear `src/agents/__init__.py`.
- [x] Crear `src/agents/data_agent.py` (clase `AgenteDatos`: generador sintético de $1\text{ año}$ realista con picos gaussianos en horas punta y efecto quincena, persistencia en SQLite, codificación cíclica seno/coseno, `MinMaxScaler` y generador de ventanas deslizantes $14 \rightarrow 7$ días sin *data leakage*).
- [x] Crear `src/agents/auditor_agent.py` (clase `AgenteAuditor`: registro en SQLite y `logs/auditoria_trafico.jsonl`, cálculo de $MAE$, $RMSE$, sesgo y prueba estadística Kolmogorov-Smirnov / Z-score para detección de *Data Drift*).

### Fase 4: Núcleo Neuronal (`AgentePredictorLSTM`) y Estratega de Ruta
- [x] Crear `src/agents/predictor_agent.py` (clase `AgentePredictorLSTM` en TensorFlow/Keras: arquitectura `Bidirectional LSTM` + `Dropout(0.25)` + `Dense`, compilación con `Adam` y pérdida `Huber`, callbacks `EarlyStopping` y `ReduceLROnPlateau`, guardado/carga de pesos en `artifacts/` y estimación de incertidumbre mediante *Monte Carlo Dropout*).
- [x] Crear `src/agents/strategist_agent.py` (clase `AgenteEstrategaRuta`: comparación entre `Ruta_A` y `Ruta_B`, simulación de desplazamientos de horario $[-30, -15, 0, +15]\text{ min}$ y motor de cálculo para el modo `"FALLBACK_HEURISTICO"`).

### Fase 5: Orquestador, Visualización, CLI y Pruebas
- [ ] Crear `src/orchestrator/__init__.py` y `src/orchestrator/traffic_orchestrator.py` (clase `OrquestadorTrafico` que conecta los 4 agentes y los Guardrails de punta a punta).
- [ ] Crear `src/utils/__init__.py` y `src/utils/visualizer.py` (gráficas de pérdida/MAE y comparativa de tendencia predicha vs. real en `artifacts/`).
- [ ] Crear `main.py` (pipeline ejecutable completo: inicialización de BD $\rightarrow$ generación/validación de datos $\rightarrow$ entrenamiento/carga LSTM $\rightarrow$ inferencia multi-agente $\rightarrow$ auditoría y gráficas) y `README.md`.
- [ ] Crear `tests/test_guardrails.py` y `tests/test_agents.py`.

---

## 5. Estado Actual y Siguiente Paso Exacto para Claude

* **Estado Actual del Repositorio (`BRIAM13/patrones-secuenciales-de-tr-fico`):** **Fases 1 a 4 completadas.** Implementados: configuración y persistencia SQLite, guardrails de entrada y salida, y los 4 agentes (`AgenteDatos`, `AgenteAuditor`, `AgentePredictorLSTM`, `AgenteEstrategaRuta`), con 141 pruebas en `tests/test_guardrails.py` / `tests/test_agents.py` (≈ 11 s; las del predictor usan una red diminuta entrenada una vez por módulo).
* **Notas de implementación relevantes para la Fase 5:**
  * **Excepciones:** base `ErrorSistemaTrafico(mensaje, *, detalles)` con `codigo` y `a_diccionario()`. Se añadieron `ErrorContratoSalida` (violaciones del contrato JSON) y `ErrorModeloNoDisponible` (no hay modelo entrenado/cargable → motivo `MODELO_NO_DISPONIBLE`). `ErrorExtrapolacionSalida` se reserva para predicciones **no acotables** (no finitas o con forma incorrecta); los valores finitos fuera de rango solo se recortan y activan `ALERTA_EXTRAPOLACION`.
  * **`ValidadorEntrada`:** esquema Pydantic estricto, continuidad horaria (ventana = 336 pasos) y regla del 15 % evaluada en **cada ventana deslizante de 14 días**; imputación lineal o aborto. Devuelve `ResultadoValidacion(datos, reporte)`.
  * **`ValidadorSalida`:** `acotar_prediccion(pred, fechas_objetivo)` acepta `(168, 2)` o `(1, 168, 2)` y devuelve `PrediccionAcotada`; `evaluar_confianza(confianza_fraccion, mae_validacion)` devuelve `EvaluacionConfianza` (modo sugerido + motivos `CONFIANZA_BAJA` / `MAE_ALTO` / `MAE_DESCONOCIDO`; además existen `MODELO_NO_DISPONIBLE` y `PREDICCION_INVALIDA` para el orquestador); `construir_recomendacion(**campos)` → `RecomendacionTrafico`; `serializar` / `validar_json`.
  * **Contrato `RecomendacionTrafico`** (Pydantic estricto, inmutable): impone el protocolo fallback (es imposible emitir `RED_LSTM` con confianza < 75 %, MAE > 12 min o MAE desconocido; `FALLBACK_HEURISTICO` exige motivos). `estado_auditoria` (`APROBADO` / `APROBADO_CON_ALERTAS`) y `banderas` son **campos calculados**: no se aceptan como entrada. Los `float` deben ser nativos de Python (convertir `np.float32` con `float()`). `a_registro_auditoria()` lo proyecta a la tabla `auditoria_inferencias`.
  * **`AgenteDatos`:** generación sintética, ingesta validada (CSV/DataFrame), `cargar_historico`, `inicializar_datos`, `construir_conjuntos` (sin *data leakage*; `x` → `(n, 336, 9)`, `y` → `(n, 168, 2)`; 235/48/48 ventanas con 1 año), `preparar_ventana_inferencia`, `invertir_escala_salida`, `guardar_scalers` / `cargar_scalers`. Acumula sus reportes en `reportes_validacion`.
  * **`AgenteAuditor`:** `registrar_inferencia(recomendacion)` (SQLite + `logs/auditoria_trafico.jsonl`); eventos MLOps inmutables en `logs/eventos_mlops.jsonl` (`registrar_reportes_validacion`, `registrar_error`, `registrar_metricas`, `registrar_evento`); `calcular_metricas` (MAE, RMSE, sesgo = predicho − real, globales y por ruta); `evaluar_drift(serie)` compara los últimos 14 días con los 28 previos: **drift si KS (horario) p < 0,05 o |Z| (medias diarias) > 3** (≈ 2 % de falsas alarmas medido sobre datos sintéticos estacionarios; detecta desplazamientos del 10 %); `lanzar_alerta=True` lanza `AlertaDataDrift` tras registrar; `resumir_auditorias()`.
  * **`AgentePredictorLSTM(invertir_escala, auditor, ...)`:** recibe `AgenteDatos.invertir_escala_salida` y el `AgenteAuditor`. Arquitectura `Bidirectional(LSTM(64))` → `Dropout(0.25)` → `Dense(128, relu)` → `Dropout(0.25)` → `Dense(336)` → `Reshape(168, 2)` (salida multi-horizonte en una pasada); `Adam` + `Huber`, `EarlyStopping` (restaura los mejores pesos) y `ReduceLROnPlateau`. `entrenar(conjuntos)` → `ResultadoEntrenamiento` (historial y métricas de validación/prueba **en minutos**, registradas por el auditor) y fija `mae_validacion`; `predecir(x)`; `predecir_con_incertidumbre(ventana.x)` → `PrediccionIncertidumbre` (media, σ, intervalo al 95 %, `nivel_confianza`); `guardar` / `cargar` (`artifacts/modelo_lstm_trafico.keras` + metadatos JSON con el MAE; sin metadatos → `mae_validacion=None` → `MAE_DESCONOCIDO`).
  * **Resultados de referencia (1 año sintético, CPU):** 60 épocas (mejor 50) en ≈ 50 s; **MAE validación 3,68 min / prueba 3,54 min** (baseline "repetir la semana anterior": 4,94 min).
  * **Confianza (`estimar_nivel_confianza` en `output_validator.py`):** fracción de horas del horizonte cuyo semiancho al 95 % (1,96·σ) es ≤ máx(10 min, 15 % del tiempo predicho). Calibrada: un modelo con MAE 3,5 min obtiene ≈ 0,96 y uno subentrenado (MAE 7,1 min) ≈ 0,48. Nota: el intervalo MC Dropout cubre ≈ 80 % de los valores reales (subestima la incertidumbre, comportamiento conocido de MC Dropout); se usa como indicador relativo, no como intervalo calibrado.
  * **`AgenteEstrategaRuta(gestor_bd)`:** `planificar_viaje(prediccion_acotada, fecha, "HH:MM", ruta_habitual="Ruta_A")` → `PlanViaje` (simula `[-30, -15, 0, +15]` min en ambas rutas interpolando linealmente entre horas; desempate: menor |desplazamiento| y luego `Ruta_A`; `ahorro` = tiempo de salir a la hora deseada por la ruta habitual − mejor opción); `resumir_tendencia(prediccion)` → `TendenciaDiaria` por día (medias en las franjas punta de mañana y tarde); `generar_prediccion_heuristica(fechas_objetivo, historico=None, semanas=12)` → `PrediccionHeuristica` (mediana por día de semana y hora de las 12 semanas previas al horizonte, sin usar datos futuros, con confianza calculada a partir de la dispersión histórica).
  * La predicción (de la red o heurística) **debe pasar por `ValidadorSalida.acotar_prediccion` antes de `planificar_viaje`**: el contrato rechaza tiempos > 180 min.
  * `src.agents.AgentePredictorLSTM` se importa de forma diferida: `import src.agents` no carga TensorFlow.
  * `ESTADO_AUDITORIA_RECHAZADO` queda reservado (el contrato solo emite los dos estados aprobados).
* **Fase Activa:** **Fase 5 — Orquestador, Visualización, CLI y Pruebas**.

### Instrucción Exacta para Claude Web (Prompt de Inicio)
> *"Hola Claude. Lee el archivo `CONTEXT.md` en la raíz del proyecto `BRIAM13/patrones-secuenciales-de-tr-fico`. Las fases 1 a 4 están completas. Tu tarea en este turno es programar **la Fase 5**: `src/orchestrator/traffic_orchestrator.py` (`OrquestadorTrafico` que conecta los 4 agentes y los guardrails de punta a punta, conmuta a `FALLBACK_HEURISTICO` ante `ErrorModeloNoDisponible`, `ErrorExtrapolacionSalida` o baja confianza/MAE alto, registra todo error con `AgenteAuditor.registrar_error` y devuelve la `RecomendacionTrafico`), `src/utils/visualizer.py` (curvas de pérdida/MAE desde `ResultadoEntrenamiento.historial` y comparativa predicción vs. real con intervalo MC en `artifacts/`), `main.py` (CLI con `argparse`: inicializar BD → datos → entrenar o cargar → inferencia → auditoría y gráficas) y `README.md`, ampliando `tests/`. Importa todas las constantes desde `src/config.py` y respeta la Sección 3. Al terminar, actualiza los checkboxes y la Sección 5 de `CONTEXT.md`."*
