# Series de tiempo y clientes — Online Retail

Soluciones de las Partes 0–6 de la tarea de Análisis y Ciencia de Datos, explicadas en español y con decisiones de limpieza justificadas.

**Empieza por el [informe en PDF](informe/main.pdf).** Incluye derivaciones, análisis, 12 figuras y las tablas de simulación, pronósticos y RFM. El [LaTeX editable](informe/main.tex) utiliza tablas y cifras generadas por el código.

> El archivo recibido se llama `BI_Tarea4.pdf`, pero su encabezado dice “Tarea 3: Introducción a Series de Tiempo”. Se resolvió ese enunciado completo.

## Ejecutar

Python 3.11 o posterior (verificado con 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python codigo/analisis.py
python codigo/verificar.py
python codigo/generar_informe.py
```

En Windows activa el entorno con `.venv\Scripts\activate`. El análisis parte de `datos/Online Retail.xlsx`, incluido sin modificaciones, y regenera todas las tablas y figuras. También acepta `python codigo/analisis.py --xlsx /ruta/al/archivo.xlsx`.

Para recompilar el PDF, con una distribución LaTeX instalada:

```bash
cd informe
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

No necesitas LaTeX para leer el PDF ni para ejecutar el análisis.

Para repetir también las versiones de dependencias indirectas, usa Python 3.12 e instala `requirements-lock.txt` en lugar de `requirements.txt`. Con Python 3.11 utiliza `requirements.txt`, para que se resuelvan dependencias indirectas compatibles.

## Qué contiene

| Ruta | Contenido |
|---|---|
| `informe/main.pdf` | Soluciones explicadas de todos los incisos |
| `informe/main.tex` | Informe editable, con tablas en `informe/tablas/` |
| `codigo/limpieza.py` | Función reutilizable `limpiar(df)` |
| `codigo/estadistica.py` | ACF, Ljung–Box y simulación de AR(1) |
| `codigo/modelos.py` | B1, B2, M1, M2, S1, S2 y validaciones |
| `codigo/analisis.py` | Ejecuta las Partes 0–6 desde el Excel |
| `codigo/verificar.py` | Controles de cálculos, limpieza, fechas y convergencia |
| `resultados/` | CSV, figuras, cifras y registro de convergencia |
| `datos/` | Excel original y atribución/licencia del dataset |
| `enunciado/BI_Tarea4.pdf` | Documento proporcionado para esta tarea |

Se simulan **5,000 réplicas por escenario** con semillas `202600` a `202605`. La CV aleatoria usa semillas `2026` a `2035`; el desempate RFM usa `2026`. Los errores semanales se calculan juntando todos los días, y cada ajuste SARIMA exige convergencia. El modelo **TRAMPA** usa información futura deliberadamente para el inciso 5.4 y no se evalúa en prueba.

La limpieza conserva devoluciones y ventas sin cliente, reconoce productos `DCGS`/`PADS` y resuelve el pedido anulado `556444`/`C556445` antes de quitar cargos manuales. La bitácora especifica orden, filas y montos. El informe explica estas decisiones y la convención para clientes históricos que solo tienen devoluciones.

Datos: Daqing Chen, [Online Retail, UCI](https://archive.ics.uci.edu/dataset/352/online+retail), [DOI 10.24432/C5BW33](https://doi.org/10.24432/C5BW33), licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Las demás fuentes están enlazadas en el informe. El enunciado conserva la autoría de la institución que lo proporcionó.
