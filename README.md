## Proyecto de Computación Paralela: Predicción de Ausentismo Escolar

### Resumen
Este proyecto predice el riesgo de ausentismo escolar usando modelos de clasificación paralelizados con joblib y RandomForest. El objetivo es comparar el rendimiento entre entrenamiento secuencial y paralelo, mostrando ventajas de la computación paralela en Machine Learning.

---

## Tabla de Contenido
1. [Resumen](#resumen)
2. [Avance Funcional](#avance-funcional)
3. [Organización del Código](#organización-del-código)
4. [Validación Inicial y Resultados](#validación-inicial-y-resultados)
5. [Gestión de Problemas y Obstáculos](#gestión-de-problemas-y-obstáculos)
6. [Requisitos y Dependencias](#requisitos-y-dependencias)
7. [Instrucciones de Ejecución](#instrucciones-de-ejecución)
8. [Enlaces Útiles](#enlaces-útiles)
9. [Contribución](#contribución)
10. [Contacto](#contacto)

---

## Avance Funcional
El código implementa un flujo completo de Machine Learning:

- **Carga de Datos:** Carga el dataset de rendimiento estudiantil.
- **Preprocesamiento:** Limpia los datos, convierte variables categóricas a numéricas (One-Hot Encoding) y define la variable objetivo. Se considera "alto riesgo de ausentismo" si un estudiante tiene más ausencias que la media del dataset.
- **Entrenamiento del Modelo:** Entrena un RandomForestClassifier en dos modalidades:
  - Secuencial: Utilizando un solo núcleo de procesador.
  - Paralelo: Utilizando todos los núcleos disponibles (`n_jobs=-1`), aprovechando joblib internamente.
- **Evaluación:** Mide el rendimiento de ambos enfoques usando métricas como precisión (accuracy), F1-score y matriz de confusión. Compara los tiempos de ejecución.
- **Análisis:** Extrae y muestra las variables más influyentes en la predicción.
- **Persistencia:** Guarda el modelo entrenado en un archivo `.pkl` para su futura reutilización, utilizando joblib.

---

## Organización del Código
El proyecto sigue los principios de Programación Orientada a Objetos (OOP) para garantizar un código modular, legible y mantenible.

- **main.py:** Orquestador principal del flujo de trabajo.
- **data_loader.py:** Clase `DataLoader` para la carga de datos.
- **preprocessor.py:** Clase `DataPreprocessor` para la preparación de los datos.
- **model_trainer.py:** Clase `ModelTrainer` para la lógica de entrenamiento.
- **evaluator.py:** Clase `ModelEvaluator` para la evaluación y análisis.

Todo el código está documentado con comentarios explicativos.

---


## Validación Inicial y Resultados
Al ejecutar `main.py`, se obtienen resultados que validan el enfoque. Ejemplo de salida:

```bash
--- Entrenamiento Secuencial (1 núcleo) ---
Tiempo de entrenamiento: 1.52 segundos.
Accuracy: 0.95
Classification Report:
              precision    recall  f1-score   support
           0       0.96      0.98      0.97       100
           1       0.92      0.85      0.88        28
    accuracy                           0.95       128
   macro avg       0.94      0.91      0.93       128
weighted avg       0.95      0.95      0.95       128

--- Entrenamiento Paralelo (usando todos los núcleos) ---
Tiempo de entrenamiento: 0.58 segundos.
Accuracy: 0.95
Classification Report:
              precision    recall  f1-score   support
           0       0.96      0.98      0.97       100
           1       0.92      0.85      0.88        28
    accuracy                           0.95       128
   macro avg       0.94      0.91      0.93       128
weighted avg       0.95      0.95      0.95       128

--- Comparación de Rendimiento ---
El entrenamiento paralelo fue 2.62x más rápido.

--- Importancia de las Variables ---
1. absences: 0.25
2. G3: 0.18
3. G2: 0.15
...

Modelo guardado en: `student_absenteeism_model.pkl`

La validación inicial demuestra que:
- El modelo predice el ausentismo con alta precisión.
- La paralelización con joblib reduce significativamente el tiempo de entrenamiento sin sacrificar precisión.

---


## Gestión de Problemas y Obstáculos
- **Definición de la Variable Objetivo:** El dataset original no tiene una variable binaria para "riesgo de ausentismo". Se transformó la variable `absences` en una categórica binaria usando la media como umbral.
- **Manejo de Datos Categóricos:** El modelo requiere entradas numéricas. Se utilizó `pandas.get_dummies()` para aplicar One-Hot Encoding.
- **Reproducibilidad:** Se fijó una semilla (`random_state`) en la división de los datos y en el entrenamiento para garantizar experimentos reproducibles.

---


## Requisitos y Dependencias
- Python 3.8+
- pandas
- scikit-learn
- joblib

Instala las dependencias con:
```bash
pip install -r requirements.txt
```

## Instrucciones de Ejecución
1. **Descargar el Dataset:**
   - Ve a [Kaggle: Student Performance Dataset](https://www.kaggle.com/datasets/uciml/student-performance)
   - Descarga el archivo `student-por.csv` y colócalo en la misma carpeta que los scripts de Python.
2. **Crear un Entorno Virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # En Windows
   # source venv/bin/activate  # En Linux/Mac
   ```
3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ejecutar el proyecto:**
   ```bash
   python main.py
   ```
   El script imprimirá los resultados en la consola y generará el archivo `student_absenteeism_model.pkl`.

---

## Enlaces Útiles
- [Dataset en Kaggle](https://www.kaggle.com/datasets/uciml/student-performance)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Documentación de joblib](https://joblib.readthedocs.io/en/latest/)

---

## Contribución
¿Quieres mejorar el proyecto? ¡Las contribuciones son bienvenidas! Abre un issue o envía un pull request.

---

## Contacto
Autor: Daniel Matarrita
Correo: [tu-email@ejemplo.com]
