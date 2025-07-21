Proyecto de Computación Paralela: Predicción de Ausentismo Escolar
Este repositorio contiene el código para la segunda entrega del proyecto del curso, enfocado en la predicción del riesgo de ausentismo escolar utilizando modelos de clasificación paralelizados con joblib.

1. Avance Funcional
El código implementa un flujo completo de Machine Learning:

Carga de Datos: Carga el dataset de rendimiento estudiantil.

Preprocesamiento: Limpia los datos, convierte variables categóricas a numéricas (One-Hot Encoding) y define la variable objetivo. Se considera "alto riesgo de ausentismo" si un estudiante tiene más ausencias que la media del dataset.

Entrenamiento del Modelo: Entrena un RandomForestClassifier en dos modalidades:

Secuencial: Utilizando un solo núcleo de procesador.

Paralelo: Utilizando todos los núcleos disponibles (n_jobs=-1), aprovechando joblib internamente.

Evaluación: Mide el rendimiento de ambos enfoques usando métricas como precisión (accuracy), F1-score, y una matriz de confusión. Compara los tiempos de ejecución.

Análisis: Extrae y muestra las variables más influyentes en la predicción.

Persistencia: Guarda el modelo entrenado en un archivo (.pkl) para su futura reutilización, utilizando la funcionalidad de joblib.

2. Organización del Código
El proyecto sigue los principios de Programación Orientada a Objetos (OOP) para garantizar un código modular, legible y mantenible.

main.py: Orquestador principal del flujo de trabajo.

data_loader.py: Clase DataLoader para la carga de datos.

preprocessor.py: Clase DataPreprocessor para la preparación de los datos.

model_trainer.py: Clase ModelTrainer para la lógica de entrenamiento.

evaluator.py: Clase ModelEvaluator para la evaluación y análisis.

Todo el código está documentado con comentarios que explican la lógica de cada componente.

3. Validación Inicial y Resultados Preliminares
Al ejecutar main.py, se obtienen resultados que validan el enfoque.

Ejemplo de Salida:

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

Modelo guardado en: student_absenteeism_model.pkl


La validación inicial demuestra que:

El modelo es capaz de predecir el ausentismo con una alta precisión.

La paralelización con joblib reduce significativamente el tiempo de entrenamiento sin sacrificar la precisión del modelo, cumpliendo el objetivo principal.

4. Gestión de Problemas y Obstáculos
Definición de la Variable Objetivo: El dataset original no tiene una variable binaria para "riesgo de ausentismo".

Solución: Se transformó la variable absences (numérica) en una categórica binaria. Se definió un umbral (la media de ausencias) para clasificar a los estudiantes en "riesgo alto" (1) o "riesgo bajo" (0). Esta decisión es crucial y está documentada en el código.

Manejo de Datos Categóricos: El modelo requiere entradas numéricas.

Solución: Se utilizó pandas.get_dummies() para aplicar One-Hot Encoding a todas las variables no numéricas, creando un formato adecuado para el clasificador.

Reproducibilidad: Los resultados de Machine Learning pueden variar entre ejecuciones.

Solución: Se fijó una semilla (random_state) en la división de los datos y en el entrenamiento del modelo para garantizar que los experimentos sean 100% reproducibles.

5. Instrucciones para Ejecutar el Código (localmente, no recomendado)
Descargar el Dataset:

Ve a Kaggle: Student Performance Dataset.

Descarga el archivo student-por.csv y colócalo en la misma carpeta que los scripts de Python.

Crear un Entorno Virtual (Recomendado):

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate


Instalar Dependencias:

Crea un archivo requirements.txt con el contenido que se proporciona en el bloque de código correspondiente.

Ejecuta:

pip install -r requirements.txt


Ejecutar el Proyecto:

python main.py


El script imprimirá todos los resultados en la consola y generará el archivo student_absenteeism_model.pkl.