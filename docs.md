Documentación Técnica del Proyecto
Este documento describe la arquitectura del código, el propósito de cada módulo y el flujo de ejecución del programa para predecir el ausentismo escolar.

1. Estructura y Organización del Código
El proyecto está diseñado siguiendo principios de Programación Orientada a Objetos (OOP) para maximizar la modularidad, legibilidad y mantenibilidad. Cada clase tiene una única responsabilidad bien definida.

main.py: El punto de entrada y orquestador del proyecto. Controla el flujo completo, desde la carga de datos hasta el guardado del modelo final.

data_loader.py: Contiene la clase DataLoader, cuya única función es encontrar y cargar el dataset desde un archivo .csv a un DataFrame de pandas.

preprocessor.py: Contiene la clase DataPreprocessor. Se encarga de toda la lógica de preparación de los datos:

Transforma el problema en uno de clasificación binaria creando la variable objetivo (high_absenteeism).

Convierte variables categóricas a numéricas mediante One-Hot Encoding.

Divide el dataset en conjuntos de entrenamiento y prueba.

model_trainer.py: Contiene la clase ModelTrainer. Es responsable de la lógica de entrenamiento. Es agnóstica al modelo, lo que significa que puede entrenar cualquier clasificador de scikit-learn que se le proporcione. Gestiona la paralelización a través del parámetro n_jobs y mide los tiempos de ejecución.

evaluator.py: Contiene la clase ModelEvaluator. Su propósito es medir el rendimiento de un modelo ya entrenado. Calcula métricas como accuracy, precision, recall, F1-score y la matriz de confusión. También puede extraer y mostrar la importancia de las variables.

requirements.txt: Archivo de texto que lista las dependencias de Python necesarias para ejecutar el proyecto.

README.md: El informe principal del proyecto, orientado a la evaluación y los resultados.

2. Flujo de Ejecución (Paso a Paso)
El flujo de código es secuencial y está controlado por el script main.py. A continuación se detalla cada paso:

Inicio: La ejecución comienza en la función main() de main.py.

Carga de Datos:

Se crea una instancia de DataLoader, pasándole la ruta del archivo student-por.csv.

Se llama al método load_data(), que devuelve un DataFrame de pandas con todos los datos crudos.

Preprocesamiento de Datos:

Se crea una instancia de DataPreprocessor con el DataFrame cargado.

Se invoca al método preprocess(). Internamente, este método:
a.  Calcula la media de ausencias (absences).
b.  Crea una nueva columna binaria high_absenteeism: 1 si las ausencias del estudiante superan la media, 0 en caso contrario. Esta es ahora nuestra variable objetivo.
c.  Separa las características (X) de la variable objetivo (y).
d.  Aplica One-Hot Encoding a X para convertir todas las columnas de texto en numéricas.
e.  Utiliza train_test_split de scikit-learn para dividir X e y en conjuntos de entrenamiento y prueba.

El método devuelve X_train, X_test, y_train, y_test.

Definición y Entrenamiento del Modelo:

En main.py, se crea una instancia del modelo de clasificación deseado (por ejemplo, RandomForestClassifier).

Se crea una instancia de ModelTrainer, pasándole el modelo recién creado.

Se llama al método train() dos veces:

Entrenamiento Secuencial: trainer.train(..., n_jobs=1). El ModelTrainer configura el modelo para usar un solo núcleo y lo entrena, midiendo el tiempo.

Entrenamiento Paralelo: trainer.train(..., n_jobs=-1). El ModelTrainer configura el modelo para usar todos los núcleos disponibles (aprovechando joblib internamente) y repite el entrenamiento, midiendo el tiempo.

Evaluación del Rendimiento:

Después de cada llamada a train(), se crea una instancia de ModelEvaluator con el modelo recién entrenado y los datos de prueba (X_test, y_test).

Se llama al método evaluate() para imprimir el reporte de clasificación y la matriz de confusión.

Análisis y Comparación:

main.py compara los tiempos de ejecución de los entrenamientos secuencial y paralelo para calcular el speedup (la mejora en velocidad).

Se invoca al método display_feature_importance() del evaluador del modelo paralelo para mostrar qué variables fueron más influyentes en la predicción.

Persistencia del Modelo:

Finalmente, se llama al método save_model() del trainer. Este utiliza joblib.dump() para serializar el objeto del modelo entrenado (el último, que fue el paralelo) y guardarlo en el disco como student_absenteeism_model.pkl.

Este flujo garantiza un experimento reproducible y claro, donde el impacto de la paralelización con joblib puede ser medido y analizado de forma directa.