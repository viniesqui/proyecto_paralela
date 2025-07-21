import time
from joblib import dump


class ModelTrainer:
    """
    Clase para entrenar un modelo de clasificación.
    Ahora es agnóstica al modelo específico, siempre que sea compatible
    con la API de scikit-learn (tenga métodos fit, get_params, set_params).
    """

    def __init__(self, model):
        """
        Inicializa el entrenador del modelo.

        Args:
            model: Una instancia de un modelo de clasificación (p. ej., de scikit-learn).
        """
        self.model = model

    def train(self, X_train, y_train, n_jobs=1):
        """
        Entrena el modelo proporcionado.

        Args:
            X_train (pd.DataFrame): Las características de entrenamiento.
            y_train (pd.Series): La variable objetivo de entrenamiento.
            n_jobs (int): El número de trabajos a ejecutar en paralelo.
                         -1 significa usar todos los procesadores disponibles.
                         1 significa ejecutar en modo secuencial.

        Returns:
            tuple: Una tupla con el modelo entrenado y el tiempo de entrenamiento.
        """
        if n_jobs == -1:
            print(
                f"--- Entrenamiento Paralelo (usando todos los núcleos) para {self.model.__class__.__name__} ---")
        else:
            print(
                f"--- Entrenamiento Secuencial ({n_jobs} núcleo) para {self.model.__class__.__name__} ---")

        # Asigna el parámetro n_jobs al modelo si este lo soporta.
        # Esto permite que modelos como RandomForest o GridSearchCV usen joblib.
        try:
            self.model.set_params(n_jobs=n_jobs)
        except ValueError:
            print(
                f"Advertencia: El modelo {self.model.__class__.__name__} no soporta el parámetro 'n_jobs'. Se entrenará secuencialmente.")

        # Medir el tiempo de entrenamiento
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        print(f"Tiempo de entrenamiento: {training_time:.2f} segundos.")

        return self.model, training_time

    def save_model(self, file_path="student_absenteeism_model.pkl"):
        """
        Guarda el modelo entrenado en un archivo usando joblib.

        Args:
            file_path (str): La ruta donde se guardará el modelo.
        """
        if self.model:
            print(f"Modelo guardado en: {file_path}")
            dump(self.model, file_path)
        else:
            print("No hay un modelo entrenado para guardar.")
