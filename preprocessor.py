import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Clase para preprocesar los datos: limpiar, transformar y dividir.
    """

    def __init__(self, data):
        """
        Inicializa el preprocesador con el DataFrame.

        Args:
            data (pd.DataFrame): El DataFrame a procesar.
        """
        self.data = data.copy()

    def preprocess(self, target_column='absences', test_size=0.2, random_state=42):
        """
        Ejecuta el flujo completo de preprocesamiento.

        Args:
            target_column (str): El nombre de la columna que se usará para crear el objetivo.
            test_size (float): La proporción del dataset a usar para el conjunto de prueba.
            random_state (int): Semilla para la reproducibilidad.

        Returns:
            tuple: Una tupla conteniendo X_train, X_test, y_train, y_test.
        """
        print("Preprocesando datos...")
        self._create_target_variable(target_column)

        # Separar características (X) y objetivo (y)
        X = self.data.drop(columns=[target_column, 'high_absenteeism'])
        y = self.data['high_absenteeism']

        # Aplicar One-Hot Encoding a las variables categóricas
        X = pd.get_dummies(X, drop_first=True)

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(
            f"Datos listos. {len(X_train)} muestras de entrenamiento y {len(X_test)} de prueba.")
        return X_train, X_test, y_train, y_test

    def _create_target_variable(self, column_name):
        """
        Crea la variable objetivo binaria 'high_absenteeism'.
        Se considera "alto ausentismo" (1) si el número de ausencias
        es mayor que la media del dataset. De lo contrario, es "bajo" (0).
        Esta es una decisión de diseño para convertirlo en un problema de clasificación.
        """
        mean_absences = self.data[column_name].mean()
        self.data['high_absenteeism'] = (
            self.data[column_name] > mean_absences).astype(int)
        print(
            f"Variable objetivo 'high_absenteeism' creada. Umbral de media de ausencias: {mean_absences:.2f}")
