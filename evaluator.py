import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelEvaluator:
    """
    Clase para evaluar el rendimiento del modelo.
    """

    def __init__(self, model, X_test, y_test):
        """
        Inicializa el evaluador.

        Args:
            model: El modelo de clasificación entrenado.
            X_test (pd.DataFrame): Las características de prueba.
            y_test (pd.Series): La variable objetivo de prueba.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        """
        Calcula e imprime las métricas de rendimiento.
        """
        # Calcular Accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Reporte de Clasificación (Precisión, Recall, F1-score)
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))

        # Matriz de Confusión
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))

    def display_feature_importance(self, top_n=10):
        """
        Muestra las características más importantes del modelo.

        Args:
            top_n (int): El número de características a mostrar.
        """
        print(f"\n--- Importancia de las Variables (Top {top_n}) ---")
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print(feature_importance_df.head(top_n).to_string(index=False))
