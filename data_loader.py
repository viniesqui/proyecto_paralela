import pandas as pd
import os


class DataLoader:
    """
    Clase responsable de cargar los datos desde un archivo CSV.
    """

    def __init__(self, file_path):
        """
        Inicializa el DataLoader con la ruta al archivo de datos.

        Args:
            file_path (str): La ruta al archivo .csv.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("FIle not found. Asegurate que sea el csv y la ruta correcta")
        self.file_path = file_path

    def load_data(self):
        """
        Carga los datos en un DataFrame de pandas.

        Returns:
            pd.DataFrame: Un DataFrame con los datos de los estudiantes.
        """
        print("Cargando datos...")
        return pd.read_csv(self.file_path, sep=';')
