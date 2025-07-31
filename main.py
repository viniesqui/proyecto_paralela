from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main():
    """
    Función principal para ejecutar el flujo de trabajo del proyecto.
    """
    # --- Configuración ---
    FILE_PATH = 'student-por.csv'
    RANDOM_STATE = 42
    
    # Define los modelos que quieres comparar
    models_to_run = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier()
    }

    # --- Carga de Datos ---
    try:
        data_loader = DataLoader(FILE_PATH)
        student_data = data_loader.load_data()
    except FileNotFoundError as e:
        print(e)
        return

    # --- Preprocesamiento ---
    preprocessor = DataPreprocessor(student_data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(random_state=RANDOM_STATE)

    # --- Entrenamiento y Evaluación ---
    for model_name, model_instance in models_to_run.items():
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        trainer = ModelTrainer(model=model_instance)

        # Entrenamiento Secuencial
        seq_model, seq_time = trainer.train(X_train, y_train, n_jobs=1)
        if seq_model:
            evaluator_seq = ModelEvaluator(seq_model, X_test, y_test)
            print("\n--- Resultados Secuenciales ---")
            evaluator_seq.evaluate()
        
        print("\n" + "-"*50 + "\n")

        # Entrenamiento Paralelo
        par_model, par_time = trainer.train(X_train, y_train, n_jobs=-1)
        if par_model:
            evaluator_par = ModelEvaluator(par_model, X_test, y_test)
            print("\n--- Resultados Paralelos ---")
            evaluator_par.evaluate()

            # Comparación de resultados
            if par_time > 0 and seq_time > par_time:
                speedup = seq_time / par_time
                print(f"\nSpeedup: {speedup:.2f}x")
            else:
                print("\nEl entrenamiento paralelo no mejoró el tiempo.")
            
            # Mostrar importancia de características si está disponible
            if hasattr(par_model, 'feature_importances_'):
                evaluator_par.display_feature_importance()

            # Guardar el último modelo paralelo entrenado
            trainer.save_model(f"{model_name}_model.pkl")

if __name__ == "__main__":
    main()

