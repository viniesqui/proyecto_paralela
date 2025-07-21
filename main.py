from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    """
    Función principal para ejecutar el flujo de trabajo del proyecto.
    """
    try:
        data_loader = DataLoader('student-por.csv')  #poner la ruta correcta. 
        student_data = data_loader.load_data()
    except FileNotFoundError as e:
        print(e)
        return

    preprocessor = DataPreprocessor(student_data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(random_state=42)

   #Cambiar al modelo que se quiera usar, esto es solo un ejemplo 
    model_instance = RandomForestClassifier(n_estimators=200, random_state=42)
    
    
    trainer = ModelTrainer(model=model_instance)

    # Secuencial 
    seq_model, seq_time = trainer.train(X_train, y_train, n_jobs=1)
    evaluator_seq = ModelEvaluator(seq_model, X_test, y_test)
    evaluator_seq.evaluate()

    print("\n" + "="*50 + "\n")

    # Paralelo
    par_model, par_time = trainer.train(X_train, y_train, n_jobs=-1)
    evaluator_par = ModelEvaluator(par_model, X_test, y_test)
    evaluator_par.evaluate()


    #comparacion resultados
    print("comparacion resultados")
    if par_time > 0 and seq_time > par_time:
        speedup = seq_time / par_time
        print(f"Speedup {speedup:.2f}x ")
    else:
        print("No mejoro el paralelo")
    
    if hasattr(par_model, 'feature_importances_'):
        evaluator_par.display_feature_importance()

    trainer.save_model("student_absenteeism_model.pkl")


if __name__ == "__main__":
    main()

