import os
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier


def load_models(model_dir):
    """
    Load all joblib models from a directory.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"The directory '{model_dir}' does not exist.")

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError(f"No joblib models found in the directory '{model_dir}'.")

    models = [joblib.load(os.path.join(model_dir, f)) for f in model_files]
    return models

def create_voting_classifier(models, weights=None):
    """
    Create a VotingClassifier from a list of models.
    """
    estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    return voting_clf


def save_model(model, model_path):
    """
    Save a model to a file.
    """
    joblib.dump(model, model_path)


def load_ensemble_model(model_path):
    """
    Load the ensemble model from a file.
    """
    return joblib.load(model_path)


def get_new_product_names():
    """
    Get new product names from user input.
    """
    new_product_names = []
    while True:
        name = input('Enter the name of a new product (or leave empty to finish): ')
        if not name:
            break
        new_product_names.append(name)
    return new_product_names


def save_results_to_excel(results, file_path):
    """
    Save prediction results to an Excel file.
    """
    df = pd.DataFrame(results, columns=['Описание', 'Код'])
    df.to_excel(file_path, index=False)


def predict_from_terminal(ensemble_model):
    """
    Predict product codes based on terminal input.
    """
    new_product_names = get_new_product_names()
    new_product_codes = ensemble_model.predict(new_product_names)
    results = list(zip(new_product_names, new_product_codes))
    for name, code in results:
        print(f'Product: {name}, Code: {code}')


def predict_from_excel(ensemble_model, input_file, output_file):
    """
    Predict product codes based on descriptions from an Excel file.
    """
    data = pd.read_excel(input_file)
    descriptions = data['Описание'].tolist()
    codes = ensemble_model.predict(descriptions)
    results = list(zip(descriptions, codes))
    save_results_to_excel(results, output_file)
    print(f'Results saved to {output_file}')


if __name__ == '__main__':
    model_dir = "models"
    ensemble_model_path = os.path.join(model_dir, "ensemble_model.joblib")

    # Load models and create the ensemble model
    models = load_models(model_dir)
    weights = [1 for _ in models]  # Adjust weights as needed
    voting_clf = create_voting_classifier(models, weights=weights)
    save_model(voting_clf, ensemble_model_path)
    print(f'Ensemble model saved to {ensemble_model_path}')

    # Load the ensemble model for making predictions
    ensemble_model = load_ensemble_model(ensemble_model_path)

    # Predict from terminal or from Excel file
    print("Choose prediction method:")
    print("1. Predict from terminal")
    print("2. Predict from Excel file")
    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        predict_from_terminal(ensemble_model)
    elif choice == '2':
        input_file = input("Enter the path to the input Excel file: ")
        output_file = input("Enter the path to the output Excel file: ")
        predict_from_excel(ensemble_model, input_file, output_file)
    else:
        print("Invalid choice. Please enter 1 or 2.")
