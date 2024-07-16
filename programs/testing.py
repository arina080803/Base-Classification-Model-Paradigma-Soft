from teaching import train_models
import pandas as pd
import joblib
import os


def get_new_product_names():
    new_product_names = []
    while True:
        name = input('Enter the name of a new product (or leave empty to finish): ')
        if not name:
            break
        new_product_names.append(name)
    return new_product_names


def save_results_to_excel(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = pd.DataFrame(results, columns=['Описание', 'Код'])
    df.to_excel(file_path, index=False)


if __name__ == '__main__':
    # Load the best model
    best_clf_loaded = joblib.load("program.models/best_model.joblib")

    new_product_names = get_new_product_names()

    new_product_codes = best_clf_loaded.predict(new_product_names)

    # Combine names and codes into a list of results
    results = list(zip(new_product_names, new_product_codes))

    # Save results to an Excel file
    file_path = "results/results.xlsx"
    save_results_to_excel(results, file_path)
