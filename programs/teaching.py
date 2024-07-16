from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from data_loading_preprocessing import load_data, process_data
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
import time


def split_data(data, test_size1=0.1, test_size2=0.2, random_state=42):
    """
    This function splits the data into training, validation, and test sets.

    Parameters:
    data (pd.DataFrame): The data to be split.
    test_size1 (float): The proportion of the data to include in the validation split. Default is 0.1.
    test_size2 (float): The proportion of the training data to include in the test split. Default is 0.2.
    random_state (int): The generator used to split the data into random train and test subsets. Default is 42.

    Returns:
    tuple: Four arrays: the training features, the test features, the training labels, and the test labels.
    """
    X = data['Описание']
    y = data['Код']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size1, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size2, random_state=random_state)
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def build_pipeline(model):
    """
    This function builds a pipeline consisting of a TF-IDF vectorizer and a classifier.

    Parameters:
    model (sklearn.linear_model.SGDClassifier or sklearn.neighbors.KNeighborsClassifier): The classifier to be used in the pipeline.

    Returns:
    sklearn.pipeline.Pipeline: The pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        (f'{model.__class__.__name__}_clf', model)
    ])


def train_models(pipeline, X_train, y_train):
    """
    This function trains the model using the training data.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): The pipeline to be trained.
    X_train (array-like): The training features.
    y_train (array-like): The training labels.

    Returns:
    sklearn.pipeline.Pipeline: The trained pipeline.
    """
    trained_pipelines = []
    for pipeline in pipelines:
        start_time = time.time()
        try:
            trained_pipeline = pipeline.fit(X_train, y_train)
            trained_pipelines.append(trained_pipeline)
            elapsed_time = time.time() - start_time
            print(f"Trained {pipeline[-1].__class__.__name__} model in {elapsed_time:.2f} seconds.")
        except KeyboardInterrupt:
            print(f"\nKeyboard interrupt detected. Stopping training of {pipeline[-1].__class__.__name__} model.")
            break
    return trained_pipelines


def test_models(trained_pipelines, X_valid, y_valid):
    scores = {}
    for pipeline in trained_pipelines:
        model_name = pipeline[-1].__class__.__name__
        start_time = time.time()
        score = cross_val_score(pipeline, X_valid, y_valid, cv=3).mean()
        elapsed_time = time.time() - start_time
        scores[model_name] = score
        print(f"Tested {model_name} model in {elapsed_time:.2f} seconds with score: {score:.4f}")
    return scores


if __name__ == '__main__':
    start_time = time.time()

    file_path = 'C:/Users/Arina/SGDclf_KNclf/pythonProject/data/Base_model_data_500000.xlsx'
    data500000 = load_data(file_path)
    data500000 = process_data(data500000)
    # data300000 = process_data(data500000, n_rows=300001)

    # Split the data
    X_train, X_test, X_valid, y_train, y_test, y_valid = split_data(data500000)

    # Define the models
    sgd_clf = SGDClassifier(random_state=42)
    knb_clf = KNeighborsClassifier(n_neighbors=10)

    # Build the pipelines
    sgd_ppl_clf = build_pipeline(sgd_clf)
    knb_ppl_clf = build_pipeline(knb_clf)

    # Train the models
    pipelines = [sgd_ppl_clf, knb_ppl_clf]
    trained_pipelines = train_models(pipelines, X_train, y_train)

    # Test the models
    scores = test_models(trained_pipelines, X_valid, y_valid)
    print(scores)

    # Save the best model
    best_model = max(trained_pipelines, key=lambda pipeline: cross_val_score(pipeline, X_valid, y_valid, cv=4).mean())
    model_dir = 'program.models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(best_model, os.path.join(model_dir, 'best_model.joblib'))

    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds.")
