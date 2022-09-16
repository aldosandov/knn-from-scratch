from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from knn import KnnClassifier


def load_data(dataset):
    datasets = {
        "digits": load_digits,
        "iris": load_iris
    }

    data = datasets[dataset]()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


def model_training(X, y, k, metric):
    model = KnnClassifier(n_neighbors=k, metric=metric)
    model.fit(X, y)

    return model