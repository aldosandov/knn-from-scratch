from utils import model_training, load_data


def make_prediction(dataset, k, metric):
    X_train, X_test, y_train, y_test = load_data(dataset) 
    model = model_training(X_train, y_train, k, metric)

    y_pred = model.predict(X_test)
    score = model.score(y_pred, y_test)
    print("*" * 50)
    print(f'Dataset: {dataset}')
    print(f'Tamaño de set de pruebas: {y_pred.shape[0]} registros')
    print("\nResultados: ")
    print(y_pred)
    print("\nRespectivamente para cada registro.")
    print(f'\nAccuracy de la prediccion: {score*100}%')
    print("*" * 50)
    

if __name__ == '__main__':
    make_prediction("iris", 5, "euclidean")
    make_prediction("digits", 5, "manhattan")