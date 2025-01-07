import sys
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Datei öffnen für Ausgabe
output_file = open("ausgabe.txt", "w")

def print_and_log(*args, **kwargs):
    """Funktion, die sowohl in die Konsole als auch in die Datei schreibt."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# Logger-Decorator
def my_logger(orig_func):
    import logging
    logging.basicConfig(filename=f"{orig_func.__name__}.log", level=logging.INFO)
    
    def wrapper(*args, **kwargs):
        logging.info(f"Ran with args: {args}, kwargs: {kwargs}")
        return orig_func(*args, **kwargs)
    return wrapper

# Timer-Decorator
def my_timer(orig_func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = orig_func(*args, **kwargs)
        end_time = time.time()
        print_and_log(f"{orig_func.__name__} ran in: {end_time - start_time:.5f} sec")
        return result
    return wrapper

@my_logger
@my_timer
def download_and_prepare_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.astype('float32'), mnist.target.astype('int64')
    X /= 255.0

    # Kleinerer Datensatz verwenden
    X_train, y_train = X[:10000], y[:10000]
    X_test, y_test = X[10000:11000], y[10000:11000]

    # Ausgabe mit print_and_log
    print_and_log(f"MNIST: {X.shape}, {y.shape}")
    return (X_train, y_train, X_test, y_test)

@my_logger
@my_timer
def train_model(X, y):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=100, penalty='l2')
    model.fit(X, y)
    y_pred = model.predict(X)

    # Genauigkeit berechnen und ausgeben
    accuracy = np.mean(y_pred == y)
    print_and_log(f"Train Accuracy: {accuracy * 100:.2f}%")
    print_and_log("Train confusion matrix:")
    print_and_log(confusion_matrix(y, y_pred))
    print_and_log("\nClassification report for classifier:")
    print_and_log(classification_report(y, y_pred))

    return model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = download_and_prepare_data()
    model = train_model(X_train, y_train)

# Datei schließen
output_file.close()
