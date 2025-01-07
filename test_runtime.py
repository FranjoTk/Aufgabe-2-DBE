import sys
import time
import unittest
import numpy as np
from main_runtime import train_model, download_and_prepare_data

def print_and_log(*args, **kwargs):
    """Funktion, die nur in die Konsole schreibt."""
    print(*args, **kwargs)


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print_and_log("Setting up class for runtime tests.\n")
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = download_and_prepare_data()
    
    def test_predict_accuracy(self):
        model = train_model(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test)
        print_and_log(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        self.assertGreater(accuracy, 0.70, "Accuracy should be greater than 70%")

    def test_training_time(self):
        start_time = time.time()
        train_model(self.X_train, self.y_train)
        end_time = time.time()
        
        # Neuer Grenzwert für die Laufzeit
        self.assertLess(end_time - start_time, 10, "Training time should be less than 10 seconds")
        print_and_log(f"Training time: {end_time - start_time:.5f} sec\n")

if __name__ == "__main__":
    unittest.main()
