from data import Data
import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score

class Model():
  def __init__(self) -> None:
    self.gpc = None
    
  def scaler(self, data):
    return data / 15 - 0.5
    
  def train(self, x, y):
    print(self.scaler(x))
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    self.gpc = GaussianProcessClassifier(kernel=kernel)
    self.gpc.fit(self.scaler(x), y)
    print("Score", self.gpc.score(self.scaler(x), y))
    
  def predict_accuracy(self, x, y):
    y_pred = self.gpc.predict(self.scaler(x))
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy
  
  def predict_probability(self, x):
    y_probs = self.gpc.predict_proba(self.scaler(x))
    y_std = np.sqrt(y_probs * (1 - y_probs))[:,1]
    y_pred = np.argmax(y_probs, axis=1)
        
    return y_pred, np.max(y_probs,axis=1), y_std
    
if __name__ == "__main__":
  data = Data()

  x = data.get_x()
  y = data.get_fractured()
  
  # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  
  model = Model()
  model.train(x, y)
  y_pred, y_probs, y_std = model.predict_probability([[0,90,0], [90,90,0], [0, 0, 0]])

  print("Prediction:\n", y_pred)
  print("Predicted probabilities:\n", np.round(y_probs*100))
  print("Uncertainty (std deviation):", y_std)

