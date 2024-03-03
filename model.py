from data import Data
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score
from scipy.special import expit
from scipy.stats import gaussian_kde

from scipy.optimize import minimize


class Model():
  def __init__(self, mode='GPy') -> None:
    self.gpc = None
    self.kde = None
    self.mode = mode
    self.scaler = StandardScaler()
        
  def train(self, x, y):
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0)
    x_scaled = self.scaler.fit_transform(x)
    
    self.kde = gaussian_kde(x.T)
    
    if self.mode == 'native':
      self.gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=9)
      self.gpc.fit(x_scaled, y)
      print("Native", self.gpc.kernel_, self.gpc.log_marginal_likelihood(self.gpc.kernel_.theta))
    else:          
      self.gpc = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
      self.gpc.fit(x_scaled, y*2-1)
      print("Custom", self.gpc.kernel_, self.gpc.log_marginal_likelihood(self.gpc.kernel_.theta))
    
  def predict_accuracy(self, x, y):
    y_pred = self.gpc.predict(x)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy
  
  def density(self, x):
    return self.kde(x.T) # np.exp(self.kde.score_samples(x))[0]
  
  def predict(self, x):
    x_scaled = self.scaler.transform(x)
    
    if self.mode == 'native':
      y_prob = self.gpc.predict_proba(x_scaled)
      y_pred = np.argmax(y_prob, axis=1)
      y_prob = np.max(y_prob, axis=1)
    else:
      y_pred = self.gpc.predict(x_scaled)
      y_prob = expit(y_pred)
      y_pred = np.where(y_pred >= 0.5, 1, 0)

    return y_pred, y_prob
    
  def bound_intersect_vec(self, ray, tolerance=.1, a=0, b=100):
    norm_ray = np.array(ray) / np.sum(ray)
    to_eval = (b+a)/2
        
    pred, _ = self.predict(np.array([norm_ray*to_eval]))
    pred = pred[0]
    
    if b-a > tolerance:
      if pred==0:
        return self.bound_intersect_vec(ray, tolerance=tolerance, a=to_eval, b=b)
      return self.bound_intersect_vec(ray, tolerance=tolerance, a=a, b=to_eval)
    
    return norm_ray*to_eval
  
  def _valuation(self, ray):
    point = self.bound_intersect_vec(ray)
    value = self.density(np.array([point]))
    #print(ray, point, value)
    return value*1e7
    
  def best_candidate(self):
    res = minimize(self._valuation, np.array([1, 1, 1]), bounds=((0,1.0),(0,1.0),(0,.01)))
    print(res)
    print(res.x)
    print(self.bound_intersect_vec(res.x))
    return self.bound_intersect_vec(res.x)
    
if __name__ == "__main__":
  from visualize import Visualisation
  
  data = Data()
  #visualisation = Visualisation()

  x = data.get_x()
  y = data.get_fractured()
   
  model = Model(mode='native')
  model.train(x, y)
  
  model.best_candidate()
