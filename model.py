from data import Data
import numpy as np
from functools import lru_cache

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde

from scipy.special import expit
from scipy.optimize import minimize

from modAL.models import ActiveLearner

# Query strategies
from modAL.expected_error import expected_error_reduction
from modAL.uncertainty import uncertainty_sampling

# Modeling strategies
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Model():
  def __init__(self) -> None:
    #self.gpc = None
    self.scaler = StandardScaler()
    self.learner = None
    self.kde = None
    self.config = {
      "query": 'EE', # EE || US
      "model": 'GP', # GP || RF || SV
      "is_2d": True,
    }
    self.cache = {}
    
  def add_sum(self, x):
    return np.concatenate((x, x.sum(axis=1).reshape((-1,1))), axis=1)
    
  def scale(self, x):
    if self.config['is_2d']:
      return self.scaler.transform(self.add_sum(x))
    return self.scaler.transform(x)
  
  def train(self, x, y):
    #kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0)
    print("Training with config:", self.config)
    
    if self.config['is_2d']:
      self.scaler.fit(self.add_sum(x))
    else:
      self.scaler.fit(x)
      
    x_scaled = self.scale(x)
    
    self.kde = gaussian_kde(x.T)
    
    # should try to avoid
    from visualize import Visualisation
    self.candidates = self.scale(Visualisation.generate_triangle(20))
    
    match self.config['query']:
      case 'EE':
        query_strategy = expected_error_reduction
      case 'US':
        query_strategy = uncertainty_sampling
      case _:
        query_strategy = uncertainty_sampling
    
    match self.config['model']:
      case 'GP':
        estimator = GaussianProcessClassifier()
      case 'RF':
        estimator = RandomForestClassifier()
      case 'SV':
        estimator = SVC(probability=True)
      case _:
        estimator = GaussianProcessClassifier()
    
    self.learner = ActiveLearner(
      estimator=estimator,
      query_strategy=query_strategy,
      X_training=x_scaled,
      y_training=y
    )
        
    # if self.mode == 'native':
    #   self.gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=9)
    #   self.gpc.fit(x_scaled, y)
    #   print("Native", self.gpc.kernel_, self.gpc.log_marginal_likelihood(self.gpc.kernel_.theta))
    # else:          
    #   self.gpc = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    #   self.gpc.fit(x_scaled, y*2-1)
    #   print("Custom", self.gpc.kernel_, self.gpc.log_marginal_likelihood(self.gpc.kernel_.theta))
    
  def predict_accuracy(self, x, y):
    y_pred = self.learner.predict(x)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy
  
  def density(self, x):
    return self.kde(x.T) # np.exp(self.kde.score_samples(x))[0]
  
  def predict(self, x):
    x_scaled = self.scale(x)
        
    # if self.mode == 'native':
    #   y_prob = self.gpc.predict_proba(x_scaled)
    #   y_pred = np.argmax(y_prob, axis=1)
    #   y_prob = np.max(y_prob, axis=1)
    # else:
    #   y_pred = self.gpc.predict(x_scaled)
    #   y_prob = expit(y_pred)
    #   y_pred = np.where(y_pred >= 0.5, 1, 0)
    
    y_prob = self.learner.predict_proba(x_scaled)
    y_pred = np.argmax(y_prob, axis=1)
    #y_prob = np.max(y_prob, axis=1)

    return y_pred, y_prob[:,0]
    
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
  
  # def _valuation(self, ray):
  #   point = self.bound_intersect_vec(ray)
  #   value = self.density(np.array([point]))
  #   #print(ray, point, value)
  #   return value*1e7
  
  def best_candidate(self):
    this_config = self.config['query'] + self.config['model']
    if this_config in self.cache:
      return self.cache[this_config]
    
    # chosen = minimize(self._valuation, np.array([1, 1]), bounds=((0,1.0),(0,1.0)))
    chosen = self.learner.query(self.candidates)
    chosen = self.scaler.inverse_transform(chosen[1])[:,:2]
    
    self.cache[this_config] = chosen
    return self.cache[this_config]
    
    # res = minimize(self._valuation, np.array([1, 1, 1]), bounds=((0,1.0),(0,1.0),(0,.01)))
    # print(res)
    # print(res.x)
    # print(self.bound_intersect_vec(res.x))
    # return self.bound_intersect_vec(res.x)
    
if __name__ == "__main__":
  from visualize import Visualisation
  
  data = Data()
  #visualisation = Visualisation()

  x = data.get_x_3d()
  y = data.get_fractured()
   
  model = Model()
  model.train(x, y)
  
  model.best_candidate()
