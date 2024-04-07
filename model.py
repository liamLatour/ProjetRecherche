from data import Data
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde

from modAL.models import ActiveLearner

# Query strategies
from modAL.expected_error import expected_error_reduction
from modAL.batch import uncertainty_batch_sampling
from modAL import batch
from my_query_strat import eer_multi, emc_multi, emc_continuous, eer_single, emc_single
from modAL.uncertainty import uncertainty_sampling

# Modeling strategies
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class Model():
  def __init__(self) -> None:
    #self.gpc = None
    self.scaler = StandardScaler()
    self.learner = None
    self.kde = None
    self.config = {
      "query": 'US', # EE || US
      "model": 'GP', # GP || RF || SV
      "add_sum": True,
    }
    self.cache = {}
    
  def add_sum(self, x):
    return np.concatenate((x, x.sum(axis=1).reshape((-1,1))), axis=1)
    
  def scale(self, x):
    if self.config['add_sum']:
      return self.scaler.transform(self.add_sum(x))
    return self.scaler.transform(x)
  
  def train(self, x, y):
    print("Training with config:", self.config)
    
    if self.config['add_sum']:
      self.scaler.fit(self.add_sum(x))
    else:
      self.scaler.fit(x)
      
    x_scaled = self.scale(x)
    
    self.kde = gaussian_kde(x.T)
    
    # should try to avoid
    from visualize import Visualisation
    self.candidates = self.scale(Visualisation.generate_triangle(50, 50, 70))
    
    match self.config['query']:
      case 'EE':
        query_strategy = expected_error_reduction
      case 'US':
        query_strategy = uncertainty_sampling
      case _:
        query_strategy = uncertainty_sampling
    
    gp_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e8))* RBF(1.0)
    
    match self.config['model']:
      case 'GP':
        estimator = GaussianProcessClassifier(kernel=gp_kernel, warm_start=False, n_jobs=-1)
      case 'RF':
        estimator = RandomForestClassifier(n_jobs=-1)
      case 'SV':
        estimator = SVC(probability=True)
      case _:
        estimator = GaussianProcessClassifier(kernel=gp_kernel, n_jobs=-1)
    
    self.learner = ActiveLearner(
      estimator=estimator,
      query_strategy=query_strategy,
      X_training=x_scaled,
      y_training=y
    )
    
  def predict_accuracy(self, x, y):
    y_pred = self.learner.predict(x)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy
  
  def density(self, x):
    return self.kde(x.T) # np.exp(self.kde.score_samples(x))[0]
  
  def predict(self, x):
    x_scaled = self.scale(x)
    
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
  
  def best_candidates(self):
    this_config = self.config['query'] + self.config['model']
    if this_config in self.cache:
      return self.cache[this_config]
  
    chosen = self.learner.query(self.candidates, n_instances=1)
    chosen_points = self.scaler.inverse_transform(chosen[1])[:,:2]
    #print(chosen_points)
    return chosen_points
    
if __name__ == "__main__":
  data = Data()

  x = data.get_x_2d()
  y = data.get_fractured(omit_200_400=True)
   
  model = Model()
  model.train(x, y)
  
  #model.best_candidates()
  params = model.learner.estimator.kernel_.get_params(deep=True)
  
  print(params['k1__constant_value'])
  print(params['k2__length_scale'])
