from data import Data
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from modAL.models import ActiveLearner

# Query strategie
from batch_uncertainty_similarity_sampling import uncertainty_batch_sampling

# Modeling strategie
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class Model():
  def __init__(self) -> None:
    self.scaler = StandardScaler()
    self.learner = None
    self.config = {
      "add_sum": True,
      "n_instances": 4,
      "ratio": 50
    }
    
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
    
    # should try to avoid
    from visualize import Visualisation
    self.candidates = self.scale(Visualisation.generate_triangle(1000, 50, 70))
    
    query_strategy = uncertainty_batch_sampling

    gp_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e8))* RBF(1.0)
    estimator = GaussianProcessClassifier(kernel=gp_kernel, warm_start=False, n_jobs=-1)

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
  
  def predict(self, x):
    x_scaled = self.scale(x)
    
    y_prob = self.learner.predict_proba(x_scaled)
    y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob[:,0]
  
  def print_candidates(self, candidates):
    print('25-100 | 100-200')
    print('-------|--------')
    for candidate in candidates:
      print("{:6.1f} |  {:6.1f}".format(candidate[0], candidate[1]))
  
  def best_candidates(self):  
    #chosen = self.learner.query(self.candidates, n_instances=self.config['n_instances'], alpha=self.config['ratio']/100.0)
    #chosen_points = self.scaler.inverse_transform(chosen[1])[:,:2]
    
    #self.print_candidates(chosen_points)
    
    return np.array([[45.8,20.3],[68,0],[28.3,39.5],[14.9,49.8]])#chosen_points

