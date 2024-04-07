import numpy as np

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt


class Data():
  def __init__(self, path='test_samples.csv') -> None:
    self.data_points = np.genfromtxt(path, skip_header=1, delimiter=',')

  def get_x_3d(self):
    return self.data_points[:, :3]
  
  def get_x_2d(self):
    without_200_400 = self.data_points[self.data_points[:,2]==0]
    return without_200_400[:, :2]
  
  def get_y(self):
    return self.data_points[:, 3:]

  def get_fractured(self, omit_200_400=False):
    if omit_200_400:
      without_200_400 = self.data_points[self.data_points[:,2]==0]
      return without_200_400[:, 7].astype(dtype=np.int64)
    return self.data_points[:, 7].astype(dtype=np.int64)
  
  def get_permeable(self):
    return self.data_points[:, 8].astype(dtype=np.int64)
  
if __name__ == "__main__":
  data = Data()
  
  x = data.get_x_2d()
  y = data.get_fractured(omit_200_400=True)
  
  x_ax = x[:,0]
  y_ax = x[:,1]

  ax = plt.subplot(111)

  size = 40

  ax.scatter(x_ax, y_ax, s=size, c=["r" if y[i]==1 else 'k' for i in range(len(y))])

  ax.set_xlabel('25-100', fontsize=14)
  ax.set_ylabel('100-200', fontsize=14)

  ax.set_xlim(-2, 102)
  ax.set_ylim(-2, 102)
  
  plt.show()