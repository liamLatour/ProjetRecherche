import numpy as np

class Data():
  def __init__(self, path='test_samples.csv') -> None:
    self.data_points = np.genfromtxt(path, skip_header=1, delimiter=',')
  
  def get_x(self):
    without_200_400 = self.data_points[self.data_points[:,2]==0]
    return without_200_400[:, :2]
  
  def get_y(self):
    without_200_400 = self.data_points[self.data_points[:,2]==0]
    return without_200_400[:, 7].astype(dtype=np.int64)
    
if __name__ == "__main__":
  data = Data()
  
  print(data.get_x())
