import numpy as np

class Data():
  def __init__(self, path='samples.csv') -> None:
    self.data_points = np.genfromtxt(path, skip_header=1, delimiter=',')

  def get_x(self):
    return self.data_points[:, :3]
  
  def get_y(self):
    return self.data_points[:, 3:]

  def get_fractured(self):
    return self.data_points[:, 7].astype(dtype=np.int64)
  
  def get_permeable(self):
    return self.data_points[:, 8].astype(dtype=np.int64)
  
if __name__ == "__main__":
  data = Data()
  
  print(data.get_x())
  print(data.get_fractured())


