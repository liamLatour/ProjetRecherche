import numpy as np
import matplotlib.pyplot as plt

class Visualisation():
  def __init__(self) -> None:
    pass
  
  def display_3d(self, x, y, std):
    x_ax = x[:,0]
    y_ax = x[:,1]
    z_ax = x[:,2]

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    ax.scatter(x_ax, y_ax, z_ax, s=np.exp(std*12)/5, c=["r" if y[i]==1 else 'k' for i in range(len(y))])

    # labels
    # for i in range(len(Xax)):
    #   ax.text(Xax[i], Yax[i], Zax[i], ', '.join(attributes[i].astype('str')), size=8, zorder=1, color='k') 

    # for loop ends
    ax.set_xlabel('25-100', fontsize=14)
    ax.set_ylabel('100-200', fontsize=14)
    ax.set_zlabel('200-400', fontsize=14)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    ax.legend()
    plt.show()
  
  def generate_pyramide_points(self, n=3):
    points = []
    
    step = 100/n
    
    for i in range(n+1):
      for j in range((n-i)+1):
        for k in range((n-i)+1-j):
          points.append([i*step, j*step, k*step])
    
    return np.array(points)
  
