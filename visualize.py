import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt 
from data import Data
from model import Model

class Visualisation():
  def __init__(self, model:Model, data:Data) -> None:
    matplotlib.use("TkAgg")
    self.figure_canvas_agg = None
    self.model = model
    self.data = data
    self.config = {
      "bounds": False,
      "original": False,
      "grid": True,
      "candidate": False
    }
  
  def draw_figure(self, canvas, figure):
    if self.figure_canvas_agg != None:
      self.figure_canvas_agg.get_tk_widget().forget()
      plt.close('all')

    self.figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    self.figure_canvas_agg.draw()
    self.figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return self.figure_canvas_agg
    
  def display_3d(self, x, y, size, special=None):
    x_ax = x[:,0]
    y_ax = x[:,1]
    z_ax = x[:,2]

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    if len(x) == 0:
      return fig

    size = size/np.min(size)

    fig.patch.set_facecolor('white')
    ax.scatter(x_ax, y_ax, z_ax, s=size, c=["r" if y[i]==1 else 'k' for i in range(len(y))])
    if type(special) == type(np.array([])):
      ax.scatter(special[0], special[1], special[2], s=3, c="b")

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
    return fig
  
  def generate_fig(self):
    points = np.empty((0, 3))
    
    if self.config['original']:
      points = np.concatenate((points, self.data.get_x()))
    if self.config['grid']:
      points = np.concatenate((points, self.generate_pyramide_points(10)))
    
    if self.config['bounds']:
      points = np.concatenate((points, self.generate_boundary(5)))

    if len(points) > 0:
      y_pred, _ = self.model.predict(points)
      y_prob = self.model.density(points)
    else:
      y_pred = []
      y_prob = []
    
    return self.display_3d(points, y_pred, y_prob)
  
  def generate_pyramide_points(self, n=3):
    points = []
    
    step = 100/n
    
    for i in range(n+1):
      for j in range((n-i)+1):
        for k in range((n-i)+1-j):
          points.append([i*step, j*step, k*step])
    
    return np.array(points)
  
  def generate_boundary(self, n=10):
    points = []
    
    step = (np.pi/2) / n
    
    for i in range(n+1):
      for j in range(n+1):
        alpha = step*i
        beta = step*j
        point = [np.sin(alpha)*np.sin(beta),
                np.sin(alpha)*np.cos(beta),
                np.cos(alpha),
        ]
        
        points.append(self.model.bound_intersect_vec(point))

    return np.array(points)
