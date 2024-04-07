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
      "bounds": True,
      "original": True,
      "grid": False,
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
    
  def display_2d(self, x, y, size, special=None):
    x_ax = x[:,0]
    y_ax = x[:,1]

    fig = Figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    if len(x) == 0:
      return fig

    size = 40 #size/np.min(size)

    fig.patch.set_facecolor('white')    
    ax.scatter(x_ax, y_ax, s=size, c=["r" if y[i]==1 else 'k' for i in range(len(y))])
    
    if type(special) == type(np.array([])):
      ax.scatter(special[:,0], special[:,1], s=40, c="b")
      for point in special:
        ax.annotate(
          str(round(point[0], 2))+'/'+str(round(point[1], 2)),
          (point[0], point[1]),
          fontsize=15
        )

    if self.config['bounds']:
      grid_x, grid_y = np.mgrid[-2:102:200j, -2:102:200j]
      grid = np.stack([grid_x, grid_y], axis=-1)
      prediction = self.model.predict(grid.reshape((-1, 2)))[1]
      prediction = prediction.reshape(*grid_x.shape)
      
      ax.contour(grid_x, grid_y, prediction,
                 levels=[0.5],
                 colors='black',
                 linestyles='dashed',
                 linewidths=1)
      
      #surf = ax.contourf(grid_x, grid_y, prediction)
      #fig.colorbar(surf)

    ax.set_xlabel('25-100', fontsize=14)
    ax.set_ylabel('100-200', fontsize=14)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    return fig
  
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

    ax.set_xlabel('25-100', fontsize=14)
    ax.set_ylabel('100-200', fontsize=14)
    ax.set_zlabel('200-400', fontsize=14)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    return fig
  
  def generate_fig_3d(self):
    points = np.empty((0, 3))
    
    if self.config['original']:
      points = np.concatenate((points, self.data.get_x_3d()))
    if self.config['grid']:
      points = np.concatenate((points, self.generate_pyramide_points(10)))
    if self.config['bounds']:
      points = np.concatenate((points, self.generate_boundary(15)))

    if len(points) > 0:
      y_pred, _ = self.model.predict(points)
      y_prob = self.model.density(points)
    else:
      y_pred = []
      y_prob = []
    
    return self.display_3d(points, [0 for _ in range(len(points))], [1 for _ in range(len(points))])#y_pred, y_prob)
  
  def generate_fig_2d(self):
    points = np.empty((0, 2))
    special = None
    
    if self.config['original']:
      points = np.concatenate((points, self.data.get_x_2d()))
    if self.config['grid']:
      points = np.concatenate((points, Visualisation.generate_triangle(30)))
    if self.config['candidate']:
      special = self.model.best_candidates()
    # if self.config['bounds']:
    #   points = np.concatenate((points, self.generate_boundary(5)))

    if len(points) > 0:
      y_pred, _ = self.model.predict(points)
      y_prob = self.model.density(points)
    else:
      y_pred = []
      y_prob = []
    
    return self.display_2d(points, y_pred, y_prob, special)
  
  @staticmethod
  def generate_triangle(n=3, min_mix=0, max_mix=100, half=None):
    points = []
    
    step = 100/n
    
    for i in range(n+1):
      for j in range((n-i)+1):
        total = i*step + j*step
        if total <= max_mix and min_mix <= total:
          if half == None or (half and i<=j) or ((not half) and j<=i):
            points.append([i*step, j*step])
    
    return np.array(points)
  
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
