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
    
  def display(self, x, y, size, special=None):
    x_ax = x[:,0]
    y_ax = x[:,1]

    fig = Figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    if len(x) == 0:
      return fig

    size = 40 #20*size/np.min(size)

    fig.patch.set_facecolor('white')    
    ax.scatter(x_ax, y_ax, s=size, c=["r" if y[i]==1 else 'k' for i in range(len(y))])
    
    if type(special) == type(np.array([])):
      ax.scatter(special[:,0], special[:,1], s=40, c="b")
      for point in special:
        ax.annotate(
          str(round(point[0], 2))+'/'+str(round(point[1], 2)),
          (point[0], point[1]),
          fontsize=15,
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
      
      #ax.contourf(grid_x, grid_y, prediction)

    ax.set_xlabel('25-100', fontsize=14)
    ax.set_ylabel('100-200', fontsize=14)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    return fig
  
  def generate_fig(self):
    points = np.empty((0, 2))
    special = None
    
    if self.config['original']:
      points = np.concatenate((points, self.data.get_x()))
    if self.config['grid']:
      points = np.concatenate((points, Visualisation.generate_triangle(30)))
    if self.config['candidate']:
      special = self.model.best_candidates()

    if len(points) > 0:
      y_pred, _ = self.model.predict(points)
      y_prob = 1
    else:
      y_pred = []
      y_prob = []
    
    return self.display(points, y_pred, y_prob, special)
  
  @staticmethod
  def generate_triangle(n=3, min_mix=0, max_mix=100):
    points = []
    
    step = 100/n
    
    for i in range(n+1):
      for j in range((n-i)+1):
        total = i*step + j*step
        if total <= max_mix and min_mix <= total:
          points.append([i*step, j*step])
    
    return np.array(points)

