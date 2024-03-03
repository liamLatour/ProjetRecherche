from data import Data
from model import Model
from visualize import Visualisation
import PySimpleGUI as sg


data = Data()
model = Model('native')

x = data.get_x()
y = data.get_fractured()
model.train(x, y)

visualisation = Visualisation(model, data)

options_list_column = [
    [sg.Checkbox('Show boundary', key='-BOUND_CB-', enable_events=True)],
    [sg.Checkbox('Show original', key='-ORIG_CB-', enable_events=True)],
    [sg.Checkbox('Show grid', default=True, key='-GRID_CB-', enable_events=True)],
    [sg.Checkbox('Show candidates', key='-CAND_CB-', enable_events=True)],
]

plot_viewer_column = [
    [sg.Text("Matplotlib plot")],
    [sg.Canvas(key="-CANVAS-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(options_list_column),
        sg.VSeperator(),
        sg.Column(plot_viewer_column),
    ]
]

# Create the form and show it without the plot
window = sg.Window(
    "Data viewer",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)

# Add the plot to the window
fig = visualisation.generate_fig()
visualisation.draw_figure(window["-CANVAS-"].TKCanvas, fig)

while True:
  event, values = window.read()
  print(event, values)
  
  match event:
    case sg.WIN_CLOSED | "Ok":
      break
    case '-BOUND_CB-':
      visualisation.config['bounds'] = values['-BOUND_CB-']
    case '-ORIG_CB-':
      visualisation.config['original'] = values['-ORIG_CB-']
    case '-GRID_CB-':
      visualisation.config['grid'] = values['-GRID_CB-']
    case '-CAND_CB-':
      visualisation.config['candidate'] = values['-CAND_CB-']
  
  fig = visualisation.generate_fig()
  visualisation.draw_figure(window["-CANVAS-"].TKCanvas, fig)
  
window.close()


# candidate = model.best_candidate()

# # Display original points
# #visualisation.display_3d(x , y, .3)

# # Display feature space, grid like
# x_test = visualisation.generate_pyramide_points(10)
# #x_test = np.concatenate((x_test, x))

# y_pred, _ = model.predict(x_test)
# #print(y_pred, y_prob)

# y_prob = model.density(x_test)

# visualisation.display_3d(x_test, y_pred, y_prob)#, candidate)

# # Display boundary
# points = visualisation.generate_boundary(model, 5)
# y_pred, _ = model.predict(points)

# y_prob = model.density(points)

# visualisation.display_3d(points, y_pred, y_prob)#, candidate)
