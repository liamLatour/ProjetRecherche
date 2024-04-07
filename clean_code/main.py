from data import Data
from model import Model
from visualize import Visualisation
import PySimpleGUI as sg


data = Data()
model = Model()

x = data.get_x()
y = data.get_y()
model.train(x, y)

visualisation = Visualisation(model, data)

options_list_column = [
    [sg.Checkbox('Show boundary',   key='-BOUND_CB-', default=True,              enable_events=True),
    sg.Checkbox('Show original',   key='-ORIG_CB-',  default=True,              enable_events=True),],
    [sg.Checkbox('Show grid',       key='-GRID_CB-',   enable_events=True),
    sg.Checkbox('Show candidate(s)', key='-CAND_CB-',                enable_events=True)],
    [sg.HorizontalSeparator(color='black')],
    [
      sg.Checkbox('Add sum to inputs',   key='-SUM_CB-', default=True,              enable_events=True),
    ],
    [
      sg.Text('Nb of instances'),
      sg.Slider(range=(1, 10), default_value=4, expand_x=True, enable_events=True, key='-INST_SL-', orientation='horizontal')
    ],
    [
      sg.Text('Proximity/Uncertainty ratio'),
      sg.Slider(range=(0, 100), default_value=50, expand_x=True, enable_events=True, key='-RATIO_SL-', orientation='horizontal')
    ],
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
    case '-SUM_CB-':
      model.config['add_sum'] = values['-SUM_CB-']
      model.train(x, y)
    case '-INST_SL-':
      model.config['n_instances'] = int(values['-INST_SL-'])
    case '-RATIO_SL-':
      model.config['ratio'] = int(values['-RATIO_SL-'])
    case _:
      print(event, values)
  
  fig = visualisation.generate_fig()
  visualisation.draw_figure(window["-CANVAS-"].TKCanvas, fig)
  
window.close()
