from data import Data
from model import Model
from visualize import Visualisation
import numpy as np

data = Data()
model = Model()
visualisation = Visualisation()

x = data.get_x()
y = data.get_fractured()
model.train(x, y)

#visualisation.display_3d(x , y, 1)

x_test = visualisation.generate_pyramide_points(10)

#x_test = np.concatenate((x_test, x))

y_pred, y_probs, y_std = model.predict_probability(x_test)

visualisation.display_3d(x_test, y_pred, 1-y_probs)

#print(y_std)
