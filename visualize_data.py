import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt('samples.csv', skip_header=1, delimiter=',')

attributes = data[:, :4]
results = data[:, 4:]

mean = np.mean(attributes, axis=0)
print("Mean", mean)

scaler = StandardScaler()
scaler.fit(attributes) 
attributes_scaled = scaler.transform(attributes)

pca = PCA(n_components=3)
pca.fit(attributes_scaled)
attributes_pca = pca.transform(attributes_scaled) 

Xax = attributes_pca[:,0]
Yax = attributes_pca[:,1]
Zax = attributes_pca[:,2]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
ax.scatter(Xax, Yax, Zax, s=40, c=["k" if data[i][15]==0 else 'r' for i in range(len(data))])

# labels
for i in range(len(Xax)):
  ax.text(Xax[i], Yax[i], Zax[i], ', '.join(attributes[i].astype('str')), size=8, zorder=1, color='k') 
  
# for loop ends
ax.set_xlabel(', '.join(np.round(pca.components_[0], decimals=3).astype('str')), fontsize=14)
ax.set_ylabel(', '.join(np.round(pca.components_[1], decimals=3).astype('str')), fontsize=14)
ax.set_zlabel(', '.join(np.round(pca.components_[2], decimals=3).astype('str')), fontsize=14)


ax.legend()
plt.show()

pca_data = np.concatenate((np.array([[1., 1., 1., 1.]]), pca.components_))

inv = np.linalg.inv(pca_data)
suggested = inv[0]*17+mean
suggested[suggested<0] = 0
print(pca.components_)
print("Suggested test", suggested, " total", np.sum(suggested))