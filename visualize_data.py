import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('samples.csv', skip_header=1, delimiter=',')

attributes = data[:, :3]
results = data[:, 3:]

mean = np.mean(attributes, axis=0)
print("Mean", mean)

Xax = data[:,0]
Yax = data[:,1]
Zax = data[:,2]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
ax.scatter(Xax, Yax, Zax, s=40, c=["r" if data[i][7]==1 else 'b' if data[i][8]==1 else 'k' for i in range(len(data))])

# labels
for i in range(len(Xax)):
  ax.text(Xax[i], Yax[i], Zax[i], ', '.join(attributes[i].astype('str')), size=8, zorder=1, color='k') 

# for loop ends
ax.set_xlabel('25-100', fontsize=14)
ax.set_ylabel('100-200', fontsize=14)
ax.set_zlabel('200-400', fontsize=14)

ax.legend()
plt.show()