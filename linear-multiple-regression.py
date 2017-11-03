import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv('auto-mpg.data',header=None,sep='\s+')
data = data[data[3] != '?']
data = np.array(data)
for i in range(len(data)):
    data[i,3] = float(data[i,3])

XT = np.array([data[:,4],data[:,3]])
XT = XT.astype(float)
X = np.transpose(XT)
t = data[:,0]

XTX = np.array(np.dot(XT,X))
XTX_inv = np.linalg.inv(XTX)
w = np.dot(np.dot(XTX_inv,XT),t)
y = np.dot(w,XT)

#plot
fig = plt.figure()
ax = Axes3D(fig)
x1 = np.arange(np.min(X[:,0]),np.max(X[:,0]),(np.max(X[:,0])-np.min(X[:,0]))/25.0)
x2 = np.arange(np.min(X[:,1]),np.max(X[:,1]),(np.max(X[:,1])-np.min(X[:,1]))/25.0)
X1,X2 = np.meshgrid(x1,x2)
Y =[]
for i in range(len(X1)):
    x = np.array([X1[i],X2[i]])
    Y.append(np.dot(w,x))
ax.plot_wireframe(X1,X2,Y)
ax.scatter(X[:,0],X[:,1],t)
fig.show()
fig.savefig("linear-multiple-regression.png")

fig2 = plt.figure()
