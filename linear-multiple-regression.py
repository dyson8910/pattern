import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loading(filename):
        data = pd.read_csv(filename,header=None,sep='\s+')
        data = data[data[3] != '?']
        data = np.array(data)
        for i in range(len(data)):
                data[i,3] = float(data[i,3])
        return data

def regression(data,x1,x2):
        Ones = np.ones(len(data))
        XT = np.array([data[:,x1],data[:,x2],Ones])
        XT = XT.astype(float)
        X = np.transpose(XT)
        t = data[:,0]
        XTX = np.array(np.dot(XT,X))
        XTX_inv = np.linalg.inv(XTX)
        wb = np.dot(np.dot(XTX_inv,XT),t)
        w = wb[:-1]
        b = wb[-1]
        print w,b
        return X,w,b

def plot(data):
        fig = plt.figure()
        ax = Axes3D(fig)
        X,w,b = regression(data,3,4)
        x1 = np.arange(np.min(X[:,0]),np.max(X[:,0]),(np.max(X[:,0])-np.min(X[:,0]))/25.0)
        x2 = np.arange(np.min(X[:,1]),np.max(X[:,1]),(np.max(X[:,1])-np.min(X[:,1]))/25.0)
        X1,X2 = np.meshgrid(x1,x2)
        Y =[]
        for i in range(len(X1)):
                x = np.array([X1[i],X2[i]])
                Y.append(np.dot(w,x))
        Y = np.array(Y)
        Y = Y + b
        ax.plot_wireframe(X1,X2,Y)
        ax.scatter(X[:,0],X[:,1],data[:,0])
        ax.set_xlabel("horsepower")
        ax.set_ylabel("wight")
        ax.set_zlabel("mpg")        
        fig.show()
        fig.savefig("result-of-linear-multiple-regression.png")

data = loading('auto-mpg.data')
plot(data)


