#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

def load_mnist_image(filename):
    with open(filename,'rb') as f:
        data = np.frombuffer(f.read(),np.uint8,offset=16)
    return data.reshape(-1,784)/255.0

def load_mnist_label(filename):
    with open(filename,'rb') as f:
        data = np.frombuffer(f.read(),np.uint8,offset=8)
        label = np.zeros((len(data),10))
        for i in range(len(data)):
            t = data[i]
            label[i,t] = 1
    return label

train_data = load_mnist_image("train-images-idx3-ubyte")
test_data = load_mnist_image("t10k-images-idx3-ubyte")
train_t = load_mnist_label("train-labels-idx1-ubyte")
test_t= load_mnist_label("t10k-labels-idx1-ubyte")

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        self.out = 1.0/(1.0 + np.exp(-x))
        return self.out
    def backward(self,dout):
        delta = dout * (1.0 - self.out) * self.out
        return delta
class Softmax:
    def __init__(self):
        self.out = None
    def forward(self,x):
        if x.ndim == 2:
            x_ = x.transpose()
            x_ = x_ - np.max(x_,axis=0)
            out = np.exp(x_)/np.sum(np.exp(x_),axis=0)
            self.out = out.transpose()
        else:
            x_ = x - np.max(x)
            out = np.exp(x_)/np.sum(np.exp(x_))
            self.out = out
        return self.out
    def backward(self,dout):
        delta = self.out - dout
        return delta
class NeuralNet:
    def __init__(self,size):
        hidden_num = len(size)-2
        self.layer = np.array([None]*(hidden_num+1))
        self.w = np.array([None]*(hidden_num+1))
        self.dw = np.array([None]*(hidden_num+1))
        self.b = np.array([None]*(hidden_num+1))
        self.db = np.array([None]*(hidden_num+1))
        for i in range(hidden_num):
            self.layer[i] = Sigmoid()
            self.w[i] = 0.01*np.random.randn(size[i],size[i+1])
            self.dw[i] = np.zeros((size[i],size[i+1]))
            self.b[i] = np.zeros(size[i+1])
            self.db[i] = np.zeros(size[i+1])
        self.layer[hidden_num] = Softmax()
        self.w[hidden_num] = 0.01*np.random.randn(size[hidden_num],size[hidden_num+1])
        self.dw[hidden_num] = np.zeros((size[hidden_num],size[hidden_num+1]))
        self.b[hidden_num] = np.zeros(size[hidden_num+1])
        self.db[hidden_num] = np.zeros(size[hidden_num+1])
        self.hidden_num = hidden_num
        self.x = [None]*(hidden_num+1)
    def forward(self,x):
        x_ = x
        for i in range(self.hidden_num+1):
            self.x[i] = x_
            u = np.dot(x_,self.w[i]) + self.b[i]
            x_ = self.layer[i].forward(u)
        return x_
    def backward(self,t,batch_size):
        d = t
        for i in range(self.hidden_num+1):
            delta = self.layer[self.hidden_num-i].backward(d)
            x_ = self.x[self.hidden_num-i].reshape(batch_size,-1)
            x_ = x_.transpose()
            delta_ = delta.reshape(batch_size,-1)
            self.dw[self.hidden_num-i] += np.dot(x_,delta_)
            self.db[self.hidden_num-i] += np.dot(np.ones((1,batch_size)),delta_).reshape(-1)
            d = np.dot(delta_,self.w[self.hidden_num-i].transpose())
    def train(self,data,label,batch_size,limit=100,learning_rate=0.1):
        for i in range(limit):
            idx = np.random.permutation(len(data))
            for j in range(len(label)/batch_size):
                y = self.forward(data[idx[j*batch_size:(j+1)*batch_size]])
                self.backward(label[idx[j*batch_size:(j+1)*batch_size]],batch_size)
                self.w -= self.dw * learning_rate / float(batch_size)
                self.b -= self.db * learning_rate / float(batch_size)
                self.dw = np.zeros_like(self.dw)
                self.db = np.zeros_like(self.db)

    def check(self,test,ans):
        correct = 0
        num = len(ans)
        for i in range(num):
            y = self.forward(test[i])
            if(np.argmax(y) == np.argmax(ans[i])):
                correct += 1
        accuracy = correct / float(num)
        return accuracy
    def graph(self,x,y,filename,xlabel,ylabel):
        plt.figure()
        plt.plot(x,y)
        plt.title(filename)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0,len(x)+1)
        plt.savefig(filename+".png")
        
    def learn(self,train_data,train_label,test_data,test_label,batch_size,filename,xlabel,ylabel,limit=30):
        x = range(1,limit+1)
        y = np.array([])
        for i in x:
            self.train(train_data,train_label,batch_size,1,1.0/np.cbrt(i))
            y_ = self.check(test_data,test_label)
            y = np.append(y,y_)
            print(i)
            print(y_)
        print(np.argmax(y)+1)
        print(y[np.argmax(y)])
        self.graph(x,y,filename,xlabel,ylabel)
        return x,y

n = NeuralNet([784,500,300,100,10])
n.learn(train_data,train_t,test_data,test_t,100,"Accuracy_of_4layers_NeuralNet","epoch","accuracy")
