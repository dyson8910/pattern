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
        dx = dout * (1.0 - self.out) * self.out
        return dx
class Softmax:
    def __init__(self,output_size):
        self.out = None
        self.output_size = output_size
    def forward(self,x):
        c = np.max(x)
        out = np.exp(x-c)
        s = np.sum(out)
        self.out = out/s
        return self.out
    def backward(self,dout):
        dx = (self.out - dout)/self.output_size
        return dx
class NeuralNet:
    def __init__(self,input_size,hidden_num,output_size):
        self.layer = []
        self.w = []
        self.b = []
        for i in range(hidden_num):
            self.layer.append(Sigmoid())
            self.w.append(0.01*np.random.randn(input_size,input_size))
            self.b.append(np.zeros(input_size))
        self.layer.append(Softmax(output_size))
        self.w.append(0.01*np.random.randn(input_size,output_size))
        self.b.append(np.zeros(output_size))
        self.x = np.zeros((hidden_num+1,input_size))
        self.hidden_num = hidden_num
    def forward(self,x):
        x_ = x
        for i in range(self.hidden_num+1):
            self.x[i] = x_
            u = np.dot(self.x[i],self.w[i]) + self.b[i]
            x_ = self.layer[i].forward(u)
        return x_
    def backward(self,t,batch_size,learning_rate):
        d = t
        for i in range(self.hidden_num+1):
            delta = self.layer[self.hidden_num-i].backward(d)
            self.w[self.hidden_num-i] = self.w[self.hidden_num-i] - learning_rate * np.dot(self.x[self.hidden_num-i].reshape(-1,batch_size),delta.reshape(batch_size,-1))
            self.b[self.hidden_num-i] = self.b[self.hidden_num-i] - learning_rate * delta
            d = np.dot(self.w[self.hidden_num-i],delta)

    def train(self,data,label,limit=100,learning_rate=0.01):
        for i in range(limit):
            flag = 0
            for j in range(len(label)):
                y = self.forward(data[j])
                self.backward(label[j],1,learning_rate)
                if(np.argmax(y) != np.argmax(label[j])):
                    flag = 1
            if(flag == 0):
                break

    def check(self,test,ans):
        correct = 0
        num = len(ans)
        for i in range(num):
            y = self.forward(test[i])
            if(np.argmax(y) == np.argmax(ans[i])):
                correct += 1
        accuracy = correct / float(num)
        return accuracy

    def graph(self,train_data,train_label,test_data,test_label,limit=50):
        x = range(1,limit+1)
        y = []
        for i in x:
            self.train(train_data,train_label,1)
            y_ = self.check(test_data,test_label)
            y.append(y_)
            print i
            print y_
        plt.plot(x,y)
        plt.title("Accuracy of NeuralNetwork")
        plt.xlabel("times")
        plt.ylabel("Accuracy")
        plt.xlim(0,limit+1)
        filename = "Accuracy_of_NeuralNetwork.png"
        plt.savefig(filename)
        plt.show()
        print np.argmax(y)+1
        print y[np.argmax(y)]
            
n = NeuralNet(784,1,10)
n.graph(train_data,train_t,test_data,test_t)
