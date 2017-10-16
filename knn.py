import numpy as np
from matplotlib import pyplot as plt

#loading data
def loading(dataset):
    data = []
    name = []
    with open('./{0}'.format(dataset)) as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            data.append([float(tk) for tk in tokens[:-1]])
            name.append(tokens[-1])
    data = np.array(data)
    name = np.array(name)
    return data,name

#euclidean distance
def distance(p0,p1):
    return np.sum( (p0-p1)**2 )

#mode
def mode(arr):
    x = []
    counter = []
    x.append(arr[0])
    counter.append(1)
    for i in range(1,len(arr)):
        name = arr[i]
        for j in range(len(x)):
            if x[j] == name:
                counter[j] += 1
                break;
            if j == len(x)-1:
                x.append(name)
                counter.append(1)
    mode_idx = np.argmax(counter)
    mode_label = x[mode_idx]
    print(x,counter)
    return mode_label   
    
#k nearest neighbor classifier 
def knn(train_features,train_labels,test_features,k):
    dists = []
    for f in train_features:
        dist = distance(f,test_features)
        dists.append(dist)
    idx = np.argsort(dists)
    neighbors = []
    for i in range(k):
        name = train_labels[idx[i]]
        neighbors.append(name)
    test_label = mode(neighbors)
    return test_label

#predict labels
def predict(features,labels,k,model = knn):
    preds = []
    for ei in range(len(features)):
        training = np.ones(len(features),bool)
        training[ei] = False
        testing = ~training
        pred = model(features[training],labels[training],features[testing],k)
        preds.append(pred)
    return preds

#calculate accuracy
def accuracy(features,labels,k,model = knn):
    preds = predict(features,labels,k,model)
    return np.mean(preds == labels)

features,labels = loading("iris.data")

x = []
y = []

for i in range(1,31):
    x.append(i)
    y.append(accuracy(features,labels,i))

#describe the result
def graph(x,y):
    plt.scatter(x,y)
    plt.title('Accuracy of KNN on iris data')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xlim(0,31)
    filename = "Accuracy_of_KNN.png"
    plt.savefig(filename)
    plt.show()

for t in range(len(x)):
    print(x[t],y[t])
    
i = np.argmax(y)
print(x[i])

graph(x,y)
