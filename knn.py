import numpy as np

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

#k nearest neighbor classifier 
def knn(train_features,train_labels,test_features,k):
    dists = []
    for f in range(len(train_features)):
        dist = distance(f,test_features)
        dists.append(dist)
    idx = np.argsort(dists)
    neighbors = []
    for i in range(k):
        name = train_labels[idx[i]]
        neighbors.append(name)
    
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

for i in range(1,30):
    
