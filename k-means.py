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

#calculate centers of mass
def center_of_mass(arr):
    sum_of_arr = arr[0]
    leng = len(arr)
    for i in range(1,leng):
        sum_of_arr += arr[i]
    ave_of_arr = sum_of_arr/(float)(leng)
    return ave_of_arr

#decide cluster features involved
def decide_cluster(features,centers):
    label =[]
    for i in range(len(features)):
        dist = distance(features[i],centers[0])
        num = 0
        for j in range(1,len(centers)):
            dist_ = distance(features[i],centers[j])
            if(dist > dist_):
                dist = dist_
                num = j
        label.append(num)
    return label

def kmeans(k,features,centers):
    labels = decide_cluster(features,centers)
    new_centers = []
    for i in range(k):
        cluster_idx = (labels == (np.ones(len(labels),int)*i))
        cluster_features = features[cluster_idx]
        if cluster_features.any():
            new_centers.append(center_of_mass(cluster_features))
        else:
            new_centers.append(centers[i])
    print k
    print new_centers
    print labels
    return new_centers,labels

def make_random(features,k):
    x = []
    max_ = []
    min_ = []
    for i in range(len(features[0])):
        max_.append(np.max(features[:,i]))
        min_.append(np.min(features[:,i]))
    for i in range(k):
        x_ = []
        for j in range(len(features[0])):
            f = np.random.rand()*(max_[j]-min_[j])+min_[j]
            x_.append(f)
        x.append(x_)
    return x

def predict(features,k):
    pre_centers = make_random(features,k)
    new_centers = []
    new_labels = []
    while True:
        new_centers,new_labels = kmeans(k,features,pre_centers)
        if(np.allclose(pre_centers,new_centers)):
            break
        else:
            pre_centers = new_centers
    return new_centers,new_labels


'''
def plot(features,k):
    centers,labels = predict(features,k)
    for t,marker,c in zip(xrange(k),">oxsd","rgbyk"):
        plt.scatter(features[labels == t,0],
                    features[labels == t,1],
                    marker = marker, c = c)
'''



features,labels = loading("iris.data")


predict(features,5)

'''
for i in range(2,6):
    predict(features,i)


for i in range(2,6):
    plot(features,i)
'''
