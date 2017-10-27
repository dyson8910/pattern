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

#execute k-means
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
    return new_centers,labels

#initialize center point
def make_random(features,k):
    idx = np.array(range(len(features)))
    np.random.shuffle(idx)
    centers = []
    for i in range(k):
        centers.append(features[idx[i]])
    return centers

#repeat k-means
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
    return new_centers,np.array(new_labels)


#plot the result
def plot(features,k_ini,k_end):
    fig = plt.figure()
    font = ['>','o','D','s','p']
    color = ['r','g','b','y','c']
    line = (k_end-k_ini)/2 + 1
    for k in range(k_ini,k_end+1):
        ax = fig.add_subplot(line,2,k-k_ini+1)
        centers,labels = predict(features,k)
        for t in range(k):
            feature = features[labels == t]
            ax.scatter(feature[:,0],
                        feature[:,1],
                        marker = font[t],
                        c = color[t])
        for t in range(k):
            center = centers[t]
            ax.scatter(center[0],
                        center[1],
                        s = 120,
                        marker = '*',
                        c = color[t])
    fig.savefig("result_of_K-means.png")
    
features,labels = loading("iris.data")

plot(features,2,5)


