import numpy as np
from sklearn.cluster import KMeans

def kmeans(name,data, colors):
    # get features table
    xtable = [list(map(float,d[:-1])) for d in data]
    # get class array
    ytable = [int(d[-1]) for d in data]
    xtable = np.array(xtable)

    # find number of classes
    nclasses = set()
    for y in ytable:
        nclasses.add(y)
    nclasses = list(nclasses)
    best = [0,None]
    for k in range(2,len(nclasses)+1):
        # apply kmean with different k
        kmean = KMeans(
            n_clusters=k
        )
        kmean.fit(xtable)

        # get predicted values
        labels = kmean.labels_

        # find number of correct matches
        a = 0
        for i,y in enumerate(ytable):
            if abs(y-kmean.labels_[i]) == 0:
                a += 1
            
        # get the best kmean execution
        if best[0] < a:
            best = [a,k]
            with open('kmeans-'+name,'w') as f:
                for i,y in enumerate(ytable):
                    f.write('{},{},{},{}\n'.format(ytable[i],labels[i],abs(y-labels[i]),"YES" if y-labels[i] == 0 else "NO"))
        print("kMeans TOTAL SIMILAR INSTANCES: {}".format(a))
    print(best)