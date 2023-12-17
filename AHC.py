from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def ahc(data):
    # get features table
    xtable = [list(map(float,d[:-1])) for d in data]
    # get class array
    ytable = [int(d[-1]) for d in data]
    # apply AHC with both methods
    """
        affinity is the distance metric to use == euclidean
        linkage (complete or average)
    """
    print("AHC USING COMPLETE LINKAGE METHOD")
    hac=AgglomerativeClustering(affinity='euclidean', linkage='complete', n_clusters=2)
    hac.fit(xtable)
    plt.title("AHC Complete Linkage")
    dendrogram(linkage(xtable,method='complete',metric='euclidean'))
    membership = hac.labels_
    score = silhouette_score(xtable, membership)
    print(score)
    plt.show()

    print("AHC USING UNWEIGHTED AVERAGE METHOD")
    hac=AgglomerativeClustering(affinity='euclidean', linkage='average', n_clusters=2)
    hac.fit(xtable)
    plt.title("AHC Unweighted Average")
    dendrogram(linkage(xtable,method='average',metric='euclidean'))
    membership = hac.labels_
    score = silhouette_score(xtable, membership)
    print(score)
    plt.show()