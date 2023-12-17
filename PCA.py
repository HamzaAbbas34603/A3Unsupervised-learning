from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def pca(data,colors):
    # get features table
    xtable = [list(map(float,d[:-1])) for d in data]
    # get class array
    ytable = [int(d[-1]) for d in data]

    # apply pca
    pca = PCA(n_components=2)
    nf = len(data[0][:-1])
    if nf > 2:
        principalComponents = pca.fit_transform(xtable)
        print("PCA Percentage of Data covered by two axis {}".format(sum(pca.explained_variance_ratio_)))
        xtable = np.transpose(principalComponents)
    else:
        xtable = np.transpose(xtable)

    # use colors to plot data
    y_pred = [colors[y-1] for y in ytable]

    # plot pca
    plt.scatter(xtable[0],xtable[1],c=y_pred)
    plt.show()