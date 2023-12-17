from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def tsne(data,colors):
    # get features table
    xtable = [list(map(float,d[:-1])) for d in data]
    # get class array
    ytable = [int(d[-1]) for d in data]

    # apply tnse
    xtable = np.array(xtable)
    tsne = TSNE(n_components=2)
    nf = len(data[0][:-1])
    if nf > 2:
        principalComponents = tsne.fit_transform(xtable)
        xtable = np.transpose(principalComponents)
    else:
        xtable = np.transpose(xtable)

    # use colors to plot data
    y_pred = [colors[y-1] for y in ytable]

    # plot result
    plt.scatter(xtable[0],xtable[1],c=y_pred)
    plt.show()