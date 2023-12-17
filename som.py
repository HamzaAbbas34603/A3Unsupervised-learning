from somlearn import SOM

# try different parameters to find best Map
"""
gridtype: Specify the grid form of the nodes. If gridtype='rectangular' use rectangular neurons. Else if gridtype='hexagonal' use hexagonal neurons.
maptype: Specify the map topology. If maptype='planar' use planar map. Else if maptype='toroid' use toroid map.
neighborhood: Specify the neighborhood. If neighborhood='gaussian' use Gaussian neighborhood. Else if neighborhood=’bubble’` use bubble neighborhood function.
std_coeff: Learning Rate
"""
def params_generator(data):
    ytable = [int(d[-1]) for d in data]
    best = [0,None,None,None,None,None,None]
    for ii in range(10,100,10):
        for j in range(10,100,10):
            for m in ['planar','toroid']:
                for g in ['rectangular','hexagonal']:
                    for n in ['gaussian','bubble']:
                        for lr in [0.1,0.2,0.5,0.9]:
                            a = 0
                            labels = test_som(data,ii,j,m,g,n,lr)
                            for i,l in enumerate(labels):
                                if ytable[i] - l == 0:
                                    a+=1
                            if a > best[0]:
                                best = [a,i,j,m,g,n,l]
    return best

def test_som(data,i,j,m,g,n,l):
    xtable = [list(map(float,d[:-1])) for d in data]
    model = SOM(n_columns=i,n_rows=j,maptype=m,gridtype=g,neighborhood=n, std_coeff=l)
    model = model.fit(xtable)
    labels = model.labels_
    return labels

def som(data):
    # get features table
    xtable = [list(map(float,d[:-1])) for d in data]
    # get class array
    ytable = [int(d[-1]) for d in data]

    # apply SOM
    model = SOM(n_columns=12,n_rows=10,maptype='planar',gridtype='hexagonal',neighborhood='bubble', std_coeff=0.5)
    model = model.fit(xtable)

    # get the predicted value
    labels = model.labels_
    a = 0
    for i,l in enumerate(labels):
        if ytable[i] - l == 0:
            a+=1
        
    # display componenet planes
    model.algorithm_.view_component_planes()
    print(labels)