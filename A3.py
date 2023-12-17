from pca import pca
from tsne import tsne
from kmeans import kmeans
from ahc import ahc
from som import som

# declare all possible colors for usage
colors = ['blue','red','green','yellow','orange','black','grey']
# declare the two datasets
datasets = ['A3-data.csv','A3-glass.csv']

# read the dataset with name
def read_dataset(name):
    f = open(name)
    f.readline()
    data = f.readlines()
    data = [d.replace('\n','').split(',') for d in data]
    f.close()
    return data

if __name__=="__main__":
    # read one dataset
    dataset = datasets[0]
    data = read_dataset(dataset)
    
    # apply all methods
    print("---     RUNNING PCA     ---")
    pca(data, colors)
    print("---     RUNNING tSNE     ---")
    tsne(data, colors)
    print("---     RUNNING kMeans     ---")
    kmeans(dataset,data, colors)
    print("---     RUNNING AHC     ---")
    ahc(data)
    print("---     RUNNING SOM     ---")
    som(data)