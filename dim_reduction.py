import pandas as pd
import numpy as np
import seaborn as sns
from pylab import subplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
import umap

# split experiment by timepoint and tissue
def split_names(names):
    timepoints = list()
    tissue = list()
    for name in names:
        tissue.append(("_").join(name.split('_')[1:3]))
        timepoints.append(("_").join(name.split('_')[-2:]))
    return timepoints, tissue

# DATA PREPROCESSING
data = pd.read_csv("http://users.wenglab.org/moorej3/Yu-Project/Mouse-Enhancer-Matrix.txt", sep='\t', header=0, index_col=0)
ccre = data.index.values
experiment = data.columns.values
timepoints, tissue = split_names(experiment)
# scale it before transpose so it happens within each experiment
data = data.replace(0, 0.0001)
data = np.log(data)
data = StandardScaler().fit_transform(data)
data = data.transpose()


def pca(data, pdf):
# PCA SCALTTER PLOT
    plt.title("PCA")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    g = sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1],
                    hue=tissue, style=timepoints, legend="brief")
    plt.legend(bbox_to_anchor=(0, 1), loc=0, borderaxespad=0., prop={'size': 5})
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def tSNE(data, pdf):
    tsne = TSNE(perplexity=10)
    plt.title("tSNE -perplexity=10")
    principalComponents= tsne.fit_transform(data)
    g = sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1],
                        hue=tissue, style=timepoints, legend="brief")
    plt.legend(bbox_to_anchor=(0, 1), loc=0, borderaxespad=0., prop={'size': 5})
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def uMap(data, pdf):
    plt.title("uMap -perplexity=10")
    principalComponents = umap.UMAP(n_neighbors=10).fit_transform(data)
    g = sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1],
                        hue=tissue, style=timepoints, legend="brief")
    plt.legend(bbox_to_anchor=(0, 1), loc=0, borderaxespad=0., prop={'size': 5})
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# SAVE FIGS; Output three figs as a three page pdf
with PdfPages("Dimension_Reduction.pdf") as pdf:
    pca(data, pdf)
    tSNE(data, pdf)
    uMap(data, pdf)





