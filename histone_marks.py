import matplotlib
matplotlib.use('TkAgg')
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


# DATA PREPROCESSING
histone_marks = pd.read_csv("http://users.wenglab.org/moorej3/Yu-Project/K562-rDHS-Histone-Matrix.txt",
                            sep='\t', header=0, index_col=0)
group = pd.read_csv("http://users.wenglab.org/moorej3/Yu-Project/K562-Group-List.txt",
                    sep='\t', header=None, index_col=0)
rdhs = group.index.values
cls = group.iloc[:,0].values
columns = histone_marks.columns
histone_marks = histone_marks.replace(0, 0.0001)
histone_marks = np.log(histone_marks)
histone_marks = StandardScaler().fit_transform(histone_marks)
histone_marks = pd.DataFrame(data=histone_marks[:],
                             index=rdhs,
                             columns=columns)

def tSNE(data, pdf):
    tsne = TSNE(perplexity=10)
    plt.title("tSNE -perplexity=10")
    principalComponents= tsne.fit_transform(data)
    my_palette = {"#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6",
                  "#6a3d9a", "#ffff99", "#b15928"}
    print(principalComponents)
    sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1],
                        hue=cls, legend="brief", palette=my_palette)
    plt.legend(bbox_to_anchor=(0, 1), loc=0, borderaxespad=0., prop={'size': 5})
    plt.tight_layout()
    plt.show()
    pdf.savefig()
    plt.close()
    print("tSNE Done")


def uMap(data, pdf):
    neighbors = [10, 15, 20, 25, 50]
    for n in neighbors:
        plt.title("uMap -perplexity = " + str(n))
        principalComponents = umap.UMAP(n_neighbors=n).fit_transform(data)
        my_palette = {"#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6",
                      "#6a3d9a",  "#ffff99", "#b15928"}
        sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1],
                        hue=cls, legend="brief", palette=my_palette)
        plt.legend(bbox_to_anchor=(0, 1), loc=0, borderaxespad=0., prop={'size': 5})
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print("n=", n, " done")

# SAVE FIGSse43
with PdfPages("Dimension_Reduction.pdf") as pdf:
    print("start tSNE")
    tSNE(histone_marks, pdf)
   # uMap(histone_marks, pdf)






