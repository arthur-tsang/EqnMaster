import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
try:
        from tsne import bh_sne
        tsne_installed = True
except Exception, e:
        tsne_installed = False

def annotate(xs,ys,labels):
        """Annotate (i.e. label the scatter-points)"""
        for x,y,char in zip(xs,ys,labels):
                plt.scatter(x,y)
                plt.annotate(char, xy=(x,y))

def svd_visualize(L, labels, outfile='figs/svd.jpg', title='Word vectors SVD'):
        """Visualize L, which we assume is given as an np array of rows"""
	U, S, V = np.linalg.svd(L)
	#datapoints = U[:,:2]
        
        data_x = U[:,0]
        data_y = U[:,1]

        #plt.xkcd() # Yes!
        fig = plt.figure()
        fig.suptitle(title)

        annotate(data_x, data_y, labels)

        # save file
        if outfile is not None:
                fig.savefig(outfile)
        plt.show()


def tsne_visualize(L, labels, outfile='figs/tsne.jpg', perplexity=1):
        """Visualize L using t-sne, which is a little complicated to setup on your system"""
        if not tsne_installed:
                print 'Sorry, tsne is not installed'
                return

        # Use t-sne algorithm to come up with points on 2d plane
        points = bh_sne(L, perplexity=perplexity)
        points = np.array(points)

        data_x, data_y = points[:,0], points[:,1]
                
        fig = plt.figure()
        fig.suptitle('Word vectors T-SNE '+str(perplexity))

        annotate(data_x, data_y, labels)

        # save file
        if outfile is not None:
                fig.savefig(outfile)
        plt.show()

def multi_tsne(L, labels):
        perplexities = [.3, .5, .7, .9, 1, 1.1, 1.4, 1.7, 2]
        for perplexity in perplexities:
                print 'perp', perplexity
                tsne_visualize(L, labels, perplexity=perplexity)

def pca_visualize(L, labels, outfile = 'figs/pca.jpg', title='Word vectors PCA'):
        pca = decomposition.PCA(n_components = 3)
        pca.fit(L)
        points = pca.transform(L)
        points = np.array(points)

        print points

        # data_x = points[:,0]
        # data_y = [0 for _ in data_x]
        data_x, data_y, data_z = points[:,0], points[:,1], points[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title)

        ax.scatter(data_x, data_y, data_z)
        #annotate(data_x, data_y, labels)

        # save file
        if outfile is not None:
                fig.savefig(outfile)
        plt.show()
