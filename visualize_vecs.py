#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def svd_visualize(L, labels, outfile='figs/svd.jpg'):
        """Visualize L, which we assume is given as an np array of rows"""
	U, S, V = np.linalg.svd(L)
	#datapoints = U[:,:2]
        
        data_x = U[:,0]
        data_y = U[:,1]

        fig = plt.figure()
        fig.suptitle('Word vectors SVD')
        plt.scatter(data_x, data_y)

        # Annotate (i.e. label the scatter-points)
        for x,y,char in zip(data_x, data_y, labels):
                plt.annotate(char, xy=(x,y))
        

        if outfile is not None:
                fig.savefig(outfile)
        plt.show()

        # TODO: annotate points
        # http://stackoverflow.com/questions/5147112/matplotlib-how-to-put-individual-tags-for-a-scatter-plot

if __name__ == '__main__':
        L = np.array([[1,2,3],[4,5,6],[7,8,9],[1,1,1]])
        svd_visualize(L)
        
