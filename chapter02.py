import sys
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn 
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier 


# # generate dataset 
# X , y = mglearn.datasets.make_forge() 
# # plot dataset 
# mglearn.discrete_scatter( X [:, 0 ], X [:, 1 ], y ) 
# plt.legend([ "Class 0" , "Class 1" ], loc = 4 ) 
# plt.xlabel( "First feature" ) 
# plt.ylabel( "Second feature" ) 
# # plt.show()
# print(y)


boston = load_boston()
X, y = mglearn.datasets.load_extended_boston()

# mglearn.plots.plot_knn_classification( n_neighbors = 5 )
# plt.show()
X , y = mglearn.datasets.make_forge()
fig , axes = plt.subplots( 1 , 3 , figsize =( 10 , 3 )) 
for n_neighbors , ax in zip([ 1 , 3 , 9 ], axes ): 
    # the fit method returns the object self, so we can instantiate 
    # and fit in one line 
    clf = KNeighborsClassifier( n_neighbors = n_neighbors ).fit( X , y ) 
    mglearn.plots.plot_2d_separator( clf , X , fill = True , eps = 0.5 , ax = ax , alpha =.4 ) 
    mglearn.discrete_scatter( X [:, 0 ], X [:, 1 ], y , ax = ax ) 
    ax.set_title( "{} neighbor(s)".format( n_neighbors )) 
    ax.set_xlabel( "feature 0" ) 
    ax.set_ylabel( "feature 1" ) 
axes [ 0 ].legend( loc = 3 )
plt.show()
