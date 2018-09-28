import sys
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn 
from IPython.display import display


# generate dataset 
X , y = mglearn.datasets.make_forge() 
# plot dataset 
mglearn.discrete_scatter( X [:, 0 ], X [:, 1 ], y ) 
plt.legend([ "Class 0" , "Class 1" ], loc = 4 ) 
plt.xlabel( "First feature" ) 
plt.ylabel( "Second feature" ) 
# plt.show()
print(y)