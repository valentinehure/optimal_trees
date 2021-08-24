import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn import tree

def dessin_erreur_CART_rapport():
    X = []
    X0 = []
    X1 = []
    Y = []
    for i in range(1,5):
        for j in range(1,10):
            X.append([i/5,j/10])
            if i < 3:
                if j > 3:
                    Y.append(0)
                    X0.append([i/5,j/10])
                else:
                    Y.append(1)
                    X1.append([i/5,j/10])
            else:
                if j < 7:
                    Y.append(0)
                    X0.append([i/5,j/10])
                else:
                    Y.append(1)
                    X1.append([i/5,j/10])
    
    X = np.array(X)
    Y = np.array(Y)
    X0 = np.array(X0)
    X1 = np.array(X1)
    
    plt.close()
    plt.scatter(X0[:,0],X0[:,1])
    plt.scatter(X1[:,0],X1[:,1])
    
    clf = tree.DecisionTreeClassifier()
    #clf = tree.DecisionTreeClassifier(max_depth=2)
    
    clf = clf.fit(X,Y)
    tree.plot_tree(clf) 
    
def score_dataset(X,Y):
    for i in range(1,7):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X,Y)
        print("######### SCORE ",i," : ", clf.score(X,Y))
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)
    print("######### SCORE : ", clf.score(X,Y)," ; depth : ", clf.tree_.max_depth)
    

