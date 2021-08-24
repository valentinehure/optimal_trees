import numpy as np
import random as rd

############## Pour des datasets déja existant

#import pandas as pd
#from pydataset import data

# df = data('titanic')
# https://github.com/iamaziz/PyDataset


############# Ecriture dans un fichier .txt

def write(X,Y,K,name):
    n, p = np.shape(X)
    with open("../data/" + name, 'w') as f:
        f.write("X = [")
        for i in range(n):
            for j in range(p):
                f.write(str(X[i,j])+" ")
            if i != (n-1):
                f.write("; ")
        f.write("]\n")
        f.write("Y = [")
        for i in range(n-1):
            f.write(str(Y[i])+", ")
        f.write(str(Y[i])+"]\n")
        f.write("K = "+str(K))

############
def read_wine(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/wine.data", 'r') as f:
        l = f.readline()
        while len(l) > 0:
            y.append(int(l[0]))
            commas = [1]
            for i in range(2,len(l)):
                if l[i] == ",":
                    commas.append(i)
            commas.append(len(l)-1)
            x.append([])
            for i in range(len(commas)-1):
                string = l[commas[i]+1:commas[i+1]]
                if "." in string:
                    x[-1].append(float(string))
                else:
                    x[-1].append(int(string))
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)
            
    if write_file:
        write(x,y,3,"wine.txt")
    if get_xy:
        return x,y
            
def read_blood_donation(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/transfusion.data", 'r') as f:
        l = f.readline()
        l = f.readline()
        while len(l) > 0:
            commas = [-1]
            for i in range(len(l)):
                if l[i] == ",":
                    commas.append(i)
            x.append([])
            for i in range(len(commas)-1):
                x[-1].append(float(l[commas[i]+1:commas[i+1]]))
            y.append(int(l[commas[-1]+1:len(l)-1]))
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)

    if write_file:
        write(x,y,2,"blood_donation.txt")
    if get_xy:
        return x,y

    
def read_breast_cancer(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/breast-cancer-wisconsin.data", 'r') as f:
        l = f.readline()
        while len(l) > 0:
            commas = []
            for i in range(len(l)):
                if l[i] == ",":
                    commas.append(i)
            x.append([])
            missing = False
            for i in range(len(commas)-1):
                string = l[commas[i]+1:commas[i+1]]
                if string == "?":
                    missing = True
                    break
                else :
                    x[-1].append(float(string))
                        
            if missing:
                x.pop()
            else:
                y.append(int(l[commas[-1]+1:len(l)-1]))
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)
    
    if write_file:
        write(x,y,2,"breast_cancer.txt")
    if get_xy:
        return x,y
    
def read_haberman(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/haberman.data", 'r') as f:
        l = f.readline()
        while len(l) > 0:
            commas = [-1]
            for i in range(len(l)):
                if l[i] == ",":
                    commas.append(i)
            x.append([])
            missing = False
            for i in range(len(commas)-1):
                string = l[commas[i]+1:commas[i+1]]
                if string == "?":
                    missing = True
                    break
                else :
                    x[-1].append(float(string))
            
            if missing:
                x.pop()
            else:
                y.append(int(l[commas[-1]+1:len(l)-1]))
            
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)
   
    if write_file:
        write(x,y,2,"haberman.txt")
    if get_xy:
        return x,y
    
def read_dermatology(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/dermatology.data", 'r') as f:
        l = f.readline()
        while len(l) > 0:
            commas = []
            for i in range(len(l)):
                if l[i] == ",":
                    commas.append(i)

            x.append([])
            missing = False
            for i in range(len(commas)-1):
                string = l[commas[i]+1:commas[i+1]]
                if string == "?":
                    missing = True
                    break
                else :
                    x[-1].append(float(string))
            if missing:
                x.pop()
            else:
                y.append(int(l[commas[-1]+1:len(l)-1]))
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)
    
    if write_file:
        write(x,y,6,"dermatology.txt")
    if get_xy:
        return x,y
    
def read_german(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/german-num.data", 'r') as f:
        l = f.readline()
        while len(l) > 0:
            spaces = []
            space_before = False
            for i in range(len(l)):
                if l[i] == " " and not space_before:
                    spaces.append(i)
                    space_before = True
                elif l[i] != " ":
                    space_before = False
                else:
                    space_before = True
            spaces.pop()

            x.append([])
            missing = False
            for i in range(len(spaces)-1):
                string = l[spaces[i]+1:spaces[i+1]]
                if string == "?":
                    missing = True
                    break
                else :
                    x[-1].append(float(string))
            if missing:
                x.pop()
            else:
                y.append(int(l[spaces[-1]+1:len(l)-1]))
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)
    
    if write_file:
        write(x,y,6,"german.txt")
    if get_xy:
        return x,y

def read_seeds(write_file,get_xy):
    y = []
    x = []
    with open("../data/unformatted/seeds_dataset.txt", 'r') as f:
        l = f.readline()
        while len(l) > 0:
            spaces = []
            for i in range(len(l)):
                if l[i] == "\t":
                    spaces.append(i)

            x.append([])
            missing = False
            for i in range(len(spaces)-1):
                string = l[spaces[i]+1:spaces[i+1]]
                if string == "?":
                    missing = True
                    break
                else :
                    x[-1].append(float(string))
            if missing:
                x.pop()
            else:
                y.append(int(l[spaces[-1]+1:len(l)-1]))
            l = f.readline()

    x = np.array(x)
    for j in range(np.shape(x)[1]):
        x_min = np.min(x[:,j])
        x_max = np.max(x[:,j])
        for i in range(np.shape(x)[0]):
            x[i,j] = (x[i,j]-x_min)/(x_max-x_min)
    
    if write_file:
        write(x,y,3,"seeds.txt")
    if get_xy:
        return x,y

############
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# col = ['royalblue','limegreen','gold','orange','red','hotpink']

def creation_dataset(n,p,K):
    X = np.zeros((n,p))
    Y = []
    for i in range(n):
        Y.append(rd.randrange(1,K+1))
        for j in range(p):
            X[i,j] = rd.random()
    return X,Y

def visu_2D(X,Y,K):
    n, p = np.shape(X)
    if p != 2 :
        print("p doit être égal à 2")
    else:
        new_X = [np.zeros((Y.count(k),p)) for k in range(1,K+1)]
        fill = [0 for k in range(K)]
        for i in range(n):
            k = Y[i]-1
            for j in range(p):
                new_X[k][fill[k],j] = X[i,j]
            fill[k] += 1
        for k in range(K):
            plt.scatter(new_X[k][:,0],new_X[k][:,1])
    plt.show()

# X,Y = creation_dataset(20,2,4)
# visu_2D(X,Y,4)

def creation_cercle(n,cx,cy,r,epsilon):
    X = np.zeros((n,2))
    Y = []
    for i in range(n):
        hors_frontiere = False
        x = 0
        y = 0
        dist = 0
        while not hors_frontiere :
            x = rd.random()
            y = rd.random()
            dist = np.sqrt((cx-x)**2+(cy-y)**2)
            hors_frontiere = dist < r - epsilon or dist > r + epsilon
        if dist > r:
            Y.append(2)
        else:
            Y.append(1)
        X[i,0] = x
        X[i,1] = y
    return X,Y

def draw_circle(cx,cy,r):
    theta = np.linspace( 0 , 2 * np.pi , 150 )

    a = cx + r * np.cos( theta )
    b = cy + r * np.sin( theta )
  
    plt.plot( a, b, color = 'black')

#plt.close()
#X,Y = creation_cercle(200,0.5,0.5,0.3,0.01)
#draw_circle(0.5,0.5,0.3)
#visu_2D(X,Y,2)

#####

def blobs(n):
    c = [[0.2,0.8],[0.6,0.7],[0.4,0.3]]
    r = [0.3,0.4,0.3]
    
    X = []
    Y = []
    for i in range(3):
        for j in range(n):
            not_in_0_1 = True
            while not_in_0_1:
                dist = r[i]*(rd.random()*2-1)
                angle = rd.random()*2*np.pi
                x = c[i][0]+dist*np.cos(angle)
                y = c[i][1]+dist*np.sin(angle)
                not_in_0_1 = 0 > x or 1 < x or 0 > y or 1 < y
            X.append([x,y])
            Y.append(i+1)
    return X,Y

for i in range(20):
    X,Y = blobs(50)
    X = np.array(X)
    write(X,Y,3,"/rd_blobs/rd_blobs_50_2_3_"+str(i+1)+".txt")


            
