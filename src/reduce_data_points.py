import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn import datasets
from numpy.linalg import norm
from quadprog import solve_qp

## https://towardsdatascience.com/clustering-using-convex-hulls-fddafeaa963c

def visu_hull():
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)

    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
    point = np.random.rand(2)
    plt.plot(point[0], point[1], 'o',color="r")
    return proj2hull(point,hull.equations)

def proj2hull(z, equations):
    G = np.eye(len(z), dtype=float)
    a = np.array(z, dtype=float)
    C = np.array(-equations[:, :-1], dtype=float)
    b = np.array(equations[:, -1], dtype=float)
    x, f, xu, itr, lag, act = solve_qp(G, a, C.T, b, meq=0, factorized=True)
    return np.linalg.norm(z-x)
            
def proj2smallensemble_rec(P,X):
    N = len(X)
    n = len(P)
    if N == 1:
        return(X[0], np.linalg.norm(P - X[0]))  
    else:    
        M = np.zeros((n + N - 1, n + N - 1))
        V = np.zeros((n + N - 1, 1))
        
        for i in range(n):
            M[i,i] = 1
            for j in range(N-1):
                M[i,n+j] = X[0][i] - X[j+1][i]
                M[n+j,i] = X[j+1][i] - X[0][i]
                V[n+j,0] += P[i]*(X[j+1][i] - X[0][i])
            V[i,0] = X[0][i]
            
        M = np.linalg.inv(M)
        result = np.dot(M,V)
        H = result[:n,:]
        H = H.transpose()
        lambd = result[n:,:][0].tolist()
        
        if sum(lambd)>1 or min(lambd) < 0:
            dist = np.inf
            Hprime = 0
            for j in range(N):
                new_Hprime, new_dist = proj2smallensemble_rec(H,[X[i] for i in range(N) if i!=j])
                if new_dist < dist:
                    dist = new_dist
                    Hprime = new_Hprime
            return(Hprime, dist)
        else:
            return(H, np.linalg.norm(P - H))

def proj2smallensemble(P,X):    
    H, dist = proj2smallensemble_rec(P,X)
    return np.linalg.norm(P - H)
    
def find_identical(X,Y):
    pair = []
    same_class = []
    for i in range(np.shape(X)[0]):
        for j in range(i+1,np.shape(X)[0]):
            if np.linalg.norm(X[i,:]-X[j,:]) <= 10**(-6):
                pair.append([i,j])
                same_class.append(Y[i]==Y[j])
    
    return pair, same_class

def remove_identical(X,Y):
    pair, same_class = find_identical(X,Y)
    
    removal_list = []
    to_be_paired = []
    alone_clusters = []
    
    for i in range(len(pair)):
        if same_class[i]:
            removal_list.append(pair[i][1])
            to_be_paired.append(pair[i])
        else:
            removal_list += pair[i]
            alone_clusters.append([pair[i][0]])
            alone_clusters.append([pair[i][1]])
    
    return removal_list,to_be_paired,alone_clusters
    
def make_hulls(X,Y):
    X = np.array(X)
    removal_list,to_be_paired,alone_clusters = remove_identical(X,Y)

    X = np.array([X[i,:] for i in range(np.shape(X)[0]) if removal_list.count(i) == 0])
    
    n = np.shape(X)[1]
    clusters = alone_clusters
    
    nb_pts = len(X[:,0])
    
    pts = [i for i in range(len(X[:,0]))]
    
    dist = np.zeros((nb_pts,nb_pts))
    for i in range(nb_pts):
        dist[i,i] = np.inf
        for j in range(i+1,nb_pts):
            dist[i,j] = norm(X[i,:]-X[j,:])
            dist[j,i] = norm(X[i,:]-X[j,:])
    
    keep_going = True
    
    while len(pts) > 0 and keep_going:
        min_dist = "nothing"
        for i in pts:
            closest = np.argmin(dist[i,:])
            if Y[closest] == Y[i] and pts.count(closest) > 0:
                closest_of_closest = np.argmin(dist[closest,:])
                if Y[closest_of_closest] == Y[i]:
                    if min_dist == "nothing":
                        min_dist = [i,closest,np.min(dist[i,:])]
                    elif min_dist[1] > np.min(dist[i,:]):
                        min_dist = [i,closest,np.min(dist[i,:])]
        
        if min_dist == "nothing":
            keep_going = False
        else :
            cluster = min_dist[:2]
            cluster_dim = np.linalg.matrix_rank(np.array([X[i,:] for i in cluster]))
            # ordonner les points les plus proches
            hull_dist = []
            for i in range(nb_pts):
                if cluster.count(i) > 0:
                    hull_dist.append(np.inf)
                else:
                    hull_dist.append(proj2smallensemble(X[i,:],[X[cluster[0],:],X[cluster[1],:]]))
            
            first_min_same_class = (Y[np.argmin(hull_dist)] == Y[cluster[0]] and pts.count(np.argmin(hull_dist)) > 0)
            while first_min_same_class:
                min_same_class = True
                added_point = False
                while min_same_class :
                    if Y[np.argmin(dist[np.argmin(hull_dist),:])] == Y[cluster[0]] :
                        added_point = True
                        if cluster_dim <= n:
                            cluster_dim = np.linalg.matrix_rank(np.array([X[i,:] for i in cluster]))
                    hull_dist[np.argmin(hull_dist)] = np.inf
                    min_same_class = (Y[np.argmin(hull_dist)] == Y[cluster[0]] and pts.count(np.argmin(hull_dist)) > 0)
                
                if added_point:
                    hull_dist = []
                    if cluster_dim > n:
                        hull = ConvexHull([X[i,:] for i in cluster])
                    for i in range(nb_pts):
                        if cluster.count(i) > 0:
                            hull_dist.append(np.inf)
                        else:
                            if cluster_dim <= n:
                                hull_dist.append(proj2smallensemble(X[i,:],[X[j,:] for j in cluster]))
                            else:
                                hull_dist.append(proj2hull(X[i,:],hull.equations))
                    first_min_same_class = (Y[np.argmin(hull_dist)] == Y[cluster[0]] and pts.count(np.argmin(hull_dist)) > 0)
                else:
                    first_min_same_class = False
            
            clusters.append(cluster)
            for i in cluster:
                pts.pop(pts.index(i))
    
    if len(pts)>0:
        for i in range(len(pts)):
            clusters.append([pts[i]])
    
    for i in range(len(to_be_paired)):
        for j in range(len(clusters)):
            if clusters[j].count(to_be_paired[0]) > 0:
                clusters[j].append(to_be_paired[1])
                break
    return(clusters)

def test(n=30,K=4):
    if K > 8:
        print("Too much colors")
        return 0
    
    color = ["blue","red","green","yellow","orange","purple","pink","grey"]
    
    X = np.random.rand(n, 2)
    Y = [np.random.randint(0,K) for i in range(n)]
    
    plt.subplot(1,2,1)
    for i in range(K):
        X_1 = []
        X_2 = []
        for j in range(n):
            if Y[j] == i:
                X_1.append(X[j,0])
                X_2.append(X[j,1])
        plt.scatter(X_1,X_2,color=color[i])      
    plt.xlim([0,1])
    plt.ylim([0,1])          
    clusters = make_hulls(X,Y)
    
    plt.subplot(1,2,2)
    for i in range(len(clusters)):
        if len(clusters[i]) > 1:
            points = np.array([[X[j,0],X[j,1]] for j in clusters[i]])
            plt.plot(points[:,0], points[:,1], 'o',color=color[Y[clusters[i][0]]])
            if len(clusters[i]) > 2 : 
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            else:
                plt.plot(points[:,0], points[:,1], 'k-')
        else:
            plt.plot(X[clusters[i][0],0], X[clusters[i][0],1], 'o',color=color[Y[clusters[i][0]]])
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    return X,Y,clusters

def iris_make_hull():
    iris = datasets.load_iris()
    Y = iris.target
    
    iris = np.array([iris.data[i,:] for i in range(np.shape(iris.data)[0]) if i != 142])
    for i in range(np.shape(iris)[1]):
        max_i = max(iris[:,i])
        min_i = min(iris[:,i])
        for j in range(np.shape(iris)[0]):
            iris[j,i] = (iris[j,i] - min_i) / (max_i - min_i)
#    color = ["blue","red","green"]
    
#    plt.subplot(1,2,1)

    X = iris[:, :2]
    clusters = make_hulls(X,Y)
    nb_cl_1 = len(clusters)
    print("Premier fait \n")
#    for i in range(len(clusters)):
#        if len(clusters[i]) > 1:
#            points = np.array([[X[j,0],X[j,1]] for j in clusters[i]])
#            plt.plot(points[:,0], points[:,1], 'o',color=color[Y[clusters[i][0]]])
#            if np.linalg.matrix_rank(points) > 2 : 
#                hull = ConvexHull(points)
#                for simplex in hull.simplices:
#                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#            else:
#                plt.plot(points[:,0], points[:,1], 'k-')
#        else:
#            plt.plot(X[clusters[i][0],0], X[clusters[i][0],1], 'o',color=color[Y[clusters[i][0]]])
#    
#    plt.subplot(1,2,2)
    X = iris[:, 2:]
    clusters = make_hulls(X,Y)
    nb_cl_2 = len(clusters)
    print("Deuxieme fait \n")
#    for i in range(len(clusters)):
#        if len(clusters[i]) > 1:
#            points = np.array([[X[j,0],X[j,1]] for j in clusters[i]])
#            plt.plot(points[:,0], points[:,1], 'o',color=color[Y[clusters[i][0]]])
#            if np.linalg.matrix_rank(points) > 2 : 
#                hull = ConvexHull(points)
#                for simplex in hull.simplices:
#                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#            else:
#                plt.plot(points[:,0], points[:,1], 'k-')
#        else:
#            plt.plot(X[clusters[i][0],0], X[clusters[i][0],1], 'o',color=color[Y[clusters[i][0]]])
#    
#    plt.show()
    
    X = iris
    clusters = make_hulls(X,Y)
    nb_cl = len(clusters)
    print("Tout fait \n")
    
    return(nb_cl_1,nb_cl_2,nb_cl)