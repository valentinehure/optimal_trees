from scipy.spatial import ConvexHull


## https://towardsdatascience.com/clustering-using-convex-hulls-fddafeaa963c

## faire boucle sur les points
##       si ya un triangle de meme classe sans rien dedans
##              boucle tant que c'est plus possible
##                  prendre point le plus proche et crée nouveau convex hull
