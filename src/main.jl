include("tests.jl")
include("2D_drawing.jl")


## screen iris_test
# test("wine",3,3600)
# test("seeds",3,3600)
# test("blood_donation",3,3600)
# test("breast_cancer",3,3600)
# test("dermatology",3,3600)

## screen audition
# include("../data/cercle_200.txt")
# X = add_powers(X,2)

# obj = 1
# D = 2
# while obj >= 1
#     T, obj, time, node, gap = oct(D,0,X,Y,K,multi_V=true,quad=true)
#     global D += 1
# end

# write_tree(T,"arbre_lin_audition.txt")

include("../data/small_random_tests/rd_dataset_20_2_4_1.txt")
X = add_powers(X,2)
T, obj, time, node, gap = oct(2,0,X,Y,K,multi_V=true,quad=true,epsi_var=true)
draw_class(X,Y,T,K,2)