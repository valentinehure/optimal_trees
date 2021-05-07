include("tests.jl")
include("2D_drawing.jl")

include("../data/small_random_tests/rd_dataset_20_2_4_1.txt")

#X = add_cross_product_and_powers(X,4)
# X = add_powers(X,2)


# T,obj = oct(2,X,Y,K,0.0,20)
T,obj = flow_adapted(3,X,Y,K,0.0,true,10^(-7),0.0)
# T,obj = main_problem(2,X,Y,K,0.0,true,0.005,0.01)

draw_class(X,Y,T,K)
#draw_leaves(X,Y,T,K,2)

# small_tests()
#epsilon_pb()

