include("tests.jl")
# include("2D_drawing.jl")

# include("../data/small_random_tests/rd_dataset_20_2_4_1.txt")
# T, obj, error, time, nodes, gap = oct_parmentier(2,0,X,Y,K,quadratic_objective=true)
# draw_class(X,Y,T,K)

test_univarie("iris",4.0,alpha=4,time_l=3600)
test_univarie("seeds",4.0,alpha=4,time_l=3600)
test_univarie("wine",4.0,alpha=4,time_l=3600)
test_univarie("dermatology",5.0,alpha=4,time_l=3600)
test_univarie("breast_cancer",6.0,alpha=4,time_l=3600)
test_univarie("blood_donation",6.0,alpha=4,time_l=3600)