include("tests.jl")
# include("2D_drawing.jl")

# include("../data/ecoli.txt")

# X = add_powers(X,2)
# T,obj,time,nodes = oct_quad(2,0,X,Y,K,0.0,20,true,10^(-4),false)
# T,obj,time,nodes = flow_adapted_quad(2,X,Y,K,0.0,true,10^(-4),false)
# println(T, obj)

# draw_class(X,Y,T,K,2)

# iris_test(5)
# iris_test(3)

# cplex_vs_gurobi()

test("../data/ecoli.txt","ecoli")
test("../data/wine.txt","wine")
test("../data/blood_donation.txt","blood_donation")
test("../data/breast_cancer.txt","breast_cancer")
