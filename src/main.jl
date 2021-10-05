include("tests.jl")
#include("2D_drawing.jl")

# function test(T,X_test,Y_test)
#     Y_pred = predict_class(T,X_test)
#     errors = Int64[]
#     for i in 1:length(Y_pred)
#         if Y_pred[i] != Y_test[i]
#             push!(errors,i)
#         end
#     end
#     return errors
# end

# include("../data/iris.txt")
# Random.seed!(0)
# train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)
# X_train = X[train_indexes[1],:]
# Y_train = Y[train_indexes[1]]
# X_test = X[test_indexes[1],:]
# Y_test = Y[test_indexes[1]]
# T1, obj1, error1, time1, nodes1, gap1 = flow(2,X_train,Y_train,K,alpha=4.0,multi_variate=true,variable_epsilon = true)
# T2, obj2, error2, time2, nodes2, gap2 = oct(2,0,X_train,Y_train,K,alpha=4.0,quadratic_objective=true,multi_variate=true,variable_epsilon = true)
#draw_class(X,Y,T,K)

# tests("iris",3;alpha=4.0,time_l=3600)
# tests("seeds",3;alpha=4.0,time_l=3600)
# tests("wine",3;alpha=4.0,time_l=3600)
# tests("dermatology",3;alpha=5.0,time_l=3600)
# tests("breast_cancer",3;alpha=6.0,time_l=3600)
# tests("blood_donation",3;alpha=6.0,time_l=3600)
# tests("german",3;alpha=7.0,time_l=3600)

# aggregate_results_of_tests("iris",3;alpha=4.0,time_l=3600)
# aggregate_results_of_tests("seeds",3;alpha=4.0,time_l=3600)
# aggregate_results_of_tests("wine",3;alpha=4.0,time_l=3600)
# aggregate_results_of_tests("dermatology",3;alpha=5.0,time_l=3600)
# aggregate_results_of_tests("breast_cancer",3;alpha=6.0,time_l=3600)
# aggregate_results_of_tests("blood_donation",3;alpha=6.0,time_l=3600)
# aggregate_results_of_tests("german",3;alpha=7.0,time_l=3600)

# epsilon_tests("iris",3;alpha=3.0,time_l=1800)
epsilon_tests("seeds",3;alpha=4.0,time_l=1800)

# aggregate_results_of_epsilon_tests("iris",3;alpha=3.0,time_l=1800)
# epsilon_tests("seeds",3;alpha=4.0,time_l=1800)
