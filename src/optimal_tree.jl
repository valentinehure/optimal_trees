include("model_parmentier.jl")
include("flow.jl")
# using DecisionTree
# pour trouver comment récupérer l'arbre CART
# https://discourse.julialang.org/t/how-to-visualise-the-structure-of-the-decision-tree-built-by-mlj/30946/7

function optimal_tree(D_max::Int64,Nmin::Int64,x_train::Array{Float64,2},x_test::Array{Float64,2},y_train::Array{Int64,1},y_test::Array{Int64,1},K::Int64;multi_variate::Bool=false,quadratic_constraints::Bool=false,quadratic_objective::Bool=false,mu::Float64=10^(-4),variable_epsilon::Bool=false,time_limit::Int64=-1,power::Int64=1,cross_prod::Bool=false)
    n_train = length(y_train)
    p = length(x_train[1,:])

    C_max = 2^D_max-1
    if multi_V
        C_max *= p
    end

    nb_errors = n_train * ones(Int64,D_max,C_max)
    tree_name = fill("",(D_max,C_max))

    for D in 1:D_max
        for C in 1:C_max
            least_errors = argmin(nb_errors[1:D,1:C])
            if tree_name[least_errors] == ""
                starting_T = null_Tree()
            else
                starting_T, power, cross_prod = read_tree(tree_name[least_errors])
            end
            T,obj,errors,time,nodes,gap = oct(D,Nmin,x_train,y_train,K,C=C,multi_variate=multi_variate,quadratic_objective=quadratic_objective,quadratic_constraints=quadratic_constraints,mu=mu,variable_epsilon=variable_epsilon,time_limit=time_limit)
            write_tree(T,string("memory/tree",D,"_",C,".txt"),power=power,cross_prod=cross_prod)
            nb_errors[D,C] = errors
            tree_name[D,C] = string("memory/tree",D,"_",C,".txt")
        end
    end

    best_trees = fill("",Dmax)
    best_C = zeros(Int64,D_max)
    for D in 1:D_max
        best_C[D] = argmin(nb_errors[D,:])
        best_trees[D] = tree_name[D,best_C[D]]
    end

    n_test = length(y_test)
    best_score = n_test
    best_tree = -1

    for D in 1:D_max
        T, power, cross_prod = read_tree(best_trees[D])
        y_pred = predict_class(T,x_test)
        score = 0
        for n in 1:n_test
            if y_pred[n] != y_test[n]
                score += 1
            end
        end
        if (best_tree == -1) || (score < best_score)
            best_score = score
            best_tree = D
        end
    end

    C_star = best_C[best_tree]
    tree, power, cross_prod = read_tree(best_trees[best_tree])
    return (tree,best_score/n_test, C_star)
end