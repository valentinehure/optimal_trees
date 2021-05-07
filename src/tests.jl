include("oct_mip.jl")
include("flow.jl")
using CSV
using DataFrames

function add_cross_product(x::Array{Float64,2})
    n = length(x[:,1])
    p = length(x[1,:])
    x_prod = zeros(Float64,n,p*(p-1)÷2)
    for i in 1:n
        count = 1
        for j in 1:p
            for j_ in (j+1):p
                x_prod[i,count] = x[i,j]*x[i,j_]
                count += 1
            end
        end
    end
    return hcat(x,x_prod)
end

function add_powers(x::Array{Float64,2},deg_max::Int64)
    n = length(x[:,1])
    p = length(x[1,:])
    new_x = copy(x)
    x_deg = copy(x)
    for deg in 2:deg_max
        for i in 1:n
            for j in 1:p
                x_deg[i,j] *= x[i,j]
            end
        end
        new_x = hcat(new_x,x_deg)
    end
    return new_x
end

function add_cross_product_and_powers(x::Array{Float64,2},deg_max::Int64)
    n = length(x[:,1])
    p = length(x[1,:])
    x_prod = zeros(Float64,n,p*(p-1)÷2)
    for i in 1:n
        count = 1
        for j in 1:p
            for j_ in (j+1):p
                x_prod[i,count] = x[i,j]*x[i,j_]
                count += 1
            end
        end
    end
    new_x = copy(x)
    x_deg = copy(x)
    for deg in 2:deg_max
        for i in 1:n
            for j in 1:p
                x_deg[i,j] *= x[i,j]
            end
        end
        new_x = hcat(new_x,x_deg)
    end
    return hcat(new_x,x_prod)
end

function epsilon_pb()
    list = cd(readdir,"../data/small_random_tests")[1:10]
    nb_datasets = length(list)
    erreur = 0
    for data in 1:nb_datasets
        include(string("../data/small_random_tests/",list[data]))
        Xpcp = add_cross_product_and_powers(X,2)
        Xp = add_powers(X,2)
        Xcp = add_cross_product(X)
        
        T,obj,gap = flow_adapted(2,X,Y,K,0.0)
        Y_pred = predict_class(T,X)
        obj_verif = 20
        for i in 1:20
            if Y_pred[i] != Y[i]
                obj_verif -= 1
            end
        end
        if obj_verif != obj
            erreur += 1
        end

        T,obj,gap = flow_adapted(3,X,Y,K,0.0)
        Y_pred = predict_class(T,X)
        obj_verif = 20
        for i in 1:20
            if Y_pred[i] != Y[i]
                obj_verif -= 1
            end
        end
        if obj_verif != obj
            erreur += 1
        end

    end
    return erreur
end

function small_tests()
    list = cd(readdir,"../data/small_random_tests")[1:10]
    nb_datasets = length(list)
    df = zeros(Float64,150,11)
    for data in 1:nb_datasets
        println(data)
        for i in 1:15
            df[15*(data-1)+i,1] = data
            df[15*(data-1)+i,2] = 0.005
            df[15*(data-1)+i,3] = 0.01
        end
        include(string("../data/small_random_tests/",list[data]))
        for D in 1:3
            print(D)
            t = time()
            T,obj = flow_adapted(D,X,Y,K,0.0,false,0.0,0.01)
            t = time() - t
            df[15*(data-1)+i,4] = 0
            df[15*(data-1)+i,5] = 0
            df[15*(data-1)+i,6] = 0
            df[15*(data-1)+i,7] = 0
            df[15*(data-1)+i,8] = D
            df[15*(data-1)+i,9] = t
            df[15*(data-1)+i,10] = obj
        end
        for D in 1:3
            print(D)
            t = time()
            T,obj = flow_adapted(D,X,Y,K,0.0,true,0.0,0.01)
            t = time() - t
            df[15*(data-1)+3+i,4] = 1
            df[15*(data-1)+3+i,5] = 0
            df[15*(data-1)+3+i,6] = 0
            df[15*(data-1)+3+i,7] = 0
            df[15*(data-1)+3+i,8] = D
            df[15*(data-1)+3+i,9] = t
            df[15*(data-1)+3+i,10] = obj
        end
        Xp2 = add_powers(X,2)
        for D in 1:3
            print(D)
            t = time()
            T,obj = flow_adapted(D,Xp2,Y,K,0.0,true,0.0,0.01)
            t = time() - t
            df[15*(data-1)+6+i,4] = 1
            df[15*(data-1)+6+i,5] = 1
            df[15*(data-1)+6+i,6] = 0
            df[15*(data-1)+6+i,7] = 0
            df[15*(data-1)+6+i,8] = D
            df[15*(data-1)+6+i,9] = t
            df[15*(data-1)+6+i,10] = obj
        end
        Xcp2 = add_cross_product_and_powers(X,2)
        for D in 1:3
            print(D)
            t = time()
            T,obj = flow_adapted(D,Xcp2,Y,K,0.0,true,0.0,0.01)
            t = time() - t
            df[15*(data-1)+9+i,4] = 1
            df[15*(data-1)+9+i,5] = 1
            df[15*(data-1)+9+i,6] = 1
            df[15*(data-1)+9+i,7] = 0
            df[15*(data-1)+9+i,8] = D
            df[15*(data-1)+9+i,9] = t
            df[15*(data-1)+9+i,10] = obj
        end
        Xcp3 = add_cross_product_and_powers(X,2)
        for D in 1:3
            print(D)
            t = time()
            T,obj = flow_adapted(D,Xcp2,Y,K,0.0,true,0.0,0.01)
            t = time() - t
            df[15*(data-1)+12+i,4] = 1
            df[15*(data-1)+12+i,5] = 1
            df[15*(data-1)+12+i,6] = 1
            df[15*(data-1)+12+i,7] = 1
            df[15*(data-1)+12+i,8] = D
            df[15*(data-1)+12+i,9] = t
            df[15*(data-1)+12+i,10] = obj
        end        
    end
    df = DataFrame(df)  
    CSV.write("../results_small_test_2.csv", df)
end