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

function small_tests()
    list = cd(readdir,"../data/small_random_tests")
    nb_datasets = length(list)
    U2_r = zeros(Float64,nb_datasets)
    U2_t = zeros(Float64,nb_datasets)
    U3_r = zeros(Float64,nb_datasets)
    U3_t = zeros(Float64,nb_datasets)
    M2_r = zeros(Float64,nb_datasets)
    M2_t = zeros(Float64,nb_datasets)
    M3_r = zeros(Float64,nb_datasets)
    M3_t = zeros(Float64,nb_datasets)
    M2p_r = zeros(Float64,nb_datasets)
    M2p_t = zeros(Float64,nb_datasets)
    M3p_r = zeros(Float64,nb_datasets)
    M3p_t = zeros(Float64,nb_datasets)
    M2cp_r = zeros(Float64,nb_datasets)
    M2cp_t = zeros(Float64,nb_datasets)
    M3cp_r = zeros(Float64,nb_datasets)
    M3cp_t = zeros(Float64,nb_datasets)
    M2pcp_r = zeros(Float64,nb_datasets)
    M2pcp_t = zeros(Float64,nb_datasets)
    M3pcp_r = zeros(Float64,nb_datasets)
    M3pcp_t = zeros(Float64,nb_datasets)
    for data in 1:nb_datasets
        include(string("../data/small_random_tests/",list[data]))
        Xpcp = add_cross_product_and_powers(X,2)
        Xp = add_powers(X,2)
        Xcp = add_cross_product(X)
        
        U2_t[data] = time()
        T,obj,gap = flow_adapted(2,X,Y,K,0.0)
        U2_t[data] = time() - U2_t[data]
        U2_r[data] = obj

        M2_t[data] = time()
        T,obj,gap = flow_adapted(2,X,Y,K,0.0,true)
        M2_t[data] = time() - M2_t[data]
        M2_r[data] = obj

        M2p_t[data] = time()
        T,obj,gap = flow_adapted(2,Xp,Y,K,0.0,true)
        M2p_t[data] = time() - M2p_t[data]
        M2p_r[data] = obj

        M2cp_t[data] = time()
        T,obj,gap = flow_adapted(2,Xcp,Y,K,0.0,true)
        M2cp_t[data] = time() - M2cp_t[data]
        M2cp_r[data] = obj

        M2pcp_t[data] = time()
        T,obj,gap = flow_adapted(2,Xpcp,Y,K,0.0,true)
        M2pcp_t[data] = time() - M2pcp_t[data]
        M2pcp_r[data] = obj

        U3_t[data] = time()
        T,obj,gap = flow_adapted(3,X,Y,K,0.0)
        U3_t[data] = time() - U3_t[data]
        U3_r[data] = obj

        M3_t[data] = time()
        T,obj,gap = flow_adapted(3,X,Y,K,0.0,true)
        M3_t[data] = time() - M3_t[data]
        M3_r[data] = obj

        M3p_t[data] = time()
        T,obj,gap = flow_adapted(3,Xp,Y,K,0.0,true)
        M3p_t[data] = time() - M3p_t[data]
        M3p_r[data] = obj

        M3cp_t[data] = time()
        T,obj,gap = flow_adapted(3,Xcp,Y,K,0.0,true)
        M3cp_t[data] = time() - M3cp_t[data]
        M3cp_r[data] = obj

        M3pcp_t[data] = time()
        T,obj,gap = flow_adapted(3,Xpcp,Y,K,0.0,true)
        M3pcp_t[data] = time() - M3pcp_t[data]
        M3pcp_r[data] = obj
    end
    df = DataFrame(Dataset = list, 
            Time_U2 = U2_t,
            Results_U2 = U2_r,
            Time_M2 = M2_t,
            Results_M2 = M2_r,
            Time_M2p = M2p_t,
            Results_M2p = M2p_r,
            Time_M2cp = M2cp_t,
            Results_M2cp = M2cp_r,
            Time_M2pcp = M2pcp_t,
            Results_M2pcp = M2pcp_r,
            Time_U3 = U3_t,
            Results_U3 = U3_r,
            Time_M3 = M3_t,
            Results_M3 = M3_r,
            Time_M3p = M3p_t,
            Results_M3p = M3p_r,
            Time_M3cp = M3cp_t,
            Results_M3cp = M3cp_r,
            Time_M3pcp = M3pcp_t,
            Results_M3pcp = M3pcp_r
               )  
    CSV.write("../results_small_test.csv", df)
end