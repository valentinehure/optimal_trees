using Random

"""
Returns the caracteristics vector with all the cross products added
"""
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

"""
Returns the caracteristics vector with each caracteristic's powers up to deg_max
"""
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

"""
Returns the caracteristics vector with all the cross products and each caracteristic's powers up to deg_max added
"""
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

"""
Returns P lists of indexes partioning data such that in each partition each class is as represented as in the original dataset
"""
function partitioning(X::Array{Float64,2},Y::Array{Int64,1},K::Int64,P::Int64)
    n = length(Y)

    indexes = Array{Int64}[]
    for p in 1:P
        push!(indexes,[])
    end

    cpt = 1
    for k in 1:K
        class_index = []
        for i in 1:n
            if Y[i] == k
                push!(class_index,i)
            end
        end
        class_index = hcat(class_index)
        rd = randperm(length(class_index))
        for i in 1:length(class_index)
            push!(indexes[cpt],class_index[rd[i]])
            cpt += 1
            if cpt > P
                cpt = 1
            end
        end  
    end

    return indexes

end
  