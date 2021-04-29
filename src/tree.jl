"""
Tree structure
"""
mutable struct Tree
    D::Int64
    a::Array{Float64,2}
    b::Array{Float64,1}
    c::Array{Int64,1}

    function Tree()
        return new()
    end
 
end

"""
Creation of a tree\n
Arguments :\n
    - D : maximum depth
    - a and b : values of the branching rules (that are a.x <= b)
    - c : leaves' labels (even if the leave cant be attained, it will be labelled, by default we will try to label those as 0)
"""
function Tree(D::Int64,a::Array{Float64,2},b::Array{Float64,1},c::Array{Int64,1})
    this=Tree()
    this.D=D
    this.a=a
    this.b=b
    this.c=c
    return(this)
end

"""
Create a Tree with depth=0 used when there is no warm-up in the algorithm
"""
function null_Tree()
    this=Tree()
    this.D=0
    this.a=zeros(Float64,0,0)
    this.b=zeros(Float64,0)
    this.c=zeros(Int64,0)
    return(this)
end

function bigger_Tree(T::Tree, new_D::Int64)
    p = length(T.a[:,1])
    a = ones(Float64,p,2^new_D-1)
    b = ones(Float64,2^new_D-1)
    c = zeros(Int64,2^new_D)

    for t in 1:(2^T.D-1)
        b[t] = T.b[t]
        for j in 1:p
            a[j,t] = T.a[j,t]
        end
    end

    for t in 2^T.D:(2^new_D-1)
        b[t] = 0
        for j in 1:p
            a[j,t] = 0
        end
    end

    dif = new_D - T.D
    for t in 1:2^T.D
        c[t*(2^dif)] = T.c[t] # only the righest leave will carry the label (others' label do not matter)
    end

    return Tree(new_D,a,b,c)
end

function predict_leaf(T::Tree, x::Array{Float64,2})
    n = length(x[:,1])
    p = length(x[1,:])
    leaf = zeros(Int64,n)
    
    for i in 1:n
        t = 1
        for d in 1:T.D
            if sum(T.a[j,t]*x[i,j] for j in 1:p) < T.b[t]
                t = t*2
            else
                t = t*2 + 1
            end
        end
        leaf[i] = t - (2^T.D - 1)
    end
    return leaf
end

function predict_class(T::Tree, x::Array{Float64,2})
    n = length(x[:,1])
    p = length(x[1,:])
    class = zeros(Int64,n)
    
    for i in 1:n
        t = 1
        for d in 1:T.D
            if sum(T.a[j,t]*x[i,j] for j in 1:p) < T.b[t]
                t = t*2
            else
                t = t*2 + 1
            end
        end
        class[i] = T.c[t - (2^T.D - 1)]
    end
    return class
end