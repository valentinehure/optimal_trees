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
Creation of a Tree with depth=0 used when there is no warm-up in the algorithm
"""
function null_Tree()
    this=Tree()
    this.D=0
    this.a=zeros(Float64,0,0)
    this.b=zeros(Float64,0)
    this.c=zeros(Int64,0)
    return(this)
end


"""
Creation of a Tree of greater dimension but with the same structure
"""
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

    return Tree(new_D,a,b,c,T)
end

"""
Prediction of the class given:\n
    - x : vectors of caracteristics
    - T : a tree
"""
function predict_leaf(T::Tree, x::Array{Float64,2})
    n = length(x[:,1])
    p = length(x[1,:])
    leaf = zeros(Int64,n)
    
    for i in 1:n
        t = 1
        for d in 1:T.D
            if sum(T.a[j,t]*x[i,j] for j in 1:p) - T.b[t] <= 0
                t = t*2
            else
                t = t*2 + 1
            end
        end
        leaf[i] = t - (2^T.D - 1)
    end
    return leaf
end

"""
Prediction of the leaf given:\n
    - x : vectors of caracteristics
    - T : a tree
"""
function predict_class(T::Tree, x::Array{Float64,2})
    n = length(x[:,1])
    p = length(x[1,:])
    class = zeros(Int64,n)
    
    for i in 1:n
        t = 1
        for d in 1:T.D
            if sum(T.a[j,t]*x[i,j] for j in 1:p) - T.b[t] <= 0
                t = t*2
            else
                t = t*2 + 1
            end
        end
        class[i] = T.c[t - (2^T.D - 1)]
    end
    return class
end

"""
Writes the tree in a file (preferably a .txt)
"""
function write_tree(T::Tree,filename::String;power::Int64=1,cross_prod::Bool=false)
    touch(filename)
    file = open(filename, "w")
    write(file, string("POWER = ",power,"\n"))
    write(file, string("CROSS_PROD = ",cross_prod,"\n"))
    write(file, string("PROF = ",T.D,"\n"))
    
    nb_attrib = length(T.a[:,1])
    write(file, string("ATTRIB = ",nb_attrib,"\n"))
    write(file, string("NODES = \n"))
    cpt = 1
    for D in 0:(T.D-1)
        for n in 1:2^D
            str = string(cpt," : a =")
            for a in 1:nb_attrib
                str = string(str," ",T.a[a,cpt])
            end
            write(file, string(str,"\n"))
            str = string(cpt," : b = ",T.b[cpt],"\n")
            write(file, str)
            cpt += 1
        end
    end
    write(file, string("LEAVES = \n"))
    for l in 1:(2^T.D)
        write(file, string(l," = ", T.c[l],"\n"))
    end
    close(file)
end

"""
Reads a tree file (preferably a .txt)
"""
function read_tree(filename::String)
    file = open(filename,"r")

    line = readline(file)
    power = parse(Int64, line[9:length(line)])

    line = readline(file)
    cross_prod = line[14] == 't'

    line = readline(file)
    D = parse(Int64, line[8:length(line)])

    line = readline(file)
    nb_attrib = parse(Int64, line[10:length(line)])

    line = readline(file)

    a = zeros(Float64,nb_attrib,2^D-1)
    b = zeros(Float64,2^D-1)

    cpt = 1
    for cpt in 1:(2^D-1)
        line = readline(file)
        debut = findnext(isequal(' '),line,7) + 1
        for att in 1:(nb_attrib-1)
            fin = findnext(isequal(' '),line,debut) - 1
            a[att,cpt] = parse(Float64,line[debut:fin])
            debut = fin + 2
        end
        a[nb_attrib,cpt] = parse(Float64,line[debut:length(line)])
        line = readline(file)
        b[cpt] = parse(Float64,line[9:length(line)])
        cpt += 1
    end
    line = readline(file)

    c = zeros(Int64,2^D)
    for l in 1:(2^T.D)
        line = readline(file)
        c[l] = parse(Int64,line[5:length(line)])
    end

    return Tree(D,a,b,c), power, cross_prod
end