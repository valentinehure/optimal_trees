using JuMP
# using CPLEX
using Gurobi
include("tree.jl")

"""
Function that computes the ancestors of t
Arguments :\n 
    - t : the number of the node (between 1 and 2^D)
    - D : the depth of the tree
Results :\n 
    - AL and AR : list of left (resp. right) ancestors
"""
function compute_LR(t::Int64,D::Int64)
    t_ = t + 2^D - 1 # the numbers for leaf-node should go from 2^D to 2^(D+1) - 1, hence the change to t_
    
    function compute_LR_aux(t::Int64)
        if t == 1
            return [],[]
        end
        AL,AR = compute_LR_aux(t ÷ 2)
        if t % 2 == 1
            AR = append!(AR,t ÷ 2)
        else
            AL = append!(AL,t ÷ 2)
        end
        return AL,AR
    end

    return(compute_LR_aux(t_))
end

function get_tree_oct(x::Array{Float64,2},D::Int64,a::Array{Float64,2},z::Array{Float64,2},c::Array{Int64,1})
    n = length(x[:,1])
    p = length(x[1,:])

    nb_br = 2^D - 1

    a_tree = zeros(Float64,p,nb_br)
    for t in 1:nb_br
        for j in 1:p
            a_tree[j,t] = a[j,t]
        end
    end

    b_tree = zeros(Float64,nb_br)
    eps_tree = zeros(Float64,nb_br)
    b_tree_l = zeros(Float64,nb_br)
    b_tree_r = ones(Float64,nb_br)

    for i in 1:n
        t = argmax(z[i,:])
        A_L,A_R = compute_LR(t,D)
        for m in A_L
            val = sum(a[f,m]*x[i,f] for f in 1:p)
            if val > b_tree_l[m]
                b_tree_l[m] = val
            end
        end
        for m in A_R
            val = sum(a[f,m]*x[i,f] for f in 1:p)
            if val < b_tree_r[m]
                b_tree_r[m] = val
            end
        end
    end

    for t in 1:nb_br
        b_tree[t] = (b_tree_l[t]+b_tree_r[t]) / 2
        eps_tree[t] = (b_tree_r[t] - b_tree_l[t])/2
    end

    return Tree(D,a_tree,b_tree,c,eps=eps_tree)
end

function LeafClassConstraintsAndVariables(m::Model,n::Int64,nb_lf::Int64,K::Int64)
    L = @variable(m, [1:nb_lf], Int, base_name="L")
    N = @variable(m, [1:nb_lf], Int, base_name="N")
    Nk = @variable(m, [1:K, 1:nb_lf], Int, base_name="Nk")
    l = @variable(m, [1:nb_lf], Bin, base_name="l")
    c = @variable(m, [1:K, 1:nb_lf], Bin, base_name="c")

    @constraint(m, [k in 1:K, t in 1:nb_lf], L[t] >= N[t] - Nk[k,t] - n*(1 - c[k,t]))
    @constraint(m, [k in 1:K, t in 1:nb_lf], L[t] <= N[t] - Nk[k,t] + n*c[k,t])
    @constraint(m, [t in 1:nb_lf], L[t] >= 0)
    @constraint(m, [t in 1:nb_lf], l[t] == sum(c[k,t] for k in 1:K))
    
    return L,N,Nk,l,c
end

function TreeVariablesandConstraints_OCTmodel(m::Model,D::Int64,p::Int64;multi_variate::Bool=false)
    nb_br = 2^D - 1
    
    b = @variable(m, [1:nb_br], base_name="b")
    d = @variable(m, d[1:nb_br], Bin, base_name="d")
    @constraint(m,[t in 1:nb_br], 0 <= b[t])
    @constraint(m,[t in 1:nb_br], b[t] <= d[t])
    if multi_variate
        a = @variable(m, [1:p, 1:nb_br], base_name="a")
        @variable(m, a_h[1:p, 1:nb_br], base_name="a_h")
        s = @variable(m, s[1:p, 1:nb_br], Bin, base_name="s")

        @constraint(m, [t in 1:nb_br], sum(a_h[j,t] for j in 1:p) <= d[t])
        @constraint(m, [t in 1:nb_br, j in 1:p], a[j,t] <= a_h[j,t])
        @constraint(m, [t in 1:nb_br, j in 1:p], -a[j,t] <= a_h[j,t])

        @constraint(m, [t in 1:nb_br, j in 1:p], -s[j,t] <= a[j,t])
        @constraint(m, [t in 1:nb_br, j in 1:p], a[j,t] <= s[j,t])

        @constraint(m, [t in 1:nb_br, j in 1:p], s[j,t] <= d[t])
        @constraint(m, [t in 1:nb_br], sum(s[j,t] for j in 1:p) >= d[t])

        @constraint(m, [t in 2:nb_br], d[t] <= d[t ÷ 2])
        return a,b,d,s
    else
        a = @variable(m, [1:p, 1:nb_br], Bin, base_name="a")
        @constraint(m,[t in 1:nb_br], sum(a[j,t] for j in 1:p) == d[t])
        @constraint(m, [t in 2:nb_br], d[t] <= d[t ÷ 2])

        return a,b,d
    end
    
end

"""
Function solving the MIO problem.\n
Arguments :\n
    - D : maximum depth of the resulting tree
    - N_min : minimum number of observation attributed to a leaf
    - x and y : data
    - K : the number of labels
    Optionnal :\n
    - alpha or C : constants linked to the complexity of the tree (one of them should be =/= 0)
    - multi_V : are the branching rules allowed multiple variables?
    - mu : precision of the algorithm
    - solv : a string that will define which solver we will use
    - warm_start : an admissible tree can be used as a warm start to shorten the computionnal time
Result :\n
    - The tree
    - The missclassification on the training set
    - The gap between the optimal solution and the solution found if the time limit was reached.
"""
function oct(D::Int64,Nmin::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64;alpha::Float64=0.0,C::Int64=0,multi_variate::Bool=false,quadratic_objective::Bool=false,quadratic_constraints::Bool=false,mu::Float64=10^(-4),variable_epsilon::Bool=false,post_process::Bool=true,time_limit::Int64=-1)
    n = length(y)
    p = length(x[1,:])

    # computation of the matrix Y (here, Y is in {0,1})
    Y = zeros(Int64,n,K)
    for i in 1:n
        Y[i,y[i]] = 1
    end

    m = Model(Gurobi.Optimizer)

    if time_limit!=-1
        set_time_limit_sec(m,time_limit)
    end

    nb_lf = 2^D
    nb_br = 2^D - 1

    if multi_variate
        a,b,d,s = TreeVariablesandConstraints_OCTmodel(m,D,p,multi_variate=true)
    else
        a,b,d = TreeVariablesandConstraints_OCTmodel(m,D,p,multi_variate=false)   
    end

    @variable(m, z[1:n, 1:nb_lf], Bin, base_name="z")
    
    @variable(m, eps[1:nb_br] >= 0, base_name="eps")
    if variable_epsilon
        @constraint(m, [t in 1:nb_br], eps[t] <= 2*d[t])
    else
        @constraint(m, [t in 1:nb_br], eps[t] == 0)
    end


    for t in 1:nb_lf
        A_L,A_R = compute_LR(t,D)

        if quadratic_constraints
            @constraint(m, [i in 1:n, g in A_L], z[i,t] * (sum(a[j,g]*x[i,j] for j in 1:p) + mu + eps[g] - b[g]) <= 0 )
            @constraint(m, [i in 1:n, g in A_R], z[i,t] * (sum(a[j,g]*x[i,j] for j in 1:p) - b[g]) >= 0)
        else
            if multi_variate
                @constraint(m, [i in 1:n, g in A_L], sum(a[j,g]*x[i,j] for j in 1:p) + mu + eps[g] <= b[g] + (2+mu)*(1-z[i,t]))
                @constraint(m, [i in 1:n, g in A_R], sum(a[j,g]*x[i,j] for j in 1:p) >= b[g] - 2*(1-z[i,t]))    
            else
                @constraint(m, [i in 1:n, g in A_L], sum(a[j,g]*x[i,j] for j in 1:p) + mu + eps[g] <= b[g] + (1+mu)*(1-z[i,t]))
                @constraint(m, [i in 1:n, g in A_R], sum(a[j,g]*x[i,j] for j in 1:p) >= b[g] - (1-z[i,t]))    
            end
        end
        @constraint(m, [i in 1:n, g in A_L], z[i,t]<=d[g])
    end
    

    if C != 0
        if multi_variate
            @constraint(m, sum(s[t] for t in 1:nb_br) <= C)
        else
            @constraint(m, sum(d[t] for t in 1:nb_br) <= C)
        end
    end

    if quadratic_objective
        @variable(m, l[1:nb_lf], Bin, base_name="l")
        @variable(m, c[1:K, 1:nb_lf], Bin, base_name="c")
        @constraint(m, [t in 1:nb_lf], l[t] == sum(c[k,t] for k in 1:K))

        @constraint(m, [i in 1:n, t in 1:nb_lf], z[i,t] <= l[t])
        @constraint(m, [t in 1:nb_lf], sum(z[i,t] for i in 1:n) >= Nmin*l[t])

        @constraint(m, [i in 1:n], sum(z[i,t] for t in 1:nb_lf) == 1)

        if alpha != 0
            if variable_epsilon
                if multi_variate
                    @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) + alpha*sum(s[j,t] for t in 1:nb_br for j in 1:p) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
                else
                    @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) + alpha*sum(d[t] for t in 1:nb_br) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
                end
            else
                if multi_variate
                    @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) + alpha*sum(s[j,t] for t in 1:nb_br for j in 1:p))
                else
                    @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) + alpha*sum(d[t] for t in 1:nb_br))
                end
            end
        else
            if variable_epsilon
                @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
            else
                @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K))
            end
        end
    else
        L,N,Nk,l,c = LeafClassConstraintsAndVariables(m,n,nb_lf,K)
        @constraint(m, [i in 1:n, t in 1:nb_lf], z[i,t] <= l[t])
        @constraint(m, [t in 1:nb_lf], sum(z[i,t] for i in 1:n) >= Nmin*l[t])
        @constraint(m, [k in 1:K, t in 1:nb_lf], Nk[k,t] == sum(Y[i,k]*z[i,t] for i in 1:n))
        @constraint(m, [t in 1:nb_lf], N[t] == sum(z[i,t] for i in 1:n))

        @constraint(m, [i in 1:n], sum(z[i,t] for t in 1:nb_lf) == 1)

        if alpha != 0
            if variable_epsilon
                if multi_variate
                    @objective(m,Min,sum(L[t] for t in 1:nb_lf) + alpha*sum(s[j,t] for t in 1:nb_br for j in 1:p) - 1/(2*nb_br) * sum(eps[n] for n in 1:nb_br))
                else
                    @objective(m,Min,sum(L[t] for t in 1:nb_lf) + alpha*sum(d[t] for t in 1:nb_br) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
                end
            else
                if multi_variate
                    @objective(m,Min,sum(L[t] for t in 1:nb_lf) + alpha*sum(s[j,t] for t in 1:nb_br for j in 1:p))
                else
                    @objective(m,Min,sum(L[t] for t in 1:nb_lf) + alpha*sum(d[t] for t in 1:nb_br))
                end
            end
        else
            if variable_epsilon
                @objective(m,Min,sum(L[t] for t in 1:nb_lf) - 1/((1+multi_variate)*nb_br) * sum(eps[n] for n in 1:nb_br))
            else
                @objective(m,Min,sum(L[t] for t in 1:nb_lf))
            end
        end
    end
        
    #JuMP.write_to_file(m, "output_lp.lp")

    optimize!(m)

    gap=0

    if termination_status(m) != MOI.OPTIMAL
        gap=abs(JuMP.objective_bound(m) - JuMP.objective_value(m)) / JuMP.objective_value(m)
    end

    class =  mapslices(argmax,value.(c),dims=1)[:] # it should give us the indexes such that c[:,t] is equal to 1
    
    nodes = MOI.get(m, MOI.NodeCount())

    if variable_epsilon
        T = Tree(D,value.(a),value.(b)-value.(eps)/2,class,eps = value.(eps))
    else
        if post_process
            T = get_tree_oct(x,D,value.(a),value.(z),class)
        else
            T = Tree(D,value.(a),value.(b),class)
        end
    end

    c_ = value.(c)
    z_ = value.(z)
    errors = sum(c_[k,t]*sum((1-Y[i,k])*z_[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K)

    return(T, objective_value(m), errors, solve_time(m), nodes, gap, c_)
end
