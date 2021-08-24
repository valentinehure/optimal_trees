using JuMP
using CPLEX
using Gurobi
include("tree.jl")

function get_tree_flow(x::Array{Float64,2},D::Int64,a::Array{Float64,2},z::Array{Float64,2},c::Array{Int64,1})
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
        if z[i,1] != 0
            for d in 1:D
                t = 2^d - 1 + argmax(z[i,2^d:(2^(d+1)-1)])
                
                if z[i,t] > 0 # it could be == 0 because of the flow thingy
                    m = t ÷ 2
                    val = sum(a[f,m]*x[i,f] for f in 1:p)
                    if t % 2 == 0
                        if val > b_tree_l[m]
                            b_tree_l[m] = val
                        end
                    else
                        if val < b_tree_r[m]
                            b_tree_r[m] = val
                        end
                    end
                end
            end
        end
    end

    for t in 1:nb_br
        b_tree[t] = (b_tree_l[t]+b_tree_r[t]) / 2
        eps_tree[t] = (b_tree_r[t] - b_tree_l[t])/2
    end

    return Tree(D,a_tree,b_tree,c,eps=eps_tree)
end


function TreeVariablesandConstraints_Flow(m::Model,D::Int64,K::Int64,p::Int64;multi_variate::Bool=false)
    nb_br = 2^D - 1
    nb_lf = 2^D
    
    b = @variable(m, [1:nb_br], base_name="b")
    d = @variable(m, d[1:nb_br], Bin, base_name="d")
    w = @variable(m,[t in 1:(nb_br+nb_lf), k in 1:K], Bin, base_name="w")

    @constraint(m, [t in (nb_br+1):(nb_br+nb_lf)], sum(w[t,k] for k in 1:K) == 1)
    @constraint(m,[t in 1:nb_br], 0 <= b[t])
    @constraint(m,[t in 1:nb_br], b[t] <= d[t])
    if multi_variate
        a = @variable(m, [1:p, 1:nb_br], base_name="a")
        @variable(m, a_h[1:p, 1:nb_br], base_name="a_h")
        @variable(m, s[1:p, 1:nb_br], Bin, base_name="s")

        @constraint(m, [t in 1:nb_br], sum(a_h[j,t] for j in 1:p) <= d[t])
        @constraint(m, [t in 1:nb_br, j in 1:p], a[j,t] <= a_h[j,t])
        @constraint(m, [t in 1:nb_br, j in 1:p], -a[j,t] <= a_h[j,t])

        @constraint(m, [t in 1:nb_br, j in 1:p], -s[j,t] <= a[j,t])
        @constraint(m, [t in 1:nb_br, j in 1:p], a[j,t] <= s[j,t])

        @constraint(m, [t in 1:nb_br, j in 1:p], s[j,t] <= d[t])
        @constraint(m, [t in 1:nb_br], sum(s[j,t] for j in 1:p) >= d[t])

        @constraint(m, [t in 1:nb_br], d[t] + sum(w[t,k] for k in 1:K) == 1)

        return a,b,d,w,s
    else
        a = @variable(m, [1:p, 1:nb_br], Bin, base_name="a")

        @constraint(m,[t in 1:nb_br], sum(a[j,t] for j in 1:p) == d[t])
        @constraint(m, [t in 1:nb_br], d[t] + sum(w[t,k] for k in 1:K) == 1)
        return a,b,d,w
    end
end

function flow_get_class(D::Int64,w::Array{Float64,2})
    class = zeros(Int64,2^D)

    nb_br = 2^D-1
    nb_lf = 2^D

    for t in 1:nb_br
        if maximum(w[t,:]) != 0
            index = t
            leaves = 1
            while index <= nb_br
                index = index*2
                leaves *= 2
            end
            index -= nb_br + 1
            for l in 1:leaves
                if class[index + l] == 0
                    class[index + l] = argmax(w[t,:])
                end
            end
        end
    end

    for t in 1:nb_lf
        if class[t] == 0
            class[t] = argmax(w[nb_br+t,:])
        end
    end

    return class

end

function flow(D::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64;alpha::Float64=0.0,C::Int64=0,multi_variate::Bool=false,quadratic_constraints::Bool=false,mu::Float64=10^(-4),variable_epsilon::Bool=false,post_process::Bool=true,time_limit::Int64 = -1)
    n = length(y)
    p = length(x[1,:])

    nb_br = 2^D - 1
    nb_lf = 2^D
    
    m = Model(Gurobi.Optimizer)

    if time_limit!=-1
        set_time_limit_sec(m,time_limit)
    end
    
    if multi_variate
        a,b,d,w,s = TreeVariablesandConstraints_Flow(m,D,K,p,multi_variate=true)
    else
        a,b,d,w = TreeVariablesandConstraints_Flow(m,D,K,p,multi_variate=false)
    end

    @variable(m,z[i in 1:n, t in 1:(nb_br+nb_lf)], Bin, base_name="z")
    @variable(m,z_t[i in 1:n, t in 1:(nb_br+nb_lf)], Bin, base_name="z_t")

    @constraint(m, [i in 1:n, t in 1:nb_br], z[i,t*2] + z[i,t*2 + 1] <= d[t])
    @constraint(m, [i in 1:n, t in 1:nb_br], z[i,t] == z[i,t*2] + z[i,t*2 + 1] + z_t[i,t])
    @constraint(m, [i in 1:n, t in (nb_br+1):(nb_br+nb_lf)], z[i,t] == z_t[i,t])
    @constraint(m, [i in 1:n, t in 1:(nb_br+nb_lf)], z_t[i,t] <= w[t,y[i]])

    @variable(m, eps[1:nb_br] >= 0, base_name="eps")
    if variable_epsilon
        @constraint(m, [t in 1:nb_br], eps[t] <= 2*d[t])
    else
        @constraint(m, [t in 1:nb_br], eps[t] == 0)
    end

    if quadratic_constraints
        @constraint(m,[i in 1:n, t in 1:nb_br], z[i,t*2]*(b[t] - mu - eps[t] - sum(x[i,j] * a[j,t] for j in 1:p)) >= 0)
        @constraint(m,[i in 1:n, t in 1:nb_br], z[i,t*2+1]*(- b[t] + sum(x[i,j] * a[j,t] for j in 1:p)) >= 0)
    else
        @constraint(m,[i in 1:n, t in 1:nb_br], z[i,t*2] <= 1 - sum(x[i,j] * a[j,t] for j in 1:p) + b[t] - mu - eps[t])
        @constraint(m,[i in 1:n, t in 1:nb_br], z[i,t*2+1] <= 1 + sum(x[i,j] * a[j,t] for j in 1:p) - b[t])
    end

    if alpha!=0
        if multi_variate
            if variable_epsilon
                @objective(m,Min, n - sum(z[i,1] for i in 1:n) + alpha*sum(s[j,t] for j in 1:p for t in 1:nb_br) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
            else
                @objective(m,Min, n - sum(z[i,1] for i in 1:n) + alpha*sum(s[j,t] for j in 1:p for t in 1:nb_br))
            end
        else
            if variable_epsilon
                @objective(m,Min, n - sum(z[i,1] for i in 1:n) + alpha*sum(d[t] for t in 1:nb_br) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
            else
                @objective(m,Min, n - sum(z[i,1] for i in 1:n) + alpha*sum(d[t] for t in 1:nb_br))
            end
        end
    else
        if C!=0
            if multi_variate
                @constraint(m, sum(s[j,t] for j in 1:p for t in 1:nb_br) <= C)
            else
                @constraint(m, sum(d[t] for t in 1:nb_br) <= C)
            end
        end
        if variable_epsilon
            @objective(m,Min, n - sum(z[i,1] for i in 1:n) - 1/nb_br * sum(eps[n] for n in 1:nb_br))
        else
            @objective(m,Min, n - sum(z[i,1] for i in 1:n))
        end
    end
    
    #JuMP.write_to_file(m, "output_lp2.lp")

    optimize!(m)

    # leaf classes have to have the same structure as Bertsimas tree 
    class = flow_get_class(D,value.(w))

    nodes = MOI.get(m, MOI.NodeCount())
    
    gap=0

    if termination_status(m) != MOI.OPTIMAL
        gap=abs(JuMP.objective_bound(m) - JuMP.objective_value(m)) / JuMP.objective_value(m)
    end

    if variable_epsilon
        T = Tree(D,value.(a),value.(b)-value.(eps)/2,class,eps = value.(eps))
    else
        if post_process
            T = get_tree_flow(x,D,value.(a),value.(z),class)
        else
            T = Tree(D,value.(a),value.(b),class)
        end
    end

    z_ = value.(z)
    errors = n - sum(z_[i,1] for i in 1:n)

    return(T, objective_value(m), errors, solve_time(m), nodes, gap)
end