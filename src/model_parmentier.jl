include("oct_mip.jl")

function get_tree_discrete(x::Array{Float64,2},D::Int64,a::Array{Float64,2},z::Array{Float64,2},c::Array{Int64,1})
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
            f = argmax(a_tree[:,m])
            if x[i,f] > b_tree_l[m]
                b_tree_l[m] = x[i,f]
            end
        end
        for m in A_R
            f = argmax(a_tree[:,m])
            if x[i,f] < b_tree_r[m]
                b_tree_r[m] = x[i,f]
            end
        end
    end

    for t in 1:nb_br
        b_tree[t] = (b_tree_l[t]+b_tree_r[t]) / 2
        eps_tree[t] = (b_tree_r[t] - b_tree_l[t])/2
    end

    return Tree(D,a_tree,b_tree,c,eps=eps_tree)
end

function compute_Xp(x::Array{Float64,2},y::Array{Int64,1})
    n = length(y)
    f = length(x[1,:])

    X_2 = zeros(Int64,n,f)
    nb_val = zeros(Int64,f)

    J = []
    new_J = []
    for attrib in 1:f
        sorted_vals = sortperm(x[:,attrib])
        push!(J,[])
        push!(J[attrib],[sorted_vals[1]])
        package_num = 1
        for i in 2:n
            if x[sorted_vals[i],attrib] == x[sorted_vals[i-1],attrib]
                push!(J[attrib][package_num],sorted_vals[i])
            else
                push!(J[attrib],[sorted_vals[i]])
                package_num += 1
            end
        end

        push!(new_J,[])
        push!(new_J[attrib],[])
        c = y[J[attrib][1][1]]
        same_class = true
        for i in J[attrib][1]
            push!(new_J[attrib][1],i)
            same_class = same_class && y[i] == c
        end

        new_package_num = 1
        for p in 2:length(J[attrib])
            same_new_class = true
            new_c = y[J[attrib][p][1]]
            for i in J[attrib][p]
                same_new_class = same_class && y[i] == new_c
            end

            if (!same_class) || (!same_new_class) || (new_c != c)
                new_package_num += 1
                push!(new_J[attrib],[])
            end  
            for i in J[attrib][p]
                push!(new_J[attrib][new_package_num],i)
            end
            same_class = same_new_class
            c = new_c
        end

        for p in 1:length(new_J[attrib])
            for i in new_J[attrib][p]
                X_2[i,attrib] = p
            end
        end
        nb_val[attrib] = length(new_J[attrib])
    end
    return X_2,nb_val
end

function TreeVariablesandConstraints_Pmodel(m::Model,D::Int64,p::Int64,nb_val::Array{Int64,1})
    nb_br = 2^D - 1
    max_nb_val = maximum(nb_val)

    
    a = @variable(m, [1:p, 1:nb_br], Bin, base_name="a")
    b = @variable(m, [1:p, 1:nb_br, 1:max_nb_val], Bin, base_name="b")
    d = @variable(m, [1:nb_br], Bin, base_name="d")

    @constraint(m, [t in 2:nb_br], d[t] <= d[t ÷ 2])
    @constraint(m,[t in 1:nb_br], sum(a[j,t] for j in 1:p) == d[t])
    @constraint(m,[t in 1:nb_br,j in 1:p], b[j,t,1] == 0)
    @constraint(m,[t in 1:nb_br,j in 1:p], a[j,t] <= b[j,t,nb_val[j]])
    @constraint(m,[t in 1:nb_br,j in 1:p,q in 2:nb_val[j]], b[j,t,q-1] <= b[j,t,q])
    @constraint(m,[t in 1:nb_br,j in 1:p,q in 2:nb_val[j]], b[j,t,q] <= a[j,t])

    return a, b, d
end

function oct_parmentier(D::Int64,Nmin::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64;quadratic_objective::Bool=false,X_p::Array{Int64,2}=zeros(Int64,1,1),nb_val::Array{Int64,1}=zeros(Int64,1),alpha::Float64=0.0,C::Int64=0,time_limit::Int64=-1)
    n = length(y)
    p = length(x[1,:])

    nb_lf = 2^D
    nb_br = 2^D - 1

    if X_p == zeros(Int64,1,1)
        X_p, nb_val = compute_Xp(x,y)
    end

    # computation of the matrix Y (here, Y is in {0,1})
    Y = zeros(Int64,n,K)
    for i in 1:n
        Y[i,y[i]] = 1
    end

    m = Model(Gurobi.Optimizer)

    if time_limit!=-1
        set_time_limit_sec(m,time_limit)
    end

    a, b, d = TreeVariablesandConstraints_Pmodel(m,D,p,nb_val)
    @variable(m, z[1:n, 1:nb_lf], Bin, base_name="z")

    for t in 1:nb_lf
        A_L,A_R = compute_LR(t,D)
        @constraint(m,[g in A_R,i in 1:n], z[i,t] <= sum(b[j,g,X_p[i,j]] for j in 1:p))
        @constraint(m,[g in A_L,i in 1:n], z[i,t] <= 1 - sum(b[j,g,X_p[i,j]] for j in 1:p))
        # @constraint(m, [i in 1:n, g in A_L], z[i,t] <= d[g])
    end

    if quadratic_objective
        @variable(m, l[1:nb_lf], Bin, base_name="l")
        @variable(m, c[1:K, 1:nb_lf], Bin, base_name="c")
        
        @constraint(m, [t in 1:nb_lf], l[t] == sum(c[k,t] for k in 1:K))
        @constraint(m, [i in 1:n, t in 1:nb_lf], z[i,t] <= l[t])
        @constraint(m, [t in 1:nb_lf], sum(z[i,t] for i in 1:n) >= Nmin*l[t])
        @constraint(m, [i in 1:n], sum(z[i,t] for t in 1:nb_lf) == 1)

        if alpha != 0
            @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) + alpha*sum(d[t] for t in 1:nb_br))
        else
            if C != 0
                @constraint(m, sum(d[t] for t in 1:nb_br) <= C)
            end
            @objective(m,Min,sum(c[k,t]*sum((1-Y[i,k])*z[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K) )
        end
    else
        L,N,Nk,l,c = LeafClassConstraintsAndVariables(m,n,nb_lf,K)

        @constraint(m, [k in 1:K, t in 1:nb_lf], Nk[k,t] == sum(Y[i,k]*z[i,t] for i in 1:n)) # here, Y is in {0,1}
        @constraint(m, [t in 1:nb_lf], N[t] == sum(z[i,t] for i in 1:n))

        @constraint(m, [t in 1:nb_lf], l[t] == sum(c[k,t] for k in 1:K))
        @constraint(m, [i in 1:n, t in 1:nb_lf], z[i,t] <= l[t])
        @constraint(m, [t in 1:nb_lf], sum(z[i,t] for i in 1:n) >= Nmin*l[t])
        @constraint(m, [i in 1:n], sum(z[i,t] for t in 1:nb_lf) == 1)

        if C != 0
            @constraint(m, sum(d[t] for t in 1:nb_br) <= C)
            @objective(m,Min,sum(L[t] for t in 1:nb_lf))
        elseif alpha != 0
            @objective(m,Min,sum(L[t] for t in 1:nb_lf) + alpha*sum(d[t] for t in 1:nb_br))
        else
            @objective(m,Min,sum(L[t] for t in 1:nb_lf))
        end
    end

    #JuMP.write_to_file(m, "output_lp.lp")

    optimize!(m)

    gap=0

    if termination_status(m) != MOI.OPTIMAL
        gap=abs(JuMP.objective_bound(m) - JuMP.objective_value(m)) / JuMP.objective_value(m)
    end

    class =  mapslices(argmax,value.(c),dims=1)[:] # it should give us the indexes such that c[:,t] is equal to 1

    T =  get_tree_discrete(x,D,value.(a),value.(z),class)
    
    nodes = MOI.get(m, MOI.NodeCount())

    c_ = value.(c)
    z_ = value.(z)
    errors = sum(c_[k,t]*sum((1-Y[i,k])*z_[i,t] for i in 1:n) for t in 1:nb_lf for k in 1:K)

    return(T, objective_value(m), errors ,solve_time(m),nodes,gap)
end
