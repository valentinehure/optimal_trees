using JuMP
using CPLEX
#using Gurobi
#const GRB_ENV = Gurobi.Env()
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

"""
Function solving the MIO problem.\n
Arguments :\n
    - D : maximum depth of the resulting tree
    - N_min : minimum number of observation attributed to a leaf
    - x and y : data
    - K : the number of labels
Optionnal :\n
    - C/alpha : constants linked to the complexity of the tree
    - multi_V : are the branching rules allowed multiple variables?
    - warm_start : a admissible tree can be used as a warm start to shorten the computionnal time
    - time_limit : time limit in seconds
Result :\n
    - The tree
    - The missclassification on the training set
    - The gap between the optimal solution and the solution found if the time limit was reached.
"""
function oct(D::Int64,Nmin::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64,alpha::Float64=0.0,C::Int64=0,multi_V::Bool=false,warm_start::Tree=null_Tree())
    n = length(y)
    p = length(x[1,:])

    print("Pre-processing\n")

    # computation of L hat
    nb_class=zeros(Int64,K)
    for i in 1:n
        nb_class[y[i]] += 1
    end

    L_h = n - maximum(nb_class)

    # computation of the matrix Y (here, Y is in {0,1})
    Y = zeros(Int64,n,K)
    for i in 1:n
        Y[i,y[i]] = 1
    end

    # computation of the vector epsilon and constant epsilon_max
    if multi_V
        mu = 0.005
    else
        eps = ones(Float64,p)
        eps_max = 0
        eps_min = Inf
        for j in 1:p
            for i in 1:n
                for i_ in i+1:n
                    dif = abs(x[i,j]-x[i_,j])
                    if dif > 0
                        eps[j] = min(eps[j],dif)
                    end
                end
            end
            eps_max = max(eps_max,eps[j])
            eps_min = min(eps_min,eps[j])
        end
    end

    md = Model(CPLEX.Optimizer)

    nb_lf = 2^D
    nb_br = 2^D - 1

    # common variables
    @variable(md, L[1:nb_lf], Int, base_name="L")
    @variable(md, N[1:nb_lf], Int, base_name="N")
    @variable(md, Nk[1:K, 1:nb_lf], Int, base_name="Nk")
    @variable(md, c[1:K, 1:nb_lf], Bin, base_name="c")
    @variable(md, z[1:n, 1:nb_lf], Bin, base_name="z")
    @variable(md, l[1:nb_lf], Bin, base_name="l")
    @variable(md, d[1:nb_br], Bin, base_name="d")

    # specific variables
    if multi_V
        @variable(md, a[1:p, 1:nb_br], base_name="a")
        @variable(md, a_h[1:p, 1:nb_br], base_name="a_h")
        @variable(md, b[1:nb_br], base_name="b")
        @variable(md, s[1:p, 1:nb_br], Bin, base_name="s")
    else
        @variable(md, a[1:p, 1:nb_br], Bin, base_name="a")
        @variable(md, b[1:nb_br], base_name="b")
    end

    # common constraints
    @constraint(md, [k in 1:K, t in 1:nb_lf], L[t] >= N[t] - Nk[k,t] - n*(1 - c[k,t]))
    @constraint(md, [k in 1:K, t in 1:nb_lf], L[t] <= N[t] - Nk[k,t] + n*c[k,t])
    @constraint(md, [t in 1:nb_lf], L[t] >= 0)
    @constraint(md, [k in 1:K, t in 1:nb_lf], Nk[k,t] == sum(Y[i,k]*z[i,t] for i in 1:n)) # here, Y is in {0,1}
    @constraint(md, [t in 1:nb_lf], N[t] == sum(z[i,t] for i in 1:n))

    @constraint(md, [t in 1:nb_lf], l[t] == sum(c[k,t] for k in 1:K))
    @constraint(md, [i in 1:n, t in 1:nb_lf], z[i,t] <= l[t])
    @constraint(md, [t in 1:nb_lf], sum(z[i,t] for i in 1:n) >= Nmin*l[t])

    @constraint(md, [i in 1:n], sum(z[i,t] for t in 1:nb_lf) == 1)

    @constraint(md, [t in 2:nb_br], d[t] <= d[t ÷ 2]) # if t is a branch-node, t ÷ 2 is the parent of t

    for t in 1:nb_lf
        A_L,A_R = compute_LR(t,D)
        if multi_V
            @constraint(md, [i in 1:n, m in A_L], sum(a[j,m]*x[i,j] for j in 1:p) + mu <= b[m] + (2+mu)*(1-z[i,t]))
            @constraint(md, [i in 1:n, m in A_R], sum(a[j,m]*x[i,j] for j in 1:p) >= b[m] - 2*(1-z[i,t]))
        else
            @constraint(md, [i in 1:n, m in A_R], sum(a[j,m]*x[i,j] for j in 1:p) >= b[m] - (1-z[i,t]))
            @constraint(md,[i in 1:n,m in A_L], sum(a[j,m]*(X[i,j]+eps[j]/2) for j in 1:p)  <= b[m]+(1+eps_max)*(1-z[i,t]))
            #@constraint(md, [i in 1:n, m in A_L], sum(a[j,m]*(x[i,j] + eps[j]) for j in 1:p) <= b[m] + (1+eps_max)*(1-z[i,t])) # contrainte du papier
        end
        @constraint(md, [i in 1:n, m in A_L], z[i,t]<=d[m])
        
    end

    if multi_V
        @constraint(md, [t in 1:nb_br], sum(a_h[j,t] for j in 1:p) <= d[t])
        @constraint(md, [t in 1:nb_br, j in 1:p], a[j,t] <= a_h[j,t])
        @constraint(md, [t in 1:nb_br, j in 1:p], -a[j,t] <= a_h[j,t])

        @constraint(md, [t in 1:nb_br, j in 1:p], -s[j,t] <= a[j,t])
        @constraint(md, [t in 1:nb_br, j in 1:p], a[j,t] <= s[j,t])

        @constraint(md, [t in 1:nb_br, j in 1:p], s[j,t] <= d[t])
        @constraint(md, [t in 1:nb_br], sum(s[j,t] for j in 1:p) >= d[t])

        @constraint(md, [t in 1:nb_br, j in 1:p], -d[t] <= b[t])
        @constraint(md, [t in 1:nb_br, j in 1:p], b[t] <= d[t])
    else
        @constraint(md,[t in 1:nb_br], sum(a[j,t] for j in 1:p) == d[t])
        @constraint(md,[t in 1:nb_br], 0 <= b[t])
        @constraint(md,[t in 1:nb_br], b[t] <= d[t])
    end

    if C != 0
        if multi_V
            @constraint(md, sum(s[t] for t in 1:nb_br) <= C)
        else
            @constraint(md, sum(d[t] for t in 1:nb_br) <= C)
        end
        @objective(md,Min,sum(L[t] for t in 1:nb_lf))
    elseif alpha != 0
        if multi_V
            @objective(md,Min,sum(L[t] for t in 1:nb_lf) + L_h*alpha*sum(s[j,t] for t in 1:nb_br for j in 1:p))
        else
            @objective(md,Min,sum(L[t] for t in 1:nb_lf) + L_h*alpha*sum(d[t] for t in 1:nb_br))
        end
    else
        println("Error : neither C nor alpha defined")
        return 0
    end

    JuMP.write_to_file(md, "output_lp.lp")

    if warm_start.D != 0
        println("Entering the warm start values")

        D_ws = warm_start.D

        if D_ws < D
            warm_start = bigger_Tree(warm_start,D)            
        end

        for t in 1:nb_br
            sum = 0
            set_start_value(b[t],warm_start.b[t])
            sum += abs(warm_start.b[t])
            for j in 1:p
                set_start_value(a[j,t],warm_start.a[j,t])
                if multi_V
                    set_start_value(a_h[j,t],abs(warm_start.a[j,t]))
                    if abs(warm_start.a[j,t]) > 0
                        set_start_value(s[j,t], 1)
                        sum += 1
                    else
                        set_start_value(s[j,t], 0)
                    end
                end
            end
            if sum>0
                set_start_value(d[t],1)
            else
                set_start_value(d[t],0)
            end
        end

        for t in 1:nb_lf # that could be problematic but if the tree is designed with the same x, there should be no problem
            for k in 1:K
                if warm_start.c[t] == k
                    set_start_value(c[k,t],1)
                else
                    set_start_value(c[k,t],0)
                end
            end
        end

        leaf = predict_leaf(warm_start,x,mu)
        current_N = zeros(Int64,nb_lf)
        current_Nk = zeros(Int64,K,nb_lf)
        current_l = zeros(Int64,nb_br)

        for i in 1:n
            current_N[leaf[i]] += 1
            current_Nk[y[i],leaf[i]] += 1

            t = leaf[i] + 2^T.D - 1

            set_start_value(z[i,1],1)
            for d in 1:(D-1)
                t = t ÷ 2
                for t_ in (2^(D-d)):(2^(D-d+1)-1)
                    set_start_value(z[i,t_],t == t_)
                end
                current_l[t] += 1
            end
        end

        for t in 1:nb_br
            set_start_value(l[t], current_l[t])
        end

        for t in 1:nb_lf
            set_start_value(N[t], current_N[t])
            for k in 1:K
                set_start_value(Nk[k,t], current_Nk[k,t])
            end
        end

        for t in 1:nb_lf # as mentionned before, that should not be a problem if warm start was designed with x and y
            set_start_value(L[t],current_N[t] - current_Nk[warm_start.c[t],t])
        end

    end

    println("Solving")

    optimize!(md)

    class =  mapslices(argmax,value.(c),dims=1)[:] # it should give us the indexes such that c[:,t] is equal to 1

    return(Tree(D,value.(a),value.(b),class),objective_value(md))
end

function oct_quad(D::Int64,Nmin::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64,alpha::Float64=0.0,C::Int64=0,multi_V::Bool=false,mu::Float64=10^(-7),warm_start::Tree=null_Tree())
    n = length(y)
    p = length(x[1,:])

    print("Pre-processing\n")

    # computation of L hat
    nb_class=zeros(Int64,K)
    for i in 1:n
        nb_class[y[i]] += 1
    end

    L_h = n - maximum(nb_class)

    # computation of the matrix Y (here, Y is in {0,1})
    Y = zeros(Int64,n,K)
    for i in 1:n
        Y[i,y[i]] = 1
    end

    md = Model(() -> Gurobi.Optimizer(GRB_ENV))

    nb_lf = 2^D
    nb_br = 2^D - 1

    # common variables
    @variable(md, L[1:nb_lf], Int, base_name="L")
    @variable(md, N[1:nb_lf], Int, base_name="N")
    @variable(md, Nk[1:K, 1:nb_lf], Int, base_name="Nk")
    @variable(md, c[1:K, 1:nb_lf], Bin, base_name="c")
    @variable(md, z[1:n, 1:(nb_br+nb_lf)], Bin, base_name="z")
    @variable(md, l[1:nb_lf], Bin, base_name="l")
    @variable(md, d[1:nb_br], Bin, base_name="d")

    # specific variables
    if multi_V
        @variable(md, a[1:p, 1:nb_br], base_name="a")
        @variable(md, a_h[1:p, 1:nb_br], base_name="a_h")
        @variable(md, b[1:nb_br], base_name="b")
        @variable(md, s[1:p, 1:nb_br], Bin, base_name="s")
    else
        @variable(md, a[1:p, 1:nb_br], Bin, base_name="a")
        @variable(md, b[1:nb_br], base_name="b")
    end

    # common constraints
    @constraint(md, [k in 1:K, t in 1:nb_lf], L[t] >= N[t] - Nk[k,t] - n*(1 - c[k,t]))
    @constraint(md, [k in 1:K, t in 1:nb_lf], L[t] <= N[t] - Nk[k,t] + n*c[k,t])
    @constraint(md, [t in 1:nb_lf], L[t] >= 0)
    @constraint(md, [k in 1:K, t in (nb_br+1):(nb_br+nb_lf)], Nk[k,t] == sum(Y[i,k]*z[i,t] for i in 1:n)) # here, Y is in {0,1}
    @constraint(md, [t in (nb_br+1):(nb_br+nb_lf)], N[t] == sum(z[i,t] for i in 1:n))

    @constraint(md, [t in 1:nb_lf], l[t] == sum(c[k,t] for k in 1:K))
    @constraint(md, [i in 1:n, t in 1:nb_lf], z[i,nb_br + t] <= l[t])
    @constraint(md, [t in 1:nb_lf], sum(z[i,nb_br + t] for i in 1:n) >= Nmin*l[t])

    @constraint(md, [i in 1:n, t in 1:nb_br], z[i,t] == z[i,t*2] + z[i,t*2+1])
    @constraint(md, [i in 1:n], z[i,1] == 1)

    #@constraint(md, [i in 1:n, t in 1:nb_br], z[i,t] <= d[t])

    @constraint(md, [t in 2:nb_br], d[t] <= d[t ÷ 2]) # if t is a branch-node, t ÷ 2 is the parent of t

    if multi_V
        @constraint(md, [t in 1:nb_br, i in 1:n], z[i,t*2] * (sum(a[j,t]*x[i,j] for j in 1:p) - b[t] + mu) <= 0)
        @constraint(md, [t in 1:nb_br, i in 1:n], z[i,t*2+1] * (sum(a[j,t]*x[i,j] for j in 1:p) - b[t]) >= 0)

        @constraint(md, [t in 1:nb_br], sum(a_h[j,t] for j in 1:p) <= d[t])
        @constraint(md, [t in 1:nb_br, j in 1:p], a[j,t] <= a_h[j,t])
        @constraint(md, [t in 1:nb_br, j in 1:p], -a[j,t] <= a_h[j,t])

        @constraint(md, [t in 1:nb_br, j in 1:p], -s[j,t] <= a[j,t])
        @constraint(md, [t in 1:nb_br, j in 1:p], a[j,t] <= s[j,t])

        @constraint(md, [t in 1:nb_br, j in 1:p], s[j,t] <= d[t])
        @constraint(md, [t in 1:nb_br], sum(s[j,t] for j in 1:p) >= d[t])

        @constraint(md, [t in 1:nb_br, j in 1:p], -d[t] <= b[t])
        @constraint(md, [t in 1:nb_br, j in 1:p], b[t] <= d[t])
    
    else
        @constraint(md, [t in 1:nb_br, i in 1:n], z[i,t*2] * (sum(a[j,t]*x[i,j] for j in 1:p) - b[t] + mu) <= 0)
        @constraint(md, [t in 1:nb_br, i in 1:n], z[i,t*2+1] * (sum(a[j,t]*x[i,j] for j in 1:p) - b[t]) >= 0)

        @constraint(md,[t in 1:nb_br], sum(a[j,t] for j in 1:p) == d[t])
        @constraint(md,[t in 1:nb_br], 0 <= b[t])
        @constraint(md,[t in 1:nb_br], b[t] <= d[t])
    end

    if C != 0
        if multi_V
            @constraint(md, sum(s[t] for t in 1:nb_br) <= C)
        else
            @constraint(md, sum(d[t] for t in 1:nb_br) <= C)
        end
        @objective(md,Min,sum(L[t] for t in 1:nb_lf) / L_h)
    elseif alpha != 0
        if multi_V
            @objective(md,Min,sum(L[t] for t in 1:nb_lf) / L_h + alpha*sum(s[j,t] for t in 1:nb_br for j in 1:p))
        else
            @objective(md,Min,sum(L[t] for t in 1:nb_lf) / L_h + alpha*sum(d[t] for t in 1:nb_br))
        end
    else
        println("Error : neither C nor alpha defined")
        return 0
    end

    # JuMP.write_to_file(md, "output_lp.lp")

    if warm_start.D != 0
        println("Entering the warm start values")

        D_ws = warm_start.D

        if D_ws < D
            warm_start = bigger_Tree(warm_start,D)            
        end

        for t in 1:nb_br
            s = 0
            set_start_value(b[t],warm_start.b[t])
            s += abs(warm_start.b[t])
            for j in 1:p
                set_start_value(a[j,t],warm_start.a[j,t])
                if multi_V
                    set_start_value(a_h[j,t],abs(warm_start.a[j,t]))
                    if abs(warm_start.a[j,t]) > 0
                        set_start_value(s[j,t], 1)
                        s += 1
                    else
                        set_start_value(s[j,t], 0)
                    end
                end
            end
            if s>0
                set_start_value(d[t],1)
            else
                set_start_value(d[t],0)
            end
        end

        for t in 1:nb_lf # that could be problematic but if the tree is designed with the same x, there should be no problem
            for k in 1:K
                if warm_start.c[t] == k
                    set_start_value(c[k,t],1)
                else
                    set_start_value(c[k,t],0)
                end
            end
        end

        leaf = predict_leaf(warm_start,x)
        current_N = zeros(Int64,nb_lf)
        current_Nk = zeros(Int64,K,nb_lf)
        current_l = zeros(Int64,nb_br)

        for i in 1:n
            current_N[leaf[i]] += 1
            current_Nk[y[i],leaf[i]] += 1

            t = leaf[i] + 2^T.D - 1

            set_start_value(z[i,1],1)
            for d in 1:(D-1)
                t = t ÷ 2
                for t_ in (2^(D-d)):(2^(D-d+1)-1)
                    set_start_value(z[i,t_],t == t_)
                end
                current_l[t] += 1
            end
        end

        for t in 1:nb_br
            set_start_value(l[t], current_l[t])
        end

        for t in 1:nb_lf
            set_start_value(N[t], current_N[t])
            for k in 1:K
                set_start_value(Nk[k,t], current_N[k,t])
            end
        end

        for t in 1:nb_lf # as mentionned before, that should not be a problem if warm start was designed with x and y
            set_start_value(L[t],N[t] - Nk[warm_start.c[t],t])
        end

    end

    println("Solving")

    optimize!(md)

    gap=0

    if termination_status(md) != MOI.OPTIMAL
        gap=abs(JuMP.objective_bound(md) - JuMP.objective_value(md)) / JuMP.objective_value(md)
    end

    class =  mapslices(argmax,value.(c),dims=1)[:] # it should give us the indexes such that c[:,t] is equal to 1

    return(Tree(D,value.(a),value.(b),class),objective_value(md),gap)
end