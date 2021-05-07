include("tree.jl")

function flow_adapted(D::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64,lambda::Float64=0.0,multi_V::Bool=false,mu::Float64=10^(-7),warm_start::Tree=null_Tree())
    I = length(y)
    F = length(x[1,:])

    L = 2^D
    N = 2^D - 1
    
    md = Model(with_optimizer(CPLEX.Optimizer, CPXPARAM_Simplex_Tolerances_Feasibility=mu))

    set_silent(md)

    @variable(md,z[i in 1:I, n in 1:(N+L)], Bin, base_name="z")
    @variable(md,z_t[i in 1:I, n in 1:(N+L)], Bin, base_name="z_t")
    @variable(md,w[n in 1:(N+L), k in 1:K], Bin, base_name="w")
    
    if multi_V
        @variable(md,a[f in 1:F,n in 1:N], base_name="a")
        @variable(md,a_h[f in 1:F,n in 1:N], base_name="a_h")
        @variable(md,S[n in 1:N], Bin, base_name="S")
        @variable(md,s[f in 1:F,n in 1:N], Bin, base_name="s")
        @variable(md,b[n in 1:N], base_name="b")
    else
        @variable(md,a[f in 1:F,n in 1:N], Bin, base_name="a")
        @variable(md,b[n in 1:N], base_name="b")
    end

    if multi_V
        @constraint(md, [n in 1:N, f in 1:F], a_h[f,n] >= a[f,n])
        @constraint(md, [n in 1:N, f in 1:F], a_h[f,n] >= -a[f,n])
        @constraint(md, [n in 1:N, f in 1:F], s[f,n] >= a[f,n])
        @constraint(md, [n in 1:N, f in 1:F], a[f,n] >= -s[f,n])
        @constraint(md, [n in N, f in 1:F], S[n] >= s[f,n])
        @constraint(md, [n in N], S[n] <= sum(s[f,n] for f in 1:F))

        @constraint(md, [n in 1:N], S[n] + sum(w[n,k] for k in 1:K) == 1)
        @constraint(md, [n in 1:N], sum(a_h[f,n] for f in 1:F) <= 1)

        @constraint(md, [n in 1:N], b[n] <= S[n])
        @constraint(md, [n in 1:N], - b[n] <= S[n])

        @constraint(md, [n in 1:N, i in 1:I], z[i,n*2] + z[i,n*2 + 1] <= S[n])
    else
        @constraint(md, [n in 1:N], sum(a[f,n] for f in 1:F) + sum(w[n,k] for k in 1:K) == 1)

        @constraint(md, [n in 1:N], b[n] <= sum(a[f,n] for f in 1:F))
        @constraint(md, [n in 1:N], b[n] >= 0)

        @constraint(md, [i in 1:I, n in 1:N], z[i,n*2] + z[i,n*2 + 1] <= sum(a[f,n] for f in 1:F))
    end

    @constraint(md, [n in (N+1):(N+L)], sum(w[n,k] for k in 1:K) == 1)

    @constraint(md, [i in 1:I, n in 1:N], z[i,n] == z[i,n*2] + z[i,n*2 + 1] + z_t[i,n])
    @constraint(md, [i in 1:I, n in (N+1):(N+L)], z[i,n] == z_t[i,n])
    @constraint(md, [i in 1:I], z[i,1] <= 1)

    @constraint(md, [i in 1:I, n in 1:N], z[i,n*2] + mu/10 <= 1 - sum(a[f,n]*x[i,f] for f in 1:F) + b[n])
    @constraint(md, [i in 1:I, n in 1:N], z[i,n*2+1] <= 1 + sum(a[f,n]*x[i,f] for f in 1:F) - b[n])

    for i in 1:I
        @constraint(md, [n in 1:(N+L)], z_t[i,n] <= w[n,y[i]])
    end
    
    if multi_V
        @objective(md, Max, (1 - lambda) * sum(z_t[i,n] for i in 1:I for n in 1:(N+L)) - lambda * sum(s[f,n] for f in 1:F for n in 1:N))
    else
        @objective(md, Max, (1 - lambda) * sum(z_t[i,n] for i in 1:I for n in 1:(N+L)) - lambda * sum(a[f,n] for f in 1:F for n in 1:N))
    end

    JuMP.write_to_file(md, "output_lp_flow.lp")

    #if warm_start.D != 0
    #end

    optimize!(md)

    # leaf classes have to have the same structure as Bertsimas tree 
    class = zeros(Int64,2^D)

    for n in 1:N
        index = n 
        while index <= N
            index = index*2 + 1
        end
        index -= N
        for k in 1:K
            if value.(w)[n,k] == 1
                class[index] = k
                break
            end
        end
    end

    for n in 1:L
        if class[n] == 0
            for k in 1:K
                if value.(w)[n+N,k] == 1
                    class[n] = k
                    break
                end
            end
        end            
    end
    return(Tree(D,value.(a),value.(b),class),objective_value(md))
end

function sub_problem_multivariate(D::Int64,a::Array{Float64,2},b::Array{Float64,1},w::Array{Int64,2},x::Array{Float64,1},y::Int64,S::Array{Int64,1},eps::Float64=0.005)
    F = length(x)
    K = length(w[1,:])

    L = 2^D
    N = 2^D - 1
    
    sub_md = Model(CPLEX.Optimizer)

    c = zeros(Float64,N+L)
    c[1] = 1
    for n in 1:N
        axb = sum(a[f,n]*x[f] for f in 1:F) - b[n]
        c[2*n] = 1 - axb - eps 
        c[2*n+1] = 1 + axb 
    end

    c_t = zeros(Int64,N+L)
    for n in N+L
        c_t[n] = w[n,y]
    end

    c_n = zeros(Int64,N)
    for n in 1:N
        c_n[n] = S[n]
    end


    @variable(sub_md, q[n in 1:(N+L)])
    @variable(sub_md, q_t[n in 1:(N+L)])
    @variable(sub_md, q_n[n in 1:N])

    @variable(sub_md, p1[n in 1:N])
    @variable(sub_md, p2[n in 1:N])
    @variable(sub_md, p3[n in 1:L])
    
    @constraint(sub_md, q[1] + p1[1] >= 1)
    @constraint(sub_md, [n in 2:N], q[n] + p1[n] - p2[n ÷ 2] >= 0)
    @constraint(sub_md, [n in (N+1):(N+L)], q[n] + p3[n-N] - p2[n ÷ 2] >= 0)
    @constraint(sub_md, [n in 1:N], q_t[n] - p1[n] >= 0)
    @constraint(sub_md, [n in (N+1):(N+L)], q_t[n] - p3[n-N] >= 0)
    @constraint(sub_md, [n in 1:N], q_n[n] - p1[n] + p2[n] >= 0)


    @objective(sub_md, Min, sum(q[n]*c[n] for n in 1:(N+L))
                            + sum(q_t[n]*c_t[n] for n in 1:(N+L))
                            + sum(q_n[n]*c_n[n] for n in 1:N)) 

    return sub_md
end

function sub_problem_univariate(D::Int64,a::Array{Int64,2},b::Array{Float64,1},w::Array{Int64,2},x::Array{Float64,1},y::Int64,eps::Float64=0.005)
    F = length(x)
    K = length(w[1,:])

    L = 2^D
    N = 2^D - 1
    
    sub_md = Model(CPLEX.Optimizer)

    c = zeros(Float,N+L)
    c[1] = 1
    for n in 1:N
        axb = sum(a[f,n]*x[f] for f in 1:F) - b[n]
        c[2*n] = 1 - axb - eps 
        c[2*n+1] = 1 + axb 
    end

    c_t = zeros(Int64,N+L)
    for n in N+L
        c_t[n] = w[n,y]
    end

    c_n = zeros(Int64,N)
    for n in 1:N
        c_n[n] = sum(a[f,n] for f in 1:F)
    end


    @variable(sub_md, q[n in 1:(N+L)])
    @variable(sub_md, q_t[n in 1:(N+L)])
    @variable(sub_md, q_n[n in 1:N])

    @variable(sub_md, p1[n in 1:N])
    @variable(sub_md, p2[n in 1:N])
    @variable(sub_md, p3[n in 1:L])
    
    @constraint(sub_md, q[1] + p1[1] >= 1)
    @constraint(sub_md, [n in 2:N], q[n] + p1[n] - p2[n] >= 0)
    @constraint(sub_md, [n in (N+1):(N+L)], q[n] + p3[n-N] - p2[n-N] >= 0)
    @constraint(sub_md, [n in 1:N], q_t[n] - p1[n] >= 0)
    @constraint(sub_md, [n in (N+1):(N+L)], q_t[n] - p3[n-N] >= 0)
    @constraint(sub_md, [n in 1:N], q_n[n] - p1[n] + p2[n] >= 0)


    @objective(sub_md, Min, sum(q[n]*c[n] for n in 1:(N+L))
                            + sum(q_t[n]*c_t[n] for n in 1:(N+L))
                            + sum(q_n[n]*c_n[n] for n in 1:N)) 

    return sub_md
end

function main_problem(D::Int64,x::Array{Float64,2},y::Array{Int64,1},K::Int64,lambda::Float64=0.0,multi_V::Bool=false,eps::Float64=0.005,EPSILON::Float64=10^(-5))
    I = length(y)
    F = length(x[1,:])

    L = 2^D
    N = 2^D - 1

    md = Model(CPLEX.Optimizer)

    @variable(md,g[i in 1:I], base_name="g")
    @variable(md,w[n in 1:(N+L), k in 1:K], Bin, base_name="w")
    
    if multi_V
        @variable(md,a[f in 1:F,n in 1:N], base_name="a")
        @variable(md,a_h[f in 1:F,n in 1:N], base_name="a_h")
        @variable(md,S[n in 1:N], Bin, base_name="S")
        @variable(md,s[f in 1:F,n in 1:N], Bin, base_name="s")
        @variable(md,b[n in 1:N], base_name="b")
    else
        @variable(md,a[f in 1:F,n in 1:N], Bin, base_name="a")
        @variable(md,b[n in 1:N], base_name="b")
    end

    if multi_V
        @constraint(md, [n in 1:N, f in 1:F], a_h[f,n] >= a[f,n])
        @constraint(md, [n in 1:N, f in 1:F], a_h[f,n] >= -a[f,n])
        @constraint(md, [n in 1:N, f in 1:F], s[f,n] >= a[f,n])
        @constraint(md, [n in 1:N, f in 1:F], a[f,n] >= -s[f,n])
        @constraint(md, [n in N, f in 1:F], S[n] >= s[f,n])
        @constraint(md, [n in N], S[n] <= sum(s[f,n] for f in 1:F))

        @constraint(md, [n in 1:N], S[n] + sum(w[n,k] for k in 1:K) == 1)
        @constraint(md, [n in 1:N], sum(a_h[f,n] for f in 1:F) <= 1)

        @constraint(md, [n in 1:N], b[n] <= S[n])
        @constraint(md, [n in 1:N], - b[n] <= S[n])
    else
        @constraint(md, [n in 1:N], sum(a[f,n] for f in 1:F) + sum(w[n,k] for k in 1:K) == 1)

        @constraint(md, [n in 1:N], b[n] <= sum(a[f,n] for f in 1:F))
        @constraint(md, [n in 1:N], b[n] >= 0)

        @constraint(md, [i in 1:I, n in 1:N], z[i,n*2] + z[i,n*2 + 1] <= sum(a[f,n] for f in 1:F))
    end

    @constraint(md, [n in (N+1):(N+L)], sum(w[n,k] for k in 1:K) == 1)
    @constraint(md, [i in 1:I], g[i] >= 0)
    @constraint(md, [i in 1:I], g[i] <= 1)
    
    if multi_V
        @objective(md, Max, (1 - lambda) * sum(g[i] for i in 1:I) - lambda * sum(s[f,n] for f in 1:F for n in 1:N))
    else
        @objective(md, Max, (1 - lambda) * sum(g[i] for i in 1:I) - lambda * sum(a[f,n] for f in 1:F for n in 1:N))
    end

    function callback_cutting_planes(callback_data)
        if multi_V
            a_star = zeros(Float64,F,N)
        else
            a_star = zeros(Int64,F,N)
        end
        for f in 1:F
            for n in 1:N
                a_star[f,n] = callback_value(callback_data,a[f,n])
            end
        end

        b_star = zeros(N)
        for n in 1:N
            b_star[n] = callback_value(callback_data,b[n])
        end

        w_star = zeros(Int64,N+L,K)
        for n in 1:(N+L)
            for k in 1:K
                w_star[n,k] = callback_value(callback_data,w[n,k])
            end
        end

        g_sub = zeros(Int,I)
        q = zeros(Float64, I, N+L)
        q_t = zeros(Float64,I, N+L)
        q_n = zeros(Float64,I, N)
        if multi_V
            S_star = zeros(Int64,N)
            for n in 1:N
                S_star[n] = callback_value(callback_data,S[n])
            end
            for i in 1:I
                sub_pb = sub_problem_multivariate(D,a_star,b_star,w_star,x[i,:],y[i],S_star,eps)
                optimize!(sub_pb)
                for n in 1:N+L
                    q[i,n] = value.(q[n])
                end
                for n in 1:N+L
                    q_t[i,n] = value.(q_t[n])
                end
                for n in 1:N
                    q_n[i,n] = value.(q_n[n])
                end
                g_sub[i] = objective_value(sub_pb)
            end
            obj_sub = (1-lambda)*sum(g_sub[i] for i in 1:I) - lambda*sum(s[f,n] for n in 1:N for f in 1:F)
        else
            for i in 1:I
                sub_pb = sub_problem_univariate(D,a_star,b_star,w_star,x[i,:],y,eps)
                optimize!(sub_pb)
                for n in 1:N+L
                    q[i,n] = value.(q[n])
                end
                for n in 1:N+L
                    q_t[i,n] = value.(q_t[n])
                end
                for n in 1:N
                    q_n[i,n] = value.(q_n[n])
                end
                g_sub[i] = objective_value(sub_pb)
            end
            obj_sub = (1-lambda)*sum(g_sub[i] for i in 1:I) - lambda*sum(a[f,n] for n in 1:N for f in 1:F)
        end
        
        if multi_V
            obj_sub -= lambda*sum(callback_value(callback_data,s[f,n]) for f in 1:F for n in 1:N)
        else
            obj_sub -= lambda*sum(a_star[f,n] for f in 1:F for n in 1:N)
        end

        if abs(obj_sub - obj_star) <= EPSILON
            return
        else
            if multi_V
                constraint = @build_constraint(g[i] <= q[1] + sum(q[2*n]*(1 - sum(a[f,n]*x[f] for f in 1:F) + b[n] - eps) for n in 1:N)
                                                        + sum(q[2*n+1]*(1 + sum(a[f,n]*x[f] for f in 1:F) - b[n]) for n in 1:N)
                                                        + sum(q_t[n]*sum(w[n,k] for k in 1:K) for n in 1:(N+L))
                                                        + sum(q_n[n]*S[n] for n in 1:N))
            else 
                constraint = @build_constraint(g[i] <= q[1] + sum(q[2*n]*(1 - sum(a[f,n]*x[f] for f in 1:F) + b[n] - eps) for n in 1:N)
                                                        + sum(q[2*n+1]*(1 + sum(a[f,n]*x[f] for f in 1:F) - b[n]) for n in 1:N)
                                                        + sum(q_t[n]*sum(w[n,k] for k in 1:K) for n in 1:(N+L))
                                                        + sum(q_n[n]*sum(a[f,n] for f in 1:F) for n in 1:N))
            end
            MOI.submit(md, MOI.LazyConstraint(callback_data), constraint)
        end
    end
    
    MOI.set(md, MOI.LazyConstraintCallback(), callback_cutting_planes)

    optimize!(md)
    return(Tree(D,value.(a),value.(b),class),objective_value(md))
end