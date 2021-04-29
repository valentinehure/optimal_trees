using Plots
using Polynomials

"""
Draws the data points
Arguments :\n 
    - x and y : data
    - K : number of classes
"""
function draw_data_points(x::Array{Float64,2},y::Array{Int,1},K::Int64,p::Int64=1,cross_product::Bool=false)
    col = ["blue","green","orange","yellow","red","pink"]
    P = length(x[1,:])
    n = length(x[:,1])
    if cross_product
        p = sqrt((p-1/2)^2+2*P) - (p-1/2)
        p = convert(Int64, round(p, digits=0))
    else
        p = convert(Int64, round(P/p, digits=0))
    end

    Count = zeros(Int64,K)
    for i in 1:n
        Count[y[i]] += 1
    end

    for k in 1:K
        new_X = zeros(Float64,Count[k],2)
        fill = 1
        for i in 1:n
            if y[i] == k
                new_X[fill,1] = x[i,1]
                new_X[fill,2] = x[i,2]
                fill += 1
            end
        end
        display(plot!(new_X[:,1],new_X[:,2],seriestype = :scatter,label="",markercolor=col[k]))
    end
end 

"""
Draws the region
Arguments :\n 
    - x and y : data
    - K : number of classes
"""
function draw_class_regions(T::Tree,K::Int64,pow::Int64=1,cross_product::Bool=false)
    col = ["blue","green","orange","yellow","red","pink"]
    pix = 400
    x = zeros(Float64,pix^2,2)
    for i in 1:pix
        for j in 1:pix
            x[(i-1)*pix+j,1] = i/pix
            x[(i-1)*pix+j,2] = j/pix
        end
    end

    if pow>1
        if cross_product
            x = add_cross_product_and_powers(x,pow)
        else
            x = add_powers(x,pow)
        end
    elseif cross_product
        x = add_cross_product(x)
    end

    y = predict_class(T,x)

    P = length(x[1,:])
    n = length(x[:,1])

    Count = zeros(Int64,K)
    for i in 1:n
        Count[y[i]] += 1
    end

    for k in 1:K
        new_X = zeros(Float64,Count[k],2)
        fill = 1
        for i in 1:n
            if y[i] == k
                new_X[fill,1] = x[i,1]
                new_X[fill,2] = x[i,2]
                fill += 1
            end
        end

        if k==1
            display(plot(new_X[:,1],new_X[:,2],seriestype = :scatter,label="classe 1",markersize=1,markercolor=col[k],markerstrokecolor=col[k],markershape=:square))
        else
            display(plot!(new_X[:,1],new_X[:,2],seriestype = :scatter,label=string("classe ",k),markersize=1,markercolor=col[k],markerstrokecolor=col[k],markershape=:square))
        end
    end
end

function draw_leaves_regions(T::Tree,pow::Int64=1,cross_product::Bool=false)
    col = ["blue","green","orange","yellow","red","pink","purple","cyan"]
    pix = 500
    x = zeros(Float64,pix^2,2)
    for i in 1:pix
        for j in 1:pix
            x[(i-1)*pix+j,1] = i/pix
            x[(i-1)*pix+j,2] = j/pix
        end
    end

    if pow>1
        if cross_product
            x = add_cross_product_and_powers(x,pow)
        else
            x = add_powers(x,pow)
        end
    elseif cross_product
        x = add_cross_product(x)
    end

    y = predict_leaf(T,x)

    P = length(x[1,:])
    n = length(x[:,1])

    Count = zeros(Int64,2^T.D)
    for i in 1:n
        Count[y[i]] += 1
    end

    for l in 1:(2^T.D)
        new_X = zeros(Float64,Count[l],2)
        fill = 1
        for i in 1:n
            if y[i] == l
                new_X[fill,1] = x[i,1]
                new_X[fill,2] = x[i,2]
                fill += 1
            end
        end

        if l==1
            display(plot(new_X[:,1],new_X[:,2],seriestype = :scatter,label="feuille 1",markersize=1,markercolor=col[l],markerstrokecolor=col[l],markershape=:square))
        else
            display(plot!(new_X[:,1],new_X[:,2],seriestype = :scatter,label=string("feuille ",l),markersize=1,markercolor=col[l],markerstrokecolor=col[l],markershape=:square))
        end
    end
end

function draw_class(x::Array{Float64,2},y::Array{Int,1},T::Tree,K::Int64,p::Int64=1,cross_product::Bool=false)
    draw_class_regions(T,K,p,cross_product)
    draw_data_points(x,y,K,p,cross_product)
end

function draw_leaves(x::Array{Float64,2},y::Array{Int,1},T::Tree,K::Int64,p::Int64=1,cross_product::Bool=false)
    draw_leaves_regions(T,p,cross_product)
    draw_data_points(x,y,K,p,cross_product)
end