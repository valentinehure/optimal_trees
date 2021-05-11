include("oct_mip.jl")
include("flow.jl")
include("data_formatting.jl")
using CSV
using DataFrames

function iris_test(nb_part::Int64)
    include("../data/iris.txt")
    caract = ["lineaire","puiss2","puiss3","quadratique"]
    
    indexes = partitioning(X,Y,K,nb_part)
    df = Array{String,2}(undef,nb_part*16,6)

    line = 1

    for part in 1:nb_part
        X_ = Array{Float64}[]
        Y_ = Y[indexes[part]]
        push!(X_,X[indexes[part],:])
        push!(X_,add_powers(X_[1],2))
        push!(X_,add_powers(X_[1],3))
        push!(X_,add_cross_product(X_[1]))
        for c in 1:4
            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas"
            T,obj,time,nodes = oct(2,0,X_[c],Y_,K,0.0,20,true,10^(-4),"C")
            df[line,4] = string(obj)
            df[line,5] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,6] = string(nodes)
            line +=1

            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas_quad"
            T,obj,time,nodes = oct_quad(2,0,X_[c],Y_,K,0.0,20,true,10^(-4),"G")
            df[line,4] = string(obj)
            df[line,5] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,6] = string(nodes)
            line +=1

            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "flot epsilon constant"
            T,obj,time,nodes = flow_adapted(2,X_[c],Y_,K,0.0,true,10^(-4))
            df[line,4] = string(150 ÷ nb_part - obj)
            df[line,5] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,6] = string(nodes)
            line +=1

            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "flot epsilon variable"
            T,obj,time,nodes = flow_adapted(2,X_[c],Y_,K,0.0,true,10^(-4),true)
            df[line,4] = string(150 ÷ nb_part - obj)
            df[line,5] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,6] = string(nodes)
            line +=1
        end
    end

    df = DataFrame(df,["Partition","Caracteristique","Formulation","Temps","Erreurs","Nodes"])  
    CSV.write(string("../results/results_iris_test_",nb_part,".csv"), df)
end


function cplex_vs_gurobi()
    include("../data/iris.txt")
    caract = ["lineaire","puiss2","puiss3","quadratique"]
    
    indexes = partitioning(X,Y,K,5)
    df = Array{String,2}(undef,60,7)

    line = 1

    for part in 1:5
        X_ = Array{Float64}[]
        Y_ = Y[indexes[part]]
        push!(X_,X[indexes[part],:])
        push!(X_,add_powers(X_[1],2))
        push!(X_,add_powers(X_[1],3))
        push!(X_,add_cross_product(X_[1]))
        for c in 1:4
            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas"
            df[line,4] = "cplex"
            T,obj,time,nodes = oct(2,0,X_[c],Y_,K,0.0,20,true,10^(-4),"C")
            df[line,5] = string(obj)
            df[line,6] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,7] = string(nodes)
            line +=1

            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas"
            df[line,4] = "gurobi"
            T,obj,time,nodes = oct(2,0,X_[c],Y_,K,0.0,20,true,10^(-4),"G")
            df[line,5] = string(obj)
            df[line,6] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,7] = string(nodes)
            line +=1

            df[line,1] = string(part)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas_quad"
            df[line,4] = "gurobi"
            T,obj,time,nodes = oct_quad(2,0,X_[c],Y_,K,0.0,20,true,10^(-4),"G")
            df[line,5] = string(obj)
            df[line,6] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,7] = string(nodes)
            line +=1
        end
    end

    df = DataFrame(df,["Partition","Caracteristique","Formulation","Solveur","Erreurs","Temps","Nodes"])  
    CSV.write("../results/results_cplex_vs_gurobi.csv", df)
end

function test(namefile::String,output_name::String)
    include(namefile)
    caract = ["lineaire","puiss2","puiss3","quadratique"]
    
    df = Array{String,2}(undef,56,7)

    line = 1

    X_ = Array{Float64}[]

    push!(X_,X)
    push!(X_,add_powers(X_[1],2))
    push!(X_,add_powers(X_[1],3))
    push!(X_,add_cross_product(X_[1]))

    for D in 2:3
        for c in 1:4
            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas"
            df[line,4] = "FALSE"
            df[line,5] = "FALSE"
            T,obj,time,nodes = oct(D,0,X_[c],Y,K,0.0,20,true,10^(-4),"G")
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1

            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas"
            df[line,4] = "TRUE"
            df[line,5] = "FALSE"
            T,obj,time,nodes = oct_quad(D,0,X_[c],Y,K,0.0,20,true,10^(-4))
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1

            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "bertsimas"
            df[line,4] = "TRUE"
            df[line,5] = "TRUE"
            T,obj,time,nodes = oct_quad(D,0,X_[c],Y,K,0.0,20,true,10^(-4),true)
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1

            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "flot"
            df[line,4] = "FALSE"
            df[line,5] = "FALSE"
            T,obj,time,nodes = flow_adapted(D,X,Y,K,0.0,true,10^(-4))
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1

            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "flot"
            df[line,4] = "TRUE"
            df[line,5] = "FALSE"
            T,obj,time,nodes = flow_adapted_quad(D,X,Y,K,0.0,true,10^(-4))
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1

            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "flot"
            df[line,4] = "FALSE"
            df[line,5] = "TRUE"
            T,obj,time,nodes = flow_adapted(D,X,Y,K,0.0,true,10^(-4),true)
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1

            df[line,1] = string(D)
            df[line,2] = caract[c]
            df[line,3] = "flot"
            df[line,4] = "TRUE"
            df[line,5] = "TRUE"
            T,obj,time,nodes = flow_adapted_quad(D,X,Y,K,0.0,true,10^(-4),true)
            df[line,6] = string(obj)
            df[line,7] = string(Int(floor(time)),",",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(nodes)
            line +=1
        end
    end
    
    df = DataFrame(df,["Caracteristique","Formulation","Contraintes quadratique","Espilon variable","Erreurs","Temps","Nodes"])  
    CSV.write(string("../results/results_",output_name,".csv"), df)
end