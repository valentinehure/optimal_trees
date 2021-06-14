include("oct_mip.jl")
include("flow.jl")
include("data_formatting.jl")
using CSV
using DataFrames

function test(namefile,depth_max::Int64,time_l::Int64=-1)
    include(string("../data/",namefile,".txt"))

    if time_l != -1
        namefile = string(namefile,"_time_",time_l)
    end

    caract = ["L","P2","P3","Q"]
    
    train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)

    df = Array{String,2}(undef,5*length(caract)*(depth_max-1)*4,10)
    
    line = 1

    for part in 1:5
        X_train = Array{Float64}[]
        Y_train = Y[train_indexes[part]]
        push!(X_train,X[train_indexes[part],:])
        push!(X_train,add_powers(X_train[1],2))
        push!(X_train,add_powers(X_train[1],3))
        push!(X_train,add_cross_product(X_train[1]))

        X_test = Array{Float64}[]
        Y_test = Y[test_indexes[part]]
        push!(X_test,X[test_indexes[part],:])
        push!(X_test,add_powers(X_test[1],2))
        push!(X_test,add_powers(X_test[1],3))
        push!(X_test,add_cross_product(X_test[1]))

        for D in 2:depth_max
            for c in 1:4
                df[line,1] = string(part)
                df[line,2] = string(D)
                df[line,3] = caract[c]
                df[line,4] = "bertsimas_quad"
                df[line,5] = "FALSE"
                T,obj,time,nodes,gap = oct(D,0,X_train[c],Y_train,K,C = 20, multi_V = true, quad=true, epsi_var=false, time_limit = time_l)
                if c>3
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_BQ_D",D,"_EPSnonVAR_part",part,".txt"),cross_prod=true)
                elseif c == 1
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_BQ_D",D,"_EPSnonVAR_part",part,".txt"))
                else
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_BQ_D",D,"_EPSnonVAR_part",part,".txt"),power = c)
                end
                df[line,6] = string(obj)
                df[line,7] = string(nodes)

                Y_pred = predict_class(T,X_test[c])
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,8] = string(errors)
                df[line,9] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,10] = string(gap)

                line +=1

                df[line,1] = string(part)
                df[line,2] = string(D)
                df[line,3] = caract[c]
                df[line,4] = "bertsimas_quad"
                df[line,5] = "TRUE"
                T,obj,time,nodes,gap = oct(D,0,X_train[c],Y_train,K,C = 20, multi_V = true, quad=true,epsi_var = true, time_limit = time_l)
                if c>3
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_BQ_D",D,"_EPSVAR_part",part,".txt"),cross_prod=true)
                elseif c == 1
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_BQ_D",D,"_EPSVAR_part",part,".txt"))
                else
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_BQ_D",D,"_EPSVAR_part",part,".txt"),power = c)
                end
                df[line,6] = string(obj)
                df[line,7] = string(nodes)

                Y_pred = predict_class(T,X_test[c])
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,8] = string(errors)
                df[line,9] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,10] = string(gap)

                line +=1

                df[line,1] = string(part)
                df[line,2] = string(D)
                df[line,3] = caract[c]
                df[line,4] = "flot"
                df[line,5] = "FALSE"
                T,obj,time,nodes,gap = flow(D,X_train[c],Y_train,K,multi_V = true, epsi_var = false, time_limit = time_l)
                if c>3
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_F_D",D,"_EPSnonVAR_part",part,".txt"),cross_prod=true)
                elseif c == 1
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_F_D",D,"_EPSnonVAR_part",part,".txt"))
                else
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_F_D",D,"_EPSnonVAR_part",part,".txt"),power = c)
                end
                df[line,6] = string(obj)
                df[line,7] = string(nodes)

                Y_pred = predict_class(T,X_test[c])
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,8] = string(errors)
                df[line,9] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,10] = string(gap)

                line +=1

                df[line,1] = string(part)
                df[line,2] = string(D)
                df[line,3] = caract[c]
                df[line,4] = "flot"
                df[line,5] = "TRUE"
                T,obj,time,nodes,gap = flow(D,X_train[c],Y_train,K,multi_V = true, epsi_var = true, time_limit = time_l)
                if c>3
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_F_D",D,"_EPSVAR_part",part,".txt"),cross_prod=true)
                elseif c == 1
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_F_D",D,"_EPSVAR_part",part,".txt"))
                else
                    write_tree(T,string("../results/",namefile,"/",caract[c],"_F_D",D,"_EPSVAR_part",part,".txt"),power = c)
                end
                df[line,6] = string(obj)
                df[line,7] = string(nodes)

                Y_pred = predict_class(T,X_test[c])
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,8] = string(errors)
                df[line,9] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,10] = string(gap)

                line +=1
            end
        end
    end

    df = DataFrame(df,["Partition","Profondeur","Caracteristique","Formulation","Epsilon_variable","Objectif","Nodes","Erreurs","Temps","Gap"])

    CSV.write(string("../results/",namefile,"/results_",namefile,"_train_test.csv"), df)
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