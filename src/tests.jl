include("optimal_tree.jl")
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

function test_parmentier()
    datasets = ["iris","wine","seeds"]
    df = Array{String,2}(undef,165,9)
    
    line = 1

    for n in 1:length(datasets)
        include(string("../data/",datasets[n],".txt"))
        train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)

        for part in 1:5
            X_train = X[train_indexes[part],:]
            Y_train = Y[train_indexes[part]]

            X_test = X[test_indexes[part],:]
            Y_test = Y[test_indexes[part]]

            t1 = time()

            X_p,nb_val = compute_Xp(X_train,Y_train)

            t2 = time()
            t = t2 -t1

            df[line,1] = datasets[n]
            df[line,2] = string(part)
            df[line,3] = "-"
            df[line,4] = "Preprocessing Parmentier"
            df[line,5] = "-"
            df[line,6] = "-"
            df[line,7] = "-"
            df[line,8] = string(Int(floor(t)),".",Int(floor((t-floor(t))*1000)))
            df[line,9] = "-"

            line += 1
            for D in 2:3
                df[line,1] = datasets[n]
                df[line,2] = string(part)
                df[line,3] = string(D)
                df[line,4] = "MIP Parmentier"

                T, obj,error,time,nodes,gap,time_post_p = oct_parmentier(D,0,X_train,Y_train,K,time_limit=3600,X_p=X_p,nb_val=nb_val)
                write_tree(T,string("../results/parmentier/",datasets[n],"_MIP_Parmentier_",D,"_part",part,".txt"))
                df[line,5] = string(obj)
                df[line,6] = string(nodes)

                Y_pred = predict_class(T,X_test)
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,7] = string(errors)
                df[line,8] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,9] = string(gap)

                line += 1

                df[line,1] = datasets[n]
                df[line,2] = string(part)
                df[line,3] = string(D)
                df[line,4] = "Postprocessing Parmentier"
                df[line,5] = "-"
                df[line,6] = "-"
                df[line,7] = "-"
                df[line,8] = string(Int(floor(time_post_p)),".",Int(floor((time_post_p-floor(time_post_p))*1000)))
                df[line,9] = "-"

                line +=1

                df[line,1] = datasets[n]
                df[line,2] = string(part)
                df[line,3] = string(D)
                df[line,4] = "MIP Bertsimas"

                T ,obj,error,time,nodes,gap = oct(D,0,X_train,Y_train,K,quad=true,time_limit=3600)
                write_tree(T,string("../results/parmentier/",datasets[n],"_MIP_Bertsimas_",D,"_part",part,".txt"))
                df[line,5] = string(obj)
                df[line,6] = string(nodes)

                Y_pred = predict_class(T,X_test)
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,7] = string(errors)
                df[line,8] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,9] = string(gap)

                line += 1

                df[line,1] = datasets[n]
                df[line,2] = string(part)
                df[line,3] = string(D)
                df[line,4] = "MIP Bertsimas epsi var"

                T ,obj,error,time,nodes,gap = oct(D,0,X_train,Y_train,K,epsi_var=true,quad=true,time_limit=3600)
                write_tree(T,string("../results/parmentier/",datasets[n],"_MIP_Bertsimas_epsi_var_",D,"_part",part,".txt"))
                df[line,5] = string(obj)
                df[line,6] = string(nodes)

                Y_pred = predict_class(T,X_test)
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,7] = string(errors)
                df[line,8] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,9] = string(gap)

                line += 1

                df[line,1] = datasets[n]
                df[line,2] = string(part)
                df[line,3] = string(D)
                df[line,4] = "MIP Bertsimas multi lin epsi var"

                T ,obj,error,time,nodes,gap = oct(D,0,X_train,Y_train,K,epsi_var=true,multi_V=true,time_limit=3600)
                write_tree(T,string("../results/parmentier/",datasets[n],"_MIP_Bertsimas_multi_lin_epsi_var_",D,"_part",part,".txt"))
                df[line,5] = string(obj)
                df[line,6] = string(nodes)

                Y_pred = predict_class(T,X_test)
                errors = 0
                for i in 1:length(Y_pred)
                    if Y_pred[i] != Y_test[i]
                        errors+=1
                    end
                end

                errors /= length(Y_pred)

                df[line,7] = string(errors)
                df[line,8] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
                df[line,9] = string(gap)

                line += 1
            end
        end
    end

    df = DataFrame(df,["Dataset","Partition","Profondeur","Caracteristiques","Objectif","Nodes","Erreurs","Temps","Gap"])

    CSV.write(string("../results/parmentier/parmentier_train_test.csv"), df)
end

function test_optimal_trees(namefile,depth_max::Int64;time_l::Int64=-1)
    include(string("../data/",namefile,".txt"))

    if time_l != -1
        namefile = string(namefile,"_time_",time_l)
    end

    caract = ["L","P2","P3","Q"]
    
    train_indexes,test_1_indexes,test_2_indexes = double_test_partitionning(Y,K,5)

    df = Array{String,2}(undef,20,4)
    
    line = 1

    for part in 1:5
        X_train = Array{Float64}[]
        Y_train = Y[train_indexes[part]]
        push!(X_train,X[train_indexes[part],:])
        push!(X_train,add_powers(X_train[1],2))
        push!(X_train,add_powers(X_train[1],3))
        push!(X_train,add_cross_product(X_train[1]))

        n_min = Int(floor(2^depth_max/(length(Y_train)*10)))

        X_test_1 = Array{Float64}[]
        Y_test_1 = Y[test_1_indexes[part]]
        push!(X_test_1,X[test_1_indexes[part],:])
        push!(X_test_1,add_powers(X_test_1[1],2))
        push!(X_test_1,add_powers(X_test_1[1],3))
        push!(X_test_1,add_cross_product(X_test_1[1]))

        X_test_2 = Array{Float64}[]
        Y_test_2 = Y[test_2_indexes[part]]
        push!(X_test_2,X[test_2_indexes[part],:])
        push!(X_test_2,add_powers(X_test_2[1],2))
        push!(X_test_2,add_powers(X_test_2[1],3))
        push!(X_test_2,add_cross_product(X_test_2[1]))

        for c in 1:4
            df[line,1] = string(part)
            df[line,2] = caract[c]

            if c == 1
                T, score = optimal_tree(depth_max,n_min,X_train[c],X_test_1[c],Y_train,Y_test_1,K,multi_V=true,epsi_var=true,time_limit=time_l)
                write_tree(T,string("../results/optimal_",namefile,"/L_depth_",T.D,"_part",part,".txt"))
            elseif c < 4
                T, score = optimal_tree(depth_max,n_min,X_train[c],X_test_1[c],Y_train,Y_test_1,K,multi_V=true,epsi_var=true,time_limit=time_l,power=c)
                write_tree(T,string("../results/optimal_",namefile,"/P",c,"_depth_",T.D,"_part",part,".txt"))
            else
                T, score = optimal_tree(depth_max,n_min,X_train[c],X_test_1[c],Y_train,Y_test_1,K,multi_V=true,epsi_var=true,time_limit=time_l,cross_prod=true)
                write_tree(T,string("../results/optimal_",namefile,"/Q_depth_",T.D,"_part",part,".txt"))
            end

            df[line,3] = string(score)
            Y_pred = predict_class(T,X_test_2[c])
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test_2[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)
            df[line,4] = string(errors)

            line += 1
        end
    end

    df = DataFrame(df,["Partition","Caracteristique","Erreurs train","Erreurs test"])

    name_of_csv = string("../results/optimal_",namefile,"/results_optimal_",namefile,"_train_test.csv")
    touch(name_of_csv)
    CSV.write(name_of_csv, df)
end

function test_univarie(namefile,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1)
    include(string("../data/",namefile,".txt"))

    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    end

    modele = ["OCT","OCT-e","F","F-e","P"]
    
    train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)

    df = Array{String,2}(undef,5*length(model)*(depth_max-1),8)

    line = 1

    for part in 1:5
        X_train = X[train_indexes[part]]
        Y_train = Y[train_indexes[part]]

        X_test = X[test_indexes[part]]
        Y_test = Y[test_indexes[part]]

        n_min = Int(floor(2^depth_max/(length(Y_train)*10)))
        for D in 2:depth_max
            
            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,time_limit = time_l)
            write_tree(T,string("../results/",namefile,"/OCT",D,part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test[c])
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = string(errors)
            df[line,7] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(gap)

            line +=1

            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "OCT-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"/OCT-e",D,part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test[c])
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = string(errors)
            df[line,7] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(gap)

            line +=1

            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "OCT"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"/F",D,part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test[c])
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = string(errors)
            df[line,7] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(gap)

            line +=1

            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "OCT-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"/F-e",D,part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test[c])
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = string(errors)
            df[line,7] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(gap)

            line +=1


            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "P"
            T, obj, error, time, nodes, gap = oct_parmentier(D,n_min,X_train,Y_train,K,alpha=alpha,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"/P",D,part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test[c])
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = string(errors)
            df[line,7] = string(Int(floor(time)),".",Int(floor((time-floor(time))*1000)))
            df[line,8] = string(gap)

            line +=1
        end
    end
    df = DataFrame(df,["Partition","Profondeur","Modele","Erreurs train","Nodes","Erreurs test","Temps","Gap"])
    
    csv_file = string("../results/univariate_test/",namefile,"_train_test.csv")
    touch(csv_file)
    CSV.write(csv_file, df)
end