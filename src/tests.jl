include("optimal_tree.jl")
include("data_formatting.jl")
using CSV
using DataFrames
using Statistics

function tests(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1,depth_min::Int64=2,random_seed::Int64=-1)
    include(string("../data/",namefile,".txt"))
    n = length(Y)

    if random_seed != -1
        Random.seed!(random_seed)
    end
    
    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    else
        namefile = string(namefile,"_alpha_",alpha)
    end

    nb_models = 14
    nb_tests = 5
    df = Array{Any,2}(undef,nb_models*nb_tests*(depth_max-depth_min+1),8)

    line = 1

    for it in 1:nb_tests
        train,test = train_test_indexes(n,0.2)

        X_train = X[train,:]
        Y_train = Y[train]
        X_test = X[test,:]
        Y_test = Y[test]

        n_min = Int(floor(2^depth_max/(length(Y_train)*10))) # what????????????????? 0 I guess then ahah

        for D in depth_min:depth_max
            # model 1 : OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 2 : OCT-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,variable_epsilon=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/OCT-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 3 : Qc-OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qc-OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_constraints=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Qc-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 4 : Qo-OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qo-OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Qo-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 5 : Qco-OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qco-OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_constraints=true,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Qco-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 6 : OCT-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/OCT-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 7 : Qco-OCT-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qco-OCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,multi_variate=true,quadratic_constraints=true,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Qco-OCT-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 8 : F
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "F"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/F_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 9 : F-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "F-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,variable_epsilon=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/F-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 10 : Q-F
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-F"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,quadratic_constraints=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Q-F_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 11 : F-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "F-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/F-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 12 : Q-F-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-F-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,quadratic_constraints=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Q-F-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 13 : D-OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "D-OCT"
            T, obj, error, time, nodes, gap = oct_parmentier(D,n_min,X_train,Y_train,K,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/D-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 14 : Q-D-OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-D-OCT"
            T, obj, error, time, nodes, gap = oct_parmentier(D,n_min,X_train,Y_train,K,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"/Q-D-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1
        end

        df_interm = DataFrame(df[(nb_models*(depth_max-depth_min+1)*(it - 1) + 1):(nb_models*(depth_max-depth_min+1)*it),2:8],["Profondeur","Modele","Erreurs_train","Nodes","Erreurs_test","Temps","Gap"])
        csv_file = string("../results/all_tests/",namefile,"/train_test_series",it,".csv")
        touch(csv_file)
        CSV.write(csv_file, df_interm)
    end

    df = DataFrame(df,["Partition","Profondeur","Modele","Erreurs_train","Nodes","Erreurs_test","Temps","Gap"])
    
    csv_file = string("../results/all_tests/",namefile,"/train_test.csv")
    touch(csv_file)
    CSV.write(csv_file, df)
end

function float_to_string(number::Float64;precision::Int64=4)
    return(string(Int(floor(number)),".",Int(round((number-floor(number))*(10^precision)))))
end

function aggregate_results_of_tests(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1,depth_min::Int64=2)
    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    else 
        namefile = string(namefile,"_alpha_",alpha)
    end

    models = ["OCT","OCT-e","Qc-OCT","Qo-OCT","Qco-OCT","OCT-H","Qco-OCT-H","F","F-e","Q-F","F-H","Q-F-H","D-OCT","Q-D-OCT"]

    nb_models = 14

    csv_file = string("../results/all_tests/",namefile,"/train_test.csv")
    #csv_file = string("../results/all_tests/",namefile,"/",namefile,"_train_test.csv") # pour les iris car changement
    df = DataFrame(CSV.File(csv_file))

    df2 = Array{String,2}(undef,nb_models,(depth_max-depth_min+1)*3 + 1)

    line = 1

    for m in 1:length(models)
        df2[line,1] = models[m]

        column = 2
        for D in depth_min:depth_max
            temps = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Modele, Ref([models[m]]))),:Temps]

            df2[line,column] = string(float_to_string(mean(temps),precision=2)," ",Char(0x000B1)," ",float_to_string(std(temps),precision=2))
            column += 1

            erreur_train = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Modele, Ref([models[m]]))),:Erreurs_train]

            df2[line,column] = string(float_to_string(mean(erreur_train)*100,precision=2)," ",Char(0x000B1)," ",float_to_string(std(erreur_train)*100,precision=2))
            column += 1

            erreur_test = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Modele, Ref([models[m]]))),:Erreurs_test]

            df2[line,column] = string(float_to_string(mean(erreur_test)*100,precision=2)," ",Char(0x000B1)," ",float_to_string(std(erreur_test)*100,precision=2))
            column += 1
        end
        line +=1
    end

    df2 = DataFrame(df2)

    csv_file = string("../results/all_tests/",namefile,"/aggregated_results.csv")
    touch(csv_file)
    CSV.write(csv_file, df2)
end

function epsilon_tests(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l=Int64=-1,depth_min::Int64=2, random_seed::Int64=-1)
    include(string("../data/",namefile,".txt"))
    n = length(Y)

    if random_seed != -1
        Random.seed!(random_seed)
    end
    
    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    else
        namefile = string(namefile,"_alpha_",alpha)
    end

    nb_models = 8
    nb_tests = 10
    df = Array{Any,2}(undef,nb_models*nb_tests*(depth_max-depth_min+1),8)

    line = 1

    for it in 1:nb_tests
        train,test = train_test_indexes(n,0.2)

        X_train = X[train,:]
        Y_train = Y[train]
        X_test = X[test,:]
        Y_test = Y[test]

        n_min = 0

        for D in depth_min:depth_max
            # model 1 : OCT-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_OCT-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 2 : OCT-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT-H-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,variable_epsilon=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_OCT-H-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 3 : Qo-OCT-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qo-OCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_objective=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_Qo-OCT-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 4 : Qo-OCT-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qo-OCT-H-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,variable_epsilon=true,quadratic_objective=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_Qo-OCT-H-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 5 : F-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "F-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_F-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 6 : F-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "F-H-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,variable_epsilon=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_F-H-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 7 : Q-F-H
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-F-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,quadratic_constraints=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_Q-F-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1

            # model 8 : Q-F-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-F-H-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,variable_epsilon=true,quadratic_constraints=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/eps_var/",namefile,"_Q-F-H-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_train)
            df[line,5] = nodes

            Y_pred = predict_class(T,X_test)
            errors = 0
            for i in 1:length(Y_pred)
                if Y_pred[i] != Y_test[i]
                    errors+=1
                end
            end

            errors /= length(Y_pred)

            df[line,6] = errors
            df[line,7] = Int(floor(time)) + Int(floor((time-floor(time))*10000))/10000
            df[line,8] = gap

            line +=1
        end
    end

    df = DataFrame(df,["Partition","Profondeur","Modele","Erreurs_train","Nodes","Erreurs_test","Temps","Gap"])
    
    csv_file = string("../results/eps_var/",namefile,"_train_test.csv")
    touch(csv_file)
    CSV.write(csv_file, df)
end

function aggregate_results_of_epsilon_tests(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1,depth_min::Int64=2)
    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    else 
        namefile = string(namefile,"_alpha_",alpha)
    end

    models = ["OCT-H","OCT-H-e","Qo-OCT-H","Qo-OCT-H-e","F-H","F-H-e","Q-F-H","Q-F-H-e"]

    nb_models = 8

    csv_file = string("../results/eps_var/",namefile,"_train_test.csv")
    df = DataFrame(CSV.File(csv_file))

    df2 = Array{String,2}(undef,nb_models,(depth_max-depth_min+1)*3 + 1)

    line = 1

    for m in 1:length(models)
        df2[line,1] = models[m]

        column = 2
        for D in depth_min:depth_max
            temps = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Modele, Ref([models[m]]))),:Temps]

            df2[line,column] = string(float_to_string(mean(temps),precision=2)," ",Char(0x000B1)," ",float_to_string(std(temps),precision=2))
            column += 1

            erreur_train = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Modele, Ref([models[m]]))),:Erreurs_train]

            df2[line,column] = string(float_to_string(mean(erreur_train)*100,precision=2)," ",Char(0x000B1)," ",float_to_string(std(erreur_train)*100,precision=2))
            column += 1

            erreur_test = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Modele, Ref([models[m]]))),:Erreurs_test]

            df2[line,column] = string(float_to_string(mean(erreur_test)*100,precision=2)," ",Char(0x000B1)," ",float_to_string(std(erreur_test)*100,precision=2))
            column += 1
        end
        line +=1
    end

    df2 = DataFrame(df2)

    csv_file = string("../results/eps_var/",namefile,"_aggregated_results.csv")
    touch(csv_file)
    CSV.write(csv_file, df2)
end