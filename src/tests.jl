include("optimal_tree.jl")
include("data_formatting.jl")
using CSV
using DataFrames
using Statistics

function test_univarie(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1)
    include(string("../data/",namefile,".txt"))

    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    end
    
    train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)

    df = Array{String,2}(undef,5*5*(depth_max-1),8)

    line = 1

    for part in 1:5
        X_train = X[train_indexes[part],:]
        Y_train = Y[train_indexes[part]]

        X_test = X[test_indexes[part],:]
        Y_test = Y[test_indexes[part]]

        n_min = Int(floor(2^depth_max/(length(Y_train)*10)))
        for D in 2:depth_max
            
            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"_OCT_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            write_tree(T,string("../results/univariate_test/",namefile,"_OCT-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "F"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"_F_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "F-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,time_limit = time_l)
            write_tree(T,string("../results/univariate_test/",namefile,"_F-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            write_tree(T,string("../results/univariate_test/",namefile,"_P_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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

function test_quadratic(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1)
    include(string("../data/",namefile,".txt"))

    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    end
    
    train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)

    df = Array{String,2}(undef,5*5*(depth_max-1),8)

    line = 1

    for part in 1:5
        X_train = X[train_indexes[part],:]
        Y_train = Y[train_indexes[part]]

        X_test = X[test_indexes[part],:]
        Y_test = Y[test_indexes[part]]

        n_min = Int(floor(2^depth_max/(length(Y_train)*10)))
        for D in 2:depth_max
            
            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "QcOCT-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,quadratic_constraints=true,time_limit = time_l)
            write_tree(T,string("../results/quadratic_test/",namefile,"_QcOCT-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QoOCT-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,quadratic_objective=true,time_limit = time_l)
            write_tree(T,string("../results/quadratic_test/",namefile,"_QoOCT-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QcoOCT-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,quadratic_constraints=true,quadratic_objective=true,time_limit = time_l)
            write_tree(T,string("../results/quadratic_test/",namefile,"_QcoOCT-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QF-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,variable_epsilon=true,quadratic_constraints=true,time_limit = time_l)
            write_tree(T,string("../results/quadratic_test/",namefile,"_QF-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QP"
            T, obj, error, time, nodes, gap = oct_parmentier(D,n_min,X_train,Y_train,K,alpha=alpha,quadratic_objective=true,time_limit = time_l)
            write_tree(T,string("../results/quadratic_test/",namefile,"_QP_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
    
    csv_file = string("../results/quadratic_test/",namefile,"_train_test.csv")
    touch(csv_file)
    CSV.write(csv_file, df)
end

function test_multivariate(namefile::String,depth_max::Int64;alpha::Float64=0.0,time_l::Int64=-1)
    include(string("../data/",namefile,".txt"))

    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    end
    
    train_indexes,test_indexes = cross_validation_partitioning(Y,K,5)

    df = Array{String,2}(undef,4*5*(depth_max-1),8)

    line = 1

    for part in 1:5
        X_train = X[train_indexes[part],:]
        Y_train = Y[train_indexes[part]]

        X_test = X[test_indexes[part],:]
        Y_test = Y[test_indexes[part]]

        n_min = Int(floor(2^depth_max/(length(Y_train)*10)))
        for D in 2:depth_max
            df[line,1] = string(part)
            df[line,2] = string(D)
            df[line,3] = "QcoOCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,multi_variate=true,quadratic_constraints=true,quadratic_objective = true,time_limit = time_l)
            write_tree(T,string("../results/multivariate_test/",namefile,"_QcoOCT-H_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QcoOCT-H-e"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,multi_variate=true,variable_epsilon=true,quadratic_constraints=true,quadratic_objective = true,time_limit = time_l)
            write_tree(T,string("../results/multivariate_test/",namefile,"_QcoOCT-H-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QF-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,multi_variate=true,quadratic_constraints=true,time_limit = time_l)
            write_tree(T,string("../results/multivariate_test/",namefile,"_QF-H_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
            df[line,3] = "QF-H-e"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,alpha=alpha,multi_variate=true,variable_epsilon=true,quadratic_constraints=true,time_limit = time_l)
            write_tree(T,string("../results/multivariate_test/",namefile,"_QF-H-e_D",D,"_",part,".txt"))

            df[line,4] = string(error)
            df[line,5] = string(nodes)

            Y_pred = predict_class(T,X_test)
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
    
    csv_file = string("../results/multivariate_test/",namefile,"_train_test.csv")
    touch(csv_file)
    CSV.write(csv_file, df)
end

function tests(namefile::String,depth_max::Int64,;alpha::Float64=0.0,time_l::Int64=-1,depth_min::Int64=2,random_seed::Int64=-1)
    include(string("../data/",namefile,".txt"))
    n = length(Y)

    if random_seed != -1
        Random.seed!(random_seed)
    end
    
    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    end

    nb_models = 14
    nb_tests = 10
    df = Array{Any,2}(undef,nb_models*nb_tests*(depth_max-depth_min+1),8)

    line = 1

    for it in 1:nb_tests
        train,test = train_test_indexes(n,0.2)

        X_train = X[train,:]
        Y_train = Y[train]
        X_test = X[test,:]
        Y_test = Y[test]

        n_min = Int(floor(2^depth_max/(length(Y_train)*10)))

        for D in depth_min:depth_max
            # model 1 : OCT
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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
            write_tree(T,string("../results/all_tests/",namefile,"_OCT-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 3 : Qc-OCT-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qc-OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_constraints=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_Qc-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 4 : Qo-OCT-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qo-OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_Qo-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 5 : Qco-OCT-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qco-OCT"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,quadratic_constraints=true,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_Qco-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 6 : OCT-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "OCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_OCT-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 7 : Qco-OCT-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Qco-OCT-H"
            T, obj, error, time, nodes, gap = oct(D,n_min,X_train,Y_train,K,multi_variate=true,quadratic_constraints=true,quadratic_objective=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_Qco-OCT-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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
            write_tree(T,string("../results/all_tests/",namefile,"_F_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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
            write_tree(T,string("../results/all_tests/",namefile,"_F-e_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 10 : Q-F-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-F"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,quadratic_constraints=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_Q-F_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 11 : F-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "F-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_F-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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

            # model 12 : Q-F-H-e
            df[line,1] = it
            df[line,2] = D
            df[line,3] = "Q-F-H"
            T, obj, error, time, nodes, gap = flow(D,X_train,Y_train,K,quadratic_constraints=true,multi_variate=true,alpha=alpha,time_limit=time_l)
            write_tree(T,string("../results/all_tests/",namefile,"_Q-F-H_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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
            write_tree(T,string("../results/all_tests/",namefile,"_D-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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
            write_tree(T,string("../results/all_tests/",namefile,"_Q-D-OCT_D",D,"_",it,".txt"))

            df[line,4] = error/length(Y_test)
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
        csv_file = string("../results/all_tests/",namefile,"_train_test_series",it,".csv")
        touch(csv_file)
        CSV.write(csv_file, df_interm)
    end

    df = DataFrame(df,["Partition","Profondeur","Modele","Erreurs_train","Nodes","Erreurs_test","Temps","Gap"])
    
    csv_file = string("../results/all_tests/",namefile,"_train_test.csv")
    touch(csv_file)
    CSV.write(csv_file, df)
end

function aggregate_results_of_tests(namefile::String,depth_max::Int64,;time_l::Int64=-1,depth_min::Int64=2)
    if time_l != -1
        namefile = string(namefile,"_time_",time_l,"_alpha_",alpha)
    end

    models = ["OCT","OCT-e","Qc-OCT","Qo-OCT","Qco-OCT","OCT-H","Qco-OCT-H","F","F-e","Q-F","F-H","Q-F-H","D-OCT","Q-D-OCT"]

    nb_models = 14

    csv_file = string("../results/all_tests/",namefile,"_train_test.csv")
    CSV.read(csv_file,df)

    df2 = Array{String,2}(undef,nb_models,(depth_max-depth_min+1)*6 + 1)

    line = 1

    for m in 1:length(models)
        df2[line,1] = models[m]

        for D in depth_min:depth_max
            temps = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Model, Ref([models[m]]))),:Temps]

            df2[line,6*(D-depth_max) + 2] = mean(temps)
            df2[line,6*(D-depth_max) + 3] = std(temps)

            erreur_train = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Model, Ref([models[m]]))),:Erreurs_train]

            df2[line,6*(D-depth_max) + 4] = mean(erreur_train)*100
            df2[line,6*(D-depth_max) + 5] = std(erreur_train)*100

            erreur_test = df[(in.(df.Profondeur, Ref([D]))) .& (in.(df.Model, Ref([models[m]]))),:Erreurs_test]

            df2[line,6*(D-depth_max) + 4] = mean(erreur_test)*100
            df2[line,6*(D-depth_max) + 5] = std(erreur_test)*100
        end
        line +=1
    end

    csv_file = string("../results/all_tests/",namefile,"_aggregated_results.csv")
    touch(csv_file)
    CSV.write(csv_file, df2)
end