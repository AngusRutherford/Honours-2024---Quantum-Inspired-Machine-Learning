include("RealRealHighDimension.jl")

using Base.Threads
using DelimitedFiles

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
encoding = stoudenmire()
encode_classes_separately = false
train_classes_separately = false
dtype = encoding.iscomplex ? ComplexF64 : Float64

function makeDataSet(N, α, β, corr_location, rng)
    x = zeros(N)
    for i in 1:N
        if i == 1
            x[i] = randn(rng)
        else
            x[i] = α * x[i-1] + randn(rng)
        end
    end
    x[1] += β * x[corr_location]
    return x
end

seed_1 = 69
seed_2 = 420 # blaze it
rng_1 = MersenneTwister(seed_1)
rng_2 = MersenneTwister(seed_2)
α_1 = 0
α_2 = 0
N = 20
M = 300
train_accs_OBC = zeros(20, 19)
test_accs_OBC = zeros(20, 19)
train_accs_PBC = zeros(20, 19)
test_accs_PBC = zeros(20, 19)

Threads.@threads for seed in 661:700
    rng_1 = MersenneTwister(seed_1)
    rng_2 = MersenneTwister(seed_2)
    
    for betas = 0.1:0.1:2
        β_1 = betas
        β_2 = -betas
        for corr_loc = 1:19
            train_OBC_vec = []
            test_OBC_vec = []
            train_PBC_vec = []
            test_PBC_vec = []
            dataset_1 = zeros(M, N)
            dataset_2 = zeros(M, N)
            
            for i in 1:M
                dataset_1[i, :] = makeDataSet(N, α_1, β_1, corr_loc+1, rng_1)
            end

            for i in 1:M
                dataset_2[i, :] = makeDataSet(N, α_2, β_2, corr_loc+1, rng_2)
            end

            X_train = vcat(dataset_1[1:Int(M/2), :], dataset_2[1:Int(M/2), :])
            X_test = vcat(dataset_1[Int(M/2)+1:M, :], dataset_2[Int(M/2)+1:M, :])
            y_train = vcat(Int.(zeros(Int(M/2))), Int.(ones(Int(M/2))))
            y_test = vcat(Int.(zeros(Int(M/2))), Int.(ones(Int(M/2))))

            # OBC Training
            opts=Options(; nsweeps=15, chi_max=3,  update_iters=1, verbosity=verbosity, dtype=dtype, 
                         loss_grad=loss_grad_KLD, bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, 
                         eta=0.3, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
                         encode_classes_separately=encode_classes_separately, 
                         train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

            W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; 
                                                                   random_state=seed, chi_init=4, opts=opts, test_run=false)
            push!(train_OBC_vec, info["train_acc"][end])
            push!(test_OBC_vec, info["test_acc"][end])

            # PBC Training
            opts=Options(; nsweeps=30, chi_max=3,  update_iters=1, verbosity=verbosity, dtype=dtype, 
                         loss_grad=loss_grad_KLD, bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, 
                         eta=0.3, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
                         encode_classes_separately=encode_classes_separately, 
                         train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

            W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; 
                                                                   random_state=seed, chi_init=4, opts=opts, test_run=false)
            push!(train_PBC_vec, info["train_acc"][end])
            push!(test_PBC_vec, info["test_acc"][end])

            # Storing results
            train_accs_OBC[Int(10*betas), corr_loc] = mean(train_OBC_vec)
            test_accs_OBC[Int(10*betas), corr_loc] = mean(test_OBC_vec)
            train_accs_PBC[Int(10*betas), corr_loc] = mean(train_PBC_vec)
            test_accs_PBC[Int(10*betas), corr_loc] = mean(test_PBC_vec)
        end
    end
end

writedlm("angus_correlation_alphapm00_eta03_train_OBC_seed_661_700.csv", train_accs_OBC, ',')
writedlm("angus_correlation_alphapm00_eta03_test_OBC_seed_661_700.csv", test_accs_OBC, ',')
writedlm("angus_correlation_alphapm00_eta03_train_PBC_seed_661_700.csv", train_accs_PBC, ',')
writedlm("angus_correlation_alphapm00_eta03_test_PBC_seed_661_700.csv", test_accs_PBC, ',')