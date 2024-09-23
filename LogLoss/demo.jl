include("RealRealHighDimension.jl")

using DelimitedFiles
using DataFrames
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")
# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST_labels.csv", ',')
# y_test = Int.(vec(y_test))

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_A.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_A_labels.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_B.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_B_labels.csv", ',')

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_A.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_A_labels.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_B.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_B_labels.csv", ',')

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_A.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_A_labels.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_B.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_B_labels.csv", ',')

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_10_1_0.1_class_A_v2.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_10_1_0.1_class_A_labels_v2.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_10_1_0.1_class_B_v2.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_10_1_0.1_class_B_labels_v2.csv", ',')

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/class1_data.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/class1_labels.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/class2_data.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/class2_labels.csv", ',')

# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TEST.txt"
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TRAIN.txt", 
# "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TEST.txt", "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TEST.txt")

# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TEST.txt"
# (X_train, y_train), (X_test, y_test) = load_splits_txt("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TRAIN.txt", 
# "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TEST.txt", "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TEST.txt")
# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/ArrowHead_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/ArrowHead_TEST.txt"

# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/MedicalImages_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/MedicalImages_TEST.txt"

# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/ECG200_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/ECG200_TEST.txt"

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_train_20_40_80_40_80_1.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_test_20_40_80_40_80_1.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_train_labels_20_40_80_40_80_1.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_test_labels_20_40_80_40_80_1.csv", ',')
# y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_shifted_train_20_40_80_40_80_0.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_shifted_test_20_40_80_40_80_0.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_shifted_train_labels_20_40_80_40_80_0.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/sine_circle_shifted_test_labels_20_40_80_40_80_0.csv", ',')
# y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_train_10_40_80_1.csv", ',')
# #X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_train_10_40_80_1_shifted.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_test_10_40_80_1.csv", ',')
# #X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_test_10_40_80_1_shifted.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_train_10_40_80_1_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_test_10_40_80_1_labels.csv", ',')
# y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_train_20_20_40_1.csv", ',')
# # #X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_train_20_20_40_1_shifted.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_test_20_20_40_1.csv", ',')
# # #X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_test_20_20_40_1_shifted.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_train_20_20_40_1_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_circle_test_20_20_40_1_labels.csv", ',')
# y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_train_alpha=1_betas_pm08.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_test_alpha=1_betas_pm08.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_train_alpha=1_betas_pm08_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_test_alpha=1_betas_pm08_labels.csv", ',')
# y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/whatisgoingon_train.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/whatisgoingon_test.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/whatisgoingon_train_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/whatisgoingon_test_labels.csv", ',')
# y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_train_alpha=1_betas_pm08_1-2.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_test_alpha=1_betas_pm08_1-2.csv", ',')
# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_train_alpha=1_betas_09_01.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_test_alpha=1_betas_09_01.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_train_alpha=1_betas_09_01_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/sahand_test_alpha=1_betas_09_01_labels.csv", ',')
# y_test = Int.(vec(y_test))

# #Read and parse the data
# parsed_data_train = read_and_parse(file_path_train)
# parsed_data_test = read_and_parse(file_path_test)

# # Convert the list of lists into a DataFrame
# training_data = DataFrame(parsed_data_train, :auto)
# testing_data = DataFrame(parsed_data_test, :auto)

# training_data_matrix = Matrix{Float64}(training_data[2:end, :])
# training_data_matrix = Matrix(transpose(training_data_matrix))

# testing_data_matrix = Matrix{Float64}(testing_data[2:end, :])
# testing_data_matrix = Matrix(transpose(testing_data_matrix))

# training_labels = Int.(Vector(training_data[1, :]))
# testing_labels = Int.(Vector(testing_data[1, :]))

# training_labels = training_labels .- 1
# testing_labels = testing_labels .- 1

# X_train = training_data_matrix
# X_test = testing_data_matrix
# y_train = training_labels
# y_test = testing_labels

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_class_A.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/sine_class_B.csv", ',')
# X_train = vcat(class_A[1:50, :], class_B[1:50, :])
# X_test = vcat(class_A[51:end, :], class_B[51:end, :])
# y_train = [zeros(Int, 50); ones(Int, 50)]
# y_test = [zeros(Int, 50); ones(Int, 50)]
# X_train = vcat(class_A[1:250, :], class_B[1:250, :])
# X_test = vcat(class_A[251:end, :], class_B[251:end, :])
# y_train = vcat(class_A_labels[1:250], class_B_labels[1:250])
# y_train = Int.(vec(y_train))
# y_test = vcat(class_A_labels[251:end], class_B_labels[251:end])
# y_test = Int.(vec(y_test))

# X_train = hcat([circshift(X_train[i, :], 50) for i in 1:size(X_train, 1)]...)
# X_train = Matrix(transpose(X_train))
# X_test = hcat([circshift(X_test[i, :], 50) for i in 1:size(X_test, 1)]...)
# X_test = Matrix(transpose(X_test))

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = legendre(norm=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=20, chi_max=40,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=12, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)


if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=1, chi_init=4, opts=opts, test_run=false)

    #print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
    sweep_summary(info)
end



# train_accs_OBC = zeros(5, 10)
# test_accs_OBC = zeros(5, 10)
# train_loss_OBC = zeros(5, 10)
# test_loss_OBC = zeros(5, 10)
# train_accs_PBC = zeros(5, 10)
# test_accs_PBC = zeros(5, 10)
# train_loss_PBC = zeros(5, 10)
# test_loss_PBC = zeros(5, 10)
# for ds = 2:2
#     for chis = 2:11
#         println(ds)
#         println(chis)
#         opts=Options(; nsweeps=20, chi_max=chis,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#         bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=ds, aux_basis_dim=2, encoding=encoding, 
#         encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

#         W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=1, chi_init=4, opts=opts, test_run=false)

#         train_accs_OBC[(ds-1), (chis-1)] = info["train_acc"][end]
#         test_accs_OBC[(ds-1), (chis-1)] = info["test_acc"][end]
#         train_loss_OBC[(ds-1), (chis-1)] = info["train_KL_div"][end]
#         test_loss_OBC[(ds-1), (chis-1)] = info["test_KL_div"][end]

#         opts=Options(; nsweeps=20, chi_max=chis,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#         bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=ds, aux_basis_dim=2, encoding=encoding, 
#         encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

#         W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=1, chi_init=4, opts=opts, test_run=false)

#         train_accs_PBC[(ds-1), (chis-1)] = info["train_acc"][end]
#         test_accs_PBC[(ds-1), (chis-1)] = info["test_acc"][end]
#         train_loss_PBC[(ds-1), (chis-1)] = info["train_KL_div"][end]
#         test_loss_PBC[(ds-1), (chis-1)] = info["test_KL_div"][end]
#     end
# end

# function makeDataSet(N, α, β, corr_location, rng)
#     x = zeros(N)
#     for i in 1:N
#         if i == 1
#             x[i] = randn(rng)
#         else
#             x[i] = α * x[i-1] + randn(rng)
#         end
#     end
#     x[1] += β * x[corr_location]
#     return x
# end

# seed_1 = 22
# seed_2 = 16
# α_1 = 0.5
# α_2 = -0.5
# N = 20
# M = 300
# #corr_loc = 2
# train_accs_OBC = zeros(20, 19)
# test_accs_OBC = zeros(20, 19)
# train_accs_PBC = zeros(20, 19)
# test_accs_PBC = zeros(20, 19)
# # train_accs_OBC = []
# # test_accs_OBC = []
# # train_accs_PBC = []
# # test_accs_PBC = []
# for betas = 0.1:0.1:2
#     β_1 = betas
#     β_2 = -betas
#     for corr_loc = 1:19
#         rng_1 = MersenneTwister(seed_1)
#         rng_2 = MersenneTwister(seed_2)
#         dataset_1 = zeros(M, N)
#         dataset_2 = zeros(M, N)
#         for i in 1:M
#             dataset_1[i, :] = makeDataSet(N, α_1, β_1, corr_loc+1, rng_1)
#         end
        
#         for i in 1:M
#             dataset_2[i, :] = makeDataSet(N, α_2, β_2, corr_loc+1, rng_2)
#         end

#         X_train = vcat(dataset_1[1:Int(M/2), :], dataset_2[1:Int(M/2), :])
#         X_test = vcat(dataset_1[Int(M/2)+1:M, :], dataset_2[Int(M/2)+1:M, :])
#         y_train = vcat(Int.(zeros(Int(M/2))), Int.(ones(Int(M/2))))
#         y_test = vcat(Int.(zeros(Int(M/2))), Int.(ones(Int(M/2))))


#         opts=Options(; nsweeps=30, chi_max=3,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#         bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.3, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
#         encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

#         W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=1, chi_init=4, opts=opts, test_run=false)
#         train_accs_OBC[Int(10*betas), corr_loc] = maximum(info["train_acc"])
#         test_accs_OBC[Int(10*betas), corr_loc] = maximum(info["test_acc"])
#         # push!(train_accs_OBC, maximum(info["train_acc"]))
#         # push!(test_accs_OBC, maximum(info["test_acc"]))

#         opts=Options(; nsweeps=30, chi_max=3,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#         bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.3, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
#         encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

#         W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=1, chi_init=4, opts=opts, test_run=false)
#         train_accs_PBC[Int(10*betas), corr_loc] = maximum(info["train_acc"])
#         test_accs_PBC[Int(10*betas), corr_loc] = maximum(info["test_acc"])
#         # push!(train_accs_PBC, maximum(info["train_acc"]))
#         # push!(test_accs_PBC, maximum(info["test_acc"]))
#     end
# end

# train_accuracy = zeros(50, 22)
# test_accuracy = zeros(50, 22)
# train_loss = zeros(50, 22)
# test_loss = zeros(50, 22)
# for seed = 1:50
#     W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
#     train_accuracy[seed, :] = info["train_acc"]
#     test_accuracy[seed, :] = info["test_acc"]
#     train_loss[seed, :] = info["train_KL_div"]
#     test_loss[seed, :] = info["test_KL_div"]
# end
# OBC_train_accuracy = []
# OBC_test_accuracy = []
# OBC_train_loss = []
# OBC_test_loss = []
# PBC_train_accuracy = []
# PBC_test_accuracy = []
# PBC_train_loss = []
# PBC_test_loss = []
# for bond_dim = 1:30
#     println(bond_dim)
#     opts=Options(; nsweeps=20, chi_max=bond_dim,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#     bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
#     encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 337)

#     W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)
#     index = find_stable_accuracy_threshold(info["train_acc"])
#     push!(OBC_train_accuracy, info["train_acc"][index])
#     push!(OBC_test_accuracy, info["test_acc"][index])
#     push!(OBC_train_loss, info["train_KL_div"][index])
#     push!(OBC_test_loss, info["test_KL_div"][index])

#     opts=Options(; nsweeps=20, chi_max=bond_dim,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#     bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
#     encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 337)

#     W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)
#     index = find_stable_accuracy_threshold(info["train_acc"])
#     push!(PBC_train_accuracy, info["train_acc"][index])
#     push!(PBC_test_accuracy, info["test_acc"][index])
#     push!(PBC_train_loss, info["train_KL_div"][index])
#     push!(PBC_test_loss, info["test_KL_div"][index])
# end
# train = []
# test = []
# for random_walk_seed = 11245:11250
#     opts=Options(; nsweeps=40, chi_max=2,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#     bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.05, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
#     encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = random_walk_seed)

#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=76, chi_init=4, opts=opts, test_run=false)

#     summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#     #index = find_stable_accuracy(info["train_acc"], 0.1)
#     # train_OBC_accuracies_matrix[seed, j] = info["train_acc"][end]
#     # test_OBC_accuracies_matrix[seed, j] = info["test_acc"][end]
#     if info["train_acc"][end] > 0.8
#         push!(train, info["train_acc"][end])
#         push!(test, info["test_acc"][end])
#     end
# end


# train = []
# test = []
# seeds = [11, 12, 15]
# for seed in seeds
#     W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
#     push!(train, info["train_acc"])
#     push!(test, info["test_acc"])
# end

# X_train = hcat([circshift(X_train[i, :], 10) for i in 1:size(X_train, 1)]...)
# X_train_global = Matrix(transpose(X_train))
# X_test = hcat([circshift(X_test[i, :], 10) for i in 1:size(X_test, 1)]...)
# X_test_global = Matrix(transpose(X_test))
# y_train_global = y_train
# y_test_global = y_test
# # seeds = [76, 47, 78, 79, 80, 81, 82, 83, 84, 48, 86, 87, 88, 89, 90, 91, 49, 93, 94, 95, 96, 97, 98, 99, 100]
# opts=Options(; nsweeps=40, chi_max=2,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
# bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
# encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 69)
# train_OBC_accuracies_matrix = zeros(100, 100)
# test_OBC_accuracies_matrix = zeros(100, 100)
# for seed = 1
#     for j = 1:20
#         println(j)
#         X_train = hcat([circshift(X_train_global[i, :], j) for i in 1:size(X_train_global, 1)]...)
#         X_train = Matrix(transpose(X_train))
#         X_test = hcat([circshift(X_test_global[i, :], j) for i in 1:size(X_test_global, 1)]...)
#         X_test = Matrix(transpose(X_test))
#         y_train = y_train_global
#         y_test = y_test_global

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)

#         summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#         #index = find_stable_accuracy(info["train_acc"], 0.1)
#         train_OBC_accuracies_matrix[seed, j] = info["train_acc"][end]
#         test_OBC_accuracies_matrix[seed, j] = info["test_acc"][end]
#     end
# end

# seed_offset = 75
# train_matrix = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/train_random_new_seeds_guassian_v6.csv", ',')
# test_matrix = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/test_random_new_seeds_guassian_v6.csv", ',')
# X_train_global = X_train
# X_test_global = X_test
# y_train_global = y_train
# y_test_global = y_test
# function find_zero_indices(matrix)
#     indices = []
#     for i in 1:size(matrix, 1)
#         for j in 1:size(matrix, 2)
#             if matrix[i, j] == 0.0
#                 push!(indices, (i, j))
#             end
#         end
#     end
#     return indices
# end

# zero_indices = find_zero_indices(train_matrix)

# for (i, j) in zero_indices
#     train = []
#     test = []
#     for random_walk_seed = 333:338
#         println("$i, $j, $random_walk_seed")
#         opts=Options(; nsweeps=40, chi_max=2,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
#         bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.05, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
#         encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = random_walk_seed)
#         X_train = hcat([circshift(X_train_global[k, :], j+10) for k in 1:size(X_train_global, 1)]...)
#         X_train = Matrix(transpose(X_train))
#         X_test = hcat([circshift(X_test_global[k, :], j+10) for k in 1:size(X_test_global, 1)]...)
#         X_test = Matrix(transpose(X_test))
#         y_train = y_train_global
#         y_test = y_test_global

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=i + seed_offset, chi_init=4, opts=opts, test_run=false)

#         summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#         if info["train_acc"][end] > 0.8
#             push!(train, info["train_acc"][end])
#             push!(test, info["test_acc"][end])
#         end
#     end
#     if length(train) == 0
#         begin
#         end
#     else
#         train_matrix[i, j] = mean(train)
#         test_matrix[i, j] = mean(test)
#     end
# end

# opts=Options(; nsweeps=20, chi_max=2,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
# bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
# encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 416)
# train_both_accuracies_matrix = zeros(25, 20)
# test_both_accuracies_matrix = zeros(25, 20)
# for seed = 25:25
#     for j = 1:20
#         println(j)
#         #println("both")
#         X_train = hcat([circshift(X_train_global[i, :], j) for i in 1:size(X_train_global, 1)]...)
#         X_train = Matrix(transpose(X_train))
#         X_test = hcat([circshift(X_test_global[i, :], j) for i in 1:size(X_test_global, 1)]...)
#         X_test = Matrix(transpose(X_test))
#         y_train = y_train_global
#         y_test = y_test_global

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
        
#         summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#         #index = find_stable_accuracy(info["train_acc"], 0.1)
#         if info["train_acc"][end] > -0.8
#             train_both_accuracies_matrix[seed, j] = info["train_acc"][end]
#             test_both_accuracies_matrix[seed, j] = info["test_acc"][end]
#         end
#         # push!(train_accuracies_OBC, info["train_acc"][end])
#         # push!(test_accuracies_OBC, info["test_acc"][end])
#     end
# end

# opts=Options(; nsweeps=20, chi_max=2,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
# bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.1, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
# encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 416)
# train_random_accuracies_matrix = zeros(25, 20)
# test_random_accuracies_matrix = zeros(25, 20)
# for seed = 1:25
#     for j = 1:20
#         println(j)
#         X_train = hcat([circshift(X_train_global[i, :], j) for i in 1:size(X_train_global, 1)]...)
#         X_train = Matrix(transpose(X_train))
#         X_test = hcat([circshift(X_test_global[i, :], j) for i in 1:size(X_test_global, 1)]...)
#         X_test = Matrix(transpose(X_test))
#         y_train = y_train_global
#         y_test = y_test_global

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
        
#         summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#         #index = find_stable_accuracy(info["train_acc"], 0.1)
#         if info["train_acc"][end] > 0.8
#             train_random_accuracies_matrix[seed, j] = info["train_acc"][end]
#             test_random_accuracies_matrix[seed, j] = info["test_acc"][end]
#         end
#         # push!(train_accuracies_OBC, info["train_acc"][end])
#         # push!(test_accuracies_OBC, info["test_acc"][end])
#     end
# end
# opts=Options(; nsweeps=15, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
# bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.05, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
# encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left")

# train_accuracies_PBC_left = []
# test_accuracies_PBC_left = []
# for j = 1:20
#     println(j)
#     X_train = hcat([circshift(X_train_global[i, :], -j) for i in 1:size(X_train_global, 1)]...)
#     X_train = Matrix(transpose(X_train))
#     X_test = hcat([circshift(X_test_global[i, :], -j) for i in 1:size(X_test_global, 1)]...)
#     X_test = Matrix(transpose(X_test))
#     y_train = y_train_global
#     y_test = y_test_global

#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=4756, chi_init=4, opts=opts, test_run=false)

#     summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#     #index = find_stable_accuracy(info["train_acc"], 0.1)
#     push!(train_accuracies_PBC_left, info["train_acc"][end])
#     push!(test_accuracies_PBC_left, info["test_acc"][end])
# end
# #medical training distribution: [35, 15, 25, 16, 10, 7, 18, 6, 46, 203]
# #circshift with samples
# opts=Options(; nsweeps=5, chi_max=30,  update_iters=1, verbosity=verbosity, dtype=Rdtype, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.1, rescale = [false, true], d=2, encoding=Encoding("Stoudenmire"))
# num_runs = 10
# num_angles = 32
# train_accuracies_stoud = zeros(num_runs, num_angles)
# test_accuracies_stoud = zeros(num_runs, num_angles)
# conf_matrices_stoud = []
# for num = 1:num_runs
#     cur_conf_mat_list = []
#     X_train, y_train, X_test, y_test, test_indices = split_data(data, labels, [10, 10], 
#     [0, 1])
#     y_train_global = y_train
#     y_test_global = y_test
#     for j = 0:num_angles-1
#         #Apply circshift to each row using a comprehension and then reconstruct the matrix
#         X_train = hcat([circshift(X_train[i, :], -16*j) for i in 1:size(X_train, 1)]...)
#         # To transpose the result back to the original orientation
#         X_train = Matrix(transpose(X_train))
#         #println(X_train)

#         X_test = hcat([circshift(X_test[i, :], -16*j) for i in 1:size(X_test, 1)]...)
#         X_test = Matrix(transpose(X_test))

#         #y_train = circshift(y_train, -j)
#         #y_test = circshift(y_test, -j)

#         # X_train_global = X_train
#         # X_test_global = X_test

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=456, chi_init=4, opts=opts)
#         summary = get_training_summary(W, train_states, test_states; print_stats=false)
#         cur_train_acc = info["train_acc"][end]
#         cur_test_acc = info["test_acc"][end]
#         cur_conf_mat = summary[:confmat]
#         train_accuracies_stoud[num, j+1] = cur_train_acc
#         test_accuracies_stoud[num, j+1] = cur_test_acc
#         push!(cur_conf_mat_list, cur_conf_mat)
#     end
#     push!(conf_matrices_stoud, cur_conf_mat_list)
# end

# train_accs = []
# test_accs = []
# for j = 0:73
#     X_train = training_data_matrix
#     X_test = testing_data_matrix
#     y_train = training_labels
#     y_test = testing_labels

#     X_train = hcat([circshift(X_train[i, :], 7*j) for i in 1:size(X_train, 1)]...)
#     X_train = Matrix(transpose(X_train))
#     X_test = hcat([circshift(X_test[i, :], 7*j) for i in 1:size(X_test, 1)]...)
#     X_test = Matrix(transpose(X_test))
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=false)
#     push!(train_accs, info["train_acc"][end])
#     push!(test_accs, info["test_acc"][end])
# end