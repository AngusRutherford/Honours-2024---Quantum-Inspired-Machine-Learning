include("RealRealHighDimension.jl")
using DelimitedFiles

dataset_dir = "LogLoss/datasets"
(X_train, y_train), (X_test, y_test) = load_splits_txt(joinpath(dataset_dir, "SwedishLeaf_TRAIN.txt"),
                                                       joinpath(dataset_dir, "SwedishLeaf_TEST.txt"),
                                                       joinpath(dataset_dir, "SwedishLeaf_TEST.txt"))

# (X_train, y_train), (X_test, y_test) = load_splits_txt(joinpath(dataset_dir, "MedicalImages_TRAIN.txt"),
# joinpath(dataset_dir, "MedicalImages_TEST.txt"),
# joinpath(dataset_dir, "MedicalImages_TEST.txt"))

verbosity = 0
test_run = false
track_cost = false
encoding = legendre()
encode_classes_separately = false
train_classes_separately = false
dtype = encoding.iscomplex ? ComplexF64 : Float64


function downsample(matrix::Array{Float64, 2}, num_points::Int)
    n, m = size(matrix)
    indices = round.(Int, range(1, stop=m, length=num_points))
    return matrix[:, indices], indices
end

m = length(X_train[1, :])
X_train_1_4, b1 = downsample(X_train, ceil(Int, m * 1/4))
X_test_1_4, b2 = downsample(X_test, ceil(Int, m * 1/4))

X_train_2_4, b3 = downsample(X_train, ceil(Int, m * 2/4))
X_test_2_4, b4 = downsample(X_test, ceil(Int, m * 2/4))

X_train_3_4, b5 = downsample(X_train, ceil(Int, m * 3/4))
X_test_3_4, b6 = downsample(X_test, ceil(Int, m * 3/4))

N = 220
seeds = 201:N
train_accs_OBC_1_4 = zeros(20, 12) #22 as 20 sweeps, plus acc before first sweep, plus acc after normalisation
test_accs_OBC_1_4 = zeros(20, 12)
train_accs_PBC_left_1_4 = zeros(20, 22)
test_accs_PBC_left_1_4 = zeros(20, 22)
train_accs_PBC_right_1_4 = zeros(20, 22)
test_accs_PBC_right_1_4 = zeros(20, 22)
train_accs_PBC_both_1_4 = zeros(20, 22)
test_accs_PBC_both_1_4 = zeros(20, 22)
train_accs_PBC_random_1_4 = zeros(20, 22)
test_accs_PBC_random_1_4 = zeros(20, 22)
for seed in seeds
    # OBC
    opts=Options(; nsweeps=10, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_1_4, y_train, X_test_1_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_OBC_1_4[seed-200, :] = info["train_acc"]
    test_accs_OBC_1_4[seed-200, :] = info["test_acc"]

    # PBC left
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_1_4, y_train, X_test_1_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_left_1_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_left_1_4[seed-200, :] = info["test_acc"]

    # PBC right
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_right", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_1_4, y_train, X_test_1_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_right_1_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_right_1_4[seed-200, :] = info["test_acc"]

    # PBC both
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_both", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_1_4, y_train, X_test_1_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_both_1_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_both_1_4[seed-200, :] = info["test_acc"]

    # PBC random
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_1_4, y_train, X_test_1_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_random_1_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_random_1_4[seed-200, :] = info["test_acc"]
end

train_accs_OBC_2_4 = zeros(20, 12) #22 as 20 sweeps, plus acc before first sweep, plus acc after normalisation
test_accs_OBC_2_4 = zeros(20, 12)
train_accs_PBC_left_2_4 = zeros(20, 22)
test_accs_PBC_left_2_4 = zeros(20, 22)
train_accs_PBC_right_2_4 = zeros(20, 22)
test_accs_PBC_right_2_4 = zeros(20, 22)
train_accs_PBC_both_2_4 = zeros(20, 22)
test_accs_PBC_both_2_4 = zeros(20, 22)
train_accs_PBC_random_2_4 = zeros(20, 22)
test_accs_PBC_random_2_4 = zeros(20, 22)
for seed in seeds
    # OBC
    opts=Options(; nsweeps=10, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_2_4, y_train, X_test_2_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_OBC_2_4[seed-200, :] = info["train_acc"]
    test_accs_OBC_2_4[seed-200, :] = info["test_acc"]

    # PBC left
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_2_4, y_train, X_test_2_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_left_2_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_left_2_4[seed-200, :] = info["test_acc"]

    # PBC right
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_right", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_2_4, y_train, X_test_2_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_right_2_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_right_2_4[seed-200, :] = info["test_acc"]

    # PBC both
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_both", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_2_4, y_train, X_test_2_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_both_2_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_both_2_4[seed-200, :] = info["test_acc"]

    # PBC random
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_2_4, y_train, X_test_2_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_random_2_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_random_2_4[seed-200, :] = info["test_acc"]
end

train_accs_OBC_3_4 = zeros(20, 12) #22 as 20 sweeps, plus acc before first sweep, plus acc after normalisation
test_accs_OBC_3_4 = zeros(20, 12)
train_accs_PBC_left_3_4 = zeros(20, 22)
test_accs_PBC_left_3_4 = zeros(20, 22)
train_accs_PBC_right_3_4 = zeros(20, 22)
test_accs_PBC_right_3_4 = zeros(20, 22)
train_accs_PBC_both_3_4 = zeros(20, 22)
test_accs_PBC_both_3_4 = zeros(20, 22)
train_accs_PBC_random_3_4 = zeros(20, 22)
test_accs_PBC_random_3_4 = zeros(20, 22)
for seed in seeds
    # OBC
    opts=Options(; nsweeps=10, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_3_4, y_train, X_test_3_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_OBC_3_4[seed-200, :] = info["train_acc"]
    test_accs_OBC_3_4[seed-200, :] = info["test_acc"]

    # PBC left
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_3_4, y_train, X_test_3_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_left_3_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_left_3_4[seed-200, :] = info["test_acc"]

    # PBC right
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_right", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_3_4, y_train, X_test_3_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_right_3_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_right_3_4[seed-200, :] = info["test_acc"]

    # PBC both
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_both", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_3_4, y_train, X_test_3_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_both_3_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_both_3_4[seed-200, :] = info["test_acc"]

    # PBC random
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train_3_4, y_train, X_test_3_4, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_random_3_4[seed-200, :] = info["train_acc"]
    test_accs_PBC_random_3_4[seed-200, :] = info["test_acc"]
end

writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_train_OBC.csv", train_accs_OBC_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_test_OBC.csv", test_accs_OBC_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_train_PBC_left.csv", train_accs_PBC_left_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_test_PBC_left.csv", test_accs_PBC_left_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_train_PBC_right.csv", train_accs_PBC_right_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_test_PBC_right.csv", test_accs_PBC_right_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_train_PBC_both.csv", train_accs_PBC_both_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_test_PBC_both.csv", test_accs_PBC_both_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_train_PBC_ranom.csv", train_accs_PBC_random_1_4, ',')
writedlm("angus_swedishleaf_1_4_leg_chi16_eta02_sweeps20_test_PBC_random.csv", test_accs_PBC_random_1_4, ',')

writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_train_OBC.csv", train_accs_OBC_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_test_OBC.csv", test_accs_OBC_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_train_PBC_left.csv", train_accs_PBC_left_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_test_PBC_left.csv", test_accs_PBC_left_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_train_PBC_right.csv", train_accs_PBC_right_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_test_PBC_right.csv", test_accs_PBC_right_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_train_PBC_both.csv", train_accs_PBC_both_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_test_PBC_both.csv", test_accs_PBC_both_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_train_PBC_ranom.csv", train_accs_PBC_random_2_4, ',')
writedlm("angus_swedishleaf_2_4_leg_chi16_eta02_sweeps20_test_PBC_random.csv", test_accs_PBC_random_2_4, ',')

writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_train_OBC.csv", train_accs_OBC_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_test_OBC.csv", test_accs_OBC_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_train_PBC_left.csv", train_accs_PBC_left_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_test_PBC_left.csv", test_accs_PBC_left_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_train_PBC_right.csv", train_accs_PBC_right_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_test_PBC_right.csv", test_accs_PBC_right_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_train_PBC_both.csv", train_accs_PBC_both_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_test_PBC_both.csv", test_accs_PBC_both_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_train_PBC_ranom.csv", train_accs_PBC_random_3_4, ',')
writedlm("angus_swedishleaf_3_4_leg_chi16_eta02_sweeps20_test_PBC_random.csv", test_accs_PBC_random_3_4, ',')

