include("RealRealHighDimension.jl")
using DelimitedFiles

dataset_dir = "LogLoss/datasets"
(X_train, y_train), (X_test, y_test) = load_splits_txt(joinpath(dataset_dir, "SwedishLeaf_TRAIN.txt"),
                                                       joinpath(dataset_dir, "SwedishLeaf_TEST.txt"),
                                                       joinpath(dataset_dir, "SwedishLeaf_TEST.txt"))

verbosity = 0
test_run = false
track_cost = false
encoding = legendre()
encode_classes_separately = false
train_classes_separately = false
dtype = encoding.iscomplex ? ComplexF64 : Float64

N = 220
seeds = 201:N
train_accs_OBC = zeros(20, 12) #22 as 20 sweeps, plus acc before first sweep, plus acc after normalisation
test_accs_OBC = zeros(20, 12)
train_accs_PBC_left = zeros(20, 22)
test_accs_PBC_left = zeros(20, 22)
train_accs_PBC_right = zeros(20, 22)
test_accs_PBC_right = zeros(20, 22)
train_accs_PBC_both = zeros(20, 22)
test_accs_PBC_both = zeros(20, 22)
train_accs_PBC_random = zeros(20, 22)
test_accs_PBC_random = zeros(20, 22)
for seed in seeds
    # OBC
    opts=Options(; nsweeps=10, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_OBC[seed-200, :] = info["train_acc"]
    test_accs_OBC[seed-200, :] = info["test_acc"]

    # PBC left
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_left[seed-200, :] = info["train_acc"]
    test_accs_PBC_left[seed-200, :] = info["test_acc"]

    # PBC right
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_right", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_right[seed-200, :] = info["train_acc"]
    test_accs_PBC_right[seed-200, :] = info["test_acc"]

    # PBC both
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_both", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_both[seed-200, :] = info["train_acc"]
    test_accs_PBC_both[seed-200, :] = info["test_acc"]

    # PBC random
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_random[seed-200, :] = info["train_acc"]
    test_accs_PBC_random[seed-200, :] = info["test_acc"]
end

writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_train_OBC_test2.csv", train_accs_OBC, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_test_OBC_test2.csv", test_accs_OBC, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_train_PBC_left_test2.csv", train_accs_PBC_left, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_test_PBC_left_test2.csv", test_accs_PBC_left, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_train_PBC_right_test2.csv", train_accs_PBC_right, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_test_PBC_right_test2.csv", test_accs_PBC_right, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_train_PBC_both_test2.csv", train_accs_PBC_both, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_test_PBC_both_test2.csv", test_accs_PBC_both, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_train_PBC_ranom_test2.csv", train_accs_PBC_random, ',')
writedlm("angus_swedishleaf_leg_chi16_eta02_sweeps20_test_PBC_random_test2.csv", test_accs_PBC_random, ',')