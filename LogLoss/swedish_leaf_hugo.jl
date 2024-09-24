include("RealRealHighDimension.jl")
using Base.Threads

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

N = 10
seeds = 1:N
train_accs_OBC = zeros(N, 22) #22 as 20 sweeps, plus acc before first sweep, plus acc after normalisation
test_accs_OBC = zeros(N, 22)
train_accs_PBC_left = zeros(N, 22)
test_accs_PBC_left = zeros(N, 22)
train_accs_PBC_right = zeros(N, 22)
test_accs_PBC_right = zeros(N, 22)
train_accs_PBC_both = zeros(N, 22)
test_accs_PBC_both = zeros(N, 22)
train_accs_PBC_random = zeros(N, 22)
test_accs_PBC_random = zeros(N, 22)
for seed = seeds
    # OBC
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "OBC", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_OBC[seed, :] = info["train_acc"]
    test_accs_OBC[seed, :] = info["test_acc"]

    # PBC left
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_left[seed, :] = info["train_acc"]
    test_accs_PBC_left[seed, :] = info["test_acc"]

    # PBC right
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_right", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_right[seed, :] = info["train_acc"]
    test_accs_PBC_right[seed, :] = info["test_acc"]

    # PBC both
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_both", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_both[seed, :] = info["train_acc"]
    test_accs_PBC_both[seed, :] = info["test_acc"]

    # PBC random
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_random[seed, :] = info["train_acc"]
    test_accs_PBC_random[seed, :] = info["test_acc"]
end
