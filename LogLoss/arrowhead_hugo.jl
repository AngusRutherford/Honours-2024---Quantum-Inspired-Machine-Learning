include("RealRealHighDimension.jl")

# might have to load it in differently
(X_train, y_train), (X_test, y_test) = load_splits_txt("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TRAIN.txt", 
"/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TEST.txt", "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/SwedishLeaf_TEST.txt")

verbosity = 0
test_run = false
track_cost = false
encoding = legendre()
encode_classes_separately = false
train_classes_separately = false
dtype = encoding.iscomplex ? ComplexF64 : Float64

function random_circshift(matrix::Matrix{Float64}, rng::MersenneTwister)
    N = size(matrix, 2)  # Number of columns (data points per sample)
    M = size(matrix, 1)  # Number of rows (datasets)
    
    # Loop over each row and apply a random circular shift using the provided rng
    for i in 1:M
        shift_amount = rand(rng, 1:N)  # Generate a random shift amount with RNG
        matrix[i, :] = circshift(matrix[i, :], shift_amount)  # Apply circular shift
    end
    
    return matrix  # Return the matrix with shifted rows
end


shift_seed = MersenneTwister(12345)
X_train = random_circshift(X_train, shift_seed) # randomly assign phase to each image
X_test = random_circshift(X_test, shift_seed)
total_seeds = N
seeds = 1:N
train_accs_OBC = zeros(N, 22) #22 as 20 sweeps, plus acc before first sweep, plus acc after normalisation
test_accs_OBC = zeros(N, 22)
train_accs_PBC_left = zeros(N, 22)
test_accs_PBC_left = zeros(N, 22)
train_accs_PBC_right = zeros(N, 22)
test_accs_PBC_right = zeros(N, 22)
train_accs_PBC_both = zeros(N, 22)
test_accs_PBC_both = zeros(N, 22)
train_accs_PBC_both_two = zeros(N, 22)
test_accs_PBC_both_two = zeros(N, 22)
train_accs_PBC_random = zeros(N, 22)
test_accs_PBC_random = zeros(N, 22)
Threads.@threads for seed = seeds
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

    # PBC both two
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_both_two", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_both_two[seed, :] = info["train_acc"]
    test_accs_PBC_both_two[seed, :] = info["test_acc"]

    # PBC random
    opts=Options(; nsweeps=20, chi_max=16,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.2, rescale = (false, true), d=4, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 100)

    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
    train_accs_PBC_random[seed, :] = info["train_acc"]
    test_accs_PBC_random[seed, :] = info["test_acc"]

end
