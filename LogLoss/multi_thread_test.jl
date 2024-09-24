Threads.nthreads()

using Base.Threads

# Shared array to store the results
results = Vector{Int}(undef, 8)

# Multithreaded loop: Each iteration sleeps for 10 seconds and appends the index to the array
@time Threads.@threads for i in 1:8
    sleep(10)  # Pause for 10 seconds
    results[i] = i  # Store the index in the results array
end

# Print the results
println("Results: ", results)
