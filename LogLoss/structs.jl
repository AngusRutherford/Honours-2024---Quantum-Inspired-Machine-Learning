using Parameters
using ITensors

import Base.convert
# type aliases
const PCache = Matrix{ITensor}
const PCacheCol = SubArray{ITensor, 1, PCache, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true} # for view mapping shenanigans
const Maybe{T} = Union{T,Nothing} 





# data structures
struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    type::String
end

const timeSeriesIterable = Vector{PState}


# Black box optimiser shell
struct BBOpt 
    name::String
    fl::String
    BBOpt(s::String, fl::String) = begin
        if !(lowercase(s) in ["optim", "optimkit", "customgd"]) 
            error("Unknown Black Box Optimiser $s, options are [CustomGD, Optim, OptimKit]")
        end
        new(s,fl)
    end
end

function BBOpt(s::String)
    sl = lowercase(s)
    if sl == "customgd"
        return BBOpt(s, "GD")
    else
        return BBOpt(s, "CGD")
    end
end

# timeseries encoding shell

struct Encoding
    name::String
    encode::Function
    Encoding(s::String, enc::Function) = begin
        if !(lowercase(s) in ["stoud", "stoudenmire", "fourier", "sahand"]) 
            error("Unknown Encoding $s, options are [\"Stoud\", \"Stoudenmire\", \"Fourier\", \"Sahand\"]")
        end
        new(s,enc)
    end
end


function Encoding(s::String)
    
    sl = lowercase(s)
    if sl == "stoud" || sl == "stoudenmire"
        enc = angle_encode
    elseif sl == "fourier"
        enc = fourier_encode
    elseif sl == "sahand"
        enc = sahand_encode
    else
        enc = identity
    end
    return Encoding(s, enc)
end

# container for options with default values

@with_kw struct Options
    nsweeps::Int
    chi_max::Int
    cutoff::Float64
    update_iters::Int
    verbosity::Int
    dtype::DataType
    lg_iter::Function
    bbopt::BBOpt
    track_cost::Bool
    eta::Float64
    rescale::Vector{Bool}
    d::Int
    encoding::Encoding
end

function Options(; nsweeps=5, chi_max=25, cutoff=1E-10, update_iters=10, verbosity=1, dtype::DataType=ComplexF64, lg_iter=KLD_iter, bbopt=BBOpt("CustomGD"),
    track_cost::Bool=(verbosity >=1), eta=0.01, rescale = [false, true], d=2, encoding=Encoding("Stoudenmire"))
    Options(nsweeps, chi_max, cutoff, update_iters, verbosity, dtype, lg_iter, bbopt, track_cost, eta, rescale, d, encoding)
end

# type conversions
# These are reasonable to implement because Encoding() and BBOpt() are just wrapper types with some validation built in
convert(::Type{Encoding}, s::String) = Encoding(s)
convert(::Type{BBOpt}, s::String) = BBOpt(s)