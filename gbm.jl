using Random
using Statistics

# Function to calculate percent returns
function returns(raw::Vector{Float64})
    return [(raw[i] - raw[i-1]) / raw[i-1] for i in 2:length(raw)]
end


function vscore(raw::Vector{Float64}, OBS::Int=60, EPOCH::Int=1000, EXT::Int=20)
    v = Float64[]  # Result vector

    for t in OBS:length(raw)-1
        temp = raw[t+1-OBS : t+1]
        ret = returns(temp)
        s0 = temp[end]
        μ, σ = mean(ret), std(ret)
        drift = μ + 0.5 * σ^2

        # Simulate paths using broadcasting
        noise = cumsum(randn(EPOCH, EXT), dims=2)  # cumsum first
        paths = s0 .* exp.(σ .* noise .+ drift .* (1:EXT)')  # Then scale and apply drift
        
        sum_exceed = count(>(s0), paths)
        push!(v, sum_exceed / (EPOCH * EXT))
    end

    return (v .- mean(v)) ./ std(v)
end