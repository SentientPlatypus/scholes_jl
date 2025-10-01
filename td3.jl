# td3_vector.jl
# TD3 with vector actions, compatible with your nn.jl:
# - Net returns Vector{Float64}, length 1 for critics, length act_dim for actor
# - All targets/ops stay vector-shaped (elementwise), like your code style

module TD3Vector

using ..OptionsEnv  # optional if you put it under same project; remove if not needed

# Bring Net/Layer/losses from your nn.jl into scope when you include this file.

mutable struct TD3
    π_       :: Net
    π_target :: Net

    Q1_       :: Net
    Q2_       :: Net
    Q1_target :: Net
    Q2_target :: Net

    replay_buffer :: Vector{Tuple{Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}, Float64}}
    γ   :: Float64
    τ   :: Float64

    policy_delay      :: Int
    target_noise_std  :: Float64
    target_noise_clip :: Float64

    step_count :: Int
end

function TD3(π::Net, Q̂::Net, γ::Float64, τ::Float64;
             policy_delay::Int=2, target_noise_std::Float64=0.2, target_noise_clip::Float64=0.5)
    Q1 = deepcopy(Q̂); Q2 = deepcopy(Q̂)
    return TD3(π, deepcopy(π), Q1, Q2, deepcopy(Q1), deepcopy(Q2),
               Tuple{Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}, Float64}[],
               γ, τ, policy_delay, target_noise_std, target_noise_clip, 0)
end

function update_target_network!(target_net::Net, main_net::Net, τ::Float64)
    for i in eachindex(main_net.layers)
        target_layer = target_net.layers[i]
        main_layer   = main_net.layers[i]
        target_layer.w .= (1 - τ) * target_layer.w .+ τ * main_layer.w
        target_layer.b .= (1 - τ) * target_layer.b .+ τ * main_layer.b
    end
end

# Add experience: a is Vector{Float64} (act_dim)
@inline function add_experience!(td3::TD3, s::Vector{Float64}, a::Vector{Float64},
                                 r::Float64, s′::Vector{Float64}, d::Float64)
    push!(td3.replay_buffer, (s, a, r, s′, d))
    if length(td3.replay_buffer) > 10_000
        popfirst!(td3.replay_buffer)
    end
end

# ∂Q1/∂a for vector action: returns Vector{Float64} of size act_dim
function dQ1_da(td3::TD3, s::Vector{Float64}, a::Vector{Float64})
    x = vcat(s, a)
    oldL′ = td3.Q1_.L′
    td3.Q1_.L′ = (ŷ, y) -> ones(eltype(ŷ), size(ŷ))
    g_in = step!(td3.Q1_, x, [0.0], 0.0, 0.0, 1.0, false)
    td3.Q1_.L′ = oldL′
    a_dim = td3.π_.output.out_features
    return g_in[end - a_dim + 1:end]
end

# One TD3 training step
function train_td3_step!(td3::TD3, α_Q::Float64, α_π::Float64, λ::Float64, batch_size::Int)
    if length(td3.replay_buffer) < batch_size; return; end
    td3.step_count += 1
    mb = [td3.replay_buffer[rand(1:end)] for _ in 1:batch_size]

    # ---- Critic updates ----
    for (s, a, r, s′, d) in mb
        # Target policy smoothing on a′ (vector)
        a′ = td3.π_target(s′)
        # add clipped Gaussian noise per-dimension
        a′ .= a′ .+ clamp.(td3.target_noise_std .* randn(length(a′)), .-td3.target_noise_clip, td3.target_noise_clip)
        a′ .= clamp.(a′, -1.0, 1.0)  # action space bounds for target

        Q1′ = td3.Q1_target(vcat(s′, a′))
        Q2′ = td3.Q2_target(vcat(s′, a′))
        Qmin′ = min.(Q1′, Q2′)   # elementwise (length-1 vectors in your Net)

        y = r .+ td3.γ .* (1 .- d) .* Qmin′

        step!(td3.Q1_, vcat(s, a), y, α_Q, λ, 1/length(mb))
        step!(td3.Q2_, vcat(s, a), y, α_Q, λ, 1/length(mb))
    end

    # ---- Delayed actor + target updates ----
    if td3.step_count % td3.policy_delay == 0
        for (s, _, _, _, _) in mb
            aπ = td3.π_(s)
            g  = dQ1_da(td3, s, aπ)
            back_custom!(td3.π_, s, -g, α_π, λ, 1/length(mb))
        end
        update_target_network!(td3.π_target,  td3.π_,  td3.τ)
        update_target_network!(td3.Q1_target, td3.Q1_, td3.τ)
        update_target_network!(td3.Q2_target, td3.Q2_, td3.τ)
    end
end

export TD3, add_experience!, train_td3_step!, update_target_network!

end # module
