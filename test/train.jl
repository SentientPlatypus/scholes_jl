# train_options_td3_ou.jl
using Random, Statistics, Plots
include("../nn.jl")
include("../data.jl")
include("../option_env.jl")
include("../td3.jl")

const TICKER = "MSFT"
const LOOK_BACK_PERIOD = 30
const NUM_EPISODES     = 150
const INIT_CAPITAL     = 1000.0
const SEED             = 3

const ACTOR_LR   = 5e-5
const CRITIC_LR  = 3e-4
const WEIGHT_DEC = 1e-4
const BATCH_SIZE = 256
const GAMMA      = 0.95
const TAU        = 0.005
const SESSION_STEPS = 300

Random.seed!(SEED)

# ----- Load features -----
all_features, all_prices = get_all_features(TICKER, LOOK_BACK_PERIOD)
nIndicators = ncol(all_features)
state_feat_dim = nIndicators * LOOK_BACK_PERIOD
extra_dim = 7  # [log_cap, τ, n_call, n_put, portΔ, portΓ, portV]
input_dim = state_feat_dim + extra_dim
act_dim   = 2

# ----- Build networks -----
π_ = Net([
    Layer(input_dim, 200, relu, relu′),
    Layer(200, 100, relu, relu′),
    Layer(100, 50, relu, relu′),
    Layer(50, 30, relu, relu′),
    Layer(30, 10, relu, relu′),
    Layer(10, act_dim, my_tanh, my_tanh′)
], mse_loss, mse_loss′)

Q = Net([
    Layer(input_dim + act_dim, 200, relu, relu′),
    Layer(200, 100, relu, relu′),
    Layer(100, 50, relu, relu′),
    Layer(50, 30, relu, relu′),
    Layer(30, 10, relu, relu′),
    Layer(10, 1, idty, idty′)
], mse_loss, mse_loss′)

td3 = TD3(π_, Q, GAMMA, TAU; policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5)

# ----- OU Noise -----
mutable struct SimpleOUNoise
    θ::Float64; μ::Float64; σ::Float64
    x_prev::Vector{Float64}
end

function SimpleOUNoise(θ::Float64, μ::Float64, σ::Float64, dim::Int)
    SimpleOUNoise(θ, μ, σ, zeros(dim))
end

function sample!(n::SimpleOUNoise)
    x = n.x_prev .+ n.θ .* (n.μ .- n.x_prev) .+ n.σ .* randn(length(n.x_prev))
    n.x_prev .= x
    return x
end

ou = SimpleOUNoise(0.15, 0.0, 0.2, act_dim)

# ----- Training loop -----
episode_equity = Float64[]
bh_equity      = Float64[]

N = nrow(all_features)
min_start = LOOK_BACK_PERIOD
max_start = max(LOOK_BACK_PERIOD, N - SESSION_STEPS - 1)

for ep in 1:NUM_EPISODES
    t0 = rand(min_start:max_start)
    t_end = min(N, t0 + SESSION_STEPS)
    sess_idx = t0-LOOK_BACK_PERIOD+1:t_end

    session_df = all_features[sess_idx, :]
    session_prices = all_prices[sess_idx]


    print(session_df)
    println(session_prices)

    env = Env(session_df, session_prices; lookback=LOOK_BACK_PERIOD, start_t = t0)
    reset!(env)

    capitals = Float64[INIT_CAPITAL]
    bh_cap   = Float64[INIT_CAPITAL]
    S_ref    = 100.0

    while true
        feat_window = flatten_window(session_df, env.steps_left / 252.0, LOOK_BACK_PERIOD)
        s = state(env, feat_window)

        # Deterministic policy
        a_det = td3.π_(s)
        # OU exploration
        ε = sample!(ou)
        a = (clamp(a_det[1] + ε[1], -1.0, 1.0),
             clamp(a_det[2] + ε[2], -1.0, 1.0))

        reward, done = step!(env, a)
        s′ = done ? s : state(env, flatten_window(session_df, env.t, LOOK_BACK_PERIOD))

        add_experience!(td3, s, collect(a), reward, s′, done ? 1.0 : 0.0)
        train_td3_step!(td3, CRITIC_LR, ACTOR_LR, WEIGHT_DEC, BATCH_SIZE)

        push!(capitals, env.capital)
        if env.t <= length(session_prices)
            r_pct = session_prices[env.t] / 100.0
            S_ref *= (1 + r_pct)
            push!(bh_cap, INIT_CAPITAL * (S_ref / 100.0))
        else
            push!(bh_cap, bh_cap[end])
        end

        if done; break; end
    end

    push!(episode_equity, capitals[end])
    push!(bh_equity, bh_cap[end])

    if ep % 10 == 0 || ep == 1
        println("Episode $ep — final equity: $(round(capitals[end], digits=2)) vs BH: $(round(bh_cap[end], digits=2))")
    end
end

mkpath("plots/options")
plt = plot(episode_equity, label="TD3 + OU (terminal equity)", xlabel="Episode",
           ylabel="Capital", lw=2, title="Options TD3 Training Performance")
plot!(plt, bh_equity, label="Buy & Hold reference", lw=2, linestyle=:dash)
savefig("plots/options/td3_ou_equity.png")
println("Saved plots/options/td3_ou_equity.png")
