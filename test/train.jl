# train_options_td3.jl
using Random, Statistics, Plots
include("../nn.jl")
include("../data.jl")
include("../env.jl")
include("../td3.jl")

const TICKER = "MSFT"
const LOOK_BACK_PERIOD = 30
const DAYS_TO_SAMPLE   = 30
const NUM_EPISODES     = 150
const INIT_CAPITAL     = 1000.0
const SEED             = 3

const ACTOR_LR   = 5e-5
const CRITIC_LR  = 3e-4
const WEIGHT_DEC = 1e-4
const BATCH_SIZE = 256

const GAMMA = 0.95
const TAU   = 0.005

Random.seed!(SEED)

# ----- Load features -----
month_features, month_prices = get_month_features(TICKER, DAYS_TO_SAMPLE, LOOK_BACK_PERIOD)
@assert !isempty(month_features) "No features found—check data.jl inputs."

nIndicators = ncol(month_features[1])
base_feat_dim = nIndicators * LOOK_BACK_PERIOD
extra_dim = 1 + 1 + 2 + 3   # [log_cap, τ, n_call, n_put, portΔ, portΓ, portV]
input_dim = base_feat_dim + extra_dim
act_dim   = 2

# ----- Build actor/critics (your Net API) -----
actor = Net([
    Layer(input_dim, 200, relu,      relu′),
    Layer(200,       100,  relu,      relu′),
    Layer(100,       50,   relu,      relu′),
    Layer(50,        30,   relu,      relu′),
    Layer(30,        10,   relu,      relu′),
    Layer(10,        act_dim, my_tanh, my_tanh′)  # outputs in [-1,1]^2
], mse_loss, mse_loss′)

critic_template = Net([
    Layer(input_dim + act_dim, 200, relu, relu′),
    Layer(200,                  100, relu, relu′),
    Layer(100,                  50,  relu, relu′),
    Layer(50,                   30,  relu, relu′),
    Layer(30,                   10,  relu, relu′),
    Layer(10,                   1,   idty, idty′)
], mse_loss, mse_loss′)

# ----- TD3 agent (vector actions) -----
using .TD3Vector
td3 = TD3Vector.TD3(actor, critic_template, GAMMA, TAU; policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5)

# Simple OU-like scalar noise we’ll reuse per-dim
mutable struct SimpleOUNoise
    θ::Float64; μ::Float64; σ::Float64
    x_prev::Float64
end
function sample!(n::SimpleOUNoise)
    x = n.x_prev + n.θ*(n.μ - n.x_prev) + n.σ*randn()
    n.x_prev = x; x
end
ou = SimpleOUNoise(0.15, 0.0, 0.2, 0.0)

# ----- Training loop -----
episode_equity = Float64[]
bh_equity      = Float64[]

for ep in 1:NUM_EPISODES
    day = rand(1:length(month_prices))
    day_df, day_change = month_features[day], month_prices[day]

    env = OptionsEnv.Env(day_df, day_change; lookback=LOOK_BACK_PERIOD)
    OptionsEnv.reset!(env)

    capitals = Float64[INIT_CAPITAL]
    bh_cap   = Float64[INIT_CAPITAL]
    S_ref    = env.S0

    for t in LOOK_BACK_PERIOD:(nrow(day_df)-1)
        feat_window = OptionsEnv.flatten_window(day_df, t, LOOK_BACK_PERIOD)
        s = OptionsEnv.state(env, feat_window)

        # Actor + OU noise per-dimension
        a_det = td3.π_(s)              # Vector length 2 in [-1,1]
        ε = sample!(ou); ou.σ = max(0.05, ou.σ * exp(-0.00005))
        a = ( clamp(a_det[1] + ε, -1.0, 1.0),
              clamp(a_det[2] + ε, -1.0, 1.0) )

        reward, done = OptionsEnv.step!(env, a)

        # Next state (if needed for storage symmetry)
        s′ = (t < nrow(day_df)-1) ? OptionsEnv.state(env, OptionsEnv.flatten_window(day_df, t+1, LOOK_BACK_PERIOD)) : s

        # Store & train
        TD3Vector.add_experience!(td3, s, collect(a_det), reward, s′, done ? 1.0 : 0.0)
        TD3Vector.train_td3_step!(td3, CRITIC_LR, ACTOR_LR, WEIGHT_DEC, BATCH_SIZE)

        push!(capitals, env.capital)  # env.capital already accounts for cash; options MTM is implicit in reward path

        # Reference buy & hold (underlier only, for context)
        r = day_change[t+1]/100.0
        S_ref *= (1 + r)
        push!(bh_cap, INIT_CAPITAL * (S_ref/100.0))

        if done; break; end
    end

    push!(episode_equity, capitals[end])
    push!(bh_equity, bh_cap[end])

    if ep % 10 == 0 || ep == 1
        println("Episode $ep: equity=$(round(capitals[end], digits=2))  BHRef=$(round(bh_cap[end], digits=2))")
    end
end

# ----- Plots -----
mkpath("plots/options")
plt = plot(episode_equity, label="TD3 Options (terminal equity)", xlabel="Episode", ylabel="Capital", lw=2,
           title="Options TD3 — Terminal Equity per Episode")
plot!(plt, bh_equity, label="Buy&Hold (underlier ref per session)", lw=2, linestyle=:dash)
savefig("plots/options/td3_terminal_equity.png")
println("Saved plots/options/td3_terminal_equity.png")
