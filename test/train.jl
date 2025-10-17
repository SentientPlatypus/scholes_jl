# train_options_td3_ou.jl
using Random, Statistics, Plots, Dates
include("../nn.jl")
include("../data.jl")
include("../option_env.jl")
include("../td3.jl")

const TICKER = "MSFT"
const LOOK_BACK_PERIOD = 90
const NUM_EPISODES     = 1500
const INIT_CAPITAL     = 1000.0
const SEED             = 10

const ACTOR_LR   = 5e-5
const CRITIC_LR  = 3e-4
const WEIGHT_DEC = 1e-4
const BATCH_SIZE = 256
const GAMMA      = 0.97
const TAU        = 0.005
const SESSION_STEPS = 21
const WARMUP_STEPS = 50000

Random.seed!(SEED)


date_str = Dates.format(now(), "yyyy-mm-dd")
save_dir = "plots/options/$(date_str)"

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


ou = SimpleOUNoise(0.3, 0.0, 0.05, act_dim)

# ----- Training loop -----
episode_equity = Float64[]
bh_equity      = Float64[]

episode_total_reward = Float64[]
episode_avg_reward   = Float64[]
last_episode_rewards = Float64[]  # per-step rewards for the most recent episode

rolling_mean(v::Vector{Float64}, w::Int) = (length(v) < w ? [mean(v[1:i]) for i in 1:length(v)]
                                            : vcat([mean(v[1:i]) for i in 1:w-1],
                                                   [mean(@view v[i-w+1:i]) for i in w:length(v)]))

N = nrow(all_features)
min_start = LOOK_BACK_PERIOD
max_start = max(LOOK_BACK_PERIOD, N - SESSION_STEPS - 1)

warmup_replay!(td3, all_features, all_prices;
               lookback=LOOK_BACK_PERIOD,
               session_steps=SESSION_STEPS,
               warmup_steps=WARMUP_STEPS,
               seed=SEED)

@info "Warmup done. Starting training..."
for ep in 1:NUM_EPISODES

    t0 = rand(min_start:max_start)
    t_end = min(N, t0 + SESSION_STEPS)

    @info "Episode $ep: session from index $t0 to $t_end (length=$(t_end - t0))"
    sess_idx = t0-LOOK_BACK_PERIOD+1:t_end

    session_df = all_features[sess_idx, :]
    session_prices = all_prices[sess_idx]



    env = Env(session_df, session_prices; lookback=LOOK_BACK_PERIOD)
    reset!(env)
    env.λΓ = (ep <= 500) ? 0.0 : 1e-4
    env.λV = (ep <= 500) ? 0.0 : 1e-7


    capitals = Float64[INIT_CAPITAL]
    bh_cap   = Float64[INIT_CAPITAL]
    S_ref    = 100.0

    empty!(last_episode_rewards)
    while true
        feat_window = flatten_window(session_df, env.t, LOOK_BACK_PERIOD)
        s = state(env, feat_window)

        # Deterministic policy
        a_det = td3.π_(s)
        # println(td3.π_.output)
        # OU exploration
        ε = sample!(ou)
        ou.σ = max(0.03, ou.σ * 0.999)   # slower decay
        a = (clamp(a_det[1] + ε[1], -1.0, 1.0),
             clamp(a_det[2] + ε[2], -1.0, 1.0))

        reward, done = step!(env, a)
        push!(last_episode_rewards, reward)

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
    push!(episode_total_reward, sum(last_episode_rewards))
    push!(episode_avg_reward,   mean(last_episode_rewards))

    if ep % 10 == 0 || ep == 1
        println("Episode $ep — final equity: $(round(capitals[end], digits=2)) vs BH: $(round(bh_cap[end], digits=2))")
    end
end

mkpath(save_dir)

#-------PLOTS----------------

#1. EQUITY
plt = plot(episode_equity, label="TD3 + OU (terminal equity)", xlabel="Episode",
           ylabel="Capital", lw=2, title="Options TD3 Training Performance")
plot!(plt, bh_equity, label="Buy & Hold reference", lw=2, linestyle=:dash)
savefig("$(save_dir)/td3_ou_equity.png")

# 1) Total reward per episode + rolling mean
w = 20  # smoothing window (episodes)
r_tot = episode_total_reward
r_tot_smooth = rolling_mean(r_tot, w)

plt_rtot = plot(r_tot, label="Total reward / episode", lw=1.5,
                xlabel="Episode", ylabel="Reward",
                title="TD3+OU — Total Reward per Episode")
plot!(plt_rtot, r_tot_smooth, label="Rolling mean (w=$(w))", lw=3, linestyle=:dash)
savefig("$(save_dir)/td3_rewards_total.png")

# 2) Average step reward per episode + rolling mean
r_avg = episode_avg_reward
r_avg_smooth = rolling_mean(r_avg, w)

plt_ravg = plot(r_avg, label="Avg step reward / episode", lw=1.5,
                xlabel="Episode", ylabel="Reward",
                title="TD3+OU — Average Step Reward per Episode")
plot!(plt_ravg, r_avg_smooth, label="Rolling mean (w=$(w))", lw=3, linestyle=:dash)
savefig("$(save_dir)/td3_rewards_avg.png")

# 3) Per-step rewards of the last episode
plt_rsteps = plot(last_episode_rewards, label="Reward (per step)", lw=1.5,
                  xlabel="Step", ylabel="Reward",
                  title="TD3+OU — Per-Step Rewards (Last Episode)")
savefig("$(save_dir)/td3_rewards_last_episode.png")

println("Saved reward plots:\n",
        "  $(save_dir)/td3_rewards_total.png\n",
        "  $(save_dir)/td3_rewards_avg.png\n",
        "  $(save_dir)/td3_rewards_last_episode.png")
