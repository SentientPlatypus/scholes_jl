# options_env.jl
# Minimal options trading environment:
# - Underlier path from your data (percent changes -> S path)
# - Option chain: 1 expiry (rolling), ATM Call + ATM Put
# - Pricing: Black–Scholes with rolling realized vol (as IV proxy)
# - State: [features window; log10(capital); τ; n_call; n_put; portfolio Δ, Γ, Vega]
# - Action: a = [a_call, a_put] ∈ [-1,1]^2 (target positions, scaled by capital)
# - Reward: log-wealth; small Γ/Vega penalty; costs & spread included

module OptionsEnv

using Statistics, Distributions

# -------- Black–Scholes utilities --------
const SQRT2 = sqrt(2.0)
Φ(x) = 0.5 * (1 + erf(x / SQRT2))

# d1 helper
@inline function bs_d1(S, K, r, q, σ, τ)
    (log(S / K) + (r - q + 0.5 * σ^2) * τ) / (σ * sqrt(τ))
end

# European call/put prices (per-share)
function bs_prices(S, K, r, q, σ, τ)
    if τ <= 0 || σ <= 0
        C = max(S - K, 0.0)
        P = max(K - S, 0.0)
        return C, P
    end
    d1 = bs_d1(S, K, r, q, σ, τ)
    d2 = d1 - σ * sqrt(τ)
    Nd1 = cdf(Normal(), d1)
    Nd2 = cdf(Normal(), d2)
    e_qτ = exp(-q * τ)
    e_rτ = exp(-r * τ)
    C = S * e_qτ * Nd1 - K * e_rτ * Nd2
    P = C - S * e_qτ + K * e_rτ   # via put-call parity
    return C, P
end

# Greeks (per-share): Δcall, Δput, Γ, Vega (dPrice/dVol, vol in absolute units)
function bs_greeks(S, K, r, q, σ, τ)
    if τ <= 0 || σ <= 0
        Δc = (S > K) ? 1.0 : 0.0
        Δp = Δc - 1.0
        Γ  = 0.0
        V  = 0.0
        return Δc, Δp, Γ, V
    end
    d1    = bs_d1(S, K, r, q, σ, τ)
    ϕd1   = pdf(Normal(), d1)
    e_qτ  = exp(-q * τ)
    Δc    = e_qτ * cdf(Normal(), d1)
    Δp    = Δc - e_qτ
    Γ     = e_qτ * ϕd1 / (S * σ * sqrt(τ))
    Vega  = S * e_qτ * ϕd1 * sqrt(τ)
    return Δc, Δp, Γ, Vega
end

# -------- Environment --------
mutable struct Env
    # Market / data
    day_df            # DataFrame of features (for this "day session")
    day_change::Vector{Float64}  # % changes (post lookback)
    lookback::Int
    t::Int
    S::Float64
    S0::Float64
    r::Float64
    q::Float64

    # Option chain (single expiry rolling ATM)
    T_steps::Int
    steps_left::Int
    K::Float64
    sigma::Float64
    vol_win::Int
    realized_ret::Vector{Float64}

    # Portfolio / accounting
    capital::Float64
    n_call::Float64
    n_put::Float64
    contract_size::Float64
    alloc_scale::Float64      # contracts per unit action magnitude (scaled by capital)
    tx_cost_rate::Float64     # proportional cost on traded premium notional
    spread_rate::Float64      # half-spread, as fraction of option price

    # Risk penalty weights
    λΓ::Float64
    λV::Float64
end

function Env(day_df, day_change; lookback=30, S0=100.0, r=0.0, q=0.0,
             T_steps=21, init_capital=1000.0, vol_win=20,
             tx_cost_rate=0.0002, spread_rate=0.002,
             contract_size=100.0, alloc_scale=0.10, σ0=0.2,
             λΓ=1e-4, λV=1e-7)
    return Env(day_df, day_change, lookback, lookback, S0, S0, r, q,
               T_steps, T_steps, S0, σ0, vol_win, Float64[],
               init_capital, 0.0, 0.0, contract_size, alloc_scale,
               tx_cost_rate, spread_rate, λΓ, λV)
end

function reset!(env::Env)
    env.t = env.lookback
    env.S = env.S0
    env.K = env.S0
    env.sigma = 0.2
    env.steps_left = env.T_steps
    empty!(env.realized_ret)
    env.capital = 1000.0
    env.n_call = 0.0
    env.n_put  = 0.0
    return
end

# Rolling realized vol -> annualized, as proxy for IV
function update_sigma!(env::Env)
    if length(env.realized_ret) < env.vol_win
        return
    end
    σp   = std(env.realized_ret[end-env.vol_win+1:end]) / 100.0
    σann = clamp(σp * sqrt(252.0), 0.05, 1.0)
    env.sigma = σann
end

# Relist a new ATM chain at expiry
function roll_chain!(env::Env)
    env.steps_left = env.T_steps
    env.K = env.S
end

# Flatten state vector
function flatten_window(df, t, L)
    vcat([df[!, c][t-L+1:t] for c in names(df)]...)
end

function state(env::Env, feat_window::Vector{Float64})
    τ = max(env.steps_left, 1) / 252.0
    Δc, Δp, Γ, V = bs_greeks(env.S, env.K, env.r, env.q, env.sigma, τ)
    portΔ = env.n_call * Δc + env.n_put * Δp
    portΓ = (env.n_call + env.n_put) * Γ
    portV = (env.n_call + env.n_put) * V
    return vcat(
        feat_window,
        [log10(env.capital)],
        [τ, env.n_call, env.n_put, portΔ, portΓ, portV]
    )
end

# One step with action a = (a_call, a_put) ∈ [-1,1]^2
# Returns: reward::Float64, done::Bool
function step!(env::Env, a::NTuple{2,Float64})
    a_call, a_put = a

    # Map actions -> target contract counts (scaled by capital & spot)
    max_contr = env.alloc_scale * env.capital / max(env.contract_size * env.S, 1e-9)
    target_call = clamp(a_call * max_contr, -max_contr, max_contr)
    target_put  = clamp(a_put  * max_contr, -max_contr, max_contr)

    # Current option prices (pre-move)
    τ   = max(env.steps_left, 1) / 252.0
    C, P = bs_prices(env.S, env.K, env.r, env.q, env.sigma, τ)

    # Trade deltas & costs
    Δc = target_call - env.n_call
    Δp = target_put  - env.n_put
    traded_prem = (abs(Δc)*C + abs(Δp)*P) * env.contract_size
    cost   = env.tx_cost_rate * traded_prem
    spread = env.spread_rate  * traded_prem
    cash_flow = (Δc*C + Δp*P) * env.contract_size + cost + spread
    env.capital -= cash_flow
    env.n_call += Δc
    env.n_put  += Δp

    # Evolve to next spot using your percent change series
    idx = env.t + 1
    if idx > length(env.day_change)
        # episode over
        return 0.0, true
    end
    r_pct = env.day_change[idx] / 100.0
    env.S *= (1 + r_pct)
    push!(env.realized_ret, env.day_change[idx])
    if length(env.realized_ret) > 200
        popfirst!(env.realized_ret)
    end
    update_sigma!(env)

    env.steps_left -= 1

    # Mark-to-market
    τ′ = max(env.steps_left, 0) / 252.0
    C′, P′ = bs_prices(env.S, env.K, env.r, env.q, env.sigma, τ′)
    opt_value = (env.n_call*C′ + env.n_put*P′) * env.contract_size
    prev_opt_val = (env.n_call*C  + env.n_put*P ) * env.contract_size
    # Approx prev equity = (pre-trade cash + trade cash) + prev_opt_val
    prev_equity = (env.capital + cash_flow) + prev_opt_val
    equity      = env.capital + opt_value
    rw = log(max(equity, 1e-9) / max(prev_equity, 1e-9))

    # Risk penalties
    Δc_g, Δp_g, Γ, V = bs_greeks(env.S, env.K, env.r, env.q, env.sigma, max(τ′, 1e-9))
    portΓ = (env.n_call + env.n_put) * Γ
    portV = (env.n_call + env.n_put) * V
    reward = rw - env.λΓ * abs(portΓ) - env.λV * abs(portV)

    # Expiry handling: settle intrinsic, relist
    if env.steps_left <= 0
        Cexp, Pexp = bs_prices(env.S, env.K, env.r, env.q, env.sigma, 0.0)
        env.capital += (env.n_call*Cexp + env.n_put*Pexp) * env.contract_size
        env.n_call = 0.0
        env.n_put  = 0.0
        roll_chain!(env)
    end

    env.t += 1
    done = false
    return reward, done
end

export Env, reset!, state, step!, flatten_window

end # module
