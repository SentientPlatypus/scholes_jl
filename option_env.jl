# options_env.jl
# Minimal options trading environment:
# - Underlier path from your data (percent changes -> S path)
# - Option chain: 1 expiry (rolling), ATM Call + ATM Put
# - Pricing: Black–Scholes with rolling realized vol (as IV proxy)
# - State: [features window; log10(capital); τ; n_call; n_put; portfolio Δ, Γ, Vega]
# - Action: a = [a_call, a_put] ∈ [-1,1]^2 (target positions, scaled by capital)
# - Reward: log-wealth; small Γ/Vega penalty; costs & spread included

using Statistics, Distributions


function bs_d1(S, K, r, q, σ, τ)
    (log(S / K) + (r - q + 0.5 * σ^2) * τ) / (σ * sqrt(τ))
end


function bs_prices(S, K, r, q, σ, τ)
    if τ <= 0 || σ <= 0 #If the time is expired, or there is no volatility (σ)
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
function greeks(S, K, r, q, σ, τ)
    if τ <= 0 || σ <= 0 #If the time is expired, or there is no volatility (σ)
        Δc = (S > K) ? 1.0 : 0.0
        Δp = Δc - 1.0
        Γ  = 0.0
        V  = 0.0
        return Δc, Δp, Γ, V
    end

    d1    = bs_d1(S, K, r, q, σ, τ)

    Δc    = e_qτ * cdf(Normal(), d1)  #δC/δS
    Δp    = Δc - e_qτ
    Γ     = e_qτ * ϕd1 / (S * σ * sqrt(τ)) #δ^2C/δS^2
    Vega  = S * e_qτ * ϕd1 * sqrt(τ) # δC/δσ
    return Δc, Δp, Γ, Vega
end

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
    T_steps::Int #Expiry time (21 days)
    steps_left::Int
    K::Float64 #Strike price
    σ::Float64 #volatility estimation
    vol_win::Int # window length for realized vol estimate.
    realized_ret::Vector{Float64} #returns for estimating σ

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
    env.σ = 0.2
    env.steps_left = env.T_steps
    empty!(env.realized_ret)
    env.capital = 1000.0
    env.n_call = 0.0
    env.n_put  = 0.0
    return
end

function update_σ!(env::Env)
    if length(env.realized_ret) < env.vol_win
        return
    end
    σp   = std(env.realized_ret[end-env.vol_win+1:end]) / 100.0
    σann = clamp(σp * sqrt(252.0), 0.05, 1.0)
    env.σ = σann
end

function flatten_window(df, t, L)
    vcat([df[!, c][t-L+1:t] for c in names(df)]...)
end

function state(env::Env, feat_window::Vector{Float64})
    τ = max(env.steps_left, 1) / 252.0
    ΔC, ΔP, Γ, V = greeks(env.S, env.K, env.r, env.q, env.σ, τ)
    portΔ = env.n_call * Δc + env.n_put * Δp
    portΓ = (env.n_call + env.n_put) * Γ
    portV = (env.n_call + env.n_put) * V
    return vcat(
        feat_window,
        [log10(env.capital)],
        [τ, env.n_call, env.n_put, portΔ, portΓ, portV]
    )
end


function step!(env::Env, a::NTuple{2, Float64})
    a_call, a_put = a

    # Scale contract counts.
    max_contr = env.alloc_scale * env.capital / max(env.contract_size * env.S, 1e-9)
    target_call = clamp(a_call * max_contr, -max_contr, max_contr) #target # of calls
    target_put  = clamp(a_put  * max_contr, -max_contr, max_contr) #target # of puts

    #pre move options prices.
    τ   = max(env.steps_left, 1) / 252.0
    C, P = bs_prices(env.S, env.K, env.r, env.q, env.sigma, τ)

    ΔC = target_call - env.n_call
    ΔP = target_put - env.n_put

    traded_premium = (abs(ΔC) * C + abs(ΔP) * P) * env.contract_size 
    cost   = env.tx_cost_rate * traded_prem
    spread = env.spread_rate  * traded_prem

    cash_flow = (ΔC*C + ΔP*P) * env.contract_size + cost + spread #total cost of this move
    env.capital -= cash_flow
    env.n_call += ΔC
    env.n_put  += ΔP

    #move to next time step.
    idx = env.t + 1
    if idx > length(env.day_change)
        #episode is over
        return 0.0, true
    end

    r_pct = env.day_change[idx]
    env.S *= (1 + r_pct / 100.0)

    push!(env.realized_ret, env.day_change[idx]) #add return to experience
    if length(env.realized_ret) > 200
        popfirst!(env.realized_ret)
    end
    update_σ!(env) #update volatility

    env.steps_left -=1
    τ′ = max(env.steps_left, 0) / 252.0
    opt_value = (env.n_call*C′ + env.n_put*P′) * env.contract_size
    prev_opt_val = (env.n_call*C  + env.n_put*P ) * env.contract_size

    # Approx prev equity = (pre-trade cash + trade cash) + prev_opt_val
    prev_equity = (env.capital + cash_flow) + prev_opt_val
    equity = env.capital + opt_value #lost the cash flow, but now have option value

    rw = log(max(equity, 1e-9) / max(prev_equity, 1e-9)) #percent reward

    # Risk penalties. Recompute greeks at new state, penalize convexity Γ and sensitivity Vega, to avoid unstable exposure
    Δc_g, Δp_g, Γ, V = bs_greeks(env.S, env.K, env.r, env.q, env.sigma, max(τ′, 1e-9))
    portΓ = (env.n_call + env.n_put) * Γ
    portV = (env.n_call + env.n_put) * V
    reward = rw - env.λΓ * abs(portΓ) - env.λV * abs(portV) #subtract from reward per weights.

    # Expiry handling: settle intrinsic, relist
    if env.steps_left <= 0
        Cexp, Pexp = bs_prices(env.S, env.K, env.r, env.q, env.sigma, 0.0)
        env.capital += (env.n_call*Cexp + env.n_put*Pexp) * env.contract_size
        env.n_call = 0.0
        env.n_put  = 0.0
        
        #roll chain
        env.steps_left = env.T_steps
        env.K = env.S
    end

    env.t += 1
    done = false
    return reward, done
end
