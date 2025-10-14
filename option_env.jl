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
    e_qτ = exp(-q * τ)
    Δc    = e_qτ * cdf(Normal(), d1)  #δC/δS
    Δp    = Δc - e_qτ

    ϕd1   = pdf(Normal(), d1)

    Γ     = e_qτ * ϕd1 / (S * σ * sqrt(τ)) #δ^2C/δS^2
    Vega  = S * e_qτ * ϕd1 * sqrt(τ) # δC/δσ
    return Δc, Δp, Γ, Vega
end

mutable struct Env
    # Market / data
    features            # DataFrame of features
    price_changes::Vector{Float64}  # % changes (post lookback)
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

function Env(features, price_changes; lookback=30, start_t=30, S0=100.0, r=0.0, q=0.0,
             T_steps=21, init_capital=1000.0, vol_win=20,
             tx_cost_rate=2e-4, spread_rate=2e-3,
             contract_size=100.0, alloc_scale=0.02, σ0=0.2,
             λΓ=1e-4, λV=1e-7)
    return Env(features, price_changes, lookback, start_t, S0, S0, r, q,
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
    portΔ = env.n_call * ΔC + env.n_put * ΔP
    portΓ = (env.n_call + env.n_put) * Γ
    portV = (env.n_call + env.n_put) * V
    return vcat(
        feat_window,
        [log10(env.capital)],
        [τ, env.n_call, env.n_put, portΔ, portΓ, portV]
    )
end

function warmup_replay!(td3, all_features, all_prices;
                        lookback::Int=30, session_steps::Int=300,
                        warmup_steps::Int=50_000, seed=nothing)

    seed === nothing || Random.seed!(seed)

    N = nrow(all_features)
    min_start = lookback
    max_start = max(lookback, N - session_steps - 1)

    # Helper to pick a fresh session + env
    pick_session = function ()
        t0 = rand(min_start:max_start)
        t_end = min(N, t0 + session_steps)
        sess_idx = (t0 - lookback + 1):t_end
        session_df = all_features[sess_idx, :]
        session_prices = all_prices[sess_idx]
        env = Env(session_df, session_prices; lookback=lookback, start_t=t0)
        reset!(env)
        return env, session_df, session_prices
    end

    env, session_df, session_prices = pick_session()

    @info "Starting warmup with $(warmup_steps) random steps…"
    for _ in 1:warmup_steps
        s  = state(env, flatten_window(session_df, env.t, lookback))
        a  = (2rand() - 1, 2rand() - 1)  # uniform in [-1,1]^2
        r, done = step!(env, a)
        s′ = done ? s : state(env, flatten_window(session_df, env.t, lookback))

        add_experience!(td3, s, collect(a), r, s′, done ? 1.0 : 0.0)

        if done
            env, session_df, session_prices = pick_session()
        end
    end

    return nothing
end

function step!(env::Env, a::NTuple{2, Float64})
    a_call, a_put = a


    # Pre-move prices:
    τ   = max(env.steps_left, 1) / 252.0
    C,P = bs_prices(env.S, env.K, env.r, env.q, env.σ, τ)

    # Premium floor avoids division blow-ups:
    prem      = max((C + P) / 2, 1e-2)            # <- tune floor
    budget    = env.alloc_scale * env.capital      # e.g. 0.05 * capital
    max_by_cash  = budget / (env.contract_size * prem)

    # Global absolute cap on contracts (per leg):
    pos_cap_abs = 2.0                              # tune

    # Final per-leg cap is the min of both:
    pos_cap = min(max_by_cash, pos_cap_abs)

    # Squash actions to [-pos_cap, pos_cap]
    target_call = pos_cap * tanh(2.0 * a_call)     # smooth & bounded
    target_put  = pos_cap * tanh(2.0 * a_put)

    ΔC = target_call - env.n_call
    ΔP = target_put  - env.n_put
    # println("DELTA C, P = $(ΔC) $(ΔP)")

    # throttle per-step change
    max_delta = 1.0
    ΔC = clamp(ΔC, -max_delta, max_delta)
    ΔP = clamp(ΔP, -max_delta, max_delta)

    traded_prem = (abs(ΔC)*C + abs(ΔP)*P) * env.contract_size
    if !isfinite(traded_prem); traded_prem = 0.0; end

    cost   = env.tx_cost_rate * traded_prem
    spread = env.spread_rate  * traded_prem
    cash_flow = (ΔC*C + ΔP*P) * env.contract_size + cost + spread

    # If cash would go negative, **scale** the trade down proportionally
    if cash_flow > env.capital
        scale = env.capital / max(cash_flow, 1e-9)   # ∈ (0,1]
        ΔC *= scale
        ΔP *= scale
        traded_prem = (abs(ΔC)*C + abs(ΔP)*P) * env.contract_size
        cost   = env.tx_cost_rate * traded_prem
        spread = env.spread_rate  * traded_prem
        cash_flow = (ΔC*C + ΔP*P) * env.contract_size + cost + spread
    end

    env.capital -= cash_flow
    env.n_call  += ΔC
    env.n_put   += ΔP


    #move to next time step.
    idx = env.t + 1
    if idx > length(env.price_changes)
        #episode is over
        if env.capital <= 0 
            return -5.0, true
        end 
        return 0.0, true
    end


    if env.capital <= 1.0
        env.capital = 1e-9
        println("SET CAPITAL TO 1")
        return -2.0, true
    end 

    r_pct = env.price_changes[idx]
    env.S *= (1 + r_pct / 100.0)

    push!(env.realized_ret, env.price_changes[idx]) #add return to experience
    if length(env.realized_ret) > 200
        popfirst!(env.realized_ret)
    end
    update_σ!(env) #update volatility


    env.steps_left -=1

    τ′ = max(env.steps_left, 0) / 252.0
    C′, P′ = bs_prices(env.S, env.K, env.r, env.q, env.σ, τ′)
    opt_value = (env.n_call*C′ + env.n_put*P′) * env.contract_size
    prev_opt_val = (env.n_call*C  + env.n_put*P ) * env.contract_size

    # Approx prev equity = (pre-trade cash + trade cash) + prev_opt_val
    prev_equity = (env.capital + cash_flow) + prev_opt_val
    equity = env.capital + opt_value #lost the cash flow, but now have option value

    rw = 100 * log(max(equity, 1e-9) / max(prev_equity, 1e-9)) #percent reward

    # Risk penalties. Recompute greeks at new state, penalize convexity Γ and sensitivity Vega, to avoid unstable exposure
    Δc_g, Δp_g, Γ, V = greeks(env.S, env.K, env.r, env.q, env.σ, max(τ′, 1e-9))
    portΓ = (env.n_call + env.n_put) * Γ
    portV = (env.n_call + env.n_put) * V
    reward = rw - env.λΓ * abs(portΓ) - env.λV * abs(portV) #subtract from reward per weights.
    # @show rw, portΓ, portV, env.λΓ*abs(portΓ), env.λV*abs(portV), cost+spread

    # Expiry handling: settle intrinsic, relist
    if env.steps_left <= 0
        Cexp, Pexp = bs_prices(env.S, env.K, env.r, env.q, env.σ, 0.0)
        env.capital += (env.n_call*Cexp + env.n_put*Pexp) * env.contract_size
        env.capital = max(env.capital, 1e-9)
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
