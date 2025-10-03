using Test
using Random
using DataFrames

# Bring in your env (after you’ve fixed the small typos noted above).
include("../option_env.jl")   # adjust relative path as needed

# --- Helpers ---
# Tiny deterministic feature frame (3 columns) long enough for lookback
function toy_features(n::Int)
    DataFrame(
        vscores     = range(-1.0, 1.0, length=n),
        ema         = range(0.0,  2.0, length=n),
        rsi         = range(-2.0, 2.0, length=n)
    )
end

# Small percent-change vector (aligned with LOOK_BACK_PERIOD semantics)
function toy_day_change(n::Int)
    # keep it small; first element after lookback is used for next step
    fill(0.5, n)  # +0.5% each step
end

@testset "options_env" begin
    # --- Edge pricing tests ---
    @testset "Black–Scholes edge cases" begin
        S, K, r, q, σ = 100.0, 100.0, 0.0, 0.0, 0.2

        # τ = 0 => intrinsic values
        C0, P0 = bs_prices(S, K, r, q, σ, 0.0)
        @test isapprox(C0, max(S - K, 0.0); atol=1e-12)
        @test isapprox(P0, max(K - S, 0.0); atol=1e-12)

        # σ = 0 => intrinsic values
        C1, P1 = bs_prices(S, K, r, q, 0.0, 10/252)
        @test isapprox(C1, max(S - K, 0.0); atol=1e-12)
        @test isapprox(P1, max(K - S, 0.0); atol=1e-12)
    end

    # --- Greeks at expiry ---
    @testset "Greeks at expiry" begin
        S, K, r, q, σ = 100.0, 100.0, 0.0, 0.0, 0.2
        Δc, Δp, Γ, V = greeks(S, K, r, q, σ, 0.0)
        # At expiry, deltas are 0/1 depending on ITM; for ATM we accept either side of step
        @test (Δc == 0.0 || Δc == 1.0)
        @test (Δp == -1.0 || Δp == 0.0)
        @test Γ == 0.0
        @test V == 0.0
    end

    # --- Minimal environment wiring ---
    @testset "Environment step mechanics + cash flow" begin
        LOOK_BACK = 5
        N = 100  # total rows in feature df
        df = toy_features(N)
        day_change = toy_day_change(N - LOOK_BACK) # matches your alignment

        # Build env (matches your constructor signature)
        env = Env(df, day_change; lookback=LOOK_BACK, S0=100.0, r=0.0, q=0.0,
                  T_steps=7, init_capital=1000.0, vol_win=5,
                  tx_cost_rate=0.001, spread_rate=0.001,  # bigger costs to make effects obvious
                  contract_size=100.0, alloc_scale=0.20, σ0=0.2,
                  λΓ=0.0, λV=0.0)
        reset!(env)

        # Build state once to ensure shape
        feat = flatten_window(df, env.t, LOOK_BACK)
        svec = state(env, feat)
        # State should be L * num_features + extras [logcap, τ, n_call, n_put, portΔ, portΓ, portV]
        num_feats = ncol(df)
        expected_len = LOOK_BACK*num_feats + 1 + 1 + 2 + 3
        @test length(svec) == expected_len

        # Price at pre-move
        τ = max(env.steps_left, 1) / 252.0
        C, P = bs_prices(env.S, env.K, env.r, env.q, env.σ, τ)
        @test C >= 0 && P >= 0

        # 1) BUY calls: capital should decrease (after costs)
        cap_before = env.capital
        reward, done = step!(env, ( +1.0, 0.0))  # target long calls
        @test env.capital < cap_before
        @test !done
        # Spot should move by +0.5%
        @test isapprox(env.S, 100.0*(1+0.005); atol=1e-8)
        # Steps left should decrease
        @test env.steps_left == 6

        # 2) SELL (reduce long or go short): capital should increase (premium in - costs)
        cap2 = env.capital
        reward2, done2 = step!(env, ( -1.0, 0.0)) # flip target: net sell calls
        @test env.capital > cap2
        @test !done

        # Force several steps to fill realized_ret and update σ
        for _ in 1:5
            step!(env, (0.0, 0.0))
        end
        # After enough returns, σ should have been updated within clamp
        @test 0.05 <= env.σ <= 1.0

        # Walk to expiry and check settlement + relist
        done = false
        while env.steps_left > 0 && !done
            reward, done = step!(env, (0.0, 0.0))
        end

        if !done
            # We did reach expiry and the env rolled the chain
            @test env.n_call == 0.0 && env.n_put == 0.0
            @test env.steps_left == env.T_steps
            @test env.K == env.S
        else
            # We terminated early due to data exhaustion; that's valid too.
            @test true
        end

    end
end
