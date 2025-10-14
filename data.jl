using CSV
using DataFrames
include("gbm.jl")


function get_historical(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    change_percent = map(row -> Float64(row.changeClosePercent), eachrow(df))
    return reverse(change_percent)
end

function get_historical_raw(ticker::String)
    df = CSV.read("data/$(ticker).csv", DataFrame)
    sort!(df, :date)
    return df
end

function get_close_features(ticker::String) 
    # Path to your CSV 
    file = "data/$(ticker).csv" 
    # Read only the close column into a DataFrame 
    df = CSV.read(file, DataFrame; select=["close"]) 
    df = df[end:-1:1, :] 
    return df 
end

function get_historical_vscores(ticker::String, OBS::Int=100, EPOCH::Int=1000, EXT::Int=20, seed::Int=3)
    df = CSV.read("data/$ticker.csv", DataFrame)
    raw = map(row -> Float64(row.close), eachrow(df))

    vscore_list = vscore(reverse(raw), OBS, EPOCH, EXT)
    return vscore_list
end

# 2. Exponential Moving Average (EMA)
function ema_series(ticker::String, window::Int=14)
    df = get_historical_raw(ticker)
    
    n = nrow(df)

    if window <= 0 || n < window
        throw(ArgumentError("Window size must be > 0 and <= length of price series"))
    end

    α = 2.0 / (window + 1)
    ema = Vector{Union{Missing, Float64}}(undef, n)

    # First EMA value is just a simple average
    ema[1:window-1] .= NaN
    ema[window] = mean(df.close[1:window])

    for t in (window+1):n
        ema[t] = α * df.close[t] + (1 - α) * ema[t-1]
    end

    return ema
end

# 3. RSI
function rsi_series(ticker::String, window::Int=14)
    df = get_historical_raw(ticker)
    
    n = nrow(df)

    if window <= 0 || n < window 
        throw(ArgumentError9("Window size must be greater than zero and larger than the length of the price series"))
    end

    deltas = diff(df.close)
    gains = max.(deltas, 0.0)
    losses = -min.(deltas, 0.0)

    avg_gain = Vector{Float64}(undef, n-1)
    avg_loss = Vector{Float64}(undef, n-1)

    # Initial averages
    avg_gain[1:window-1] .= NaN
    avg_loss[1:window-1] .= NaN
    avg_gain[window] = mean(gains[1:window])
    avg_loss[window] = mean(losses[1:window])

    # Wilder's smoothing
    for i in (window+1):(n-1)
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i]) / window
    end

    # RSI computation
    rsi = Vector{Float64}(undef, n)
    rsi[1:window] .= NaN
    for i in (window+1):n
        rs = avg_loss[i-1] == 0 ? Inf : avg_gain[i-1] / avg_loss[i-1]
        rsi[i] = 100.0 - (100.0 / (1 + rs))
    end

    return rsi
end

# 5. Bollinger Band %B
function bb_percentb_series(ticker::String, window::Int=20)
    df = get_historical_raw(ticker)
    ma = [i < window ? NaN : mean(df.close[i-window+1:i]) for i in 1:nrow(df)]
    stddev = [i < window ? NaN : std(df.close[i-window+1:i]) for i in 1:nrow(df)]
    upper = ma .+ 2 .* stddev
    lower = ma .- 2 .* stddev
    percentb = (df.close .- lower) ./ (upper .- lower)
    return percentb
end

function get_all_features(ticker::String, LOOK_BACK_PERIOD::Int=100)
    df = DataFrame()


    df.vscores = get_historical_vscores(ticker, LOOK_BACK_PERIOD)
    df.ema = ema_series(ticker)[LOOK_BACK_PERIOD+1:end]  
    df.rsi = rsi_series(ticker)[LOOK_BACK_PERIOD+1:end]  
    df.bb_percentb = bb_percentb_series(ticker)[LOOK_BACK_PERIOD+1:end]
    df.percent_change = get_historical_raw(ticker).changeClosePercent[LOOK_BACK_PERIOD+1 : end]
    
    df_standardized = deepcopy(df)
    cols_to_standardize = [:rsi, :ema]
    for col in cols_to_standardize
        μ = mean(df[!, col])
        σ = std(df[!, col])
        df_standardized[!, col] = (df[!, col] .- μ) ./ σ
    end

    day_price = get_historical_raw(ticker).changeClosePercent[LOOK_BACK_PERIOD+1 : end]
    return df_standardized, day_price
end

