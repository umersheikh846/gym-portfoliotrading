import numpy as np


def RSI(stock, df):
    df = df[df["symbol"]==stock]
    res = df[["close"]]

    res.insert(1, "diff", res["close"].shift(periods=1).values)
    # res["diff"] = res["close"].shift(periods=1)
    res = res.dropna(how="any")
    res.loc[:,"diff"] = res["close"] - res["diff"]

    res["U"] = res["diff"]
    res.loc[res["U"]<0,"U"] = 0

    res["D"] = - res["diff"]
    res.loc[res["D"]<0,"D"] = 0

    N = 14
    multiplier = 2/(N+1)
    U_EMA = res["U"].tolist()
    D_EMA = res["D"].tolist()
    for n in range(len(U_EMA)):
        if n<N-1:
            U_EMA[n] = None
            D_EMA[n] = None
        elif n == N-1:
            U_EMA[n] = np.mean(res["U"].values[:N])
            D_EMA[n] = np.mean(res["D"].values[:N])
        else:
            U_EMA[n] = (U_EMA[n]-U_EMA[n-1])*multiplier+U_EMA[n-1]
            D_EMA[n] = (D_EMA[n]-D_EMA[n-1])*multiplier+D_EMA[n-1]
    
    res["U_EMA"] = U_EMA
    res["D_EMA"] = D_EMA

    res = res.dropna(how="any")
    res["RSI"] = (1-1/(res["U_EMA"]/res["D_EMA"]+1))*100

    return res[["close", "RSI"]]


def MACD(stock, df):

    df = df[df["symbol"]==stock]
    res = df[["close"]]

    EMA_12 = res["close"].tolist()
    N = 12
    multiplier = 2/(N+1)
    for n in range(len(EMA_12)):
        if n<N-1:
            EMA_12[n] = None
        elif n == N-1:
            EMA_12[n] = np.mean(res["close"].values[:N])
        else:
            EMA_12[n] = (EMA_12[n]-EMA_12[n-1])*multiplier+EMA_12[n-1]
    res.insert(1, "EMA_12", EMA_12)

    EMA_26 = res["close"].tolist()
    N = 26
    multiplier = 2/(N+1)
    for n in range(len(EMA_26)):
        if n<N-1:
            EMA_26[n] = None
        elif n == N-1:
            EMA_26[n] = np.mean(res["close"].values[:N])
        else:
            EMA_26[n] = (EMA_26[n]-EMA_26[n-1])*multiplier+EMA_26[n-1]
    res.insert(1, "EMA_26", EMA_26)

    res = res.dropna(how="any")
    res["MACD"] = res["EMA_12"] - res["EMA_26"]

    return res[["close", "MACD"]]
