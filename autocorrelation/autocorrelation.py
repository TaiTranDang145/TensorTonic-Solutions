def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0..max_lag
    """
    n = len(series)
    mean = sum(series) / n

    # total variance γ0
    gamma0 = sum((x - mean) ** 2 for x in series)
    if gamma0 == 0:
        return [1] + [0] * max_lag
    ans = []

    for k in range(max_lag + 1):
        num = 0
        for t in range(n - k):
            num += (series[t] - mean) * (series[t + k] - mean)

        ans.append(num / gamma0)

    return ans