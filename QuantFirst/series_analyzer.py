def sum_range(series, left, right):
    series_len = len(series)
    start_index = left if left > 0 else 0
    end_index = right if right < series_len else series_len
    part_sum = 0
    for index in range(start_index, end_index):
        part_sum += series[index]
    return part_sum


def EMA(series, lastN=-1):
    series_len = len(series)
    series_left = 0 if lastN < 0 else (series_len - lastN)
    series_right = series_len if series_len > 0 else 0
    if series_left < 0 or series_right < series_left:
        return 0
    return sum_range(series, series_left, series_right) / lastN


def sliding_window_sum(series, window_size):
    series_len = len(series)
    range_sums = [0] * series_len
    if window_size > series_len:
        return range_sums
    init_sum = 0
    for index in range(0, window_size):
        init_sum += series[index]
    range_sums[window_size - 1] = init_sum
    for index in range(window_size, series_len):
        range_sums[index] = range_sums[index - 1] + series[index] - series[index - window_size]
    return range_sums


def series_MA(series, window_size):
    acc_sums = sliding_window_sum(series, window_size)
    for index in range(0, len(series)):
        acc_sums[index] = acc_sums[index] / window_size
    return acc_sums


def series_EMA(series, period):
    series_len = len(series)
    nm1 = period - 1
    denominator = period + 1
    ema_values = [0] * series_len
    ema_values[0] = series[0]
    for index in range(1, series_len):
        ema_values[index] = (2 * series[index] + nm1 * ema_values[index - 1]) / denominator
    return ema_values


def series_dif(series, fast = 12, slow = 26):
    series_len = len(series)
    fast_ema = series_EMA(series, fast)
    slow_ema = series_EMA(series, slow)
    difs = [0] * series_len
    for index in range(0, series_len):
        difs[index] = fast_ema[index] - slow_ema[index]
    return difs


def series_macd(series):
    # test_series = [x for x in range(0, 50)]
    series_len = len(series)
    diff = series_dif(series)
    dea = series_EMA(diff, 9)
    macd = [0] * series_len
    for index in range(0, series_len):
        macd[index] = 2 * (diff[index] - dea[index])
    # print(diff[25: series_len])
    # print(dea[25: series_len])
    # print(macd[25: series_len])
    return macd
