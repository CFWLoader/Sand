base_invest = 20000
inflate_rate = 1.02
profile_rate = 1.024


def retire_prod():
    start_age = 25
    acc_invest = base_invest
    current_value = base_invest
    actual_value = base_invest
    for it_year in range(start_age + 1, 81):
        if it_year < 35:
            actual_value += base_invest
            acc_invest += base_invest
            current_value += base_invest
        current_value = current_value * inflate_rate
        actual_value = actual_value * profile_rate
        # print('ini[%d] %d = %f <%f>' % (acc_invest, it_year, current_value, actual_value))
        print(acc_invest, it_year, current_value, actual_value)


def exceed_prod():
    pass


if __name__ == '__main__':
    retire_prod()
    # for idx in range(0, 10):
    #     print(idx)