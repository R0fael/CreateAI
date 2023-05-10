def simple_out(number, max):
    """
    simple formating outpus
    """
    if number[0] > max:
        return max
    else:
        return round(number[0]*max)


def simple_in(number, max):
    """
    simple formating outpus
    """
    if number > max:
        return max
    else:
        return number/max
