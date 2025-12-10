
def cantor_pairing(x, y, mod = (1 << 32)):
    # https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
    # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    result = (x + y) * (x + y + 1) // 2 + y

    if mod is not None:
        result = result % mod

    return result

