def smallest_denominations(M):
    denominations = [500, 200, 100, 50, 20, 10, 5, 2, 1]

    result = []

    for coin in denominations:
        while M >= coin:
            M -= coin
            result.append(coin)

    return result

M = 1043
coins = smallest_denominations(M)

print("the smallest denominations", M, "are:", coins)


output = ""
for i in range(len(coins)):
    if i == 0:
        output = str(coins[i])
    else:
        output = output + "+" + str(coins[i])

print("formatted:", output)

