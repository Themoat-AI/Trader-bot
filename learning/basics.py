def percent_return(old_price, new_price):
    """
    Calculate the percentage return between two prices.

    Parameters:
    old_price (float): The initial price.
    new_price (float): The final price.

    Returns:
    float: The percentage return.
    """
    if old_price == 0:
        raise ValueError("Old price cannot be zero.")
    
    return ((new_price - old_price) / old_price) * 100

def shares_to_buy(cash, price, pct):
    """
    Calculate the number of shares that can be bought with a given amount of cash.

    Parameters:
    cash (float): The amount of cash available.
    price (float): The price per share.
    pct (float): The percentage of cash to use for the purchase.

    Returns:
    int: The number of shares that can be bought.
    """
    if price == 0:
        raise ValueError("Price cannot be zero.")

    # Accept 0.10 as 10% as well as 10.
    percentage = pct * 100 if 0 < pct < 1 else pct
    return int((cash * percentage / 100) / price)

def position_value(shares, price):
    """
    Calculate the total value of a position based on the number of shares and the price per share.

    Parameters:
    shares (int): The number of shares held.
    price (float): The price per share.

    Returns:
    float: The total value of the position.
    """
    return shares * price

assert percent_return(100, 110) == 10.0
assert shares_to_buy(1000, 30, 0.10) == 3
assert position_value(3, 30) == 90
print("All tests passed!")