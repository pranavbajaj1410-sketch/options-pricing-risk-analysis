# Calculate call and put option payoffs at expiry.

import numpy as np
from scipy.stats import norm

def call_payoff(price_at_expiry, strike_price):
    # Calculate the call payoff per underlying unit.
    price_at_expiry = np.asarray(price_at_expiry, dtype=float)

    return np.maximum(price_at_expiry - strike_price, 0.0)


def put_payoff(price_at_expiry, strike_price):
    # Calculate the put payoff per underlying unit.
    price_at_expiry = np.asarray(price_at_expiry, dtype=float)

    return np.maximum(strike_price - price_at_expiry, 0.0)


if __name__ == "__main__":
    # Check the payoffs using three example prices at expiry.
    strike_price = 100

    for price_at_expiry in [80, 100, 120]:
        call_payoff_amount = call_payoff(price_at_expiry, strike_price)
        put_payoff_amount = put_payoff(price_at_expiry, strike_price)

        print(
            f"Price at expiry: {price_at_expiry} | "
            f"Call payoff: {call_payoff_amount:.2f} | "
            f"Put payoff: {put_payoff_amount:.2f}"
        )
