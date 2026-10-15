# Calculate call and put option payoffs at expiry.

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

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

def black_scholes_price(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
    option_type="call",
):
    # Check the inputs.

    if current_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive.")

    if time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("Time and volatility must be positive.")

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    # Calculate d1 and d2.

    d1 = (
        np.log(current_price / strike_price)
        + (risk_free_rate + volatility**2 / 2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    # Calculate the European call or put price.

    discounted_strike_price = strike_price * np.exp(
        -risk_free_rate * time_to_expiry
    )

    if option_type == "call":
        option_price = (
            current_price * norm.cdf(d1)
            - discounted_strike_price * norm.cdf(d2)
        )
    else:
        option_price = (
            discounted_strike_price * norm.cdf(-d2)
            - current_price * norm.cdf(-d1)
        )

    return float(option_price)

def black_scholes_call_price(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
):
    # Price a European call using the existing Black-Scholes function.

    return black_scholes_price(
        current_price=current_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type="call",
    )


def black_scholes_put_price(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
):
    # Price a European put using the existing Black-Scholes function.

    return black_scholes_price(
        current_price=current_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type="put",
    )


def _calculate_d1_d2(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
):
    # Check that the numerical inputs are finite.

    numerical_inputs = [
        current_price,
        strike_price,
        risk_free_rate,
        time_to_expiry,
        volatility,
    ]

    if not np.all(np.isfinite(numerical_inputs)):
        raise ValueError("All numerical inputs must be finite.")

    if min(current_price, strike_price, time_to_expiry, volatility) <= 0:
        raise ValueError("Prices, time and volatility must be positive.")

    # Calculate the two standard Black-Scholes terms.

    d1 = (
        np.log(current_price / strike_price)
        + (risk_free_rate + volatility**2 / 2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    return d1, d2


def black_scholes_greeks(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
    option_type="call",
):
    # Check the selected option type.

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    # Calculate the common Black-Scholes terms.

    d1, d2 = _calculate_d1_d2(
        current_price,
        strike_price,
        risk_free_rate,
        time_to_expiry,
        volatility,
    )

    normal_density = norm.pdf(d1)

    discounted_strike_price = strike_price * np.exp(
        -risk_free_rate * time_to_expiry
    )

    # Gamma and vega are the same for matching calls and puts.

    gamma = normal_density / (
        current_price * volatility * np.sqrt(time_to_expiry)
    )

    vega = current_price * normal_density * np.sqrt(time_to_expiry)

    time_decay_component = -(
        current_price * normal_density * volatility
    ) / (2 * np.sqrt(time_to_expiry))

    # Calculate the option-specific Greeks.

    if option_type == "call":
        delta = norm.cdf(d1)

        theta = (
            time_decay_component
            - risk_free_rate * discounted_strike_price * norm.cdf(d2)
        )

        rho = time_to_expiry * discounted_strike_price * norm.cdf(d2)

    else:
        delta = norm.cdf(d1) - 1

        theta = (
            time_decay_component
            + risk_free_rate * discounted_strike_price * norm.cdf(-d2)
        )

        rho = -time_to_expiry * discounted_strike_price * norm.cdf(-d2)

    # Theta is per year of elapsed time.
    # Vega and rho use decimal volatility and interest-rate changes.

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def binomial_option_price(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
    num_steps,
    option_type="call",
):
    # Check the inputs supported by this version.

    if not isinstance(num_steps, (int, np.integer)) or num_steps < 1:
        raise ValueError("Number of steps must be a positive integer.")

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    if min(current_price, strike_price, time_to_expiry, volatility) <= 0:
        raise ValueError("Prices, time and volatility must be positive.")

    # Calculate the movement and discount factors.

    time_step = time_to_expiry / num_steps
    up_factor = np.exp(volatility * np.sqrt(time_step))
    down_factor = 1 / up_factor
    discount_factor = np.exp(-risk_free_rate * time_step)

    up_probability = (
        np.exp(risk_free_rate * time_step) - down_factor
    ) / (up_factor - down_factor)

    if not 0 < up_probability < 1:
        raise ValueError("Invalid tree probability. Increase num_steps.")

    # Build the stock and option price trees.

    stock_prices = np.zeros((num_steps + 1, num_steps + 1))
    option_prices = np.zeros_like(stock_prices)

    for step in range(num_steps + 1):
        for up_moves in range(step + 1):
            down_moves = step - up_moves

            stock_prices[step, up_moves] = (
                current_price
                * up_factor**up_moves
                * down_factor**down_moves
            )

    # Calculate option payoffs at expiry.

    prices_at_expiry = stock_prices[-1, :]

    if option_type == "call":
        option_prices[-1, :] = np.maximum(
            prices_at_expiry - strike_price,
            0.0,
        )
    else:
        option_prices[-1, :] = np.maximum(
            strike_price - prices_at_expiry,
            0.0,
        )

    # Work backwards through the tree.

    for step in range(num_steps - 1, -1, -1):
        for up_moves in range(step + 1):
            option_prices[step, up_moves] = discount_factor * (
                up_probability * option_prices[step + 1, up_moves + 1]
                + (1 - up_probability) * option_prices[step + 1, up_moves]
            )

    return float(option_prices[0, 0])


def monte_carlo_option_price(
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    volatility,
    num_simulations=100_000,
    option_type="call",
    random_seed=42,
):
    # Check the inputs supported by this version.

    if not isinstance(num_simulations, int) or num_simulations < 2:
        raise ValueError("Number of simulations must be an integer above 1.")

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    numerical_inputs = [
        current_price,
        strike_price,
        risk_free_rate,
        time_to_expiry,
        volatility,
    ]

    if not np.all(np.isfinite(numerical_inputs)):
        raise ValueError("All numerical inputs must be finite.")

    if min(current_price, strike_price, time_to_expiry, volatility) <= 0:
        raise ValueError("Prices, time and volatility must be positive.")

    # Simulate terminal prices under the risk-neutral model.

    random_generator = np.random.default_rng(random_seed)
    random_shocks = random_generator.standard_normal(num_simulations)

    prices_at_expiry = current_price * np.exp(
        (risk_free_rate - 0.5 * volatility**2) * time_to_expiry
        + volatility * np.sqrt(time_to_expiry) * random_shocks
    )

    # Calculate and discount the option payoffs.

    if option_type == "call":
        option_payoffs = np.maximum(
            prices_at_expiry - strike_price,
            0.0,
        )
    else:
        option_payoffs = np.maximum(
            strike_price - prices_at_expiry,
            0.0,
        )

    discounted_payoffs = (
        np.exp(-risk_free_rate * time_to_expiry) * option_payoffs
    )

    # Estimate the option price and standard error.

    option_price = np.mean(discounted_payoffs)

    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(
        num_simulations
    )

    confidence_margin = norm.ppf(0.975) * standard_error

    return {
        "option_price": float(option_price),
        "standard_error": float(standard_error),
        "ci_lower": float(option_price - confidence_margin),
        "ci_upper": float(option_price + confidence_margin),
    }


def implied_volatility(
    market_option_price,
    current_price,
    strike_price,
    risk_free_rate,
    time_to_expiry,
    option_type="call",
    volatility_lower_bound=0.0001,
    volatility_upper_bound=3.0,
):
    # Check the inputs supported by this version.

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    if volatility_lower_bound <= 0:
        raise ValueError("The lower volatility bound must be positive.")

    if volatility_upper_bound <= volatility_lower_bound:
        raise ValueError("The upper bound must exceed the lower bound.")

    # Check the theoretical option-price bounds.

    discounted_strike_price = strike_price * np.exp(
        -risk_free_rate * time_to_expiry
    )

    if option_type == "call":
        minimum_price = max(current_price - discounted_strike_price, 0.0)
        maximum_price = current_price
        pricing_function = black_scholes_call_price
    else:
        minimum_price = max(discounted_strike_price - current_price, 0.0)
        maximum_price = discounted_strike_price
        pricing_function = black_scholes_put_price

    if not minimum_price < market_option_price < maximum_price:
        raise ValueError("Market price must be inside the option price bounds.")

    # Measure the pricing error at a trial volatility.

    def pricing_error(trial_volatility):
        model_price = pricing_function(
            current_price,
            strike_price,
            risk_free_rate,
            time_to_expiry,
            trial_volatility,
        )

        return model_price - market_option_price

    # Find the volatility that makes the pricing error zero.

    return float(
        brentq(
            pricing_error,
            volatility_lower_bound,
            volatility_upper_bound,
        )
    )