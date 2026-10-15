# Test the option pricing functions.

import unittest

import numpy as np

from pricing import (
    black_scholes_call_price,
    black_scholes_greeks,
    black_scholes_put_price,
    binomial_option_price,
    call_payoff,
    implied_volatility,
    monte_carlo_option_price,
    put_payoff,
)


class TestPricing(unittest.TestCase):
    def setUp(self):
        # Set shared test inputs.

        self.market_inputs = {
            "current_price": 100,
            "strike_price": 100,
            "risk_free_rate": 0.05,
            "time_to_expiry": 1.0,
            "volatility": 0.20,
        }

    def test_option_payoffs(self):
        # Check call and put payoffs at expiry.

        prices_at_expiry = np.array([80, 100, 120])

        np.testing.assert_array_equal(
            call_payoff(prices_at_expiry, 100),
            [0, 0, 20],
        )

        np.testing.assert_array_equal(
            put_payoff(prices_at_expiry, 100),
            [20, 0, 0],
        )

    def test_black_scholes_benchmark_prices(self):
        # Check against known Black-Scholes values.

        call_price = black_scholes_call_price(**self.market_inputs)
        put_price = black_scholes_put_price(**self.market_inputs)

        self.assertAlmostEqual(call_price, 10.4506, places=4)
        self.assertAlmostEqual(put_price, 5.5735, places=4)

    def test_put_call_parity(self):
        # Check the relationship between call and put prices.

        call_price = black_scholes_call_price(**self.market_inputs)
        put_price = black_scholes_put_price(**self.market_inputs)

        expected_difference = (
            self.market_inputs["current_price"]
            - self.market_inputs["strike_price"]
            * np.exp(
                -self.market_inputs["risk_free_rate"]
                * self.market_inputs["time_to_expiry"]
            )
        )

        self.assertAlmostEqual(
            call_price - put_price,
            expected_difference,
            places=10,
        )

    def test_greek_delta(self):
        # Compare call delta with a small price bump.

        greeks = black_scholes_greeks(
            **self.market_inputs,
            option_type="call",
        )

        price_bump = 0.01

        higher_inputs = dict(self.market_inputs)
        lower_inputs = dict(self.market_inputs)

        higher_inputs["current_price"] += price_bump
        lower_inputs["current_price"] -= price_bump

        numerical_delta = (
            black_scholes_call_price(**higher_inputs)
            - black_scholes_call_price(**lower_inputs)
        ) / (2 * price_bump)

        self.assertAlmostEqual(
            greeks["delta"],
            numerical_delta,
            places=5,
        )

    def test_binomial_converges_to_black_scholes(self):
        # Check that a large tree approaches the Black-Scholes price.

        call_price = black_scholes_call_price(**self.market_inputs)

        tree_price = binomial_option_price(
            **self.market_inputs,
            num_steps=500,
            option_type="call",
        )

        self.assertLess(abs(tree_price - call_price), 0.01)

    def test_implied_volatility_recovers_input(self):
        # Recover the volatility used to generate the call price.

        market_option_price = black_scholes_call_price(
            **self.market_inputs
        )

        estimated_volatility = implied_volatility(
            market_option_price=market_option_price,
            current_price=self.market_inputs["current_price"],
            strike_price=self.market_inputs["strike_price"],
            risk_free_rate=self.market_inputs["risk_free_rate"],
            time_to_expiry=self.market_inputs["time_to_expiry"],
            option_type="call",
        )

        self.assertAlmostEqual(estimated_volatility, 0.20, places=6)

    def test_monte_carlo_is_reproducible(self):
        # The same seed should return the same result.

        first_result = monte_carlo_option_price(
            **self.market_inputs,
            num_simulations=100_000,
            option_type="call",
            random_seed=42,
        )

        second_result = monte_carlo_option_price(
            **self.market_inputs,
            num_simulations=100_000,
            option_type="call",
            random_seed=42,
        )

        self.assertEqual(first_result, second_result)
        self.assertGreater(first_result["standard_error"], 0)
        self.assertLess(
            abs(first_result["option_price"] - 10.4506),
            0.20,
        )

    def test_invalid_option_type_is_rejected(self):
        # Reject an unsupported option type.

        with self.assertRaises(ValueError):
            binomial_option_price(
                **self.market_inputs,
                num_steps=100,
                option_type="straddle",
            )


if __name__ == "__main__":
    unittest.main()