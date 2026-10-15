# Options Pricing and Risk Analysis

A Python project for pricing European options and examining how their values
respond to changes in market inputs.

## Models and analysis

- Black-Scholes call and put prices
- Cox-Ross-Rubinstein binomial prices
- Monte Carlo prices with standard errors and 95% confidence intervals
- Implied volatility
- Delta, gamma, vega, theta, and rho
- Greek-based scenario P&L compared with full repricing

## Assumptions

- European options on an underlying with no dividends
- Constant risk-free rate and volatility during the option's life
- Risk-neutral pricing for the binomial and Monte Carlo models
- Scenario P&L is per underlying unit unless scaled to a position
- Example values are illustrative, not live market quotes

## Project files

- `pricing.py` contains the reusable pricing functions.
- `analysis.ipynb` demonstrates the models and scenario analysis.
- `test_pricing.py` checks benchmark values and model behaviour.
- `requirements.txt` lists the Python dependencies.

## Setup

Activate the project's virtual environment, then install the requirements:

```powershell
python -m pip install -r requirements.txt