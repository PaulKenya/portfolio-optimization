from typing import Dict

def calculate_return(portfolio, historical_data):
    returns = {}
    for asset, investment in portfolio.items():
        returns[asset] = investment * historical_data[asset].pct_change().iloc[-1]
    total_return = sum(returns.values())
    return total_return


def calculate_portfolio_profit(pivoted_data, selected_assets, weights):
    portfolio_returns = pivoted_data[selected_assets].dot(weights)
    total_return = portfolio_returns.sum()
    profit_percentage = total_return * 100  # Convert to percentage
    return profit_percentage
