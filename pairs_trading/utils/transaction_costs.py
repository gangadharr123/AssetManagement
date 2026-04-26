import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def compute_transaction_cost(trade_date):
    # For 2015-2025, one-way cost = commission (5bps) + market impact (20bps)
    # Each pair trade = 4 one-way trades
    one_way_bps = config.COMMISSION_BPS + config.MARKET_IMPACT_BPS
    total_bps = 4 * one_way_bps
    return total_bps / 10000.0

def apply_costs(trades):
    for trade in trades:
        cost = compute_transaction_cost(trade['entry_date'])
        trade['transaction_cost'] = cost
        trade['return_after_cost'] = trade['return_before_cost'] - cost
    return trades