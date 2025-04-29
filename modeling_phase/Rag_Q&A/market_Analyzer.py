import pandas as pd
import numpy as np
from typing import Dict
import math
from typing import List

# --- Market Metrics Analyzer ---
class MarketAnalyzer:
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}

        df.columns = df.columns.str.strip().str.lower()
        price_col = 'price' if 'price' in df.columns else 'close'
        volume_col = 'vol.' if 'vol.' in df.columns else 'volume'

        metrics = {
            'current_price': df[price_col].iloc[-1],
            '30_day_volatility': df[price_col].pct_change().std() * np.sqrt(365),
            'max_drawdown': (df[price_col] / df[price_col].cummax() - 1).min(),
            'avg_daily_volume': df[volume_col].mean() if volume_col in df.columns else None,
            'last_30_day_return': (
                df[price_col].iloc[-1] / df[price_col].iloc[-30] - 1
            ) if len(df) >= 30 else None
        }
        return metrics

    def analyze_all_markets(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        results = {}
        for asset, df in market_data.items():
            try:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                results[asset] = self.calculate_metrics(df)
            except Exception as e:
                print(f"Error analyzing {asset}: {str(e)}")
                results[asset] = {}
        return results


# --- Financial Statement Processor ---
class FinancialStatementProcessor:
    @staticmethod
    def filter_table(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        zero_percentage = (df == 0).sum(axis=1) / df.shape[0]
        return df.loc[zero_percentage < threshold]

    def process_statements(self, IS, BS, CF) -> Dict[str, pd.DataFrame]:
        IS_final = self.filter_table(table_transform(IS[0]))

        BS_annual = table_transform(BS[0])
        if IS_final.index[-1] == 'TTM ':
            BS_quarterly = table_transform(BS[1])
            last_row = BS_quarterly.tail(1)
            BS_annual = pd.concat([BS_annual, last_row])
        BS_final = self.filter_table(BS_annual)

        CF_final = self.filter_table(table_transform(CF[0]))

        return {
            'Income Statement': IS_final,
            'Balance Sheet': BS_final,
            'Cash Flow': CF_final
        }


# --- Dividends Example ---
divs = pd.DataFrame({
    "time": np.round(np.arange(0.11, 2.01, 0.25), 2),
    "fixed": np.linspace(1.5, 1, 8),
    "proportional": np.linspace(1, 1.5, 8)
})


# --- Interest Rate & Survival Functions ---
def disc_factor_fcn(T, t, **kwargs):
    return np.exp(-0.03 * (T - t))

def surv_prob_fcn(T, t, **kwargs):
    return np.exp(-0.07 * (T - t))


# --- Volatility Term Structure ---
def variance_cumulation_from_vols(df: pd.DataFrame):
    def vc(T, t):
        # Simple interpolation method for cumulative variance
        vol_interp = np.interp([t, T], df["time"], df["volatility"])
        avg_vol = np.mean(vol_interp)
        return avg_vol ** 2 * (T - t)
    return vc

vc = variance_cumulation_from_vols(pd.DataFrame({
    "time": [0.1, 2, 3],
    "volatility": [0.2, 0.5, 1.2]
}))

print(f"Cumulated variance to 18 months is {vc(1.5, 0):.6f}")



class FinancialCalculator:

    @staticmethod
    def calc_dividend_discount_model_value(dividend: float, required_rate: float) -> float:
        return dividend / required_rate

    @staticmethod
    def calc_preferred_stock_value(dividend: float, required_rate: float) -> float:
        return dividend / required_rate

    @staticmethod
    def calc_constant_growth_valuation(dividend: float, required_rate: float, dividend_growth: float) -> float:
        return dividend / (required_rate - dividend_growth)

    @staticmethod
    def calc_expected_rate(dividend: float, value: float, dividend_growth: float) -> float:
        return (dividend / value) + dividend_growth

    @staticmethod
    def calc_capm_model(expected_return_market: float, risk_free_rate: float, beta: float) -> float:
        return risk_free_rate + beta * (expected_return_market - risk_free_rate)

    @staticmethod
    def calc_weighted_average_cost_of_capital(
        total_value: float, cost_equity: float, cost_debt: float,
        market_value_equity: float, market_value_debt: float, tax_rate: float
    ) -> float:
        equity_weight = market_value_equity / total_value
        debt_weight = market_value_debt / total_value
        return (equity_weight * cost_equity) + (debt_weight * cost_debt * (1 - tax_rate))

    @staticmethod
    def calc_equity_multiplier(total_assets: float, stockholders_equity: float) -> float:
        return total_assets / stockholders_equity

    @staticmethod
    def calc_equity_multiplier_from_equity_ratio(equity_ratio: float) -> float:
        return 1 / equity_ratio

    @staticmethod
    def calc_market_to_book_ratio(market_price: float, book_value_per_share: float) -> float:
        return market_price / book_value_per_share

    @staticmethod
    def calc_return_on_equity(net_income: float, total_equity: float) -> float:
        return net_income / total_equity

    @staticmethod
    def calc_earnings_per_share(net_income: float, number_of_common_shares: float) -> float:
        return net_income / number_of_common_shares

    @staticmethod
    def calc_price_earnings_ratio(price_per_share: float, earnings_per_share: float) -> float:
        return price_per_share / earnings_per_share

    @staticmethod
    def calc_dividend_payout_ratio(dividends_per_share: float, earnings_per_share: float) -> float:
        return dividends_per_share / earnings_per_share

    @staticmethod
    def calc_dividend_yield(dividends_per_share: float, price_per_share: float) -> float:
        return dividends_per_share / price_per_share

    @staticmethod
    def calc_book_value_per_share(total_common_equity: float, number_of_common_shares: float) -> float:
        return total_common_equity / number_of_common_shares

    @staticmethod
    def calc_market_value_per_share(market_capitalization: float, number_of_common_shares: float) -> float:
        return market_capitalization / number_of_common_shares

    @staticmethod
    def calc_book_value_per_bond(bond_equity_value: float, number_of_bonds: float) -> float:
        return bond_equity_value / number_of_bonds

    @staticmethod
    def calc_market_value_per_bond(bond_price: float, number_of_bonds: float) -> float:
        return bond_price / number_of_bonds

    @staticmethod
    def calc_holding_period_return(initial_value: float, final_value: float, cash_flows: float = 0.0) -> float:
        return (final_value - initial_value + cash_flows) / initial_value

    @staticmethod
    def calc_arithmetic_mean_return(returns: List[float]) -> float:
        return sum(returns) / len(returns)

    @staticmethod
    def calc_geometric_mean_return(returns: List[float]) -> float:
        product = 1.0
        for r in returns:
            product *= (1 + r)
        return product ** (1 / len(returns)) - 1

    @staticmethod
    def calc_zero_coupon_bond_value(face_value: float, rate_or_yield: float, time_to_maturity: float) -> float:
        return face_value / (1 + rate_or_yield) ** time_to_maturity
