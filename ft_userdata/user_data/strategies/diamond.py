# IMPORTANT: Diamond strategy is designed to be pure and
# cuz of that it have not any indicator population. idea is that
# It is just use the pure dataframe ohlcv data for calculation
# of buy/sell signals, But you can add your indicators and add
# your key names inside catagorical hyperoptable params and
# than you be able to hyperopt them as well.
# thanks to: @Kroissan, @drakes00 And @xmatthias for his patience and helps
# Author: @Mablue (Masoud Azizi)
# github: https://github.com/mablue/
# * freqtrade backtesting --strategy Diamond

# freqtrade hyperopt --hyperopt-loss ShortTradeDurHyperOptLoss --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
# *    3/10:     76 trades. 51/18/7 Wins/Draws/Losses. Avg profit   1.92%. Median profit   2.40%. Total profit  0.04808472 BTC (  48.08%). Avg duration 5:06:00 min. Objective: 1.75299
# freqtrade hyperopt --hyperopt-loss OnlyProfitHyperOptLoss --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
# *   10/10:     76 trades. 39/34/3 Wins/Draws/Losses. Avg profit   0.61%. Median profit   0.05%. Total profit  0.01528359 BTC (  15.28%). Avg duration 17:32:00 min. Objective: -0.01528
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
# *    4/10:     15 trades. 10/2/3 Wins/Draws/Losses. Avg profit   1.52%. Median profit   7.99%. Total profit  0.00754274 BTC (   7.54%). Avg duration 1 day, 0:04:00 min. Objective: -0.90653
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
# *    7/10:    130 trades. 68/54/8 Wins/Draws/Losses. Avg profit   0.71%. Median profit   0.06%. Total profit  0.03050369 BTC (  30.50%). Avg duration 10:07:00 min. Objective: -11.08185
# freqtrade hyperopt --hyperopt-loss SortinoHyperOptLoss --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
# *    2/10:     10 trades. 7/0/3 Wins/Draws/Losses. Avg profit   5.50%. Median profit   7.05%. Total profit  0.01817970 BTC (  18.18%). Avg duration 0:27:00 min. Objective: -11.72450
# freqtrade hyperopt --hyperopt-loss SortinoHyperOptLossDaily --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
#   | * Best |    3/10 |      165 |     98   63    4 |        1.00% |    0.05453885 BTC   (54.54%) | 0 days 08:02:00 |    0.00442974 BTC   (13.41%) |     -41.371 |
#   | * Best |    7/10 |      101 |     56   42    3 |        0.73% |    0.02444518 BTC   (24.45%) | 0 days 13:08:00 |    0.00107122 BTC    (3.24%) |    -66.7687 |
# *    7/10:    101 trades. 56/42/3 Wins/Draws/Losses. Avg profit   0.73%. Median profit   0.13%. Total profit  0.02444518 BTC (  24.45%). Avg duration 13:08:00 min. Objective: -66.76866
# freqtrade hyperopt --hyperopt-loss OnlyProfitHyperOptLoss --spaces buy sell roi trailing stoploss --strategy Diamond -j 2 -e 10
# *    7/10:    117 trades. 74/41/2 Wins/Draws/Losses. Avg profit   1.91%. Median profit   1.50%. Total profit  0.07370921 BTC (  73.71%). Avg duration 9:26:00 min. Objective: -0.07371

# --- Do not remove these libs ---
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Diamond(IStrategy):
    # ###################### RESULT PLACE ######################
    #    Config: 5 x UNLIMITED STOCK costume pair list,
    #    hyperopt : 5000 x SortinoHyperOptLossDaily,
    #    34/5000: 297 trades. 136/156/5 Wins/Draws/Losses. Avg profit   0.49%. Median profit   0.00%. Total profit  45.84477237 USDT (  33.96Σ%). Avg duration 11:54:00 min. Objective: -46.50379

    # Buy hyperspace params:
    buy_params = {
        "buy_fast_key": "volume",
        "buy_horizontal_push": 10,
        "buy_slow_key": "volume",
        "buy_vertical_push": 1.197,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_fast_key": "open",
        "sell_horizontal_push": 5,
        "sell_slow_key": "high",
        "sell_vertical_push": 1.066,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.179,
        "21": 0.03,
        "61": 0.012,
        "179": 0

    }

    # Stoploss:
    stoploss = -0.342

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.011  # value loaded from strategy
    trailing_stop_positive_offset = 0.054  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    # timeframe
    timeframe = '5m'
    # #################### END OF RESULT PLACE ####################

    buy_vertical_push = DecimalParameter(0.5, 1.5, decimals=3, default=1, space='buy')
    buy_horizontal_push = IntParameter(0, 10, default=0, space='buy')
    buy_fast_key = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',
                                         #  you can not enable this lines befour you
                                         #  populate an indicator for them and set
                                         #  the same key name for it
                                         #  'ma_fast', 'ma_slow', {...}
                                         ], default='ma_fast', space='buy')
    buy_slow_key = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',
                                         #  'ma_fast', 'ma_slow', {...}
                                         ], default='ma_slow', space='buy')

    sell_vertical_push = DecimalParameter(0.5, 1.5, decimals=3, default=1, space='sell')
    sell_horizontal_push = IntParameter(0, 10, default=0, space='sell')
    sell_fast_key = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',
                                          #  'ma_fast', 'ma_slow', {...}
                                          ], default='ma_fast', space='sell')
    sell_slow_key = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',
                                          #  'ma_fast', 'ma_slow', {...}
                                          ], default='ma_slow', space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # you can add new indicators and enable them inside
        # hyperoptable categorical params on the top
        # dataframe['ma_fast'] = ta.SMA(dataframe, timeperiod=9)
        # dataframe['ma_slow'] = ta.SMA(dataframe, timeperiod=18)
        # dataframe['{...}'] = ta.{...}(dataframe, timeperiod={...})
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            qtpylib.crossed_above
                (
                dataframe[self.buy_fast_key.value].shift(self.buy_horizontal_push.value),
                dataframe[self.buy_slow_key.value] * self.buy_vertical_push.value
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            qtpylib.crossed_below
                (
                dataframe[self.sell_fast_key.value].shift(self.sell_horizontal_push.value),
                dataframe[self.sell_slow_key.value] * self.sell_vertical_push.value
            )
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        return dataframe
