# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class GPTScalping(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Don't do anything if DataProvider is not available. 
        if not self.dp:
            return dataframe
        
        ## Overlap Studies
        # ------------------------------------   
        dataframe["ema_5"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ma_volume"] = ta.SMA(dataframe, timeperiod=50, price="volume")
        
        ## Volume Indicators
        # ------------------------------------   
        #VWAP
        dataframe["rolling_vwap"] = qtpylib.rolling_vwap(dataframe)        
        
        ## Momentum Indicators
        # ------------------------------------   
        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        
        # Bollinger Bands
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe["bb_upperband"] = bbands["upperband"]
        dataframe["bb_middleband"] = bbands["middleband"]
        dataframe["bb_lowerband"] = bbands["lowerband"]
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # Long Signal
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"]) &
                (dataframe["ema_50"] > dataframe["ema_200"]) &
                ## the 50 EMA is sloping upwards
                (dataframe["ema_50"] > dataframe["ema_50"].shift()) &
                # (dataframe["ema_50"] > dataframe["ema_50"].shift(2)) &
                # (dataframe["ema_50"] > dataframe["ema_50"].shift(3)) &
                ## up trend
                (dataframe["close"] > dataframe["rolling_vwap"]) &
                
                ## TODO:
                ## --------------------------------------------- ##
                ## Pullback to vwap or ema50: 
                ##      touch -> bounce/ break the vwap or ema50 then check price action
                ## --------------------------------------------- ##
                
                ## MACD line crosses above the signal line
                # (dataframe["macd"] > dataframe["macdsignal"]) &
                ## BB signal
                (
                    (
                        ## Price touch bb_lowerband
                        (dataframe["close"] > dataframe["bb_lowerband"]) &
                        (dataframe["low"] <= dataframe["bb_lowerband"])    
                    ) |
                    (
                        ## Price breaks bb_lowerband
                        (dataframe["open"] > dataframe["bb_lowerband"]) &
                        (dataframe["close"] <= dataframe["bb_lowerband"])
                    )
                ) &
                
                ## TODO:
                ## --------------------------------------------- ##
                ## Add conditions with volume profile or the average volume: 
                ##      volume profile:
                ##      volume is greater than MA or mean of volume
                ## --------------------------------------------- ##
                
                # (dataframe["volume"] > 0)  # Make sure Volume is greater than average of volume at 200 candles
                (dataframe["volume"] > dataframe["ma_volume"])  # Make sure Volume is greater than average of volume at 200 candles
            ),
            "enter_long"] = 1
        
        # Short Signal
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_200"]) &
                (dataframe["ema_50"] < dataframe["ema_200"]) &
                ## the 50 EMA is sloping downwards
                (dataframe["ema_50"] < dataframe["ema_50"].shift()) &
                # (dataframe["ema_50"] > dataframe["ema_50"].shift(2)) &
                # (dataframe["ema_50"] > dataframe["ema_50"].shift(3)) &
                ## downtrend
                (dataframe["close"] < dataframe["rolling_vwap"]) &
                
                ## TODO:
                ## --------------------------------------------- ##
                ## Pullback to vwap or ema50: 
                ##      touch -> bounce/ break the vwap or ema50 then check price action
                ## --------------------------------------------- ##
                
                ## MACD line crosses below the signal line
                # (dataframe["macd"] < dataframe["macdsignal"]) &
                ## BB signal
                (
                    (
                        ## Price touch bb_upperband
                        (dataframe["close"] < dataframe["bb_upperband"]) &
                        (dataframe["high"] >= dataframe["bb_upperband"])    
                    ) |
                    (
                        ## Price breaks bb_upperband
                        (dataframe["open"] < dataframe["bb_upperband"]) &
                        (dataframe["close"] >= dataframe["bb_upperband"])
                    )
                ) &
                
                ## TODO:
                ## --------------------------------------------- ##
                ## Add conditions with volume profile or the average volume: 
                ##      volume profile:
                ##      volume is greater than MA or mean of volume
                ## --------------------------------------------- ##
                
                # (dataframe["volume"] > 0)  # Make sure Volume is greater than average of volume at 200 candles
                (dataframe["volume"] > dataframe["ma_volume"])  # Make sure Volume is greater than average of volume at 200 candles
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # dataframe.loc[
        #     (
        #         ## 1. below the 200ema | previous swing low
        #         ## 2. If volatility is high, use a fixed 0.5% SL for BTC/USDT.
        #         ## 3. 1:1 / 1:2 / ema5 when trend is strong

        #         # (dataframe["close"] < dataframe["ema_200"]) &
        #         (qtpylib.crossed_above(dataframe["close"], dataframe["ema_200"])) &
        #         (dataframe["volume"] > 0)  # Make sure Volume is not 0
        #     ),
        #     "exit_long"] = 1

        # Short exit signal
        # dataframe.loc[
        #     (
        #         # (dataframe["close"] > dataframe["ema_200"]) &
        #         (qtpylib.crossed_below(dataframe["close"], dataframe["ema_200"])) &
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     'exit_short'] = 1
        
        ## Deactivated sell signal to allow the strategy to work correctly
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return 10.0