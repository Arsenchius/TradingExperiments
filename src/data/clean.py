import pandas as pd
import numpy as np


def read_data(df: pd.DataFrame) -> pd.DataFrame:
    # df = pd.read_csv(path_to_data, sep="|")
    # df = df.iloc[10:-10]
    df = df.reset_index().drop("index", axis=1)
    df.adapter_time = pd.to_datetime(df.adapter_time)
    df.host_time = pd.to_datetime(df.host_time)
    df.sent_time = pd.to_datetime(df.sent_time)
    df.set_index("adapter_time", inplace=True)

    return df


# def cumAGreaterBtwopfive(A, B, C):
#     a = A > B * 2.5
#     b = a & a.shift(1) & a.shift(2)
#     c = b * C
#     return c


# def delay(A, n):
#     temp = A.shift(n)
#     temp = temp.fillna(0)
#     return temp


# @jit(nopython = True)
# def pass_resistance_ask(thres_arr, trueprice_arr):
#     looked1, looked2, looked3 = 0, 0, 0
#     toReturn = np.zeros(len(thres_arr))
#     for i in range(len(thres_arr)):
#         thres = thres_arr[i]
#         price = trueprice_arr[i]
#         # print('threshold is ', max(looked1, looked2, looked3))
#         # print('price is ', price)
#         # print('thres is ', thres)
#         # print('------------------')
#         if thres != 0:
#             looked2, looked3 = looked1, looked2
#             looked1 = thres
#         elif price >= max(looked1, looked2, looked3):
#             continue
#         elif price < max(looked1, looked2, looked3):
#             toReturn[i] = 1
#             looked1, looked2, looked3 = 0, 0, 0
#     return toReturn


# @jit(nopython = True)
# def pass_resistance_bid(thres_arr, trueprice_arr):
#     looked1, looked2, looked3 = 10000, 10000, 10000
#     toReturn = np.zeros(len(thres_arr))
#     for i in range(len(thres_arr)):
#         thres = thres_arr[i]
#         price = trueprice_arr[i]
#         # print('threshold is ', min(looked1, looked2, looked3))
#         # print('price is ', price)
#         # print('thres is ', thres)
#         # print('------------------')
#         if thres != 0:
#             looked2, looked3 = looked1, looked2
#             looked1 = thres
#         elif price <= min(looked1, looked2, looked3):
#             continue
#         elif price > min(looked1, looked2, looked3):
#             toReturn[i] = 1
#             looked1, looked2, looked3 = 10000, 10000, 10000
#     return toReturn


"""
Multi-Level Order-Flow Imbalance in a Limit Order Book by Ken et al.
"""


def deltaW(BidPrice, BidVolume):
    case1 = BidPrice > BidPrice.shift(1)
    case2 = BidPrice == BidPrice.shift(1)
    case3 = BidPrice < BidPrice.shift(1)
    toReturn = (
        case1 * BidVolume
        + case2 * (BidVolume - BidVolume.shift(1).fillna(0))
        + case3 * -1 * BidVolume.shift(1)
    )
    return toReturn


def deltaV(AskPrice, AskVolume):
    case1 = AskPrice > AskPrice.shift(1)
    case2 = AskPrice == AskPrice.shift(1)
    case3 = AskPrice < AskPrice.shift(1)
    toReturn = (
        case1 * -1 * AskVolume
        + case2 * (AskVolume - AskVolume.shift(1).fillna(0))
        + case3 * AskVolume.shift(1)
    )
    return toReturn


# def deltaW1(BidPrice, BidVolume):
#     case1 = BidPrice < BidPrice.shift(1)
#     case2 = BidPrice == BidPrice.shift(1)
#     case3 = BidPrice < BidPrice.shift(1)
#     toReturn = (
#         case1 * -1 * BidVolume
#         + case2 * (BidVolume - BidVolume.shift(1).fillna(0))
#         + case3 * -1 * BidVolume.shift(1)
#     )
#     return toReturn


# def deltaV1(AskPrice, AskVolume):
#     case1 = AskPrice < AskPrice.shift(1)
#     case2 = AskPrice == AskPrice.shift(1)
#     case3 = AskPrice < AskPrice.shift(1)
#     toReturn = (
#         case1 * AskVolume
#         + case2 * (AskVolume - AskVolume.shift(1).fillna(0))
#         + case3 * AskVolume.shift(1)
#     )
#     return toReturn


def OFI(BidPrice, BidVolume, AskPrice, AskVolume):
    case1 = BidPrice >= BidPrice.shift(1)
    case2 = BidPrice <= BidPrice.shift(1)
    case3 = AskPrice >= AskPrice.shift(1)
    case4 = AskPrice <= AskPrice.shift(1)
    toReturn = (
        case1 * BidVolume
        - case2 * BidVolume.shift(1).fillna(0)
        - case4 * AskVolume
        - case3 * AskVolume.shift(1)
    )
    return toReturn



# def Press(BidPrice, AskPrice, BidVolume, AskVolume, TruePrice):
#     case1 = BidPrice - TruePrice
#     case2 = AskPrice - TruePrice
#     p = case1 * BidVolume - case2 * AskVolume
#     return p


# def VolumeRatio(tick, df):
#     toReturn = np.zeros(len(df["Volume"]))
#     for i in range(int(np.floor(len(df["Volume"]) / tick))):
#         vol = df["TrueVolume"][i * tick : (i + 1) * tick]
#         mr = df["MidReturn"][i * tick : (i + 1) * tick]
#         vr = (sum((mr > 0) * vol) + np.mean(vol) / 2) / (
#             sum((mr < 0) * vol + np.mean(vol) / 2)
#         )
#         toReturn[(i + 1) * tick : (i + 2) * tick] = vr
#     return toReturn


def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    # df = data.copy()
    # df["time"] = df.index
    # df = df[~df["time"].duplicated(keep="first")]

    for i in range(5):
        df[f"ImBalance{i+1}"] = ((df[f"amt_buy_{i+1}"]) - (df[f"amt_sell_{i+1}"])) / (
            (df[f"amt_buy_{i+1}"]) + (df[f"amt_sell_{i+1}"])
        )
    df["ImBalance10"] = ((df["amt_buy_10"]) - (df["amt_sell_10"])) / (
        (df["amt_buy_10"]) + (df["amt_sell_10"])
    )

    df["ImBalance1v"] = df["ImBalance1"] - df["ImBalance1"].shift(1).fillna(0)
    df["ImBalance2v"] = df["ImBalance2"] - df["ImBalance2"].shift(1).fillna(0)
    df["ImBalance3v"] = df["ImBalance3"] - df["ImBalance3"].shift(1).fillna(0)

    df["OFI1"] = (
        deltaW(df["px_buy_1"], df["amt_buy_1"])
        - deltaV(df["px_sell_1"], df["amt_sell_1"])
    ).fillna(0)
    df["OFI2"] = (
        deltaW(df["px_buy_2"], df["amt_buy_2"])
        - deltaV(df["px_sell_2"], df["amt_sell_2"])
    ).fillna(0)
    df["OFI3"] = (
        deltaW(df["px_buy_3"], df["amt_buy_3"])
        - deltaV(df["px_sell_3"], df["amt_sell_3"])
    ).fillna(0)

    # df["OFIb"] = (
    #     deltaW1(df["px_buy_1"], df["amt_buy_1"])
    #     - deltaV1(df["px_sell_1"], df["amt_sell_1"])
    # ).fillna(0)
    # df["OFI2b"] = (
    #     deltaW1(df["px_buy_2"], df["amt_buy_2"])
    #     - deltaV1(df["px_sell_2"], df["amt_sell_2"])
    # ).fillna(0)
    # df["OFI3b"] = (
    #     deltaW1(df["px_buy_3"], df["amt_buy_3"])
    #     - deltaV1(df["px_sell_3"], df["amt_sell_3"])
    # ).fillna(0)

    df["OFI4"] = (
        deltaW(df["px_buy_4"], df["amt_buy_4"])
        - deltaV(df["px_sell_4"], df["amt_sell_4"])
    ).fillna(0)
    df["OFI5"] = (
        deltaW(df["px_buy_5"], df["amt_buy_5"])
        - deltaV(df["px_sell_5"], df["amt_sell_5"])
    ).fillna(0)
    df["OFI10"] = (
        deltaW(df["px_buy_10"], df["amt_buy_10"])
        - deltaV(df["px_sell_10"], df["amt_sell_10"])
    ).fillna(0)
    df["OFI1v"] = df["OFI1"] - df["OFI1"].shift(1).fillna(0)

    df["OFIV1"] = (
        (
            deltaW(df["px_buy_1"], df["amt_buy_1"])
            - deltaV(df["px_sell_1"], df["amt_sell_1"])
        ).fillna(0)
    ) / (
        np.abs(deltaW(df["px_buy_1"], df["amt_buy_1"]))
        + np.abs(deltaV(df["px_sell_1"], df["amt_sell_1"]))
    )
    df["OFIV2"] = (
        (
            deltaW(df["px_buy_2"], df["amt_buy_2"])
            - deltaV(df["px_sell_2"], df["amt_sell_2"])
        ).fillna(0)
    ) / (
        np.abs(deltaW(df["px_buy_2"], df["amt_buy_2"]))
        + np.abs(deltaV(df["px_sell_2"], df["amt_sell_2"]))
    )
    df["OFIV3"] = (
        (
            deltaW(df["px_buy_3"], df["amt_buy_3"])
            - deltaV(df["px_sell_3"], df["amt_sell_3"])
        ).fillna(0)
    ) / (
        np.abs(deltaW(df["px_buy_3"], df["amt_buy_3"]))
        + np.abs(deltaV(df["px_sell_3"], df["amt_sell_3"]))
    )

    s5b = (df["px_buy_5"] - df["px_buy_1"]) / (df["amt_buy_5"] - df["amt_buy_1"])
    s5a = (df["px_sell_5"] - df["px_sell_1"]) / (df["amt_sell_5"] - df["amt_sell_1"])
    df["slope5"] = (s5b - s5a) / (np.abs(s5b) + np.abs(s5a))

    s10b = (df["px_buy_10"] - df["px_buy_1"]) / (df["amt_buy_10"] - df["amt_buy_1"])
    s10a = (df["px_sell_10"] - df["px_sell_1"]) / (df["amt_sell_10"] - df["amt_sell_1"])
    df["slope10"] = (s10b - s10a) / (np.abs(s10b) + np.abs(s10a))

    df["TruePrice"] = df[["px_buy_1", "px_sell_1"]].mean(axis=1)
    # df["LagTruePrice"] = (df['TruePrice'] - df['TruePrice'].shift(1)).fillna(0)
    df["LagTruePrice"] = (df['TruePrice'].shift(-1) - df['TruePrice']).fillna(0)
    df["MeanRev2"] = (df["TruePrice"] - df["TruePrice"].rolling(20).mean()).fillna(0)
    df["MeanRev1"] = (df["TruePrice"] - df["TruePrice"].rolling(10).mean()).fillna(0)

    df["MidReturn"] = df["TruePrice"].pct_change(1).shift(-1).fillna(0)
    df["Momentum1"] = df["MidReturn"].shift(1)
    df["Momentum2"] = df["MidReturn"].shift(2)
    df["Momentum3"] = df["MidReturn"].shift(3)

    # additional features
    # thres_arrA = cumAGreaterBtwopfive(
    #     df["amt_sell_1"], df["amt_buy_1"], df["TruePrice"]
    # )
    # df["overA"] = pass_resistance_ask(thres_arrA.to_numpy(), df["TruePrice"].to_numpy())
    # thres_arrB = cumAGreaterBtwopfive(
    #     df["amt_buy_1"], df["amt_sell_1"], df["TruePrice"]
    # )
    # df["overB"] = pass_resistance_bid(thres_arrB.to_numpy(), df["TruePrice"].to_numpy())

    df["Volume"] = (
        df["amt_buy_1"]
        + df["amt_buy_2"]
        + df["amt_buy_3"]
        + df["amt_buy_4"]
        + df["amt_buy_5"]
        + df["amt_sell_1"]
        + df["amt_sell_2"]
        + df["amt_sell_3"]
        + df["amt_sell_4"]
        + df["amt_sell_5"]
    )
    df["TrueVolume"] = df["Volume"] - df["Volume"].shift(1)
    df["TrueVolume"] = df["TrueVolume"].fillna(0)
    # df["Mid20VWAP"] = (df["TrueVolume"] * df["TruePrice"]).rolling(20).sum() / df[
    #     "TrueVolume"
    # ].rolling(20).sum()
    # df["SampleY"] = (df["Mid20VWAP"].shift(-20) - df["TruePrice"]) / df["TruePrice"]
    df = df.drop(["host_time", "sent_time"], axis=1)
    return df
