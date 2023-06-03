import os

import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, threshold, size, max_position, adj_coeff):
        self.threshold = threshold
        self.size = size
        self.max_position = max_position
        self.adj_coeff = adj_coeff

    def get_signal(self, data, fee, cur_pos):
        # train_data = data.drop(["SampleY", "host_time", "sent_time"])
        # train_data = data.drop(["LagTruePrice"])

        # price_predicted = self.model.predict(data.values[:-4].reshape(1,-1))[0]
        # convert it to a NumPy array
        # arr = train_data.values

        # reshape the array
        # train_data_reshaped = arr.reshape((1, -1))

        # # price_predicted = self.model.predict(train_data.reshape(1,-1))[0]
        # price_predicted = self.model.predict(train_data_reshaped)[0]
        bid_0 = data["px_buy_1"]
        ask_0 = data["px_sell_1"]
        price_predicted = data["predicted"]
        # mid_price = (bid_0 + ask_0) / 2.0
        mid_price = data["TruePrice"]
        if mid_price + price_predicted - ask_0 - self.adj_coeff * cur_pos > self.threshold:
            signal = 1
        elif mid_price + price_predicted - bid_0 - self.adj_coeff * cur_pos < -self.threshold:
            signal = -1
        else:
            signal = 0 # не отсылаем ордер на текущем тике

        return signal, mid_price
