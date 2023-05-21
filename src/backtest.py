import pandas as pd
import numpy as np
from tqdm import tqdm

from strategy import Strategy
from order import Order



class Backtest:

    def __init__(self, strategy):
        # self.data = data
        self.strategy = strategy
        self.closed_orders = []
        self.open_orders = []
        self.cur_pos = 0

    def dump_results(self, path_to_folder: str):
        chunk_size = 50000
        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)
        for i in range(0, len(self.closed_orders), chunk_size):
            full_path = os.path.join(path_to_folder, f"orders_{int(i/chunk_size) +1}.json")
            with open(full_path, "w") as f:
                chunk = orders[i : i + chunk_size]
                json.dump(chunk, f)

    def summary(self, fee):
        # while self.open_orders:
        #     current_order = self.open_orders.pop()
        #     index = 1
        #     if current_order.action == "sell":
        #         # while current_order.volume >
        #         pass
        #     elif current_order.action == "buy":
        #         pass
        results = {
            "total_trades": len(self.closed_orders)
        }
        buy_trades = 0
        buy_trades_amt = 0
        sell_trades = 0
        sell_trades_amt = 0
        returns = 0
        side = 1
        for i in range(len(self.closed_orders)):
            order = self.closed_orders[i]
            if order.action == "buy":
                buy_trades += 1
                buy_trades_amt += order.volume
                side = 1
            elif order.action == "sell":
                sell_trades += 1
                sell_trades_amt += order.volume
                side = -1
            returns += order.volume / order.price * (side - fee)

        results["buy_orders_cnt"] = buy_trades
        results["buy_orders_amt"] = buy_trades_amt
        results["sell_orders_cnt"] = sell_trades
        results["sell_orders_amt"] = sell_trades_amt
        results["returns"] = returns

        return results

    def run_backtest(self, data, latency, fee):
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            # last_received_time = float(row["adapter_time"].timestamp())
            last_received_time = float(index.timestamp())
            if self.open_orders:
                current_order = self.open_orders.pop()
                if current_order.time_received + latency <= last_received_time:
                    if current_order.action == "sell":
                        # for i in range(25):
                        #     if row[f"px_buy_{i+1}"] >= current_order.price and row[f"amt_buy_{i+1}"] >= current_order.volume:
                        #         closed_orders.append(current_order)
                        if row["amt_buy_1"] >= current_order.volume:
                            current_order.price = row["px_buy_1"]
                            self.closed_orders.append(current_order)
                            self.cur_pos -= current_order.volume
                    elif current_order.action == "buy":
                        # for i in range(25):
                        #     if row[f"px_sell_{i+1}"] <= current_order.price and row[f"amt_sell_{i+1}"] >= current_order.volume:
                        #         closed_orders.append(current_order)
                        if row["amt_sell_1"] >= current_order.volume:
                            current_order.price = row["px_sell_1"]
                            self.closed_orders.append(current_order)
                            self.cur_pos += current_order.volume
            else:
                # signal, price = self.strategy.get_signal(self.data.loc[index], fee, cur_pos)
                signal, price = self.strategy.get_signal(row, fee, self.cur_pos)
                if signal == 1:
                    order = Order(order_type="market", action="buy", price=price, volume=self.strategy.size, max_volume=self.strategy.max_position, time_received=last_received_time, stop_loss=None, take_profit= None)
                    self.open_orders.append(order)
                elif signal == 0:
                    continue
                elif signal == -1:
                    order = Order(order_type="market", action="sell", price=price, volume=self.strategy.size, max_volume=self.strategy.max_position, time_received=last_received_time, stop_loss=None, take_profit=None)
                    self.open_orders.append(order)

        # results = self.summary(fee)

        # return results

    def plot_results(self):
        return
