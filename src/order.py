class Order:
    def __init__(
        self,
        order_type,
        action,
        price,
        volume,
        max_volume,
        time_received,
        stop_loss: None,
        take_profit: None,
    ):
        self.order_type = order_type
        self.action = action
        self.price = price
        self.volume = volume
        self.max_volume = max_volume
        self.time_received = time_received
        self.stop_loss = stop_loss
        self.take_profit = take_profit


    def __str__(self):
        return f'Order: \n order_type={self.order_type}\n action={self.action} \n price={self.price} \n volume={self.volume} \n time_received={self.time_received}'
