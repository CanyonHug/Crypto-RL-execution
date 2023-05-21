import gym
from gym import spaces
import numpy as np

# initial_inventory = 3
# period = 600
# alpha = 10

class LimitOrderEnv(gym.Env):
    def __init__(self, initial_inventory = 3, period = 600, alpha = 10):
        super(LimitOrderEnv, self).__init__()
        self.initial_inventory = initial_inventory
        self.period = period
        self.alpha = alpha

        self.twap = initial_inventory / period

        self.elapsed_time = 0
        self.inventory = initial_inventory
        self.orderbook = []                 # top 10 bids asks prices
        self.bid_ask_spread = 0             # the value of bid/ask spread
        self.making_list = []               # list of bid makings      (price, qty)
        self.executed_list = []             # save the executed orders (price, qty)

        self.action_space = spaces.Discrete(36)     # 1 + 7 * 5 (5 actions for quantity, 7 actions for price)
        self.observation_space = spaces.Box(        # state : [elapsed time, remaining inventory, LOB, bid/ask spread] (1 + 1 + 2*10*2 + 1 = 43) 
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32
        )

    def reset(self, first_orderbook):
        self.elapsed_time = 0
        self.inventory = self.initial_inventory
        self.orderbook = first_orderbook
        self.bid_ask_spread = 0
        self.making_list = []
        self.executed_list = []

        return self._get_state()

    # action : qty (6) = 0, 0.4 TWAP ~ 2 TWAP, price (7) : best bid price + 0.2 alpha ~ best bid - alpha (alpha = 10)
    def step(self, action, next_orderbook, low_price):      # next_orderbook = data['b'][t+1] + data['a'][t+1]

        if action != 0:
            temp = action - 1
            price = (temp // 7 + 1) * 0.4 * (self.twap)
            qty = float(self.orderbook[0][0]) + (2 - (temp % 7) * 0.2 * self.alpha)

            self.making_list.append((price, qty))
            self.making_list = sorted(self.making_list, key=lambda x: (-x[0], x[1])) # As it's bid making, decreasing order sorting
            
        self.elapsed_time += 1

        if self.elapsed_time == self.period:    # at final step : sell remaining inventory
            if self.inventory > 0:
                while self.inventory > 0:
                    for i in range(10, 20):     # orderbook[10~20] : ask
                        if self.inventory > float(self.orderbook[i][1]):
                            self.executed_list.append(
                                (float(self.orderbook[i][0]), float(self.orderbook[i][1]))
                            )
                            self.inventory -= float(self.orderbook[i][1])
                        elif 0 < self.inventory < float(self.orderbook[i][1]):
                            self.executed_list.append(
                                (float(self.orderbook[i][0]), self.inventory)
                            )
                            self.inventory = 0

            reward = self._get_reward(low_price)
            done = True
        else:
            # update the making_li based on current LOB and next LOB
            if self.inventory > 0:
                for i in self.making_list:
                    if i[0] > float(next_orderbook[0][0]):         # next order book?
                        if self.inventory >= i[1]: 
                            self.executed_list.append(i)
                            self.making_list.remove(i)
                            self.inventory -= i[1]
                        else:
                            i[1] = self.inventory
                            self.executed_list.append(i)
                            self.making_list = []
                            self.inventory = 0
                    else:
                        break

            reward = self._get_reward(low_price)
            done = False
            self.orderbook = next_orderbook

        self.bid_ask_spread = float(self.orderbook[10][0]) - float(self.orderbook[0][0])

        return self._get_state(), reward, done, {}

    def _get_state(self):
        state = [self.elapsed_time, self.inventory]
        state = state + self._flatten_orderbook() + [self.bid_ask_spread]
        state = np.array(state, dtype=np.float32)
        return state

    def _flatten_orderbook(self):
        orderbook_flattened = []
        for bid_ask_prices in self.orderbook:       # orderbook = data['b'][t] + data['a'][t]
            for price_qty in bid_ask_prices:        # [price, qty]
                orderbook_flattened.extend([float(price_qty)])
        return orderbook_flattened

    def _get_reward(self, low_price):
        executed_price = self._calculate_executed_price()
        reward = 0
        reward -= float(abs(executed_price - low_price))

        if self.elapsed_time == self.period:
            reward *= 1000
        
        return reward

    def _calculate_executed_price(self):
        temp = 0
        qty = 0
        for price, quantity in self.executed_list:      # executed_li : list of executed(price, qty)
            temp += price * quantity
            qty += quantity
        if qty > 0:
            executed_price = temp / qty
        else:
            executed_price = 0
        return executed_price