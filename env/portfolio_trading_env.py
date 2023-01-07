import gym
from gym import spaces
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from env.indicators import *


# trading fee:  min 2 USD/deal  0.01 USD/stock
def get_trading_fee(price, trading_shares):
    if trading_shares == 0:
        fee = 0
    else:
        fee = max(np.round(0.03*np.abs(trading_shares),2), 2)
    return fee


class PortfolioTradingEnv(gym.Env):

    def __init__(self, portfolio_stocks, initial_cash, start_day, end_day, out_csv_name='results/rewards', verbose=False):
        super().__init__()
        self.verbose = verbose
        self.max_share_per_trade = 1000
        self.reward_scale = 1e-4
        assert len(portfolio_stocks) > 0, f'Portfolio should include at least one stock'
        self.portfolio_stocks = portfolio_stocks
        self.n_stocks = len(self.portfolio_stocks)
        assert initial_cash > 0, f'Initial investment should be positive, but got {initial_cash}'
        self.initial_cash = initial_cash
        self.start_day = datetime.strptime(start_day, '%Y-%m-%d') 
        self.end_day = datetime.strptime(end_day, '%Y-%m-%d')

        self.stocks_history = pd.read_csv("data/prices-split-adjusted.csv")
        self.stocks_history = self.stocks_history[self.stocks_history["symbol"].isin(self.portfolio_stocks)]
        self.stocks_history.set_index(keys="date", drop=True, inplace=True)
        self.stocks_history.index = pd.to_datetime(self.stocks_history.index)
        self.stocks_history_dic = dict.fromkeys(self.portfolio_stocks)

        while(self.start_day not in self.stocks_history.index):
            self.start_day += timedelta(days=1)
        print(f"Experiment Start: {self.start_day}")

        for idx, stock in enumerate(self.portfolio_stocks):
            self.stocks_history_dic[stock] = self.stocks_history[self.stocks_history["symbol"]==stock]
            self.stocks_history_dic[stock] = self.stocks_history_dic[stock].join(RSI(stock,self.stocks_history)["RSI"], how="right")
            self.stocks_history_dic[stock] = self.stocks_history_dic[stock].join(MACD(stock,self.stocks_history)["MACD"], how="right")
            self.stocks_history_dic[stock].drop(["symbol","open","low","high"], axis=1, inplace=True)
            
            if idx == 0:
                self.day_list =  self.stocks_history_dic[stock].index
        
        self.n_days = 0
        for day in self.day_list:
            if day >= self.start_day and day <self.end_day:
                self.n_days += 1

            # print("MACD MAX", max(self.stocks_history_dic[stock]["MACD"]), min(self.stocks_history_dic[stock]["MACD"]))
            # print("RSI MAX", max(self.stocks_history_dic[stock]["RSI"]), min(self.stocks_history_dic[stock]["RSI"]))

        self.action_space = spaces.Box(low=-1, high=1,shape = (self.n_stocks,))
        # Shape 5*N+1 : [Current Balance] + [prices 1-N] + [owned shares 1-N] + [Volume 1-N]
        # + [MACD 1-N] + [RSI 1-N]
        self.observation_space = spaces.Box(low=-1, high=1, shape = (5*self.n_stocks+1,))

        self.state_max = [self.initial_cash*10] + \
            [ 200 for i in range(self.n_stocks)] + \
            [ 2000 for i in range(self.n_stocks)] + \
            [500000000 for i in range(self.n_stocks)] + \
            [5 for i in range(self.n_stocks)] + \
            [100 for i in range(self.n_stocks)]

        self.account = dict.fromkeys(['Cash'] + self.portfolio_stocks)
        self.total_asset = None
        self.reward = None
        self.total_reward = None

        self.current_day = None
        self.current_prices = dict.fromkeys(self.portfolio_stocks)
        self.state = None
        self.norm_state = None
        self.terminal = False

        self.action_summary = dict.fromkeys(self.portfolio_stocks)
        self.counter_list = ["buy", "sell", "no-action"]
        for key in self.action_summary:
            self.action_summary[key] = dict.fromkeys(self.counter_list)

        self.out_csv_name = out_csv_name
        self.metrics = []
        self.run = 0
        self.step_index = 0

    def _update_state(self):

        self.state = [self.account["Cash"]] + \
                    [ self.current_prices[stock] for stock in self.portfolio_stocks]  + \
                    [ self.account[stock] for stock in self.portfolio_stocks] + \
                    [ self.stocks_history_dic[stock].loc[self.current_day,"volume"] for stock in self.portfolio_stocks] + \
                    [ self.stocks_history_dic[stock].loc[self.current_day,"MACD"] for stock in self.portfolio_stocks]  + \
                    [ self.stocks_history_dic[stock].loc[self.current_day,"RSI"] for stock in self.portfolio_stocks]  
        return


    def _update_norm_state(self):
        self.norm_state = self.state
        for i in range(len(self.state)):
            if i>3*self.n_stocks and i<=4*self.n_stocks:
                self.norm_state[i] = self.norm_state[i]/self.state_max[i]
            else:
                self.norm_state[i] = (self.norm_state[i]/self.state_max[i] - 0.5) / 0.5
        return


    def _update_current_price(self):
        for stock in self.portfolio_stocks:
            self.current_prices[stock] = self.stocks_history_dic[stock].loc[self.current_day, "close"]
        return


    def reset(self):
        # this will return the initial state
        self.account['Cash'] = self.initial_cash
        for stock in self.portfolio_stocks:
            self.account[stock] = 0

        self.current_day = self.start_day
        self._update_current_price()
        
        self.total_asset = self.initial_cash
        self.reward = 0
        self.total_reward = 0
        self.terminal = False

        for stcok in self.action_summary.keys():
            for act in self.counter_list:
                self.action_summary[stcok][act] = 0

        self._update_state()
        self._update_norm_state()
        
        return np.array(self.norm_state).astype(np.float32)
    

    def _sell_stock(self, index, shares):
        stock = self.portfolio_stocks[index]
        available_shares = self.account[stock]
        shares = int(-shares)
        shares = min(available_shares, shares)

        self.account[stock] -= shares
        fee = get_trading_fee(self.current_prices[stock], shares)
        self.account["Cash"] += (shares*self.current_prices[stock] - fee)
        
        if fee > 0:
            self.action_summary[stock]["sell"] += 1
        else:
            self.action_summary[stock]["no-action"] += 1

        if shares != 0 and self.verbose:
            print(f"Sell {shares} {stock} @ Price {self.current_prices[stock]:.3f} Service fee $ {fee:.2f}.")

        return
    

    def _buy_stock(self, index, shares):
        stock = self.portfolio_stocks[index]
        available_shares = (self.account["Cash"]-2) // (self.current_prices[stock] + 0.03) # trading service fee
        shares = int(shares)
        shares = min(available_shares, shares)

        self.account[stock] += shares
        fee = get_trading_fee(self.current_prices[stock], shares)
        self.account["Cash"] -= (shares*self.current_prices[stock] + fee)

        if fee > 0:
            self.action_summary[stock]["buy"] += 1
        else:
            self.action_summary[stock]["no-action"] += 1
        if shares != 0 and self.verbose:
            print(f"Buy {shares} {stock} @ Price {self.current_prices[stock]:.3f} Service fee $ {fee:.2f}.")

        return

    def step(self, actions):
        # this will perform an action, return the next state, reward
        self.step_index +=1
        if self.verbose:
            print("\nBefore step:")
            self.render()

        actions = actions*self.max_share_per_trade
        argsort_actions = np.argsort(actions)
        
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        hold_index = [x for x in argsort_actions if (x not in sell_index) and (x not in buy_index)]
        for index in hold_index:
            self.action_summary[self.portfolio_stocks[index]]["no-action"] += 1

        if self.verbose:
            print("During step:")
        
        for index in sell_index:
            self._sell_stock(index, actions[index])

        for index in buy_index:
            self._buy_stock(index, actions[index])

        assert(self.account["Cash"]>=0)

        # update total asset value
        summation = self.account['Cash']
        for stock in self.portfolio_stocks:
            summation += self.account[stock]*self.current_prices[stock]
        self.reward = (summation - self.total_asset)*self.reward_scale
        self.total_reward += self.reward
        self.total_asset = summation

        info = self._compute_step_info()
        self.metrics.append(info)

        # update to next date
        self.current_day = self.day_list[np.where(self.day_list == self.current_day)[0] + 1][0]
        self._update_current_price()

        if self.current_day >= self.end_day:
            self.terminal = True

        self._update_state()
        self._update_norm_state()

        if self.terminal:
            self.save_csv(self.out_csv_name, self.run)
            self.run += 1
            self.metrics = []
            self.step_index = 0

        return np.array(self.norm_state).astype(np.float32), self.reward, self.terminal, {"total_reward": self.total_reward,"total_asset": self.total_asset, "last_account": self.account, "action_summary": self.action_summary}


    def _compute_step_info(self):
        return {
            'current_step': self.step_index,
            'last_account': self.account,
            'assets': self.total_asset,
            'action_summary': self.action_summary,
            'Total Reward': self.total_reward,
            'Reward': self.reward
        }

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)
    
    def render(self):
        message = " | ".join([ f"{stock} - "+ f"{np.round(self.account[stock],0):4}" for stock in self.portfolio_stocks])
        print(f"{str(self.current_day)[:10]}" + f" Account Status: " + f"Cash - {np.round(self.account['Cash'],2):8} | " + message)
        print(f"Total Assets Value: {np.round(self.total_asset,2)}\n")
        return

