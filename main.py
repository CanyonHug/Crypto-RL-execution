import json
import matplotlib.pyplot as plt
from DQN import *
from gym_env import *
import requests

# Function to load the order book data
def load_orderbook_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Define the training parameters
num_episodes = 400
episode_length = 600

# Create the environment and agent
env = LimitOrderEnv()
agent = DoubleDQNAgent(state_size=43, action_size=36, batch_size=16, gamma=0.9, replay_capacity= 300, epsilon=0.3, epsilon_decay=0.8)

# Lists to store the results
low_prices = []
high_prices = []
executed_prices = []

# Training loop
for episode in range(num_episodes):
    # Define the file path for the order book data
    orderbook_file = f'./LOB/{episode + 1}.json'

    # Load the order book data for the episode
    orderbook_data = load_orderbook_data(orderbook_file)

    # Reset the environment
    state = env.reset(orderbook_data['b']["0"] + orderbook_data['a']["0"])
    total_reward = 0

    # Episode-specific variables
    low_price = float('inf')
    high_price = float('-inf')

    # There is an error with kline data of websocket LOB so I get kline data from binanace API.
    base_url = 'https://fapi.binance.com'

    def get_kline_data(symbol, interval, start_time, end_time=None, limit=None):
        endpoint = '/fapi/v1/klines'
        url = base_url + endpoint

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        return data
    
    symbol = 'btcusdt'
    interval = '5m'
    start_time = orderbook_data['E']['0']

    kline = get_kline_data(symbol, interval, start_time)

    # Update episode-specific variables
    low_price = float(kline[0][3])
    high_price = float(kline[0][2])


    for t in range(episode_length):    
        # Select an action
        action = agent.select_action(state)


        # Take a step in the environment
        if t != episode_length - 1:
            next_state, reward, done, _ = env.step(action, orderbook_data['b'][str(t + 1)] + orderbook_data['a'][str(t + 1)], low_price)
        else:
            next_state, reward, done, _ = env.step(action, None, low_price)


        # Save the results
        if done:
            low_prices.append(low_price)
            high_prices.append(high_price)
            executed_price = env._calculate_executed_price()
            executed_prices.append(executed_price)
        
        # Store the transition in the replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train the agent
        agent.train()

        # Update the current state
        state = next_state
        total_reward += reward

        if t % 10 == 0:
            print(total_reward)

        if done:
            break

    # Print the episode results
    print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward}, High price: {high_price}, Low price: {low_price}, Executed price: {executed_price}")

    # Save the model parameters every 100 episodes
    if (episode + 1) % 100 == 0:
        agent.save_model(f'dqn_model_{episode + 1}.pt')

# Plot the learning progress
plt.plot(low_prices, label='Low Prices')
plt.plot(high_prices, label='High Prices')
plt.plot(executed_prices, label='Executed Prices')
plt.xlabel('Episode')
plt.ylabel('Price')
plt.title('Learning Progress')
plt.legend()
plt.show()

# Load the trained model parameters
agent.load_model('dqn_model_final.pt')