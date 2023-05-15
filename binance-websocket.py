import asyncio
import json
import pandas as pd
import websockets
from datetime import datetime

async def save_order_book():
    url = 'wss://stream.binance.com:9443/ws/btcusdt@depth5'  # Replace with your desired symbol and depth level

    data = []

    async with websockets.connect(url) as websocket:
        while True:
            try:
                response = await websocket.recv()
                order_book = json.loads(response)

                # Append the order book data
                data.append(order_book)

                # Convert to pandas DataFrame every 10 minutes
                if len(data) % 300 == 0:  # 5 minutes = 5 * 60 seconds = 300 seconds
                    df = pd.DataFrame(data)
                    now = datetime.now()
                    df.to_json(f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}.json')
                    print("Order book data saved to JSON file.")
                    data = []  # Reset data

            except websockets.exceptions.ConnectionClosedOK:
                print("WebSocket connection closed.")
                break

            except Exception as e:
                print(f"An error occurred: {e}")

async def main():
    await save_order_book()

asyncio.run(main())