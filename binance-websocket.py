import asyncio
import json
import pandas as pd
import websockets
from datetime import datetime

async def save_order_book():
    # partial book depth stream
    # update speed: 500ms = 0.5s
    url = 'wss://fstream.binance.com/ws/btcusdt@depth10@500ms'  # Replace with your desired symbol and depth level
    kline_url = 'wss://fstream.binance.com/ws/btcusdt@kline_5m' # for kline data using at last step

    
    num = 469   # should be updated as the last number of the file + 1

    while True:
        data = []

        try:
            websocket = await websockets.connect(url)
            kline_websocket = await websockets.connect(kline_url)

            while True:
                response = await websocket.recv()
                order_book = json.loads(response)

                # Append the order book data
                data.append(order_book)

                # Convert to pandas DataFrame every 10 minutes
                if len(data) % 600 == 0:  # 5 minutes = 5 * 60 seconds = 5 * 60 * 2 = 600 (0.5 seconds)
                    kline_response = await kline_websocket.recv()
                    kline = json.loads(kline_response)
                    data.append(kline)

                    df = pd.DataFrame(data)
                    now = datetime.now()
                    df.to_json(f'./LOB/{num}.json')
                    print(f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute} Order book data saved to JSON file.")
                    data = []  # Reset data
                    num += 1

        except websockets.exceptions.ConnectionClosedOK:
            print("WebSocket connection closed. Reconnecting...")
            continue

        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket exception occurred: {e}")
            continue

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

async def main():
    await save_order_book()

asyncio.run(main())
