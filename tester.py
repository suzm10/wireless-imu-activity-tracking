import asyncio
from bleak import BleakClient, BleakScanner
import bleak

#UART_RX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" # RX characteristic UUID - shouldn't need to change?
UART_RX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
FILENAME_BASE="timed_velostat_35_"

def handle_rx(sender, data):
    try:
        text = data.decode()
        text = text.strip()
        number, time, velostat, ax, ay, az, gx, gy, gz, mx, my, mz = text.split(',')
        #time, val = text.split(',')
    except Exception:
        print(f"text:  {text} could not be decoded")
    try:
        #with open(FILENAME_BASE + f"{number}" + ".csv", "a") as f:
        #    f.write(text + "\n")
        with open(FILENAME_BASE + ".csv", "a") as f:
            f.write(text + "\n")
        print(text)
    except Exception:
        print("FAIL: ", text)
        pass

async def connect_and_listen(address):
    while True:
        try:
            async with BleakClient(address) as client:
                print(f"Connected to {address}")
                await client.start_notify(UART_RX_UUID, handle_rx)
                while client.is_connected:
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Disconnected or failed to connect: {e}")
            print("Retrying in 3 seconds...")
            await asyncio.sleep(3)

async def main():
    print("Scanning for devices...")
    connectedDevices = []
    tasks = []
    expectedNumberOfDevices = 1
    
    while expectedNumberOfDevices > len(connectedDevices):
        devices = await BleakScanner.discover()
        for d in devices:
            if d and d.name and d.name.startswith("Feather") and d not in connectedDevices:
                print(f"Connecting to new device: {d.name} ({d.address})")
                connectedDevices.append(d)
                # Create a task to run connect_and_listen concurrently
                task = asyncio.create_task(connect_and_listen(d.address))
                tasks.append(task)
        if len(connectedDevices) < expectedNumberOfDevices:
            print("Still awaiting connections, scanning again...")
            await asyncio.sleep(1)
    
    print(f"Connected to all {len(connectedDevices)} devices. Listening for data...")
    # Wait for all tasks to complete (they won't unless there's an unrecoverable error)
    await asyncio.gather(*tasks)

asyncio.run(main())
