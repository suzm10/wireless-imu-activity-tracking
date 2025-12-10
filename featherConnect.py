import asyncio
from bleak import BleakClient, BleakScanner
import bleak

# Initializing variables
UART_RX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
FILENAME_BASE="insert_random_filename_here"

# Handler for receiving data from the Feather IMU/Microcontroller
def handle_rx(sender, data):
    try:
        # Parse the incoming data
        text = data.decode()
        text = text.strip()

        # Extract the outputted component values (was not ultimately utilized in this data receiving file)
        number, time, velostat, ax, ay, az, gx, gy, gz, mx, my, mz = text.split(',')
    except Exception:
        print(f"text:  {text} could not be decoded")
    try:
        # Write the data to a CSV file
        with open(FILENAME_BASE + ".csv", "a") as f:
            f.write(text + "\n")

        # Print it to stdout as well for confirmation during testing
        print(text)
    except Exception:
        # In case something goes horribly wrong with file writing
        print("Failed to write to CSV: ", text)
        pass

# Establishes a connection to Feather node and runs handler_rx callback to receive data
async def connect_and_listen(address):
    while True:
        try:
            async with BleakClient(address) as client:
                # Start the notification handler to begin receiving data
                print(f"Awaiting data transfer from {address}...")
                await client.start_notify(UART_RX_UUID, handle_rx)

                # Keep the connection alive
                while client.is_connected:
                    await asyncio.sleep(1)
        except Exception as e:
            # Something went wrong, just try reconnecting...
            print(f"Disconnected or failed to connect: {e}")
            print("Retrying in 3 seconds...")
            await asyncio.sleep(3)

async def main():
    print("Scanning for devices...")

    # Initializing variables
    connectedDevices = []
    tasks = []

    # Note: This code was originally set up to listen to multiple devices at once
    # for faster data collection. This never ended up being necessary for the project.
    expectedNumberOfDevices = 1
    
    # Continue scanning for devices until we reach the target number
    while expectedNumberOfDevices > len(connectedDevices):

        # Get the list of available devices
        devices = await BleakScanner.discover()

        # For each possible connection, check if it's one of our devices and connect to it if so
        for d in devices:
            if d and d.name and d.name.startswith("Feather") and d not in connectedDevices:
                print(f"Connecting to new device: {d.name} ({d.address})")

                # Note the connected device to avoid connecting to it again
                connectedDevices.append(d)

                # Create a task to run connect_and_listen concurrently
                task = asyncio.create_task(connect_and_listen(d.address))
                tasks.append(task)

        # Continue scanning if we haven't reached the target number of devices yet        
        if len(connectedDevices) < expectedNumberOfDevices:
            print("Still awaiting connections, scanning again...")
            await asyncio.sleep(1)
    
    print(f"Connected to all {len(connectedDevices)} devices. Listening for data...")

    # Wait for all tasks to complete (though in reality we just use an Interrupt signal to stop the program)
    await asyncio.gather(*tasks)

asyncio.run(main())
