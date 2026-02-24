import asyncio
import csv
from datetime import datetime
from bleak import BleakClient, BleakScanner

# Standard Heart Rate Service characteristic UUID
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# CSV file setup
csv_file = open("polar_hr_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "heart_rate_bpm"])

def handle_hr(sender, data):
    """Callback for handling heart rate notifications"""
    
    heart_rate = data[1]
    timestamp = datetime.now().isoformat()
    print(f"{timestamp} | Heart Rate: {heart_rate} bpm")
    csv_writer.writerow([timestamp, heart_rate])

async def main():
    print("🔍 Scanning for devices...")
    devices = await BleakScanner.discover()

    target_device = None
    for d in devices:
        if "Polar H10" in d.name and "0188AC34" in d.name:
            target_device = d
            break

    if not target_device:
        print("⚠️ Polar H10 not found. Make sure it’s worn and electrodes are moist.")
        return

    print(f"✅ Found device: {target_device.name} ({target_device.address})")

    async with BleakClient(target_device.address) as client:
        print("📡 Connected. Listening for heart rate data...")
        await client.start_notify(HR_CHAR_UUID, handle_hr)
        await asyncio.sleep(60)  # Collect data for 60s
        await client.stop_notify(HR_CHAR_UUID)

    csv_file.close()
    print("✅ Data saved to polar_hr_data.csv")

asyncio.run(main())
