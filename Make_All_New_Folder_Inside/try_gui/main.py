import eel
import threading
import time
import random

eel.init('web')

running = False
connected = False


def monitor_loop():
    global running, connected
    while running and connected:
        bpm = random.randint(55, 1100)
        eel.updateHR(bpm)
        time.sleep(5)


@eel.expose
def on_cmd(cmd):
    global running, connected
    if cmd == 'connect':
        connected = True
        print("✅ Connected")
    elif cmd == 'disconnect':
        connected = False
        running = False
        print("❌ Disconnected")
    elif cmd == 'start' and connected and not running:
        running = True
        threading.Thread(target=monitor_loop, daemon=True).start()
        print("▶️ Monitoring started")
    elif cmd == 'stop' and running:
        running = False
        print("⏸️ Monitoring stopped")


if __name__ == '__main__':
    eel.start('index.html', size=(640, 400))
