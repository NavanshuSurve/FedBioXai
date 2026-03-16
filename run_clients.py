import threading
import time
import subprocess
import sys

subjects = ["S2","S3","S4","S5","S6","S7"]

def run_client(subject):
    subprocess.run([sys.executable, "client.py", subject])

threads = []

for s in subjects:
    t = threading.Thread(target=run_client, args=(s,))
    t.start()
    threads.append(t)
    time.sleep(2)

for t in threads:
    t.join()