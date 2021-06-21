import time
import threading
import psutil
# gives a single float value
import sys
from datetime import datetime


def count_processes():
    count = 0
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()
            processID = proc.pid


            #print(processName, ' ::: ', processID)
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count


class Log_init:

    def __init__(self):
        log_file_path = "./log_statistics.csv"
        import os
        print(os.path.abspath(log_file_path))
        self.log_file = open(log_file_path, "w")

        print("utc_time,swap_percent,virtual_percent,cpu_percent,process_count", file=self.log_file)

    def log(self):

        while True:
            time.sleep(0.5)
            print("{},{},{},{},{}".format(datetime.utcnow(),
                                          psutil.swap_memory().percent
                                          ,psutil.virtual_memory().percent
                                          ,psutil.cpu_percent()
                                          ,count_processes()),file=self.log_file)


if __name__ == "__main__":
    Log_init().log()
