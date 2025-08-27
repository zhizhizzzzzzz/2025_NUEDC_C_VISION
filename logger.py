import datetime


def error(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[91m[{timestamp}][ERROR] {message}\033[0m")


def warn(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[93m[{timestamp}][WARN] {message}\033[0m")


def info(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {message}")


def debug(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[94m[{timestamp}][DEBUG] {message}\033[0m")
