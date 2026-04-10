import configparser
import os

import requests
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)

# Telegram credentials are loaded from an optional gitignored config file.
# Override with TG2_CONFIG_PATH; defaults to a path relative to TG2_HOME so
# the script is portable. Original training-time location was
# /home/srojas/tg2/Stuff/config.ini.
CONFIG_PATH = os.environ.get(
    "TG2_CONFIG_PATH",
    os.path.join(os.environ.get("TG2_HOME", "."), "Stuff", "config.ini"),
)
DEFAULT_FLAG_FILE = os.environ.get(
    "TG2_FLAG_FILE",
    os.path.join(os.environ.get("TG2_HOME", "."), "scripts", "flag.txt"),
)


def send_telegram_message(message):
    """Send a Telegram notification, silently no-op if not configured."""
    if not os.path.exists(CONFIG_PATH):
        return
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        apiToken = config.get("Telegram", "apiToken")
        chatID = config.get("Telegram", "chatID")
    except (configparser.Error, KeyError):
        return
    apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage"
    try:
        requests.post(apiURL, json={"chat_id": chatID, "text": message}, timeout=10)
        print(message)
    except requests.RequestException:
        pass


def read_flag_from_file(filename=DEFAULT_FLAG_FILE):
    try:
        with open(filename, "r") as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0


def write_flag_to_file(flag, filename=DEFAULT_FLAG_FILE):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as file:
        file.write(str(flag))


def get_gpu_memory_usage():
    # Obtener uso de memoria y memoria total de todas las tarjetas gráficas
    memory_usage_list = []
    for i in range(8):  # Asumiendo que tienes 8 GPUs
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        used_memory = mem_info.used / 1e6  # Convertir bytes a MB
        total_memory = mem_info.total / 1e6  # Convertir bytes a MB
        memory_usage_list.append((used_memory, total_memory))
    return memory_usage_list


def run():
    nvmlInit()
    old_flag = read_flag_from_file()
    memory_usages = get_gpu_memory_usage()
    new_flag = 0
    message = "[SERVER - CRATOS] [📊 GPU]\n\n"

    for i, (used, total) in enumerate(memory_usages):
        percentage = (used / total) * 100
        if percentage <= 25:
            message += f"🟢 GPU {i}: [{percentage:.1f} %] {used:.2f} / {total:.2f} MiB\n"
            new_flag += 4
        elif percentage > 25 and percentage <= 50:
            message += f"🟡 GPU {i}: [{percentage:.1f} %] {used:.2f} / {total:.2f} MiB\n"
            new_flag += 3
        elif percentage > 50 and percentage <= 75:
            message += f"🟠 GPU {i}: [{percentage:.1f} %] {used:.2f} / {total:.2f} MiB\n"
            new_flag += 2
        else:
            message += f"🔴 GPU {i}: [{percentage:.1f} %] {used:.2f} / {total:.2f} MiB\n"
            new_flag += 1

    if new_flag != old_flag:
        send_telegram_message(message)
    print(f"New flag: {new_flag}\nOld flag: {old_flag}")
    write_flag_to_file(new_flag)
    nvmlShutdown()


if __name__ == "__main__":
    run()
