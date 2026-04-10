import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import log10, sqrt

COVER_FOLDER_PATH = os.environ.get("TG2_DATA_ROOT", "/HDDmedia/srojas") + "/input-data"
STEGO_FOLDER_PATH = os.environ.get("TG2_DATA_ROOT", "/HDDmedia/srojas") + "/output-dwt"
SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
METHOD = "DWT"
IMAGES = ["IMAGE000001", "IMAGE000002", "IMAGE000003", "IMAGE000004", "IMAGE000005"]


def extract_number_from_filename(filename):
    global METHOD
    pattern = r"_{METHOD}_(\d+).png$".format(METHOD=METHOD)
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        return None


def psnr(cover, stego):
    mse = np.mean((cover - stego) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse))
    return psnr_value


def run():
    psnr_mean = []
    default_x_ticks = []

    for image in IMAGES:
        cover_path = os.path.join(COVER_FOLDER_PATH, f"{image}.png")
        stego_files_name = os.listdir(STEGO_FOLDER_PATH)

        for stego_file_name in stego_files_name:
            if stego_file_name.startswith(image):
                print(f"Procesando archivo estego: {stego_file_name}")
                char_value = extract_number_from_filename(stego_file_name)
                if char_value is not None:
                    char_value = int(char_value)
                    print(f"Valor extraído del nombre del archivo: {char_value}")

                    stego_path = os.path.join(STEGO_FOLDER_PATH, stego_file_name)
                    print(f"Ruta de la imagen estego: {stego_path}")

                    cover = cv2.imread(cover_path)
                    stego = cv2.imread(stego_path)
                    if cover is None or stego is None:
                        print(
                            f"No se pudo leer la imagen de portada o la imagen estego para {stego_file_name}"
                        )
                        continue

                    value = psnr(cover, stego)

                    psnr_mean.append(value)
                    default_x_ticks.append(char_value)

    # Verifica si las listas tienen el mismo tamaño
    print(
        f"Tamaño de default_x_ticks: {len(default_x_ticks)}, Tamaño de psnr_mean: {len(psnr_mean)}"
    )

    # Ordenar los valores de acuerdo al eje x (default_x_ticks)
    sorted_indices = np.argsort(default_x_ticks)
    default_x_ticks_sorted = np.array(default_x_ticks)[sorted_indices]
    psnr_mean_sorted = np.array(psnr_mean)[sorted_indices]

    print(f"Valores de X ordenados: {default_x_ticks_sorted}")
    print(f"Valores de PSNR ordenados: {psnr_mean_sorted}")

    # Guardar y cerrar la figura
    file_path = os.path.join(os.environ.get("TG2_HOME", "/home/srojas/tg2") + "/Resultados/", f"{SCRIPT_NAME}.pdf")
    plt.figure(figsize=(10, 7))
    plt.plot(default_x_ticks_sorted, psnr_mean_sorted)
    plt.title(f"PSNR del Método: {METHOD}", fontsize=16)  # Aumentar el tamaño del título
    plt.xlabel("Número de Caracteres", fontsize=14)  # Aumentar el tamaño de la etiqueta del eje x
    plt.ylim([50, 70])
    plt.ylabel("PSNR (dB)", fontsize=14)  # Aumentar el tamaño de la etiqueta del eje y
    plt.grid()
    plt.savefig(file_path)
    plt.close()
    print(f"Gráfico guardado en {file_path}")


if __name__ == "__main__":
    run()
