import os
import requests
import traceback
import configparser
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from vit_pytorch import ViT
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
)
import matplotlib.pyplot as plt


# ======================== INICIO - TELEGRAM ========================= #

config = configparser.ConfigParser()
config.read("/home/srojas/tg2/Stuff/config.ini")
apiToken = config.get("Telegram", "apiToken")
chatID = config.get("Telegram", "chatID")


def send_telegram_message(message):
    apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage"
    try:
        response = requests.post(apiURL, json={"chat_id": chatID, "text": message})
    except Exception as e:
        print(e)


# ======================== FIN - TELEGRAM ========================= #

send_telegram_message(
    "[CRATOS] Tu programa {} ha empezado a ejecutarse.".format(
        os.path.basename(__file__)
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancia las métricas
accuracy = BinaryAccuracy().to(device)
auroc = BinaryAUROC().to(device)
precision = BinaryPrecision().to(device)
recall = BinaryRecall().to(device)
f1 = BinaryF1Score().to(device)
mcc = BinaryMatthewsCorrCoef().to(device)

# 1. Preparación de los datos.
print("Iniciando etapa de preparación de datos...")
send_telegram_message("Iniciando etapa de preparación de datos...")

folder_paths = [
    "/data/estudiantes/srojas/input-data-test",
    "/data/estudiantes/srojas/output-lsb-test",
    "/data/estudiantes/srojas/output-dct-test",
    "/data/estudiantes/srojas/output-dwt-test",
]

labels = [0, 1, 1, 1]
all_image_paths = []
all_image_labels = []

for folder_path, label in zip(folder_paths, labels):
    for image_file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, image_file)
        try:
            Image.open(full_path).close()  # Abre y cierra inmediatamente para verificar
            all_image_paths.append(full_path)
            all_image_labels.append(label)
        except Exception as e:
            print(f"Error al abrir la imagen: {full_path}. Error: {e}")
            send_telegram_message(f"Error al abrir la imagen: {full_path}. Error: {e}")
    print(f"Carpeta {folder_path} procesada.")
    send_telegram_message(f"Carpeta {folder_path} procesada.")


# División de los datos en entrenamiento, validación y pruebas
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_image_paths, all_image_labels, test_size=0.3, random_state=42
)

validation_paths, test_paths, validation_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42
)

unique_labels, counts = np.unique(all_image_labels, return_counts=True)
class_counts = [train_labels.count(i) for i in unique_labels]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
samples_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=samples_weights, num_samples=len(samples_weights), replacement=True
)

EPOCHS = 20


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose(
    [
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
    ]
)

train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
validation_dataset = CustomDataset(
    validation_paths, validation_labels, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
)
validation_loader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

test_dataset = CustomDataset(test_paths, test_labels, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)


def plot_roc_curve(y_true, y_pred_prob):
    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    # Calcular el AUC
    roc_auc = auc(fpr, tpr)

    # Trazar la curva ROC
    plt.figure(figsize=(10, 7))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    script_name = os.path.basename(__file__).split(".")[0]
    file_path = os.path.join("/home/srojas/tg2/Resultados/", f"ROC_{script_name}.png")
    plt.savefig(file_path)
    plt.close()  # Cierre de la figura

    send_telegram_message(f"[CRATOS] Tu curva ROC se guardó en {file_path}")


# Definición del modelo
v = ViT(
    image_size=480,
    patch_size=30,
    num_classes=1,  # Hay dos clases según tus etiquetas
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
)

if torch.cuda.device_count() > 1:
    v = torch.nn.DataParallel(v)

v = v.to(device)

# Definir el optimizador
optimizer = optim.Adam(v.parameters(), lr=0.001)


# Función para entrenar el modelo
def train():
    v.train()
    accuracy.reset()
    auroc.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    mcc.reset()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = v(data)
        loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())
        loss.backward()
        optimizer.step()
        print(target[:10])

        preds = torch.sigmoid(output).squeeze()
        print(preds.min(), preds.max())

        # Actualizar métricas (sin calcularlas aún)
        accuracy.update(preds, target)
        auroc.update(preds, target)
        precision.update(preds, target)
        recall.update(preds, target)
        f1.update(preds, target)
        mcc.update(preds, target)

    # Calcular y mostrar las métricas
    message = (
        f"-- TRAINING --\n"
        f"Loss: {loss.item():.6f}\n"
        f"Accuracy: {accuracy.compute()}\n"
        f"Precision: {precision.compute()}\n"
        f"Recall: {recall.compute()}\n"
        f"F1 Score: {f1.compute()}\n"
    )
    print(message)
    send_telegram_message(message)


best_val_loss = float("inf")  # Inicializa con un valor alto


# Función para validar el modelo
def validate():
    global best_val_loss  # Usa la variable global

    v.eval()
    accuracy.reset()
    auroc.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    mcc.reset()

    total_loss = 0.0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = v(data)
            preds = torch.sigmoid(output).squeeze()
            loss_val = F.binary_cross_entropy_with_logits(
                output.squeeze(), target.float()
            )
            total_loss += loss_val.item()

            # Actualizar métricas (sin calcularlas aún)
            accuracy.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1.update(preds, target)

    # Calcular el loss promedio
    avg_loss = total_loss / len(validation_loader)

    # Comprobar si esta es la mejor pérdida y guardar el modelo si es necesario
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        base_path = "/home/srojas/tg2/Models"
        model_path = os.path.join(base_path, f"Best_Model_{script_name}.pth")
        torch.save(v.state_dict(), model_path)
        print(
            f"Guardado el mejor modelo con pérdida de validación: {avg_loss} en {model_path}"
        )
        send_telegram_message(
            f"Guardado el mejor modelo con pérdida de validación: {avg_loss} en {model_path}"
        )

    message = (
        f"-- VALIDATION --\n"
        f"Loss: {avg_loss:.6f}\n"
        f"Accuracy: {accuracy.compute()}\n"
        f"Precision: {precision.compute()}\n"
        f"Recall: {recall.compute()}\n"
        f"F1 Score: {f1.compute()}"
    )

    print(message)
    send_telegram_message(message)


# Test del modelo
def test():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    base_path = "/home/srojas/tg2/Models"
    model_path = os.path.join(base_path, f"Best_Model_{script_name}.pth")
    v.load_state_dict(torch.load(model_path))
    v.eval()
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    mcc.reset()
    auroc.reset()  # Añadido el reset para auroc

    all_targets = []
    all_preds = []

    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = v(data)
            preds = torch.sigmoid(output).squeeze()
            loss_test = F.binary_cross_entropy_with_logits(
                output.squeeze(), target.float()
            )
            total_loss += loss_test.item()

            # Actualizar métricas
            accuracy.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1.update(preds, target)
            mcc.update(preds, target)
            auroc.update(preds, target)  # Añadido el update para auroc

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calcular el loss promedio
    avg_loss = total_loss / len(test_loader)

    # Calcular AUROC
    auc_value = auroc.compute()

    message = (
        f"-- TEST -- \n"
        f"Loss: {avg_loss:.6f}\n"
        f"Accuracy: {accuracy.compute()}\n"
        f"Precision: {precision.compute()}\n"
        f"Recall: {recall.compute()}\n"
        f"F1 Score: {f1.compute()}\n"
        f"MCC: {mcc.compute()}\n"
        f"AUROC: {auc_value}"
    )

    print(message)
    send_telegram_message(message)
    plot_roc_curve(all_targets, all_preds)


def run():
    try:
        # Entrenamiento y validación
        for epoch in range(1, EPOCHS + 1):
            print("Epoca " + str(epoch) + "/" + str(EPOCHS))
            send_telegram_message("Epoca " + str(epoch) + "/" + str(EPOCHS))
            train()
            validate()
        # Ejecutar prueba
        print("Iniciando etapa de prueba...")
        send_telegram_message("Iniciando etapa de prueba...")
        test()

    except Exception as e:
        error_message = str(e) + "\n\n" + traceback.format_exc()
        send_telegram_message(f"[CRATOS] Error durante la ejecucion:\n{error_message}")
        print(f"Error durante la ejecucion:\n{error_message}")

    send_telegram_message(
        "[CRATOS] Tu programa {} ha terminado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )


if __name__ == "__main__":
    run()
