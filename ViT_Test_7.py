import os

# ---------------------------------------------------------------------------
# Configurable paths (env vars with original training-time defaults).
# Override TG2_HOME and TG2_DATA_ROOT to relocate models, results, dataset.
# These defaults match the on-prem GPU server "Cratos" where this code was
# trained for the MSc thesis; they are kept for academic provenance.
# ---------------------------------------------------------------------------
TG2_HOME = os.environ.get("TG2_HOME", ".")
TG2_DATA_ROOT = os.environ.get("TG2_DATA_ROOT", "./data")
MODELS_DIR = os.path.join(TG2_HOME, "Models")
RESULTS_DIR = os.path.join(TG2_HOME, "Resultados")
CSV_DIR = os.path.join(RESULTS_DIR, "CSV_Files")
CONFIG_INI = os.path.join(TG2_HOME, "Stuff", "config.ini")

import requests
import traceback
import configparser
import random
import itertools
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
EPOCHS = 20

# ======================== INICIO - TELEGRAM ========================= #


def send_telegram_message(message):
    """Send a Telegram notification, silently no-op if not configured.

    Reads ``apiToken`` and ``chatID`` from the ``[Telegram]`` section of
    ``$TG2_HOME/Stuff/config.ini`` (or whatever ``CONFIG_INI`` resolves to).
    If the file is missing, the section is missing, or the HTTP request
    fails, this function returns silently — no warnings, no exceptions —
    so the training pipeline never crashes because of optional notifications.
    """
    if not os.path.exists(CONFIG_INI):
        return
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_INI)
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


# ======================== FIN - TELEGRAM ========================= #

send_telegram_message(
    "[ViT] El programa {} ha empezado a ejecutarse.".format(os.path.basename(__file__))
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
send_telegram_message("[ViT] Iniciando etapa de preparación de datos...")

folder_paths = [
    os.path.join(TG2_DATA_ROOT, "input-data"),  # Hay 510612 imagenes
    os.path.join(TG2_DATA_ROOT, "output-lsb"),  # Hay 638265 imagenes
    os.path.join(TG2_DATA_ROOT, "output-dct"),  # Hay 638265 imagenes
    os.path.join(TG2_DATA_ROOT, "output-dwt"),  # Hay 638265 imagenes
]

labels = [0, 1, 1, 1]
all_image_paths = []
all_image_labels = []

sample_size = len(os.listdir(folder_paths[0])) // 3

for folder_path, label in zip(folder_paths, labels):
    if label == 0:
        # Si es la clase 0 (input-data), simplemente añadimos todas las imágenes
        for image_file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, image_file)
            all_image_paths.append(full_path)
            all_image_labels.append(label)
    else:
        # Si es clase 1 (output-lsb, output-dct o output-dwt), tomamos una muestra aleatoria
        image_files_sample = random.sample(os.listdir(folder_path), sample_size)
        for image_file in image_files_sample:
            full_path = os.path.join(folder_path, image_file)
            all_image_paths.append(full_path)
            all_image_labels.append(label)


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
    train_dataset, batch_size=48, sampler=sampler, num_workers=24, pin_memory=True
)
validation_loader = DataLoader(
    validation_dataset, batch_size=48, shuffle=False, num_workers=24, pin_memory=True
)

test_dataset = CustomDataset(test_paths, test_labels, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=48, shuffle=False, num_workers=24, pin_memory=True
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

    file_path = os.path.join(RESULTS_DIR, f"ROC_{SCRIPT_NAME}.pdf")
    plt.savefig(file_path)
    plt.close()  # Cierre de la figura

    send_telegram_message(f"[ViT] La curva ROC se guardó en {file_path}")


def calculate_far_frr(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    far = fpr
    frr = 1 - tpr
    return far, frr, thresholds


def save_frr_far_data_as_csv(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    far_values = fpr
    frr_values = 1 - tpr

    data = {"Thresholds": thresholds, "FAR": far_values, "FRR": frr_values}

    file_path = os.path.join(
        CSV_DIR, f"FRR_FAR_Data_{SCRIPT_NAME}.csv"
    )

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    send_telegram_message(f"[CNN] Los datos de FAR y FRR se guardaron en {file_path}")


def plot_far_frr(y_true, y_pred_prob):
    save_frr_far_data_as_csv(y_true, y_pred_prob)
    far, frr, thresholds = calculate_far_frr(y_true, y_pred_prob)

    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2
    threshold_eer = thresholds[idx]

    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, far, color="blue", lw=2, label="FAR")
    plt.plot(thresholds, frr, color="red", lw=2, label="FRR")
    plt.axvline(threshold_eer, color="k", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Accuracy")
    plt.ylabel("Percentage")
    plt.title("FAR vs. FRR")
    plt.legend(loc="lower right")

    file_path = os.path.join(
        RESULTS_DIR, f"FAR_FRR_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(f"[ViT] La gráfica de FAR vs FRR se guardó en {file_path}")

    return eer, threshold_eer


def plot_classification_distribution(y_true, y_pred_prob):
    # Convertir a arrays de NumPy
    y_true = np.array(y_true).ravel()
    y_pred_prob = np.array(y_pred_prob).ravel()

    data = pd.DataFrame({"Classification_Value": y_pred_prob, "Class": y_true})

    # Clasificación positiva y negativa
    positive_class = y_pred_prob[y_true == 1]
    negative_class = y_pred_prob[y_true == 0]

    plt.figure(figsize=(10, 7))
    plt.hist(
        positive_class,
        bins=50,
        color="blue",
        label="Clase Positiva",
        alpha=0.7,
        density=True,
    )
    plt.hist(
        negative_class,
        bins=50,
        color="red",
        label="Clase Negativa",
        alpha=0.7,
        density=True,
    )
    plt.xlabel("Valor de Clasificación")
    plt.ylabel("Frecuencia")
    plt.title("Distribución del Valor de Clasificación")
    plt.legend()

    file_path = os.path.join(
        CSV_DIR,
        f"Distribucion_{SCRIPT_NAME}.csv",
    )
    data.to_csv(file_path, index=False)

    send_telegram_message(
        f"[ViT] Los datos de la distribucion del valor de clasificación se guardaron en {file_path}"
    )

    file_path = os.path.join(
        RESULTS_DIR, f"Distribucion_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[ViT] La grafica de distribucion del valor de clasificación se guardo en {file_path}"
    )


def plot_classification_density(y_true, y_pred_prob):
    y_pred_prob = np.array(y_pred_prob).ravel()
    y_true = np.array(y_true).ravel()

    # Convertir a un DataFrame de pandas
    data = pd.DataFrame({"Classification_Value": y_pred_prob, "Class": y_true})

    # Configurar el estilo de Seaborn
    sns.set_style("white")

    # Crear un objeto de figura y ejes para tener un control preciso
    fig, ax = plt.subplots(figsize=(10, 7))

    # Gráfico de la clase positiva
    sns.kdeplot(
        data=data[data["Class"] == 1],
        x="Classification_Value",
        fill=True,
        color="blue",
        label="Clase Positiva",
        alpha=0.7,
        ax=ax,
    )

    # Gráfico de la clase negativa
    sns.kdeplot(
        data=data[data["Class"] == 0],
        x="Classification_Value",
        fill=True,
        color="red",
        label="Clase Negativa",
        alpha=0.7,
        ax=ax,
    )

    ax.set_xlabel("Valor de Clasificación")
    ax.set_ylabel("Densidad")
    ax.set_title("Densidad del Valor de Clasificación")

    if data["Classification_Value"].mean() > 0.5:
        legend_loc = "upper left"
    else:
        legend_loc = "upper right"

    ax.legend(loc=legend_loc)

    # Guardar CSV de los datos de la gráfica
    file_path = os.path.join(
        CSV_DIR, f"Densidad_{SCRIPT_NAME}.csv"
    )
    data.to_csv(file_path, index=False)

    send_telegram_message(
        f"[ViT] Los datos de la densidad del valor de clasificación se guardaron en {file_path}"
    )

    file_path = os.path.join(
        RESULTS_DIR, f"Densidad_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[ViT] La densidad del valor de clasificación se guardó en {file_path}"
    )


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))  # Incrementa el tamaño de la figura

    # Invierte el orden de la matriz y las clases para mejor visualización
    cm = cm[::-1, ::-1]
    classes = classes[::-1]

    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    # Elige un umbral que es un promedio de los valores de la matriz para mejor contraste
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = (
            "white" if cm[i, j] > thresh else "black"
        )  # Contraste de color basado en umbral
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            verticalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")

    file_path = os.path.join(
        RESULTS_DIR, f"Confusion_Matrix_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(f"[ViT] La matriz de confusión se guardo en {file_path}")


# Definición del modelo
v = ViT(
    image_size=480,
    patch_size=20,  # Antes 30. Esto permitirá al modelo captar detalles más finos al trabajar con parches más pequeños de las imágenes.
    num_classes=1,
    dim=1024,  # Este es el tamaño de las características (dimensiones) de cada token después de pasar por el embedding.
    depth=8,  # Antes 6. Esto añadirá capas adicionales al modelo, lo que podría ayudar a aprender representaciones más complejas.
    heads=32,  # Antes 16. Esto permitirá una atención más diversa sobre los datos, lo que podría mejorar el rendimiento en tareas de clasificación complejas.
    mlp_dim=2048,  # Este es el tamaño de la capa densa en el bloque Transformer.
    dropout=0.1,  
    emb_dropout=0.1,  
)

if torch.cuda.device_count() > 1:
    v = torch.nn.DataParallel(v, device_ids=[0, 1, 2, 3, 4, 5])

v = v.to(device)

# Definir el optimizador
optimizer = optim.Adam(v.parameters(), lr=0.001)

# Definir el scheduler
scheduler = ReduceLROnPlateau(optimizer, "min", patience=2)

best_val_loss = float("inf")
best_val_metrics = {}
best_train_metrics = {}


def train_and_validate(epochs):
    global best_val_loss
    global best_val_metrics
    global best_train_metrics

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        send_telegram_message(f"[ViT] Inicio de la epoca {epoch}/{EPOCHS}")

        # Resetear las métricas al inicio de cada época para el entrenamiento
        accuracy.reset()
        auroc.reset()
        precision.reset()
        recall.reset()
        f1.reset()

        # Entrenamiento
        v.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = v(data)
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(output).squeeze()
            # Actualizar métricas (sin calcularlas aún)
            accuracy.update(preds, target)
            auroc.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1.update(preds, target)

        # Guardar métricas temporales de entrenamiento
        temp_train_metrics = {
            "Loss": loss.item(),
            "Accuracy": accuracy.compute().item(),
            "AUROC": auroc.compute().item(),
            "Precision": precision.compute().item(),
            "Recall": recall.compute().item(),
            "F1 Score": f1.compute().item(),
        }

        message = (
            f"-- TRAINING (EPOCH {epoch}) --\n"
            f"Loss: {loss.item()}\n"
            f"Accuracy: {accuracy.compute()}\n"
            f"AUROC: {auroc.compute()}\n"
            f"Precision: {precision.compute()}\n"
            f"Recall: {recall.compute()}\n"
            f"F1 Score: {f1.compute()}"
        )
        send_telegram_message(message)

        train_losses.append(loss.item())
        train_accuracies.append(accuracy.compute().item())

        accuracy.reset()
        auroc.reset()
        precision.reset()
        recall.reset()
        f1.reset()

        # Validación
        v.eval()
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
                auroc.update(preds, target)
                precision.update(preds, target)
                recall.update(preds, target)
                f1.update(preds, target)

        avg_loss = total_loss / len(validation_loader)
        val_losses.append(avg_loss)
        val_accuracies.append(accuracy.compute().item())
        scheduler.step(avg_loss)

        message = (
            f"-- VALIDATION (EPOCH {epoch}) -- \n"
            f"Loss: {avg_loss:.6f}\n"
            f"Accuracy: {accuracy.compute()}\n"
            f"AUROC: {auroc.compute()}\n"
            f"Precision: {precision.compute()}\n"
            f"Recall: {recall.compute()}\n"
            f"F1 Score: {f1.compute()}"
        )
        send_telegram_message(message)

        # Comprobar si esta es la mejor pérdida y actualizar todo si es necesario
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_train_metrics = temp_train_metrics
            best_val_metrics = {  # Actualizar las métricas de validación
                "Loss": avg_loss,
                "Accuracy": accuracy.compute().item(),
                "AUROC": auroc.compute().item(),
                "Precision": precision.compute().item(),
                "Recall": recall.compute().item(),
                "F1 Score": f1.compute().item(),
            }

            # Guardar el mejor modelo
            base_path = MODELS_DIR
            model_path = os.path.join(base_path, f"Best_Model_{SCRIPT_NAME}.pth")
            torch.save(v.state_dict(), model_path)
            send_telegram_message(
                f"[ViT] Guardado el mejor modelo con pérdida de validación: {avg_loss:.6f} en {model_path}"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == 5:  # Número de épocas sin mejora
            send_telegram_message(f"[ViT] EarlyStopping activado en la época: {epoch}.")
            break

    # Al final de todas las épocas, enviar las métricas del mejor modelo
    if best_val_metrics:
        train_msg = "-- BEST TRAINING METRICS --\n" + "\n".join(
            [f"{k}: {v:.6f}" for k, v in best_train_metrics.items()]
        )
        val_msg = "-- BEST VALIDATION METRICS --\n" + "\n".join(
            [f"{k}: {v:.6f}" for k, v in best_val_metrics.items()]
        )
        send_telegram_message(train_msg)
        send_telegram_message(val_msg)
        # Gráfica de precisión y pérdida por época
        file_path = os.path.join(
            RESULTS_DIR, f"Accuracy_Loss_{SCRIPT_NAME}.pdf"
        )
        plt.figure(figsize=(12, 4))

        # Gráfica de precisión
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="upper left")

        # Gráfica de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        send_telegram_message(
            f"[ViT] La gráfica Accuracy vs Loss se guardó en {file_path}"
        )

    else:
        send_telegram_message(
            "[ViT] No se encontró una mejora en la pérdida de validación a lo largo de las épocas."
        )


# Test del modelo
def test():
    base_path = MODELS_DIR
    model_path = os.path.join(base_path, f"Best_Model_{SCRIPT_NAME}.pth")
    v.load_state_dict(torch.load(model_path))
    v.eval()
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    mcc.reset()
    auroc.reset()

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
    send_telegram_message(message)

    predicted_labels = np.where(np.array(all_preds) > 0.5, 1, 0).flatten()

    # Calcular la matriz de confusión
    cm = confusion_matrix(all_targets, predicted_labels)

    # Crear gráficos
    plot_roc_curve(all_targets, all_preds)
    plot_classification_distribution(all_targets, all_preds)
    plot_classification_density(all_targets, all_preds)
    plot_confusion_matrix(cm, classes=[0, 1])
    eer, threshold_eer = plot_far_frr(all_targets, all_preds)

    send_telegram_message(f"EER: {eer:.2f} at threshold: {threshold_eer:.2f}\n")


def run():
    try:
        send_telegram_message("[ViT] Iniciando etapa de entrenamiento y validacion...")
        train_and_validate(EPOCHS)
        send_telegram_message("[ViT] Iniciando etapa de prueba...")
        test()

    except Exception as e:
        error_message = str(e) + "\n\n" + traceback.format_exc()
        send_telegram_message(f"[ViT] Error durante la ejecucion:\n{error_message}")

    send_telegram_message(
        "[ViT] El programa {} ha terminado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )


if __name__ == "__main__":
    run()
