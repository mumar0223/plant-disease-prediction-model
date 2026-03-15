import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

print("TF Version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

IMG_SIZE = (128,128)
BATCH_SIZE = 32
TEST_DIR = "Dataset_Split/test"

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:", len(class_names))
print(class_names)

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, base_lr, total_steps, warmup_steps):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):

        step = tf.cast(step, tf.float32)

        warmup_lr = self.base_lr * (step / self.warmup_steps)

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

        cosine_lr = 0.5 * self.base_lr * (1 + tf.cos(np.pi * progress))

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):

        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps
        }

MODEL_PATH = "trained_plant_disease_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"WarmupCosineDecay": WarmupCosineDecay}
)

print("\n Model loaded successfully")

print("\nRunning predictions...")

y_pred_probs = model.predict(test_dataset, verbose=1)

predicted_classes = np.argmax(y_pred_probs, axis=1)

confidences = np.max(y_pred_probs, axis=1) * 100

true_labels = np.concatenate(
    [np.argmax(y, axis=1) for x, y in test_dataset],
    axis=0
)

true_class_names = [class_names[i] for i in true_labels]
pred_class_names = [class_names[i] for i in predicted_classes]

df = pd.DataFrame({

    "true_index": true_labels,
    "pred_index": predicted_classes,
    "true_class": true_class_names,
    "pred_class": pred_class_names,
    "confidence": confidences

})

df.to_csv("test_predictions.csv", index=False)

print("\n CSV saved successfully")
print("Rows saved:", len(df))

print("\nClassification Report\n")

print(
    classification_report(
        true_class_names,
        pred_class_names,
        target_names=class_names
    )
)

cm = confusion_matrix(true_class_names, pred_class_names, labels=class_names)

plt.figure(figsize=(40,40))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Plant Disease Confusion Matrix")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("confusion_matrix_test.png")

plt.show()

print(" Confusion matrix saved")