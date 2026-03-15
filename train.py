#  train.py  —  Custom ResNet-SE CNN | Plant Disease Detection (From Scratch)
#  Run split_dataset.py FIRST to generate Dataset_Split/train & val

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, classification_report

print("TF Version  :", tf.__version__)
print("GPUs        :", tf.config.list_physical_devices('GPU'))

# Mixed precision — faster training on GPU, no accuracy loss
tf.keras.mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 50
NUM_CLASSES = 23
AUTOTUNE    = tf.data.AUTOTUNE

TRAIN_DIR = 'Dataset_Split/train'
VAL_DIR   = 'Dataset_Split/val'

# Verify directories exist
for d in [TRAIN_DIR, VAL_DIR]:
    if not os.path.exists(d):
        raise FileNotFoundError(
            f" Directory not found: '{d}'\n"
            "   Please run split_dataset.py first."
        )

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="augmentation")


# Training Image Preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    crop_to_aspect_ratio=False
)

# Validation Image Preprocessing
validation_set = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
    seed=42,
    interpolation="bilinear",
    crop_to_aspect_ratio=False
)

class_names = training_set.class_names
print(f"\nTotal Classes : {len(class_names)}")
print(f"Train Batches : {len(training_set)}")
print(f"Val   Batches : {len(validation_set)}")
print(f"Class Names   : {class_names}")

# Save class names — test.py will load this
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("\n class_names.json saved.")

def prepare(ds, augment=False):
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(training_set, augment=True)
val_ds   = prepare(validation_set, augment=False)

def conv_bn_relu(x, filters, kernel_size=3, strides=1, padding='same'):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def residual_block(x, filters, strides=1):
    shortcut = x

    # Main path
    x = conv_bn_relu(x, filters, kernel_size=3, strides=strides, padding='same')
    x = tf.keras.layers.Conv2D(
        filters, 3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Shortcut — match dimensions if strides or filters changed
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def squeeze_excitation(x, ratio=16):
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu',
                                kernel_initializer='he_normal')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    return tf.keras.layers.Multiply()([x, se])

def build_custom_cnn(num_classes, input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Normalize 0-255 → 0-1
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    # Stem Block — (128,128,3) → (32,32,64)
    x = conv_bn_relu(x, 64, kernel_size=3, strides=2)        # → 64x64
    x = conv_bn_relu(x, 64, kernel_size=3, strides=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)  # → 32x32

    # Stage 1 — 64 filters  (32×32)
    x = residual_block(x, filters=64,  strides=1)
    x = residual_block(x, filters=64,  strides=1)
    x = squeeze_excitation(x, ratio=8)

    # Stage 2 — 128 filters (16×16)
    x = residual_block(x, filters=128, strides=2)
    x = residual_block(x, filters=128, strides=1)
    x = residual_block(x, filters=128, strides=1)
    x = squeeze_excitation(x, ratio=8)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Stage 3 — 256 filters (8×8)
    x = residual_block(x, filters=256, strides=2)
    x = residual_block(x, filters=256, strides=1)
    x = residual_block(x, filters=256, strides=1)
    x = residual_block(x, filters=256, strides=1)
    x = squeeze_excitation(x, ratio=16)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Stage 4 — 512 filters (4×4)
    x = residual_block(x, filters=512, strides=2)
    x = residual_block(x, filters=512, strides=1)
    x = residual_block(x, filters=512, strides=1)
    x = squeeze_excitation(x, ratio=16)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output — float32
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32'
    )(x)

    return tf.keras.Model(inputs, outputs, name="Custom_ResNet_SE")


# Build
tf.keras.backend.clear_session()
cnn = build_custom_cnn(NUM_CLASSES)
cnn.summary()
print(f"\nTotal Parameters : {cnn.count_params():,}")

steps_per_epoch = len(training_set)
total_steps     = EPOCHS * steps_per_epoch
warmup_steps    = 5 * steps_per_epoch

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        self.base_lr      = base_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        progress  = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = 0.5 * self.base_lr * (1 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr"     : self.base_lr,
            "total_steps" : self.total_steps,
            "warmup_steps": self.warmup_steps
        }

lr_schedule = WarmupCosineDecay(
    base_lr=0.001,
    total_steps=total_steps,
    warmup_steps=warmup_steps
)

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    ),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='trained_plant_disease_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger('training_log.csv'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

print("\n" + "="*60)
print("       TRAINING CUSTOM RESIDUAL CNN FROM SCRATCH")
print(f"       Train Dir : {TRAIN_DIR}")
print(f"       Val   Dir : {VAL_DIR}")
print("="*60 + "\n")

training_history = cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Training Set Accuracy
train_loss, train_acc, train_top3 = cnn.evaluate(train_ds, verbose=1)
print(f'\nTraining   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | Top-3: {train_top3:.4f}')

# Validation Set Accuracy
val_loss, val_acc, val_top3 = cnn.evaluate(val_ds, verbose=1)
print(f'Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Top-3: {val_top3:.4f}')

cnn.save('trained_plant_disease_model.keras')
print("\n Model saved  : trained_plant_disease_model.keras")

with open('training_hist.json', 'w') as f:
    json.dump(training_history.history, f)
print(" History saved: training_hist.json")

print("\nHistory Keys:", list(training_history.history.keys()))

acc           = training_history.history['accuracy']
val_acc_hist  = training_history.history['val_accuracy']
loss          = training_history.history['loss']
val_loss_hist = training_history.history['val_loss']
top3          = training_history.history.get('top3_acc', [])
val_top3_hist = training_history.history.get('val_top3_acc', [])
epochs_range  = range(1, len(acc) + 1)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Custom ResNet-SE CNN — Training Results', fontsize=16, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, acc,          'b-o', markersize=3, label='Training Accuracy')
axes[0].plot(epochs_range, val_acc_hist, 'r-o', markersize=3, label='Validation Accuracy')
axes[0].set_title('Accuracy');  axes[0].set_xlabel('No. of Epochs')
axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(epochs_range, loss,          'b-o', markersize=3, label='Training Loss')
axes[1].plot(epochs_range, val_loss_hist, 'r-o', markersize=3, label='Validation Loss')
axes[1].set_title('Loss');  axes[1].set_xlabel('No. of Epochs')
axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Top-3 Accuracy
if top3:
    axes[2].plot(epochs_range, top3,          'b-o', markersize=3, label='Train Top-3')
    axes[2].plot(epochs_range, val_top3_hist, 'r-o', markersize=3, label='Val   Top-3')
    axes[2].set_title('Top-3 Accuracy'); axes[2].set_xlabel('No. of Epochs')
    axes[2].set_ylabel('Top-3 Accuracy'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Plot saved   : training_curves.png")

eval_set = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=1,
    image_size=IMG_SIZE,
    shuffle=False
)

y_pred_probs         = cnn.predict(eval_set, verbose=1)
predicted_categories = np.argmax(y_pred_probs, axis=1)
true_categories      = np.concatenate([np.argmax(y, axis=1) for _, y in eval_set])

print("\n" + "="*60)
print("   CLASSIFICATION REPORT  (Validation Set)")
print("="*60)
print(classification_report(true_categories, predicted_categories, target_names=class_names))

cm = confusion_matrix(true_categories, predicted_categories)

plt.figure(figsize=(40, 40))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    annot_kws={"size": 10}
)
plt.xlabel('Predicted Class', fontsize=20)
plt.ylabel('Actual Class',    fontsize=20)
plt.title('Crop Disease Prediction — Confusion Matrix (Val Set)', fontsize=25)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_val.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Confusion matrix saved: confusion_matrix_val.png")