import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121, InceptionV3, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ------------------- CONFIG -------------------
# Paths to your datasets (you have two)
DATASET_DIRS = [
    r"C:\Users\vishwanath g\Desktop\Cotton_disease_detection_datasets\Original Dataset (1)\Original Dataset",
    r"C:\Users\vishwanath g\Desktop\Cotton_disease_detection_datasets\roboflow_dataset"
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Directory to save trained models
SAVE_DIR = r"C:\Users\vishwanath g\Desktop\cotton_disease_project\webapp\model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------- COMBINE DATASET -------------------
# Copy all images into a single folder structure for flow_from_directory
# For simplicity, we will assume both datasets already have subfolders per class
# You can merge them manually or with a script if needed
# Here, we will just pick one path at a time (update if merging is needed)
DATASET_DIR = DATASET_DIRS[0]  # <-- Start with Original Dataset, you can change to roboflow_dataset

# ------------------- DATA GENERATORS -------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print("✅ Classes found:", train_gen.class_indices)

# ------------------- MODEL BUILDER -------------------
def build_model(base_model, num_classes):
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------- MODELS TO TRAIN -------------------
models_to_train = {
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
}

# ------------------- TRAIN AND SAVE -------------------
results = {}
for name, base in models_to_train.items():
    print(f"\n🔹 Training {name} ...")
    model = build_model(base, num_classes)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    acc = max(history.history['val_accuracy'])
    results[name] = round(acc * 100, 2)
    
    save_path = os.path.join(SAVE_DIR, f"{name}_model.h5")
    model.save(save_path)
    print(f"✅ Saved {name}_model.h5 — Best Validation Accuracy: {results[name]}%")

# ------------------- SUMMARY -------------------
print("\n📊 Model Comparison Results:")
for k, v in results.items():
    print(f"{k} → {v}%")
