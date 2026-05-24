import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('model/pneumonet.keras')
THRESHOLD = 0.3

datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    'chest_xray/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

preds = model.predict(test_gen, verbose=1)
pred_labels = (preds > THRESHOLD).astype(int).flatten()
true_labels = test_gen.classes

normal_total    = np.sum(true_labels == 0)
pneumonia_total = np.sum(true_labels == 1)
normal_correct    = np.sum((pred_labels == 0) & (true_labels == 0))
pneumonia_correct = np.sum((pred_labels == 1) & (true_labels == 1))

print("\n========== BALANCE CHECK ==========")
print(f"Normal    — Correct: {normal_correct}/{normal_total} "
      f"({normal_correct/normal_total*100:.1f}%)")
print(f"Pneumonia — Correct: {pneumonia_correct}/{pneumonia_total} "
      f"({pneumonia_correct/pneumonia_total*100:.1f}%)")
print(f"Overall Accuracy: {np.mean(pred_labels == true_labels)*100:.2f}%")
print("====================================")

if normal_correct/normal_total < 0.4:
    print("\nWARNING: Model still biased! Normal detection bahut kam hai.")
    print("Threshold aur kam karo ya training dobara karo.")
else:
    print("\nModel balanced hai!")