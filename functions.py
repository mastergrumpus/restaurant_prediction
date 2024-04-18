from sklearn.metrics import f1_score, log_loss
import tensorflow as tf
from tensorflow import keras

def sugar_classifier(sugar):
    if sugar >= 30:
        return 5
    elif (sugar < 30) & (sugar >= 7):
        return 4
    elif (sugar < 7) & (sugar > 2):
        return 3
    elif (sugar <= 2) & (sugar > 0):
        return 2
    elif sugar == 0:
        return 1

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)
    y_proba_test = model.predict_proba(X_test)

    print(f"Weighted F1 Score (Train): {f1_score(y_train, y_pred_train, average = 'weighted')}")
    print(f"Weighted F1 Score (Test): {f1_score(y_test, y_pred_test, average = 'weighted')}")
    print(f"Log Loss (Train):  {log_loss(y_train, y_proba_train)}")
    print(f"Log Loss (Test):  {log_loss(y_test, y_proba_test)}")

def runtime(start, end):
    runtime = end - start
    hours = runtime // 3600
    runtime %= 3600
    minutes = runtime // 60
    runtime %= 60
    if (hours == 0) & (minutes == 0):
        print(f"Runtime: {runtime:.2f} seconds")
    elif (hours == 0) & (minutes == 1):
        print(f"Runtime: 1 minute, {runtime:.2f} seconds")
    elif (hours == 0) & (minutes > 0):
        print(f"Runtime: {int(minutes)} minutes, {runtime:.2f} seconds")
    elif (hours == 1) & (minutes == 0):
        print(f"Runtime: 1 hour, {runtime:.2f} seconds")
    elif (hours == 1) & (minutes == 1):
        print(f"Runtime: 1 hour, 1 minute, {runtime:.2f} seconds")
    elif (hours == 1) & (minutes > 0):
        print(f"Runtime: 1 hour, {int(minutes)} minutes, {runtime:.2f} seconds")
    elif (hours > 1) & (minutes == 0):
        print(f"Runtime: {int(hours)} hours, {runtime:.2f} seconds")
    elif (hours > 1) & (minutes == 1):
        print(f"Runtime: {int(hours)} hours, 1 minute, {runtime:.2f} seconds")
    else:
        print(f"{int(hours)} hours, {int(minutes)} minutes, {runtime:.2f} seconds")

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
    predicted_positives = tf.reduce_sum(tf.cast(y_pred, tf.float32), axis=0)
    actual_positives = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=0)
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)