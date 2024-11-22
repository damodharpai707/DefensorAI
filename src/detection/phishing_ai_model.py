import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Load tokenizer for BERT
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(data, max_length=128):
    """
    Tokenize textual data using BERT tokenizer.
    Args:
        data (list): List of textual inputs (e.g., URLs or domains).
        max_length (int): Maximum token length for input.
    Returns:
        dict: Tokenized data suitable for BERT model.
    """
    return TOKENIZER(
        data,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )

def build_ai_model():
    """
    Build an advanced deep learning model using BERT for feature extraction.
    """
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    cls_output = bert_output.last_hidden_state[:, 0, :]  # CLS token output

    # Dense layers for classification
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(cls_output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc')])
    return model

def adversarial_training(model, X_train, y_train, epsilon=0.01):
    """
    Add adversarial noise to input for robust training.
    Args:
        model (tf.keras.Model): Trained model.
        X_train (tf.Tensor): Training input data.
        y_train (tf.Tensor): Training labels.
        epsilon (float): Magnitude of adversarial noise.
    Returns:
        tf.Tensor, tf.Tensor: Adversarially perturbed data and labels.
    """
    with tf.GradientTape() as tape:
        tape.watch(X_train)
        predictions = model(X_train)
        loss = tf.keras.losses.binary_crossentropy(y_train, predictions)

    gradients = tape.gradient(loss, X_train)
    adversarial_data = X_train + epsilon * tf.sign(gradients)
    adversarial_data = tf.clip_by_value(adversarial_data, 0, 1)  # Clip values to valid range
    return adversarial_data, y_train

def train_ai_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Train the AI model with optional validation data and adversarial training.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint("best_ai_model.h5", monitor='val_loss', save_best_only=True)
    ]

    # Adversarial training
    X_train_adv, y_train_adv = adversarial_training(model, X_train, y_train)

    # Combine original and adversarial examples
    X_combined = tf.concat([X_train, X_train_adv], axis=0)
    y_combined = tf.concat([y_train, y_train_adv], axis=0)

    history = model.fit(
        X_combined, y_combined,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_ai_model(model, X_test, y_test):
    """
    Evaluate the AI model and generate metrics.
    """
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC-ROC Score: {auc_score}")

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc}")

    plot_confusion_matrix(cm)
    plot_precision_recall_curve(precision, recall)

    return y_pred

def plot_precision_recall_curve(precision, recall):
    """
    Plot Precision-Recall curve.
    """
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, labels=["Legitimate", "Phishing"]):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = "d"
    thresh = cm.max() / 2
    for i, j in enumerate(cm.flatten()):
        plt.text(j % 2, j // 2, fmt.format(i),
                 horizontalalignment="center",
                 color="white" if i > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
