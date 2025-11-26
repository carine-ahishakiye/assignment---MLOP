
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score)


class BirdSoundClassifier:
    """Deep Neural Network for bird sound classification"""
    
    def __init__(self, input_dim=95, num_classes=64, random_seed=42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.model = None
        self.history = None
        
        # Set seed
        np.random.seed(random_seed)
        keras.utils.set_random_seed(random_seed)
    
    def build_model(self):
        """Build optimized neural network architecture"""
        
        self.model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            # Layer 1: 512 neurons
            layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Layer 2: 256 neurons
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Layer 3: 128 neurons
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
       
        print("MODEL ARCHITECTURE")
        self.model.summary()
        return self.model
    
    def train(self, X_train, y_train, epochs=30, batch_size=32, 
              validation_split=0.2, checkpoint_path='models/'):
        """Train model with callbacks"""
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                os.path.join(checkpoint_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print(f"\nTRAINING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\nTRAINING COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return self.history
    
    def plot_training_history(self, save_path='training_history.png'):
        """Visualize training history"""
        
        if self.history is None:
            print("No training history found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training History', fontsize=20, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.show()
    
    def evaluate_on_test_set(self, X_test, y_test, label_encoder):
        """Evaluate model on test set with detailed metrics"""
        
        
        print("MODEL EVALUATION ON TEST SET")
      
        
        # Predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Print metrics
        print("\nPERFORMANCE METRICS (4+ Required)")
        print(f"1. ACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"2. PRECISION: {precision:.4f} ({precision*100:.2f}%)")
        print(f"3. RECALL:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"4. F1-SCORE:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"5. TEST LOSS: {test_loss[0]:.4f}")
        
        if accuracy > 0.95:
            print("\n  WARNING: Very high accuracy might indicate:")
            print("   - Small dataset or limited species diversity")
            print("   - Very distinct species with clear acoustic differences")
            print("   - Potential overfitting (check validation curves)")
        else:
            print("\n✓ Realistic accuracy indicates proper train/test separation")
        
        # Detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_, cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_loss': test_loss[0],
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def save_model(self, path='models/'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'best_model.h5')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, path):
        """Load pre-trained model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict_single(self, features, scaler=None):
        """Make prediction on single audio features"""
        if scaler:
            features = scaler.transform([features])
        else:
            features = np.array([features])
        
        prediction_probs = self.model.predict(features, verbose=0)[0]
        return prediction_probs
    
    def retrain(self, X_new, y_new, epochs=10, batch_size=32):
        """Retrain model with new data (fine-tuning)"""
        if self.model is None:
            print("Model not initialized. Build model first.")
            return
        
        print(f"\nRETRAINING with {len(X_new)} new samples")
        
        history = self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        print("Retraining Complete")
        return history


if __name__ == "__main__":
    # usage
    classifier = BirdSoundClassifier(input_dim=95, num_classes=64)
    classifier.build_model()
    print("\nModel built successfully!")