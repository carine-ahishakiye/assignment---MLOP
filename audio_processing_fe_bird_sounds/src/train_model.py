import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime


class BirdSoundPredictor:
    """Make predictions on audio files"""
    
    def __init__(self, model_path, scaler_path, encoder_path):
        """Initialize predictor with pre-trained model"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.sr = 22050
        self.top_db = 23
        
        # Load model
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"ERROR: Model not found at {model_path}")
        
        # Load scaler
        if os.path.exists(scaler_path):
            self.scaler = pickle.load(open(scaler_path, 'rb'))
            print(f"✓ Scaler loaded from {scaler_path}")
        else:
            print(f"ERROR: Scaler not found at {scaler_path}")
        
        # Load label encoder
        if os.path.exists(encoder_path):
            self.label_encoder = pickle.load(open(encoder_path, 'rb'))
            print(f"✓ Label encoder loaded from {encoder_path}")
        else:
            print(f"ERROR: Label encoder not found at {encoder_path}")
    
    def load_and_trim_audio(self, file_path):
        """Load and trim silence from audio file"""
        try:
            data, sample_rate = librosa.load(file_path, sr=self.sr)
            trimmed_data, _ = librosa.effects.trim(data, top_db=self.top_db)
            
            if len(trimmed_data) < self.sr * 0.5:
                print(f"Audio too short: {len(trimmed_data)/self.sr:.2f} seconds")
                return None, None
            
            return trimmed_data, sample_rate
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_features(self, audio_data):
        """Extract 95 audio features"""
        if audio_data is None or len(audio_data) == 0:
            return None
        try:
            features = []
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=20)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            features.extend(np.max(mfccs, axis=1))
            features.extend(np.min(mfccs, axis=1))
            
            # Spectral Centroid
            sc = librosa.feature.spectral_centroid(y=audio_data, sr=self.sr)
            features.extend([np.mean(sc), np.std(sc), np.max(sc)])
            
            # Spectral Rolloff
            sr_off = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sr)
            features.extend([np.mean(sr_off), np.std(sr_off)])
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Chroma STFT
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sr)
            features.extend([np.mean(chroma), np.std(chroma)])
            
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sr, n_mels=128)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db)])
            
            # Spectral Bandwidth
            sbw = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sr)
            features.extend([np.mean(sbw), np.std(sbw)])
            
            # RMS Energy
            rms = librosa.feature.rms(y=audio_data)
            features.append(np.mean(rms))
            
            return np.array(features)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict(self, audio_file_path, top_k=5):
        """Make prediction on audio file"""
        if not self.model or not self.scaler or not self.label_encoder:
            print("ERROR: Model, scaler, or label encoder not loaded")
            return None
        
        audio_data, sr = self.load_and_trim_audio(audio_file_path)
        if audio_data is None:
            return None
        
        features = self.extract_features(audio_data)
        if features is None:
            return None
        
        features_scaled = self.scaler.transform([features])
        prediction_probs = self.model.predict(features_scaled, verbose=0)[0]
        
        top_indices = np.argsort(prediction_probs)[-top_k:][::-1]
        
        result = {
            'filename': os.path.basename(audio_file_path),
            'predicted_species': self.label_encoder.classes_[np.argmax(prediction_probs)],
            'confidence': float(prediction_probs[np.argmax(prediction_probs)]),
            'confidence_percent': f"{prediction_probs[np.argmax(prediction_probs)]*100:.2f}%",
            'top_predictions': {
                self.label_encoder.classes_[i]: float(prediction_probs[i])
                for i in top_indices
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, audio_folder_path, file_extensions=('.wav', '.mp3')):
        """Make predictions on multiple audio files"""
        if not os.path.exists(audio_folder_path):
            os.makedirs(audio_folder_path)
            print(f"Created folder: {audio_folder_path}")
            return []
        
        audio_files = [f for f in os.listdir(audio_folder_path) if f.lower().endswith(file_extensions)]
        if not audio_files:
            print(f"No audio files found in {audio_folder_path} with extensions {file_extensions}")
            return []
        
        print(f"Predicting on {len(audio_files)} files...")
        results = []
        for audio_file in audio_files:
            file_path = os.path.join(audio_folder_path, audio_file)
            result = self.predict(file_path)
            if result:
                results.append(result)
                print(f"  {audio_file}: {result['predicted_species']} ({result['confidence_percent']})")
        
        return results


if __name__ == "__main__":
    # Paths to your files
    MODEL_PATH = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\final_model.h5"
    SCALER_PATH = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\scaler.pkl"
    ENCODER_PATH = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\label_encoder.pkl"
    TEST_AUDIO_DIR = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\test_audio"
    
    # Initialize predictor
    predictor = BirdSoundPredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        encoder_path=ENCODER_PATH
    )
    
    # Single prediction
    print("\nMaking single prediction...")
    single_audio_file = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\data\Voice of Birds\Voice of Birds\Andean Tinamou_sound\Andean Tinamou2.mp3"
    result = predictor.predict(single_audio_file)
    if result:
        print(f"Predicted: {result['predicted_species']} ({result['confidence_percent']})")
    
    # Batch prediction
    print("\nMaking batch predictions...")
    batch_results = predictor.predict_batch(TEST_AUDIO_DIR)
    if batch_results:
        print(f"Processed {len(batch_results)} files")
