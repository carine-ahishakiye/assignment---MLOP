import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

class AudioProcessor:
    """Process audio files and extract 95 features"""
    
    def __init__(self, sr=22050, top_db=23):
        self.sr = sr
        self.top_db = top_db
    
    def load_and_trim_audio(self, file_path, max_duration=10):
        """Load MP3 audio and trim silence, skip corrupted files quickly"""
        try:
            # Load only first `max_duration` seconds to avoid freezing
            data, sample_rate = librosa.load(file_path, sr=self.sr, duration=max_duration)
            trimmed_data, _ = librosa.effects.trim(data, top_db=self.top_db)
            if len(trimmed_data) < self.sr * 0.5:
                return None, None
            return trimmed_data, sample_rate
        except Exception:
            # Skip problematic/corrupted MP3s
            return None, None
    
    def extract_features(self, audio_data):
        """Extract 95 audio features from audio data"""
        if audio_data is None or len(audio_data) == 0:
            return None
        
        try:
            features = []
            
            # MFCC features
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
        except Exception:
            return None


class DataProcessor:
    """Prepare data for model training and prevent data leakage"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = None
        self.scaler = None
    
    def load_dataset(self, csv_path):
        """Load dataset from CSV and parse date columns"""
        df = pd.read_csv(csv_path)
        df['year'] = df['Date'].apply(lambda x: str(x).split('-')[0])
        df['month'] = df['Date'].apply(lambda x: str(x).split('-')[1])
        df['day_of_month'] = df['Date'].apply(lambda x: str(x).split('-')[2])
        
        print("DATASET LOADED")
        print(f"Shape: {df.shape} | Species: {df['common_name'].nunique()}")
        return df
    
    def process_audio_files(self, df, audio_base_path, samples_per_species=25, min_samples_per_class=15):
        """Process audio files and extract features, track skipped files"""
        
        audio_processor = AudioProcessor(sr=22050, top_db=23)
        features_list, labels_list, file_paths_list = [], [], []
        skipped_files = []
        
        # Filter species with enough samples
        species_counts = df['common_name'].value_counts()
        valid_species = species_counts[species_counts >= min_samples_per_class].index
        print(f"Processing {len(valid_species)} species with >= {min_samples_per_class} samples each")
        
        available_folders = os.listdir(audio_base_path)
        
        def find_species_folder(common_name):
            for folder in available_folders:
                folder_clean = folder.lower().replace('_', ' ').replace('sound', '').strip()
                if common_name.lower() in folder_clean or folder_clean in common_name.lower():
                    return folder
            return None
        
        for species in tqdm(valid_species, desc="Processing species"):
            species_folder = find_species_folder(species)
            if not species_folder:
                continue
            
            folder_path = os.path.join(audio_base_path, species_folder)
            mp3_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
            if len(mp3_files) == 0:
                continue
            
            np.random.shuffle(mp3_files)
            files_to_process = mp3_files[:samples_per_species]
            
            for mp3_file in files_to_process:
                file_path = os.path.join(folder_path, mp3_file)
                try:
                    audio_data, sr = audio_processor.load_and_trim_audio(file_path)
                    if audio_data is None or len(audio_data) < sr * 0.5:
                        skipped_files.append(file_path)
                        continue
                    
                    features = audio_processor.extract_features(audio_data)
                    if features is None or np.isnan(features).any():
                        skipped_files.append(file_path)
                        continue
                    
                    features_list.append(features)
                    labels_list.append(species)
                    file_paths_list.append(file_path)
                    
                except Exception:
                    skipped_files.append(file_path)
                    continue
        
        print(f"\n✓ Processed: {len(features_list)} samples from {len(np.unique(labels_list))} species")
        if skipped_files:
            print(f"⚠️ Skipped {len(skipped_files)} files due to errors or invalid audio")
            with open("skipped_files.txt", "w") as f:
                for path in skipped_files:
                    f.write(path + "\n")
            print("List of skipped files saved to skipped_files.txt")
        
        return np.array(features_list), np.array(labels_list), file_paths_list
    
    def prepare_train_test_data(self, X, y):
        """Encode labels, split train-test, scale features"""
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train_enc, y_test_enc = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        y_train = to_categorical(y_train_enc)
        y_test = to_categorical(y_test_enc)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nData Preparation Complete:")
        print(f"  Train: {X_train_scaled.shape[0]} samples ({X_train_scaled.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test: {X_test_scaled.shape[0]} samples ({X_test_scaled.shape[0]/len(X)*100:.1f}%)")
        print(f"  Classes: {len(self.label_encoder.classes_)}")
        print(f"  Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
        print(f"  Test mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, y_train_enc, y_test_enc
    
    def save_preprocessor(self, path):
        """Save scaler and label encoder"""
        os.makedirs(path, exist_ok=True)
        pickle.dump(self.scaler, open(os.path.join(path, 'scaler.pkl'), 'wb'))
        pickle.dump(self.label_encoder, open(os.path.join(path, 'label_encoder.pkl'), 'wb'))
        print(f"Preprocessor objects saved to {path}")
    
    def load_preprocessor(self, path):
        """Load scaler and label encoder"""
        self.scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
        self.label_encoder = pickle.load(open(os.path.join(path, 'label_encoder.pkl'), 'rb'))
        print(f"Preprocessor objects loaded from {path}")


if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load CSV
    csv_path = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\data\raw\Birds Voice.csv"
    df = processor.load_dataset(csv_path)
    
    # Process audio files
    audio_base_path = r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\data\Voice of Birds\Voice of Birds"
    X, y, file_paths = processor.process_audio_files(df, audio_base_path, samples_per_species=25)
    
    # Prepare train test data
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = processor.prepare_train_test_data(X, y)
    
    # Save preprocessing objects
    processor.save_preprocessor("models")
