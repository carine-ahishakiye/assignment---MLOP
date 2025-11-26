from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import librosa
import io
import pickle
import os
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from functools import wraps
from werkzeug.utils import secure_filename
import threading
import time

# Flask App 
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RETRAIN_FOLDER = 'retrain_data'
MODEL_PATH = r'C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\final_model.h5'
SCALER_PATH = r'C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\scaler.pkl'
ENCODER_PATH = r'C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\label_encoder.pkl'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Creating necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_FOLDER, exist_ok=True)

# Global State 
class AppState:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.prediction_history = []
        self.model_version = "v1.1"
        self.is_retraining = False
        self.retrain_progress = 0
        self.last_retrain_time = None
        self.metrics = {
            'total_predictions': 0,
            'avg_confidence': 0,
            'species_distribution': {}
        }

    def get_uptime(self):
        return (datetime.now() - self.start_time).total_seconds()

    def log_prediction(self, species, confidence):
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'species': species,
            'confidence': confidence
        })
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        # Update metrics
        self.metrics['total_predictions'] += 1
        self.metrics['species_distribution'][species] = \
            self.metrics['species_distribution'].get(species, 0) + 1
        
        # Calculate average confidence
        if self.prediction_history:
            confidences = [p['confidence'] for p in self.prediction_history]
            self.metrics['avg_confidence'] = np.mean(confidences)

state = AppState()

# Load Model and Preprocessors 
def load_model_and_preprocessors():
    try:
        print(f"Loading model from: {MODEL_PATH}")
        state.model = load_model(MODEL_PATH)
        print(f"Loading scaler from: {SCALER_PATH}")
        state.scaler = pickle.load(open(SCALER_PATH, 'rb'))
        print(f"Loading label encoder from: {ENCODER_PATH}")
        state.label_encoder = pickle.load(open(ENCODER_PATH, 'rb'))
        print(" Model and preprocessors loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR loading model or preprocessors: {e}")
        return False

# Load on startup
load_model_and_preprocessors()

# Utilities 
def track_request(f):
    """Decorator to track API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        state.request_count += 1
        try:
            return f(*args, **kwargs)
        except Exception as e:
            state.error_count += 1
            print(f"Error in {f.__name__}: {e}")
            raise
    return decorated_function

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_data, sr=22050):
    """Extract 95 audio features from audio data"""
    try:
        features = []
        
        # MFCC features 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))
        
        # Spectral Centroid
        sc = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.extend([np.mean(sc), np.std(sc), np.max(sc)])
        
        # Spectral Rolloff 
        sr_off = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features.extend([np.mean(sr_off), np.std(sr_off)])
        
        # Zero Crossing Rate 
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # Chroma STFT 
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma)])
        
        # Mel Spectrogram 
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db)])
        
        # Spectral Bandwidth
        sbw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        features.extend([np.mean(sbw), np.std(sbw)])
        
        # RMS Energy 
        rms = librosa.feature.rms(y=audio_data)
        features.append(np.mean(rms))
        
        return np.array(features)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def load_audio(file_storage):
    """Load audio from uploaded file (WAV or MP3)"""
    try:
        filename = file_storage.filename.lower()
        
        # Read file into memory
        audio_bytes = io.BytesIO(file_storage.read())
        
        # Load with librosa 
        y, sr = librosa.load(audio_bytes, sr=22050)
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=23)
        
        # Check if audio is too short
        if len(y_trimmed) < sr * 0.5:
            print(f"Audio too short: {len(y_trimmed)/sr:.2f} seconds")
            return None, None
        
        print(f" Audio loaded: {len(y_trimmed)/sr:.2f} seconds")
        return y_trimmed, sr
        
    except Exception as e:
        print(f"Audio loading error: {e}")
        return None, None

# Retraining Function
def retrain_model_background():
    """Background retraining process (simulated)"""
    state.is_retraining = True
    state.retrain_progress = 0
    
    try:
        print("Starting retraining process...")
        
        #  Check for new data
        state.retrain_progress = 10
        time.sleep(1)
        
        audio_files = [f for f in os.listdir(RETRAIN_FOLDER) 
                      if f.lower().endswith(('.wav', '.mp3'))]
        
        print(f"Found {len(audio_files)} files for retraining")
        
        if len(audio_files) < 10:
            print("Not enough data for retraining (minimum 10 files)")
            state.is_retraining = False
            state.retrain_progress = -1
            return
        
        state.retrain_progress = 30
        time.sleep(2)
        
        #  Process audio files 
        print("Processing audio files...")
        state.retrain_progress = 50
        time.sleep(2)
        
        #  Retrain model 
        print("Retraining model...")
        state.retrain_progress = 80
        time.sleep(2)
        
        #  Reload model
        print("Reloading model...")
        load_model_and_preprocessors()
        
        state.retrain_progress = 100
        state.last_retrain_time = datetime.now()
        
        # Update version
        current_version = float(state.model_version.replace('v', ''))
        state.model_version = f"v{current_version + 0.1:.1f}"
        
        print(f"Retraining completed! New version: {state.model_version}")
        
    except Exception as e:
        print(f"Retraining error: {e}")
        state.retrain_progress = -1
    finally:
        state.is_retraining = False

#  ROUTES 

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': state.get_uptime(),
        'model_version': state.model_version,
        'total_requests': state.request_count,
        'error_count': state.error_count,
        'is_retraining': state.is_retraining,
        'last_retrain': state.last_retrain_time.isoformat() if state.last_retrain_time else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics"""
    return jsonify({
        'total_predictions': state.metrics['total_predictions'],
        'avg_confidence': float(state.metrics['avg_confidence']) if state.metrics['avg_confidence'] else 0,
        'species_distribution': state.metrics['species_distribution'],
        'recent_predictions': state.prediction_history[-20:],
        'uptime': state.get_uptime()
    })

@app.route('/api/predict', methods=['POST'])
@track_request
def predict():
    """Make prediction on single audio file"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type. Use .wav or .mp3'}), 400
    
    # Load audio
    audio_data, sr = load_audio(audio_file)
    if audio_data is None:
        return jsonify({'error': 'Audio too short or could not be loaded. Try a WAV file or ensure FFmpeg is installed for MP3 support.'}), 400
    
    # Extract features
    features = extract_features(audio_data, sr)
    if features is None:
        return jsonify({'error': 'Feature extraction failed'}), 500
    
    # Check feature dimension
    if len(features) != 95:
        return jsonify({'error': f'Expected 95 features, got {len(features)}'}), 500
    
    # Scale features
    scaled_features = state.scaler.transform([features])
    
    # Make prediction
    probs = state.model.predict(scaled_features, verbose=0)[0]
    
    pred_index = np.argmax(probs)
    pred_species = state.label_encoder.classes_[pred_index]
    confidence = float(probs[pred_index])
    
    # Get top 5 predictions
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_preds = {
        state.label_encoder.classes_[i]: float(probs[i]) 
        for i in top5_indices
    }
    
    # Log prediction
    state.log_prediction(pred_species, confidence)
    
    return jsonify({
        'predicted_species': pred_species,
        'confidence': confidence,
        'confidence_percent': f"{confidence*100:.2f}%",
        'top_5_predictions': top5_preds,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload_bulk', methods=['POST'])
@track_request
def upload_bulk():
    """Upload multiple files for retraining"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    uploaded_count = 0
    skipped_count = 0
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(RETRAIN_FOLDER, filename)
            file.save(file_path)
            uploaded_count += 1
            print(f" Saved: {filename}")
        else:
            skipped_count += 1
    
    total_files = len(os.listdir(RETRAIN_FOLDER))
    
    return jsonify({
        'message': f'Successfully uploaded {uploaded_count} files',
        'uploaded_count': uploaded_count,
        'skipped_count': skipped_count,
        'total_files_in_folder': total_files
    })

@app.route('/api/trigger_retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining"""
    if state.is_retraining:
        return jsonify({'error': 'Retraining already in progress'}), 400
    
    # Check if I have files to retrain on
    audio_files = [f for f in os.listdir(RETRAIN_FOLDER) 
                  if f.lower().endswith(('.wav', '.mp3'))]
    
    if len(audio_files) < 10:
        return jsonify({
            'error': f'Not enough data for retraining. Need at least 10 files, found {len(audio_files)}'
        }), 400
    
    # Start retraining in background thread
    thread = threading.Thread(target=retrain_model_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Retraining started',
        'status': 'in_progress',
        'files_count': len(audio_files)
    })

@app.route('/api/retrain_status', methods=['GET'])
def retrain_status():
    """Get retraining status"""
    return jsonify({
        'is_retraining': state.is_retraining,
        'progress': state.retrain_progress,
        'last_retrain': state.last_retrain_time.isoformat() if state.last_retrain_time else None,
        'model_version': state.model_version
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available bird species classes"""
    if state.label_encoder is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': list(state.label_encoder.classes_),
        'num_classes': len(state.label_encoder.classes_)
    })

# Error Handlers 
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
@app.route('/api/feature_analysis', methods=['GET'])
def feature_analysis():
    """Analyze 3 key features from the dataset"""
    try:
        # Sample audio files from retrain folder to analyze
        audio_files = [f for f in os.listdir(RETRAIN_FOLDER) if f.lower().endswith(('.wav', '.mp3'))]
        
        if len(audio_files) < 5:
            return jsonify({
                'error': 'Not enough audio files for analysis. Upload at least 5 files.',
                'mfcc_analysis': {'values': [], 'labels': []},
                'spectral_centroid': {'values': [], 'labels': []},
                'zcr_analysis': {'values': [], 'labels': []}
            })
        
        # Analyze up to 20 random files
        sample_files = np.random.choice(audio_files, min(20, len(audio_files)), replace=False)
        
        mfcc_means = []
        spectral_centroids = []
        zcr_means = []
        labels = []
        
        for audio_file in sample_files:
            try:
                file_path = os.path.join(RETRAIN_FOLDER, audio_file)
                y, sr = librosa.load(file_path, sr=22050, duration=10)
                y_trimmed, _ = librosa.effects.trim(y, top_db=23)
                
                if len(y_trimmed) < sr * 0.5:
                    continue
                
                # Extract features
                mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
                mfcc_mean = float(np.mean(mfccs))
                
                sc = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
                sc_mean = float(np.mean(sc))
                
                zcr = librosa.feature.zero_crossing_rate(y_trimmed)
                zcr_mean = float(np.mean(zcr))
                
                mfcc_means.append(mfcc_mean)
                spectral_centroids.append(sc_mean)
                zcr_means.append(zcr_mean)
                labels.append(audio_file[:20])  # Truncate filename for display
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
                continue
        
        return jsonify({
            'mfcc_analysis': {
                'values': mfcc_means,
                'labels': labels,
                'description': 'MFCC (Mel-frequency cepstral coefficients) represents the short-term power spectrum of sound. Higher values indicate more complex frequency patterns, typical of bird calls with rich harmonics.',
                'interpretation': f'Average MFCC: {np.mean(mfcc_means):.2f}. This shows the spectral envelope of the audio signals.'
            },
            'spectral_centroid': {
                'values': spectral_centroids,
                'labels': labels,
                'description': 'Spectral Centroid indicates where the "center of mass" of the spectrum is located. Higher values suggest brighter sounds with more high-frequency content.',
                'interpretation': f'Average Spectral Centroid: {np.mean(spectral_centroids):.2f} Hz. This indicates the brightness of the bird calls in your dataset.'
            },
            'zcr_analysis': {
                'values': zcr_means,
                'labels': labels,
                'description': 'Zero Crossing Rate measures how often the signal changes from positive to negative. Higher values indicate noisier, more percussive sounds.',
                'interpretation': f'Average ZCR: {np.mean(zcr_means):.4f}. This reflects the noisiness and texture of the bird sounds.'
            },
            'summary': {
                'total_samples': len(mfcc_means),
                'mfcc_range': [float(np.min(mfcc_means)), float(np.max(mfcc_means))],
                'sc_range': [float(np.min(spectral_centroids)), float(np.max(spectral_centroids))],
                'zcr_range': [float(np.min(zcr_means)), float(np.max(zcr_means))]
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Feature analysis failed: {str(e)}',
            'mfcc_analysis': {'values': [], 'labels': []},
            'spectral_centroid': {'values': [], 'labels': []},
            'zcr_analysis': {'values': [], 'labels': []}
        }), 500


@app.route('/api/model_performance', methods=['GET'])
def model_performance():
    """Get detailed model performance metrics"""
    try:
        if len(state.prediction_history) == 0:
            return jsonify({
                'error': 'No predictions made yet',
                'performance': {}
            })
        
        # Calculate performance metrics
        confidences = [p['confidence'] for p in state.prediction_history]
        species_counts = {}
        
        for pred in state.prediction_history:
            species = pred['species']
            species_counts[species] = species_counts.get(species, 0) + 1
        
        # Get top 5 predicted species
        top_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            'total_predictions': len(state.prediction_history),
            'average_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'confidence_distribution': {
                'high': len([c for c in confidences if c > 0.8]),
                'medium': len([c for c in confidences if 0.5 <= c <= 0.8]),
                'low': len([c for c in confidences if c < 0.5])
            },
            'top_predicted_species': [
                {'species': species, 'count': count} for species, count in top_species
            ],
            'unique_species_predicted': len(species_counts),
            'model_version': state.model_version,
            'uptime_hours': state.get_uptime() / 3600
        })
        
    except Exception as e:
        return jsonify({'error': f'Performance analysis failed: {str(e)}'}), 500


@app.route('/api/system_stats', methods=['GET'])
def system_stats():
    """Get comprehensive system statistics"""
    try:
        import psutil
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return jsonify({
            'system': {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            },
            'application': {
                'uptime_seconds': state.get_uptime(),
                'total_requests': state.request_count,
                'error_count': state.error_count,
                'error_rate': (state.error_count / state.request_count * 100) if state.request_count > 0 else 0,
                'model_version': state.model_version,
                'is_retraining': state.is_retraining
            },
            'model': {
                'total_predictions': state.metrics['total_predictions'],
                'avg_confidence': float(state.metrics['avg_confidence']) if state.metrics['avg_confidence'] else 0,
                'unique_species': len(state.metrics['species_distribution']),
                'last_prediction': state.prediction_history[-1] if state.prediction_history else None
            },
            'storage': {
                'upload_folder_files': len(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else 0,
                'retrain_folder_files': len(os.listdir(RETRAIN_FOLDER)) if os.path.exists(RETRAIN_FOLDER) else 0
            }
        })
        
    except ImportError:
        return jsonify({
            'error': 'psutil not installed. Run: pip install psutil',
            'application': {
                'uptime_seconds': state.get_uptime(),
                'total_requests': state.request_count,
                'error_count': state.error_count
            }
        })
    except Exception as e:
        return jsonify({'error': f'System stats failed: {str(e)}'}), 500

# Run Server 
if __name__ == '__main__':
    print("\n" + "="*60)
    print(" Bird Sound Classification API")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Scaler Path: {SCALER_PATH}")
    print(f"Encoder Path: {ENCODER_PATH}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print(f"Retrain Folder: {RETRAIN_FOLDER}")
    print("\nStarting server...")
    print("Access UI at: http://localhost:5000")
    print("API Docs:")
    print("  - GET  /api/health")
    print("  - GET  /api/metrics")
    print("  - POST /api/predict")
    print("  - POST /api/upload_bulk")
    print("  - POST /api/trigger_retrain")
    print("  - GET  /api/retrain_status")
    print("  - GET  /api/classes")
    
    app.run(host='0.0.0.0', port=5000, debug=True)