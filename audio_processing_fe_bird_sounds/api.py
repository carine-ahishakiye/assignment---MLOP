from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import io
import pickle
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import threading
import time

# Flask App Configuration
app = Flask(__name__)
CORS(app)

# Configuration
RETRAIN_FOLDER = 'retrain_data'
MODEL_PATH = os.getenv('MODEL_PATH', 'models/final_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
ENCODER_PATH = os.getenv('ENCODER_PATH', 'models/label_encoder.pkl')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Create necessary directories
os.makedirs(RETRAIN_FOLDER, exist_ok=True)

# APPLICATION STATE
class AppState:
    """Global application state management"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.start_time = datetime.now()
        self.prediction_history = []
        self.model_version = "v1.0"
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

# MODEL LOADING
def load_model_and_preprocessors():
    """Load trained model and preprocessing objects"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        state.model = load_model(MODEL_PATH)
        state.scaler = pickle.load(open(SCALER_PATH, 'rb'))
        state.label_encoder = pickle.load(open(ENCODER_PATH, 'rb'))
        print("✓ Model and preprocessors loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

# Load on startup
load_model_and_preprocessors()

# UTILITIES
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_data, sr=22050):
    """Extract 95 audio features from audio data"""
    try:
        features = []
        
        # MFCC features (80 features: 20 coefficients × 4 statistics)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))
        
        # Spectral Centroid (3 features)
        sc = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.extend([np.mean(sc), np.std(sc), np.max(sc)])
        
        # Spectral Rolloff (2 features)
        sr_off = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features.extend([np.mean(sr_off), np.std(sr_off)])
        
        # Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # Chroma STFT (2 features)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma)])
        
        # Mel Spectrogram (3 features)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db)])
        
        # Spectral Bandwidth (2 features)
        sbw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        features.extend([np.mean(sbw), np.std(sbw)])
        
        # RMS Energy (1 feature)
        rms = librosa.feature.rms(y=audio_data)
        features.append(np.mean(rms))
        
        return np.array(features)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def load_audio(file_storage):
    """Load audio from uploaded file"""
    try:
        audio_bytes = io.BytesIO(file_storage.read())
        y, sr = librosa.load(audio_bytes, sr=22050)
        y_trimmed, _ = librosa.effects.trim(y, top_db=23)
        
        if len(y_trimmed) < sr * 0.5:
            return None, None
        
        return y_trimmed, sr
    except Exception as e:
        print(f"Audio loading error: {e}")
        return None, None

# RETRAINING
def retrain_model_background():
    """Background retraining process (simulated)"""
    state.is_retraining = True
    state.retrain_progress = 0
    
    try:
        audio_files = [f for f in os.listdir(RETRAIN_FOLDER) 
                      if f.lower().endswith(('.wav', '.mp3'))]
        
        if len(audio_files) < 10:
            state.is_retraining = False
            state.retrain_progress = -1
            return
        
        # Simulate retraining process
        for progress in [10, 30, 50, 80, 100]:
            state.retrain_progress = progress
            time.sleep(2)
        
        # Reload model
        load_model_and_preprocessors()
        state.last_retrain_time = datetime.now()
        
        # Update version
        current_version = float(state.model_version.replace('v', ''))
        state.model_version = f"v{current_version + 0.1:.1f}"
        
    except Exception as e:
        print(f"Retraining error: {e}")
        state.retrain_progress = -1
    finally:
        state.is_retraining = False

# API ROUTES
@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Bird Sound Classification API',
        'version': state.model_version,
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'metrics': '/api/metrics',
            'predict': '/api/predict [POST]',
            'upload_bulk': '/api/upload_bulk [POST]',
            'trigger_retrain': '/api/trigger_retrain [POST]',
            'retrain_status': '/api/retrain_status',
            'feature_analysis': '/api/feature_analysis',
            'model_performance': '/api/model_performance'
        },
        'documentation': 'Use /api/health to check system status'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': state.get_uptime(),
        'model_version': state.model_version,
        'is_retraining': state.is_retraining,
        'last_retrain': state.last_retrain_time.isoformat() if state.last_retrain_time else None
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics"""
    return jsonify({
        'total_predictions': state.metrics['total_predictions'],
        'avg_confidence': float(state.metrics['avg_confidence']) if state.metrics['avg_confidence'] else 0,
        'species_distribution': state.metrics['species_distribution'],
        'recent_predictions': state.prediction_history[-20:]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on audio file"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if not audio_file.filename or not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Load audio
    audio_data, sr = load_audio(audio_file)
    if audio_data is None:
        return jsonify({'error': 'Audio too short or invalid'}), 400
    
    # Extract features
    features = extract_features(audio_data, sr)
    if features is None or len(features) != 95:
        return jsonify({'error': 'Feature extraction failed'}), 500
    
    # Scale and predict
    scaled_features = state.scaler.transform([features])
    probs = state.model.predict(scaled_features, verbose=0)[0]
    
    pred_index = np.argmax(probs)
    pred_species = state.label_encoder.classes_[pred_index]
    confidence = float(probs[pred_index])
    
    # Get top 5
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
def upload_bulk():
    """Upload multiple files for retraining"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded_count = 0
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(RETRAIN_FOLDER, filename)
            file.save(file_path)
            uploaded_count += 1
    
    total_files = len(os.listdir(RETRAIN_FOLDER))
    
    return jsonify({
        'uploaded_count': uploaded_count,
        'total_files_in_folder': total_files
    })

@app.route('/api/trigger_retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining"""
    if state.is_retraining:
        return jsonify({'error': 'Retraining in progress'}), 400
    
    audio_files = [f for f in os.listdir(RETRAIN_FOLDER) 
                  if f.lower().endswith(('.wav', '.mp3'))]
    
    if len(audio_files) < 10:
        return jsonify({'error': f'Need at least 10 files, found {len(audio_files)}'}), 400
    
    # Start retraining
    thread = threading.Thread(target=retrain_model_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Retraining started',
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

@app.route('/api/feature_analysis', methods=['GET'])
def feature_analysis():
    """Analyze audio features from retrain dataset"""
    try:
        audio_files = [f for f in os.listdir(RETRAIN_FOLDER) 
                      if f.lower().endswith(('.wav', '.mp3'))]
        
        if len(audio_files) < 5:
            return jsonify({
                'error': 'Need at least 5 files',
                'mfcc_analysis': {'values': [], 'labels': [], 'description': ''},
                'spectral_centroid': {'values': [], 'labels': [], 'description': ''},
                'zcr_analysis': {'values': [], 'labels': [], 'description': ''}
            })
        
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
                
                mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
                sc = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y_trimmed)
                
                mfcc_means.append(float(np.mean(mfccs)))
                spectral_centroids.append(float(np.mean(sc)))
                zcr_means.append(float(np.mean(zcr)))
                labels.append(audio_file[:20])
            except:
                continue
        
        return jsonify({
            'mfcc_analysis': {
                'values': mfcc_means,
                'labels': labels,
                'description': 'MFCC represents spectral characteristics'
            },
            'spectral_centroid': {
                'values': spectral_centroids,
                'labels': labels,
                'description': 'Spectral Centroid indicates brightness'
            },
            'zcr_analysis': {
                'values': zcr_means,
                'labels': labels,
                'description': 'Zero Crossing Rate measures noisiness'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance', methods=['GET'])
def model_performance():
    """Get model performance metrics"""
    if not state.prediction_history:
        return jsonify({'error': 'No predictions yet'})
    
    confidences = [p['confidence'] for p in state.prediction_history]
    species_counts = {}
    
    for pred in state.prediction_history:
        species = pred['species']
        species_counts[species] = species_counts.get(species, 0) + 1
    
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
            {'species': s, 'count': c} for s, c in top_species
        ],
        'unique_species_predicted': len(species_counts),
        'model_version': state.model_version,
        'uptime_hours': state.get_uptime() / 3600
    })

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'GET /': 'API information',
            'GET /api/health': 'Health check',
            'GET /api/metrics': 'System metrics',
            'POST /api/predict': 'Make prediction (requires audio file)',
            'POST /api/upload_bulk': 'Upload training files',
            'POST /api/trigger_retrain': 'Start model retraining',
            'GET /api/retrain_status': 'Retraining status',
            'GET /api/feature_analysis': 'Audio feature analysis',
            'GET /api/model_performance': 'Model performance metrics'
        },
        'tip': 'Try GET / for API information'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# SERVER STARTUP
if __name__ == '__main__':
    print(" BIRD SOUND CLASSIFICATION API")
    print(f"\nModel: {MODEL_PATH}")
    print(f"Retrain Folder: {RETRAIN_FOLDER}\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)