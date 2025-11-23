from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import numpy as np
import librosa
import io
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
from functools import wraps
from pydub import AudioSegment

# Flask App 
app = Flask(__name__)
CORS(app)

#  Global State 
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

    def get_uptime(self):
        return (datetime.now() - self.start_time).total_seconds()

    def log_prediction(self, species, confidence):
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'species': species,
            'confidence': confidence
        })
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

state = AppState()

# Load Model & Preprocessors 
def load_model_and_preprocessors():
    try:
        state.model = load_model(r'C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\final_model.h5')
        state.scaler = pickle.load(open(r'C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\scaler.pkl', 'rb'))
        state.label_encoder = pickle.load(open(r'C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\models\label_encoder.pkl', 'rb'))
        print("✓ Model and preprocessors loaded successfully")
    except Exception as e:
        print(f"ERROR loading model or preprocessors: {e}")
        raise

load_model_and_preprocessors()

#  Utilities 
def track_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        state.request_count += 1
        try:
            return f(*args, **kwargs)
        except Exception as e:
            state.error_count += 1
            raise
    return decorated_function

def extract_features(audio_data, sr=22050):
    try:
        features = []
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))
        sc = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.extend([np.mean(sc), np.std(sc), np.max(sc)])
        sr_off = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features.extend([np.mean(sr_off), np.std(sr_off)])
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.extend([np.mean(zcr), np.std(zcr)])
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma)])
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db)])
        sbw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        features.extend([np.mean(sbw), np.std(sbw)])
        rms = librosa.feature.rms(y=audio_data)
        features.append(np.mean(rms))
        return np.array(features)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def load_audio(file_storage):
    try:
        filename = file_storage.filename.lower()
        audio_bytes = io.BytesIO(file_storage.read())
        if filename.endswith('.mp3'):
            audio_segment = AudioSegment.from_mp3(audio_bytes)
            wav_bytes = io.BytesIO()
            audio_segment.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            y, sr = librosa.load(wav_bytes, sr=22050)
        else:
            y, sr = librosa.load(audio_bytes, sr=22050)
        y_trimmed, _ = librosa.effects.trim(y, top_db=23)
        if len(y_trimmed) < sr * 0.5:
            return None, None
        return y_trimmed, sr
    except Exception as e:
        print(f"Audio loading error: {e}")
        return None, None

#  HTTPS Redirect Fix 
@app.before_request
def fix_bad_https_requests():
    """
    This prevents the 400 Bad Request errors caused by HTTPS requests hitting
    Flask dev server without proper TLS. Redirects to HTTP if detected.
    """
    if request.is_secure:
        url = request.url.replace("https://", "http://", 1)
        return redirect(url, code=301)

# === Routes ===
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Bird Sound API is running!',
        'available_endpoints': ['/health', '/predict', '/bulk_predict', '/classes'],
        'model_version': state.model_version,
        'uptime_seconds': state.get_uptime()
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': state.get_uptime(),
        'model_version': state.model_version,
        'total_requests': state.request_count,
        'error_count': state.error_count,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
@track_request
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    audio_data, sr = load_audio(audio_file)
    if audio_data is None:
        return jsonify({'error': 'Audio too short or could not be loaded'}), 400
    features = extract_features(audio_data, sr)
    if features is None:
        return jsonify({'error': 'Feature extraction failed'}), 500
    scaled_features = state.scaler.transform([features])
    probs = state.model.predict(scaled_features, verbose=0)[0]
    pred_index = np.argmax(probs)
    pred_species = state.label_encoder.classes_[pred_index]
    confidence = float(probs[pred_index])
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_preds = {state.label_encoder.classes_[i]: float(probs[i]) for i in top5_indices}
    state.log_prediction(pred_species, confidence)
    return jsonify({
        'predicted_species': pred_species,
        'confidence': confidence,
        'confidence_percent': f"{confidence*100:.2f}%",
        'top_5_predictions': top5_preds,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/bulk_predict', methods=['POST'])
@track_request
def bulk_predict():
    if 'audio_files' not in request.files:
        return jsonify({'error': 'No audio files provided'}), 400
    files = request.files.getlist('audio_files')
    results = []
    for f in files:
        audio_data, sr = load_audio(f)
        if audio_data is None:
            results.append({'filename': f.filename, 'status': 'failed'})
            continue
        features = extract_features(audio_data, sr)
        scaled_features = state.scaler.transform([features])
        probs = state.model.predict(scaled_features, verbose=0)[0]
        pred_index = np.argmax(probs)
        pred_species = state.label_encoder.classes_[pred_index]
        confidence = float(probs[pred_index])
        results.append({
            'filename': f.filename,
            'predicted_species': pred_species,
            'confidence_percent': f"{confidence*100:.2f}%",
            'status': 'success'
        })
        state.log_prediction(pred_species, confidence)
    return jsonify({'total_processed': len(files), 'results': results})

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': list(state.label_encoder.classes_)})

# Error Handlers 
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

#  Run Server 
if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
