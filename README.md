#  Bird Sound Classification System - ML Pipeline Project

##  Video Demonstration

**YouTube Demo Link**: [INSERT YOUR VIDEO LINK HERE]

**Live Deployment URL**: https://assignment---mlop-8npooe4tivumqtbxpimrh3.streamlit.app/

---

##  Project Description

This is an **end to end Machine Learning pipeline** for **Bird Sound Classification** using Deep Neural Networks. The system processes audio files (WAV/MP3), extracts 95 acoustic features, and classifies bird species with high accuracy.

### ✨ Key Features

1. **🎤 Single Prediction**: Upload individual audio files for instant species identification
2. ** Bulk Upload & Retraining**: Upload multiple audio files and trigger model retraining
3. ** Real-time Metrics**: System health, uptime, prediction confidence trends
4. ** Data Visualizations**: Species distribution, confidence trends, prediction history
5. **Scalable Architecture**: Horizontal scaling with Docker containers + NGINX load balancer
6. ** Web Interface**: Modern, responsive UI with dark theme and interactive charts

---

##  Model Performance

| Metric          | Score    |
|-----------------|----------|
| **Accuracy**    | 68.00%   |
| **Precision**   | 70.63%   |
| **Recall**      | 68.00%   |
| **F1-Score**    | 66.85%   |
| **Test Loss**   | 2.0806   |

###  Model Architecture

- **Input Layer**: 95 acoustic features
- **Layer 1**: 512 neurons (ReLU, Dropout 0.5, L2 regularization)
- **Layer 2**: 256 neurons (ReLU, Dropout 0.4, L2 regularization)
- **Layer 3**: 128 neurons (ReLU, Dropout 0.3, L2 regularization)
- **Output Layer**: 64 classes (Softmax activation)

###  Feature Engineering (95 Features)

- **MFCC** (Mel-Frequency Cepstral Coefficients): 80 features (mean, std, max, min)
- **Spectral Centroid**: 3 features (mean, std, max)
- **Spectral Rolloff**: 2 features (mean, std)
- **Zero Crossing Rate**: 2 features (mean, std)
- **Chroma STFT**: 2 features (mean, std)
- **Mel Spectrogram**: 3 features (mean, std, max)
- **Spectral Bandwidth**: 2 features (mean, std)
- **RMS Energy**: 1 feature (mean)

---

##  Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.8+** (for local development)
- **Audio files** in WAV or MP3 format

###  Docker Deployment 
```bash

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.8+** (for local development)
- **Audio files** in WAV or MP3 format

```bash
# 1. Clone the repository
git clone <https://github.com/carine-ahishakiye/assignment---MLOP.git>
cd audio_processing_fe_bird_sounds

# 2. Build and start containers
docker-compose up -d --build

# 3. Access the application
# Frontend: http://localhost:5000
# API Docs: http://localhost:5000/docs

# 4. Check container status
docker-compose ps

# 5. View logs
docker-compose logs -f bird-classifier

# 6. Scale containers 
docker-compose up -d --scale bird-classifier=3

# 7. Stop containers
docker-compose down
```

###  Local Development
```bash
# 1. Create virtual environment
python -m venv venv
source venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python api.py

# 4. Access at http://localhost:5000
```

---

##  Project Structure
```
audio_processing_fe_bird_sounds/
│
├── README.md                          
├── requirements.txt                  
├── Dockerfile                         
├── docker-compose.yml                
├── nginx.conf                         
│
├── notebook/
│   └── bird_sound_classification.ipynb  
│
├── src/
│   ├── preprocessing.py               
│   ├── model.py                     
│   └── prediction.py  
|               
│
├── data/
│   ├── raw/
│   │   └── Birds Voice.csv            
│   ├── Voice of Birds/                
│   ├── train/                        
│   └── test/                          
│
├── models/
│   ├── final_model.h5                 
│   ├── scaler.pkl                     
│   └── label_encoder.pkl              
│
├── templates/
│   └── index.html                  
│
├── uploads/                           
├── retrain_data/                   
├── test_audio/                        
│
├── api.py                             
├── locustfile.py                      
└── run_load_tests.sh                  
```

---

##  API Endpoints

### Core Endpoints

| Method | Endpoint               | Description                          |
|--------|------------------------|--------------------------------------|
| GET    | `/`                    | Web UI dashboard                     |
| GET    | `/api/health`          | System health check                  |
| GET    | `/api/metrics`         | Prediction metrics & stats           |
| GET    | `/api/classes`         | List of bird species (64 classes)    |
| POST   | `/api/predict`         | Single audio file prediction         |
| POST   | `/api/upload_bulk`     | Bulk upload for retraining           |
| POST   | `/api/trigger_retrain` | Trigger model retraining             |
| GET    | `/api/retrain_status`  | Check retraining progress            |

### Example: Prediction Request
```bash
curl -X POST "http://localhost:5000/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio/sample_bird.wav"
```

**Response**:
```json
{
  "predicted_species": "Andean Tinamou",
  "confidence": 0.9523,
  "confidence_percent": "95.23%",
  "top_5_predictions": {
    "Andean Tinamou": 0.9523,
    "Chilean Tinamou": 0.0312,
    "Elegant Crested Tinamou": 0.0089,
    "Ornate Tinamou": 0.0047,
    "Tataupa Tinamou": 0.0029
  },
  "timestamp": "2025-01-15T14:32:10.123456"
}
```

---

##  Retraining Pipeline

### How It Works

1. **Upload Audio Files**: Use the "Bulk Upload" section to upload multiple WAV/MP3 files
2. **Trigger Retraining**: Click "🔄 Trigger Retraining" button
3. **Monitor Progress**: Real-time progress bar shows training status
4. **Automatic Model Update**: New model replaces the old one upon completion

### Manual Retraining (
```bash
# 1. Add new audio files to retrain_data/
cp new_birds/*.mp3 retrain_data/

# 2. Run retraining script
python src/model.py --retrain --data retrain_data/

# 3. Model will be saved to models/final_model.h5
```

---

##  Load Testing Results

### Test Scenarios

I tested with 1, 2, and 4 Docker containers using Locust.

#### 1 Container
```bash
docker-compose up -d --scale bird-classifier=1
locust -f locustfile.py --host=http://localhost:5000 \
  --users 50 --spawn-rate 5 --run-time 2m \
  --headless --html=results/1_container.html
```

**Results**:
- Requests/sec: 12.3 RPS
- Average Response Time: 487ms
- P95 Latency: 1,250ms
- Failure Rate: 0.2%

#### 2 Containers
```bash
docker-compose up -d --scale bird-classifier=2
locust -f locustfile.py --host=http://localhost:5000 \
  --users 100 --spawn-rate 10 --run-time 2m \
  --headless --html=results/2_containers.html
```

**Results**:
- Requests/sec: 23.7 RPS
- Average Response Time: 312ms
- P95 Latency: 780ms
- Failure Rate: 0.1%

#### 4 Containers
```bash
docker-compose up -d --scale bird-classifier=4
locust -f locustfile.py --host=http://localhost:5000 \
  --users 200 --spawn-rate 20 --run-time 2m \
  --headless --html=results/4_containers.html
```

**Results**:
- Requests/sec: 45.2 RPS
- Average Response Time: 178ms
- P95 Latency: 420ms
- Failure Rate: 0.0%

### Performance Insights

| Containers | RPS   | Avg Latency | P95 Latency | Throughput Improvement |
|------------|-------|-------------|-------------|------------------------|
| 1          | 12.3  | 487ms       | 1,250ms     | Baseline               |
| 2          | 23.7  | 312ms       | 780ms       | +93% throughput        |
| 4          | 45.2  | 178ms       | 420ms       | +267% throughput       |

**Conclusion**: Horizontal scaling with Docker containers significantly improves throughput and reduces latency. The system handles 20+ concurrent users with 4 containers while maintaining sub-500ms P95 latency.

---

##  Data Insights and Visualizations

The UI provides three key visualizations:

###  Species Distribution 
- Shows the top 10 most predicted bird species
- **Insight**: Identifies which species are most common in predictions
- **Use Case**: Helps understand model usage patterns and dataset balance

###  Prediction Confidence Trend 
- Displays the last 20 predictions' confidence scores
- **Insight**: Monitors model confidence over time
- **Use Case**: Detect model degradation or need for retraining

###  System Metrics Dashboard
- **Uptime**: Tracks system availability
- **Total Predictions**: Cumulative prediction count
- **Average Confidence**: Overall model certainty
- **Model Version**: Current model identifier

---

##  Troubleshooting

###  Model not loading
```bash
# Check if model files exist
ls -lh models/

# If missing, retrain the model
python src/model.py
```

### Audio file format error
```bash
# Convert MP3 to WAV
ffmpeg -i audio.mp3 audio.wav
```

###  Docker container not starting
```bash
# Check logs
docker-compose logs bird-classifier

# Rebuild containers
docker-compose down
docker-compose up -d --build
```

###  Port 5000 already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8080:5000"  
```

---

##  Academic Context

**Course**: Machine Learning Pipeline (BSE)  
**Institution**: African Leadership University  
**Assignment**: MLOps Summative -  ML System

**Learning Outcomes Demonstrated**:
1. ✅ Data acquisition and preprocessing (audio feature engineering)
2. ✅ Model creation and optimization (deep neural network)
3. ✅ Model evaluation (4+ metrics: accuracy, precision, recall, F1)
4. ✅ Model retraining pipeline (with trigger mechanism)
5. ✅ API development (FastAPI with 8 endpoints)
6. ✅ UI creation (real-time metrics, visualizations, upload)
7. ✅ Cloud deployment (Dockerized, scalable architecture)
8. ✅ Load testing (Locust with 1/2/4 container scenarios)

---

## 👨 Author

**[Your Name]**  
- GitHub: [carine-ahishakiye](https://github.com/carine-ahishakiye)
- Email: c.yibukabay@alustudent.com

---

**Last Updated**: November 2025  
