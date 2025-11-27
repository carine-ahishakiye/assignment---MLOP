import os
import random
import time
from pathlib import Path
from typing import List, Optional

from locust import HttpUser, task, between, events, tag

# Configuration
TEST_AUDIO_DIR = os.getenv("TEST_AUDIO_DIR", "test_audio")
RESULTS_DIR = os.getenv("RESULTS_DIR", "load_test_results")

# Helper Functions
def get_test_audio_files() -> List[str]:
    """Get list of available test audio files"""
    test_dir = Path(TEST_AUDIO_DIR)
    
    if not test_dir.exists():
        print(f"Warning: Test audio directory '{TEST_AUDIO_DIR}' not found")
        return []
    
    audio_files = []
    for ext in ['.wav', '.mp3']:
        audio_files.extend([str(f) for f in test_dir.glob(f'*{ext}')])
    
    return audio_files

# Event Handlers
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment"""
    print("\n" + "="*70)
    print("🔥 LOAD TEST STARTED")
    print("="*70)
    print(f"Host: {environment.host}")
    print(f"Test audio files: {len(get_test_audio_files())}")
    print("="*70 + "\n")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Summarize test results"""
    print("\n" + "="*70)
    print("✅ LOAD TEST COMPLETED")
    print("="*70)
    
    if hasattr(environment.runner, 'stats'):
        stats = environment.runner.stats
        print(f"Total Requests: {stats.total.num_requests}")
        print(f"Total Failures: {stats.total.num_failures}")
        print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
        print(f"Requests/Second: {stats.total.total_rps:.2f}")
        if stats.total.num_requests > 0:
            print(f"Failure Rate: {(stats.total.num_failures/stats.total.num_requests*100):.2f}%")
    print("="*70 + "\n")

# Base User Class
class BaseAPIUser(HttpUser):
    """Base class with common functionality"""
    
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_files = get_test_audio_files()
    
    def on_start(self):
        """Check API health on start"""
        try:
            response = self.client.get("/api/health", timeout=10)
            if response.status_code == 200:
                print(f"✓ User started - API healthy")
            else:
                print(f"⚠ API health check returned {response.status_code}")
        except Exception as e:
            print(f"⚠ Could not reach API: {e}")
    
    def get_random_audio_file(self) -> Optional[str]:
        """Get a random test audio file"""
        if not self.audio_files:
            return None
        return random.choice(self.audio_files)

# User Classes
class NormalUser(BaseAPIUser):
    """
    Simulates normal user behavior
    - Realistic wait times between actions
    - Balanced mix of viewing and predicting
    - Weight: 70% of users
    """
    
    wait_time = between(3, 10)
    weight = 7
    
    @tag('health')
    @task(2)
    def check_health(self):
        """Check system health"""
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure(f"Unhealthy: {data.get('status')}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @tag('metrics')
    @task(3)
    def view_metrics(self):
        """View system metrics"""
        self.client.get("/api/metrics")
    
    @tag('predict')
    @task(10)
    def make_prediction(self):
        """Make a bird sound prediction"""
        audio_file = self.get_random_audio_file()
        
        if not audio_file:
            return
        
        try:
            with open(audio_file, 'rb') as f:
                files = {
                    'audio': (
                        os.path.basename(audio_file),
                        f,
                        'audio/wav' if audio_file.endswith('.wav') else 'audio/mpeg'
                    )
                }
                
                with self.client.post(
                    "/api/predict",
                    files=files,
                    catch_response=True,
                    timeout=30
                ) as response:
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'predicted_species' in data and 'confidence' in data:
                            confidence = data.get('confidence', 0)
                            if 0 <= confidence <= 1:
                                response.success()
                            else:
                                response.failure(f"Invalid confidence: {confidence}")
                        else:
                            response.failure("Missing required fields")
                    else:
                        response.failure(f"Status: {response.status_code}")
        
        except FileNotFoundError:
            print(f"⚠ Audio file not found: {audio_file}")
        except Exception as e:
            print(f"⚠ Prediction error: {e}")

class PowerUser(BaseAPIUser):
    
    wait_time = between(1, 3)
    weight = 2
    
    @tag('predict')
    @task(10)
    def rapid_predictions(self):
        """Make rapid predictions"""
        audio_file = self.get_random_audio_file()
        
        if not audio_file:
            return
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': (os.path.basename(audio_file), f, 'audio/wav')}
                self.client.post("/api/predict", files=files, timeout=30)
        except Exception:
            pass
    
    @tag('metrics')
    @task(1)
    def check_metrics(self):
        """Occasionally check metrics"""
        self.client.get("/api/metrics")

class MonitoringUser(BaseAPIUser):

    
    wait_time = between(10, 15)
    weight = 1
    
    @tag('health')
    @task(5)
    def monitor_health(self):
        """Regular health monitoring"""
        self.client.get("/api/health")
    
    @tag('metrics')
    @task(3)
    def monitor_metrics(self):
        """Regular metrics collection"""
        self.client.get("/api/metrics")
    
    @task(1)
    def check_retrain_status(self):
        """Check retraining status"""
        self.client.get("/api/retrain_status")

