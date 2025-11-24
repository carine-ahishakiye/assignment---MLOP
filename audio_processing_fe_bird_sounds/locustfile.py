import os
import random
import json
import time
from pathlib import Path
from typing import List, Optional

from locust import HttpUser, task, between, events, tag
from locust.exception import StopUser

# Configuration

TEST_AUDIO_DIR = os.getenv(
    "TEST_AUDIO_DIR",
    r"C:\Users\PC\Desktop\assignment---MLOP\audio_processing_fe_bird_sounds\test_auidio"
)
RESULTS_DIR = os.getenv("RESULTS_DIR", "load_test_results")
LOG_FAILURES = os.getenv("LOG_FAILURES", "true").lower() == "true"


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


def log_response_time(name: str, response_time: float, success: bool):
    """Log response time for analysis"""
    log_file = Path(RESULTS_DIR) / "response_times.log"
    log_file.parent.mkdir(exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(f"{time.time()},{name},{response_time},{success}\n")


# Event Handlers for Custom Metrics

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log each request for detailed analysis"""
    if LOG_FAILURES and exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment"""
    print("LOAD TEST STARTED")
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print(f"Test audio files: {len(get_test_audio_files())}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Clean up and summarize test results"""
    print("LOAD TEST COMPLETED")
    
    if hasattr(environment.runner, 'stats'):
        stats = environment.runner.stats
        print(f"Total requests: {stats.total.num_requests}")
        print(f"Total failures: {stats.total.num_failures}")
        print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
        print(f"Requests per second: {stats.total.total_rps:.2f}")
 

# Base User Class with Common Utilities

class BaseAPIUser(HttpUser):
    """Base class with common functionality"""
    
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_files = get_test_audio_files()
        self.request_count = 0
        self.error_count = 0
    
    def on_start(self):
        """Called when a user starts"""
        # check if API is ready
        try:
            response = self.client.get("/api/health", timeout=10)
            if response.status_code != 200:
                print(f"Warning: API health check returned {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not reach API: {e}")
    
    def get_random_audio_file(self) -> Optional[str]:
        """Get a random test audio file"""
        if not self.audio_files:
            return None
        return random.choice(self.audio_files)


# Normal User Behavior

class NormalUser(BaseAPIUser):
    """
    Simulates normal user behavior with realistic wait times
    - Checks health occasionally
    - Views metrics
    - Makes predictions with pauses
    """
    
    wait_time = between(3, 10)  
    weight = 3  
    
    @tag('health')
    @task(2)
    def check_health(self):
        """Check system health - lightweight request"""
        with self.client.get(
            "/api/health",
            catch_response=True,
            name="GET /api/health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @tag('metrics')
    @task(3)
    def view_metrics(self):
        """View system metrics"""
        with self.client.get(
            "/api/metrics",
            catch_response=True,
            name="GET /api/metrics"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'total_predictions' in data:
                    response.success()
                else:
                    response.failure("Missing expected fields")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @tag('classes')
    @task(1)
    def get_classes(self):
        """Get available bird species classes"""
        with self.client.get(
            "/api/classes",
            catch_response=True,
            name="GET /api/classes"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'classes' in data and 'num_classes' in data:
                    response.success()
                else:
                    response.failure("Missing classes data")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @tag('predict')
    @task(5)
    def make_prediction(self):
        """Make a bird sound prediction - most important task"""
        audio_file = self.get_random_audio_file()
        
        if not audio_file:
           
            self.client.get("/api/health", name="GET /api/predict (no file)")
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
                    name="POST /api/predict",
                    timeout=30
                ) as response:
                    self.request_count += 1
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Validate response structure
                        required_fields = [
                            'predicted_species',
                            'confidence',
                            'top_5_predictions'
                        ]
                        
                        if all(field in data for field in required_fields):
                            # Check confidence is reasonable
                            confidence = data.get('confidence', 0)
                            if 0 <= confidence <= 1:
                                response.success()
                            else:
                                response.failure(f"Invalid confidence: {confidence}")
                        else:
                            response.failure("Missing required fields in response")
                    
                    elif response.status_code == 400:
                        response.failure("Bad request - check audio file format")
                    
                    elif response.status_code == 500:
                        self.error_count += 1
                        response.failure("Server error during prediction")
                    
                    else:
                        response.failure(f"Unexpected status: {response.status_code}")
        
        except FileNotFoundError:
            print(f"Audio file not found: {audio_file}")
        except Exception as e:
            print(f"Error making prediction: {e}")
            self.error_count += 1


class HeavyUser(BaseAPIUser):
    """
    Simulates power users who make frequent predictions
    - Minimal wait time
    - Mostly predictions
    - Occasional metrics checks
    """
    
    wait_time = between(1, 3) 
    weight = 1 
    
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
                
                response = self.client.post(
                    "/api/predict",
                    files=files,
                    name="POST /api/predict (heavy)",
                    timeout=30
                )
                
                if response.status_code != 200:
                    self.error_count += 1
        
        except Exception as e:
            self.error_count += 1
    
    @tag('metrics')
    @task(1)
    def check_metrics(self):
        """Occasionally check metrics"""
        self.client.get("/api/metrics")


class StressTestUser(BaseAPIUser):
    
    wait_time = between(0.1, 0.5)  
    weight = 0 
    
    @task(5)
    def rapid_health_checks(self):
        """Rapid fire health checks"""
        self.client.get("/api/health", name="GET /api/health (stress)")
    
    @task(3)
    def rapid_metrics(self):
        """Rapid fire metrics requests"""
        self.client.get("/api/metrics", name="GET /api/metrics (stress)")
    
    @task(1)
    def rapid_classes(self):
        """Rapid fire classes requests"""
        self.client.get("/api/classes", name="GET /api/classes (stress)")




class MonitoringUser(BaseAPIUser):
    """
    Simulates monitoring systems
    - Regular health checks
    - Metrics polling
    - Predictable intervals
    """
    
    wait_time = between(10, 15) 
    weight = 1  
    
    @task(5)
    def monitor_health(self):
        """Regular health monitoring"""
        self.client.get("/api/health", name="GET /api/health (monitor)")
    
    @task(3)
    def monitor_metrics(self):
        """Regular metrics collection"""
        self.client.get("/api/metrics", name="GET /api/metrics (monitor)")
    
    @task(1)
    def validate_system(self):
        """Comprehensive system validation"""
        # Check multiple endpoints
        endpoints = ["/api/health", "/api/metrics", "/api/classes"]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint, name=f"GET {endpoint} (validate)")
            if response.status_code != 200:
                print(f"Validation failed for {endpoint}: {response.status_code}")




"""
COMMAND LINE USAGE:

1. BASIC TEST (Web UI - Interactive):
   locust -f locustfile.py --host=http://localhost:5000
   Then open: http://localhost:8089

2. QUICK SMOKE TEST (10 users, 30 seconds):
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 10 --spawn-rate 2 --run-time 30s \
     --headless --csv=results/quick_test

3. LOAD TEST (50 users, 5 minutes):
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 50 --spawn-rate 5 --run-time 5m \
     --headless --csv=results/load_test \
     --html=results/load_test_report.html

4. STRESS TEST (100 users, heavy load):
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 100 --spawn-rate 10 --run-time 10m \
     --headless --csv=results/stress_test

5. SCALING TEST - 1 Container:
   docker-compose up -d --scale bird-classifier=1
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 50 --spawn-rate 5 --run-time 2m \
     --headless --csv=results/1_container \
     --html=results/1_container.html

6. SCALING TEST - 2 Containers:
   docker-compose up -d --scale bird-classifier=2
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 100 --spawn-rate 10 --run-time 2m \
     --headless --csv=results/2_containers \
     --html=results/2_containers.html

7. SCALING TEST - 4 Containers:
   docker-compose up -d --scale bird-classifier=4
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 200 --spawn-rate 20 --run-time 2m \
     --headless --csv=results/4_containers \
     --html=results/4_containers.html

8. TAGGED TESTS (specific scenarios):
   # Only prediction tests
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 30 --spawn-rate 5 --run-time 2m \
     --tags predict --headless

   # Only health and metrics
   locust -f locustfile.py --host=http://localhost:5000 \
     --users 50 --spawn-rate 10 --run-time 1m \
     --tags health metrics --headless

9. SPECIFIC USER CLASS:
   locust -f locustfile.py --host=http://localhost:5000 \
     NormalUser --users 20 --spawn-rate 4 --run-time 2m \
     --headless --csv=results/normal_users

10. DISTRIBUTED LOAD TESTING (Master-Worker):
    # Master
    locust -f locustfile.py --master --host=http://localhost:5000
    
    # Workers (run on multiple machines)
    locust -f locustfile.py --worker --master-host=<master-ip>

ENVIRONMENT VARIABLES:
  export TEST_AUDIO_DIR=./test_audio
  export RESULTS_DIR=./load_test_results
  export LOG_FAILURES=true

ANALYZING RESULTS:
  - Check CSV files in results/ directory:
    * *_stats.csv - Request statistics
    * *_stats_history.csv - Time series data
    * *_failures.csv - Failed requests
  - Open HTML report in browser
  - Compare response times across different container counts
  - Look for P95, P99 latencies and failure rates

KEY METRICS TO MONITOR:
  - Response time (average, median, P95, P99)
  - Requests per second (RPS/throughput)
  - Failure rate (%)
  - CPU and memory usage (docker stats)
  - Network I/O
"""