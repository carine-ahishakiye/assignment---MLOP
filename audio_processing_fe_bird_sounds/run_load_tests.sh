set -e  

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HOST="http://localhost:5000"
RESULTS_DIR="results"
COMPOSE_FILE="docker-compose.yml"

# Create results directory
mkdir -p ${RESULTS_DIR}

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}  LOAD TESTING SCRIPT${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Function to wait for service to be healthy
wait_for_service() {
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for service to be healthy...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "${HOST}/api/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Service is healthy${NC}"
            sleep 5  
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ Service failed to become healthy${NC}"
    return 1
}

# Function to run load test
run_test() {
    local containers=$1
    local users=$2
    local spawn_rate=$3
    local duration=$4
    local test_name="${containers}_containers"
    
    echo ""
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}  TEST: ${test_name}${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo -e "Containers: ${containers}"
    echo -e "Users: ${users}"
    echo -e "Spawn Rate: ${spawn_rate}"
    echo -e "Duration: ${duration}"
    echo ""
    
    # Scale containers
    echo -e "${YELLOW}Scaling to ${containers} container(s)...${NC}"
    docker-compose up -d --scale bird-classifier=${containers}
    
    # Wait for service
    if ! wait_for_service; then
        echo -e "${RED}Skipping test due to health check failure${NC}"
        return 1
    fi
    
    # Run load test
    echo -e "${GREEN}Starting load test...${NC}"
    locust -f locustfile.py \
        --host=${HOST} \
        --users ${users} \
        --spawn-rate ${spawn_rate} \
        --run-time ${duration} \
        --headless \
        --csv=${RESULTS_DIR}/${test_name} \
        --html=${RESULTS_DIR}/${test_name}.html \
        --logfile=${RESULTS_DIR}/${test_name}.log
    
    echo -e "${GREEN}✓ Test completed: ${test_name}${NC}"
    
    # Show container stats
    echo ""
    echo -e "${YELLOW}Container Statistics:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    
    # Cool down period
    echo ""
    echo -e "${YELLOW}Cooling down for 10 seconds...${NC}"
    sleep 10
}

# Function to generate comparison report
generate_report() {
    echo ""
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}  GENERATING COMPARISON REPORT${NC}"
    echo -e "${BLUE}=================================${NC}"
    
    python3 << 'EOF'
import pandas as pd
import os

results_dir = "results"
test_configs = ["1_containers", "2_containers", "4_containers"]

print("\n LOAD TEST COMPARISON REPORT\n")
print("=" * 100)

summary_data = []

for config in test_configs:
    stats_file = f"{results_dir}/{config}_stats.csv"
    if os.path.exists(stats_file):
        df = pd.read_csv(stats_file)
        
        # Get aggregated stats
        agg_row = df[df['Name'] == 'Aggregated']
        if not agg_row.empty:
            summary_data.append({
                'Configuration': config.replace('_', ' ').title(),
                'Total Requests': int(agg_row['Request Count'].values[0]),
                'Failures': int(agg_row['Failure Count'].values[0]),
                'Avg Response (ms)': round(agg_row['Average Response Time'].values[0], 2),
                'Min Response (ms)': round(agg_row['Min Response Time'].values[0], 2),
                'Max Response (ms)': round(agg_row['Max Response Time'].values[0], 2),
                'RPS': round(agg_row['Requests/s'].values[0], 2),
                'Failure Rate (%)': round(agg_row['Failure Count'].values[0] / agg_row['Request Count'].values[0] * 100, 2) if agg_row['Request Count'].values[0] > 0 else 0
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(f"{results_dir}/comparison_summary.csv", index=False)
    print(f"\n✓ Comparison summary saved to {results_dir}/comparison_summary.csv")
else:
    print("No test results found!")

print("=" * 100)
EOF
}

# Main execution
main() {
    echo -e "${GREEN}Starting automated load testing...${NC}"
    echo ""
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: docker-compose not found${NC}"
        exit 1
    fi
    
    # Check if locust is available
    if ! command -v locust &> /dev/null; then
        echo -e "${RED}Error: locust not found. Install with: pip install locust${NC}"
        exit 1
    fi
    
    # Build images if needed
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose build
    
    # Run tests with different configurations
    # TContainer 
    run_test 1 50 5 "2m"
    
    #   Containers 
    run_test 2 100 10 "2m"
    
    #   Containers 
    run_test 4 200 20 "2m"
    
    # Generate comparison report
    generate_report
    
    # Cleanup
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    docker-compose down
    
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}  ALL TESTS COMPLETED!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo -e "Results saved in: ${RESULTS_DIR}/"
    echo -e "  - CSV files: ${RESULTS_DIR}/*_stats.csv"
    echo -e "  - HTML reports: ${RESULTS_DIR}/*.html"
    echo -e "  - Log files: ${RESULTS_DIR}/*.log"
    echo -e "  - Comparison: ${RESULTS_DIR}/comparison_summary.csv"
    echo ""
    echo -e "${BLUE}View HTML reports in your browser:${NC}"
    for config in 1 2 4; do
        echo -e "  file://${PWD}/${RESULTS_DIR}/${config}_containers.html"
    done
    echo ""
}

# Run main function
main

